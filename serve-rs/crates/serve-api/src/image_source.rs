//! serve/engine/image_source.py: where a request's images and videos may come from, and the bounded read of each.

use base64::Engine as _;
use regex::Regex;
use serve_text::pyjson::py_str_ascii;
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

pub const IMAGE_MAX_BYTES: u64 = 32 << 20;
pub const VIDEO_MAX_BYTES: u64 = 1 << 30;
pub const IMAGE_READ_S: f64 = 60.0;
pub const USER_AGENT: &str = "local-server";
pub const USAGE: &str = "an image must be a data: URL, an http(s) URL or a local file path";
pub const POLICY: &str = "image source refused by server policy";
pub const DATA_WORKS: &str = "A data: URL always works.";

pub fn why_remote(why: &str) -> &'static str {
    match why {
        "peer" => "this client is not on the server's machine",
        "forwarded" => "the request came through a proxy",
        "headers" => "the request's headers are malformed",
        "host" => "the request's Host is not a loopback name",
        "origin" => "the request came from a web page",
        "remote-only" => "the server treats every client as remote (--image-remote-only)",
        _ => "the client's address is unknown",
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Kind {
    pub noun: &'static str,
    pub max_bytes: u64,
    pub usage: &'static str,
}

pub const IMAGE: Kind = Kind { noun: "image", max_bytes: IMAGE_MAX_BYTES, usage: USAGE };
pub const VIDEO: Kind =
    Kind { noun: "video", max_bytes: VIDEO_MAX_BYTES, usage: "a video must be a data: URL, an http(s) URL or a local file path" };

/// What one request may name.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageSources {
    pub url_hosts: Vec<(String, Option<u16>)>,
    pub dirs: Vec<String>,
    pub remote_only: bool,
    pub local: bool,
    pub why: String,
    pub peer: String,
}

impl Default for ImageSources {
    fn default() -> Self {
        ImageSources { url_hosts: vec![], dirs: vec![], remote_only: false, local: false, why: "no-peer".into(), peer: String::new() }
    }
}

impl ImageSources {
    /// dataclasses.replace(policy, local=, why=, peer=)
    pub fn with_access(&self, local: bool, why: String, peer: String) -> ImageSources {
        ImageSources { local, why, peer, ..self.clone() }
    }
}

/// A refused or failed source: `message` for the client, `log_line` for the operator (None: a plain ValueError).
#[derive(Clone, Debug)]
pub struct SourceErr {
    pub message: String,
    pub log_line: Option<String>,
}

impl SourceErr {
    pub fn value(m: impl Into<String>) -> Self {
        SourceErr { message: m.into(), log_line: None }
    }
    fn refused(message: String, log: String) -> Self {
        SourceErr { message, log_line: Some(log) }
    }
}

impl std::fmt::Display for SourceErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// shown: client-named text as one safe, short line.
pub fn shown(text: &str) -> String {
    let cut: String = text.chars().take(200).collect();
    py_str_ascii(&cut)
}

static ENTRY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:(\[[0-9a-f:.]+\]|[a-z0-9._-]+)(?::([0-9]{1,5}))?)$").unwrap());
static SCHEME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z][a-z0-9+.-]+://").unwrap());
static UNSAFE_URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^\x21-\x7e]|\\").unwrap());

fn py_lower(s: &str) -> String {
    s.to_lowercase()
}

/// Python's ipaddress.ip_address: strict dotted IPv4 (no leading zeros) or IPv6 (an optional %scope is kept apart).
pub fn parse_ip(text: &str) -> Option<(IpAddr, Option<String>)> {
    if let Ok(v4) = text.parse::<Ipv4Addr>() {
        // Python refuses leading zeros, as Rust does; both refuse fewer than four parts
        return Some((IpAddr::V4(v4), None));
    }
    let (addr, scope) = match text.split_once('%') {
        Some((a, s)) if !s.is_empty() && !s.contains('%') => (a, Some(s.to_string())),
        Some(_) => return None,
        None => (text, None),
    };
    addr.parse::<Ipv6Addr>().ok().map(|a| (IpAddr::V6(a), scope))
}

fn ip_text(ip: &IpAddr, scope: &Option<String>) -> String {
    match scope {
        Some(s) => format!("{ip}%{s}"),
        None => ip.to_string(),
    }
}

pub fn is_ip(name: &str) -> bool {
    parse_ip(name).is_some()
}

fn canonical_host(host: &str) -> String {
    match parse_ip(host) {
        Some((ip, scope)) => ip_text(&ip, &scope),
        None => host.to_string(),
    }
}

/// parse_url_host: a --image-url-host entry, HOST or HOST:PORT ([ADDR] for IPv6), as (host, port).
pub fn parse_url_host(text: &str) -> Result<(String, Option<u16>), String> {
    let low = py_lower(crate::py::strip(text));
    let bad = || format!("expected HOST or HOST:PORT, not {}", serve_text::pyjson::py_str_repr(text));
    let m = ENTRY.captures(&low);
    let port: Option<i64> = m.as_ref().and_then(|m| m.get(2)).map(|p| p.as_str().parse().unwrap());
    let host = m.as_ref().map(|m| m.get(1).unwrap().as_str().to_string()).unwrap_or_default();
    if host.is_empty()
        || host.starts_with('.')
        || host.ends_with('.')
        || host.contains("..")
        || port.map(|p| !(0 < p && p < 65536)).unwrap_or(false)
    {
        return Err(bad());
    }
    let host = if host.starts_with('[') {
        match host[1..host.len() - 1].parse::<Ipv6Addr>() {
            Ok(a) => a.to_string(),
            Err(_) => return Err(bad()),
        }
    } else {
        host
    };
    Ok((canonical_host(&host), port.map(|p| p as u16)))
}

/// realpath(expanduser(d))
fn realpath(p: &str) -> String {
    let expanded = if let Some(rest) = p.strip_prefix('~') {
        let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
        if rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\') {
            format!("{home}{rest}")
        } else {
            p.to_string()
        }
    } else {
        p.to_string()
    };
    real_of(&expanded)
}

/// os.path.realpath: absolute, symlinks resolved as far as the path exists.
pub fn real_of(path: &str) -> String {
    let abs = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    // canonicalize the longest existing prefix, keep the rest
    let mut head = abs.clone();
    let mut rest: Vec<std::ffi::OsString> = vec![];
    loop {
        if let Ok(c) = std::fs::canonicalize(&head) {
            let mut out = c;
            for r in rest.iter().rev() {
                out.push(r);
            }
            return strip_verbatim(&normpath(&out));
        }
        match (head.parent(), head.file_name()) {
            (Some(p), Some(n)) => {
                rest.push(n.to_os_string());
                head = p.to_path_buf();
            }
            _ => return strip_verbatim(&normpath(&abs)),
        }
    }
}

fn normpath(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn strip_verbatim(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    if let Some(r) = s.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{r}");
    }
    s.strip_prefix(r"\\?\").map(|x| x.to_string()).unwrap_or(s)
}

/// image_sources_of: the operator's settings, --image-url-host, --image-dir, --image-remote-only.
pub fn image_sources_of(hosts: &[String], dirs: &[String], remote_only: bool) -> Result<ImageSources, String> {
    let mut out: Vec<String> = vec![];
    for d in dirs {
        let real = realpath(d);
        if !Path::new(&real).is_dir() {
            return Err(format!("--image-dir {d}: not a directory"));
        }
        if !out.contains(&real) {
            out.push(real);
        }
    }
    let mut hs: Vec<(String, Option<u16>)> = vec![];
    for h in hosts {
        let e = parse_url_host(h)?;
        if !hs.contains(&e) {
            hs.push(e);
        }
    }
    Ok(ImageSources { url_hosts: hs, dirs: out, remote_only, ..Default::default() })
}

fn v4_net(a: [u8; 4], bits: u32) -> (u32, u32) {
    let mask = if bits == 0 { 0 } else { u32::MAX << (32 - bits) };
    (u32::from_be_bytes(a) & mask, mask)
}

fn v6_in(ip: u128, net: u128, bits: u32) -> bool {
    let mask = if bits == 0 { 0 } else { u128::MAX << (128 - bits) };
    ip & mask == net & mask
}

/// is_public_address: an address on the public internet, by this module's own tables.
pub fn is_public_address(text: &str) -> bool {
    let Some((ip, scope)) = parse_ip(text) else { return false };
    match ip {
        IpAddr::V4(v4) => {
            let x = u32::from_be_bytes(v4.octets());
            let nets: [([u8; 4], u32); 18] = [
                ([0, 0, 0, 0], 8), ([10, 0, 0, 0], 8), ([100, 64, 0, 0], 10), ([127, 0, 0, 0], 8), ([169, 254, 0, 0], 16),
                ([172, 16, 0, 0], 12), ([192, 0, 0, 0], 24), ([192, 0, 2, 0], 24), ([192, 31, 196, 0], 24),
                ([192, 52, 193, 0], 24), ([192, 88, 99, 0], 24), ([192, 168, 0, 0], 16), ([192, 175, 48, 0], 24),
                ([198, 18, 0, 0], 15), ([198, 51, 100, 0], 24), ([203, 0, 113, 0], 24), ([224, 0, 0, 0], 4),
                ([240, 0, 0, 0], 4),
            ];
            !nets.iter().any(|(a, b)| {
                let (n, m) = v4_net(*a, *b);
                x & m == n
            })
        }
        IpAddr::V6(v6) => {
            let x = u128::from_be_bytes(v6.octets());
            let p = |s: &str| u128::from_be_bytes(s.parse::<Ipv6Addr>().unwrap().octets());
            scope.is_none()
                && v6_in(x, p("2000::"), 3)
                && ![("2001::", 23), ("2001:db8::", 32), ("2002::", 16), ("2620:4f:8000::", 48), ("3fff::", 20)]
                    .iter()
                    .any(|(n, b)| v6_in(x, p(n), *b))
        }
    }
}

fn refused(allowed: &ImageSources, what: &str, detail: &str) -> SourceErr {
    let why = why_remote(&allowed.why);
    SourceErr::refused(
        format!("{POLICY}: {why}, {what}. {DATA_WORKS}"),
        format!("{POLICY}: peer={} why={} {detail}", shown(&allowed.peer), allowed.why).trim_end().to_string(),
    )
}

fn failed(allowed: &ImageSources, message: &str, detail: &str) -> SourceErr {
    SourceErr::refused(message.to_string(), format!("{message}: peer={} {detail}", shown(&allowed.peer)).trim_end().to_string())
}

fn local_failed(allowed: &ImageSources, message: &str, source: &str, reason: &str) -> SourceErr {
    let said = if reason.is_empty() { message.to_string() } else { format!("{message} ({reason})") };
    SourceErr::refused(format!("{said}: {}", shown(source)), format!("{message}: peer={}", shown(&allowed.peer)))
}

/// urlsplit: (scheme, netloc, path...) enough for hostname/port/userinfo.
struct Split {
    scheme: String,
    username: Option<String>,
    password: Option<String>,
    hostname: Option<String>,
    port: Result<Option<u16>, ()>,
}

fn urlsplit(url: &str) -> Result<Split, ()> {
    let (scheme, rest) = match url.find(':') {
        Some(i) if url[..i].chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
            && url[..i].chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)) =>
        {
            (url[..i].to_lowercase(), &url[i + 1..])
        }
        _ => (String::new(), url),
    };
    let netloc = if let Some(r) = rest.strip_prefix("//") {
        let end = r.find(['/', '?', '#']).unwrap_or(r.len());
        &r[..end]
    } else {
        ""
    };
    if (netloc.contains('[') && !netloc.contains(']')) || (netloc.contains(']') && !netloc.contains('[')) {
        return Err(());
    }
    let (userinfo, hostport) = match netloc.rfind('@') {
        Some(i) => (Some(&netloc[..i]), &netloc[i + 1..]),
        None => (None, netloc),
    };
    let (username, password) = match userinfo {
        Some(u) => match u.split_once(':') {
            Some((a, b)) => (Some(a.to_string()), Some(b.to_string())),
            None => (Some(u.to_string()), None),
        },
        None => (None, None),
    };
    let (host, port_s) = if let Some(r) = hostport.strip_prefix('[') {
        let end = r.find(']').ok_or(())?;
        let h = &r[..end];
        let after = &r[end + 1..];
        (h.to_string(), after.strip_prefix(':'))
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p)),
            None => (hostport.to_string(), None),
        }
    };
    let port = match port_s {
        None | Some("") => Ok(None),
        Some(p) => {
            if p.chars().all(|c| c.is_ascii_digit()) {
                match p.parse::<u32>() {
                    Ok(n) if n <= 65535 => Ok(Some(n as u16)),
                    _ => Err(()),
                }
            } else {
                Err(())
            }
        }
    };
    let hostname = if host.is_empty() { None } else { Some(host.to_lowercase()) };
    Ok(Split { scheme, username, password, hostname, port })
}

fn url_host(source: &str) -> String {
    match urlsplit(source) {
        Ok(s) => shown(s.hostname.as_deref().unwrap_or("")),
        Err(_) => "''".into(),
    }
}

// -------------------------------------------------------------------------------------------- data: URLs
fn read_data(source: &str, kind: Kind) -> Result<Vec<u8>, SourceErr> {
    let Some((head, payload)) = source.split_once(',') else {
        return Err(SourceErr::value("a data: URL needs a comma before its payload"));
    };
    if !head.to_lowercase().contains(";base64") {
        return Err(SourceErr::value("a data: URL must be base64"));
    }
    let payload: String = payload.chars().filter(|c| !" \t\r\n\x0c\x0b".contains(*c)).collect();
    if payload.is_empty() {
        return Err(SourceErr::value("a data: URL's payload is empty"));
    }
    if payload.len() as u64 > (kind.max_bytes + 2) / 3 * 4 {
        return Err(SourceErr::value(format!("the {} is larger than {} MiB", kind.noun, kind.max_bytes >> 20)));
    }
    b64decode_validate(&payload).ok_or_else(|| SourceErr::value("a data: URL's payload is not valid base64"))
}

/// base64.b64decode(validate=True): the alphabet then at most two pads; too little padding is an error.
pub fn b64decode_validate(s: &str) -> Option<Vec<u8>> {
    let data = s.trim_end_matches('=');
    let pads = s.len() - data.len();
    if pads > 2 || !data.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/') {
        return None;
    }
    let r = data.len() % 4;
    if r == 1 || (r != 0 && pads < 4 - r) {
        return None;
    }
    let cfg = base64::engine::GeneralPurposeConfig::new()
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::RequireNone)
        .with_decode_allow_trailing_bits(true);
    base64::engine::GeneralPurpose::new(&base64::alphabet::STANDARD, cfg).decode(data).ok()
}

// -------------------------------------------------------------------------------------------- files
static DRIVE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z]:[\\/]").unwrap());
static SLASH_DRIVE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^/[A-Za-z]:[\\/]").unwrap());

fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 0 && i + 2 <= b.len() - 1 + 0 {
            let h = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|x| u8::from_str_radix(x, 16).ok());
            if let Some(v) = h {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// file_url_path: the path of a file: URL.
pub fn file_url_path(source: &str, windows: bool, relative: bool, decode: bool) -> Result<String, String> {
    let mut rest: &str = &source[5..];
    let owned;
    if let Some(r) = rest.strip_prefix("//") {
        rest = r;
        let drive = DRIVE.is_match(rest);
        if rest.chars().take(10).collect::<String>().to_lowercase() == "localhost/" {
            owned = rest[9..].to_string();
            rest = &owned;
        } else if !rest.starts_with('/') && !(windows && drive) && !relative {
            return Err("a file: URL may not name another host".into());
        }
    }
    let mut path = if decode { unquote(rest) } else { rest.to_string() };
    if windows && SLASH_DRIVE.is_match(&path) {
        path = path[1..].to_string();
    }
    Ok(path)
}

const NT_RESERVED_BASE: [&str; 6] = ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"];

fn nt_reserved(name: &str) -> bool {
    if NT_RESERVED_BASE.contains(&name) {
        return true;
    }
    for d in ["COM", "LPT"] {
        if let Some(r) = name.strip_prefix(d) {
            let mut c = r.chars();
            if let (Some(x), None) = (c.next(), c.next()) {
                if x.is_ascii_digit() || "\u{b9}\u{b2}\u{b3}".contains(x) {
                    return true;
                }
            }
        }
    }
    false
}

/// windows_path_refused: forms a remote request may not name on Windows.
pub fn windows_path_refused(path: &str) -> bool {
    if !DRIVE.is_match(path) {
        return true;
    }
    // ntpath.splitdrive of "C:\..." is "C:"
    static BAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"[:*?"<>|\x00-\x1f]"#).unwrap());
    for part in path[3..].split(['\\', '/']) {
        if part.is_empty() || part == "." || part == ".." {
            continue;
        }
        if BAD.is_match(part) || part != part.trim_end_matches(['.', ' ']) {
            return true;
        }
        let stem = part.split('.').next().unwrap_or("").trim_end_matches(' ').to_uppercase();
        if nt_reserved(&stem) {
            return true;
        }
    }
    false
}

enum ReadErr {
    Overflow,
    Io(String),
}

fn read_capped(f: &mut std::fs::File, limit: u64) -> Result<Vec<u8>, ReadErr> {
    let meta = f.metadata().map_err(|e| ReadErr::Io(strerror(&e)))?;
    if !meta.is_file() {
        return Err(ReadErr::Io("Invalid argument".into()));
    }
    if meta.len() > limit {
        return Err(ReadErr::Overflow);
    }
    let mut data = Vec::new();
    f.take(limit + 1).read_to_end(&mut data).map_err(|e| ReadErr::Io(strerror(&e)))?;
    if data.len() as u64 > limit {
        return Err(ReadErr::Overflow);
    }
    Ok(data)
}

/// e.strerror as Python words it for the common cases.
pub fn strerror(e: &std::io::Error) -> String {
    use std::io::ErrorKind::*;
    match e.kind() {
        NotFound => "No such file or directory".into(),
        PermissionDenied => "Permission denied".into(),
        IsADirectory => "Is a directory".into(),
        _ => {
            let s = e.to_string();
            s.split(" (os error").next().unwrap_or(&s).to_string()
        }
    }
}

fn read_local_file(source: &str, allowed: &ImageSources, kind: Kind) -> Result<Vec<u8>, SourceErr> {
    let mut path = source.to_string();
    if source.chars().take(6).collect::<String>().to_lowercase() == "file:/" {
        path = file_url_path(source, cfg!(windows), true, true).map_err(SourceErr::value)?;
        if source.contains('%') {
            let literal = file_url_path(source, cfg!(windows), true, false).map_err(SourceErr::value)?;
            if !(Path::new(&path).exists() || !Path::new(&literal).exists()) {
                path = literal;
            }
        }
    }
    if path.is_empty() {
        return Err(SourceErr::value(kind.usage));
    }
    let noun = kind.noun;
    match std::fs::metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(local_failed(allowed, &format!("the {noun} file does not exist"), &path, ""))
        }
        Err(e) => return Err(local_failed(allowed, &format!("the {noun} file could not be read"), &path, &strerror(&e))),
        Ok(m) if !m.is_file() => return Err(local_failed(allowed, &format!("the {noun} path is not a regular file"), &path, "")),
        Ok(_) => {}
    }
    let mut f = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(local_failed(allowed, &format!("the {noun} file does not exist"), &path, ""))
        }
        Err(e) => return Err(local_failed(allowed, &format!("the {noun} file could not be read"), &path, &strerror(&e))),
    };
    match read_capped(&mut f, kind.max_bytes) {
        Ok(d) => Ok(d),
        Err(ReadErr::Overflow) => {
            Err(local_failed(allowed, &format!("the {noun} is larger than {} MiB", kind.max_bytes >> 20), &path, ""))
        }
        Err(ReadErr::Io(r)) => Err(local_failed(allowed, &format!("the {noun} file could not be read"), &path, &r)),
    }
}

fn normcase(p: &str) -> String {
    if cfg!(windows) {
        p.replace('/', "\\").to_lowercase()
    } else {
        p.to_string()
    }
}

fn within(path: &str, root: &str) -> bool {
    let (p, r) = (normcase(path), normcase(root));
    let sep = if cfg!(windows) { '\\' } else { '/' };
    p == r || p.starts_with(&format!("{}{sep}", r.trim_end_matches(sep)))
}

fn is_abs(p: &str) -> bool {
    if cfg!(windows) {
        // ntpath.isabs: a drive and a root, or a UNC/root-relative path
        DRIVE.is_match(p) || p.starts_with('\\') || p.starts_with('/')
    } else {
        p.starts_with('/')
    }
}

fn read_allowed_file(source: &str, allowed: &ImageSources, kind: Kind) -> Result<Vec<u8>, SourceErr> {
    let windows = cfg!(windows);
    let what = if !allowed.dirs.is_empty() {
        "and the path is not a file under the directories allowed with --image-dir"
    } else {
        "so a file path needs --image-dir on the server"
    };
    let refusal = refused(allowed, what, "kind=path");
    if allowed.dirs.is_empty() {
        return Err(refusal);
    }
    let path = if source.chars().take(6).collect::<String>().to_lowercase() == "file:/" {
        match file_url_path(source, windows, false, true) {
            Ok(p) => p,
            Err(_) => return Err(refusal),
        }
    } else {
        source.to_string()
    };
    if path.contains('\0') || !is_abs(&path) || (windows && windows_path_refused(&path)) {
        return Err(refusal);
    }
    let real = real_of(&path);
    let Some(root) = allowed.dirs.iter().find(|r| within(&real, r)) else { return Err(refusal) };
    if &real == root {
        return Err(refusal);
    }
    // the opened file must be the one the path still names, under the root
    let mut f = match std::fs::File::open(&real) {
        Ok(f) => f,
        Err(_) => return Err(refusal),
    };
    if !within(&real_of(&real), root) {
        return Err(refusal);
    }
    match read_capped(&mut f, kind.max_bytes) {
        Ok(d) => Ok(d),
        Err(ReadErr::Overflow) => {
            Err(failed(allowed, &format!("the {} is larger than {} MiB", kind.noun, kind.max_bytes >> 20), "kind=path"))
        }
        Err(ReadErr::Io(_)) => Err(refusal),
    }
}

// -------------------------------------------------------------------------------------------- URLs
enum Fetch {
    Body(Vec<u8>),
    Status(u16),
    Overflow,
    Failed(String),
}

fn fetch(url: &str, pinned: Option<(String, u16, Vec<String>)>, limit: u64, deadline: Instant) -> Fetch {
    let left = deadline.saturating_duration_since(Instant::now());
    let mut b = ureq::AgentBuilder::new().redirects(0).timeout(left).user_agent(USER_AGENT);
    if let Some((_, port, addrs)) = pinned.clone() {
        b = b.resolver(move |_: &str| -> std::io::Result<Vec<SocketAddr>> {
            Ok(addrs.iter().filter_map(|a| parse_ip(a).map(|(ip, _)| SocketAddr::new(ip, port))).collect())
        });
    }
    let agent = b.build();
    let resp = match agent.get(url).call() {
        Ok(r) => r,
        Err(ureq::Error::Status(code, _)) => return Fetch::Status(code),
        Err(ureq::Error::Transport(t)) => return Fetch::Failed(transport_reason(&t)),
    };
    let code = resp.status();
    if code >= 300 {
        return Fetch::Status(code);
    }
    let length: Option<u64> = resp.header("Content-Length").and_then(|v| v.trim().parse().ok());
    let mut data = Vec::new();
    match resp.into_reader().take(limit + 1).read_to_end(&mut data) {
        Ok(_) => {}
        Err(e) => return Fetch::Failed(e.to_string()),
    }
    if data.len() as u64 > limit {
        return Fetch::Overflow;
    }
    if let Some(l) = length {
        if (data.len() as u64) < l {
            return Fetch::Failed(format!("IncompleteRead({} bytes read, {} more expected)", data.len(), l - data.len() as u64));
        }
    }
    Fetch::Body(data)
}

fn transport_reason(t: &ureq::Transport) -> String {
    t.message().map(|m| m.to_string()).unwrap_or_else(|| t.to_string())
}

fn read_local_url(source: &str, allowed: &ImageSources, kind: Kind, deadline: Instant) -> Result<Vec<u8>, SourceErr> {
    let noun = kind.noun;
    match fetch(source, None, kind.max_bytes, deadline) {
        Fetch::Body(d) => Ok(d),
        Fetch::Overflow => Err(local_failed(allowed, &format!("the {noun} is larger than {} MiB", kind.max_bytes >> 20), source, "")),
        Fetch::Status(c) => Err(local_failed(allowed, &format!("the {noun} URL answered HTTP {c}"), source, "")),
        Fetch::Failed(r) => Err(local_failed(allowed, &format!("the {noun} URL could not be read"), source, &shown(&r))),
    }
}

fn resolve(host: &str, port: u16) -> std::io::Result<Vec<String>> {
    let mut out: Vec<String> = vec![];
    for a in (host, port).to_socket_addrs()? {
        let s = a.ip().to_string();
        if !out.contains(&s) {
            out.push(s);
        }
    }
    Ok(out)
}

fn read_allowed_url(source: &str, allowed: &ImageSources, kind: Kind, deadline: Instant) -> Result<Vec<u8>, SourceErr> {
    let what = if !allowed.url_hosts.is_empty() {
        "and the URL's host and port are not among those allowed with --image-url-host"
    } else {
        "so an http(s) URL needs --image-url-host HOST on the server"
    };
    let refusal = refused(allowed, what, &format!("kind=url host={}", url_host(source)));
    if UNSAFE_URL.is_match(source) {
        return Err(refusal);
    }
    let Ok(parts) = urlsplit(source) else { return Err(refusal) };
    let Ok(port) = parts.port else { return Err(refusal) };
    let host = canonical_host(&parts.hostname.clone().unwrap_or_default().to_lowercase());
    if parts.username.is_some() || parts.password.is_some() || host.is_empty() {
        return Err(refusal);
    }
    let default = if parts.scheme == "https" { 443 } else { 80 };
    if !allowed.url_hosts.iter().any(|(h, p)| *h == host && p.unwrap_or(default) == port.unwrap_or(default)) {
        return Err(refusal);
    }
    let port = port.unwrap_or(default);
    let noun = kind.noun;
    let fetch_failed = |reason: Option<String>| {
        let mut e = failed(allowed, &format!("the {noun} URL could not be fetched"), &format!("host={}", shown(&host)));
        if let Some(r) = reason {
            e.log_line = Some(format!("{} reason={}", e.log_line.unwrap(), shown(&r)));
        }
        e
    };
    let addresses = if is_ip(&host) {
        vec![host.clone()]
    } else {
        let a = match resolve(&host, port) {
            Ok(a) => a,
            Err(e) => return Err(fetch_failed(Some(e.to_string()))),
        };
        if a.is_empty() || !a.iter().all(|x| is_public_address(x)) {
            return Err(SourceErr::refused(
                format!(
                    "{POLICY}: the allowed host {} resolves to a non-public address; name the address itself in --image-url-host to allow it. {DATA_WORKS}",
                    shown(&host)
                ),
                format!("{POLICY}: peer={} host={} non-public", shown(&allowed.peer), shown(&host)),
            ));
        }
        a
    };
    match fetch(source, Some((host.clone(), port, addresses)), kind.max_bytes, deadline) {
        Fetch::Body(d) => Ok(d),
        Fetch::Overflow => Err(failed(allowed, &format!("the {noun} is larger than {} MiB", kind.max_bytes >> 20), &format!("host={}", shown(&host)))),
        Fetch::Status(c) if (300..400).contains(&c) => Err(SourceErr::refused(
            format!(
                "{POLICY}: the {noun} URL redirected, and redirects are not followed for this client; name the final URL's host with --image-url-host on the server. {DATA_WORKS}"
            ),
            format!("{POLICY}: peer={} redirect={c} host={}", shown(&allowed.peer), shown(&host)),
        )),
        Fetch::Status(c) => Err(failed(allowed, &format!("the {noun} URL answered HTTP {c}"), &format!("host={}", shown(&host)))),
        Fetch::Failed(r) => Err(fetch_failed(Some(r))),
    }
}

// -------------------------------------------------------------------------------------------- the one entry
fn bounded(
    read: impl FnOnce(Instant) -> Result<Vec<u8>, SourceErr> + Send + 'static,
    allowed: &ImageSources,
    source: &str,
    kind: Kind,
) -> Result<Vec<u8>, SourceErr> {
    let (tx, rx) = mpsc::sync_channel(1);
    let deadline = Instant::now() + Duration::from_secs_f64(IMAGE_READ_S);
    std::thread::Builder::new()
        .name("image-read".into())
        .spawn(move || {
            let _ = tx.send(read(deadline));
        })
        .map_err(|e| SourceErr::value(e.to_string()))?;
    match rx.recv_timeout(Duration::from_secs_f64(IMAGE_READ_S)) {
        Ok(r) => r,
        Err(_) => {
            let message = format!("the {} source did not answer within {:.0} s", kind.noun, IMAGE_READ_S);
            Err(SourceErr::refused(
                if !allowed.local { message.clone() } else { format!("{message}: {}", shown(source)) },
                format!("{message}: peer={}", shown(&allowed.peer)),
            ))
        }
    }
}

/// load_image: one request image's (or video's) bytes, under the policy; None is the remote one with nothing allowed.
pub fn load_image(source: &serde_json::Value, allowed: Option<&ImageSources>, kind: Kind) -> Result<Vec<u8>, SourceErr> {
    let default = ImageSources::default();
    let allowed = allowed.unwrap_or(&default).clone();
    let Some(source) = source.as_str().filter(|s| !s.is_empty()) else {
        return Err(SourceErr::value(kind.usage));
    };
    let low: String = source.chars().take(32).collect::<String>().to_lowercase();
    if low.starts_with("data:") {
        return read_data(source, kind);
    }
    let s = source.to_string();
    if low.starts_with("http://") || low.starts_with("https://") {
        let a = allowed.clone();
        return bounded(
            move |dl| if a.local { read_local_url(&s, &a, kind, dl) } else { read_allowed_url(&s, &a, kind, dl) },
            &allowed,
            source,
            kind,
        );
    }
    if SCHEME.is_match(&low) && !low.starts_with("file://") {
        return Err(SourceErr::value(kind.usage));
    }
    let a = allowed.clone();
    if allowed.local {
        return bounded(move |_| read_local_file(&s, &a, kind), &allowed, source, kind);
    }
    bounded(move |_| read_allowed_file(&s, &a, kind), &allowed, source, kind)
}

/// The shared bytes of one read source.
pub type Bytes = Arc<Vec<u8>>;

#[allow(dead_code)]
fn _unused(_: &Mutex<()>, _: TcpStream) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts() {
        assert_eq!(parse_url_host("Example.COM:8080").unwrap(), ("example.com".into(), Some(8080)));
        assert_eq!(parse_url_host("[::1]").unwrap(), ("::1".into(), None));
        assert!(parse_url_host("a..b").is_err());
        assert!(parse_url_host("x:0").is_err());
        assert!(is_public_address("8.8.8.8"));
        assert!(!is_public_address("192.168.1.1"));
        assert!(is_public_address("2606:4700::1111"));
        assert!(!is_public_address("fe80::1"));
    }

    #[test]
    fn b64() {
        assert_eq!(b64decode_validate("aGk=").unwrap(), b"hi");
        assert!(b64decode_validate("aGk").is_none());
        assert!(b64decode_validate("a$Gk").is_none());
    }
}
