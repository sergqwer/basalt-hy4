//! serve/security.py: host and origin checks, the allowed names, and the exposure test without an API key.
use std::collections::BTreeSet;
use std::net::{IpAddr, SocketAddr, UdpSocket};

use crate::http::Headers;
use serve_text::pyjson::py_str_repr;

pub const LOOPBACK_NAMES: [&str; 3] = ["localhost", "127.0.0.1", "::1"];

const FORWARDING_HEADERS: [&str; 16] = [
    "forwarded", "via", "x-real-ip", "x-client-ip", "client-ip", "true-client-ip", "x-cluster-client-ip",
    "cf-connecting-ip", "fastly-client-ip", "x-originating-ip", "x-remote-ip", "x-remote-addr", "x-host",
    "x-proxyuser-ip", "wl-proxy-client-ip", "proxy-connection",
];
const FORWARDING_PREFIXES: [&str; 8] = ["x-forwarded-", "x-original-", "x-envoy-", "x-amzn-", "x-azure-", "x-appengine-",
    "cf-", "tailscale-"];

pub fn exposed_without_key(host: &str, api_key: &str) -> bool {
    !LOOPBACK_NAMES.contains(&host) && api_key.is_empty()
}

fn py_isspace_only(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

/// The key as a client can send it; Err when it is only whitespace.
pub fn api_key_of(value: &str) -> Result<String, String> {
    if py_isspace_only(value) {
        return Err("an API key was given but it is empty".into());
    }
    Ok(py_strip(value).to_string())
}

/// The key in the first line of `path`.
pub fn api_key_from_file(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("--api-key-file {path}: {}", py_os_error(&e, path)))?;
    let text = String::from_utf8(bytes).map_err(|e| {
        let pos = e.utf8_error().valid_up_to();
        let b = e.as_bytes()[pos];
        format!("--api-key-file {path}: 'utf-8' codec can't decode byte 0x{b:02x} in position {pos}: invalid start byte")
    })?;
    // readline(): up to and including the first "\n" (universal newlines: \r and \r\n end a line too, as \n)
    let mut line = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            line.push('\n');
            break;
        }
        line.push(c);
        if c == '\n' {
            break;
        }
    }
    if py_strip(&line).is_empty() {
        return Err(format!("--api-key-file {path}: no key in its first line"));
    }
    Ok(line)
}

pub fn py_os_error(e: &std::io::Error, path: &str) -> String {
    let code = e.raw_os_error().unwrap_or(0);
    let msg = match e.kind() {
        std::io::ErrorKind::NotFound => "No such file or directory".to_string(),
        std::io::ErrorKind::PermissionDenied => "Permission denied".to_string(),
        _ => e.to_string(),
    };
    let errno = match e.kind() {
        std::io::ErrorKind::NotFound => 2,
        std::io::ErrorKind::PermissionDenied => 13,
        _ => code,
    };
    format!("[Errno {errno}] {msg}: {}", py_str_repr(path))
}

/// `given` is a header value as http.server read it (each byte one char); the key passes in UTF-8 or Latin-1 form.
pub fn key_matches(given: &str, key: &str) -> bool {
    let raw: Vec<u8> = given.chars().map(|c| if (c as u32) < 256 { c as u32 as u8 } else { b'?' }).collect();
    let mut ok = ct_eq(&raw, key.as_bytes());
    if key.chars().all(|c| (c as u32) < 256) {
        let l1: Vec<u8> = key.chars().map(|c| c as u32 as u8).collect();
        ok |= ct_eq(&raw, &l1);
    }
    ok
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = (a.len() ^ b.len()) as u8 | ((a.len() != b.len()) as u8);
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    diff == 0
}

/// This PC's non-loopback, non-link-local IPv4 addresses.
pub fn lan_addresses() -> Vec<String> {
    let mut first: Option<String> = None;
    if let Ok(s) = UdpSocket::bind("0.0.0.0:0") {
        if s.connect("10.255.255.255:1").is_ok() {
            if let Ok(SocketAddr::V4(a)) = s.local_addr() {
                first = Some(a.ip().to_string());
            }
        }
    }
    let mut ips: BTreeSet<String> = BTreeSet::new();
    if let Some(name) = hostname() {
        use std::net::ToSocketAddrs;
        if let Ok(it) = (name.as_str(), 0u16).to_socket_addrs() {
            for a in it {
                if let IpAddr::V4(v4) = a.ip() {
                    ips.insert(v4.to_string());
                }
            }
        }
    }
    let ok = |ip: &str| !ip.is_empty() && !ip.starts_with("127.") && !ip.starts_with("169.254.") && !ip.starts_with("0.");
    let mut out = Vec::new();
    if let Some(f) = &first {
        if ok(f) {
            out.push(f.clone());
        }
    }
    // sorted() of strings
    for ip in ips {
        if ok(&ip) && Some(&ip) != first.as_ref() {
            out.push(ip);
        }
    }
    out
}

pub fn hostname() -> Option<String> {
    #[cfg(windows)]
    {
        std::env::var("COMPUTERNAME").ok()
    }
    #[cfg(unix)]
    {
        let mut buf = [0u8; 256];
        let r = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
        if r != 0 {
            return None;
        }
        let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Some(String::from_utf8_lossy(&buf[..n]).into_owned())
    }
}

fn py_isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// The lowercase name of a Host header or origin host[:port]; "" when malformed.
pub fn host_name(value: Option<&str>) -> String {
    let v = py_strip(value.unwrap_or("")).to_lowercase();
    if let Some(rest0) = v.strip_prefix('[') {
        return match rest0.split_once(']') {
            Some((name, rest)) if rest.is_empty() || (rest.starts_with(':') && py_isdigit(&rest[1..])) => name.to_string(),
            _ => String::new(),
        };
    }
    let mut v = v;
    if v.matches(':').count() == 1 {
        let (h, port) = v.split_once(':').unwrap();
        if !py_isdigit(port) {
            return String::new();
        }
        v = h.to_string();
    } else if v.contains(':') {
        return v;
    }
    let v = v.trim_end_matches('.');
    if !v.is_empty() && v.chars().all(|c| c.is_alphanumeric() || "-._".contains(c)) {
        v.to_string()
    } else {
        String::new()
    }
}

/// ipaddress.ip_address(name) parses
pub fn is_ip(name: &str) -> bool {
    parse_ip(name).is_some()
}

fn parse_ip(name: &str) -> Option<IpAddr> {
    // Python rejects IPv4 with leading zeros ("01.2.3.4"); so does Rust's parser; scope ids ("fe80::1%eth0") pass in Python
    let base = name.split('%').next().unwrap_or(name);
    if name.contains('%') && !base.contains(':') {
        return None;
    }
    base.parse::<IpAddr>().ok()
}

fn name_in(name: &str, names: &BTreeSet<String>) -> bool {
    names.contains(name) || names.iter().any(|n| n.starts_with('.') && (name.ends_with(n.as_str()) || name == &n[1..]))
}

pub fn allowed_hosts_of(items: &[String]) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for raw in items {
        let x = py_strip(raw).to_lowercase();
        if x == "*" {
            if !out.contains(&x) {
                out.push(x);
            }
            continue;
        }
        let x = x.rsplit_once("://").map(|(_, b)| b.to_string()).unwrap_or(x.clone());
        let x = x.split_once("://").map(|(_, b)| b.to_string()).unwrap_or(x);
        let x = x.split('/').next().unwrap_or("").to_string();
        let dot = x.starts_with('.');
        let name = host_name(Some(if dot { &x[1..] } else { &x }));
        if name.is_empty() {
            return Err(format!("allowed_hosts: {} is not a host name", py_str_repr(raw)));
        }
        let v = if dot { format!(".{name}") } else { name };
        if !out.contains(&v) {
            out.push(v);
        }
    }
    Ok(out)
}

pub fn host_names_for(bind_host: &str, allowed_hosts: &[String], trusted_origins: &[String]) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = LOOPBACK_NAMES.iter().map(|s| s.to_string()).collect();
    let bind = host_name(Some(bind_host));
    if !bind.is_empty() && bind != "0.0.0.0" && bind != "::" {
        names.insert(bind.clone());
    }
    if !LOOPBACK_NAMES.contains(&bind.as_str()) {
        if let Some(pc) = hostname() {
            let pc = pc.to_lowercase();
            names.insert(format!("{pc}.local"));
            names.insert(pc);
        }
        names.insert("host.docker.internal".into());
        names.extend(lan_addresses());
    }
    names.extend(allowed_hosts.iter().filter(|x| *x != "*").cloned());
    for o in trusted_origins {
        let rest = o.split_once("://").map(|(_, b)| b).unwrap_or(o);
        let n = host_name(Some(rest));
        if !n.is_empty() {
            names.insert(n);
        }
    }
    names
}

pub fn host_allowed(host: Option<&str>, names: &BTreeSet<String>, any_host: bool) -> bool {
    if any_host || py_strip(host.unwrap_or("")).is_empty() {
        return true;
    }
    let name = host_name(host);
    !name.is_empty() && (is_ip(&name) || name == "localhost" || name.ends_with(".localhost") || name_in(&name, names))
}

pub fn origin_allowed(origin: &str, host: Option<&str>, names: &BTreeSet<String>, origins: &[String]) -> bool {
    let origin = py_strip(origin).trim_end_matches('/').to_string();
    if origins.iter().any(|o| *o == origin || o == "*") {
        return true;
    }
    let low = origin.to_lowercase();
    let Some((scheme, rest)) = low.split_once("://") else { return false };
    if scheme.is_empty() {
        return false;
    }
    if scheme != "http" && scheme != "https" {
        return true;
    }
    if let Some(h) = host {
        if !h.is_empty() && rest == py_strip(h).to_lowercase() {
            return true;
        }
    }
    let name = host_name(Some(rest));
    !name.is_empty() && (LOOPBACK_NAMES.contains(&name.as_str()) || name.ends_with(".localhost") || name_in(&name, names))
}

pub fn peer_of(addr: Option<&SocketAddr>) -> String {
    addr.map(|a| a.ip().to_string()).unwrap_or_default()
}

pub fn loopback_peer(address: &str) -> bool {
    match parse_ip(address) {
        Some(IpAddr::V4(v)) => v.is_loopback(),
        Some(IpAddr::V6(v)) => match v.to_ipv4_mapped() {
            Some(m) => m.is_loopback(),
            None => v.is_loopback(),
        },
        None => false,
    }
}

fn loopback_host(host: &str) -> bool {
    let name = host_name(Some(host));
    name == "localhost" || loopback_peer(&name) || name == "0.0.0.0" || name == "::"
}

/// "" when a request is provably from this machine, else why not.
pub fn remote_reason(addr: Option<&SocketAddr>, headers: Option<&Headers>) -> String {
    let peer = peer_of(addr);
    if peer.is_empty() {
        return "no-peer".into();
    }
    if !loopback_peer(&peer) {
        return "peer".into();
    }
    let Some(h) = headers else { return "headers".into() };
    if h.defect {
        return "headers".into();
    }
    let forwarded = h.items.iter().any(|(k, _)| {
        let n = k.to_lowercase().replace('_', "-");
        FORWARDING_HEADERS.contains(&n.as_str()) || FORWARDING_PREFIXES.iter().any(|p| n.starts_with(p))
    });
    if forwarded {
        return "forwarded".into();
    }
    let hosts = h.get_all("Host");
    if hosts.len() != 1 || !loopback_host(hosts[0]) {
        return "host".into();
    }
    if !h.get_all("Origin").is_empty() || h.get_all("Sec-Fetch-Site").iter().any(|v| py_strip(v).to_lowercase() != "none") {
        return "origin".into();
    }
    String::new()
}

/// An origin or list of origins; Err is the SystemExit text.
pub fn origins_of(items: &[String], key: &str, wildcard: bool) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for x in items {
        let x = py_strip(x).trim_end_matches('/').to_string();
        if x == "*" && wildcard {
            out.push(x);
            continue;
        }
        let (scheme, sep, rest) = match x.split_once("://") {
            Some((a, b)) => (a.to_string(), true, b.to_string()),
            None => (x.clone(), false, String::new()),
        };
        if (scheme != "http" && scheme != "https") || !sep || rest.is_empty() || rest.contains('/') || rest.contains('*') {
            return Err(format!("[server] error: {key}: {} is not an origin like https://chat.example.com{}",
                               py_str_repr(&x), if wildcard { "" } else { " (no wildcards)" }));
        }
        out.push(x);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names() {
        assert_eq!(host_name(Some("[::1]:8095")), "::1");
        assert_eq!(host_name(Some("Example.COM:80")), "example.com");
        assert_eq!(host_name(Some("a:b")), "");
        assert_eq!(host_name(Some("::1")), "::1");
        let n: BTreeSet<String> = ["localhost".to_string(), ".ex.com".to_string()].into_iter().collect();
        assert!(host_allowed(Some("a.ex.com"), &n, false));
        assert!(host_allowed(Some("10.0.0.1:5"), &n, false));
        assert!(!host_allowed(Some("evil.com"), &n, false));
        assert!(origin_allowed("http://localhost:3000", None, &n, &[]));
        assert!(!origin_allowed("null", None, &n, &[]));
        assert!(origin_allowed("vscode-webview://x", None, &n, &[]));
        assert!(key_matches("abc", "abc"));
        assert!(!key_matches("abd", "abc"));
        assert_eq!(api_key_of("  k ").unwrap(), "k");
        assert!(api_key_of("  ").is_err());
        assert!(loopback_peer("::ffff:127.0.0.1"));
    }
}
