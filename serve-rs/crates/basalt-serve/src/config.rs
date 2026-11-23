//! serve/config.py: the command-line parser, the --config file and engine flag checks.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use serve_engine::config_checks::{num, whole, GRAMMAR_DEFAULT, SAMPLING_ORDER};
use serve_engine::flags::flag_index;
use serve_engine::util::console::tagged;
use serve_text::pyjson::{self, py_repr, py_str_repr};

use crate::argparse::{invalid, Arg, Exit, Kind, Namespace, Parser};

pub const MIN_SLOT_CONTEXT: i64 = 256;
pub const CONFIG_VERSION: i64 = 2;
pub const PARALLEL_MAX: i64 = 8;
pub const CONFIG_PATHS: [&str; 3] = ["model", "tokenizer", "image-dir"];
pub const ENTRY_OPTIONS: [&str; 2] = ["model", "alias"];
/// A model entry's keys for the launcher (basalt serve fetches a missing model: tools/get_model.py); the server skips them.
pub const LAUNCHER_KEYS: [&str; 2] = ["hf", "download"];
pub const MODEL_OPTIONS: [&str; 24] = [
    "gpu", "vision", "parallel", "slot-full-context", "idle-sleep", "idle-unload", "min-free-vram-mib", "fit-max-tokens",
    "grammar", "reasoning-effort", "reasoning-budget-tokens", "thinking-loop-tokens", "answer-room", "temperature",
    "top-p", "top-k", "min-p", "presence-penalty", "frequency-penalty", "repetition-penalty", "penalty-last-n", "seed",
    "video-max-tokens", "idle-sleep",
];
pub const DESCRIPTION: &str = "Server settings: the command-line parser, the --config file and engine flag checks.";

// ---- serve/engine/vision.py and image_source.py pieces the parser needs (TODO(serve-api): use theirs)
pub const VIDEO_MAX_TOKENS: i64 = 8192;
pub const VISION_MODES: [&str; 3] = ["off", "dynamic", "cpu"];

pub fn vision_usage() -> String {
    format!("vision: use one of {} or a GPU number, as nvidia-smi numbers them", VISION_MODES.join(", "))
}

/// vision_mode(value): one of VISION_MODES or a GPU number as a string; Err for anything else.
pub fn vision_mode(value: &Value) -> Result<String, String> {
    if value.is_null() {
        return Ok("off".into());
    }
    let text = pyjson::py_str(value).trim().to_lowercase();
    if VISION_MODES.contains(&text.as_str()) {
        return Ok(text);
    }
    if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) {
        let t = text.trim_start_matches('0');
        return Ok(if t.is_empty() { "0".into() } else { t.to_string() });
    }
    Err(format!("{} (got {})", vision_usage(), py_repr(value)))
}

pub fn parse_url_host(text: &str) -> Result<(String, Option<u16>), String> {
    let low = text.trim().to_lowercase();
    let bad = || format!("expected HOST or HOST:PORT, not {}", py_str_repr(text));
    let re = regex::Regex::new(r"^(\[[0-9a-f:.]+\]|[a-z0-9._-]+)(?::([0-9]{1,5}))?$").unwrap();
    let (host, port) = match re.captures(&low) {
        Some(c) => (c.get(1).unwrap().as_str().to_string(), c.get(2).map(|p| p.as_str().parse::<u32>().unwrap())),
        None => (String::new(), None),
    };
    if host.is_empty() || host.starts_with('.') || host.ends_with('.') || host.contains("..")
        || port.map(|p| !(0 < p && p < 65536)).unwrap_or(false) {
        return Err(bad());
    }
    let host = if host.starts_with('[') {
        match host[1..host.len() - 1].parse::<std::net::Ipv6Addr>() {
            Ok(a) => a.to_string(),
            Err(_) => return Err(bad()),
        }
    } else {
        host.parse::<std::net::IpAddr>().map(|a| a.to_string()).unwrap_or(host)
    };
    Ok((host, port.map(|p| p as u16)))
}

// ---- argparse types
fn int_text(text: &str) -> Option<i64> {
    // Python int(): whitespace allowed, underscores between digits, a sign
    let t = text.trim();
    let t2 = t.replace('_', "");
    if t2.is_empty() || t.starts_with('_') || t.ends_with('_') || t.contains("__") {
        return None;
    }
    t2.parse::<i64>().ok()
}

fn t_int(text: &str) -> Result<Value, String> {
    int_text(text).map(Value::from).ok_or_else(invalid)
}

fn t_float(text: &str) -> Result<Value, String> {
    serve_engine::py::float_of(text).map(|x| json!(x)).ok_or_else(invalid)
}

fn t_lower(text: &str) -> Result<Value, String> {
    Ok(Value::String(text.to_lowercase()))
}

fn t_vision(text: &str) -> Result<Value, String> {
    vision_mode(&Value::String(text.into())).map(Value::String)
}

fn t_url_host(text: &str) -> Result<Value, String> {
    parse_url_host(text)?;
    Ok(Value::String(text.trim().to_string()))
}

fn t_parallel(text: &str) -> Result<Value, String> {
    let n = int_text(text).ok_or_else(|| format!("expected a whole number 1..{PARALLEL_MAX}, not {}", py_str_repr(text)))?;
    if !(1..=PARALLEL_MAX).contains(&n) {
        return Err(format!("expected 1..{PARALLEL_MAX}, not {n}"));
    }
    Ok(json!(n))
}

fn t_budget(text: &str) -> Result<Value, String> {
    let n = int_text(text).ok_or_else(|| format!("expected a whole number >= 0, not {}", py_str_repr(text)))?;
    if n < 0 {
        return Err(format!("expected a whole number >= 0, not {n}"));
    }
    Ok(json!(n))
}

pub fn answer_room(text: &str) -> Result<Value, String> {
    if text == "auto" {
        return Ok(json!("auto"));
    }
    let n = int_text(text)
        .ok_or_else(|| format!("expected auto or a whole number of tokens >= 0, not {}", py_str_repr(text)))?;
    if n < 0 {
        return Err(format!("expected auto or a whole number of tokens >= 0, not {n}"));
    }
    Ok(json!(n))
}

fn t_video(text: &str) -> Result<Value, String> {
    let n = int_text(text).ok_or_else(|| format!("expected a whole number from 64 to 65536, not {}", py_str_repr(text)))?;
    if !(64..=65536).contains(&n) {
        return Err(format!("expected a whole number from 64 to 65536, not {n}"));
    }
    Ok(json!(n))
}

/// sampling key -> (is int, test, expected)
pub fn sampling_rule(key: &str) -> Option<(bool, fn(&Value) -> bool, String)> {
    serve_engine::config_checks::sampling_rule(key)
}

fn sampling_parse(key: &str, text: &str) -> Result<Value, String> {
    let (is_int, ok, expected) = sampling_rule(key).unwrap();
    let v = serve_engine::py::float_of(text).ok_or_else(|| format!("expected {expected}, not {}", py_str_repr(text)))?;
    let val = json!(v);
    if !ok(&val) {
        return Err(format!("expected {expected}, not {}", py_str_repr(text)));
    }
    Ok(if is_int { json!(v as i64) } else { val })
}

macro_rules! sampling_type {
    ($name:ident, $key:expr) => {
        fn $name(text: &str) -> Result<Value, String> {
            sampling_parse($key, text)
        }
    };
}
sampling_type!(s_temperature, "temperature");
sampling_type!(s_top_p, "top_p");
sampling_type!(s_top_k, "top_k");
sampling_type!(s_min_p, "min_p");
sampling_type!(s_presence, "presence_penalty");
sampling_type!(s_frequency, "frequency_penalty");
sampling_type!(s_repetition, "repetition_penalty");
sampling_type!(s_last_n, "penalty_last_n");
sampling_type!(s_seed, "seed");

/// The repository root (serve/ and pack/ live under it): BASALT_ROOT, else the nearest parent of the binary that
/// holds serve/web/index.html, else the working directory.
pub fn root() -> PathBuf {
    if let Ok(r) = std::env::var("BASALT_ROOT") {
        if !r.is_empty() {
            return PathBuf::from(r);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut d = exe.parent().map(|p| p.to_path_buf());
        while let Some(p) = d {
            if p.join("serve").join("web").join("index.html").is_file() {
                return p;
            }
            d = p.parent().map(|x| x.to_path_buf());
        }
    }
    std::env::current_dir().unwrap_or_default()
}

pub fn build_parser() -> Parser {
    let mut ap = Parser::new("server.py", DESCRIPTION);
    let pm = PARALLEL_MAX;
    ap.add(Arg::new("--mock-engine").kind(Kind::StoreTrue)
        .help("run a scripted engine instead of the model, for clients and tests (no --model needed)"));
    ap.add(Arg::new("--model").help("the .basalt model file: the engine, tokenizer, sampling and vision come from it"));
    ap.add(Arg::new("--engine-bin").suppress());
    ap.add(Arg::new("--alias").help("the model name the API reports (default: the name stored in the model file)"));
    ap.add(Arg::new("--parallel").ty(t_parallel, "parallel_count").metavar("N").help(&format!(
        "requests served at once, 1..{pm} (default 1: one at a time); more requests wait their turn. --max-context is \
         shared: each slot holds that divided by N (so 4 slots of 262144 need --max-context 1048576, or \
         --slot-full-context). The slots' KV cache comes out of the expert cache")));
    ap.add(Arg::new("--slot-full-context").kind(Kind::StoreTrue).help(
        "with --parallel, give every slot the whole --max-context instead of an equal part: N slots cost N times the KV cache"));
    ap.add(Arg::new("--config").metavar("FILE").help(
        "a JSON file {\"server\": {...}, \"engine\": {...}, \"models\": [{\"model\": FILE, ...}]}: settings named without \
         the dashes. \"server\" holds what the server shares, \"engine\" what applies to every model that takes it, a \
         model entry that model's own; a list gives a flag several values; the command line wins over it, for every model"));
    ap.add(Arg::new("--vision").nargs_optional("").default(json!("off")).ty(t_vision, "vision_setting").metavar("MODE|GPU").help(
        "image input: off (default), dynamic (inside the engine: it borrows expert-cache slots on the main GPU while it \
         reads an image, no standing VRAM), cpu (no VRAM, slow) or a GPU number, as nvidia-smi numbers them: the encoder \
         sidecar, always on that card"));
    ap.add(Arg::new("--image-url-host").kind(Kind::Append).default(json!([])).metavar("HOST[:PORT]").ty(t_url_host, "image_url_host").help(
        "let requests from other machines name http(s) image URLs on exactly this host (repeatable; without a port, 80 \
         and 443). A name must resolve to public addresses; an IP address is taken as written. Requests from this \
         machine need no flag"));
    ap.add(Arg::new("--image-dir").kind(Kind::Append).default(json!([])).metavar("DIR").help(
        "let requests from other machines name image files under this directory (repeatable). Requests from this machine need no flag"));
    ap.add(Arg::new("--image-remote-only").kind(Kind::StoreTrue).help(
        "treat requests from this machine as remote too, so they get only data: URLs and what --image-url-host and \
         --image-dir allow. For a server behind anything on this machine that forwards other people's requests"));
    ap.add(Arg::new("--host").default(json!("127.0.0.1")).help(
        "the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other devices on your network (give --api-key)"));
    ap.add(Arg::new("--script").kind(Kind::Append).help(
        "the mock engine's answer (default: a short greeting); given more than once, requests get them in turn and the last one repeats"));
    ap.add(Arg::new("--port").ty(t_int, "int").default(json!(8080)).help("the TCP port to listen on (default 8080)"));
    ap.add(Arg::new("--gpu").help(
        "the GPUs to use, as nvidia-smi numbers them: exactly these, the first is the main card and the engine gives the \
         rest their roles (\"0,1\"); auto = every Blackwell card, the fastest as main (default: GPU 0 alone)"));
    let tok_default = root().join("pack/full/tokenizer");
    ap.add(Arg::new("--tokenizer").default(json!(py_path(&tok_default)))
        .help("tokenizer directory for the mock engine (a byte tokenizer if absent)"));
    ap.add(Arg::new("--open").kind(Kind::StoreTrue).help("open the local page in the browser once the model is ready"));
    ap.add(Arg::new("--fit-max-tokens").kind(Kind::StoreTrue)
        .help("clamp max_tokens to the remaining context instead of rejecting the request with 400"));
    ap.add(Arg::new("--grammar").kind(Kind::BoolOpt).default(json!(GRAMMAR_DEFAULT)).help(
        "hold every tool call and JSON response format to a grammar in the engine (on by default; a request with strict: \
         true gets it either way); --no-grammar turns it off"));
    ap.add(Arg::new("--api-key").default(json!("")).help(
        "require this key on /v1/* (Authorization: Bearer ... or x-api-key); command line only, not allowed in a --config file"));
    ap.add(Arg::new("--api-key-file").default(json!("")).metavar("FILE").help(
        "read that key from the first line of FILE, which keeps it out of the process list; command line only"));
    ap.add(Arg::new("--allowed-host").kind(Kind::Append).default(json!([])).metavar("NAME")
        .help("also answer to this Host name (repeatable; \"*\" turns the check off)"));
    ap.add(Arg::new("--cors-origin").kind(Kind::Append).default(json!([])).metavar("ORIGIN")
        .help("let web pages from this origin use /v1/* (repeatable)"));
    ap.add(Arg::new("--max-payload-mib").ty(t_int, "int").default(json!(256)).metavar("MIB")
        .help("the largest request body the server reads, in MiB; a bigger one gets a 413 (default 256)"));
    ap.add(Arg::new("--debug").kind(Kind::StoreTrue).help("log one line per request and the raw model text after each reply"));
    ap.add(Arg::new("--request-lines").kind(Kind::StoreTrue)
        .help("echo the engine's per-request summary on stdout, for a supervisor that only sees this output"));
    ap.add(Arg::new("--claude-code-prompt").choices(&["local", "as_sent"]).default(json!("local")).help(
        "Claude Code's system prompt: local (the default) drops its Claude-only parts (identity, Anthropic's security          policy, model ids, pronoun rule) and shortens its memory how-to, the same text every turn; as_sent passes it as sent"));
    ap.add(Arg::new("--bang-restart-tokens").ty(t_budget, "budget_count").default(json!(256)).metavar("N").help(
        "a reply that repeats token 0 this many times in a row (\"!\" on most vocabularies: what the sampler answers \
         when no logit is finite) ends there as length, and the next request starts the engine again with fresh state \
         (0 = never)"));
    ap.add(Arg::new("--stop-token-text").kind(Kind::BoolOpt).default(json!(true)).help(
        "a stop token the model writes where no turn can end (inside a tool call's argument, after an opening backtick          or quote, after a space, closing a turn written out as an example) is written as its text and the reply goes          on (on by default); --no-stop-token-text ends the reply there"));
    ap.add(Arg::new("--lazy").kind(Kind::StoreTrue)
        .help("start the API unloaded; load on first request; a sidecar image encoder is dropped, --vision dynamic stays"));
    ap.add(Arg::new("--idle-unload").ty(t_float, "float").metavar("SECONDS")
        .help("unload the model after this many seconds without requests; the next request loads it again"));
    ap.add(Arg::new("--idle-sleep").ty(t_float, "float").metavar("SECONDS").help(
        "put the model to sleep after this many seconds without requests: its VRAM is freed, its RAM copy, conversations \
         and learned expert placement stay, and the next request wakes it in a few seconds (Linux; the driver suspends \
         the engine's GPU state)"));
    let samp: [(&str, &str, crate::argparse::TypeFn); 9] = [
        ("temperature", "0 = greedy; higher is more random", s_temperature),
        ("top-p", "nucleus cutoff, 0 < value <= 1 (1 = off)", s_top_p),
        ("top-k", "keep the K likeliest tokens (0 = off: every token)", s_top_k),
        ("min-p", "drop tokens less likely than this times the likeliest, 0..1 (0 = off)", s_min_p),
        ("presence-penalty", "flat penalty on a token seen in the window", s_presence),
        ("frequency-penalty", "penalty growing with how often a token was seen", s_frequency),
        ("repetition-penalty", "multiplicative penalty on seen tokens, > 0 (1 = off)", s_repetition),
        ("penalty-last-n", "the window the penalties look back over, in tokens (at most 4096)", s_last_n),
        ("seed", "fix the random draw, an integer", s_seed),
    ];
    for (key, text, f) in samp {
        let k = key.replace('-', "_");
        ap.add(Arg::new(&format!("--{key}")).ty(f, "parse").metavar("N")
            .help(&format!("default {k} for requests that leave it out, over the model file's: {text}")));
    }
    ap.add(Arg::new("--min-free-vram-mib").ty(t_int, "int")
        .help("load an unloaded model only when this much VRAM is free, else answer 503"));
    ap.add(Arg::new("--reasoning-effort").ty(t_lower, "lower").choices(&["none", "low", "medium", "high", "xhigh", "max"]).help(
        "default reasoning effort for requests that name none (no effort, thinking switch or budget); unset, the model's template decides"));
    ap.add(Arg::new("--reasoning-budget-tokens").ty(t_budget, "budget_count").metavar("N").help(
        "default thinking budget for requests that leave reasoning_budget_tokens out (0 = none): at the budget the \
         thinking is closed and the answer starts, so set it below the request's max_tokens"));
    ap.add(Arg::new("--thinking-loop-tokens").ty(t_budget, "budget_count").default(json!(256)).metavar("N").help(
        "end the thinking and let the model answer when it repeats one token this many times in a row inside the thinking (0 = never)"));
    ap.add(Arg::new("--video-max-tokens").ty(t_video, "video_tokens").default(json!(VIDEO_MAX_TOKENS)).metavar("N").help(&format!(
        "the most image tokens one video may take over all its frames (default {VIDEO_MAX_TOKENS}): frames are sampled at \
         2 a second (4 to 768; a request's video part may ask fps and max_frames) and scaled down together to fit")));
    ap.add(Arg::new("--answer-room").ty(answer_room, "answer_room").default(json!("auto")).metavar("auto|N").help(
        "output tokens kept for closing the thinking and the answer: auto (a quarter of the cap, at least 512, at most \
         half) or a whole number of tokens (0 = none); an explicit request budget of 0 opts out"));
    ap
}

/// str(Path): Windows paths with backslashes
pub fn py_path(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    if cfg!(windows) { s.replace('/', "\\") } else { s }
}

/// The parsed settings of one model (Python's Namespace plus what parse_settings adds).
#[derive(Debug, Clone)]
pub struct Args {
    pub ns: Namespace,
    pub skipped: Vec<String>,
    pub sampling_launch: Map<String, Value>,
    pub sampling_cli: BTreeSet<String>,
    pub split_default: bool,
    pub shared_context: Option<i64>,
}

impl Args {
    pub fn get(&self, k: &str) -> &Value {
        self.ns.get(k).unwrap_or(&Value::Null)
    }
    pub fn s(&self, k: &str) -> Option<String> {
        self.get(k).as_str().map(|s| s.to_string())
    }
    /// truthiness of a string option
    pub fn st(&self, k: &str) -> Option<String> {
        self.s(k).filter(|s| !s.is_empty())
    }
    pub fn b(&self, k: &str) -> bool {
        serve_text::pyjson::truthy(self.get(k))
    }
    pub fn i(&self, k: &str) -> Option<i64> {
        self.get(k).as_i64()
    }
    pub fn f(&self, k: &str) -> Option<f64> {
        self.get(k).as_f64()
    }
    pub fn list(&self, k: &str) -> Vec<String> {
        match self.get(k) {
            Value::Array(a) => a.iter().map(pyjson::py_str).collect(),
            _ => vec![],
        }
    }
    pub fn set(&mut self, k: &str, v: Value) {
        self.ns.insert(k.into(), v);
    }
}

/// (args, argv, file_words, cli_words) of one model
pub type Setting = (Args, Vec<String>, Vec<String>, Vec<String>);

/// A start-up failure: argparse's exit, or a ValueError text (`--config: ...` is added by the caller).
#[derive(Debug)]
pub enum ConfigErr {
    Exit(Exit),
    Value(String),
    /// SystemExit(text): printed to stderr, status 1
    System(String),
}

impl From<Exit> for ConfigErr {
    fn from(e: Exit) -> Self {
        ConfigErr::Exit(e)
    }
}

fn verr<T>(s: String) -> Result<T, ConfigErr> {
    Err(ConfigErr::Value(s))
}

pub fn grammar_words(known: Option<&BTreeSet<String>>) -> Vec<String> {
    match known {
        Some(k) if k.contains("--grammar") => vec!["--grammar".into(), "on".into()],
        _ => vec![],
    }
}

/// The long flags the engine binary accepts; None when it cannot be read.
pub fn engine_flags_of(engine_bin: &str) -> Option<BTreeSet<String>> {
    let data = std::fs::read(engine_bin).ok()?;
    let printable = |b: u8| (0x21..=0x7e).contains(&b);
    let mut out = BTreeSet::new();
    let n = data.len();
    let mut i = 0;
    while i + 2 < n {
        if data[i] == b'-' && data[i + 1] == b'-' && (i == 0 || !printable(data[i - 1]))
            && (data[i + 2].is_ascii_lowercase() || data[i + 2].is_ascii_digit()) {
            let mut j = i + 3;
            while j < n && (data[j].is_ascii_lowercase() || data[j].is_ascii_digit() || data[j] == b'-') {
                j += 1;
            }
            if j >= n || !printable(data[j]) {
                out.insert(String::from_utf8_lossy(&data[i..j]).into_owned());
            } else {
                // the regex backtracks: a shorter match must also be followed by a non-printable byte; none is
            }
            i = j;
            continue;
        }
        i += 1;
    }
    Some(out)
}

pub fn split_engine_extras(extras: &[String]) -> Result<Vec<String>, String> {
    if let Some(first) = extras.first() {
        if !(first.starts_with("--") && first.len() > 2) {
            return Err(format!("unexpected {}", py_str_repr(first)));
        }
    }
    let mut out = Vec::new();
    for word in extras {
        let (flag, eq, value) = serve_engine::py::partition(word, "=");
        if eq && word.starts_with("--") && flag.len() > 2 {
            out.push(flag.to_string());
            out.push(value.to_string());
        } else {
            out.push(word.clone());
        }
    }
    Ok(out)
}

pub fn config_name(key: &str) -> String {
    let s = key.trim().to_lowercase().replace('_', "-");
    s.strip_prefix("--").map(|x| x.to_string()).unwrap_or(s)
}

pub fn engine_words(key: &str, value: &Value) -> Result<Vec<String>, String> {
    let name = config_name(key);
    let flag = format!("--{name}");
    if let Value::Bool(b) = value {
        return Ok(if *b { vec![flag] } else { vec![] });
    }
    let form: Option<[&str; 2]> = if name == "control-vector-layer-range" { Some(["start", "end"]) } else { None };
    if let Some(form) = form {
        if !matches!(value, Value::Array(a) if a.len() == form.len()) {
            return Err(format!("{name} must be a list [{}], got {}", form.join(", "), pyjson::dumps(value, true)));
        }
    } else if value.is_array() {
        return Err(format!("{name} takes one value, got {}", pyjson::dumps(value, true)));
    }
    let items: Vec<Value> = match value {
        Value::Array(a) => a.clone(),
        v => vec![v.clone()],
    };
    for item in &items {
        let ok = matches!(item, Value::String(_) | Value::Number(_)) && !pyjson::py_str(item).trim().is_empty();
        if !ok {
            return Err(format!("engine.{key}: expected a string or number, or true/false for a switch, not {}", py_repr(value)));
        }
    }
    let mut out = vec![flag];
    out.extend(items.iter().map(pyjson::py_str));
    Ok(out)
}

#[derive(Debug, Clone)]
struct ModelEntry {
    path: String,
    argv: Vec<String>,
    shared_words: Vec<String>,
    own_words: Vec<String>,
}

fn join_path(here: &Path, item: &str) -> String {
    let expanded = expand_user(item);
    py_path(&here.join(expanded))
}

fn expand_user(p: &str) -> String {
    if p == "~" || p.starts_with("~/") || p.starts_with("~\\") {
        if let Some(h) = home_dir() {
            return format!("{}{}", h, &p[1..]);
        }
    }
    p.to_string()
}

fn home_dir() -> Option<String> {
    if cfg!(windows) {
        std::env::var("USERPROFILE").ok().or_else(|| {
            let d = std::env::var("HOMEDRIVE").ok()?;
            let p = std::env::var("HOMEPATH").ok()?;
            Some(format!("{d}{p}"))
        })
    } else {
        std::env::var("HOME").ok()
    }
}

fn option_words(where_: &str, key: &str, value: &Value, act: &Arg, here: &Path) -> Result<Vec<String>, String> {
    let name = config_name(key);
    let flag = format!("--{name}");
    if name == "vision" {
        return vision_mode(value).map(|m| vec!["--vision".into(), m]).map_err(|e| format!("{where_}.{key}: {e}"));
    }
    if name == "answer-room" {
        if !matches!(value, Value::String(_)) && !(matches!(value, Value::Number(n) if n.is_i64() || n.is_u64())) {
            return Err(format!("{where_}.{key}: \"auto\" or a whole number of tokens (0 = none), not {}", py_repr(value)));
        }
        return answer_room(&pyjson::py_str(value))
            .map(|v| vec![flag.clone(), pyjson::py_str(&v)])
            .map_err(|e| format!("{where_}.{key}: {e}"));
    }
    if matches!(act.kind, Kind::StoreTrue | Kind::BoolOpt) {
        let Value::Bool(b) = value else { return Err(format!("{where_}.{key}: true or false")) };
        return Ok(if *b { vec![flag] } else if act.kind == Kind::BoolOpt { vec![format!("--no-{name}")] } else { vec![] });
    }
    let items: Vec<Value> = match value {
        Value::Array(a) => a.clone(),
        v => vec![v.clone()],
    };
    if value.is_array() && act.kind != Kind::Append {
        return Err(format!("{where_}.{key}: one value, not a list"));
    }
    let mut words = Vec::new();
    for item in &items {
        if !matches!(item, Value::String(_) | Value::Number(_)) {
            return Err(format!("{where_}.{key}: expected a string or number, not {}", py_repr(item)));
        }
        let mut s = pyjson::py_str(item);
        if CONFIG_PATHS.contains(&name.as_str()) && !s.is_empty() {
            s = join_path(here, &s);
        }
        words.push(flag.clone());
        words.push(s);
    }
    Ok(words)
}

fn read_config(path: &str) -> Result<Map<String, Value>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read: {}", os_strerror(&e)))?;
    let text = String::from_utf8(bytes.clone()).map_err(|_| "not valid JSON: invalid UTF-8".to_string())?;
    let raw = pyjson::loads(&text).map_err(|e| format!("not valid JSON: {}", e.0))?;
    let Value::Object(raw) = raw else {
        return Err(format!("expected {{\"version\": {CONFIG_VERSION}, \"server\": {{...}}, \"engine\": {{...}}, \"models\": [...]}}"));
    };
    let version = raw.get("version").cloned().unwrap_or(json!(CONFIG_VERSION));
    let v = match &version {
        Value::Number(n) if n.is_i64() => n.as_i64().unwrap(),
        _ => return Err("\"version\" must be a whole number".into()),
    };
    if v > CONFIG_VERSION {
        return Err(format!("this is a v{v} config; this Basalt reads v{CONFIG_VERSION}"));
    }
    if v < CONFIG_VERSION || !raw.contains_key("models") || raw.contains_key("v") {
        let old = std::fs::canonicalize(path).map(|p| strip_unc(&p)).unwrap_or_else(|_| PathBuf::from(path));
        let stem = old.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let suffix = old.extension().map(|s| format!(".{}", s.to_string_lossy())).unwrap_or_default();
        let new = old.with_file_name(format!("{stem}.v{CONFIG_VERSION}{suffix}"));
        return Err(format!("Basalt requires v{CONFIG_VERSION} config, please upgrade: python3 {} {} -o {}",
                           py_path(&root().join("tools/migrations/config_v1_to_v2.py")), py_path(&old), py_path(&new)));
    }
    let mut extra: Vec<&String> = raw.keys().filter(|k| !["version", "server", "engine", "models"].contains(&k.as_str())).collect();
    extra.sort();
    if let Some(k) = extra.first() {
        return Err(format!("unexpected key {}", py_str_repr(k)));
    }
    Ok(raw)
}

pub fn strip_unc(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(r) = s.strip_prefix(r"\\?\") {
        PathBuf::from(r)
    } else {
        p.to_path_buf()
    }
}

fn os_strerror(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => "No such file or directory".into(),
        std::io::ErrorKind::PermissionDenied => "Permission denied".into(),
        _ => {
            if e.raw_os_error() == Some(21) || (cfg!(windows) && e.raw_os_error() == Some(5)) {
                "Permission denied".into()
            } else {
                e.to_string()
            }
        }
    }
}

struct ConfigFile {
    server_argv: Vec<String>,
    models: Vec<ModelEntry>,
}

fn server_options(ap: &Parser) -> Vec<(String, Arg)> {
    let mut out = Vec::new();
    for a in &ap.args {
        for o in &a.flags {
            if let Some(n) = o.strip_prefix("--") {
                if n != "help" && n != "config" {
                    out.push((n.to_string(), a.clone()));
                }
            }
        }
    }
    out
}

fn load_config_file(path: &str, ap: &Parser) -> Result<ConfigFile, String> {
    let raw = read_config(path)?;
    let server = raw.get("server").cloned().unwrap_or(json!({}));
    let shared = raw.get("engine").cloned().unwrap_or(json!({}));
    let entries = raw.get("models").cloned().unwrap_or(Value::Null);
    for (k, v) in [("server", &server), ("engine", &shared)] {
        if !v.is_object() {
            return Err(format!("\"{k}\" must be an object"));
        }
    }
    let entries = match &entries {
        Value::Array(a) if !a.is_empty() && a.iter().all(|e| e.is_object()) => a.clone(),
        _ => return Err("\"models\" must be a list of at least one {\"model\": FILE, ...}".into()),
    };
    let server = server.as_object().unwrap().clone();
    let shared = shared.as_object().unwrap().clone();
    let secs: Vec<&Map<String, Value>> =
        std::iter::once(&server).chain(std::iter::once(&shared)).chain(entries.iter().map(|e| e.as_object().unwrap())).collect();
    for section in secs {
        for key in section.keys() {
            let letters: String = key.to_lowercase().chars().filter(|c| c.is_ascii_lowercase()).collect();
            if letters.starts_with("apikey") {
                return Err("api-key can't be set in a config file; pass --api-key on the command line".into());
            }
        }
    }
    let here = std::fs::canonicalize(path).map(|p| strip_unc(&p)).unwrap_or_else(|_| PathBuf::from(path));
    let here = here.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let known = server_options(ap);
    let find = |n: &str| known.iter().find(|(k, _)| k == n).map(|(_, a)| a);
    let mut server_argv = Vec::new();
    for (key, value) in &server {
        let name = config_name(key);
        if name == "engine-bin" {
            return Err("server.engine-bin: engine-bin is set by basalt serve".into());
        }
        if ENTRY_OPTIONS.contains(&name.as_str()) || MODEL_OPTIONS.contains(&name.as_str()) {
            return Err(format!("server.{key}: a model setting, set it under \"engine\" or in a model entry"));
        }
        let Some(act) = find(&name) else {
            return Err(format!("server.{key}: not a server option (engine flags go under \"engine\")"));
        };
        server_argv.extend(option_words("server", key, value, act, &here)?);
    }
    let mut shared_argv = Vec::new();
    let mut shared_words = Vec::new();
    for (key, value) in &shared {
        let name = config_name(key);
        if ENTRY_OPTIONS.contains(&name.as_str()) {
            return Err(format!("engine.{key}: set it in a model entry"));
        }
        if MODEL_OPTIONS.contains(&name.as_str()) {
            shared_argv.extend(option_words("engine", key, value, find(&name).unwrap(), &here)?);
        } else if find(&name).is_some() {
            return Err(format!("engine.{key}: a server option, set it under \"server\""));
        } else {
            shared_words.extend(engine_words(key, value)?);
        }
    }
    let mut models = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        let entry = entry.as_object().unwrap();
        let where_ = format!("models[{i}]");
        let model = match entry.get("model") {
            Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
            _ => return Err(format!("{where_}: \"model\" names the model file")),
        };
        let mut out = ModelEntry { path: join_path(&here, &model), argv: shared_argv.clone(), shared_words: shared_words.clone(), own_words: vec![] };
        for (key, value) in entry {
            let name = config_name(key);
            if LAUNCHER_KEYS.contains(&name.as_str()) {
                continue;
            }
            if ENTRY_OPTIONS.contains(&name.as_str()) || MODEL_OPTIONS.contains(&name.as_str()) {
                out.argv.extend(option_words(&where_, key, value, find(&name).unwrap(), &here)?);
            } else if find(&name).is_some() {
                return Err(format!("{where_}.{key}: a server option, set it under \"server\""));
            } else {
                out.own_words.extend(engine_words(key, value)?);
            }
        }
        models.push(out);
    }
    Ok(ConfigFile { server_argv, models })
}

pub type Refused<'a> = &'a dyn Fn(&str, &[String]) -> bool;

/// parse_settings() for each model the server runs: the --config file's entries, else the command line's one.
pub fn model_settings(ap: &Parser, cli: &[String], refused: Option<Refused>) -> Result<Vec<Setting>, ConfigErr> {
    let a_cli = ap.parse_known_args(cli)?.0;
    let cfg_path = a_cli.get("config").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());
    let Some(cfg_path) = cfg_path else { return Ok(vec![parse_settings(ap, cli, &[], &[], &[])?]) };
    let mut given = Vec::new();
    for (f, k) in [("--model", "model"), ("--alias", "alias")] {
        if a_cli.get(k).map(pyjson::truthy).unwrap_or(false) {
            given.push(f);
        }
    }
    if !given.is_empty() {
        let verb = if given.len() == 1 { "is" } else { "are" };
        return verr(format!("{} {verb} not supported when using a config file", given.join(" and ")));
    }
    let cfg = load_config_file(&cfg_path, ap).map_err(ConfigErr::Value)?;
    let mut out = Vec::new();
    for entry in &cfg.models {
        let mut kept = Vec::new();
        let mut skipped = Vec::new();
        for group in engine_groups(&entry.shared_words) {
            if refused.map(|f| f(&entry.path, &group)).unwrap_or(false) {
                skipped.push(group[0].clone());
            } else {
                kept.extend(group);
            }
        }
        kept.extend(entry.own_words.iter().cloned());
        let mut s = parse_settings(ap, cli, &cfg.server_argv, &entry.argv, &kept)?;
        s.0.skipped = skipped;
        out.push(s);
    }
    Ok(out)
}

pub fn parse_settings(ap: &Parser, cli: &[String], server_argv: &[String], model_argv: &[String], words: &[String])
                      -> Result<Setting, ConfigErr> {
    let (a_cli, extras) = ap.parse_known_args(cli)?;
    let mut cli_words = split_engine_extras(&extras).map_err(ConfigErr::Value)?;
    let file_argv: Vec<String> = server_argv.iter().chain(model_argv).cloned().collect();
    let mut file_words: Vec<String> = words.to_vec();
    let all: Vec<String> = file_argv.iter().chain(cli).cloned().collect();
    let ns = ap.parse_known_args(&all)?.0;
    let mut a = Args { ns, skipped: vec![], sampling_launch: Map::new(), sampling_cli: BTreeSet::new(), split_default: false, shared_context: None };
    for k in SAMPLING_ORDER {
        let v = a.get(k).clone();
        if !v.is_null() {
            a.sampling_launch.insert(k.to_string(), v);
            if !a_cli.get(k).map(|x| x.is_null()).unwrap_or(true) {
                a.sampling_cli.insert(k.to_string());
            }
        }
    }
    if a.get("vision").as_str() == Some("") {
        return Err(ConfigErr::Exit(ap.error(&vision_usage())));
    }
    for act in &ap.args {
        if act.kind == Kind::Append && a_cli.get(&act.dest) != Some(&act.default) {
            a.ns.insert(act.dest.clone(), a_cli.get(&act.dest).cloned().unwrap_or(Value::Null));
        }
    }
    let parallel = a.i("parallel").unwrap_or(0);
    if parallel > 1 {
        for flag in ["--batch", "--slots"] {
            if file_words.iter().any(|w| w == flag) || cli_words.iter().any(|w| w == flag) {
                return verr(format!("parallel and engine {flag} both set: use parallel"));
            }
        }
        let cli_par = a_cli.get("parallel").map(pyjson::truthy).unwrap_or(false);
        let target = if cli_par { &mut cli_words } else { &mut file_words };
        target.push("--batch".into());
        target.push(parallel.to_string());
        if !a.b("slot_full_context") {
            let use_cli = cli_words.iter().any(|w| w == "--max-context") || !file_words.iter().any(|w| w == "--max-context");
            let src = if use_cli { &mut cli_words } else { &mut file_words };
            let at = flag_index(src, "--max-context");
            if let Some(at) = at {
                if at + 1 >= src.len() {
                    return verr("--max-context needs a value".into());
                }
            }
            match at {
                None => a.split_default = true,
                Some(at) => {
                    let ctx = match int_text(&src[at + 1]) {
                        Some(c) => c,
                        None => return verr(format!("invalid literal for int() with base 10: {}", py_str_repr(&src[at + 1]))),
                    };
                    src[at + 1] = slot_share(ctx, parallel).to_string();
                    a.shared_context = Some(ctx);
                }
            }
        }
    }
    Ok((a, all, file_words, cli_words))
}

pub fn check_engine_flags(words: &[String], known: Option<&BTreeSet<String>>, where_: &str) -> Result<(), String> {
    for word in words {
        if word == "--gpu" {
            return Err(format!("{where_}: gpu is a server setting (--gpu, or \"gpu\" in the server section)"));
        }
        if let Some(k) = known {
            if word.starts_with("--") && !k.contains(word) {
                return Err(format!("{where_}: {word} is not an engine flag"));
            }
        }
    }
    Ok(())
}

pub fn engine_groups(words: &[String]) -> Vec<Vec<String>> {
    let mut groups: Vec<Vec<String>> = Vec::new();
    for w in words {
        if w.starts_with("--") || groups.is_empty() {
            groups.push(vec![w.clone()]);
        } else {
            groups.last_mut().unwrap().push(w.clone());
        }
    }
    groups
}

const ENGINE_NAMES: [(&str, &str); 5] = [("max-context", "ctx"), ("pool-workers", "pool"),
    ("conversation-cache-disk-mib", "disk-cache"), ("conversation-cache-mib", "ram-cache"), ("vram-reserve-mib", "reserve")];

pub fn engine_item(group: &[String]) -> String {
    let flag = &group[0][2.min(group[0].len())..];
    let values = &group[1..];
    let unit = [("-mib", "MiB"), ("-gib", "GiB")].iter()
        .find(|(s, _)| flag.ends_with(s) && values.len() == 1).map(|(_, u)| *u).unwrap_or("");
    let name = ENGINE_NAMES.iter().find(|(f, _)| *f == flag).map(|(_, n)| n.to_string())
        .unwrap_or_else(|| if !unit.is_empty() { flag[..flag.len() - 4].to_string() } else { flag.to_string() });
    if values.is_empty() {
        name
    } else {
        format!("{name}={}{unit}", values.join(","))
    }
}

pub fn start_items(file_words: &[String], cli_words: &[String], shared: Option<(i64, i64)>, grammar: Option<(bool, String)>,
                   debug: bool) -> Vec<String> {
    let mut groups = engine_groups(file_words);
    groups.extend(engine_groups(cli_words));
    let mut last = std::collections::HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        last.insert(g[0].clone(), i);
    }
    let mut items = Vec::new();
    for (i, g) in groups.iter().enumerate() {
        if last[&g[0]] != i || (g[0] == "--conversation-cache-dir" && !debug) {
            continue;
        }
        let mut item = engine_item(g);
        if let Some((total, slots)) = shared {
            if g[0] == "--max-context" {
                item += &format!(" ({total}/{slots} slots)");
            }
        }
        items.push(item);
    }
    if let Some((on, _)) = grammar {
        items.push(format!("grammar={}", if on { "on" } else { "off" }));
    }
    items
}

pub fn slot_share(total: i64, parallel: i64) -> i64 {
    MIN_SLOT_CONTEXT.max(total.div_euclid(parallel))
}

pub fn slots_text(parallel: usize, per_slot: i64, shared: Option<i64>) -> String {
    let text = format!("slots {parallel} \u{d7} {per_slot} ctx");
    match shared.filter(|s| *s != 0) {
        Some(s) => format!("{text} (slot-full-context: {s} each)"),
        None => text,
    }
}

pub fn with_spec(items: Vec<String>, spec: Option<i64>) -> Vec<String> {
    match spec {
        None => items,
        Some(n) => {
            let mut v: Vec<String> = items.into_iter().filter(|x| !x.starts_with("spec=")).collect();
            v.push(format!("spec={n}"));
            v
        }
    }
}

pub fn shared_line(shared: &[String], sleeps: &[Option<f64>]) -> (String, bool) {
    let mut s: Vec<f64> = Vec::new();
    for x in sleeps {
        let v = x.unwrap_or(0.0);
        if !s.contains(&v) {
            s.push(v);
        }
    }
    let sleep = sleeps.len() > 1 && s.len() == 1 && s[0] > 0.0;
    let mut items: Vec<String> = shared.to_vec();
    if sleep {
        items.push(format!("sleep={}s", pyjson::py_format_g(s[0])));
    }
    (items.join(" "), sleep)
}

pub fn model_line(name: &str, width: usize, items: &[String]) -> String {
    format!("[server] {name:<width$} {}", items.join(" ")).trim_end().to_string()
}

/// The model file's sampling defaults; Err(SystemExit text) for a bad value.
pub fn sampling_defaults_from_config(cfg: &Map<String, Value>) -> Result<Map<String, Value>, String> {
    let mut out = Map::new();
    let Some(Value::Object(s)) = cfg.get("sampling") else { return Ok(out) };
    for (key, value) in s {
        if value.is_null() || (key == "top_k" && num(value).map(|x| x == 0.0).unwrap_or(false)) {
            continue;
        }
        if let Some((is_int, ok, expected)) = sampling_rule(key) {
            if !ok(value) {
                return Err(format!("[server] error: sampling.{key}={}: expected {expected}", py_repr(value)));
            }
            let x = num(value).unwrap();
            out.insert(key.clone(), if is_int { json!(x as i64) } else { json!(x) });
        } else if key == "experimental_speed_projection" {
            if !value.is_boolean() {
                return Err(format!("[server] error: sampling.{key}={}: expected true or false (the default for requests \
                                    that leave it out, when the engine has the vector)", py_repr(value)));
            }
            out.insert(key.clone(), value.clone());
        } else {
            serve_engine::util::stdout::out(&format!("[server] warning: sampling.{key} is not a known key, ignored"));
        }
    }
    Ok(out)
}

const SAMPLING_NAMES: [(&str, &str); 4] = [("temperature", "temp"), ("presence_penalty", "pres"), ("frequency_penalty", "freq"),
    ("repetition_penalty", "rep")];

pub fn sampling_items(effective: &Map<String, Value>) -> Vec<String> {
    let off = |k: &str, v: &Value| -> bool {
        let x = num(v);
        match k {
            "min_p" | "presence_penalty" | "frequency_penalty" => x == Some(0.0),
            "repetition_penalty" => x == Some(1.0),
            "top_k" => x == Some(-1.0) || x == Some(0.0),
            _ => false,
        }
    };
    effective.iter().filter(|(k, v)| !off(k, v)).map(|(k, v)| {
        let n = SAMPLING_NAMES.iter().find(|(a, _)| a == k).map(|(_, b)| *b).unwrap_or(k);
        format!("{n}={}", pyjson::py_str(v))
    }).collect()
}

#[allow(dead_code)]
fn _w(v: &Value) -> bool {
    whole(v).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn w(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }
    #[test]
    fn items() {
        assert_eq!(engine_item(&w(&["--vram-reserve-mib", "512"])), "reserve=512MiB");
        assert_eq!(engine_item(&w(&["--foo-gib", "2"])), "foo=2GiB");
        assert_eq!(engine_item(&w(&["--spec"])), "spec");
        assert_eq!(slot_share(1000, 8), 256);
        assert_eq!(split_engine_extras(&w(&["--a=1", "b"])).unwrap(), w(&["--a", "1", "b"]));
        assert!(split_engine_extras(&w(&["x"])).is_err());
    }
    #[test]
    fn launcher_keys_are_not_engine_flags() {
        let dir = std::env::temp_dir().join(format!("basalt-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("c.json");
        std::fs::write(&p, r#"{"version": 2, "models": [{"model": "a.basalt", "hf": "o/r/a.basalt", "download": false, "kv": "int8"}]}"#).unwrap();
        let cfg = load_config_file(p.to_str().unwrap(), &build_parser()).unwrap();
        assert_eq!(cfg.models[0].own_words, w(&["--kv", "int8"]));
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn parse_basics() {
        let ap = build_parser();
        let (ns, extras) = ap.parse_known_args(&w(&["--port", "9", "--spec", "2", "--grammar", "--no-grammar", "--vision"])).unwrap();
        assert_eq!(ns["port"], json!(9));
        assert_eq!(extras, w(&["--spec", "2"]));
        assert_eq!(ns["grammar"], json!(false));
        let e = ap.parse_known_args(&w(&["--port", "x"])).unwrap_err();
        assert!(e.stderr.ends_with("server.py: error: argument --port: invalid int value: 'x'\n"));
    }
}
