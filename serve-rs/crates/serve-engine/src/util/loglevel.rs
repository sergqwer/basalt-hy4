//! serve/util/loglevel.py: the level of a log line, read from the marker its writer put on it (engine `basalt <area>:
//! [warning:|error:]`, server `[server] [warning:|error:]`, a Python traceback); no message text is searched for words.
use regex::Regex;
use std::sync::OnceLock;

pub const INFO: &str = "info";
pub const WARN: &str = "warn";
pub const ERR: &str = "err";
pub const TRACEBACK: &str = "Traceback (most recent call last):";
pub const CHAINED: [&str; 2] = ["During handling of the above exception", "The above exception was the direct cause"];

fn name(lvl: Option<&str>) -> &'static str {
    match lvl {
        Some("warning") => WARN,
        Some("error") => ERR,
        _ => INFO,
    }
}

fn engine_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?s)\A(?P<tag>basalt(?:-[a-z]+)?(?: [\w-]+)*):(?: (?P<lvl>warning|error):)? (?P<text>.*)\z").unwrap()
    })
}

fn legacy_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?s)\Abasalt (?P<lvl>warning|error): (?P<text>.*)\z").unwrap())
}

fn server_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\A\[server\] (?:(?P<lvl>warning|error)(?:: | {2,}))?").unwrap())
}

/// The parts of an engine log line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineLine {
    pub tag: String,        // "basalt generate"; "basalt" for a line of no area
    pub level: &'static str,
    pub text: String,       // the message, without the tag and the marker
}

/// The parts of an engine log line, or None when the line is not one.
pub fn engine_line(line: &str) -> Option<EngineLine> {
    if let Some(m) = legacy_re().captures(line) {
        return Some(EngineLine { tag: "basalt".into(), level: name(m.name("lvl").map(|x| x.as_str())),
                                 text: m["text"].to_string() });
    }
    let m = engine_re().captures(line)?;
    Some(EngineLine { tag: m["tag"].to_string(), level: name(m.name("lvl").map(|x| x.as_str())), text: m["text"].to_string() })
}

/// The level a `[server] ` line states, or None when the line is not a server line.
pub fn server_level(line: &str) -> Option<&'static str> {
    server_re().captures(line).map(|m| name(m.name("lvl").map(|x| x.as_str())))
}

/// Reads the level of the lines of one stream, in order (a traceback spans lines).
#[derive(Debug, Default, Clone)]
pub struct LevelReader {
    in_traceback: bool,
}

impl LevelReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read(&mut self, line: &str) -> &'static str {
        if line.starts_with(TRACEBACK) {
            self.in_traceback = true;
            return ERR;
        }
        if CHAINED.iter().any(|c| line.starts_with(c)) {
            self.in_traceback = false;
            return ERR;
        }
        if self.in_traceback {
            let first = line.chars().next();
            if !matches!(first, Some(' ') | Some('\t')) && !crate::pyfmt::strip(line).is_empty() {
                self.in_traceback = false;      // the exception line ends the block
            }
            return ERR;
        }
        if let Some(e) = engine_line(line) {
            return e.level;
        }
        server_level(line).unwrap_or(INFO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(lines: &[&str]) -> Vec<&'static str> {
        let mut r = LevelReader::new();
        lines.iter().map(|x| r.read(x)).collect()
    }

    #[test]
    fn engine_markers() {
        assert_eq!(read(&["basalt generate: ready", "basalt generate: warning: low", "basalt serve: error: x",
                          "basalt: error: CUDA out of memory", "basalt batch adapt: window 3: 2 swaps",
                          "basalt-bench: warning: NVML is unavailable"]),
                   ["info", "warn", "err", "err", "info", "warn"]);
        assert_eq!(read(&["basalt error: boom", "basalt warning: slow disk"]), ["err", "warn"]);
    }

    #[test]
    fn words_decide_nothing() {
        for text in ["basalt serve: no error, 0 failures", "basalt generate: failed over to the CPU path",
                     "basalt load: cuda0 10 100", "basalt serve: prompt 5 tokens = 0 reused, error=0",
                     "request failed with a timeout", "WARNING: low on RAM", "CUDA error: out of memory",
                     "a fatal exception was aborted", "[server] stuck? no, retrying is not a marker"] {
            assert_eq!(read(&[text]), ["info"], "{text}");
        }
    }

    #[test]
    fn server_markers() {
        assert_eq!(read(&["[server] ready", "[server] warning: no API key", "[server] error: boom", "[server] error-free run"]),
                   ["info", "warn", "err", "info"]);
        assert_eq!(read(&["[server] warning  no api key"]), ["warn"]);
    }

    #[test]
    fn tracebacks() {
        assert_eq!(read(&["ready", TRACEBACK, "  File \"x.py\", line 3, in f", "    g()", "ValueError: nope", "[server] ready"]),
                   ["info", "err", "err", "err", "err", "info"]);
        assert_eq!(read(&[TRACEBACK, "  x", "KeyError: 1", "During handling of the above exception, another exception occurred:",
                          TRACEBACK, "  y", "OSError: 2", "ok"]),
                   ["err", "err", "err", "err", "err", "err", "err", "info"]);
    }

    #[test]
    fn parts() {
        let e = engine_line("basalt mtp: warning: the draft head over 5 tokens").unwrap();
        assert_eq!((e.tag.as_str(), e.level, e.text.as_str()), ("basalt mtp", "warn", "the draft head over 5 tokens"));
        let e = engine_line("basalt error: x").unwrap();
        assert_eq!((e.tag.as_str(), e.level, e.text.as_str()), ("basalt", "err", "x"));
        assert!(engine_line("CUDA error: x").is_none());
        assert!(engine_line("  chunk share 0.50: 12 ms").is_none());
    }
}
