//! serve/engine/errors.py: errors raised for a request by the engine process and the server.
use std::collections::BTreeMap;
use std::fmt;

/// The ERR codes the engine gives a machine-readable form: `ERR <code> key=N ...: <message>`.
pub const ERR_CODES: [&str; 2] = ["context_overflow", "context_no_room"];

/// (code, fields, message) of an ERR text; (None, {}, text) when it is not a coded error.
pub fn split_err(text: &str) -> (Option<String>, BTreeMap<String, i64>, String) {
    let (head, sep, message) = match text.find(": ") {
        Some(i) => (&text[..i], true, &text[i + 2..]),
        None => (text, false, ""),
    };
    let words: Vec<&str> = head.split_whitespace().collect();
    if sep && words.len() > 1 && ERR_CODES.contains(&words[0]) {
        let mut fields = BTreeMap::new();
        for w in &words[1..] {
            let (k, v) = match w.find('=') {
                Some(i) => (&w[..i], &w[i + 1..]),
                None => (*w, ""),
            };
            if !crate::py::int_like(v) {
                return (None, BTreeMap::new(), text.to_string());
            }
            match v.parse::<i64>() {
                Ok(n) => {
                    fields.insert(k.to_string(), n);
                }
                Err(_) => return (None, BTreeMap::new(), text.to_string()),
            }
        }
        return (Some(words[0].to_string()), fields, message.to_string());
    }
    (None, BTreeMap::new(), text.to_string())
}

/// The Python exception classes, one variant each. Subclassing: StartError and Silent are EngineDied, Error is a
/// ValueError (see is_died / is_value).
#[derive(Debug, Clone, PartialEq)]
pub enum EngineErr {
    /// EngineDied: the engine process ended mid-request.
    Died(String),
    /// EngineStartError(EngineDied): the engine ended before READY.
    StartError { msg: String, log: Option<String> },
    /// EngineSilent(EngineDied): silent too long; the server ended it.
    Silent(String),
    /// EngineStarting: no READY yet (a 503).
    Starting(String),
    /// ModelBusy: a model control refused (a 409).
    ModelBusy(String),
    /// EngineError(ValueError): the engine failed for a reason of its own (a 500).
    Error(String),
    /// ValueError: the request's fault (a 400).
    Value(String),
    /// ClientGone: the client left while waiting.
    ClientGone,
    /// EngineStuck: survived QUIT, terminate and kill.
    Stuck(String),
    /// WakeBlocked: the VRAM a sleeping engine needs is not free.
    WakeBlocked(String),
    /// GpuBusy: unloaded and free VRAM below min_free_vram_mib.
    GpuBusy(String),
    /// Any other Python exception (TypeError, KeyError, ...): the handler does not catch it; the connection closes
    /// with nothing more written.
    Internal(String),
}

impl EngineErr {
    pub fn died(s: impl Into<String>) -> Self {
        EngineErr::Died(s.into())
    }
    pub fn value(s: impl Into<String>) -> Self {
        EngineErr::Value(s.into())
    }
    /// isinstance(e, EngineDied)
    pub fn is_died(&self) -> bool {
        matches!(self, EngineErr::Died(_) | EngineErr::StartError { .. } | EngineErr::Silent(_))
    }
    /// isinstance(e, ValueError)
    pub fn is_value(&self) -> bool {
        matches!(self, EngineErr::Value(_) | EngineErr::Error(_))
    }
    /// The Python class name.
    pub fn class_name(&self) -> &'static str {
        match self {
            EngineErr::Died(_) => "EngineDied",
            EngineErr::StartError { .. } => "EngineStartError",
            EngineErr::Silent(_) => "EngineSilent",
            EngineErr::Starting(_) => "EngineStarting",
            EngineErr::ModelBusy(_) => "ModelBusy",
            EngineErr::Error(_) => "EngineError",
            EngineErr::Value(_) => "ValueError",
            EngineErr::ClientGone => "ClientGone",
            EngineErr::Stuck(_) => "EngineStuck",
            EngineErr::WakeBlocked(_) => "WakeBlocked",
            EngineErr::GpuBusy(_) => "GpuBusy",
            EngineErr::Internal(_) => "Exception",
        }
    }
    /// str(e)
    pub fn message(&self) -> String {
        match self {
            EngineErr::Died(s)
            | EngineErr::Silent(s)
            | EngineErr::Starting(s)
            | EngineErr::ModelBusy(s)
            | EngineErr::Error(s)
            | EngineErr::Value(s)
            | EngineErr::Stuck(s)
            | EngineErr::WakeBlocked(s)
            | EngineErr::GpuBusy(s)
            | EngineErr::Internal(s) => s.clone(),
            EngineErr::StartError { msg, .. } => msg.clone(),
            EngineErr::ClientGone => String::new(),
        }
    }
    /// The same kind with another message.
    pub fn with_message(&self, text: String) -> Self {
        match self {
            EngineErr::Died(_) => EngineErr::Died(text),
            EngineErr::StartError { log, .. } => EngineErr::StartError { msg: text, log: log.clone() },
            EngineErr::Silent(_) => EngineErr::Silent(text),
            EngineErr::Starting(_) => EngineErr::Starting(text),
            EngineErr::ModelBusy(_) => EngineErr::ModelBusy(text),
            EngineErr::Error(_) => EngineErr::Error(text),
            EngineErr::Value(_) => EngineErr::Value(text),
            EngineErr::ClientGone => EngineErr::ClientGone,
            EngineErr::Stuck(_) => EngineErr::Stuck(text),
            EngineErr::WakeBlocked(_) => EngineErr::WakeBlocked(text),
            EngineErr::GpuBusy(_) => EngineErr::GpuBusy(text),
            EngineErr::Internal(_) => EngineErr::Internal(text),
        }
    }
}

impl fmt::Display for EngineErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for EngineErr {}

pub type EResult<T> = Result<T, EngineErr>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split() {
        let (c, f, m) = split_err("context_overflow prompt=10 max_new=5 context=12: too long");
        assert_eq!(c.as_deref(), Some("context_overflow"));
        assert_eq!(f["prompt"], 10);
        assert_eq!(m, "too long");
        let (c, _, m) = split_err("context_overflow prompt=x: y");
        assert!(c.is_none());
        assert_eq!(m, "context_overflow prompt=x: y");
        let (c, _, m) = split_err("something: else");
        assert!(c.is_none());
        assert_eq!(m, "something: else");
    }
}
