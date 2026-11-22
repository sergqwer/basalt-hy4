//! The engine-protocol half of serve/api/logprobs.py: P/BP line parsing and the per-token record, and the GEN-line
//! keys of logprobs, logit_bias (serve/api/logit_bias.py) and grammar (serve/api/grammar.py).
use serde_json::Value;

pub const LOGPROBS_KEY: &str = "_logprobs";
pub const GRAMMAR_KEY: &str = "_grammar";
pub const BIAS_KEY: &str = "_logit_bias";

/// A P line's facts for a pick: its log probability and the likeliest alternatives.
#[derive(Debug, Clone, PartialEq)]
pub struct LogProb {
    pub lp: f64,
    pub alts: Vec<(u32, f64)>,
}

/// A generated token: the id, with its P line when the engine sent one for it (Python's logprobs.Token).
#[derive(Debug, Clone, PartialEq)]
pub struct Tok {
    pub id: u32,
    pub lp: Option<LogProb>,
}

impl Tok {
    pub fn plain(id: u32) -> Self {
        Tok { id, lp: None }
    }
}

/// `<id> <lp> [<id>:<lp> ...]`; None if malformed.
pub fn parse_line(fields: &[&str]) -> Option<(u32, LogProb)> {
    let mut alts = Vec::new();
    for f in fields.iter().skip(2) {
        let (i, _, v) = crate::py::partition(f, ":");
        let i = i.trim().parse::<i64>().ok()?;
        let v = crate::py::float_of(v)?;
        alts.push((i as u32, v));
    }
    let id = fields.first()?.trim().parse::<i64>().ok()?;
    let lp = crate::py::float_of(fields.get(1)?)?;
    Some((id as u32, LogProb { lp, alts }))
}

/// `tid` with its P line when the line is for this id.
pub fn token(tid: u32, parsed: Option<&(u32, LogProb)>) -> Tok {
    match parsed {
        Some((id, lp)) if *id == tid => Tok { id: tid, lp: Some(lp.clone()) },
        _ => Tok::plain(tid),
    }
}

/// ` logprobs=K`, or nothing (an int, not a bool, >= 0).
pub fn logprobs_key(top: Option<&Value>) -> String {
    match top {
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => {
            let k = n.as_i64().unwrap_or(i64::MAX);
            if k >= 0 {
                format!(" logprobs={k}")
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

/// ` logit_bias=id:value,...` (sorted by id), or nothing; only the server-normalized map is sent.
pub fn bias_key(bias: Option<&[(u32, f64)]>) -> String {
    let Some(b) = bias else { return String::new() };
    if b.is_empty() {
        return String::new();
    }
    let mut v: Vec<(u32, f64)> = b.to_vec();
    v.sort_by(|a, c| a.0.cmp(&c.0).then(a.1.partial_cmp(&c.1).unwrap_or(std::cmp::Ordering::Equal)));
    let parts: Vec<String> = v.iter().map(|(i, x)| format!("{i}:{}", serve_text::pyjson::py_format_g(*x))).collect();
    format!(" logit_bias={}", parts.join(","))
}

/// ` grammar=<JSON>`: compact, ASCII, spaces escaped; nothing for an empty spec.
pub fn grammar_key(spec: Option<&Value>) -> Result<String, String> {
    let Some(spec) = spec else { return Ok(String::new()) };
    if !py_truthy(spec) {
        return Ok(String::new());
    }
    let text = serve_text::pyjson::dumps_compact(spec);
    Ok(format!(" grammar={}", text.replace(' ', "\\u0020")))
}

/// Python truthiness of a JSON value.
pub fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|x| x != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse() {
        let (id, lp) = parse_line(&["5", "-0.25", "5:-0.25", "7:-1.5"]).unwrap();
        assert_eq!(id, 5);
        assert_eq!(lp.alts, vec![(5, -0.25), (7, -1.5)]);
        assert!(parse_line(&["5"]).is_none());
        let t = token(5, Some(&(5, lp.clone())));
        assert!(t.lp.is_some());
        assert!(token(6, Some(&(5, lp))).lp.is_none());
    }
}
