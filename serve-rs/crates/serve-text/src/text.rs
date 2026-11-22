//! util/text.py: streaming detokenizing and stop strings.

use crate::pyjson::py_repr;
use crate::tokenizer::Tokenizer;
use serde_json::{Map, Value};
use std::sync::Arc;

pub const IM_END: &str = "<|im_end|>";
pub const OPENAI_MAX_STOP: usize = 4;

/// Python's incremental UTF-8 decoder with errors="replace".
#[derive(Default)]
pub struct Utf8Incremental {
    held: Vec<u8>,
}

impl Utf8Incremental {
    pub fn decode(&mut self, bytes: &[u8]) -> String {
        self.held.extend_from_slice(bytes);
        let mut out = String::new();
        let mut buf = std::mem::take(&mut self.held);
        let mut at = 0;
        loop {
            match std::str::from_utf8(&buf[at..]) {
                Ok(s) => {
                    out.push_str(s);
                    break;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&buf[at..at + ok]).unwrap());
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{fffd}');
                            at += ok + n;
                        }
                        None => {
                            self.held = buf.split_off(at + ok);
                            buf.clear();
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    pub fn pending(&self) -> bool {
        !self.held.is_empty()
    }
}

/// Constant-time incremental decode for a tokenizer with token_bytes; a re-decode otherwise.
pub struct Detokenizer {
    tok: Arc<dyn Tokenizer>,
    ids: Vec<u32>,
    sent: usize, // characters of the decoded text sent (re-decode path)
    inc: Option<Utf8Incremental>,
}

impl Detokenizer {
    pub fn new(tok: Arc<dyn Tokenizer>) -> Self {
        let inc = if tok.has_token_bytes() { Some(Utf8Incremental::default()) } else { None };
        Detokenizer { tok, ids: Vec::new(), sent: 0, inc }
    }

    pub fn pending(&self) -> bool {
        match &self.inc {
            Some(inc) => inc.pending(),
            None => self.tok.decode(&self.ids).map(|t| t.ends_with('\u{fffd}')).unwrap_or(false),
        }
    }

    pub fn push(&mut self, t: u32) -> String {
        if let Some(inc) = &mut self.inc {
            let b = self.tok.token_bytes(t).unwrap_or_default();
            return inc.decode(&b);
        }
        self.ids.push(t);
        let text = self.tok.decode(&self.ids).unwrap_or_default();
        if text.ends_with('\u{fffd}') {
            return String::new();
        }
        let n = text.chars().count();
        let delta: String = text.chars().skip(self.sent).collect();
        self.sent = n;
        delta
    }
}

/// stop_strings(req): OpenAI `stop` or Anthropic `stop_sequences`, empties dropped; Err(ValueError text).
pub fn stop_strings(req: &Map<String, Value>) -> Result<Vec<String>, String> {
    let stop = match req.get("stop") {
        Some(v) if !v.is_null() => {
            let list = match v {
                Value::String(s) => vec![Value::String(s.clone())],
                Value::Array(a) => a.clone(),
                _ => return Err("stop must be a string or a list of strings".into()),
            };
            if !list.iter().all(|x| x.is_string()) {
                return Err("stop must be a string or a list of strings".into());
            }
            if list.len() > OPENAI_MAX_STOP {
                return Err(format!("stop takes at most {OPENAI_MAX_STOP} strings"));
            }
            list
        }
        _ => match req.get("stop_sequences") {
            Some(v) if !v.is_null() => match v {
                Value::Array(a) if a.iter().all(|x| x.is_string()) => a.clone(),
                _ => return Err("stop_sequences must be a list of strings".into()),
            },
            _ => return Ok(vec![]),
        },
    };
    Ok(stop.into_iter().filter_map(|x| x.as_str().map(String::from)).filter(|s| !s.is_empty()).collect())
}

/// Cuts text at the first stop string, holding back only a tail that could start one.
pub struct StopMatcher {
    pub stops: Vec<String>,
    pub held: String,
    pub hit: Option<String>,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        StopMatcher { stops, held: String::new(), hit: None }
    }

    pub fn push(&mut self, text: &str) -> String {
        if self.hit.is_some() {
            return String::new();
        }
        let buf = format!("{}{}", self.held, text);
        let mut first: Option<(usize, &String)> = None;
        for s in &self.stops {
            if let Some(i) = buf.find(s.as_str()) {
                if first.map(|(f, _)| i < f).unwrap_or(true) {
                    first = Some((i, s));
                }
            }
        }
        if let Some((i, s)) = first {
            self.hit = Some(s.clone());
            self.held.clear();
            return buf[..i].to_string();
        }
        let buf_chars = buf.chars().count();
        let mut keep_bytes = 0;
        let mut keep = 0usize; // characters
        for s in &self.stops {
            let sc: Vec<char> = s.chars().collect();
            let top = (sc.len().saturating_sub(1)).min(buf_chars);
            let mut k = top;
            while k > keep {
                let prefix: String = sc[..k].iter().collect();
                if buf.ends_with(&prefix) {
                    keep = k;
                    keep_bytes = prefix.len();
                    break;
                }
                k -= 1;
            }
        }
        let cut = buf.len() - keep_bytes;
        self.held = if keep > 0 { buf[cut..].to_string() } else { String::new() };
        buf[..cut].to_string()
    }

    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// `{value!r}` of a request field, for the messages that quote it.
pub fn r(v: &Value) -> String {
    py_repr(v)
}
