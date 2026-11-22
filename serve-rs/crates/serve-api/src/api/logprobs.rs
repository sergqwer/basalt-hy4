//! api/logprobs.py (the API half): OpenAI `logprobs` / `top_logprobs`, entries and their channels.

use crate::error::{AResult, ApiError};
use crate::frontend::Event;
use crate::py::{as_int, Obj};
use serde_json::{json, Map, Value};
use serve_engine::logprobs::Tok;
use serve_text::tokenizer::Tokenizer;
use std::collections::BTreeMap;

pub const REQUEST_KEY: &str = "_logprobs";
pub const MAX_TOP: i64 = 20;

/// checked: the alternatives a Chat Completions request asks for (0..20), or None for no logprobs.
pub fn checked(req: &Obj) -> AResult<Option<i64>> {
    let on = req.get("logprobs").filter(|v| !v.is_null());
    let top = req.get("top_logprobs").filter(|v| !v.is_null());
    if let Some(o) = on {
        if !o.is_boolean() {
            return Err(ApiError::value("logprobs: expected true or false"));
        }
    }
    if let Some(t) = top {
        match as_int(t) {
            Some(n) if (0..=MAX_TOP).contains(&n) => {}
            _ => return Err(ApiError::value(format!("top_logprobs: an integer from 0 to {MAX_TOP}"))),
        }
    }
    let on = on.map(|o| o == &Value::Bool(true)).unwrap_or(false);
    if top.is_some() && !on {
        return Err(ApiError::value("top_logprobs: requires logprobs to be true"));
    }
    Ok(if on { Some(top.and_then(as_int).unwrap_or(0)) } else { None })
}

fn f(x: f64) -> Value {
    serde_json::Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
}

/// entry: OpenAI's logprobs entry for pick `t`.
pub fn entry(tok: &dyn Tokenizer, t: &Tok) -> Value {
    let one = |tid: u32, lp: f64| -> Value {
        let raw = if tok.has_token_bytes() {
            tok.token_bytes(tid).unwrap_or_default()
        } else {
            tok.decode(&[tid]).unwrap_or_default().into_bytes()
        };
        json!({"token": String::from_utf8_lossy(&raw), "logprob": f(lp), "bytes": raw})
    };
    let lp = t.lp.as_ref().expect("an entry needs the P line");
    let mut e = one(t.id, lp.lp).as_object().unwrap().clone();
    e.insert("top_logprobs".into(), Value::Array(lp.alts.iter().map(|(i, v)| one(*i, *v)).collect()));
    Value::Object(e)
}

/// logprobs_part: the `logprobs` object.
pub fn logprobs_part(content: &[Value], reasoning: &[Value], tools: &BTreeMap<i64, Vec<Value>>) -> Value {
    let mut part = Map::new();
    part.insert("content".into(), Value::Array(content.to_vec()));
    if !reasoning.is_empty() {
        part.insert("reasoning_content".into(), Value::Array(reasoning.to_vec()));
    }
    if !tools.is_empty() {
        part.insert(
            "tool_calls".into(),
            Value::Array(tools.iter().map(|(i, c)| json!({"index": i, "content": c})).collect()),
        );
    }
    Value::Object(part)
}

/// (channel, entry, call index)
pub type LpItem = (&'static str, Value, Option<i64>);

/// Channels: which text each generated token's entry belongs to.
pub struct Channels {
    thinking: bool,
    pub pending: Vec<Value>,
    calls: i64,
    answered: bool,
    last: Option<(&'static str, Option<i64>)>,
}

fn shows(evs: &[Event]) -> Vec<&Event> {
    evs.iter().filter(|e| !e.text.is_empty() || e.kind == "tool_start" || e.kind == "tool_call").collect()
}

impl Channels {
    pub fn new(thinking: bool) -> Self {
        Channels { thinking, pending: vec![], calls: 0, answered: false, last: None }
    }

    pub fn token(&mut self, entry: Option<Value>, state0: &str, state1: &str, evs: &[Event]) -> Vec<LpItem> {
        let sh = shows(evs);
        let mut entry = entry;
        if state0 == "reasoning" && state1 != "reasoning" {
            entry = None;
            if !evs.iter().any(|e| e.kind == "reasoning" && !e.text.is_empty()) {
                self.pending.clear();
            }
        }
        if let Some(e) = entry {
            self.pending.push(e);
        }
        if sh.is_empty() {
            return vec![];
        }
        let out = self.flush(sh[0].kind, None);
        self.calls += evs.iter().filter(|e| e.kind == "tool_call").count() as i64;
        out
    }

    pub fn finish(&mut self, evs: &[Event]) -> Vec<LpItem> {
        let sh = shows(evs);
        if !sh.is_empty() {
            return self.flush(sh[0].kind, None);
        }
        if !self.pending.is_empty() && matches!(self.last, Some(("tool", _))) {
            return self.flush("tool_call", Some(self.calls - 1));
        }
        if !self.pending.is_empty() {
            self.flush("content", None)
        } else {
            vec![]
        }
    }

    fn flush(&mut self, kind: &str, index: Option<i64>) -> Vec<LpItem> {
        let channel: &'static str = match kind {
            "reasoning" => "reasoning",
            "content" => "content",
            _ => "tool",
        };
        let index = if channel == "tool" && index.is_none() { Some(self.calls) } else { index };
        let mut out = std::mem::take(&mut self.pending);
        if channel == "content" && self.thinking && !self.answered {
            while !out.is_empty()
                && crate::py::strip(out[0].get("token").and_then(|t| t.as_str()).unwrap_or("")).is_empty()
            {
                out.remove(0);
            }
        }
        self.answered = self.answered || channel == "content";
        self.last = Some((channel, if channel == "tool" { index } else { None }));
        let idx = self.last.unwrap().1;
        out.into_iter().map(|e| (channel, e, idx)).collect()
    }
}
