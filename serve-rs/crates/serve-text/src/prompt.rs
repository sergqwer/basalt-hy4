//! serve/prompt.py: a prompt sent to the engine as text (the engine tokenizes it), and the context errors.

use crate::tokenizer::Tokenizer;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// The engine rejects prompt + max_new + 8 > context.
pub const CTX_SLACK: i64 = 8;

pub fn room_error(n: i64, ctx: i64) -> String {
    format!(
        "request exceeds the context window: prompt ({n} tokens) leaves no room to answer in the context ({ctx}); \
         requests are never truncated"
    )
}

pub fn overflow_error(n: i64, max_new: i64, ctx: i64) -> String {
    let room = ctx - CTX_SLACK - n;
    format!(
        "request exceeds the context window: prompt ({n} tokens) + max tokens ({max_new}) exceeds the context \
         ({ctx}); requests are never truncated. Send a smaller max_tokens (at most {} here), or start the server with \
         --fit-max-tokens (or set \"fit-max-tokens\": true in the \"server\" section of the --config file) to shorten \
         it to the room left",
        room.max(0)
    )
}

pub const ERR_CODES: [&str; 2] = ["context_overflow", "context_no_room"];

/// errors.split_err: (code, integer fields, message) of an ERR text.
pub fn split_err(text: &str) -> (Option<String>, BTreeMap<String, i64>, String) {
    let (head, message) = match text.split_once(": ") {
        Some((h, m)) => (h, m),
        None => return (None, BTreeMap::new(), text.to_string()),
    };
    let words: Vec<&str> = head.split_whitespace().collect();
    if words.len() > 1 && ERR_CODES.contains(&words[0]) {
        let mut fields = BTreeMap::new();
        for w in &words[1..] {
            let (k, v) = w.split_once('=').unwrap_or((w, ""));
            let digits = v.strip_prefix('-').unwrap_or(v);
            // Python: v.lstrip("-").isdigit() (any number of leading minus signs)
            let digits = digits.trim_start_matches('-');
            if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                return (None, BTreeMap::new(), text.to_string());
            }
            let Ok(n) = v.parse::<i64>() else { return (None, BTreeMap::new(), text.to_string()) };
            fields.insert(k.to_string(), n);
        }
        return (Some(words[0].to_string()), fields, message.to_string());
    }
    (None, BTreeMap::new(), text.to_string())
}

#[derive(Default)]
struct Counted {
    n: Option<i64>,
    max_new: Option<i64>,
    ids: Option<Vec<u32>>,
}

/// PromptText: the rendered prompt the engine tokenizes, plus what the engine said of it.
pub struct PromptText {
    tok: Arc<dyn Tokenizer>,
    pub text: String,
    pub body: String,
    pub tail: String,
    /// byte spans of `body` where special-token text stays text
    pub plain: Vec<(usize, usize)>,
    pub clip: bool,
    state: Mutex<Counted>,
}

impl PromptText {
    pub fn new(tok: Arc<dyn Tokenizer>, text: String, plain: Vec<(usize, usize)>, tail: &str, clip: bool) -> Self {
        let body = if tail.is_empty() { text.clone() } else { text[..text.len() - tail.len()].to_string() };
        PromptText { tok, text, body, tail: tail.to_string(), plain, clip, state: Mutex::new(Counted::default()) }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn clip(&self) -> bool {
        self.clip
    }

    /// The prompt tokens from the engine's PT line (None: not counted yet).
    pub fn n(&self) -> Option<i64> {
        self.state.lock().unwrap().n
    }

    /// The answer cap the engine settled on.
    pub fn max_new(&self) -> Option<i64> {
        self.state.lock().unwrap().max_new
    }

    pub fn ids(&self) -> Vec<u32> {
        let mut st = self.state.lock().unwrap();
        if st.ids.is_none() {
            let mut ids = self.tok.encode(&self.body, true, &self.plain);
            if !self.tail.is_empty() {
                ids.extend(self.tok.encode(&self.tail, true, &[]));
            }
            st.n = Some(ids.len() as i64);
            st.ids = Some(ids);
        }
        st.ids.clone().unwrap()
    }

    /// len(): the count once known, else a display estimate.
    pub fn len(&self) -> usize {
        if let Some(n) = self.n() {
            return n.max(0) as usize;
        }
        (self.text.chars().count() / 4).max(1)
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// prompt + other: the ids extended.
    pub fn add(&self, other: &[u32]) -> Vec<u32> {
        let mut ids = self.ids();
        ids.extend_from_slice(other);
        ids
    }

    /// The plain spans as byte offsets into the UTF-8 text (they already are).
    pub fn byte_spans(&self) -> Vec<(usize, usize)> {
        self.plain.clone()
    }

    /// `prompt.max_new = max_new` before the engine has counted (Service._prepare_text).
    pub fn preset_max_new(&self, max_new: i64) {
        self.state.lock().unwrap().max_new = Some(max_new);
    }

    pub fn set_counted(&self, n: i64, max_new: i64) {
        let mut st = self.state.lock().unwrap();
        st.n = Some(n);
        st.max_new = Some(max_new);
    }

    /// The answer cap for a prompt of `n` tokens as the engine settles it; Err(ValueError text).
    pub fn settle(&self, n: i64, max_new: i64, ctx: i64) -> Result<i64, String> {
        let room = ctx - CTX_SLACK - n;
        let mut max_new = max_new;
        if self.clip {
            if room < 1 {
                return Err(room_error(n, ctx));
            }
            max_new = max_new.min(room);
        } else if max_new > room {
            return Err(overflow_error(n, max_new, ctx));
        }
        self.set_counted(n, max_new);
        Ok(max_new)
    }

    pub fn resolve(&self, max_new: i64, ctx: i64) -> Result<(Vec<u32>, i64), String> {
        let ids = self.ids();
        let m = self.settle(ids.len() as i64, max_new, ctx)?;
        Ok((ids, m))
    }

    /// The engine's ERR text in the server's words.
    pub fn explain(&self, err: &str) -> String {
        let (code, f, message) = split_err(err);
        let has = |k: &str| f.contains_key(k);
        match code.as_deref() {
            Some("context_overflow") if !has("slot") && has("prompt") && has("max_new") && has("context") => {
                self.state.lock().unwrap().n = Some(f["prompt"]);
                overflow_error(f["prompt"], f["max_new"], f["context"])
            }
            Some("context_no_room") if has("prompt") && has("context") => {
                self.state.lock().unwrap().n = Some(f["prompt"]);
                room_error(f["prompt"], f["context"])
            }
            _ => message,
        }
    }
}
