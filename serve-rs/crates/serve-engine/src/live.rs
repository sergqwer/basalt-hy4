//! serve/live.py: the requests being answered, by id and by session, so a client can steer one (answer now).
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

pub const FINISHED_KEPT: usize = 1024;
pub const STATES: [&str; 5] = ["queued", "prompt", "thinking", "answering", "finished"];

/// A settable flag (Python threading.Event without the wait).
#[derive(Debug, Default, Clone)]
pub struct Flag(Arc<AtomicBool>);

impl Flag {
    pub fn new() -> Self {
        Flag(Arc::new(AtomicBool::new(false)))
    }
    pub fn set(&self) {
        self.0.store(true, Ordering::SeqCst)
    }
    pub fn clear(&self) {
        self.0.store(false, Ordering::SeqCst)
    }
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[derive(Debug)]
pub struct LiveState {
    pub state: String,
    /// whether the reply thinks; None until the prompt is rendered
    pub thinking: Option<bool>,
}

/// One request's id, session and state, and its answer-now signal.
#[derive(Debug)]
pub struct LiveRequest {
    pub id: String,
    pub session: Option<String>,
    pub client_id: Option<String>,
    pub st: Mutex<LiveState>,
    pub answer_now: Flag,
}

impl LiveRequest {
    pub fn state(&self) -> String {
        self.st.lock().unwrap().state.clone()
    }
    pub fn set_state(&self, s: &str) {
        self.st.lock().unwrap().state = s.to_string();
    }
    pub fn thinking(&self) -> Option<bool> {
        self.st.lock().unwrap().thinking
    }
    pub fn set_thinking(&self, t: Option<bool>) {
        self.st.lock().unwrap().thinking = t;
    }
}

/// An answer-now call that cannot apply.
#[derive(Debug, Clone, PartialEq)]
pub struct AnswerNowRefused {
    pub status: u16,
    pub code: String,
    pub message: String,
    pub state: Option<String>,
}

impl AnswerNowRefused {
    fn new(status: u16, code: &str, message: String, state: Option<&str>) -> Self {
        AnswerNowRefused { status, code: code.into(), message, state: state.map(|s| s.to_string()) }
    }
    pub fn body(&self) -> Value {
        let mut err = Map::new();
        err.insert("type".into(), json!(if self.status == 404 { "not_found_error" } else { "invalid_request_error" }));
        err.insert("code".into(), json!(self.code));
        err.insert("message".into(), json!(self.message));
        if let Some(s) = &self.state {
            err.insert("state".into(), json!(s));
        }
        let mut out = Map::new();
        out.insert("error".into(), Value::Object(err));
        Value::Object(out)
    }
}

struct Inner {
    live: Vec<(String, Arc<LiveRequest>)>,        // insertion order, like a dict
    finished: VecDeque<(String, String)>,         // key -> request id, oldest first
}

/// Every running request across the served models, and the ids of the last ones that finished.
pub struct LiveRequests {
    inner: Mutex<Inner>,
}

impl Default for LiveRequests {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveRequests {
    pub fn new() -> Self {
        LiveRequests { inner: Mutex::new(Inner { live: Vec::new(), finished: VecDeque::new() }) }
    }

    pub fn open(&self, rid: &str, session: Option<String>, client_id: Option<String>) -> Arc<LiveRequest> {
        let req = Arc::new(LiveRequest {
            id: rid.to_string(),
            session,
            client_id,
            st: Mutex::new(LiveState { state: "queued".into(), thinking: None }),
            answer_now: Flag::new(),
        });
        let mut g = self.inner.lock().unwrap();
        // dict assignment: an existing key keeps its place
        if let Some(e) = g.live.iter_mut().find(|(k, _)| k == rid) {
            e.1 = req.clone();
        } else {
            g.live.push((rid.to_string(), req.clone()));
        }
        req
    }

    pub fn close(&self, req: &Arc<LiveRequest>) {
        let mut g = self.inner.lock().unwrap();
        req.set_state("finished");
        if let Some(i) = g.live.iter().position(|(k, r)| k == &req.id && Arc::ptr_eq(r, req)) {
            g.live.remove(i);
        }
        for key in [Some(req.id.clone()), req.client_id.clone()].into_iter().flatten() {
            if key.is_empty() {
                continue;
            }
            if let Some(i) = g.finished.iter().position(|(k, _)| *k == key) {
                g.finished.remove(i);
            }
            g.finished.push_back((key, req.id.clone()));
        }
        while g.finished.len() > FINISHED_KEPT {
            g.finished.pop_front();
        }
    }

    fn find(g: &Inner, request_id: Option<&str>, session: Option<&str>) -> Result<Arc<LiveRequest>, AnswerNowRefused> {
        if let Some(rid) = request_id.filter(|s| !s.is_empty()) {
            let hit = g.live.iter().find(|(k, _)| k == rid).map(|(_, r)| r.clone()).or_else(|| {
                g.live.iter().rev().find(|(_, r)| r.client_id.as_deref() == Some(rid)).map(|(_, r)| r.clone())
            });
            if let Some(r) = hit {
                return Ok(r);
            }
            if let Some((_, id)) = g.finished.iter().find(|(k, _)| k == rid) {
                return Err(AnswerNowRefused::new(409, "not_thinking", format!("request {id} has finished"), Some("finished")));
            }
            return Err(AnswerNowRefused::new(404, "unknown_request", format!("no running or recent request {rid}"), None));
        }
        let hit = g.live.iter().rev().find(|(_, r)| r.session.as_deref() == session).map(|(_, r)| r.clone());
        hit.ok_or_else(|| {
            AnswerNowRefused::new(404, "unknown_session", format!("no running request in session {}", session.unwrap_or("None")), None)
        })
    }

    /// Ask a request to end its thinking and answer; the reply names its state.
    pub fn answer_now(&self, request_id: Option<&str>, session: Option<&str>) -> Result<Value, AnswerNowRefused> {
        let rid = request_id.filter(|s| !s.is_empty());
        let ses = session.filter(|s| !s.is_empty());
        if rid.is_none() && ses.is_none() {
            return Err(AnswerNowRefused::new(400, "no_target", "name the request: request_id, or session_id (or a session header)".into(), None));
        }
        let g = self.inner.lock().unwrap();
        let req = Self::find(&g, rid, session)?;
        let (state, thinking) = {
            let s = req.st.lock().unwrap();
            (s.state.clone(), s.thinking)
        };
        if thinking == Some(false) {
            return Err(AnswerNowRefused::new(409, "not_thinking", format!("request {} runs with thinking off", req.id), Some(&state)));
        }
        if !["queued", "prompt", "thinking"].contains(&state.as_str()) {
            return Err(AnswerNowRefused::new(409, "not_thinking", format!("request {} is already {}", req.id, state), Some(&state)));
        }
        req.answer_now.set();
        let mut out = Map::new();
        out.insert("object".into(), json!("answer_now"));
        out.insert("request_id".into(), json!(req.id));
        out.insert("state".into(), json!(state));
        Ok(Value::Object(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn answer_now_flow() {
        let l = LiveRequests::new();
        let r = l.open("chatcmpl-1", Some("s".into()), Some("c1".into()));
        let v = l.answer_now(Some("c1"), None).unwrap();
        assert_eq!(v["state"], "queued");
        assert!(r.answer_now.is_set());
        l.close(&r);
        let e = l.answer_now(Some("c1"), None).unwrap_err();
        assert_eq!(e.status, 409);
        assert_eq!(e.message, "request chatcmpl-1 has finished");
        let e = l.answer_now(None, None).unwrap_err();
        assert_eq!(e.status, 400);
        let e = l.answer_now(None, Some("zz")).unwrap_err();
        assert_eq!(e.message, "no running request in session zz");
    }
}
