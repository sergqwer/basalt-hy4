//! serve/handler.py: the request handler: routes, authentication, CORS and streaming.
use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use serve_api::error::{ApiError, ResponsesError};
use serve_api::image_source::ImageSources;
use serve_engine::errors::EngineErr;
use serve_engine::live::{Flag, LiveRequest};
use serve_engine::logprobs::{BIAS_KEY, GRAMMAR_KEY, LOGPROBS_KEY};
use serve_engine::util::logbuf::LogBuffer;
use serve_engine::util::stdout;
use serve_text::pyjson;

use crate::http::{self, App, Req};
use crate::router::{Router, Svc, Turn};
use crate::security::{host_allowed, key_matches, origin_allowed, peer_of, remote_reason};

pub const CONTROL_BODY_MAX: usize = 64 << 10;
pub const CAPABILITIES: [&str; 3] = ["answer_now", "logprobs", "documents"];
pub const TURN_POLL_S: f64 = 0.5;
pub const QUEUE_BEAT_S: f64 = 10.0;
pub const LOG_FIRST_LINES: usize = 300;
pub const LOG_MAX_LINES: usize = 500;
pub const SESSION_HEADERS: [&str; 6] = ["x-llm-flow-id", "x-claude-code-session-id", "x-session-id", "x-session-affinity",
    "x-grok-conv-id", "x-grok-session-id"];

type Obj = Map<String, Value>;

/// The settings every model shares (Python keeps them on router.front).
pub struct ServerConf {
    pub api_key: String,
    pub cors_origins: Vec<String>,
    pub trusted_origins: Vec<String>,
    pub allowed_hosts: Vec<String>,
    pub host_names: BTreeSet<String>,
    pub max_payload: usize,
    pub image_sources: ImageSources,
    pub logs: Arc<LogBuffer>,
    pub root: PathBuf,
}

/// What ends a request in handler.py's except-chain.
#[derive(Debug)]
enum Fail {
    Api(ApiError),
    BadBody(u16, String),
    /// the client left while queued
    Gone,
    /// already answered (a `return` inside the try)
    Done,
}

impl From<ApiError> for Fail {
    fn from(e: ApiError) -> Self {
        match e {
            ApiError::Engine(EngineErr::ClientGone) => Fail::Gone,
            e => Fail::Api(e),
        }
    }
}

impl From<EngineErr> for Fail {
    fn from(e: EngineErr) -> Self {
        Fail::from(ApiError::Engine(e))
    }
}

type R<T> = Result<T, Fail>;

pub struct Handler {
    pub router: Arc<Router>,
    pub conf: Arc<RwLock<ServerConf>>,
}

/// The per-request state of handler.py's Handler instance.
struct Ctx<'a> {
    h: &'a Handler,
    req: &'a mut Req,
    watch_done: Option<Flag>,
    cancel: Option<Flag>,
    turn: Option<Turn>,
    sse_open: bool,
    live: Option<Arc<LiveRequest>>,
    svc: Option<Svc>,
}

fn dumps(v: &Value) -> Vec<u8> {
    pyjson::dumps(v, false).into_bytes()
}

fn err_obj(kind: Option<&str>, message: &str) -> Value {
    let mut e = Map::new();
    if let Some(k) = kind {
        e.insert("type".into(), json!(k));
    }
    e.insert("message".into(), json!(message));
    let mut o = Map::new();
    o.insert("error".into(), Value::Object(e));
    Value::Object(o)
}

fn size_text(n: usize) -> String {
    if n >= 1 << 20 { format!("{} MiB", n >> 20) } else { format!("{} KiB", n >> 10) }
}

/// parse_qs(urlsplit(path).query).get(name, [None])[0]
fn query_first(path: &str, name: &str) -> Option<String> {
    let q = path.split_once('?').map(|(_, q)| q).unwrap_or("");
    let q = q.split('#').next().unwrap_or("");
    for part in q.split(['&']) {
        if part.is_empty() {
            continue;
        }
        let (k, v) = part.split_once('=').unwrap_or((part, ""));
        if v.is_empty() {
            continue; // parse_qs drops blank values
        }
        if unquote_plus(k) == name {
            return Some(unquote_plus(v));
        }
    }
    None
}

fn query_has(path: &str, name: &str) -> bool {
    query_first(path, name).is_some()
}

fn unquote_plus(s: &str) -> String {
    let s = s.replace('+', " ");
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 0 && i + 2 <= b.len() - 1 + 0 {
            if let (Some(h), Some(l)) = ((b[i + 1] as char).to_digit(16), (b[i + 2] as char).to_digit(16)) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

impl<'a> Ctx<'a> {
    fn conf(&self) -> std::sync::RwLockReadGuard<'_, ServerConf> {
        self.h.conf.read().unwrap()
    }
    fn router(&self) -> &Arc<Router> {
        &self.h.router
    }
    fn header(&self, name: &str) -> Option<String> {
        self.req.headers.get(name).map(|s| s.to_string())
    }
    fn route(&self) -> String {
        self.req.path.split('?').next().unwrap_or("").trim_end_matches('/').to_string()
    }

    fn session_id(&self, req: Option<&Obj>) -> Option<String> {
        for h in SESSION_HEADERS {
            let v = self.header(h).unwrap_or_default();
            let v = v.trim();
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
        match req.and_then(|r| r.get("session_id")) {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            _ => None,
        }
    }

    fn open_live(&mut self, rid: &str, req: &Obj) {
        let client: String = self.header("x-client-request-id").unwrap_or_default().trim().chars().take(256).collect();
        let client = if client.is_empty() { None } else { Some(client) };
        let ses = self.session_id(Some(req));
        let live = self.router().live.open(rid, ses, client);
        self.req.end_header_extra = Some(("X-Request-Id".into(), live.id.clone()));
        self.live = Some(live);
    }

    fn chunked(&self) -> bool {
        let te = self.header("Transfer-Encoding").unwrap_or_default();
        te.rsplit(',').next().unwrap_or("").trim().to_lowercase() == "chunked"
    }

    fn read(&mut self, n: usize) -> R<Vec<u8>> {
        self.req.conn.read(n).map_err(|_| {
            self.req.close_connection = true;
            Fail::BadBody(408, "the request body stopped arriving".into())
        })
    }

    fn readline(&mut self, n: usize) -> R<Vec<u8>> {
        self.req.conn.readline(n).map_err(|_| {
            self.req.close_connection = true;
            Fail::BadBody(408, "the request body stopped arriving".into())
        })
    }

    fn read_chunked(&mut self, limit: usize, keep: bool) -> R<Vec<u8>> {
        let mut parts = Vec::new();
        let mut total = 0usize;
        loop {
            let line = self.readline(1025)?;
            if !line.ends_with(b"\n") {
                return Err(Fail::BadBody(400, "malformed chunked request body".into()));
            }
            let head = line.split(|&b| b == b';').next().unwrap_or(b"");
            let text = String::from_utf8_lossy(head).trim().to_string();
            let size = parse_hex(&text).ok_or_else(|| Fail::BadBody(400, "malformed chunk size in the request body".into()))?;
            if size < 0 {
                return Err(Fail::BadBody(400, "malformed chunk size in the request body".into()));
            }
            if size == 0 {
                for _ in 0..64 {
                    let l = self.readline(8193)?;
                    if l.iter().all(|b| b.is_ascii_whitespace()) {
                        break;
                    }
                }
                return Ok(parts);
            }
            let size = size as usize;
            total += size;
            if total > limit {
                self.req.close_connection = true;
                return Err(Fail::BadBody(413, format!("the request body is larger than {}", size_text(limit))));
            }
            let mut left = size;
            while left > 0 {
                let piece = self.read(left.min(1 << 20))?;
                if piece.is_empty() {
                    return Err(Fail::BadBody(400, "the chunked request body ended early".into()));
                }
                left -= piece.len();
                if keep {
                    parts.extend_from_slice(&piece);
                }
            }
            if self.read(2)? != b"\r\n" {
                return Err(Fail::BadBody(400, "malformed chunked request body".into()));
            }
        }
    }

    fn body(&mut self) -> R<Vec<u8>> {
        let limit = self.conf().max_payload;
        if self.chunked() {
            return self.read_chunked(limit, true);
        }
        let length = match self.header("Content-Length") {
            None => 0i64,
            Some(v) => py_int(&v).ok_or_else(|| Fail::BadBody(400, "invalid Content-Length".into()))?,
        };
        if length < 0 {
            return Err(Fail::BadBody(400, "invalid Content-Length".into()));
        }
        if length as u128 > limit as u128 {
            self.req.close_connection = true;
            return Err(Fail::BadBody(413, format!("the request body is larger than {}", size_text(limit))));
        }
        self.read(length as usize)
    }

    fn foreign_page(&mut self) -> std::io::Result<bool> {
        let origin = self.header("Origin");
        let (key, names, origins) = {
            let c = self.conf();
            (c.api_key.clone(), c.host_names.clone(), [c.trusted_origins.clone(), c.cors_origins.clone()].concat())
        };
        let Some(origin) = origin.filter(|o| !o.is_empty()) else { return Ok(false) };
        if !key.is_empty() {
            return Ok(false);
        }
        if !origin_allowed(&origin, self.header("Host").as_deref(), &names, &origins) {
            stdout::out(&format!("[server] error: refused a request from web page {} (--api-key KEY, --allowed-host, --cors-origin)",
                                 pyjson::py_str_repr(&origin)));
            self.json(403, &json!({"error": {"type": "forbidden", "message": format!(
                "web pages of {origin} may not use this server without an API key; start the server with --api-key KEY, \
                 or add the page's host with --allowed-host (or its origin with --cors-origin)")}}))?;
            return Ok(true);
        }
        if !self.header("Content-Type").unwrap_or_default().starts_with("application/json") {
            self.json(415, &err_obj(None, "send application/json"))?;
            return Ok(true);
        }
        Ok(false)
    }

    fn image_access(&self) -> ImageSources {
        let base = self.conf().image_sources.clone();
        let why = if base.remote_only { "remote-only".to_string() } else { remote_reason(self.req.client_address(), Some(&self.req.headers)) };
        base.with_access(why.is_empty(), why, peer_of(self.req.client_address()))
    }

    fn client_cancel(&mut self) -> Flag {
        if self.cancel.is_none() {
            let c = Flag::new();
            self.watch_client(c.clone());
            self.cancel = Some(c);
        }
        self.cancel.clone().unwrap()
    }

    fn watch_client(&mut self, cancel: Flag) {
        let done = Flag::new();
        self.watch_done = Some(done.clone());
        let Ok(sock) = self.req.conn.writer.try_clone() else { return };
        std::thread::Builder::new().name("client-watch".into()).spawn(move || loop {
            std::thread::sleep(Duration::from_millis(500));
            if done.is_set() || cancel.is_set() {
                return;
            }
            match http::peer_gone(&sock) {
                None => return,
                Some(true) => {
                    cancel.set();
                    return;
                }
                Some(false) => {}
            }
        }).ok();
    }

    fn take_turn(&mut self, stream: bool) -> R<()> {
        let svc = self.svc.clone().expect("svc");
        let turn = self.router().join(&svc);
        self.turn = Some(turn.clone());
        if self.router().wait(&turn, Some(0.0))? {
            return Ok(());
        }
        let cancel = self.client_cancel();
        let mut beat = Instant::now();
        while !self.router().wait(&turn, Some(TURN_POLL_S))? {
            if cancel.is_set() {
                return Err(Fail::Gone);
            }
            if stream && beat.elapsed().as_secs_f64() >= QUEUE_BEAT_S {
                beat = Instant::now();
                let r = (|| -> std::io::Result<()> {
                    self.sse()?;
                    self.req.write(b": keep-alive\n\n")?;
                    self.req.flush()
                })();
                if r.is_err() {
                    return Err(Fail::Gone);
                }
            }
        }
        Ok(())
    }

    fn cors(&mut self) {
        let (origins,) = (self.conf().cors_origins.clone(),);
        if origins.is_empty() || !self.req.path.split('?').next().unwrap_or("").starts_with("/v1/") {
            return;
        }
        let origin = self.header("Origin").unwrap_or_default().trim_end_matches('/').to_string();
        let allow = if origins.iter().any(|o| o == "*") {
            "*".to_string()
        } else if !origin.is_empty() && origins.contains(&origin) {
            self.req.send_header("Vary", "Origin");
            origin
        } else {
            return;
        };
        self.req.send_header("Access-Control-Allow-Origin", &allow);
        self.req.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        let asked = self.header("Access-Control-Request-Headers").filter(|s| !s.is_empty());
        self.req.send_header("Access-Control-Allow-Headers",
            &asked.unwrap_or_else(|| "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta".into()));
        self.req.send_header("Access-Control-Max-Age", "600");
        self.req.send_header("Access-Control-Expose-Headers", "X-Request-Id");
    }

    fn do_options(&mut self) -> std::io::Result<()> {
        self.req.send_response(204, None);
        self.cors();
        self.req.send_header("Content-Length", "0");
        self.req.end_headers()
    }

    fn stream_error(&mut self, obj: &Value) -> std::io::Result<()> {
        let path = self.route();
        let err = match obj.get("error") {
            Some(Value::Object(e)) => Value::Object(e.clone()),
            _ => json!({"message": pyjson::py_str(obj)}),
        };
        let data = if path == "/v1/messages" {
            [b"event: error\ndata: ".to_vec(), dumps(&json!({"type": "error", "error": err})), b"\n\n".to_vec()].concat()
        } else if path == "/v1/responses" {
            let mut m = Map::new();
            m.insert("type".into(), json!("error"));
            m.insert("code".into(), err.get("code").cloned().unwrap_or(Value::Null));
            m.insert("message".into(), err.get("message").cloned().unwrap_or(Value::Null));
            [b"event: error\ndata: ".to_vec(), dumps(&Value::Object(m)), b"\n\n".to_vec()].concat()
        } else {
            [b"data: ".to_vec(), dumps(&json!({"error": err})), b"\n\ndata: [DONE]\n\n".to_vec()].concat()
        };
        self.req.write(&data)
    }

    fn json(&mut self, code: u16, obj: &Value) -> std::io::Result<()> {
        if self.sse_open {
            return self.stream_error(obj);
        }
        let body = dumps(obj);
        self.req.send_response(code, None);
        self.req.send_header("Content-Type", "application/json");
        self.cors();
        self.req.send_header("Content-Length", &body.len().to_string());
        self.req.end_headers()?;
        self.req.write(&body)
    }

    fn died(&self, e: &EngineErr) -> String {
        if self.router().stopping() { e.message() } else { format!("{}; the next request restarts it", e.message()) }
    }

    fn authorized(&mut self) -> std::io::Result<bool> {
        let key = self.conf().api_key.clone();
        if key.is_empty() {
            return Ok(true);
        }
        let auth = self.header("Authorization").unwrap_or_default();
        let given = if auth.to_lowercase().starts_with("bearer ") {
            auth.chars().skip(7).collect::<String>().trim().to_string()
        } else {
            self.header("x-api-key").unwrap_or_default()
        };
        if key_matches(&given, &key) {
            return Ok(true);
        }
        self.json(401, &json!({"error": {"type": "authentication_error", "message": "missing or wrong API key"}}))?;
        Ok(false)
    }

    fn static_file(&mut self, file: &std::path::Path, ctype: &str, cache: Option<&str>) -> std::io::Result<()> {
        let body = std::fs::read(file)?;
        self.req.send_response(200, None);
        self.req.send_header("Content-Type", ctype);
        if let Some(c) = cache {
            self.req.send_header("Cache-Control", c);
        }
        self.req.send_header("Content-Length", &body.len().to_string());
        self.req.end_headers()?;
        self.req.write(&body)
    }

    fn do_get(&mut self) -> std::io::Result<()> {
        let path = self.route();
        let web = self.conf().root.join("serve").join("web");
        if let Some(name) = path.strip_prefix("/fonts/") {
            let f = web.join("fonts").join(name);
            if name.contains('/') || name.contains('\\') || !name.ends_with(".woff2") || !f.is_file() {
                return self.json(404, &err_obj(None, "not found"));
            }
            return self.static_file(&f, "font/woff2", Some("max-age=86400"));
        }
        if let Some(name) = path.strip_prefix("/web/") {
            let ext = name.rfind('.').filter(|&i| i > 0 && !name[..i].ends_with(['/', '\\'])).map(|i| &name[i..]).unwrap_or("");
            let ty = match ext {
                ".css" => Some("text/css; charset=utf-8"),
                ".js" => Some("text/javascript; charset=utf-8"),
                ".svg" => Some("image/svg+xml"),
                ".png" => Some("image/png"),
                _ => None,
            };
            let f = web.join(name);
            if name.contains('/') || name.contains('\\') || ty.is_none() || !f.is_file() {
                return self.json(404, &err_obj(None, "not found"));
            }
            return self.static_file(&f, ty.unwrap(), Some("no-cache"));
        }
        let router = self.router().clone();
        match path.as_str() {
            "/metrics" => {
                if self.authorized()? {
                    let shown = match query_first(&self.req.path, "model") {
                        None => Some(router.shown()),
                        Some(n) => router.service_for_str(&n),
                    };
                    match shown {
                        None => self.json(404, &err_obj(None, "model not found"))?,
                        Some(s) => self.json(200, &Value::Object(router.metrics(&s)))?,
                    }
                }
            }
            "/api/logs" => {
                if self.authorized()? {
                    self.logs()?;
                }
            }
            "" => {
                let f = web.join("index.html");
                let body = std::fs::read(f)?;
                self.req.send_response(200, None);
                self.req.send_header("Content-Type", "text/html; charset=utf-8");
                self.req.send_header("Content-Length", &body.len().to_string());
                self.req.end_headers()?;
                self.req.write(&body)?;
            }
            "/health" | "/api/health" => {
                let shown = router.shown();
                let load = shown.load_progress();
                let mut m = Map::new();
                m.insert("status".into(), json!("ok"));
                m.insert("max_context".into(), json!(shown.engine.max_context()));
                m.insert("model".into(), json!(shown.model));
                m.insert("images".into(), json!(shown.has_vision()));
                m.insert("api_key".into(), json!(!self.conf().api_key.is_empty()));
                m.insert("loaded".into(), json!(shown.loaded()));
                m.insert("loading".into(), json!(shown.loading()));
                m.insert("service".into(), json!("basalt"));
                m.insert("percent".into(), load.as_ref().and_then(|l| l.get("percent").cloned()).unwrap_or(Value::Null));
                m.insert("swap".into(), router.swap_status());
                self.json(200, &Value::Object(m))?;
            }
            "/status" => {
                if !self.authorized()? {
                    return Ok(());
                }
                let shown = router.shown();
                let s = shown.status_view();
                self.json(200, &Value::Object(s))?;
            }
            "/v1/models" | "/models" => {
                if self.authorized()? {
                    let models = Value::Array(router.models());
                    self.json(200, &json!({"object": "list", "data": models.clone(), "models": models}))?;
                }
            }
            "/props" => {
                if self.authorized()? {
                    self.props()?;
                }
            }
            "/slots" => {
                if self.authorized()? {
                    let shown = router.shown();
                    let loaded = !shown.engine.has_alive() || shown.engine.alive();
                    let v = if loaded { Value::Array(shown.slot_table().0) } else { json!([]) };
                    self.json(200, &v)?;
                }
            }
            "/v1/stats" => {
                if self.authorized()? {
                    let n = query_first(&self.req.path, "n")
                        .and_then(|v| py_int(&v))
                        .map(|v| v.clamp(0, serve_api::STATS_REQUESTS_MAX as i64) as usize)
                        .unwrap_or(serve_api::STATS_REQUESTS_DEFAULT);
                    let facts = self.conf().logs.facts();
                    let s = router.shown().stats(n, &facts);
                    self.json(200, &Value::Object(s))?;
                }
            }
            "/v1/status" => {
                if self.authorized()? {
                    let mut m = router.shown().v1_status();
                    m.insert("models".into(), Value::Array(router.models()));
                    m.insert("swap".into(), router.swap_status());
                    self.json(200, &Value::Object(m))?;
                }
            }
            _ => self.json(404, &err_obj(None, "not found"))?,
        }
        Ok(())
    }

    /// GET /api/logs?after=<id>&limit=<n>: the newest lines after that id, and the newest id.
    fn logs(&mut self) -> std::io::Result<()> {
        let path = self.req.path.clone();
        let number = |name: &str, default: Option<i64>| -> Option<i64> {
            match query_first(&path, name) {
                None => default,
                Some(v) => py_int(&v).map(|n| n.max(0)).or(default),
            }
        };
        let after = if query_has(&path, "after") { number("after", None) } else { None };
        let limit = if after.is_none() {
            number("limit", Some(LOG_FIRST_LINES as i64)).unwrap().min(LOG_MAX_LINES as i64)
        } else {
            number("limit", Some(LOG_MAX_LINES as i64)).unwrap().min(LOG_MAX_LINES as i64)
        };
        let logs = self.conf().logs.clone();
        let (lines, last) = logs.since(after.unwrap_or(0), limit);
        self.json(200, &json!({"lines": lines, "last": last}))
    }

    fn sse(&mut self) -> std::io::Result<()> {
        if self.sse_open {
            return Ok(());
        }
        self.sse_open = true;
        self.req.send_response(200, None);
        self.req.send_header("Content-Type", "text/event-stream");
        self.req.send_header("Cache-Control", "no-cache");
        self.req.send_header("X-Accel-Buffering", "no");
        self.cors();
        self.req.end_headers()
    }

    fn props(&mut self) -> std::io::Result<()> {
        let router = self.router().clone();
        let svc = match query_first(&self.req.path, "model") {
            None => Some(router.shown()),
            Some(n) => router.service_for_str(&n),
        };
        let Some(svc) = svc else { return self.json(404, &err_obj(None, "model not found")) };
        if !router.multi() {
            if let Err(e) = svc.wait_load() {
                if matches!(e, EngineErr::GpuBusy(_)) || e.is_died() {
                    return self.json(503, &err_obj(Some("server_error"), &e.message()));
                }
            }
        }
        if !svc.loaded() && !svc.engine.unloaded() {
            return self.json(503, &err_obj(None, "the engine is not running"));
        }
        let mut params = Map::new();
        for (k, v) in svc.sampling_defaults().iter() {
            if ["temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty", "presence_penalty", "frequency_penalty",
                "penalty_last_n"].contains(&k.as_str()) {
                let name = match k.as_str() {
                    "repetition_penalty" => "repeat_penalty",
                    "penalty_last_n" => "repeat_last_n",
                    x => x,
                };
                params.insert(name.into(), v.clone());
            }
        }
        params.insert("n_predict".into(), json!(-1));
        let mut props = Map::new();
        props.insert("default_generation_settings".into(), json!({"n_ctx": svc.engine.max_context(), "params": params}));
        props.insert("total_slots".into(), svc.slot_stats().get("total").cloned().unwrap_or(Value::Null));
        props.insert("model_alias".into(), json!(svc.model));
        props.insert("chat_template".into(), json!(svc.template_source()));
        props.insert("modalities".into(), json!({"vision": svc.has_vision(), "video": svc.video_on()}));
        props.insert("models_autoload".into(), json!(svc.engine.has_process()));
        props.insert("is_sleeping".into(), json!(!svc.loaded()));
        props.insert("chat_template_caps".into(), Value::Object(svc.dialect_caps()));
        if let Some(p) = svc.engine.model_path().filter(|p| !p.is_empty()) {
            props.insert("model_path".into(), json!(p));
        }
        let version = svc.engine.info().and_then(|i| i.get("version").cloned()).filter(pyjson::truthy);
        if let Some(v) = &version {
            props.insert("build_info".into(), json!(format!("basalt {}", pyjson::py_str(v))));
        }
        let mut caps: Vec<Value> = CAPABILITIES.iter().map(|c| json!(c)).collect();
        if svc.video_on() {
            caps.push(json!("video"));
        }
        props.insert("basalt".into(), json!({"version": version.as_ref().map(|v| Value::from(pyjson::py_str(v))).unwrap_or(Value::Null),
                                              "capabilities": caps}));
        self.json(200, &Value::Object(props))
    }

    fn control_body(&mut self) -> std::io::Result<bool> {
        let chunked = self.chunked();
        let mut length = 0usize;
        if !chunked {
            let l = py_int(&self.header("Content-Length").unwrap_or_else(|| "0".into()));
            let Some(l) = l else {
                self.json(400, &err_obj(None, "invalid Content-Length"))?;
                return Ok(false);
            };
            if !(0..=CONTROL_BODY_MAX as i64).contains(&l) {
                self.json(413, &err_obj(None, "control request body is limited to 64 KiB"))?;
                return Ok(false);
            }
            length = l as usize;
        }
        self.req.conn.set_timeout(Some(Duration::from_secs(2)));
        let complete = if chunked {
            match self.read_chunked(CONTROL_BODY_MAX, false) {
                Ok(_) => true,
                Err(Fail::BadBody(status, m)) => {
                    self.req.conn.set_timeout(Some(Duration::from_secs_f64(http::HTTP_BODY_S)));
                    self.json(status, &err_obj(None, &format!("control request: {m}")))?;
                    return Ok(false);
                }
                Err(_) => false,
            }
        } else {
            match self.req.conn.read(length) {
                Ok(b) => b.len() == length,
                Err(_) => false,
            }
        };
        self.req.conn.set_timeout(Some(Duration::from_secs_f64(http::HTTP_BODY_S)));
        if !complete {
            self.json(400, &err_obj(None, "incomplete control request body"))?;
        }
        Ok(complete)
    }

    fn own_page(&mut self, what: &str) -> std::io::Result<bool> {
        if !self.header("Content-Type").unwrap_or_default().starts_with("application/json") {
            self.json(415, &err_obj(None, "send application/json"))?;
            return Ok(false);
        }
        let origin = self.header("Origin").unwrap_or_default().trim_end_matches('/').to_string();
        let host = self.header("Host").unwrap_or_default();
        let trusted = self.conf().trusted_origins.clone();
        let tail = origin.split_once("://").map(|(_, b)| b).unwrap_or(&origin);
        if !origin.is_empty() && tail != host && !trusted.contains(&origin) {
            self.json(403, &err_obj(None, &format!("{what} only from the server's own page")))?;
            return Ok(false);
        }
        Ok(true)
    }

    fn unknown_model(&mut self, path: &str, name: Option<&Value>) -> std::io::Result<()> {
        let anthropic = path.starts_with("/v1/messages");
        let text = match name {
            None | Some(Value::Null) => if anthropic { "model: required".to_string() } else { "model is required".to_string() },
            Some(n) => {
                let s = pyjson::py_str(n);
                if anthropic { format!("model: {s}") } else { format!("The model '{s}' does not exist") }
            }
        };
        if anthropic {
            self.json(404, &json!({"type": "error", "error": {"type": "not_found_error", "message": text}}))
        } else {
            self.json(404, &serve_api::api::responses::error_body(&text, "invalid_request_error", Some("model"), Some("model_not_found")))
        }
    }

    fn end_request(&mut self) {
        if let Some(t) = self.turn.take() {
            self.router().leave(&t);
        }
        if let Some(d) = &self.watch_done {
            d.set();
        }
        if let Some(l) = &self.live {
            self.router().live.close(l);
        }
    }

    fn do_post(&mut self) -> std::io::Result<()> {
        if !self.authorized()? {
            return Ok(());
        }
        let path = self.route();
        if path.starts_with("/v1/") && self.foreign_page()? {
            return Ok(());
        }
        if (path == "/unload" || path == "/load") && !self.control_body()? {
            return Ok(());
        }
        if (path == "/unload" || path == "/load") && !self.own_page("the model can be loaded or unloaded")? {
            return Ok(());
        }
        let router = self.router().clone();
        if path == "/unload" {
            return match router.shown().unload() {
                Err(EngineErr::Stuck(m)) => self.json(503, &err_obj(Some("server_error"), &m)),
                Err(e) => self.json(503, &err_obj(Some("server_error"), &e.message())),
                Ok(r) => self.json(if r == "busy" { 409 } else { 200 }, &json!({"status": r})),
            };
        }
        if path == "/load" {
            self.svc = Some(router.shown());
            let r = (|| -> R<()> {
                self.take_turn(false)?;
                self.svc.clone().unwrap().load()?;
                Ok(())
            })();
            let out = match r {
                Ok(()) => self.json(200, &json!({"status": "loaded"})),
                Err(Fail::Api(ApiError::Engine(e))) if matches!(e, EngineErr::GpuBusy(_)) || e.is_died() => {
                    self.json(503, &err_obj(Some("server_error"), &e.message()))
                }
                Err(Fail::Gone) => Ok(()),
                Err(Fail::Api(e)) => {
                    self.end_request();
                    return Err(internal(e.message()));
                }
                Err(_) => Ok(()),
            };
            self.end_request();
            return out;
        }
        let r = self.post_body(&path);
        let out = match r {
            Ok(()) | Err(Fail::Done) | Err(Fail::Gone) => Ok(()),
            Err(Fail::BadBody(status, m)) => {
                if path == "/v1/responses" {
                    self.json(status, &serve_api::api::responses::error_body(&m, "invalid_request_error", None, None))
                } else {
                    self.json(status, &err_obj(Some("invalid_request_error"), &m))
                }
            }
            Err(Fail::Api(e)) => self.post_error(&path, e),
        };
        self.end_request();
        out
    }

    /// do_POST's except-chain
    fn post_error(&mut self, path: &str, e: ApiError) -> std::io::Result<()> {
        let responses = path == "/v1/responses";
        let rb = |m: &str, kind: &str, code: Option<&str>| serve_api::api::responses::error_body(m, kind, None, code);
        match e {
            ApiError::Engine(EngineErr::Error(m)) => {
                if responses { self.json(500, &rb(&m, "server_error", Some("server_error"))) } else { self.json(500, &err_obj(Some("server_error"), &m)) }
            }
            e if e.is_value() => {
                let m = e.message();
                if responses { self.json(400, &rb(&m, "invalid_request_error", None)) } else { self.json(400, &err_obj(Some("invalid_request_error"), &m)) }
            }
            ApiError::Engine(EngineErr::ModelBusy(m)) => self.json(409, &err_obj(Some("model_busy"), &m)),
            ApiError::Engine(EngineErr::GpuBusy(m)) | ApiError::Engine(EngineErr::Starting(m)) => self.json(503, &err_obj(Some("server_error"), &m)),
            ApiError::Engine(e) if e.is_died() => {
                let m = self.died(&e);
                self.json(503, &err_obj(Some("server_error"), &m))
            }
            ApiError::Engine(EngineErr::Stuck(m)) => self.json(503, &err_obj(Some("server_error"), &m)),
            ApiError::Engine(EngineErr::ClientGone) => Ok(()),
            other => Err(internal(other.message())),
        }
    }

    fn post_body(&mut self, path: &str) -> R<()> {
        let raw = self.body()?;
        let raw = if raw.is_empty() { b"{}".to_vec() } else { raw };
        let parsed = pyjson::loads_bytes(&raw).map_err(|e| Fail::Api(ApiError::value(e.0)))?;
        let Value::Object(mut req) = parsed else { return Err(Fail::Api(ApiError::value("send a JSON object"))) };
        let router = self.router().clone();
        let io = |r: std::io::Result<()>| -> R<()> { r.map_err(|e| Fail::Api(ApiError::Internal(e.to_string()))).and(Err(Fail::Done)) };
        if path.starts_with("/v1/responses/") {
            return io(self.json(404, &serve_api::api::responses::error_body(
                "this server keeps no responses (stateless); send the whole conversation to POST /v1/responses",
                "invalid_request_error", None, Some("not_found"))));
        }
        if path == "/v1/load" || path == "/v1/unload" {
            if !self.own_page("the model can be loaded or unloaded").map_err(|e| Fail::Api(ApiError::Internal(e.to_string())))? {
                return Err(Fail::Done);
            }
            let target = match req.get("model") {
                None | Some(Value::Null) => Some(router.shown()),
                Some(m) => router.service_for(Some(m)),
            };
            let Some(target) = target else { return io(self.json(404, &err_obj(None, "model not found"))) };
            self.svc = Some(target.clone());
            let result;
            if path == "/v1/unload" {
                let r = target.unload()?;
                if r == "busy" {
                    return Err(EngineErr::ModelBusy("a request is running or queued".into()).into());
                }
                result = r;
            } else if target.loading() {
                self.take_turn(false)?;
                target.load()?;
                result = "loaded".to_string();
            } else {
                self.take_turn(false)?;
                if !target.fifo().acquire(Some(Duration::ZERO)) {
                    return Err(EngineErr::ModelBusy("a request is running or queued".into()).into());
                }
                let r = (|| -> R<()> {
                    if target.busy_or_queued() {
                        return Err(EngineErr::ModelBusy("a request is running or queued".into()).into());
                    }
                    target.ensure_loaded()?;
                    target.touch_last_request();
                    Ok(())
                })();
                target.fifo().release();
                r?;
                result = "loaded".to_string();
            }
            let mut m = Map::new();
            m.insert("status".into(), json!(result));
            for (k, v) in target.v1_status() {
                m.insert(k, v);
            }
            return io(self.json(200, &Value::Object(m)));
        }
        if path == "/v1/answer_now" {
            let rid = req.get("request_id").cloned().unwrap_or(Value::Null);
            if !rid.is_null() && !rid.is_string() {
                return Err(Fail::Api(ApiError::value("request_id: a string")));
            }
            let rid_s = rid.as_str().map(|s| s.to_string());
            let ses = if rid_s.as_deref().map(|s| !s.is_empty()).unwrap_or(false) { None } else { self.session_id(Some(&req)) };
            let r = router.live.answer_now(rid_s.as_deref(), ses.as_deref());
            return io(match r {
                Ok(v) => self.json(200, &v),
                Err(e) => self.json(e.status, &e.body()),
            });
        }
        if path == "/v1/vram" {
            if !self.own_page("the VRAM reserve can be changed").map_err(|e| Fail::Api(ApiError::Internal(e.to_string())))? {
                return Err(Fail::Done);
            }
            let r = req.get("reserve_mib").cloned().unwrap_or(Value::Null);
            let ok = r.is_null() || matches!(&r, Value::Number(n) if (n.is_i64() && n.as_i64().unwrap() >= 0) || n.is_u64());
            if !ok {
                return Err(Fail::Api(ApiError::value("reserve_mib: a whole number of MiB (0 or more), or null for the start's")));
            }
            return io(match router.vram(r.as_i64()) {
                Ok(out) => {
                    let code = if out.get("status").and_then(|s| s.as_str()) == Some("pending") { 202 } else { 200 };
                    self.json(code, &Value::Object(out))
                }
                Err(e) if e.is_died() => self.json(503, &err_obj(Some("server_error"), &e.message())),
                Err(e) => return Err(e.into()),
            });
        }
        let requested = req.get("model").cloned();
        let svc = if requested.as_ref().map(|v| v.is_null()).unwrap_or(true) && (path == "/tokenize" || path == "/detokenize") {
            Some(router.shown())
        } else {
            router.service_for(requested.as_ref())
        };
        let Some(svc) = svc else {
            return io(self.unknown_model(path, requested.as_ref().filter(|v| !v.is_null())));
        };
        self.svc = Some(svc.clone());
        svc.set_request_path(Some(path));
        let access = self.image_access();
        let _docs = serve_api::api::documents::sources(access);
        match path {
            "/v1/responses" => self.responses(&svc, req),
            "/v1/chat/completions" => self.openai(&svc, &mut req, path),
            "/v1/messages" => self.anthropic(&svc, &mut req, path),
            "/v1/messages/count_tokens" => self.count_tokens(&svc, &req),
            "/tokenize" => self.tokenize(&svc, &req),
            "/detokenize" => self.detokenize(&svc, &req),
            _ => io(self.json(404, &err_obj(None, "not found"))),
        }
    }

    fn openai(&mut self, svc: &Svc, req: &mut Obj, path: &str) -> R<()> {
        use serve_api::api::{logit_bias, logprobs, openai, structured};
        let (messages, tools, kw) = serve_api::frontend::openai_to_messages(req)?;
        let (tools, force) = serve_api::frontend::forced_call(req.get("tool_choice"), tools, &|n| svc.forced_opening(n))?;
        structured::check_format(req.get("response_format"))?;
        svc.require_grammar(req.get("response_format"))?;
        serve_engine::config_checks::checked_sampling(req).map_err(ApiError::value)?;
        openai::checked_return_progress(req)?;
        self.open_live(&format!("chatcmpl-{}", &uuid_hex()[..24]), req);
        let lp = logprobs::checked(req)?;
        req.insert(LOGPROBS_KEY.into(), lp.map(Value::from).unwrap_or(Value::Null));
        let stream = req.get("stream").map(pyjson::truthy).unwrap_or(false);
        self.take_turn(stream)?;
        svc.load()?;
        let bias = logit_bias::checked(req, &*svc.engine)?;
        let grammar = svc.grammar_for(tools.as_ref(), req.get("tool_choice"), req.get("tools"), req.get("response_format"))?;
        req.insert(GRAMMAR_KEY.into(), grammar.unwrap_or(Value::Null));
        let max_new = int_of_first(&[req.get("max_completion_tokens"), req.get("max_tokens")])?;
        svc.reasoning_budget(req)?;
        serve_text::text::stop_strings(req).map_err(ApiError::value)?;
        let (ids, thinking, max_new) = svc.prepare(&messages, tools.as_ref(), &kw, max_new, force.as_deref(), Some(self.image_access()))?;
        serve_api::service::debug_req("openai", req, &messages, tools.as_ref(), max_new, thinking, ids.len());
        let cancel = self.client_cancel();
        let session = self.session_id(Some(req));
        let chunks = openai::openai_chunks(svc, req, bias, &ids, thinking, tools.clone(), max_new, cancel.clone(), force.clone(),
                                           session, self.live.clone());
        let mut chunks = serve_api::prompt::primed(chunks, &ids)?;
        if !stream {
            let out = openai::openai_collect(&mut chunks)?;
            return ok(self.json(200, &out));
        }
        ok(self.sse())?;
        let _ = path;
        let mut write_err = false;
        loop {
            let item = match chunks.next() {
                None => break,
                Some(Ok(c)) => c,
                Some(Err(e)) => {
                    // mid-stream
                    if e.is_died() {
                        let err = json!({"error": {"type": "server_error", "message": self.died(&e)}});
                        let _ = self.req.write(&[b"data: ".to_vec(), pyjson::dumps(&err, true).into_bytes(), b"\n\ndata: [DONE]\n\n".to_vec()].concat());
                    } else if e.is_value() {
                        let err = json!({"error": {"type": "server_error", "message": e.message()}});
                        let _ = self.req.write(&[b"data: ".to_vec(), pyjson::dumps(&err, true).into_bytes(), b"\n\ndata: [DONE]\n\n".to_vec()].concat());
                    } else {
                        return Err(Fail::Api(ApiError::Engine(e)));
                    }
                    return Err(Fail::Done);
                }
            };
            let r = match item {
                None => self.req.write(b": keep-alive\n\n"),
                Some(c) => self.req.write(&[b"data: ".to_vec(), dumps(&c), b"\n\n".to_vec()].concat()),
            }
            .and_then(|_| self.req.flush());
            if r.is_err() {
                write_err = true;
                break;
            }
        }
        if write_err {
            cancel.set();
            let _ = chunks.close();
            return Err(Fail::Done);
        }
        if self.req.write(b"data: [DONE]\n\n").is_err() {
            cancel.set();
            let _ = chunks.close();
        }
        Err(Fail::Done)
    }


    fn anthropic(&mut self, svc: &Svc, req: &mut Obj, _path: &str) -> R<()> {
        use serve_api::api::{anthropic, logprobs, structured};
        let (messages, tools, kw) = serve_api::frontend::anthropic_to_messages(req, svc.anthropic_think_unasked())?;
        let (tools, force) = serve_api::frontend::forced_call(req.get("tool_choice"), tools, &|n| svc.forced_opening(n))?;
        let max_new = int_of_first(&[req.get("max_tokens")])?;
        let response_format = structured::anthropic_format(req)?;
        if let Err(e) = svc.require_grammar(response_format.as_ref()) {
            if e.is_value() {
                return Err(Fail::Api(ApiError::value(e.message().replace(
                    "response_format json_schema and json_object", "output_config.format json_schema"))));
            }
            return Err(e.into());
        }
        let grammar = svc.grammar_for(tools.as_ref(), req.get("tool_choice"), req.get("tools"), response_format.as_ref())?;
        req.insert(GRAMMAR_KEY.into(), grammar.unwrap_or(Value::Null));
        svc.reasoning_budget(req)?;
        serve_engine::config_checks::checked_sampling(req).map_err(ApiError::value)?;
        let lp = logprobs::checked(req)?;
        req.insert(LOGPROBS_KEY.into(), lp.map(Value::from).unwrap_or(Value::Null));
        serve_text::text::stop_strings(req).map_err(ApiError::value)?;
        self.open_live(&format!("msg_{}", &uuid_hex()[..24]), req);
        let stream = req.get("stream").map(pyjson::truthy).unwrap_or(false);
        self.take_turn(stream)?;
        svc.load()?;
        let (ids, thinking, max_new) = svc.prepare(&messages, tools.as_ref(), &kw, max_new, force.as_deref(), Some(self.image_access()))?;
        serve_api::service::debug_req("anthropic", req, &messages, tools.as_ref(), max_new, thinking, ids.len());
        let cancel = self.client_cancel();
        let session = self.session_id(Some(req));
        let events = anthropic::anthropic_events(svc, req, &ids, thinking, tools.clone(), max_new, cancel.clone(), force.clone(),
                                                 session, self.live.clone());
        let mut events = serve_api::prompt::primed(events, &ids)?;
        if !stream {
            let out = anthropic::anthropic_collect(&mut events)?;
            return ok(self.json(200, &out));
        }
        ok(self.sse())?;
        loop {
            let item = match events.next() {
                None => break,
                Some(Ok(c)) => c,
                Some(Err(e)) => {
                    if e.is_died() || e.is_value() {
                        let m = if e.is_died() { self.died(&e) } else { e.message() };
                        let err = json!({"type": "error", "error": {"type": "api_error", "message": m}});
                        let _ = self.req.write(&[b"event: error\ndata: ".to_vec(), pyjson::dumps(&err, true).into_bytes(), b"\n\n".to_vec()].concat());
                        return Err(Fail::Done);
                    }
                    return Err(Fail::Api(ApiError::Engine(e)));
                }
            };
            let r = match item {
                None => self.req.write(b": keep-alive\n\n"),
                Some((name, e)) => self.req.write(&[format!("event: {name}\n").into_bytes(), b"data: ".to_vec(), dumps(&e), b"\n\n".to_vec()].concat()),
            }
            .and_then(|_| self.req.flush());
            if r.is_err() {
                cancel.set();
                let _ = events.close();
                break;
            }
        }
        Err(Fail::Done)
    }

    fn responses(&mut self, svc: &Svc, req: Obj) -> R<()> {
        use serve_api::api::responses as rs;
        let prepared = self.responses_prepare(svc, req);
        let (ids, thinking, tools, max_new, asm, req, force) = match prepared {
            Ok(p) => p,
            Err(Fail::Api(ApiError::Responses(e))) => return ok(self.json(e.status, &e.body())).and(Err(Fail::Done)),
            Err(Fail::Api(ApiError::Engine(EngineErr::ModelBusy(m)))) => {
                return ok(self.json(409, &rs::error_body(&m, "server_error", None, Some("model_busy")))).and(Err(Fail::Done));
            }
            Err(Fail::Api(ApiError::Engine(e))) if matches!(e, EngineErr::GpuBusy(_) | EngineErr::Starting(_) | EngineErr::Stuck(_)) || e.is_died() => {
                return ok(self.json(503, &rs::error_body(&e.message(), "server_error", None, Some("server_error")))).and(Err(Fail::Done));
            }
            Err(e) => return Err(e),
        };
        let cancel = self.client_cancel();
        let session = self.session_id(Some(&req));
        let events = rs::response_events(svc, &asm, &ids, thinking, tools, max_new, &req, cancel.clone(), force, session, self.live.clone());
        let mut items = match serve_api::prompt::primed(events, &ids) {
            Ok(i) => i,
            Err(ApiError::Engine(EngineErr::Error(m))) => return Err(Fail::Api(ApiError::Engine(EngineErr::Error(m)))),
            Err(e) if e.is_value() => {
                let m = e.message();
                let code = if m.contains("context") { Some("context_length_exceeded") } else { None };
                let err = ResponsesError::new(m, Some("input"), code);
                return ok(self.json(err.status, &err.body())).and(Err(Fail::Done));
            }
            Err(e) => return Err(e.into()),
        };
        let stream = req.get("stream").map(pyjson::truthy).unwrap_or(false);
        if !stream {
            return match rs::collect(&mut items) {
                Ok(result) => ok(self.json(200, &result)).and(Err(Fail::Done)),
                Err(ApiError::Engine(e)) if e.is_died() => {
                    let m = self.died(&e);
                    ok(self.json(503, &rs::error_body(&m, "server_error", None, Some("server_error")))).and(Err(Fail::Done))
                }
                Err(ApiError::Engine(EngineErr::Error(m))) => {
                    ok(self.json(500, &rs::error_body(&m, "server_error", None, Some("server_error")))).and(Err(Fail::Done))
                }
                Err(e) if e.is_value() => {
                    let m = e.message();
                    let code = if m.contains("context") { Some("context_length_exceeded") } else { None };
                    let err = ResponsesError::new(m, Some("input"), code);
                    ok(self.json(err.status, &err.body())).and(Err(Fail::Done))
                }
                Err(e) => Err(e.into()),
            };
        }
        let mut last = Instant::now();
        let r: R<()> = (|| {
            if self.sse().is_err() {
                cancel.set();
                return Ok(());
            }
            loop {
                let item = match items.next() {
                    None => break,
                    Some(Ok(x)) => x,
                    Some(Err(e)) => {
                        if e.is_died() || e.is_value() {
                            let m = if e.is_died() { self.died(&e) } else { e.message() };
                            let ev = asm.lock().unwrap().failed(&m, "server_error");
                            let _ = self.send_event(&ev).and_then(|_| self.req.flush());
                            return Ok(());
                        }
                        return Err(Fail::Api(ApiError::Engine(e)));
                    }
                };
                let w = match item {
                    None => {
                        if last.elapsed().as_secs_f64() >= 15.0 {
                            let ev = asm.lock().unwrap().in_progress();
                            last = Instant::now();
                            self.send_event(&ev)
                        } else {
                            self.req.write(b": keep-alive\n\n")
                        }
                    }
                    Some(ev) => {
                        last = Instant::now();
                        self.send_event(&ev)
                    }
                }
                .and_then(|_| self.req.flush());
                if w.is_err() {
                    cancel.set();
                    return Ok(());
                }
            }
            Ok(())
        })();
        let _ = items.close();
        r.and(Err(Fail::Done))
    }

    fn send_event(&mut self, e: &Value) -> std::io::Result<()> {
        let ty = e.get("type").map(pyjson::py_str).unwrap_or_default();
        self.req.write(&[format!("event: {ty}\n").into_bytes(), b"data: ".to_vec(), dumps(e), b"\n\n".to_vec()].concat())
    }

    #[allow(clippy::type_complexity)]
    fn responses_prepare(&mut self, svc: &Svc, mut req: Obj)
        -> R<(serve_engine::iface::Prompt, bool, Option<Value>, i64, Arc<std::sync::Mutex<serve_api::api::responses::Assembler>>, Obj, Option<String>)> {
        use serve_api::api::responses as rs;
        rs::check_request(&req)?;
        let messages = rs::input_messages(&req)?;
        let (tools, names, skipped) = rs::request_tools(&req)?;
        let (tools, force) = serve_api::frontend::forced_call(req.get("tool_choice"), tools, &|n| svc.forced_opening(n))
            .map_err(|e| ApiError::Responses(ResponsesError::new(e.message(), Some("tool_choice"), None)))?;
        let kw = rs::template_kwargs(&req)?;
        // handler.py wraps text_format's own ResponsesError (a ValueError) the same way: param "text.format"
        let tf = rs::text_format(&req).map_err(|e| {
            ApiError::Responses(ResponsesError::new(e.message().replace("response_format", "text.format"), Some("text.format"), None))
        })?;
        serve_api::api::structured::check_format(tf.as_ref())
            .map_err(|e| ApiError::Responses(ResponsesError::new(e.message().replace("response_format", "text.format"), Some("text.format"), None)))?;
        if let Err(e) = svc.require_grammar(tf.as_ref()) {
            if e.is_value() {
                return Err(ApiError::Responses(ResponsesError::new(e.message().replace("response_format", "text.format"), Some("text.format"), None)).into());
            }
            return Err(e.into());
        }
        let grammar = svc.grammar_for(tools.as_ref(), req.get("tool_choice"), req.get("tools"), tf.as_ref())?;
        req.insert(GRAMMAR_KEY.into(), grammar.unwrap_or(Value::Null));
        let fresh = svc.note_hosted_tools(&skipped);
        if !fresh.is_empty() {
            stdout::out(&format!("[server] warning: /v1/responses ignores hosted tools {}", fresh.join(", ")));
        }
        if let Err(e) = svc.reasoning_budget(&req) {
            if e.is_value() {
                return Err(ApiError::Responses(ResponsesError::new(e.message(), Some("reasoning_budget_tokens"), None)).into());
            }
            return Err(e.into());
        }
        if let Err(m) = serve_engine::config_checks::checked_sampling(&req) {
            let param = m.split(':').next().unwrap_or("").to_string();
            return Err(ApiError::Responses(ResponsesError::new(m, Some(&param), None)).into());
        }
        let lp = rs::requested_logprobs(&req)?;
        req.insert(LOGPROBS_KEY.into(), lp.map(Value::from).unwrap_or(Value::Null));
        if let Err(m) = serve_text::text::stop_strings(&req) {
            return Err(ApiError::Responses(ResponsesError::new(m, Some("stop"), None)).into());
        }
        self.open_live(&rs::new_id("resp"), &req);
        let stream = req.get("stream").map(pyjson::truthy).unwrap_or(false);
        self.take_turn(stream)?;
        svc.load()?;
        let max_out = py_int_or_zero(req.get("max_output_tokens")).unwrap_or(0);
        let (ids, thinking, max_new) = match svc.prepare(&messages, tools.as_ref(), &kw, max_out, force.as_deref(), Some(self.image_access())) {
            Ok(x) => x,
            Err(ApiError::Responses(e)) => return Err(ApiError::Responses(e).into()),
            Err(e) if e.is_value() => {
                let m = e.message();
                let code = if m.contains("context") { Some("context_length_exceeded") } else { None };
                return Err(ApiError::Responses(ResponsesError::new(m, Some("input"), code)).into());
            }
            Err(e) => return Err(e.into()),
        };
        serve_api::service::debug_req("responses", &req, &messages, tools.as_ref(), max_new, thinking, ids.len());
        let include = match req.get("include") {
            Some(Value::Array(a)) => a.clone(),
            _ => vec![],
        };
        let enc = include.iter().any(|v| v.as_str() == Some("reasoning.encrypted_content"));
        let rid = self.live.as_ref().map(|l| l.id.clone()).unwrap_or_default();
        let asm = rs::Assembler::new(&req, &svc.model, ids.len(), names, enc, &rid);
        let _ = BIAS_KEY;
        Ok((ids, thinking, tools, max_new, Arc::new(std::sync::Mutex::new(asm)), req, force))
    }

    fn tokenize(&mut self, svc: &Svc, req: &Obj) -> R<()> {
        let content = req.get("content");
        let parse = req.get("parse_special").cloned().unwrap_or(Value::Bool(true));
        let Some(Value::String(content)) = content else { return Err(Fail::Api(ApiError::value("content: a string"))) };
        let Value::Bool(parse) = parse else { return Err(Fail::Api(ApiError::value("parse_special: true or false"))) };
        let ids = svc.tok.encode(content, parse, &[]);
        ok(self.json(200, &json!({"tokens": ids}))).and(Err(Fail::Done))
    }

    fn detokenize(&mut self, svc: &Svc, req: &Obj) -> R<()> {
        let bad = || Fail::Api(ApiError::value("tokens: a list of token ids"));
        let Some(Value::Array(ids)) = req.get("tokens") else { return Err(bad()) };
        let mut out = Vec::new();
        for t in ids {
            match t {
                Value::Number(n) if n.is_u64() => {
                    let v = n.as_u64().unwrap();
                    if v > u32::MAX as u64 {
                        return Err(Fail::Api(ApiError::value("tokens: an id outside the model's vocabulary")));
                    }
                    out.push(v as u32);
                }
                _ => return Err(bad()),
            }
        }
        let text = svc.tok.decode(&out).map_err(|_| Fail::Api(ApiError::value("tokens: an id outside the model's vocabulary")))?;
        ok(self.json(200, &json!({"content": text}))).and(Err(Fail::Done))
    }

    fn count_tokens(&mut self, svc: &Svc, req: &Obj) -> R<()> {
        let (messages, tools, kw) = serve_api::frontend::anthropic_to_messages(req, svc.anthropic_think_unasked())?;
        let (tools, _) = serve_api::frontend::forced_call(req.get("tool_choice"), tools, &|n| svc.forced_opening(n))?;
        serve_text::text::stop_strings(req).map_err(ApiError::value)?;
        let n = svc.encode_prompt(&messages, tools.as_ref(), &kw)?.len();
        ok(self.json(200, &json!({"input_tokens": n}))).and(Err(Fail::Done))
    }
}

fn ok(r: std::io::Result<()>) -> R<()> {
    r.map_err(|e| Fail::Api(ApiError::Internal(e.to_string())))
}

fn internal(m: String) -> std::io::Error {
    std::io::Error::other(format!("internal: {m}"))
}

/// int(text) as Python reads a header or query value
pub fn py_int(text: &str) -> Option<i64> {
    let t = text.trim();
    let t2 = t.replace('_', "");
    if t2.is_empty() || t.starts_with('_') || t.ends_with('_') || t.contains("__") {
        return None;
    }
    let digits = t2.trim_start_matches(['+', '-']);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) || t2.len() - digits.len() > 1 {
        return None;
    }
    t2.parse::<i64>().ok()
}

fn parse_hex(text: &str) -> Option<i64> {
    let t = text.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let body = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")).unwrap_or(body);
    let body = body.replace('_', "");
    if body.is_empty() {
        return None;
    }
    let v = i64::from_str_radix(&body, 16).ok()?;
    Some(if neg { -v } else { v })
}

/// `int(a or b or ... or 0)` over JSON values, as handler.py reads max_tokens: ValueError's text for a bad string or
/// NaN, a TypeError (an unhandled exception in Python: Internal) for a list or an object.
fn int_of_first(vals: &[Option<&Value>]) -> R<i64> {
    let v = vals.iter().flatten().find(|v| pyjson::truthy(v)).copied();
    let Some(v) = v else { return Ok(0) };
    match v {
        Value::Bool(b) => Ok(*b as i64),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Ok(i),
            None => {
                let x = n.as_f64().unwrap_or(0.0);
                if x.is_nan() {
                    Err(Fail::Api(ApiError::value("cannot convert float NaN to integer")))
                } else if x.is_infinite() {
                    Err(Fail::Api(ApiError::Internal("OverflowError: cannot convert float infinity to integer".into())))
                } else {
                    Ok(x.trunc().clamp(i64::MIN as f64, i64::MAX as f64) as i64)
                }
            }
        },
        Value::String(s) => py_int(s).ok_or_else(|| Fail::Api(ApiError::value(format!(
            "invalid literal for int() with base 10: {}", pyjson::py_str_repr(s))))),
        other => Err(Fail::Api(ApiError::Internal(format!(
            "TypeError: int() argument must be a string, a bytes-like object or a real number, not '{}'",
            pyjson::type_name(other))))),
    }
}

/// `req.get(key) or 0` for max_output_tokens (prepare compares it with 0)
fn py_int_or_zero(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|x| x.trunc() as i64)),
        Some(Value::Bool(b)) => Some(*b as i64),
        _ => None,
    }
}

pub fn uuid_hex() -> String {
    let mut b = [0u8; 16];
    fill_random(&mut b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn fill_random(b: &mut [u8]) {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    for chunk in b.chunks_mut(8) {
        let mut h = RandomState::new().build_hasher();
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        let v = h.finish().to_le_bytes();
        chunk.copy_from_slice(&v[..chunk.len()]);
    }
}

impl App for Handler {
    fn parse_request_extra(&self, req: &mut Req) -> bool {
        let host = req.headers.get("Host").map(|s| s.to_string());
        let (key, names, any) = {
            let c = self.conf.read().unwrap();
            (c.api_key.clone(), c.host_names.clone(), c.allowed_hosts.iter().any(|h| h == "*"))
        };
        if !key.is_empty() || host_allowed(host.as_deref(), &names, any) {
            return true;
        }
        let hr = match &host {
            Some(h) => pyjson::py_str_repr(h),
            None => "None".into(),
        };
        stdout::out(&format!("[server] error: refused Host {hr} from {} (--api-key KEY, or --allowed-host)", peer_of(req.client_address())));
        let mut ctx = Ctx { h: self, req, watch_done: None, cancel: None, turn: None, sse_open: false, live: None, svc: None };
        let _ = ctx.json(403, &json!({"error": {"type": "forbidden", "message": format!(
            "Host {hr} is not allowed (DNS rebinding protection). Reaching the server under this name on purpose? Start \
             the server with --allowed-host NAME, or with --api-key KEY, which turns this check off")}}));
        false
    }

    fn dispatch(&self, req: &mut Req) -> bool {
        let cmd = req.command.clone();
        let mut ctx = Ctx { h: self, req, watch_done: None, cancel: None, turn: None, sse_open: false, live: None, svc: None };
        let r = match cmd.as_str() {
            "GET" => ctx.do_get(),
            "POST" => ctx.do_post(),
            "OPTIONS" => ctx.do_options(),
            _ => return false,
        };
        if let Err(e) = r {
            let m = e.to_string();
            if let Some(inner) = m.strip_prefix("internal: ") {
                self.handle_error(req_peer(&ctx), inner);
            }
            // a write that failed: the client left (Python: the exception ends the request quietly or in handle_error)
        }
        true
    }

    fn handle_error(&self, peer: Option<&SocketAddr>, what: &str) {
        let p = match peer {
            Some(a) => format!("('{}', {})", a.ip(), a.port()),
            None => "None".into(),
        };
        stdout::err(&format!("----------------------------------------\nException occurred during processing of request from {p}\n{what}\n----------------------------------------"));
    }
}

fn req_peer<'a>(ctx: &'a Ctx<'_>) -> Option<&'a SocketAddr> {
    ctx.req.client_address()
}
