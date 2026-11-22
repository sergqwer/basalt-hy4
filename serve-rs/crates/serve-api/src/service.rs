//! serve/service.py: the engine behind a lock, with request bookkeeping and the metrics the stats page reads.

use crate::api::grammar::{grammar_spec, unenforced};
use crate::api::logprobs::LpItem;
use crate::error::{AResult, ApiError};
use crate::frontend::{effort_kwargs, images_of, media_of, mark_think_literals, unmark_think_literals, Event, Tags};
use crate::image_source::ImageSources;
use crate::models::Dialect;
use crate::py::{self, get, getv, py_repr, Obj};
use crate::template::{ChatTemplate, TemplateErr};
use crate::vision::{Vision, IMAGE_WAITING_BYTES, VIDEO_MAX_TOKENS, VIDEO_REQUEST_BYTES};
use serde_json::{json, Map, Value};
use serve_engine::errors::EngineErr;
use serve_engine::gen::{Gen, Stop, Yielder};
use serve_engine::iface::{thread_ident, Engine, Flag, Prompt, PyLock};
use serve_engine::util::stdout::out;
use serve_text::prompt::{overflow_error, room_error, PromptText, CTX_SLACK};
use serve_text::pyjson::{py_fixed, py_round};
use serve_text::tokenizer::{control_literals, Tokenizer};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

pub const ANSWER_ROOM_MIN_CAP: i64 = 32;
pub const ANSWER_RESERVE_MIN: i64 = 512;
pub const QUEUE_BEAT_S: f64 = 10.0;
pub const KEEPALIVE_S: f64 = 10.0;
pub const RATE_WINDOW_S: f64 = 2.0;
pub const RATE_SAMPLES: usize = 1024;
pub const LOG_FIRST_LINES: i64 = 300;
pub const LOG_MAX_LINES: i64 = 500;
pub const REQUESTS_SHOWN: usize = 12;
pub const BEST_MIN_OUT: i64 = 256;
pub const BEST_MIN_READ: i64 = 4096;
pub const BEST_MAX_READ_TOK_S: f64 = 200000.0;
pub const RATE_MIN_SPAN_S: f64 = 0.25;
pub const READ_PRINT_EVERY: f64 = 5.0;
pub const LIVE_EVERY_S: f64 = 2.0;
pub const LIVE_MIN_S: f64 = 0.5;
pub const MIN_RATE_TOKENS: i64 = 16;
pub const MIN_PP_RATE_TOKENS: i64 = 512;
pub const WAKE_VRAM_WAIT_S: f64 = 15.0;
pub const SLEEP_RAM_MARGIN_MIB: i64 = 8192;

/// --debug
pub static DEBUG: AtomicBool = AtomicBool::new(false);

pub fn effort_text(effort: &str) -> &'static str {
    match effort {
        "low" => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        _ => "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary elaboration.",
    }
}

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn monotonic() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

pub fn fv(x: f64) -> Value {
    serde_json::Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
}

/// The answer reserve when thinking: "auto" or a fixed token count (0: none).
#[derive(Clone, Debug, PartialEq)]
pub enum AnswerRoom {
    Auto,
    Fixed(i64),
}

impl std::fmt::Display for AnswerRoom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnswerRoom::Auto => f.write_str("auto"),
            AnswerRoom::Fixed(n) => write!(f, "{n}"),
        }
    }
}

/// The settings serve() and the config set after construction.
#[derive(Clone)]
pub struct Settings {
    pub api_key: String,
    pub cors_origins: Vec<String>,
    pub max_payload: usize,
    pub trusted_origins: Vec<String>,
    pub allowed_hosts: Vec<String>,
    pub host_names: HashSet<String>,
    pub image_sources: ImageSources,
    pub effort_end: bool,
    pub idle_unload_s: f64,
    pub idle_sleep_s: f64,
    pub min_free_vram_mib: i64,
    /// a command (string: shell) or argv run before loading
    pub before_load: Option<Value>,
    pub reasoning_budget_tokens: i64,
    pub answer_room: AnswerRoom,
    pub reasoning_effort: Option<Value>,
    pub thinking_loop_tokens: i64,
    pub anthropic_think_unasked: bool,
    pub start_tries: Option<u32>,
    pub gpu_indices: Vec<u32>,
    pub gpu_index: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            api_key: String::new(),
            cors_origins: vec![],
            max_payload: 256 << 20,
            trusted_origins: vec![],
            allowed_hosts: vec![],
            host_names: ["localhost", "127.0.0.1", "::1"].iter().map(|s| s.to_string()).collect(),
            image_sources: ImageSources::default(),
            effort_end: false,
            idle_unload_s: 0.0,
            idle_sleep_s: 0.0,
            min_free_vram_mib: 0,
            before_load: None,
            reasoning_budget_tokens: 0,
            answer_room: AnswerRoom::Auto,
            reasoning_effort: None,
            thinking_loop_tokens: 256,
            anthropic_think_unasked: true,
            start_tries: None,
            gpu_indices: vec![0],
            gpu_index: 0,
        }
    }
}

pub type Rate = VecDeque<(f64, i64)>;

/// Everything Python keeps under status_lock.
pub struct StatusAll {
    pub status: Obj,
    pub rate: Rate,
    /// parallel mode: each running request's own status and rate window, in start order
    pub live_reqs: Vec<(u64, Obj, Rate)>,
    pub slot_last: BTreeMap<i64, Obj>,
    pub waiting: Vec<(u64, Obj)>,
    pub best: Obj,
    pub best_gen: u64,
    pub history: VecDeque<Obj>,
    pub requests_done: i64,
    pub totals: Obj,
    pub last_timings: Option<Obj>,
    pub last_request_at: Option<f64>,
}

struct VramQ {
    queued: Option<Obj>,
    queued_gen: u64,
    applied: Option<Obj>,
    error: Option<String>,
    worker: bool,
}

/// What Service.run yields.
#[derive(Debug, Clone)]
pub enum RunItem {
    Event(Event),
    Ping,
    Logprobs(LpItem),
    Done(Obj),
}

pub type RunGen = Gen<RunItem>;

/// The identity of the thread a run's body runs on (the one that calls engine.generate), for the readers of the
/// engine's per-request state from other threads.
#[derive(Clone, Default)]
pub struct RunThread(pub Arc<AtomicU64>);

impl RunThread {
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// What prepare() hands to run(): the prompt and the answer cap.
pub struct Prepared {
    pub ids: Prompt,
    pub thinking: bool,
    pub max_new: i64,
    /// the request's embeddings file (images), removed when the run ends
    pub embeddings: Option<String>,
}

pub struct Service {
    pub engine: Arc<dyn Engine>,
    pub tok: Arc<dyn Tokenizer>,
    pub template: ChatTemplate,
    pub model: String,
    pub vision: Option<Arc<Vision>>,
    pub dialect: Dialect,
    pub literals: Tags,
    pub fit_max_tokens: bool,
    pub video_max_tokens: i64,
    pub grammar: bool,
    pub grammar_engine: bool,
    pub sampling_defaults: Obj,
    pub fifo: PyLock,
    vram_q: Mutex<VramQ>,
    vram_cv: Condvar,
    pub stopping: Flag,
    pub cfg: RwLock<Settings>,
    images_waiting: Mutex<u64>,
    pub status_lock: Mutex<StatusAll>,
    live_at: Mutex<f64>,
    pub loading_flag: AtomicBool,
    /// per handler thread: this request's embeddings file (prepare -> run) and API path
    embeddings: Mutex<std::collections::HashMap<u64, String>>,
    request_path: Mutex<std::collections::HashMap<u64, String>>,
    pub booting: AtomicBool,
    pub load_kind: Mutex<&'static str>,
    pub boot_error: Mutex<Option<EngineErr>>,
    pub on_loaded: RwLock<Option<Arc<dyn Fn() + Send + Sync>>>,
    pub stop_ids: HashSet<u32>,
    pub started_at: f64,
    pub hosted_tools_noted: Mutex<HashSet<String>>,
    sleep_refused_at: Mutex<Option<f64>>,
    pub vram_reserve: Mutex<Option<i64>>,
    pub telemetry: RwLock<Option<Arc<serve_engine::util::telemetry::Telemetry>>>,
}

fn int(v: Option<&Value>) -> i64 {
    v.and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64))).unwrap_or(0)
}

fn fl(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| x.as_f64())
}

fn some(v: Option<&Value>) -> Option<&Value> {
    v.filter(|x| !x.is_null())
}

impl Service {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: Arc<dyn Engine>,
        tok: Arc<dyn Tokenizer>,
        template: ChatTemplate,
        model_name: &str,
        vision: Option<Arc<Vision>>,
        sampling_defaults: Option<Obj>,
        fit_max_tokens: bool,
        grammar: bool,
        grammar_engine: bool,
        dialect: Option<Dialect>,
        video_max_tokens: Option<i64>,
    ) -> Self {
        let dialect = dialect.unwrap_or_else(Dialect::qwen);
        let literals = dialect.literal_tags(&control_literals(tok.as_ref()));
        let mut stop_ids = HashSet::new();
        for s in &dialect.stop_texts {
            stop_ids.extend(tok.encode(s, true, &[]));
        }
        let status = json!({"busy": false, "queued": 0}).as_object().unwrap().clone();
        let totals = json!({"prompt": 0, "reused": 0, "written": 0, "prompt_ms": 0.0, "decode_ms": 0.0})
            .as_object()
            .unwrap()
            .clone();
        let best_gen = engine.gen();
        Service {
            engine,
            tok,
            template,
            model: model_name.to_string(),
            vision,
            dialect,
            literals,
            fit_max_tokens,
            video_max_tokens: video_max_tokens.unwrap_or(VIDEO_MAX_TOKENS),
            grammar,
            grammar_engine,
            sampling_defaults: sampling_defaults.unwrap_or_default(),
            fifo: PyLock::new(),
            vram_q: Mutex::new(VramQ { queued: None, queued_gen: 0, applied: None, error: None, worker: false }),
            vram_cv: Condvar::new(),
            stopping: Flag::new(),
            cfg: RwLock::new(Settings::default()),
            images_waiting: Mutex::new(0),
            status_lock: Mutex::new(StatusAll {
                status,
                rate: VecDeque::new(),
                live_reqs: vec![],
                slot_last: BTreeMap::new(),
                waiting: vec![],
                best: Obj::new(),
                best_gen,
                history: VecDeque::new(),
                requests_done: 0,
                totals,
                last_timings: None,
                last_request_at: None,
            }),
            live_at: Mutex::new(0.0),
            loading_flag: AtomicBool::new(false),
            embeddings: Mutex::new(Default::default()),
            request_path: Mutex::new(Default::default()),
            booting: AtomicBool::new(false),
            load_kind: Mutex::new("weights"),
            boot_error: Mutex::new(None),
            on_loaded: RwLock::new(None),
            stop_ids,
            started_at: now(),
            hosted_tools_noted: Mutex::new(HashSet::new()),
            sleep_refused_at: Mutex::new(None),
            vram_reserve: Mutex::new(None),
            telemetry: RwLock::new(None),
        }
    }

    pub fn settings(&self) -> Settings {
        self.cfg.read().unwrap().clone()
    }

    pub fn is_loading(&self) -> bool {
        self.loading_flag.load(Ordering::SeqCst)
    }

    pub fn set_last_request_at(&self, t: f64) {
        self.status_lock.lock().unwrap().last_request_at = Some(t);
    }

    /// This model's row in /v1/models.
    pub fn model_entry_v(&self) -> Value {
        let value = if self.is_loading() {
            "loading"
        } else if self.loaded() {
            "loaded"
        } else {
            "unloaded"
        };
        let input = if self.vision.is_some() {
            let mut v = vec![json!("text"), json!("image")];
            if self.video_on() {
                v.push(json!("video"));
            }
            v
        } else {
            vec![json!("text")]
        };
        json!({"id": self.model, "object": "model", "created": self.started_at as i64, "owned_by": "basalt",
               "status": {"value": value}, "asleep": self.engine.asleep(),
               "meta": {"n_ctx": self.engine.max_context()},
               "architecture": {"input_modalities": input, "output_modalities": ["text"]}})
    }

    pub fn served_names(&self) -> Vec<String> {
        let mut names = vec![self.model.clone()];
        if let Some(p) = self.engine.model_path().filter(|p| !p.is_empty()) {
            names.push(p.replace('\\', "/").rsplit('/').next().unwrap_or("").to_string());
        }
        names
    }

    pub fn loaded(&self) -> bool {
        !self.engine.has_alive() || self.engine.alive()
    }

    /// reasoning_budget: the request's thinking budget in tokens, or None; Err for a non-integer.
    pub fn reasoning_budget_of(&self, req: Option<&Obj>, use_default: bool) -> AResult<Option<i64>> {
        let empty = Obj::new();
        let req = req.unwrap_or(&empty);
        let mut value = some(req.get("reasoning_budget_tokens")).cloned();
        if value.is_none() {
            value = some(req.get("thinking_budget_tokens")).cloned();
        }
        if value.is_none() {
            value = some(req.get("reasoning_budget")).cloned();
        }
        if value.is_none() {
            if let Some(t @ Value::Object(_)) = req.get("thinking") {
                value = some(get(t, "budget_tokens")).cloned();
            }
        }
        if value.is_none() && use_default {
            value = Some(Value::from(self.cfg.read().unwrap().reasoning_budget_tokens));
        }
        let Some(value) = value else { return Ok(None) };
        let Some(n) = py::whole(&value) else {
            return Err(ApiError::value(format!(
                "reasoning_budget_tokens: expected a whole number (0 = none), not {}",
                py_repr(&value)
            )));
        };
        Ok(if n > 0 { Some(n) } else { None })
    }

    /// answer_budget: limit thinking to leave answer space; an explicit non-positive request budget opts out.
    pub fn answer_budget(&self, req: Option<&Obj>, max_new: i64, budget: Option<i64>) -> AResult<Option<i64>> {
        let room_cfg = self.cfg.read().unwrap().answer_room.clone();
        let reserve = match room_cfg {
            AnswerRoom::Auto => {
                if max_new < ANSWER_ROOM_MIN_CAP {
                    return Ok(budget);
                }
                (ANSWER_RESERVE_MIN.max(max_new.div_euclid(4))).min(max_new.div_euclid(2))
            }
            AnswerRoom::Fixed(n) if n > 0 => n.min(max_new - 1),
            _ => return Ok(budget),
        };
        let request_budget = self.reasoning_budget_of(req, false)?;
        let empty = Obj::new();
        let r = req.unwrap_or(&empty);
        let thinking = r.get("thinking");
        let explicit = ["reasoning_budget_tokens", "thinking_budget_tokens", "reasoning_budget"]
            .iter()
            .any(|k| some(r.get(*k)).is_some())
            || thinking.map(|t| t.is_object() && some(get(t, "budget_tokens")).is_some()).unwrap_or(false);
        if explicit && request_budget.is_none() {
            return Ok(budget);
        }
        let room = max_new - reserve;
        if budget.map(|b| b > room).unwrap_or(true) {
            out(&format!("[server] thinking budget {room} tokens, output cap {max_new} (answer-room {room_cfg})"));
            return Ok(Some(room));
        }
        Ok(budget)
    }

    fn vision_down(&self) -> bool {
        self.vision.as_ref().map(|v| !v.alive()).unwrap_or(false)
    }

    pub fn stop(&self) {
        self.stopping.set();
    }

    /// load_progress: {"kind", and for weights the engine's load status}, None when no load runs.
    pub fn load_progress_v(&self) -> Option<Value> {
        if !self.is_loading() {
            return None;
        }
        let kind = *self.load_kind.lock().unwrap();
        let mut m = Obj::new();
        m.insert("kind".into(), Value::from(kind));
        if kind == "weights" {
            if let Some(s) = self.engine.load_status() {
                m.extend(s);
            }
        }
        Some(Value::Object(m))
    }

    /// start_load: load in the background at start.
    pub fn start_load(self: &Arc<Self>) {
        *self.load_kind.lock().unwrap() = "weights";
        self.loading_flag.store(true, Ordering::SeqCst);
        let me = self.clone();
        std::thread::spawn(move || {
            me.booting.store(true, Ordering::SeqCst);
            let _ = me.load();
            me.booting.store(false, Ordering::SeqCst);
            me.loading_flag.store(false, Ordering::SeqCst);
        });
    }

    /// ensure_loaded: start the engine if it is not running. Caller holds fifo.
    pub fn ensure_loaded(&self) -> Result<(), EngineErr> {
        if let Some(e) = self.boot_error.lock().unwrap().clone() {
            let log = match &e {
                EngineErr::StartError { log, .. } => log.clone(),
                _ => None,
            };
            return Err(EngineErr::StartError { msg: e.message(), log });
        }
        if self.loaded() && !self.vision_down() && !self.engine.asleep() {
            return Ok(());
        }
        *self.load_kind.lock().unwrap() = if self.engine.asleep() { "wake" } else { "weights" };
        self.loading_flag.store(true, Ordering::SeqCst);
        let r = self.ensure_loaded_inner().and_then(|_| {
            if let Some(f) = self.on_loaded.read().unwrap().clone() {
                f();
            }
            Ok(())
        });
        if let Err(e) = &r {
            if (e.is_died() || matches!(e, EngineErr::GpuBusy(_))) && self.booting.load(Ordering::SeqCst) {
                let be = match e {
                    EngineErr::StartError { .. } => e.clone(),
                    other => EngineErr::StartError { msg: other.message(), log: None },
                };
                *self.boot_error.lock().unwrap() = Some(be);
            }
        }
        self.loading_flag.store(false, Ordering::SeqCst);
        r
    }

    fn ensure_loaded_inner(&self) -> Result<(), EngineErr> {
        if self.stopping.is_set() {
            return Err(EngineErr::Died("the server is stopping".into()));
        }
        if self.engine.asleep() {
            self.wake()?;
        }
        if self.loaded() && !self.vision_down() {
            return Ok(());
        }
        let cfg = self.settings();
        if let Some(cmd) = &cfg.before_load {
            let shown = match cmd {
                Value::String(s) => s.clone(),
                Value::Array(a) => a.iter().map(py::py_str).collect::<Vec<_>>().join(" "),
                other => py::py_str(other),
            };
            out(&format!("[server] before_load: {shown}"));
            if let Err(e) = run_hook(cmd) {
                out(&format!("[server] warning: before_load command failed ({e}), loading anyway"));
            }
        }
        if cfg.min_free_vram_mib != 0 {
            let read = || serve_engine::util::vram::free_mib(cfg.gpu_index).map(|x| x as i64);
            let mut free = read();
            let deadline = now() + 15.0;
            while free.map(|f| f < cfg.min_free_vram_mib).unwrap_or(false) && now() < deadline {
                std::thread::sleep(Duration::from_millis(500));
                free = read();
            }
            if let Some(f) = free.filter(|f| *f < cfg.min_free_vram_mib) {
                let held = serve_engine::util::vram::holders(Some(&[cfg.gpu_index]), None);
                return Err(EngineErr::GpuBusy(format!(
                    "the GPU is in use by another program: {f} MiB of VRAM free, the model needs {} (min_free_vram_mib){} - it stays unloaded until that is free",
                    cfg.min_free_vram_mib,
                    if held.is_empty() { String::new() } else { format!("; held by {held}") }
                )));
            }
        }
        if self.vision_down() {
            out("[server] restarting the vision encoder");
            if let Some(v) = &self.vision {
                v.restart().map_err(EngineErr::Died)?;
            }
        }
        if self.loaded() {
            return Ok(());
        }
        if self.stopping.is_set() {
            return Err(EngineErr::Died("the server is stopping".into()));
        }
        let again = self.engine.known_ctx() != 0;
        if !again {
        } else if self.engine.unloaded() {
            out("[server] loading the model again");
        } else {
            let mut note = self.engine.death_note();
            if note.is_empty() {
                let code = self.engine.exit_code().map(|c| c.to_string()).unwrap_or_else(|| "None".into());
                note = format!("the engine stopped (exit code {code})");
            }
            let log = self.engine.log_path();
            out(&format!(
                "[server] warning: {note}, restarting it{}",
                log.map(|l| format!(", log {l}")).unwrap_or_default()
            ));
        }
        let tries = if self.booting.load(Ordering::SeqCst) { Some(1) } else { cfg.start_tries };
        if let Err(e) = self.engine.restart(tries) {
            if let EngineErr::StartError { log, .. } = &e {
                out(&format!("[server] error: {}", e.message()));
                if let Some(l) = log.as_ref().filter(|l| !l.is_empty()) {
                    out(&format!("[server] engine log: {l}"));
                }
            }
            return Err(e);
        }
        if again {
            out("[server] engine running again");
        }
        let reserve = *self.vram_reserve.lock().unwrap();
        if let Some(r) = reserve {
            if self.engine.has_process() {
                if let Err(e) = self.engine.vram(Some(r)) {
                    if e.is_value() || e.is_died() {
                        out(&format!("[server] warning: VRAM reserve of {r} MiB not applied ({})", e.message()));
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------ VRAM
    /// POST /v1/vram.
    pub fn vram(self: &Arc<Self>, reserve_mib: Option<i64>) -> Result<Map<String, Value>, EngineErr> {
        self.vram_v(reserve_mib).map(|v| v.as_object().cloned().unwrap_or_default())
    }

    fn vram_v(self: &Arc<Self>, reserve_mib: Option<i64>) -> Result<Value, EngineErr> {
        if !self.engine.has_process() {
            return Err(EngineErr::Value("this engine cannot resize its VRAM use".into()));
        }
        let replaced;
        {
            let mut q = self.vram_q.lock().unwrap();
            replaced = q.queued.is_some();
            q.queued = Some(json!({"reserve_mib": reserve_mib}).as_object().unwrap().clone());
            q.queued_gen += 1;
        }
        if let Some(o) = self.apply_vram(0.0)? {
            return Ok(o);
        }
        let mut q = self.vram_q.lock().unwrap();
        let Some(job) = q.queued.clone() else {
            let mut m = json!({"status": "ok"}).as_object().unwrap().clone();
            if let Some(a) = &q.applied {
                m.extend(a.clone());
            }
            return Ok(Value::Object(m));
        };
        if !q.worker {
            q.worker = true;
            let me = self.clone();
            std::thread::spawn(move || me.vram_worker());
        }
        Ok(json!({"status": "pending", "reserve_mib": job.get("reserve_mib").cloned().unwrap_or(Value::Null),
                  "replaced": replaced}))
    }

    fn apply_vram(&self, wait: f64) -> Result<Option<Value>, EngineErr> {
        let take = |l: &PyLock| l.acquire(Some(Duration::from_secs_f64(wait)));
        if !take(&self.fifo) {
            return Ok(None);
        }
        let ctl = if self.engine.batch() > 0 { self.engine.ctl() } else { None };
        if let Some(c) = ctl {
            if !take(c) {
                self.fifo.release();
                return Ok(None);
            }
        }
        let mut job_gen = None;
        let result = (|| -> Result<Option<Value>, EngineErr> {
            if !self.status_lock.lock().unwrap().live_reqs.is_empty() {
                return Ok(None);
            }
            let (job, g) = {
                let q = self.vram_q.lock().unwrap();
                (q.queued.clone(), q.queued_gen)
            };
            let Some(job) = job else { return Ok(None) };
            job_gen = Some(g);
            let reserve = job.get("reserve_mib").and_then(|v| v.as_i64());
            let out_v = if !self.loaded() {
                *self.vram_reserve.lock().unwrap() = reserve;
                json!({"status": "not loaded", "reserve_mib": reserve, "note": "applied when the model loads"})
            } else {
                let got = self.engine.vram(reserve)?;
                *self.vram_reserve.lock().unwrap() = reserve;
                let cache = if some(got.get("expert_cache_mib")).is_some() {
                    format!(
                        "expert cache {} of {} MiB ",
                        py::py_str(&got["expert_cache_mib"]),
                        got.get("expert_cache_full_mib").map(py::py_str).unwrap_or_else(|| "None".into())
                    )
                } else {
                    String::new()
                };
                out(&format!(
                    "[server] vram: {} MiB free, {cache}({} experts)",
                    got.get("vram_free_mib").map(py::py_str).unwrap_or_else(|| "None".into()),
                    got.get("expert_slots").map(py::py_str).unwrap_or_else(|| "None".into())
                ));
                let mut m = json!({"status": "ok"}).as_object().unwrap().clone();
                m.extend(got);
                Value::Object(m)
            };
            let mut q = self.vram_q.lock().unwrap();
            q.applied = Some(json!({"reserve_mib": reserve, "at": now() as i64}).as_object().unwrap().clone());
            q.error = None;
            Ok(Some(out_v))
        })();
        if let Some(c) = ctl {
            c.release();
        }
        self.fifo.release();
        if let Some(g) = job_gen {
            let mut q = self.vram_q.lock().unwrap();
            if q.queued_gen == g {
                q.queued = None;
            }
            self.vram_cv.notify_all();
        }
        result
    }

    fn vram_worker(self: Arc<Self>) {
        while !self.stopping.is_set() {
            if self.vram_q.lock().unwrap().queued.is_none() {
                break;
            }
            match self.apply_vram(0.2) {
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Ok(Some(_)) => {}
                Err(e) if e.is_value() || e.is_died() => {
                    self.vram_q.lock().unwrap().error = Some(e.message());
                    out(&format!("[server] warning: queued VRAM reserve not applied ({})", e.message()));
                }
                Err(_) => {}
            }
        }
        let mut q = self.vram_q.lock().unwrap();
        q.worker = false;
        if self.stopping.is_set() {
            q.queued = None;
        }
        self.vram_cv.notify_all();
    }

    /// _vram_gate: wait while a resize is queued, with keep-alives; false when cancelled.
    pub(crate) fn vram_gate_pub(&self, y: &Yielder<RunItem>, cancel: &Flag) -> Result<bool, Stop> {
        loop {
            {
                let q = self.vram_q.lock().unwrap();
                if q.queued.is_none() {
                    return Ok(true);
                }
                let (q, _) = self.vram_cv.wait_timeout(q, Duration::from_secs_f64(QUEUE_BEAT_S)).unwrap();
                if q.queued.is_none() {
                    return Ok(true);
                }
            }
            if cancel.is_set() {
                return Ok(false);
            }
            y.yield_(RunItem::Ping)?;
        }
    }

    pub fn vram_state(&self) -> Value {
        let q = self.vram_q.lock().unwrap();
        json!({"pending": q.queued.clone(), "applied": q.applied.clone(), "error": q.error.clone()})
    }

    fn say_died(&self, e: &EngineErr) {
        let note = self.engine.death_note();
        let log = self.engine.log_path();
        let msg = e.message();
        let why = if !note.is_empty() && msg.contains(&note) {
            msg
        } else if !note.is_empty() {
            format!("{msg}: {note}")
        } else {
            msg
        };
        out(&format!(
            "[server] warning: {why} (the next request restarts it{})",
            log.map(|l| format!(", log {l}")).unwrap_or_default()
        ));
    }

    /// require_grammar: a JSON response format the engine would not enforce is refused.
    pub fn require_grammar(&self, response_format: Option<&Value>) -> AResult<()> {
        if unenforced(self.grammar, response_format)? {
            return Err(ApiError::value(
                "response_format json_schema and json_object (structured output) need grammar enabled on this server; start it without --no-grammar, or send strict: true",
            ));
        }
        Ok(())
    }

    pub fn grammar_for(
        &self,
        tools: Option<&Value>,
        tool_choice: Option<&Value>,
        raw_tools: Option<&Value>,
        response_format: Option<&Value>,
    ) -> AResult<Option<Value>> {
        if !self.grammar_engine {
            return Ok(None);
        }
        grammar_spec(self.grammar, tools, tool_choice, raw_tools, response_format)
    }

    /// load: start the engine if unloaded.
    pub fn load(&self) -> Result<(), EngineErr> {
        self.set_last_request_at(now());
        if self.loaded() && !self.vision_down() && !self.engine.asleep() {
            return Ok(());
        }
        self.fifo.acquire(None);
        let r = self.ensure_loaded();
        self.fifo.release();
        r
    }

    pub fn wait_load(&self) -> Result<(), EngineErr> {
        if self.is_loading() {
            self.load()?;
        }
        Ok(())
    }

    pub fn swap_out(&self) -> Result<(), EngineErr> {
        self.fifo.acquire(None);
        let r = (|| {
            if self.engine.has_process() {
                self.engine.unload()?;
            }
            if let Some(v) = &self.vision {
                v.unload();
            }
            Ok(())
        })();
        self.fifo.release();
        r
    }

    fn wake(&self) -> Result<(), EngineErr> {
        let t = now();
        let check = |parked: Option<&BTreeMap<u32, u64>>| self.wake_vram_check(parked);
        match self.engine.wake(Some(&check)) {
            Err(EngineErr::WakeBlocked(m)) => {
                out(&format!("[server] error: the model cannot wake: {m}"));
                return Err(EngineErr::GpuBusy(format!("the model cannot wake: {m}")));
            }
            Err(e) if e.is_value() || e.is_died() => {
                out(&format!("[server] warning: waking the model failed ({}); loading it again", e.message()));
                return Ok(());
            }
            Err(e) => return Err(e),
            Ok(_) => {}
        }
        out(&format!(
            "[server] model awake in {} s (driver restore {} s)",
            py_fixed(now() - t, 1),
            py_fixed(self.engine.restore_s(), 1)
        ));
        Ok(())
    }

    fn wake_vram_check(&self, parked: Option<&BTreeMap<u32, u64>>) -> Option<String> {
        let parked = parked.filter(|p| !p.is_empty())?;
        let deadline = now() + WAKE_VRAM_WAIT_S;
        loop {
            let mut short = Vec::new();
            for (g, need) in parked {
                if let Some(free) = serve_engine::util::vram::free_mib(*g) {
                    if free < *need {
                        short.push((*g, *need, free));
                    }
                }
            }
            if short.is_empty() {
                return None;
            }
            if now() >= deadline {
                let gs: Vec<u32> = short.iter().map(|s| s.0).collect();
                let held = serve_engine::util::vram::holders(Some(&gs), None);
                let what: Vec<String> =
                    short.iter().map(|(g, n, f)| format!("GPU {g} needs {n} MiB and has {f} MiB free")).collect();
                return Some(format!(
                    "{}{} - the model stays asleep until that is free",
                    what.join("; "),
                    if held.is_empty() { String::new() } else { format!("; held by {held}") }
                ));
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// sleep: "asleep", "not loaded", "busy", "no ram" or "unsupported".
    pub fn sleep(&self, idle_for: Option<f64>) -> Result<&'static str, EngineErr> {
        if !self.engine.has_process() || !self.engine.can_sleep() {
            return Ok("unsupported");
        }
        if self.stopping.is_set() {
            return Ok("busy");
        }
        if !self.fifo.acquire(Some(Duration::ZERO)) {
            return Ok("busy");
        }
        let ctl = if self.engine.batch() > 0 { self.engine.ctl() } else { None };
        if let Some(c) = ctl {
            if !c.acquire(Some(Duration::ZERO)) {
                self.fifo.release();
                return Ok("busy");
            }
        }
        let r = (|| -> Result<&'static str, EngineErr> {
            if !self.engine.alive() {
                return Ok("not loaded");
            }
            if self.engine.asleep() {
                return Ok("asleep");
            }
            let last = {
                let s = self.status_lock.lock().unwrap();
                if py::truthy(s.status.get("busy").unwrap_or(&Value::Null))
                    || py::truthy(s.status.get("queued").unwrap_or(&Value::Null))
                {
                    return Ok("busy");
                }
                s.last_request_at.unwrap_or(self.started_at)
            };
            if let Some(idle) = idle_for {
                if now() - last < idle {
                    return Ok("busy");
                }
                if *self.sleep_refused_at.lock().unwrap() == Some(last) {
                    return Ok("unsupported");
                }
            }
            let t = now();
            let check = || self.sleep_fits();
            match self.engine.sleep(Some(&check)) {
                Err(e) if e.is_value() => {
                    if idle_for.is_some() {
                        *self.sleep_refused_at.lock().unwrap() = Some(last);
                    }
                    out(&format!("[server] warning: the model stays awake ({})", e.message()));
                    return Ok(if e.message().contains("RAM") { "no ram" } else { "unsupported" });
                }
                Err(e) => return Err(e),
                Ok(_) => {}
            }
            out(&format!(
                "[server] model asleep{} in {} s: its VRAM is free, the next request wakes it",
                idle_for.map(|i| format!(" after {} s idle", py_fixed(i, 0))).unwrap_or_default(),
                py_fixed(now() - t, 1)
            ));
            Ok("asleep")
        })();
        if let Some(c) = ctl {
            c.release();
        }
        self.fifo.release();
        r
    }

    fn sleep_fits(&self) -> Option<String> {
        let used = self
            .engine
            .pid()
            .and_then(|p| serve_engine::util::vram::process_mib(p, None).map(|x| x as i64))
            .or_else(|| self.used_vram_mib());
        let avail = available_ram_mib();
        match (used, avail) {
            (_, None) => Some("RAM unreadable".into()),
            (None, Some(_)) => None,
            (Some(u), Some(a)) => {
                if a - u < SLEEP_RAM_MARGIN_MIB {
                    Some(format!(
                        "RAM: {a} MiB free, {u} MiB of GPU state to hold and {SLEEP_RAM_MARGIN_MIB} MiB to keep"
                    ))
                } else {
                    None
                }
            }
        }
    }

    pub fn used_vram_mib(&self) -> Option<i64> {
        let cfg = self.settings();
        let gpus = if cfg.gpu_indices.is_empty() { vec![cfg.gpu_index] } else { cfg.gpu_indices.clone() };
        let mut total = 0i64;
        for i in gpus {
            let nv = serve_engine::util::telemetry::Nvml::new(i);
            if !nv.ok() {
                return None;
            }
            let (_, _, used) = nv.memory_info()?;
            total += (used >> 20) as i64;
        }
        Some(total)
    }

    pub fn start_idle_sleep(self: &Arc<Self>, announce: bool, name: Option<&str>) {
        let idle = self.cfg.read().unwrap().idle_sleep_s;
        if idle == 0.0 || !self.engine.has_process() {
            return;
        }
        if !self.engine.can_sleep() {
            out("[server] warning: idle sleep is not available here (the driver cannot suspend a process); use idle_unload");
            return;
        }
        if announce {
            out(&format!("[server] {} sleeps after {} s idle", name.unwrap_or("model"), serve_text::pyjson::py_format_g(idle)));
        }
        let me = self.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs_f64((idle / 4.0).clamp(1.0, 30.0)));
            if let Err(e) = me.sleep(Some(idle)) {
                if e.is_died() {
                    out(&format!("[server] warning: idle sleep failed ({})", e.message()));
                }
            }
        });
    }

    /// unload: "unloaded", "not loaded", "busy" or "unsupported".
    pub fn unload_idle(&self, idle_for: Option<f64>) -> Result<&'static str, EngineErr> {
        if !self.engine.has_process() {
            return Ok("unsupported");
        }
        if !self.fifo.acquire(Some(Duration::ZERO)) {
            return Ok("busy");
        }
        let r = (|| -> Result<&'static str, EngineErr> {
            if !self.engine.alive() {
                return Ok("not loaded");
            }
            let last = {
                let s = self.status_lock.lock().unwrap();
                if py::truthy(s.status.get("busy").unwrap_or(&Value::Null))
                    || py::truthy(s.status.get("queued").unwrap_or(&Value::Null))
                {
                    return Ok("busy");
                }
                s.last_request_at.unwrap_or(self.started_at)
            };
            if let Some(idle) = idle_for {
                if now() - last < idle {
                    return Ok("busy");
                }
            }
            self.engine.unload()?;
            if let Some(v) = &self.vision {
                v.unload();
            }
            out(&format!(
                "[server] model unloaded{}, the next request loads it",
                idle_for.map(|i| format!(" after {} s idle", py_fixed(i, 0))).unwrap_or_default()
            ));
            Ok("unloaded")
        })();
        self.fifo.release();
        r
    }

    pub fn start_idle_unload(self: &Arc<Self>) {
        let idle = self.cfg.read().unwrap().idle_unload_s;
        if idle == 0.0 || !self.engine.has_process() {
            return;
        }
        out(&format!("[server] model unloads after {} s idle", py_num(idle)));
        let me = self.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs_f64((idle / 4.0).clamp(1.0, 30.0)));
            if let Err(EngineErr::Stuck(m)) = me.unload_idle(Some(idle)) {
                out(&format!("[server] warning: idle unload failed ({m})"));
            }
        });
    }

    // ------------------------------------------------------------------------------------------------ prompts
    /// render_prompt: with effort_end, a non-default effort follows as a short system turn right before the answer.
    pub fn render_prompt(&self, messages: &[Value], tools: Option<&Value>, kwargs: &Obj) -> AResult<String> {
        let mut kwargs = kwargs.clone();
        if let Some(e) = kwargs.get("reasoning_effort").cloned() {
            if !(e.is_null() || e == "") {
                kwargs.insert("reasoning_effort".into(), self.dialect.effort_level(&e)?);
            }
        }
        let kwargs = self.dialect.template_kwargs(&kwargs);
        let effort = kwargs.get("reasoning_effort").cloned().unwrap_or(Value::Null);
        let off = kwargs.get("enable_thinking") == Some(&Value::Bool(false));
        let msgs = Value::Array(messages.to_vec());
        let render = |agp: bool, kw: &Obj| -> AResult<String> {
            self.template.render(&msgs, tools, agp, kw).map_err(|e| match e {
                TemplateErr::Request(m) => ApiError::value(m),
                TemplateErr::Other(m) => ApiError::internal(m),
            })
        };
        let default_effort = effort.is_null() || effort == "" || effort == "xhigh" || effort == "high";
        if !self.cfg.read().unwrap().effort_end || self.dialect.effort_turn.is_empty() || (!off && default_effort) {
            // add_generation_prompt from kwargs wins as **kwargs would pass it
            let agp = kwargs.get("add_generation_prompt").map(py::truthy).unwrap_or(true);
            let mut kw = kwargs.clone();
            kw.shift_remove("add_generation_prompt");
            return render(agp, &kw);
        }
        let base: Obj = kwargs
            .iter()
            .filter(|(k, _)| k.as_str() != "reasoning_effort" && k.as_str() != "enable_thinking")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let history = render(false, &base)?;
        let full = render(true, &kwargs)?;
        let mine = render(false, &kwargs)?;
        if !full.starts_with(&mine) {
            return Ok(full);
        }
        let tail = if off {
            String::new()
        } else {
            self.dialect.effort_turn.replacen("{}", effort_text(effort.as_str().unwrap_or("medium")), 1)
        };
        Ok(format!("{history}{tail}{}", &full[mine.len()..]))
    }

    /// render_text: the rendered prompt and the byte spans where special-token text stays plain text.
    pub fn render_text(&self, messages: &[Value], tools: Option<&Value>, kwargs: &Obj) -> AResult<(String, Vec<(usize, usize)>)> {
        let (marked, marked_tools, changed) = mark_think_literals(messages, tools, &self.literals);
        let prompt = self.render_prompt(&marked, marked_tools.as_ref(), kwargs)?;
        if !changed {
            return Ok((prompt, vec![]));
        }
        Ok(unmark_think_literals(&prompt, &self.literals))
    }

    /// thinking_kwargs: the server's default effort when the request names no effort and no thinking switch.
    pub fn thinking_kwargs(&self, kwargs: &Obj) -> AResult<Obj> {
        let effort = self.cfg.read().unwrap().reasoning_effort.clone();
        let Some(effort) = effort else { return Ok(kwargs.clone()) };
        if kwargs.contains_key("reasoning_effort") || kwargs.contains_key("enable_thinking") {
            return Ok(kwargs.clone());
        }
        let mut out = kwargs.clone();
        out.extend(effort_kwargs(&effort)?);
        Ok(out)
    }

    /// encode_prompt: the rendered, tokenized prompt.
    pub fn encode_prompt_videos(&self, messages: &[Value], tools: Option<&Value>, kwargs: &Obj, videos: Option<&[String]>) -> AResult<Vec<u32>> {
        let kwargs = self.thinking_kwargs(kwargs)?;
        let (mut prompt, mut plain) = self.render_text(messages, tools, &kwargs)?;
        if let Some(v) = videos {
            (prompt, plain) = self.splice_videos(&prompt, &plain, v)?;
        }
        Ok(self.tok.encode(&prompt, true, &plain))
    }

    fn splice_videos(&self, prompt: &str, plain: &[(usize, usize)], videos: &[String]) -> AResult<(String, Vec<(usize, usize)>)> {
        let marker = &self.dialect.video_marker;
        let mut found = Vec::new();
        let mut at = 0;
        while let Some(j) = prompt[at..].find(marker.as_str()).map(|j| j + at) {
            if !plain.iter().any(|&(a, b)| a < j + marker.len() && j < b) {
                found.push(j);
            }
            at = j + 1;
            while at < prompt.len() && !prompt.is_char_boundary(at) {
                at += 1;
            }
        }
        if found.len() != videos.len() {
            return Err(refuse_images("the prompt and its videos do not match"));
        }
        let mut out = String::new();
        let mut at = 0;
        for (j, text) in found.iter().zip(videos) {
            out.push_str(&prompt[at..*j]);
            out.push_str(text);
            at = j + marker.len();
        }
        out.push_str(&prompt[at..]);
        let growth: Vec<(usize, i64)> =
            found.iter().zip(videos).map(|(j, t)| (*j, t.len() as i64 - marker.len() as i64)).collect();
        let spans = plain
            .iter()
            .map(|&(a, b)| {
                let d: i64 = growth.iter().filter(|(j, _)| *j < a).map(|(_, d)| d).sum();
                ((a as i64 + d) as usize, (b as i64 + d) as usize)
            })
            .collect();
        Ok((out, spans))
    }

    /// video_on: image input is on, the model reads video and its encoder found FFmpeg.
    pub fn video_on(&self) -> bool {
        let Some(v) = &self.vision else { return false };
        if self.dialect.video_pad.is_empty() {
            return false;
        }
        v.video_state().map(|s| s.starts_with("on")).unwrap_or(false)
    }

    fn check_videos(&self, items: &[Value]) -> AResult<()> {
        if self.vision.is_none() {
            return Err(refuse_images("this server was started without image input (--vision), so it cannot read videos"));
        }
        if self.dialect.video_pad.is_empty() {
            return Err(refuse_images("this model reads no video"));
        }
        for item in items {
            if getv(item, "role") == "system" {
                return Err(refuse_images("a system prompt cannot hold a video"));
            }
            if getv(item, "role") == "tool" {
                return Err(refuse_images("a tool result cannot hold a video"));
            }
            if !py::truthy(getv(item, "source")) {
                return Err(refuse_images(crate::image_source::VIDEO.usage));
            }
            if let Some(fps) = some(get(item, "fps")) {
                let ok = fps.as_f64().map(|f| f > 0.0 && f <= 60.0).unwrap_or(false);
                if !ok {
                    return Err(refuse_images(&format!(
                        "a video's fps must be a number above 0 and at most 60, not {}",
                        py_repr(fps)
                    )));
                }
            }
            if let Some(fr) = some(get(item, "max_frames")) {
                let ok = py::as_int(fr).map(|n| (4..=768).contains(&n)).unwrap_or(false);
                if !ok {
                    return Err(refuse_images(&format!(
                        "a video's max_frames must be a whole number from 4 to 768, not {}",
                        py_repr(fr)
                    )));
                }
            }
        }
        Ok(())
    }

    fn video_state_check(&self) -> AResult<()> {
        let state = self
            .vision
            .as_ref()
            .and_then(|v| v.video_state())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "off its encoder has not said whether it reads video".into());
        if !state.starts_with("on") {
            let rest: String = state.chars().skip(4).collect();
            return Err(refuse_images(&format!("this server cannot read video: {}", py::strip(&rest))));
        }
        Ok(())
    }

    /// prepare: (ids, thinking, max_new); an unset or non-positive max_new means the rest of the context.
    pub fn prepare_full(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        kwargs: &Obj,
        max_new: Option<i64>,
        force: Option<&str>,
        image_access: Option<&ImageSources>,
    ) -> AResult<Prepared> {
        let kwargs = self.thinking_kwargs(kwargs)?;
        let access = image_access.cloned().unwrap_or_else(|| self.settings().image_sources);
        let mut emb: Option<String> = None;
        self.embeddings.lock().unwrap().remove(&thread_ident());
        let r = self.prepare_inner(messages, tools, &kwargs, max_new, force, &access, &mut emb);
        match r {
            Ok(mut p) => {
                if let Some(e) = &emb {
                    self.embeddings.lock().unwrap().insert(thread_ident(), e.clone());
                }
                p.embeddings = emb;
                Ok(p)
            }
            Err(e) => {
                if let Some(path) = emb {
                    let _ = std::fs::remove_file(path);
                }
                Err(e)
            }
        }
    }

    fn known_ctx_or_starting(&self) -> AResult<i64> {
        let mut ctx = self.engine.max_context();
        if ctx <= 0 {
            let starting = || {
                ApiError::Engine(EngineErr::Starting("the engine is starting (a minute or two); try again shortly".into()))
            };
            if self.engine.starting() {
                return Err(starting());
            }
            ctx = self.engine.known_ctx();
            if ctx <= 0 {
                return Err(starting());
            }
        }
        Ok(ctx)
    }

    fn prepare_text(&self, messages: &[Value], tools: Option<&Value>, kwargs: &Obj, max_new: Option<i64>, force: Option<&str>) -> AResult<Prepared> {
        let (text, plain) = self.render_text(messages, tools, kwargs)?;
        let thinking_off = kwargs.get("enable_thinking") == Some(&Value::Bool(false));
        let tail = if thinking_off { force.filter(|f| !f.is_empty()).unwrap_or("") } else { "" };
        let ctx = self.known_ctx_or_starting()?;
        let unset = max_new.map(|m| m <= 0).unwrap_or(true);
        let max_new = if unset { ctx - CTX_SLACK } else { max_new.unwrap() };
        let prompt = PromptText::new(self.tok.clone(), format!("{text}{tail}"), plain, tail, unset || self.fit_max_tokens);
        // prompt.max_new = max_new (the cap the engine is asked for, before it settles)
        prompt.preset_max_new(max_new);
        Ok(Prepared { ids: Prompt::Text(Arc::new(prompt)), thinking: !thinking_off, max_new, embeddings: None })
    }

    fn check_images(&self, messages: &[Value], images: &[Value]) -> AResult<()> {
        if self.vision.is_none() {
            return Err(refuse_images("this server was started without image input (--vision), so it cannot read images"));
        }
        if !images_of(messages.iter().filter(|m| getv(m, "role") == "system")).is_empty() {
            return Err(refuse_images("a system prompt cannot hold an image"));
        }
        if !images.iter().all(py::truthy) {
            return Err(refuse_images(crate::image_source::USAGE));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_inner(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        kwargs: &Obj,
        max_new: Option<i64>,
        force: Option<&str>,
        access: &ImageSources,
        emb: &mut Option<String>,
    ) -> AResult<Prepared> {
        let media = media_of(messages);
        let images: Vec<Value> =
            media.iter().filter(|i| getv(i, "type") == "image").map(|i| getv(i, "source").clone()).collect();
        let videos: Vec<Value> = media.iter().filter(|i| getv(i, "type") == "video").cloned().collect();
        if !images.is_empty() {
            self.check_images(messages, &images)?;
        }
        if !videos.is_empty() {
            self.check_videos(&videos)?;
        }
        if media.is_empty() && self.engine.takes_text() {
            return self.prepare_text(messages, tools, kwargs, max_new, force);
        }
        let mut results = Vec::new();
        let mut spliced = None;
        if !media.is_empty() {
            results = self.encode_media(&media, &images, !videos.is_empty(), access, emb)?;
            if !videos.is_empty() {
                spliced = Some(
                    results
                        .iter()
                        .filter_map(|r| match r {
                            crate::vision::MediaResult::Video(v) => Some(self.dialect.video_text(&v.times, v.rows as usize)),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut ids = self.encode_prompt_videos(messages, tools, kwargs, spliced.as_deref())?;
        let thinking_off = kwargs.get("enable_thinking") == Some(&Value::Bool(false));
        if let Some(f) = force.filter(|f| !f.is_empty()) {
            if thinking_off {
                ids.extend(self.tok.encode(&format!("{}{}", self.dialect.nothink_lead, f), true, &[]));
            }
        }
        if !images.is_empty() {
            let counts: Vec<i64> = results
                .iter()
                .filter_map(|r| match r {
                    crate::vision::MediaResult::Image(n) => Some(*n),
                    _ => None,
                })
                .collect();
            ids = self.expand_images(&ids, &counts)?;
        }
        self.fit(ids, kwargs, max_new)
    }

    fn encode_media(
        &self,
        media: &[Value],
        images: &[Value],
        has_videos: bool,
        access: &ImageSources,
        emb: &mut Option<String>,
    ) -> AResult<Vec<crate::vision::MediaResult>> {
        let vision = self.vision.as_ref().unwrap();
        if has_videos && vision.in_engine() {
            self.fifo.acquire(None);
            let r = self.ensure_loaded();
            self.fifo.release();
            r?;
        }
        if has_videos {
            self.video_state_check()?;
        }
        let datas = (|| -> Result<Vec<Arc<Vec<u8>>>, crate::image_source::SourceErr> {
            let mut read = if images.is_empty() { vec![] } else { vision.read(images, access)? }.into_iter();
            let mut datas = Vec::new();
            for item in media {
                if getv(item, "type") == "image" {
                    datas.push(read.next().unwrap());
                } else {
                    datas.push(Arc::new(vision.read_video(getv(item, "source"), access)?));
                }
            }
            Ok(datas)
        })();
        let datas = match datas {
            Ok(d) => d,
            Err(e) => {
                if let Some(l) = &e.log_line {
                    out(&format!("[server] error: {l}"));
                }
                return Err(ApiError::value(e.message));
            }
        };
        let video_bytes: usize =
            media.iter().zip(&datas).filter(|(i, _)| getv(i, "type") == "video").map(|(_, d)| d.len()).sum();
        if video_bytes as u64 > VIDEO_REQUEST_BYTES {
            return Err(refuse_images(&format!(
                "the request's videos are larger than {} MiB",
                VIDEO_REQUEST_BYTES >> 20
            )));
        }
        let mut seen: Vec<*const Vec<u8>> = Vec::new();
        let mut held = 0u64;
        for d in &datas {
            let p = Arc::as_ptr(d);
            if !seen.contains(&p) {
                seen.push(p);
                held += d.len() as u64;
            }
        }
        {
            let mut w = self.images_waiting.lock().unwrap();
            if *w + held > IMAGE_WAITING_BYTES {
                return Err(ApiError::Engine(EngineErr::ModelBusy(
                    "too many images are waiting to be encoded; try again shortly".into(),
                )));
            }
            *w += held;
        }
        let r = (|| -> AResult<Vec<crate::vision::MediaResult>> {
            self.fifo.acquire(None);
            let r = (|| -> AResult<Vec<crate::vision::MediaResult>> {
                if vision.in_engine() {
                    self.ensure_loaded()?;
                }
                let jobs: Vec<crate::vision::Job> = media
                    .iter()
                    .zip(&datas)
                    .map(|(item, d)| {
                        if getv(item, "type") == "image" {
                            crate::vision::Job::Image(d.clone())
                        } else {
                            let mut opts = Obj::new();
                            opts.insert("max_tokens".into(), Value::from(self.video_max_tokens));
                            for k in ["fps", "max_frames"] {
                                if let Some(v) = get(item, k) {
                                    opts.insert(k.into(), v.clone());
                                }
                            }
                            crate::vision::Job::Video(d.clone(), opts)
                        }
                    })
                    .collect();
                match vision.encode_media(&jobs) {
                    Ok((path, results)) => {
                        *emb = Some(path);
                        Ok(results)
                    }
                    Err(e) => {
                        out(&format!("[server] error: {e}"));
                        Err(ApiError::value(e))
                    }
                }
            })();
            self.fifo.release();
            r
        })();
        *self.images_waiting.lock().unwrap() -= held;
        r
    }

    fn expand_images(&self, ids: &[u32], counts: &[i64]) -> AResult<Vec<u32>> {
        let pad = self.tok.encode(&self.dialect.image_pad, true, &[])[0];
        let start = if self.dialect.image_start.is_empty() {
            None
        } else {
            Some(self.tok.encode(&self.dialect.image_start, true, &[])[0])
        };
        let literal = self.tok.encode(&self.dialect.image_pad, false, &[]);
        let mut out = Vec::new();
        let mut k = 0;
        for (j, &t) in ids.iter().enumerate() {
            if t == pad && start.map(|s| j > 0 && ids[j - 1] == s).unwrap_or(true) {
                if k == counts.len() {
                    return Err(refuse_images("the prompt and its images do not match"));
                }
                out.extend(std::iter::repeat_n(pad, counts[k].max(0) as usize));
                k += 1;
            } else if t == pad {
                out.extend(&literal);
            } else {
                out.push(t);
            }
        }
        if k != counts.len() {
            return Err(refuse_images("the prompt and its images do not match"));
        }
        Ok(out)
    }

    fn fit(&self, ids: Vec<u32>, kwargs: &Obj, max_new: Option<i64>) -> AResult<Prepared> {
        let ctx = self.known_ctx_or_starting()?;
        let n = ids.len() as i64;
        let room = ctx - CTX_SLACK - n;
        let mut max_new = max_new.unwrap_or(0);
        if max_new <= 0 || (self.fit_max_tokens && room < 1) {
            if room < 1 {
                return Err(ApiError::value(room_error(n, ctx)));
            }
            max_new = room;
        } else if max_new > room {
            if !self.fit_max_tokens {
                return Err(ApiError::value(overflow_error(n, max_new, ctx)));
            }
            max_new = room.max(1);
        }
        let thinking = kwargs.get("enable_thinking") != Some(&Value::Bool(false));
        Ok(Prepared { ids: Prompt::Ids(ids), thinking, max_new, embeddings: None })
    }
}

fn run_hook(cmd: &Value) -> Result<(), String> {
    let mut c = match cmd {
        Value::String(s) => {
            let mut c = if cfg!(windows) { std::process::Command::new("cmd") } else { std::process::Command::new("sh") };
            if cfg!(windows) {
                c.args(["/C", s]);
            } else {
                c.args(["-c", s]);
            }
            c
        }
        Value::Array(a) if !a.is_empty() => {
            let mut c = std::process::Command::new(py::py_str(&a[0]));
            c.args(a[1..].iter().map(py::py_str));
            c
        }
        _ => return Err("bad command".into()),
    };
    c.stdin(std::process::Stdio::null());
    let mut child = c.spawn().map_err(|e| e.to_string())?;
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    loop {
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            return Err(format!("Command '{}' timed out after 120 seconds", py::py_str(cmd)));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// MemAvailable from /proc/meminfo in MiB; None where there is no such file.
pub fn available_ram_mib() -> Option<i64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            return rest.split_whitespace().next()?.parse::<i64>().ok().map(|k| k >> 10);
        }
    }
    None
}

/// An int-valued float as Python prints a number from the config (`{x}`): 30 for an int, 30.0 for a float.
fn py_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        serve_text::pyjson::py_float_repr(x)
    }
}

/// _refuse_images: the client gets the 400, the log says why.
pub fn refuse_images(why: &str) -> ApiError {
    out(&format!("[server] error: {why}"));
    ApiError::value(why)
}

pub fn slot_tag(slot: Option<i64>) -> String {
    match slot {
        None => "s?".into(),
        Some(s) => format!("s{}", s + 1),
    }
}

/// prompt_tokens_seen: all, unless the DONE line shows a cancel mid-read.
pub fn prompt_tokens_seen(prompt_tokens: i64, last: &Obj) -> i64 {
    let read = some(last.get("prompt_read"));
    if read.is_none() || last.get("finish").and_then(|f| f.as_str()) != Some("cancel") {
        return prompt_tokens;
    }
    prompt_tokens.min(int(last.get("reused")) + int(read))
}

/// spill_of: the experts only the file holds, from the engine's INFO words.
pub fn spill_of(info: &mut Obj) -> Option<Value> {
    let mut words = Obj::new();
    for k in ["experts", "total", "bytes"] {
        words.insert(k.into(), info.shift_remove(&format!("spill_{k}")).unwrap_or(Value::Null));
    }
    if py::as_int(&words["experts"]).map(|e| e > 0).unwrap_or(false) {
        Some(Value::Object(words))
    } else {
        None
    }
}

/// request_timings: llama.cpp's `timings` from the engine's DONE figures; None without an engine clock.
pub fn request_timings(prompt_tokens: i64, generated: i64, last: &Obj) -> Option<Obj> {
    let pm = fl(some(last.get("prompt_ms")))?;
    let cache_n = int(last.get("reused"));
    let prompt_n = (prompt_tokens - cache_n).max(0);
    let decode_ms = fl(last.get("decode_ms")).unwrap_or(0.0);
    let g = int(last.get("generated"));
    let decoded = if g != 0 { g } else { generated };
    let mut t = Obj::new();
    t.insert("cache_n".into(), Value::from(cache_n));
    t.insert("prompt_n".into(), Value::from(prompt_n));
    t.insert("prompt_ms".into(), fv(py_round(pm, 1)));
    t.insert(
        "prompt_per_token_ms".into(),
        if prompt_n != 0 { fv(py_round(pm / prompt_n as f64, 3)) } else { Value::Null },
    );
    t.insert(
        "prompt_per_second".into(),
        if prompt_n != 0 && pm > 0.0 { fv(py_round(prompt_n as f64 / (pm / 1000.0), 1)) } else { Value::Null },
    );
    t.insert("predicted_n".into(), Value::from(generated));
    t.insert("predicted_ms".into(), fv(py_round(decode_ms, 1)));
    t.insert(
        "predicted_per_token_ms".into(),
        if decoded != 0 { fv(py_round(decode_ms / decoded as f64, 3)) } else { Value::Null },
    );
    t.insert(
        "predicted_per_second".into(),
        if decoded != 0 && decode_ms > 0.0 { fv(py_round(decoded as f64 / (decode_ms / 1000.0), 1)) } else { Value::Null },
    );
    if some(last.get("drafts_offered")).is_some() {
        t.insert("draft_n".into(), Value::from(int(last.get("drafts_offered"))));
        t.insert("draft_n_accepted".into(), Value::from(int(last.get("drafts_accepted"))));
    }
    Some(t)
}

#[allow(clippy::too_many_arguments)]
pub fn done_line(
    slot: Option<i64>,
    prompt_total: i64,
    generated: i64,
    finish: &str,
    cancelled: bool,
    timings: Option<&Obj>,
    ttft: Option<f64>,
    wall_rate: f64,
    dl: Option<f64>,
    hit: Option<f64>,
    pcie: Option<f64>,
    ssd: Option<f64>,
) -> String {
    let empty = Obj::new();
    let t = timings.unwrap_or(&empty);
    let mut pp = format!("pp={prompt_total}/{}c", int(t.get("cache_n")));
    if let Some(pm) = fl(some(t.get("prompt_ms"))) {
        pp += &format!(" {}s", py_fixed(pm / 1000.0, 1));
        if int(t.get("prompt_n")) >= MIN_PP_RATE_TOKENS {
            if let Some(r) = fl(t.get("prompt_per_second")).filter(|r| *r != 0.0) {
                pp += &format!(" {}t/s", py_fixed(r, 0));
            }
        }
    }
    let rate = fl(t.get("predicted_per_second"))
        .filter(|r| *r != 0.0)
        .unwrap_or(if generated >= MIN_RATE_TOKENS { wall_rate } else { 0.0 });
    let mut tg = format!("tg={generated}");
    if generated != 0 && rate != 0.0 {
        tg += &format!(" {}t/s", py_fixed(rate, 1));
    }
    if int(t.get("draft_n")) != 0 {
        tg += &format!(
            " acc={}%",
            py_fixed(int(t.get("draft_n_accepted")) as f64 / int(t.get("draft_n")) as f64 * 100.0, 0)
        );
        if let Some(d) = dl {
            tg += &format!(" dl={}", py_fixed(d, 1));
        }
    }
    let mut groups = vec![pp, tg];
    if let Some(tt) = ttft {
        groups.push(format!("ttft={}s", py_fixed(tt, 1)));
    }
    groups.push(format!("end={}", if cancelled || finish == "cancel" { "cancelled" } else { finish }));
    if let Some(h) = hit {
        groups.push(format!("hit={}%", py_fixed(h * 100.0, 1)));
        if let Some(p) = pcie {
            groups.push(format!("pcie={}%", py_fixed(p * 100.0, 1)));
        }
        if let Some(s) = ssd.filter(|s| *s != 0.0) {
            groups.push(format!("ssd={}%", py_fixed(s * 100.0, 1)));
        }
    }
    format!("[server] {} done  {}", slot_tag(slot), groups.join("  "))
}

/// _debug_req: --debug, one compact line per request.
#[allow(clippy::too_many_arguments)]
pub fn debug_req(api: &str, req: &Obj, messages: &[Value], tools: Option<&Value>, max_new: i64, thinking: bool, prompt_tokens: usize) {
    if !DEBUG.load(Ordering::Relaxed) {
        return;
    }
    let empty = Value::Object(Map::new());
    let last = messages.last().unwrap_or(&empty);
    let body = getv(last, "content");
    let body = match body {
        Value::Array(a) => Value::String(
            a.iter().filter(|p| p.is_object()).map(|p| get(p, "text").map(py::py_str).unwrap_or_default()).collect::<Vec<_>>().join(" "),
        ),
        other => other.clone(),
    };
    let s = if py::truthy(&body) { py::py_str(&body) } else { String::new() };
    let preview = py::head(&s, 80).replace('\n', " ");
    let ntools = tools.and_then(|t| t.as_array()).map(|a| a.len()).unwrap_or(0);
    let r = |k: &str| py_repr(req.get(k).unwrap_or(&Value::Null));
    out(&format!(
        "[server] req {api}: msgs={} tools={ntools} max_tokens_raw={}/{} max_new={max_new} thinking={} stream={} prompt_tokens={prompt_tokens} last={}:{}",
        messages.len(),
        r("max_tokens"),
        r("max_completion_tokens"),
        if thinking { "True" } else { "False" },
        if req.get("stream").map(py::truthy).unwrap_or(false) { "True" } else { "False" },
        py_repr(getv(last, "role")),
        serve_text::pyjson::py_str_repr(&preview)
    ));
}

pub use crate::service_run::RunArgs;

pub use crate::service_api::close_inner;

impl Service {
    pub(crate) fn embeddings_map(&self) -> &Mutex<std::collections::HashMap<u64, String>> {
        &self.embeddings
    }
    pub(crate) fn request_path_map(&self) -> &Mutex<std::collections::HashMap<u64, String>> {
        &self.request_path
    }
    pub(crate) fn live_at_lock(&self) -> std::sync::MutexGuard<'_, f64> {
        self.live_at.lock().unwrap()
    }
}
