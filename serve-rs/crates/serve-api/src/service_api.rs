//! The Service surface the HTTP layer (serve-http) calls: Python's attribute reads and the call shapes handler.py uses.

use crate::error::AResult;
use crate::image_source::ImageSources;
use crate::models::Dialect;
use crate::py::Obj;
use crate::service::{AnswerRoom, RunArgs, RunThread, Service};
use serde_json::{Map, Value};
use serve_engine::errors::EngineErr;
use serve_engine::gen::{Gen, Stop};
use serve_engine::iface::{thread_ident, Flag, Prompt, PyLock, Sampling};
use serve_engine::util::telemetry::Telemetry;
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn map(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

/// A Python generator's `for x in inner:` ended by the consumer or an error: the inner run is closed (its cleanup
/// error, as CPython's collection of it, is not raised), the outer outcome stands.
pub fn close_inner<T: Send + 'static>(mut gen: Gen<T>, r: Result<(), Stop>) -> Result<(), Stop> {
    let _ = gen.close();
    r
}

impl Service {
    pub fn sampling_defaults(&self) -> Obj {
        self.sampling_defaults.clone()
    }
    pub fn template_source(&self) -> String {
        self.template.source.clone()
    }
    pub fn dialect(&self) -> &Dialect {
        &self.dialect
    }
    pub fn dialect_caps(&self) -> Map<String, Value> {
        self.dialect.caps.as_object().cloned().unwrap_or_default()
    }
    pub fn forced_opening(&self, name: Option<&str>) -> String {
        self.dialect.forced_opening(name)
    }
    pub fn anthropic_think_unasked(&self) -> bool {
        self.cfg.read().unwrap().anthropic_think_unasked
    }
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }
    pub fn fifo(&self) -> &PyLock {
        &self.fifo
    }
    /// status busy or queued, under status_lock
    pub fn busy_or_queued(&self) -> bool {
        let s = self.status_lock.lock().unwrap();
        s.status.get("busy").map(crate::py::truthy).unwrap_or(false)
            || s.status.get("queued").map(crate::py::truthy).unwrap_or(false)
    }
    pub fn touch_last_request(&self) {
        self.set_last_request_at(crate::service::now());
    }
    /// The hosted tools not noted yet, sorted; they are noted.
    pub fn note_hosted_tools(&self, skipped: &[String]) -> Vec<String> {
        let mut noted = self.hosted_tools_noted.lock().unwrap();
        let mut fresh: Vec<String> = skipped.iter().filter(|s| !noted.contains(*s)).cloned().collect();
        fresh.sort();
        fresh.dedup();
        if !fresh.is_empty() {
            noted.extend(skipped.iter().cloned());
        }
        fresh
    }

    pub fn reasoning_budget(&self, req: &Obj) -> AResult<Option<i64>> {
        self.reasoning_budget_of(Some(req), true)
    }

    /// prepare: (ids, thinking, max_new) as handler.py unpacks it; max_new 0 means the rest of the context.
    pub fn prepare(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        kwargs: &Obj,
        max_new: i64,
        force: Option<&str>,
        image_access: Option<ImageSources>,
    ) -> AResult<(Prompt, bool, i64)> {
        let p = self.prepare_full(messages, tools, kwargs, Some(max_new), force, image_access.as_ref())?;
        Ok((p.ids, p.thinking, p.max_new))
    }

    pub fn encode_prompt(&self, messages: &[Value], tools: Option<&Value>, kwargs: &Obj) -> AResult<Vec<u32>> {
        self.encode_prompt_videos(messages, tools, kwargs, None)
    }

    /// The API path of the request this thread serves (handler.py's request_ctx.path).
    pub fn set_request_path(&self, path: Option<&str>) {
        let mut m = self.request_path_map().lock().unwrap();
        match path {
            Some(p) => {
                m.insert(thread_ident(), p.to_string());
            }
            None => {
                m.remove(&thread_ident());
            }
        }
    }

    /// RunArgs for run(), with this thread's embeddings file and API path taken (call on the handler's thread).
    pub fn run_args(&self, ids: Prompt, thinking: bool, tools: Option<Value>, max_new: i64, sampling: Sampling, cancel: Flag) -> RunArgs {
        let tid = thread_ident();
        RunArgs {
            ids,
            thinking,
            tools,
            max_new,
            sampling,
            cancel,
            force: None,
            session: None,
            live: None,
            path: self.request_path_map().lock().unwrap().get(&tid).cloned(),
            model: Some(self.model.clone()),
            embeddings: self.embeddings_map().lock().unwrap().remove(&tid),
            thread: RunThread::default(),
        }
    }

    pub fn unload(&self) -> Result<String, EngineErr> {
        self.unload_idle(None).map(String::from)
    }
    pub fn sleep_now(&self) -> Result<String, EngineErr> {
        self.sleep(None).map(String::from)
    }
    pub fn model_entry(&self) -> Map<String, Value> {
        map(self.model_entry_v())
    }
    pub fn metrics(&self) -> Map<String, Value> {
        map(self.metrics_v())
    }
    pub fn telemetry_extra(&self) -> Map<String, Value> {
        map(self.telemetry_extra_v())
    }
    pub fn v1_status(&self) -> Map<String, Value> {
        map(self.v1_status_v())
    }
    pub fn slot_stats(&self) -> Map<String, Value> {
        map(self.slot_stats_v())
    }
    pub fn status_view(&self) -> Map<String, Value> {
        map(self.status_view_v())
    }
    pub fn stats(&self, n: usize, facts: &[Map<String, Value>]) -> Map<String, Value> {
        map(self.stats_v(n, facts))
    }
    pub fn slot_table(&self) -> (Vec<Value>, Vec<Value>) {
        let (a, b) = self.slot_table_v();
        (a.into_iter().map(Value::Object).collect(), b)
    }
    pub fn load_progress(&self) -> Option<Map<String, Value>> {
        self.load_progress_v().map(map)
    }

    pub fn set_vram_reserve(&self, mib: Option<i64>) {
        *self.vram_reserve.lock().unwrap() = mib;
    }
    pub fn telemetry(&self) -> Option<Arc<Telemetry>> {
        self.telemetry.read().unwrap().clone()
    }
    pub fn set_telemetry(&self, t: Option<Arc<Telemetry>>) {
        *self.telemetry.write().unwrap() = t;
    }
    pub fn gpu_indices(&self) -> Vec<i64> {
        self.cfg.read().unwrap().gpu_indices.iter().map(|x| *x as i64).collect()
    }
    pub fn set_load_kind(&self, kind: &str) {
        *self.load_kind.lock().unwrap() = match kind {
            "wake" => "wake",
            "swap" => "swap",
            _ => "weights",
        };
    }
    pub fn set_loading(&self, v: bool) {
        self.loading_flag.store(v, Ordering::SeqCst);
    }
    pub fn loading(&self) -> bool {
        self.is_loading()
    }
    pub fn boot_error(&self) -> Option<EngineErr> {
        self.boot_error.lock().unwrap().clone()
    }

    // ------------------------------------------------------------------------------------ settings server.py sets
    pub fn set_idle_unload_s(&self, v: f64) {
        self.cfg.write().unwrap().idle_unload_s = v;
    }
    pub fn set_idle_sleep_s(&self, v: f64) {
        self.cfg.write().unwrap().idle_sleep_s = v;
    }
    pub fn set_min_free_vram_mib(&self, v: i64) {
        self.cfg.write().unwrap().min_free_vram_mib = v;
    }
    pub fn set_reasoning_budget_tokens(&self, v: i64) {
        self.cfg.write().unwrap().reasoning_budget_tokens = v;
    }
    /// "auto" or a token count
    pub fn set_answer_room(&self, v: &Value) {
        self.cfg.write().unwrap().answer_room = match v.as_i64() {
            Some(n) => AnswerRoom::Fixed(n),
            None => AnswerRoom::Auto,
        };
    }
    pub fn set_reasoning_effort(&self, v: Option<Value>) {
        self.cfg.write().unwrap().reasoning_effort = v.filter(|x| !x.is_null());
    }
    pub fn set_effort_end(&self, v: bool) {
        self.cfg.write().unwrap().effort_end = v;
    }
    pub fn set_bang_restart_tokens(&self, v: i64) {
        self.cfg.write().unwrap().bang_restart_tokens = v;
    }

    pub fn set_stop_token_text(&self, v: bool) {
        self.cfg.write().unwrap().stop_token_text = v;
    }

    pub fn set_thinking_loop_tokens(&self, v: i64) {
        self.cfg.write().unwrap().thinking_loop_tokens = v;
    }
    pub fn set_anthropic_think_unasked(&self, v: bool) {
        self.cfg.write().unwrap().anthropic_think_unasked = v;
    }
    pub fn set_gpu_indices(&self, v: Vec<i64>) {
        let mut c = self.cfg.write().unwrap();
        c.gpu_indices = v.iter().map(|x| *x as u32).collect();
        c.gpu_index = c.gpu_indices.first().copied().unwrap_or(0);
    }
    pub fn set_start_tries(&self, v: Option<u32>) {
        self.cfg.write().unwrap().start_tries = v;
    }
    pub fn set_before_load(&self, v: Option<Value>) {
        self.cfg.write().unwrap().before_load = v.filter(|x| !x.is_null());
    }
    pub fn set_image_sources(&self, v: ImageSources) {
        self.cfg.write().unwrap().image_sources = v;
    }
    /// on_loaded, chained after any earlier one (server.py's open_when_ready wraps the previous hook).
    pub fn set_on_loaded(&self, f: Box<dyn Fn() + Send + Sync>) {
        let prev = self.on_loaded.read().unwrap().clone();
        let f: Arc<dyn Fn() + Send + Sync> = Arc::from(f);
        let chained: Arc<dyn Fn() + Send + Sync> = match prev {
            Some(p) => Arc::new(move || {
                p();
                f();
            }),
            None => f,
        };
        *self.on_loaded.write().unwrap() = Some(chained);
    }
}
