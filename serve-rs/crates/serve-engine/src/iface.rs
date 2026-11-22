//! The engine surface Service uses: ProcessEngine's and MockEngine's attributes and methods, names 1:1 with Python.
//! Python's `hasattr(engine, x)` / `getattr(engine, x, default)` become `has_*()` flags and defaulted methods, so
//! MockEngine leaves out what it does not have.
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Map, Value};

use crate::console::ConsoleLines;
use crate::errors::{EResult, EngineErr};
use crate::gen::Gen;
pub use crate::live::Flag;
pub use crate::logprobs::{LogProb, Tok};

pub type JsonMap = Map<String, Value>;

/// What generate() reads: token ids, or a text the engine tokenizes (serve/prompt.py PromptText).
#[derive(Clone)]
pub enum Prompt {
    Ids(Vec<u32>),
    Text(Arc<serve_text::prompt::PromptText>),
}

impl Prompt {
    /// len(prompt): the count once known, else PromptText's estimate.
    pub fn len(&self) -> usize {
        match self {
            Prompt::Ids(v) => v.len(),
            Prompt::Text(t) => t.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A request's sampling: the dict Service builds (temperature, top_p, ..., `_grammar`, `_logprobs`) and the
/// server-normalized logit bias (Python's int-keyed `_logit_bias`, which a JSON map cannot carry).
#[derive(Debug, Clone, Default)]
pub struct Sampling {
    pub map: JsonMap,
    pub logit_bias: Option<Vec<(u32, f64)>>,
}

/// The tokens of a generate(): Some(tok) per token, None as a heartbeat; an Err item is the exception.
pub type GenStream = Gen<Option<Tok>>;

/// `engine.last`: the figures of the last DONE line. `id` changes whenever the dict is replaced (Python's identity
/// test `engine.last is not engine_last0`).
#[derive(Debug, Clone, Default)]
pub struct Last {
    pub id: u64,
    pub map: JsonMap,
}

/// A batch slot's draft counts from its last BDONE: (accepted, offered, time).
pub type SlotDrafts = BTreeMap<usize, (Option<i64>, Option<i64>, f64)>;

/// A lock that one thread takes and any thread may release (Python threading.Lock).
#[derive(Debug, Default)]
pub struct PyLock {
    held: Mutex<bool>,
    cv: Condvar,
}

impl PyLock {
    pub fn new() -> Self {
        Self::default()
    }
    /// acquire(timeout=None|t); acquire(blocking=False) is `Some(Duration::ZERO)`.
    pub fn acquire(&self, timeout: Option<Duration>) -> bool {
        let mut g = self.held.lock().unwrap();
        let end = timeout.map(|t| Instant::now() + t);
        while *g {
            match end {
                None => g = self.cv.wait(g).unwrap(),
                Some(e) => {
                    let now = Instant::now();
                    if now >= e {
                        return false;
                    }
                    g = self.cv.wait_timeout(g, e - now).unwrap().0;
                }
            }
        }
        *g = true;
        true
    }
    pub fn release(&self) {
        *self.held.lock().unwrap() = false;
        self.cv.notify_one();
    }
    pub fn locked(&self) -> bool {
        *self.held.lock().unwrap()
    }
}

/// `ram_check()` of sleep(): why the rest would not fit in RAM (None: it fits).
pub type RamCheck<'a> = &'a (dyn Fn() -> Option<String> + Sync);
/// `vram_check(parked)` of wake(): why the VRAM the state needs is not free (None: it is).
pub type VramCheck<'a> = &'a (dyn Fn(Option<&BTreeMap<u32, u64>>) -> Option<String> + Sync);

/// The engine. Defaults are what Python's getattr defaults give for an engine without the attribute (MockEngine).
#[allow(unused_variables)]
pub trait Engine: Send + Sync {
    fn as_any(&self) -> &dyn Any;

    // ---- generation
    /// Yield token ids, and None as a 10 s heartbeat while the engine is quiet.
    fn generate(&self, prompt: Prompt, max_new: i64, sampling: &Sampling, cancel: Flag, embeddings: Option<String>,
                session: Option<String>, stop_soon: Option<Flag>) -> GenStream;
    /// `"stop_soon" in inspect.signature(engine.generate).parameters`
    fn supports_stop_soon(&self) -> bool {
        false
    }
    /// takes_text: generate() accepts a PromptText
    fn takes_text(&self) -> bool {
        false
    }

    // ---- facts
    fn max_context(&self) -> i64;
    /// `last` as the thread `tid` (the one that called generate) sees it
    fn last_for(&self, tid: u64) -> Option<Last> {
        let _ = tid;
        self.last()
    }
    fn prompt_progress_for(&self, tid: u64) -> Option<JsonMap> {
        let _ = tid;
        self.prompt_progress()
    }
    fn window_stats_for(&self, tid: u64) -> Option<(u64, u64)> {
        let _ = tid;
        self.window_stats()
    }
    fn known_ctx(&self) -> i64 {
        0
    }
    fn starting(&self) -> bool {
        false
    }
    fn batch(&self) -> usize {
        0
    }
    fn gen(&self) -> u64 {
        0
    }
    /// `getattr(engine, "last", None)`
    fn last(&self) -> Option<Last> {
        None
    }
    /// `engine.progress`: (read, total) prompt tokens
    fn progress(&self) -> Option<(i64, i64)> {
        None
    }
    /// `engine.prompt_progress()`; None when the engine has no such method
    fn prompt_progress(&self) -> Option<JsonMap> {
        None
    }
    fn prefill_tok_s_mean(&self) -> Option<f64> {
        None
    }
    /// `getattr(engine, "info", None)`
    fn info(&self) -> Option<JsonMap> {
        None
    }
    fn eos_ids(&self) -> Vec<u32> {
        Vec::new()
    }
    fn alive(&self) -> bool {
        true
    }
    fn has_alive(&self) -> bool {
        false
    }
    fn asleep(&self) -> bool {
        false
    }
    fn unloaded(&self) -> bool {
        false
    }
    fn loading(&self) -> bool {
        false
    }
    fn load_status(&self) -> Option<JsonMap> {
        None
    }
    fn set_load_status(&self, status: JsonMap) {}
    fn model_path(&self) -> Option<String> {
        None
    }
    fn log_path(&self) -> Option<String> {
        None
    }
    /// the engine's command-line words (ProcessEngine.spawn[1])
    fn spawn_args(&self) -> Vec<String> {
        Vec::new()
    }
    fn conversation(&self) -> JsonMap {
        JsonMap::new()
    }
    /// (windows, tokens) of the run in progress on this thread; None when the engine has no window_stats
    fn window_stats(&self) -> Option<(u64, u64)> {
        None
    }
    /// per batch slot: {"owner": thread or null, "held": n}; None when the engine has no slot_view
    fn slot_view(&self) -> Option<Vec<JsonMap>> {
        None
    }
    fn slot_drafts(&self) -> SlotDrafts {
        SlotDrafts::new()
    }
    fn console(&self) -> Option<Arc<ConsoleLines>> {
        None
    }
    /// requests waiting for the engine or a free slot; None when the engine has no queue_depth
    fn queue_depth(&self) -> Option<usize> {
        None
    }
    /// the control lines of a batch engine (`engine.ctl`)
    fn ctl(&self) -> Option<&PyLock> {
        None
    }
    fn pid(&self) -> Option<u32> {
        None
    }

    // ---- process control (ProcessEngine only)
    fn has_process(&self) -> bool {
        false
    }
    fn death_note(&self) -> String {
        String::new()
    }
    fn exit_code(&self) -> Option<i64> {
        None
    }
    fn restart(&self, tries: Option<u32>) -> EResult<()> {
        Err(EngineErr::died("the engine cannot be restarted"))
    }
    fn unload(&self) -> EResult<()> {
        Ok(())
    }
    fn close(&self) -> EResult<()> {
        Ok(())
    }
    fn kill(&self) {}
    fn can_sleep(&self) -> bool {
        false
    }
    fn sleep(&self, ram_check: Option<RamCheck>) -> EResult<JsonMap> {
        Err(EngineErr::value("the engine cannot sleep"))
    }
    fn wake(&self, vram_check: Option<VramCheck>) -> EResult<JsonMap> {
        Err(EngineErr::died("the engine cannot wake"))
    }
    fn restore_s(&self) -> f64 {
        0.0
    }
    fn parked(&self) -> Option<BTreeMap<u32, u64>> {
        None
    }
    fn vram(&self, reserve_mib: Option<i64>) -> EResult<JsonMap> {
        Err(EngineErr::died("the engine is not running"))
    }
    fn encode_images(&self, jobs: &[(String, String)], timeout: Option<f64>) -> EResult<Vec<String>> {
        Err(EngineErr::died("the engine is not running"))
    }
    fn encode_video(&self, video: &str, out: &str, words: &str, timeout: Option<f64>) -> EResult<String> {
        Err(EngineErr::died("the engine is not running"))
    }
    /// the in-engine encoder's VIDEO line ("on ..." / "off <reason>")
    fn video(&self) -> Option<String> {
        None
    }
}

/// `threading.get_ident()`: a number for the calling thread, stable for its life.
pub fn thread_ident() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static ID: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    ID.with(|v| *v)
}
