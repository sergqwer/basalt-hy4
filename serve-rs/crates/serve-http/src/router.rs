//! serve/router.py: one Service per configured model, one model loaded at a time, requests picking theirs by name.
//! Turn rules as in the Python module docstring.
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use serve_api::service::Service;
use serve_engine::errors::EngineErr;
use serve_engine::live::LiveRequests;
use serve_engine::util::stdout;
use serve_engine::util::telemetry::Telemetry;

pub type Svc = Arc<Service>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    Waiting,
    Granted,
    Failed,
    Left,
}

struct TurnInner {
    svc: Svc,
    holds: bool,
    state: TurnState,
    error: Option<EngineErr>,
    at: f64,
}

/// A request's place.
#[derive(Clone)]
pub struct Turn(Arc<Mutex<TurnInner>>);

impl Turn {
    fn new(svc: Svc, holds: bool) -> Turn {
        Turn(Arc::new(Mutex::new(TurnInner { svc, holds, state: TurnState::Waiting, error: None, at: now() })))
    }
    fn st(&self) -> TurnState {
        self.0.lock().unwrap().state
    }
    fn svc(&self) -> Svc {
        self.0.lock().unwrap().svc.clone()
    }
    fn same(&self, o: &Turn) -> bool {
        Arc::ptr_eq(&self.0, &o.0)
    }
}

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// The error a waiting request gets for a failed load, naming the model.
pub fn load_failure(name: &str, e: &EngineErr) -> EngineErr {
    let text = format!("{name} did not load: {}", e.message());
    match e {
        EngineErr::StartError { log, .. } => EngineErr::StartError { msg: text, log: log.clone() },
        EngineErr::GpuBusy(_) => EngineErr::GpuBusy(text),
        EngineErr::Stuck(_) => EngineErr::Stuck(text),
        _ => EngineErr::Died(text),
    }
}

struct Swap {
    svc: Svc,
    from: Option<String>,
    phase: &'static str,
    since: f64,
}

struct State {
    current: Option<Svc>,
    queue: VecDeque<Turn>,
    running: usize,
    swap: Option<Swap>,
    swaps: u64,
    errors: HashMap<usize, String>,
    stopping: bool,
    worker: bool,
}

pub struct Router {
    pub services: Vec<Svc>,
    pub front: Svc,
    st: Mutex<State>,
    cv: Condvar,
    pub live: LiveRequests,
}

fn key(s: &Svc) -> usize {
    Arc::as_ptr(s) as usize
}

impl Router {
    pub fn new(services: Vec<Svc>) -> Arc<Router> {
        let front = services[0].clone();
        let current = if services.len() == 1 { Some(front.clone()) } else { None };
        Arc::new(Router {
            services,
            front,
            st: Mutex::new(State { current, queue: VecDeque::new(), running: 0, swap: None, swaps: 0,
                                   errors: HashMap::new(), stopping: false, worker: false }),
            cv: Condvar::new(),
            live: LiveRequests::new(),
        })
    }

    pub fn multi(&self) -> bool {
        self.services.len() > 1
    }

    pub fn stopping(&self) -> bool {
        self.st.lock().unwrap().stopping
    }

    /// The service answering to `name`: a listed name or a model file's name.
    pub fn service_for(&self, name: Option<&Value>) -> Option<Svc> {
        let Some(Value::String(name)) = name else { return None };
        self.services.iter().find(|s| s.served_names().iter().any(|n| n == name)).cloned()
    }

    pub fn service_for_str(&self, name: &str) -> Option<Svc> {
        self.service_for(Some(&Value::String(name.to_string())))
    }

    /// The service the status routes describe: the swap's target, else the loaded model, else the first.
    pub fn shown(&self) -> Svc {
        let st = self.st.lock().unwrap();
        let t = st.swap.as_ref().map(|s| s.svc.clone()).or_else(|| st.current.clone());
        t.unwrap_or_else(|| self.front.clone())
    }

    /// What /v1/models lists.
    pub fn models(&self) -> Vec<Value> {
        let errors = self.st.lock().unwrap().errors.clone();
        let mut out = Vec::new();
        for svc in &self.services {
            let mut model = svc.model_entry();
            if let Some(why) = errors.get(&key(svc)) {
                let unloaded = model.get("status").and_then(|s| s.get("value")).and_then(|v| v.as_str()) == Some("unloaded");
                if !why.is_empty() && unloaded {
                    let mut st = Map::new();
                    st.insert("value".into(), json!("failed"));
                    st.insert("error".into(), json!(why));
                    model.insert("status".into(), Value::Object(st));
                }
            }
            out.push(Value::Object(model));
        }
        out
    }

    pub fn swap_status(&self) -> Value {
        let st = self.st.lock().unwrap();
        let Some(s) = &st.swap else { return Value::Null };
        let mut m = Map::new();
        m.insert("from".into(), s.from.clone().map(Value::from).unwrap_or(Value::Null));
        m.insert("to".into(), json!(s.svc.model));
        m.insert("phase".into(), json!(s.phase));
        m.insert("elapsed_s".into(), json!(serve_text::pyjson::py_round(now() - s.since, 1)));
        m.insert("waiting".into(), json!(st.queue.len()));
        Value::Object(m)
    }

    /// GET /metrics for `svc`, with the requests waiting for any model's turn and every model's row.
    pub fn metrics(&self, svc: &Svc) -> Map<String, Value> {
        let mut m = svc.metrics();
        let t = now();
        let waiting: Vec<Value> = {
            let st = self.st.lock().unwrap();
            st.queue.iter().filter_map(|turn| {
                let i = turn.0.lock().unwrap();
                if !i.holds {
                    return None;
                }
                let mut w = Map::new();
                w.insert("api".into(), Value::Null);
                w.insert("model".into(), json!(i.svc.model));
                w.insert("prompt_tokens".into(), Value::Null);
                w.insert("waited_s".into(), json!(serve_text::pyjson::py_round(t - i.at, 1)));
                Some(Value::Object(w))
            }).collect()
        };
        if let Some(Value::Object(live)) = m.get_mut("live") {
            let q = live.get("queued").and_then(|v| v.as_i64()).unwrap_or(0) + waiting.len() as i64;
            live.insert("queued".into(), json!(q));
            let mut queue = match live.get("queue") {
                Some(Value::Array(a)) => a.clone(),
                _ => vec![],
            };
            queue.extend(waiting);
            live.insert("queue".into(), Value::Array(queue));
        }
        m.insert("models".into(), Value::Array(self.models()));
        m.insert("swap".into(), self.swap_status());
        m
    }

    /// Take a turn for `svc`: granted at once when its model is loaded and nothing waits, else queued.
    pub fn join(self: &Arc<Self>, svc: &Svc) -> Turn {
        let turn = Turn::new(svc.clone(), true);
        let mut st = self.st.lock().unwrap();
        if st.stopping {
            let mut t = turn.0.lock().unwrap();
            t.state = TurnState::Failed;
            t.error = Some(EngineErr::died("the server is stopping"));
        } else if st.current.as_ref().map(|c| Arc::ptr_eq(c, svc)).unwrap_or(false) && st.swap.is_none() && st.queue.is_empty() {
            turn.0.lock().unwrap().state = TurnState::Granted;
            st.running += 1;
        } else {
            st.queue.push_back(turn.clone());
            self.start_worker(&mut st);
            self.cv.notify_all();
        }
        turn
    }

    pub fn preload(self: &Arc<Self>, svc: &Svc) {
        let mut st = self.st.lock().unwrap();
        st.queue.push_back(Turn::new(svc.clone(), false));
        self.start_worker(&mut st);
        self.cv.notify_all();
    }

    /// Whether `turn` is granted after up to `timeout` seconds; Err with the load's error when it failed.
    pub fn wait(&self, turn: &Turn, timeout: Option<f64>) -> Result<bool, EngineErr> {
        let mut st = self.st.lock().unwrap();
        let end = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t));
        while turn.st() == TurnState::Waiting {
            match end {
                None => st = self.cv.wait(st).unwrap(),
                Some(e) => {
                    let n = Instant::now();
                    if n >= e {
                        break;
                    }
                    st = self.cv.wait_timeout(st, e - n).unwrap().0;
                }
            }
        }
        drop(st);
        let t = turn.0.lock().unwrap();
        if t.state == TurnState::Failed {
            return Err(t.error.clone().unwrap_or_else(|| EngineErr::died("")));
        }
        Ok(t.state == TurnState::Granted)
    }

    pub fn leave(&self, turn: &Turn) {
        let mut st = self.st.lock().unwrap();
        let mut t = turn.0.lock().unwrap();
        if t.state == TurnState::Granted && t.holds {
            st.running = st.running.saturating_sub(1);
        } else if t.state == TurnState::Waiting {
            if let Some(i) = st.queue.iter().position(|x| x.same(turn)) {
                st.queue.remove(i);
            }
        }
        t.state = TurnState::Left;
        self.cv.notify_all();
    }

    pub fn stop(&self) {
        {
            let mut st = self.st.lock().unwrap();
            st.stopping = true;
            for turn in st.queue.drain(..) {
                let mut t = turn.0.lock().unwrap();
                t.state = TurnState::Failed;
                t.error = Some(EngineErr::died("the server is stopping"));
            }
            self.cv.notify_all();
        }
        for svc in &self.services {
            svc.stop();
        }
        self.stop_telemetry();
    }

    /// POST /v1/vram: the reserve is the machine's, so a model loaded later applies it too.
    pub fn vram(&self, reserve_mib: Option<i64>) -> Result<Map<String, Value>, EngineErr> {
        let active = self.shown();
        for svc in &self.services {
            if !Arc::ptr_eq(svc, &active) {
                svc.set_vram_reserve(reserve_mib);
            }
        }
        active.vram(reserve_mib)
    }

    /// One hardware sampler over the cards of every model; the speeds are the shown model's.
    pub fn start_telemetry(self: &Arc<Self>) {
        let cur = self.front.telemetry();
        if cur.as_ref().map(|t| !t.stopped()).unwrap_or(false) {
            return;
        }
        let mut cards: Vec<i64> = Vec::new();
        for svc in &self.services {
            let g = svc.gpu_indices();
            let g = if g.is_empty() { vec![0] } else { g };
            for i in g {
                if !cards.contains(&i) {
                    cards.push(i);
                }
            }
        }
        let me = Arc::downgrade(self);
        let extra: serve_engine::util::telemetry::Extra = Box::new(move || me.upgrade().map(|r| r.shown().telemetry_extra()));
        let cards: Vec<u32> = cards.into_iter().map(|c| c.max(0) as u32).collect();
        let sampler = Arc::new(Telemetry::new(Some(extra), Some(&cards), None));
        for svc in &self.services {
            svc.set_telemetry(Some(sampler.clone()));
        }
    }

    pub fn stop_telemetry(&self) {
        if let Some(t) = self.front.telemetry() {
            t.stop();
        }
    }

    fn start_worker(self: &Arc<Self>, st: &mut State) {
        if !st.worker {
            st.worker = true;
            let me = self.clone();
            std::thread::Builder::new().name("router".into()).spawn(move || me.run()).ok();
        }
    }

    fn grant_batch(&self, st: &mut State, svc: &Svc) {
        let mut keep = VecDeque::new();
        for turn in st.queue.drain(..).collect::<Vec<_>>() {
            let same = Arc::ptr_eq(&turn.svc(), svc);
            if same {
                let mut t = turn.0.lock().unwrap();
                t.state = TurnState::Granted;
                if t.holds {
                    st.running += 1;
                }
            } else {
                keep.push_back(turn);
            }
        }
        st.queue = keep;
        self.cv.notify_all();
    }

    fn fail_batch(&self, st: &mut State, svc: &Svc, error: EngineErr) {
        let mut keep = VecDeque::new();
        for turn in st.queue.drain(..).collect::<Vec<_>>() {
            if Arc::ptr_eq(&turn.svc(), svc) {
                let mut t = turn.0.lock().unwrap();
                t.state = TurnState::Failed;
                t.error = Some(error.clone());
            } else {
                keep.push_back(turn);
            }
        }
        st.queue = keep;
        self.cv.notify_all();
    }

    fn run(self: Arc<Self>) {
        loop {
            let (old, head, waiting) = {
                let mut st = self.st.lock().unwrap();
                let head = loop {
                    if st.stopping {
                        return;
                    }
                    let head = st.queue.front().map(|t| t.svc());
                    match head {
                        Some(h) if st.current.as_ref().map(|c| Arc::ptr_eq(c, &h)).unwrap_or(false) => {
                            self.grant_batch(&mut st, &h);
                        }
                        Some(h) if st.running == 0 => break h,
                        _ => st = self.cv.wait(st).unwrap(),
                    }
                };
                let from = st.current.as_ref().map(|c| c.model.clone());
                st.swap = Some(Swap { svc: head.clone(), from, phase: if st.current.is_some() { "unloading" } else { "loading" }, since: now() });
                (st.current.clone(), head, st.queue.len())
            };
            let failure = self.do_swap(old.as_ref(), &head, waiting);
            let mut st = self.st.lock().unwrap();
            match failure {
                None => {
                    st.current = Some(head.clone());
                    st.swaps += 1;
                    st.errors.remove(&key(&head));
                    self.grant_batch(&mut st, &head);
                }
                Some(e) => {
                    st.errors.insert(key(&head), e.message());
                    self.fail_batch(&mut st, &head, load_failure(&head.model, &e));
                }
            }
            st.swap = None;
            self.cv.notify_all();
        }
    }

    fn do_swap(&self, old: Option<&Svc>, new: &Svc, waiting: usize) -> Option<EngineErr> {
        let t = Instant::now();
        new.set_load_kind("weights");
        new.set_loading(true);
        new.engine.set_load_status(Map::new());
        match old {
            Some(o) => stdout::out(&format!("[server] swap {} -> {} ({waiting} waiting)", o.model, new.model)),
            None => stdout::out(&format!("[server] loading {} ({waiting} waiting)", new.model)),
        }
        let r = (|| -> Result<(), EngineErr> {
            if let Some(o) = old {
                o.swap_out()?;
                let mut st = self.st.lock().unwrap();
                st.current = None;
                if let Some(s) = st.swap.as_mut() {
                    s.phase = "loading";
                }
            }
            new.load()
        })();
        new.set_loading(false);
        if let Err(e) = r {
            stdout::out(&format!("[server] error: {} did not load: {}", new.model, e.message()));
            return Some(e);
        }
        stdout::out(&format!("[server] swap: {} ready in {} s", new.model, serve_text::pyjson::py_fixed(t.elapsed().as_secs_f64(), 1)));
        None
    }
}
