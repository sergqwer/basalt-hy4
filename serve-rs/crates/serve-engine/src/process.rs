//! serve/engine/process.py: ProcessEngine, one resident `basalt --serve` child driven over stdin and stdout.
//! The protocol is described in the Python module's docstring; this is a line-for-line port. Python's thread-locals
//! (last, read_ms, prompt progress, window counts) are keyed by the request thread (`iface::thread_ident`), since a
//! generator body runs on its own thread here.
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

use crate::console::{ConsoleLines, LINES as CONSOLE_LINES};
use crate::errors::{split_err, EResult, EngineErr};
use crate::flags::flag_value;
use crate::gen::{Gen, Stop, Yielder};
use crate::iface::{thread_ident, Engine, Flag, GenStream, JsonMap, Last, Prompt, PyLock, RamCheck, Sampling,
                   SlotDrafts, VramCheck};
use crate::logprobs::{self, LogProb, Tok};
use crate::progress;
use crate::py;
use crate::slots::{AdmissionQueue, Ticket};
use crate::suspend;
use crate::util::console::tagged;
use crate::util::stdout;
use serve_text::pyjson::py_float_repr;
use serve_text::PromptText;

pub const ENGINE_SILENCE_S: f64 = 300.0;
pub const PP_CHUNK_MAX: i64 = 32768;
pub const SLOT_MARGIN: i64 = 8;
pub const PP_FLOOR_TOK_S: f64 = 50.0;
pub const PP_SLACK: f64 = 3.0;
pub const ENGINE_READY_S: f64 = 900.0;
pub const QUIT_S: f64 = 20.0;
pub const QUIT_DISK_S: f64 = 300.0;
pub const PENALTY_LAST_N_DEFAULT: i64 = 64;
pub const RESTART_RETRY_S: f64 = 15.0;
pub const ENC_TIMEOUT_S: f64 = 300.0;
pub const BDONE_WAIT_S: f64 = 2.0;
pub const YIELDS_MAX: usize = 2;
pub const SOLO_AGAIN_MAX: usize = 2;
pub const SOLO_AGAIN_MIN_LEFT: i64 = 32;

/// --request-lines: echo each request's DONE figures on stdout.
pub static REQUEST_LINES: AtomicBool = AtomicBool::new(false);

pub const AUTO_FLAGS: &str = "auto flags: ";

pub const FATAL_PREFIXES: [&str; 4] = [
    "ERR verify: timed out at layer ",
    "ERR verify batch: timed out at layer ",
    "ERR verify: an earlier window never finished",
    "ERR verify batch: an earlier window never finished",
];
pub const REQUEST_ERRS: [&str; 9] = [
    "prompt (", "bad request:", "sampling ", "logit_bias:", "grammar:", "ppl:",
    "a token id is outside the vocabulary", "this engine was started without --vision",
    "the prompt cannot end in an image",
];

fn protocol_line(line: &str) -> bool {
    const P: [&str; 21] = [AUTO_FLAGS, "INFO ", "CONV ", "READY", "T ", "P ", "PP ", "PL ", "PT ", "RESUME ",
        "REUSED ", "DONE", "ERR", "BADM ", "BT ", "BP ", "BDONE ", "YIELDED ", "VRAM ", "ENC ", "VIDEO "];
    P.iter().any(|p| line.starts_with(p)) || CONSOLE_LINES.iter().any(|p| line.starts_with(p))
}

fn now_s() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s.max(0.0))
}

/// The facts of a `CONV` line.
pub fn conversation_facts(line: &str) -> JsonMap {
    let mut facts = Map::new();
    for kv in py::split(line).into_iter().skip(1) {
        let (k, _, v) = py::partition(kv, "=");
        if k.ends_with("_ms") {
            if let Some(x) = py::float_of(v) {
                facts.insert(k.into(), json!(x));
            }
        } else if let Ok(n) = v.trim().parse::<i64>() {
            facts.insert(k.into(), json!(n));
        }
    }
    facts
}

/// Drop the `\r` before a line's final `\n`.
pub fn engine_line(line: &str) -> String {
    if let Some(s) = line.strip_suffix("\r\n") {
        format!("{s}\n")
    } else {
        line.to_string()
    }
}

/// The key=value facts of an `INFO` line, integers as ints.
pub fn info_facts(line: &str) -> JsonMap {
    let mut facts = Map::new();
    for kv in py::split(line).into_iter().skip(1) {
        let (k, _, v) = py::partition(kv, "=");
        let val = if py::int_like(v) {
            v.parse::<i64>().map(Value::from).unwrap_or_else(|_| Value::from(v))
        } else {
            Value::from(v)
        };
        facts.insert(k.into(), val);
    }
    facts
}

const AUTO_NAMES: [(&str, &str); 3] = [("peer-exclusive", "peer-excl"), ("pool-workers", "pool"), ("resident", "ram-experts")];

/// Print the engine's auto-picked flags as a server line; false when the line is not that line.
pub fn relay_auto_flags(line: &str) -> bool {
    let Some(rest) = line.strip_prefix(AUTO_FLAGS) else { return false };
    let rest = rest.trim_end_matches('\n');
    let mut pairs = Vec::new();
    for item in rest.split(", ") {
        let (name, _, value) = py::partition(item, " ");
        let name = AUTO_NAMES.iter().find(|(a, _)| *a == name).map(|(_, b)| *b).unwrap_or(name);
        pairs.push(if !value.is_empty() { format!("{}={}", name, value.replace(' ', "")) } else { name.to_string() });
    }
    progress::server_line(&tagged("auto", &pairs.join(" ")));
    true
}

/// The engine log as the server writes it (Python's text-mode handle: `\n` is `\r\n` on Windows).
pub struct LogFile {
    f: Mutex<File>,
}

impl LogFile {
    pub fn write_line(&self, line: &str) {
        let mut text = if line.ends_with('\n') { line.to_string() } else { format!("{line}\n") };
        if cfg!(windows) {
            text = text.replace('\n', "\r\n");
        }
        if let Ok(mut f) = self.f.lock() {
            let _ = f.write_all(text.as_bytes());
            let _ = f.flush();
        }
    }
}

/// Write an engine stdout line the server does not parse to the engine log, as it came.
pub fn log_unparsed(line: &str, log: Option<&LogFile>) {
    if protocol_line(line) || py::strip(line).is_empty() {
        return;
    }
    if let Some(l) = log {
        l.write_line(line);
    }
}

/// The exception for a request's ERR text: ValueError when the request caused it, else EngineError.
pub fn request_err(err: &str, shown: Option<String>) -> EngineErr {
    let message = split_err(err).2;
    let text = shown.unwrap_or_else(|| message.clone());
    if REQUEST_ERRS.iter().any(|p| message.starts_with(p)) {
        EngineErr::Value(text)
    } else {
        EngineErr::Error(text)
    }
}

fn number(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Number(n)) => n.as_f64().filter(|x| x.is_finite()),
        _ => None,
    }
}

fn is_whole(v: Option<&Value>) -> Option<i64> {
    let x = number(v)?;
    if x == x.trunc() {
        Some(match v {
            Some(Value::Number(n)) if n.is_i64() => n.as_i64().unwrap(),
            _ => x as i64,
        })
    } else {
        None
    }
}

/// The GEN line's sampling keys, each value as sent; a field left out is not sent.
pub fn sampling_keys(s: &Sampling) -> String {
    let m = &s.map;
    let mut keys = String::new();
    if let Some(t) = number(m.get("temperature")) {
        if t > 0.0 {
            keys += &format!(" temperature={}", py_float_repr(t));
        }
    }
    if let Some(tp) = number(m.get("top_p")) {
        keys += &format!(" top_p={}", py_float_repr(tp));
    }
    if let Some(tk) = is_whole(m.get("top_k")) {
        keys += &format!(" top_k={tk}");
    }
    if let Some(mp) = number(m.get("min_p")) {
        keys += &format!(" min_p={}", py_float_repr(mp));
    }
    let rp = number(m.get("repetition_penalty")).filter(|x| *x != 1.0);
    let pf = number(m.get("frequency_penalty")).filter(|x| *x != 0.0);
    let pp = number(m.get("presence_penalty")).filter(|x| *x != 0.0);
    if let Some(x) = rp {
        keys += &format!(" penalty_repeat={}", py_float_repr(x));
    }
    if let Some(x) = pf {
        keys += &format!(" penalty_freq={}", py_float_repr(x));
    }
    if let Some(x) = pp {
        keys += &format!(" penalty_present={}", py_float_repr(x));
    }
    if rp.is_some() || pf.is_some() || pp.is_some() {
        let pln = number(m.get("penalty_last_n")).map(|x| x.trunc() as i64).unwrap_or(PENALTY_LAST_N_DEFAULT);
        keys += &format!(" penalty_last_n={pln}");
    }
    if let Some(seed) = is_whole(m.get("seed")) {
        keys += &format!(" seed={seed}");
    }
    match m.get("relaxed_drafting") {
        Some(Value::Bool(false)) => keys += " relax=0",
        v => {
            if let Some(x) = number(v) {
                keys += &format!(" relax={}", py_float_repr(x));
            }
        }
    }
    keys + &projection_key(m)
        + &logprobs::bias_key(s.logit_bias.as_deref())
        + &logprobs::grammar_key(m.get(logprobs::GRAMMAR_KEY)).unwrap_or_default()
        + &logprobs::logprobs_key(m.get(logprobs::LOGPROBS_KEY))
}

/// `cvec=0|1` for the speed-projection control vector.
pub fn projection_key(m: &JsonMap) -> String {
    match m.get("experimental_speed_projection") {
        Some(Value::Bool(b)) => format!(" cvec={}", *b as i32),
        _ => String::new(),
    }
}

const NT_CRASH_CODES: [(u32, &str); 11] = [
    (0x40000015, "STATUS_FATAL_APP_EXIT"),
    (0xC0000005, "STATUS_ACCESS_VIOLATION"),
    (0xC0000006, "STATUS_IN_PAGE_ERROR"),
    (0xC000001D, "STATUS_ILLEGAL_INSTRUCTION"),
    (0xC0000094, "STATUS_INTEGER_DIVIDE_BY_ZERO"),
    (0xC00000FD, "STATUS_STACK_OVERFLOW"),
    (0xC0000135, "STATUS_DLL_NOT_FOUND"),
    (0xC000013A, "STATUS_CONTROL_C_EXIT"),
    (0xC0000374, "STATUS_HEAP_CORRUPTION"),
    (0xC0000409, "STATUS_STACK_BUFFER_OVERRUN"),
    (0xE06D7363, "C++ exception"),
];

fn signal_name(sig: i64) -> Option<&'static str> {
    Some(match sig {
        1 => "SIGHUP", 2 => "SIGINT", 3 => "SIGQUIT", 4 => "SIGILL", 5 => "SIGTRAP", 6 => "SIGABRT", 7 => "SIGBUS",
        8 => "SIGFPE", 9 => "SIGKILL", 10 => "SIGUSR1", 11 => "SIGSEGV", 12 => "SIGUSR2", 13 => "SIGPIPE",
        14 => "SIGALRM", 15 => "SIGTERM", 16 => "SIGSTKFLT", 17 => "SIGCHLD", 18 => "SIGCONT", 19 => "SIGSTOP",
        20 => "SIGTSTP", 21 => "SIGTTIN", 22 => "SIGTTOU", 23 => "SIGURG", 24 => "SIGXCPU", 25 => "SIGXFSZ",
        26 => "SIGVTALRM", 27 => "SIGPROF", 28 => "SIGWINCH", 29 => "SIGIO", 30 => "SIGPWR", 31 => "SIGSYS",
        _ => return None,
    })
}

/// Human-readable exit status, decoding POSIX signals and Windows NTSTATUS crash codes.
pub fn format_exit_status(rc: Option<i64>) -> String {
    let Some(rc) = rc else { return "the engine ended (no exit status)".into() };
    if rc < 0 {
        return match signal_name(-rc) {
            Some(n) if cfg!(unix) => format!("the engine was killed by {n}"),
            _ => format!("the engine was killed by signal {}", -rc),
        };
    }
    if let Some((_, name)) = NT_CRASH_CODES.iter().find(|(c, _)| *c as i64 == rc) {
        return format!("the engine crashed (0x{rc:08X} {name})");
    }
    if rc >= 0x8000_0000 {
        return format!("the engine crashed (0x{rc:08X})");
    }
    format!("the engine exited (code {rc})")
}

fn exit_code_of(st: std::process::ExitStatus) -> Option<i64> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = st.signal() {
            return Some(-(sig as i64));
        }
        st.code().map(|c| c as i64)
    }
    #[cfg(windows)]
    {
        st.code().map(|c| c as u32 as i64)
    }
}

// ------------------------------------------------------------------------------------------------ one process

type Line = String;

/// One engine process: its handle, stdin and the queues its pump fills (`proc`, `lines`, `prompt_q`, `slot_q`).
pub struct Io {
    pub gen: u64,
    pub pid: u32,
    child: Mutex<Child>,
    stdin: Mutex<Option<ChildStdin>>, // wlock
    lines: Mutex<Option<Receiver<Line>>>,
    prompt_q: Mutex<Option<Receiver<Line>>>,
    slot_q: Vec<Mutex<Option<Receiver<Line>>>>,
    log: Option<Arc<LogFile>>,
    pump: Mutex<Option<JoinHandle<()>>>,
    console: Arc<ConsoleLines>,
}

enum Got {
    Line(String),
    Timeout,
    Eof,
}

fn recv(q: &Mutex<Option<Receiver<Line>>>, timeout: Option<f64>) -> Got {
    let g = q.lock().unwrap();
    let Some(r) = g.as_ref() else { return Got::Eof };
    match timeout {
        None => match r.recv() {
            Ok(l) => Got::Line(l),
            Err(_) => Got::Eof,
        },
        Some(t) => match r.recv_timeout(secs(t)) {
            Ok(l) => Got::Line(l),
            Err(RecvTimeoutError::Timeout) => Got::Timeout,
            Err(RecvTimeoutError::Disconnected) => Got::Eof,
        },
    }
}

fn try_recv(q: &Mutex<Option<Receiver<Line>>>) -> Option<String> {
    q.lock().unwrap().as_ref().and_then(|r| r.try_recv().ok())
}

impl Io {
    fn poll(&self) -> Option<i64> {
        let mut c = self.child.lock().unwrap();
        match c.try_wait() {
            Ok(Some(st)) => Some(exit_code_of(st).unwrap_or(-1)),
            _ => None,
        }
    }
    /// proc.wait(timeout): Err(()) on timeout
    fn wait(&self, timeout: f64) -> Result<i64, ()> {
        let end = Instant::now() + secs(timeout);
        loop {
            if let Some(rc) = self.poll() {
                return Ok(rc);
            }
            if Instant::now() >= end {
                return Err(());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn kill(&self) {
        let _ = self.child.lock().unwrap().kill();
    }
    /// proc.terminate(): SIGTERM on POSIX, TerminateProcess on Windows
    fn terminate(&self) {
        #[cfg(unix)]
        unsafe {
            if self.poll().is_none() {
                libc::kill(self.pid as libc::pid_t, libc::SIGTERM);
            }
        }
        #[cfg(windows)]
        self.kill();
    }
    fn write(&self, text: &str) -> std::io::Result<()> {
        let mut g = self.stdin.lock().unwrap();
        match g.as_mut() {
            Some(w) => {
                w.write_all(text.as_bytes())?;
                w.flush()
            }
            None => Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "closed")),
        }
    }
    fn close_stdin(&self) {
        self.stdin.lock().unwrap().take();
    }
}

// ------------------------------------------------------------------------------------------------ state

struct Spawn {
    bin: String,
    args: Vec<String>,
    cwd: Option<String>,
    log: Option<String>,
    env: Option<Vec<(String, String)>>,
}

#[derive(Default)]
struct Tl {
    last: Option<Last>,
    read_ms: f64,
    ptl: Option<JsonMap>,
    windows: u64,
    tokens: u64,
}

struct St {
    io: Option<Arc<Io>>,           // the queues of the latest process (Python keeps self.lines after close)
    proc_set: bool,                // `self.proc is not None`
    ended: bool,
    unloaded: bool,
    loading: bool,
    load_status: JsonMap,
    starting: bool,
    asleep: bool,
    parked: Option<BTreeMap<u32, u64>>,
    restore_s: f64,
    max_context: i64,
    known_ctx: i64,
    can_stop: bool,
    info: JsonMap,
    video: Option<String>,
    conversation: JsonMap,
    prefill_tok_s_mean: Option<f64>,
    progress: Option<(i64, i64)>,
    silent_note: Option<String>,
    last_err: Option<String>,
    gen: u64,
    batch: usize,
    log_start: u64,
    log_ready_pos: u64,
    last_solo: Last,
    read_ms_solo: f64,
    slot_drafts: SlotDrafts,
    // the control lines' holder's state
    ctl_mode: &'static str,
    ctl_result: Option<(bool, bool)>, // BADM: (continues, no token yet); None also for "done"
    ctl_done: bool,
    yielded: Option<(i64, i64)>,
    ctl_erred: bool,
}

struct Slots {
    gen: u64,
    order: Vec<usize>,
    busy: Vec<bool>,
    held: Vec<i64>,
    used: Vec<f64>,
    epoch: Vec<u64>,
    owner: BTreeMap<usize, String>,
    thread: Vec<Option<u64>>,
    waiting: usize,
    wait_lens: Vec<(u64, i64)>,
    next_wait: u64,
    ctl_epoch: u64,
    admission: AdmissionQueue,
}

pub struct Inner {
    spawn: Spawn,
    pub model_path: String,
    quit_s: f64,
    pub log_path: Option<String>,
    st: Mutex<St>,
    slots: Mutex<Slots>,
    slot_cv: Condvar,
    ctl: PyLock,
    prompt_lock: Mutex<()>,
    sleep_lock: Mutex<()>,
    tl: Mutex<HashMap<u64, Tl>>,
    pub silence_s: Mutex<f64>,
}

static LAST_IDS: AtomicU64 = AtomicU64::new(1);

fn new_last(map: JsonMap) -> Last {
    Last { id: LAST_IDS.fetch_add(1, Ordering::Relaxed), map }
}

/// `engine.stderr_sink`: engine stderr when there is no log file.
pub static STDERR_SINK: Mutex<Option<Arc<dyn Fn(&str) + Send + Sync>>> = Mutex::new(None);

pub struct ProcessEngine {
    pub inner: Arc<Inner>,
}

impl ProcessEngine {
    /// ProcessEngine(engine_bin, args, cwd, log, env, lazy)
    pub fn new(engine_bin: &str, args: Vec<String>, cwd: Option<String>, log: Option<String>,
               env: Option<Vec<(String, String)>>, lazy: bool) -> EResult<Self> {
        let disk = flag_value(&args, "--conversation-cache-disk-mib").unwrap_or_else(|| "0".into());
        let inner = Arc::new(Inner {
            model_path: flag_value(&args, "--model").unwrap_or_default(),
            quit_s: if disk != "0" { QUIT_DISK_S } else { QUIT_S },
            log_path: log.clone(),
            spawn: Spawn { bin: engine_bin.to_string(), args, cwd, log, env },
            st: Mutex::new(St {
                io: None,
                proc_set: false,
                ended: true,
                unloaded: true,
                loading: false,
                load_status: Map::new(),
                starting: false,
                asleep: false,
                parked: None,
                restore_s: 0.0,
                max_context: 0,
                known_ctx: 0,
                can_stop: false,
                info: Map::new(),
                video: None,
                conversation: Map::new(),
                prefill_tok_s_mean: None,
                progress: None,
                silent_note: None,
                last_err: None,
                gen: 0,
                batch: 0,
                log_start: 0,
                log_ready_pos: 0,
                last_solo: new_last(Map::new()),
                read_ms_solo: 0.0,
                slot_drafts: SlotDrafts::new(),
                ctl_mode: "solo",
                ctl_result: None,
                ctl_done: false,
                yielded: None,
                ctl_erred: false,
            }),
            slots: Mutex::new(Slots {
                gen: 0,
                order: vec![],
                busy: vec![],
                held: vec![],
                used: vec![],
                epoch: vec![],
                owner: BTreeMap::new(),
                thread: vec![],
                waiting: 0,
                wait_lens: vec![],
                next_wait: 0,
                ctl_epoch: 0,
                admission: AdmissionQueue::default(),
            }),
            slot_cv: Condvar::new(),
            ctl: PyLock::new(),
            prompt_lock: Mutex::new(()),
            sleep_lock: Mutex::new(()),
            tl: Mutex::new(HashMap::new()),
            silence_s: Mutex::new(ENGINE_SILENCE_S),
        });
        inner.init(lazy)?;
        Ok(ProcessEngine { inner })
    }
}

impl Inner {
    fn st(&self) -> MutexGuard<'_, St> {
        self.st.lock().unwrap()
    }
    fn sl(&self) -> MutexGuard<'_, Slots> {
        self.slots.lock().unwrap()
    }
    fn io(&self) -> Option<Arc<Io>> {
        self.st().io.clone()
    }
    fn batch(&self) -> usize {
        self.st().batch
    }
    fn gen(&self) -> u64 {
        self.st().gen
    }
    fn info(&self) -> JsonMap {
        self.st().info.clone()
    }
    fn info_flag(&self, k: &str) -> bool {
        self.st().info.get(k).map(logprobs::py_truthy).unwrap_or(false)
    }

    // ---- thread-locals
    fn tl_with<R>(&self, owner: u64, f: impl FnOnce(&mut Tl) -> R) -> R {
        let mut g = self.tl.lock().unwrap();
        if g.len() > 4096 && !g.contains_key(&owner) {
            g.clear();
        }
        f(g.entry(owner).or_default())
    }
    /// `self.last` as the thread `owner` sees it
    fn last_of(&self, owner: u64) -> Last {
        if self.batch() > 0 {
            self.tl_with(owner, |t| t.last.get_or_insert_with(|| new_last(Map::new())).clone())
        } else {
            self.st().last_solo.clone()
        }
    }
    fn set_last(&self, owner: u64, l: Last) {
        let batch = {
            let mut st = self.st();
            st.last_solo = l.clone();
            st.batch
        };
        if batch > 0 {
            self.tl_with(owner, |t| t.last = Some(l));
        }
    }
    /// mutate `self.last` in place (same identity)
    fn edit_last(&self, owner: u64, f: impl FnOnce(&mut JsonMap)) {
        if self.batch() > 0 {
            self.tl_with(owner, |t| f(&mut t.last.get_or_insert_with(|| new_last(Map::new())).map));
        } else {
            f(&mut self.st().last_solo.map);
        }
    }
    fn read_ms(&self, owner: u64) -> f64 {
        if self.batch() > 0 {
            self.tl_with(owner, |t| t.read_ms)
        } else {
            self.st().read_ms_solo
        }
    }
    fn set_read_ms(&self, owner: u64, v: f64) {
        let batch = {
            let mut st = self.st();
            st.read_ms_solo = v;
            st.batch
        };
        if batch > 0 {
            self.tl_with(owner, |t| t.read_ms = v);
        }
    }
    fn note_read(&self, owner: u64, f: &[&str]) {
        let (a, b) = (f[1].parse::<i64>().unwrap_or(0), f[2].parse::<i64>().unwrap_or(0));
        self.st().progress = Some((a, b));
        let ms = f.get(3).and_then(|x| py::float_of(x));
        self.tl_with(owner, |t| {
            let mut state = t.ptl.clone().unwrap_or_default();
            state.insert("read".into(), json!(a));
            state.insert("total".into(), json!(b));
            if let Some(ms) = ms {
                state.insert("ms".into(), json!(ms));
            }
            t.ptl = Some(state);
        });
    }
    fn note_resume(&self, owner: u64, line: &str) {
        let Some(n) = py::split(line).get(1).and_then(|x| x.parse::<i64>().ok()) else { return };
        self.tl_with(owner, |t| {
            let mut state = t.ptl.clone().unwrap_or_default();
            state.insert("cache".into(), json!(n));
            t.ptl = Some(state);
        });
    }
    fn reset_read(&self, owner: u64) {
        self.tl_with(owner, |t| t.ptl = None);
    }
    fn count_window(&self, owner: u64, tokens: u64) {
        self.tl_with(owner, |t| {
            t.windows += 1;
            t.tokens += tokens;
        });
    }
    fn reset_windows(&self, owner: u64) {
        self.tl_with(owner, |t| {
            t.windows = 0;
            t.tokens = 0;
        });
    }
    fn note_live(&self, owner: u64, line: &str) {
        let f = py::split(line);
        if f.len() >= 5 && py::isdigit(f[1]) && py::isdigit(f[2]) {
            self.note_read(owner, &f);
            if let Some(x) = py::float_of(f[4]) {
                self.st().prefill_tok_s_mean = Some(x);
            }
        }
    }

    // ---- start
    fn rotate_log(path: &str) {
        if let Ok(md) = std::fs::metadata(path) {
            if md.len() >= 10 << 20 {
                let _ = std::fs::rename(path, format!("{path}.1"));
            }
        }
    }

    /// ProcessEngine.__init__ (also restart()'s new start).
    fn init(self: &Arc<Self>, lazy: bool) -> EResult<()> {
        let sp = &self.spawn;
        {
            let mut st = self.st();
            st.io = st.io.take(); // the queues stay until a new process replaces them
            st.proc_set = false;
            st.ended = true;
            st.unloaded = true;
            st.asleep = false;
            st.parked = None;
            st.restore_s = 0.0;
            st.max_context = flag_value(&sp.args, "--max-context").and_then(|v| v.parse().ok()).unwrap_or(0);
            st.can_stop = false;
            self.set_last_locked(&mut st, Map::new());
            st.slot_drafts = SlotDrafts::new();
            st.info = Map::new();
            st.video = None;
            st.conversation = Map::new();
            st.prefill_tok_s_mean = None;
            st.progress = None;
            st.silent_note = None;
            st.last_err = None;
            let version = std::fs::read(Path::new(&sp.bin).parent().unwrap_or(Path::new("")).join("BUILD.json"))
                .ok()
                .and_then(|b| serve_text::pyjson::loads_bytes(&b).ok())
                .map(|v| match v {
                    Value::Object(o) => o.get("version").cloned().unwrap_or(Value::Null),
                    _ => Value::Null, // .get on a non-dict: AttributeError in Python; treated as no version
                })
                .unwrap_or(Value::Null);
            st.info.insert("version".into(), version);
        }
        if lazy {
            return Ok(());
        }
        self.st().unloaded = false;
        let log = sp.log.clone();
        let mut log_file: Option<File> = None;
        if let Some(l) = &log {
            Self::rotate_log(l);
            log_file = Some(
                OpenOptions::new().create(true).append(true).open(l)
                    .map_err(|e| EngineErr::StartError { msg: format!("the engine could not be started: {e}"), log: log.clone() })?,
            );
        }
        let loading = Arc::new(progress::Event::new());
        let log_start = log.as_ref().and_then(|l| std::fs::metadata(l).ok()).map(|m| m.len()).unwrap_or(0);
        self.st().log_start = log_start;
        let mut narrator: Option<JoinHandle<()>> = None;
        if let Some(l) = &log {
            let visible = match &sp.env {
                Some(env) => env.iter().find(|(k, _)| k == "CUDA_VISIBLE_DEVICES").map(|(_, v)| v.clone()).unwrap_or_default(),
                None => std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
            };
            let gpus: Vec<i64> = visible.split(',').filter(|x| py::isdigit(x.trim())).map(|x| x.trim().parse().unwrap()).collect();
            let (l2, ev, me) = (l.clone(), loading.clone(), Arc::downgrade(self));
            let size = std::fs::metadata(l).map(|m| m.len()).unwrap_or(0);
            narrator = Some(std::thread::spawn(move || {
                progress::narrate_start(&l2, size, &ev, 20.0, None, Some(gpus), Some(Box::new(move |status: JsonMap| {
                    if let Some(me) = me.upgrade() {
                        me.st().load_status = status;
                    }
                })));
            }));
        }
        let sink = if log.is_some() { None } else { STDERR_SINK.lock().unwrap().clone() };
        self.st().max_context = 0;
        let mut cmd = Command::new(&sp.bin);
        cmd.arg("--serve").args(&sp.args).stdin(Stdio::piped()).stdout(Stdio::piped());
        if let Some(c) = &sp.cwd {
            cmd.current_dir(c);
        }
        if let Some(env) = &sp.env {
            cmd.env_clear();
            cmd.envs(env.iter().map(|(k, v)| (k, v)));
        }
        if sink.is_some() {
            cmd.stderr(Stdio::piped());
        } else if let Some(f) = &log_file {
            match f.try_clone() {
                Ok(h) => {
                    cmd.stderr(Stdio::from(h));
                }
                Err(_) => {
                    cmd.stderr(Stdio::null());
                }
            }
        } else {
            cmd.stderr(Stdio::null());
        }
        let mut child = match crate::lifeline::spawn(cmd) {
            Ok(c) => c,
            Err(e) => {
                loading.set();
                return Err(EngineErr::StartError {
                    msg: format!("the engine could not be started: {}", py_os_error(&e, &sp.bin)),
                    log: log.clone(),
                });
            }
        };
        let pid = child.id();
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("stdout");
        if let (Some(sink), Some(err)) = (sink, child.stderr.take()) {
            std::thread::spawn(move || {
                let mut r = BufReader::new(err);
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    match r.read_until(b'\n', &mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => sink(&engine_line(&String::from_utf8_lossy(&buf))),
                    }
                }
            });
        }
        let logw = log_file.map(|f| Arc::new(LogFile { f: Mutex::new(f) }));
        // one reader: the lines up to READY go to ready_q, then it waits for the pump's queues
        let (ready_tx, ready_rx) = channel::<Option<String>>();
        let (setup_tx, setup_rx) = channel::<PumpSetup>();
        let logr = logw.clone();
        let reader = std::thread::spawn(move || {
            let mut r = BufReader::new(stdout);
            let mut buf = Vec::new();
            let mut ready = false;
            loop {
                buf.clear();
                match r.read_until(b'\n', &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                let line = engine_line(&String::from_utf8_lossy(&buf));
                log_unparsed(&line, logr.as_deref());
                let is_ready = line.starts_with("READY");
                let _ = ready_tx.send(Some(line));
                if is_ready {
                    ready = true;
                    break;
                }
            }
            if !ready {
                let _ = ready_tx.send(None);
                return;
            }
            drop(ready_tx);
            if let Ok(setup) = setup_rx.recv() {
                pump(r, setup);
            }
        });
        let child = Mutex::new(child);
        let deadline = Instant::now() + secs(ENGINE_READY_S);
        let mut timed_out = false;
        let (mut max_ctx, mut can_stop) = (0i64, false);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                timed_out = true;
                break;
            }
            let line = match ready_rx.recv_timeout(left) {
                Ok(Some(l)) => l,
                Ok(None) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => continue,
            };
            if relay_auto_flags(&line) {
                continue;
            }
            if line.starts_with("INFO ") {
                let f = info_facts(&line);
                let mut st = self.st();
                for (k, v) in f {
                    st.info.insert(k, v);
                }
            }
            if let Some(v) = line.strip_prefix("VIDEO ") {
                self.st().video = Some(py::strip(v).to_string());
            }
            if line.starts_with("READY") {
                let f = py::split(&line);
                max_ctx = f.get(1).and_then(|x| x.parse().ok()).unwrap_or(0);
                can_stop = f[2..].contains(&"stop");
                break;
            }
        }
        {
            let mut st = self.st();
            st.max_context = max_ctx;
            st.can_stop = can_stop;
        }
        loading.set();
        if let Some(l) = &log {
            if let Some(n) = narrator.take() {
                join_timeout(n, 2.0);
            }
            self.st().log_ready_pos = std::fs::metadata(l).map(|m| m.len()).unwrap_or(0);
        }
        if max_ctx <= 0 {
            let io_tmp = Io {
                gen: 0,
                pid,
                child,
                stdin: Mutex::new(stdin),
                lines: Mutex::new(None),
                prompt_q: Mutex::new(None),
                slot_q: vec![],
                log: logw,
                pump: Mutex::new(None),
                console: Arc::new(ConsoleLines::new()),
            };
            if timed_out {
                io_tmp.kill();
            }
            let _ = io_tmp.wait(5.0);
            io_tmp.close_stdin();
            drop(setup_tx);
            let reason = if timed_out {
                format!("the engine did not report READY within {:.0} s", ENGINE_READY_S)
            } else {
                let errors = progress::start_errors(log.as_deref(), log_start);
                if !errors.is_empty() {
                    let mut seen: Vec<String> = Vec::new();
                    for e in errors {
                        if !seen.contains(&e) {
                            seen.push(e);
                        }
                    }
                    seen.join("; ")
                } else {
                    let last = progress::start_last_line(log.as_deref(), log_start);
                    if last.is_empty() { "the engine exited before it was ready".into() } else { last }
                }
            };
            // the process (exited) stays as `self.proc` for restart()'s wait, as in Python
            {
                let mut st = self.st();
                st.proc_set = true;
                let g = st.gen;
                st.io = Some(Arc::new(Io { gen: g, ..io_tmp }));
            }
            let _ = reader;
            return Err(EngineErr::StartError { msg: reason + &progress::start_failure_hint(log.as_deref(), log_start), log });
        }
        // alive from here
        let asked = ["--batch", "--slots"].iter().filter_map(|k| flag_value(&sp.args, k)).next()
            .and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
        let (lines_tx, lines_rx) = channel();
        let (prompt_tx, prompt_rx) = channel();
        let console = Arc::new(ConsoleLines::new());
        let (gen, batch) = {
            let mut st = self.st();
            st.known_ctx = st.max_context;
            if let Some(e) = st.info.get("engine").cloned() {
                if logprobs::py_truthy(&e) {
                    st.info.insert("version".into(), Value::from(serve_text::pyjson::py_str(&e)));
                }
            }
            st.gen += 1;
            st.batch = st.info.get("batch_slots").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as usize;
            st.yielded = None;
            st.ctl_erred = false;
            (st.gen, st.batch)
        };
        if asked != 0 && batch as i64 != asked {
            let n = if batch > 0 { batch.to_string() } else { "one".into() };
            stdout::out(&format!("[server] warning: {asked} parallel requests asked, the engine runs {n}"));
        }
        let mut slot_tx = Vec::new();
        let mut slot_rx = Vec::new();
        for _ in 0..batch {
            let (t, r) = channel();
            slot_tx.push(t);
            slot_rx.push(Mutex::new(Some(r)));
        }
        {
            let mut s = self.sl();
            s.gen = gen;
            s.order = (0..batch).collect();
            s.busy = vec![false; batch];
            s.held = vec![0; batch];
            s.used = vec![0.0; batch];
            s.epoch = vec![0; batch];
            s.owner = BTreeMap::new();
            s.thread = vec![None; batch];
        }
        let io = Arc::new(Io {
            gen,
            pid,
            child,
            stdin: Mutex::new(stdin),
            lines: Mutex::new(Some(lines_rx)),
            prompt_q: Mutex::new(Some(prompt_rx)),
            slot_q: slot_rx,
            log: logw.clone(),
            pump: Mutex::new(Some(reader)),
            console: console.clone(),
        });
        {
            let mut st = self.st();
            st.io = Some(io.clone());
            st.proc_set = true;
            st.ended = false;
        }
        let _ = setup_tx.send(PumpSetup {
            me: Arc::downgrade(self),
            io: Arc::downgrade(&io),
            gen,
            lines: lines_tx,
            prompt_q: prompt_tx,
            slot_q: slot_tx,
            log: logw,
            console,
        });
        let _ = io.write("STATS\n");
        Ok(())
    }

    fn set_last_locked(&self, st: &mut St, map: JsonMap) {
        st.last_solo = new_last(map);
        self.tl.lock().unwrap().clear();
    }

    fn alive(&self) -> bool {
        let io = {
            let st = self.st();
            if !st.proc_set || st.ended {
                return false;
            }
            st.io.clone()
        };
        io.map(|p| p.poll().is_none()).unwrap_or(false)
    }

    fn exit_code(&self) -> Option<i64> {
        let io = {
            let st = self.st();
            if !st.proc_set {
                return None;
            }
            st.io.clone()
        };
        io.and_then(|p| p.wait(5.0).ok())
    }

    fn died(&self) -> EngineErr {
        EngineErr::Died(self.death_note())
    }

    /// What is known of why the engine ended.
    fn death_note(&self) -> String {
        if let Some(n) = self.st().silent_note.clone() {
            return n;
        }
        let mut tail = String::new();
        if let Some(p) = &self.log_path {
            if let Ok(mut f) = File::open(p) {
                let end = f.seek(SeekFrom::End(0)).unwrap_or(0);
                let _ = f.seek(SeekFrom::Start(end.saturating_sub(4096)));
                let mut b = Vec::new();
                let _ = f.read_to_end(&mut b);
                tail = String::from_utf8_lossy(&b).into_owned();
            }
        }
        let raw_lines = py::splitlines(&tail);
        for line in raw_lines.iter().rev() {
            if line.contains("issue #29") {
                return format!("the engine's watchdog stopped it: {}", py::strip(line));
            }
        }
        let rc = {
            let st = self.st();
            if st.proc_set { st.io.clone() } else { None }
        }
        .and_then(|p| p.poll());
        let status = format_exit_status(rc);
        let last_err = self.st().last_err.clone();
        if let (Some(rc), Some(le)) = (rc, last_err.filter(|s| !s.is_empty())) {
            if rc >= 0 {
                return format!("{status} after it reported: {le}");
            }
        }
        let err_idx = (0..raw_lines.len()).rev()
            .find(|&i| raw_lines[i].contains(": error: ") || raw_lines[i].trim_start().starts_with("ERR"));
        if let Some(i) = err_idx {
            let mut block = vec![py::strip(raw_lines[i]).to_string()];
            for nxt in &raw_lines[i + 1..] {
                if (nxt.starts_with("  ") || nxt.starts_with('\t')) && !py::strip(nxt).is_empty() {
                    block.push(py::strip(nxt).to_string());
                } else {
                    break;
                }
            }
            return format!("{status}, last error in its log: {}", block.join(" | "));
        }
        let last = raw_lines.iter().rev().map(|x| py::strip(x)).find(|x| x.starts_with("basalt")).unwrap_or("");
        if !last.is_empty() {
            format!("{status}, last log line: {last}")
        } else {
            format!("{status}, nothing in its log")
        }
    }

    fn send(&self, text: &str) -> EResult<()> {
        let io = {
            let st = self.st();
            if st.proc_set { st.io.clone() } else { None }
        };
        match io {
            Some(p) => p.write(&format!("{text}\n")).map_err(|_| self.died()),
            None => Err(self.died()),
        }
    }

    fn write_raw(&self, text: &str) -> std::io::Result<()> {
        let io = {
            let st = self.st();
            if st.proc_set { st.io.clone() } else { None }
        };
        match io {
            Some(p) => p.write(text),
            None => Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "no process")),
        }
    }

    fn lines_get(&self, timeout: Option<f64>) -> Got {
        match self.io() {
            Some(p) => recv(&p.lines, timeout),
            None => Got::Eof,
        }
    }

    fn parse_done(&self, owner: u64, line: &str) {
        let f = py::split(line);
        let fl = |i: usize| f.get(i).and_then(|x| py::float_of(x)).unwrap_or(0.0);
        let int = |i: usize| f.get(i).and_then(|x| x.parse::<i64>().ok()).unwrap_or(0);
        let rm = self.read_ms(owner) + fl(3);
        self.set_read_ms(owner, rm);
        let mut m = Map::new();
        m.insert("generated".into(), json!(int(1)));
        m.insert("prompt_tokens".into(), json!(int(2)));
        m.insert("prompt_ms".into(), json!(fl(3)));
        m.insert("decode_ms".into(), json!(fl(4)));
        m.insert("finish".into(), json!(f.get(5).copied().unwrap_or("")));
        if f.len() >= 9 {
            m.insert("drafts_accepted".into(), json!(int(6)));
            m.insert("drafts_offered".into(), json!(int(7)));
            m.insert("reused".into(), json!(int(8)));
        }
        if f.len() >= 11 {
            m.insert("hits".into(), json!(int(9)));
            m.insert("lookups".into(), json!(int(10)));
        }
        if f.len() >= 13 {
            m.insert("file_reads".into(), json!(int(12)));
        }
        if f.len() >= 15 {
            m.insert("prompt_read".into(), json!(int(14)));
        }
        if f.len() >= 16 {
            m.insert("offloaded".into(), json!(int(15)));
        }
        if f.len() >= 17 {
            m.insert("windows".into(), json!(int(16)));
        }
        self.set_last(owner, new_last(m.clone()));
        if REQUEST_LINES.load(Ordering::Relaxed) {
            echo_request(&m);
        }
    }

    fn vram_command(&self, command: &str, timeout: f64) -> EResult<JsonMap> {
        if !self.alive() {
            return Err(EngineErr::died("the engine is not running"));
        }
        let what = command.split(' ').next().unwrap_or("").to_string();
        let _ = self.write_raw(&format!("{command}\n"));
        let deadline = Instant::now() + secs(timeout);
        loop {
            let left = deadline.saturating_duration_since(Instant::now()).as_secs_f64().max(0.1);
            let line = match self.lines_get(Some(left)) {
                Got::Line(l) => l,
                Got::Timeout => return Err(EngineErr::Died(format!("the engine did not answer the {what} command"))),
                Got::Eof => return Err(EngineErr::died("the engine ended")),
            };
            let line = py::strip(&line).to_string();
            if line.starts_with("ERR") {
                let rest = py::strip(line.get(4..).unwrap_or("")).to_string();
                return Err(EngineErr::Value(if rest.is_empty() { format!("the engine refused the {what} command") } else { rest }));
            }
            if line.starts_with("VRAM ") {
                let out = info_facts(&line);
                let mut st = self.st();
                for k in ["expert_slots", "vram_free_mib"] {
                    if let Some(v) = out.get(k) {
                        st.info.insert(k.into(), v.clone());
                    }
                }
                st.info.insert("vram".into(), Value::Object(out.clone()));
                return Ok(out);
            }
            if Instant::now() > deadline {
                return Err(EngineErr::Died(format!("the engine did not answer the {what} command")));
            }
        }
    }

    fn silent(&self, what: &str) -> EngineErr {
        let io = {
            let mut st = self.st();
            st.silent_note = Some("the engine and the server lost step (engine_silence_s, 0 = wait forever)".into());
            st.ended = true;
            if st.proc_set { st.io.clone() } else { None }
        };
        if let Some(p) = io {
            p.kill();
            let _ = p.wait(20.0);
        }
        EngineErr::Silent(format!("{what}; the server ended the engine"))
    }

    /// End the engine: QUIT, then terminate, then kill, 20 s each; EngineStuck if it still runs.
    fn close(&self) -> EResult<()> {
        let io = {
            let st = self.st();
            if !st.proc_set {
                return Ok(());
            }
            st.io.clone()
        };
        let Some(p) = io else { return Ok(()) };
        {
            let _g = self.sleep_lock.lock().unwrap();
            let asleep = std::mem::replace(&mut self.st().asleep, false);
            if asleep && suspend::resume(p.pid).is_some() {
                p.kill();
                let _ = p.wait(20.0);
            }
        }
        let mut result = Ok(());
        if p.poll().is_none() {
            let quit_ok = p.write("QUIT\n").is_ok();
            p.close_stdin();
            let waited = quit_ok && p.wait(self.quit_s).is_ok();
            if !waited {
                p.terminate();
                if p.wait(20.0).is_err() {
                    p.kill();
                    if p.wait(20.0).is_err() {
                        result = Err(EngineErr::Stuck("the engine is still releasing GPU/RAM; retry unloading after it exits".into()));
                    }
                }
            }
        }
        if p.poll().is_some() {
            if let Some(h) = p.pump.lock().unwrap().take() {
                join_timeout(h, 2.0);
            }
            p.close_stdin();
            let mut st = self.st();
            st.proc_set = false;
            st.ended = true;
            st.progress = None;
            st.conversation = Map::new();
            self.set_last_locked(&mut st, Map::new());
        }
        result
    }

    fn restart(self: &Arc<Self>, tries: u32) -> EResult<()> {
        let info = self.info();
        {
            let mut st = self.st();
            st.starting = true;
            st.loading = true;
            st.load_status = Map::new();
        }
        let r = (|| {
            self.close()?;
            for i in 0..tries {
                self.st().ended = true;
                match self.init(false) {
                    Ok(()) => break,
                    Err(e) => {
                        if !matches!(e, EngineErr::StartError { .. }) {
                            return Err(e);
                        }
                        let p = {
                            let st = self.st();
                            if st.proc_set { st.io.clone() } else { None }
                        };
                        let Some(p) = p else { return Err(e) };
                        let _ = p.wait(60.0);
                        if i == tries - 1 {
                            return Err(e);
                        }
                        stdout::out(&format!(
                            "[server] warning: the engine did not start (try {} of {}), retrying in {} s",
                            i + 1, tries, serve_text::pyjson::py_format_g(RESTART_RETRY_S)));
                        std::thread::sleep(secs(RESTART_RETRY_S));
                    }
                }
            }
            Ok(())
        })();
        {
            let mut st = self.st();
            st.starting = false;
            st.loading = false;
        }
        {
            let _s = self.sl();
            self.slot_cv.notify_all();
        }
        r?;
        let mut st = self.st();
        let mut merged = info;
        for (k, v) in std::mem::take(&mut st.info) {
            merged.insert(k, v);
        }
        st.info = merged;
        Ok(())
    }
}

/// --request-lines: one stdout line for the request the last DONE ended.
fn echo_request(d: &JsonMap) {
    let f = |k: &str| d.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
    let i = |k: &str| d.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
    let (read, gen, pms, dms) = (i("prompt_read"), i("generated"), f("prompt_ms"), f("decode_ms"));
    stdout::out(&format!(
        "[server] request prompt {} cached {} output {} prompt_read {} ms total {} ms prefill {} tok/s decode {} tok/s",
        i("prompt_tokens"), i("reused"), gen, pyf(pms, 0), pyf(pms + dms, 0),
        pyf(if pms > 0.0 { 1000.0 * read as f64 / pms } else { 0.0 }, 1),
        pyf(if dms > 0.0 { 1000.0 * gen as f64 / dms } else { 0.0 }, 1)));
}

/// `"%.<n>f" % x`
fn pyf(x: f64, n: usize) -> String {
    serve_text::pyjson::py_fixed(x, n)
}

fn py_os_error(e: &std::io::Error, bin: &str) -> String {
    // Python: "[WinError 2] The system cannot find the file specified" / "[Errno 2] No such file or directory: 'x'"
    let code = e.raw_os_error().unwrap_or(0);
    if cfg!(windows) {
        let msg = match code {
            2 => "The system cannot find the file specified".to_string(),
            3 => "The system cannot find the path specified".to_string(),
            5 => "Access is denied".to_string(),
            193 => format!("%1 is not a valid Win32 application"),
            _ => e.to_string(),
        };
        format!("[WinError {code}] {msg}")
    } else {
        let msg = match code {
            2 => "No such file or directory",
            13 => "Permission denied",
            8 => "Exec format error",
            _ => "",
        };
        if msg.is_empty() { e.to_string() } else { format!("[Errno {code}] {msg}: {}", serve_text::pyjson::py_str_repr(bin)) }
    }
}

fn join_timeout(h: JoinHandle<()>, s: f64) {
    let end = Instant::now() + secs(s);
    while !h.is_finished() && Instant::now() < end {
        std::thread::sleep(Duration::from_millis(5));
    }
    if h.is_finished() {
        let _ = h.join();
    }
}

struct PumpSetup {
    me: std::sync::Weak<Inner>,
    io: std::sync::Weak<Io>,
    gen: u64,
    lines: Sender<Line>,
    prompt_q: Sender<Line>,
    slot_q: Vec<Sender<Line>>,
    log: Option<Arc<LogFile>>,
    console: Arc<ConsoleLines>,
}

/// ProcessEngine._pump: route the engine's stdout lines after READY.
fn pump(mut r: BufReader<std::process::ChildStdout>, s: PumpSetup) {
    let current = |s: &PumpSetup| s.me.upgrade().map(|m| m.st().gen == s.gen && m.st().proc_set).unwrap_or(false);
    let mut buf = Vec::new();
    let mut last: Option<String> = None;
    loop {
        buf.clear();
        match r.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = engine_line(&String::from_utf8_lossy(&buf));
        last = Some(line.clone());
        log_unparsed(&line, s.log.as_deref());
        if line.starts_with(AUTO_FLAGS) {
            continue;
        }
        if line.starts_with("INFO ") {
            if current(&s) {
                if let Some(m) = s.me.upgrade() {
                    let f = info_facts(&line);
                    let mut st = m.st();
                    for (k, v) in f {
                        st.info.insert(k, v);
                    }
                }
            }
            continue;
        }
        if CONSOLE_LINES.iter().any(|p| line.starts_with(p)) {
            if current(&s) {
                s.console.feed(&line);
            }
            continue;
        }
        if line.starts_with("CONV ") {
            if current(&s) {
                if let Some(m) = s.me.upgrade() {
                    let f = conversation_facts(&line);
                    let mut st = m.st();
                    for (k, v) in f {
                        st.conversation.insert(k, v);
                    }
                }
            }
            continue;
        }
        if FATAL_PREFIXES.iter().any(|p| line.starts_with(p)) {
            if current(&s) {
                if let Some(m) = s.me.upgrade() {
                    let mut st = m.st();
                    st.silent_note = Some(format!("unrecoverable verification failure: {}", py::strip(&line[4..])));
                    st.ended = true;
                }
            }
            if let Some(io) = s.io.upgrade() {
                io.terminate();
            }
            break;
        }
        if line.starts_with("PROMPT ") {
            let _ = s.prompt_q.send(line);
            continue;
        }
        if (line.starts_with("BT ") || line.starts_with("BP ") || line.starts_with("BDONE ")) && !s.slot_q.is_empty() {
            if let Some(slot) = py::split(&line).get(1).and_then(|x| x.parse::<i64>().ok()) {
                if slot >= 0 && (slot as usize) < s.slot_q.len() {
                    let _ = s.slot_q[slot as usize].send(line);
                    continue;
                }
            }
        }
        let _ = s.lines.send(line);
    }
    if current(&s) {
        if let Some(m) = s.me.upgrade() {
            let mut st = m.st();
            st.ended = true;
            if let Some(l) = last.filter(|l| l.starts_with("ERR")) {
                st.last_err = Some(py::strip(l.get(4..).unwrap_or("")).to_string());
            }
        }
    }
    // dropping the senders is the None every reader sees
}

// ------------------------------------------------------------------------------------------------ generation

type Y = Yielder<Option<Tok>>;

/// What _control returned: ("done") or ("badm", continues, no token yet).
#[derive(Debug, Clone, Copy)]
enum CtlEnd {
    Done,
    Badm(bool, bool),
}

impl Inner {
    fn eos_ids(&self) -> Vec<u32> {
        let v = self.st().info.get("eos_ids").cloned();
        let text = match v {
            Some(Value::String(s)) => s,
            Some(Value::Number(n)) => n.to_string(),
            _ => String::new(),
        };
        text.split(',').filter(|i| !i.is_empty()).filter_map(|i| i.trim().parse().ok()).collect()
    }

    /// Wait for the control lines, heartbeats through `beat`; Ok(false) when cancelled.
    fn take_control(&self, cancel: &Flag, plen: i64, after_epoch: Option<u64>,
                    beat_fn: &mut dyn FnMut() -> Result<(), Stop>) -> Result<bool, Stop> {
        let id = {
            let mut s = self.sl();
            s.waiting += 1;
            s.next_wait += 1;
            let id = s.next_wait;
            s.wait_lens.push((id, plen));
            id
        };
        let mut beat = Instant::now();
        let r = (|| -> Result<bool, Stop> {
            loop {
                if !self.alive() && !self.st().starting {
                    return Err(Stop::Raised(EngineErr::died("the engine stopped while this request waited; it was not sent")));
                }
                let mut turn = {
                    let s = self.sl();
                    after_epoch.is_none() || s.ctl_epoch > after_epoch.unwrap() || s.waiting <= 1
                };
                if turn && self.ctl.acquire(Some(Duration::from_millis(500))) {
                    if self.alive() {
                        return Ok(true);
                    }
                    self.ctl.release();
                    turn = false;
                }
                if !turn {
                    let s = self.sl();
                    let _ = self.slot_cv.wait_timeout(s, Duration::from_millis(500)).unwrap();
                }
                if cancel.is_set() {
                    return Ok(false);
                }
                if beat.elapsed().as_secs_f64() >= 10.0 {
                    beat = Instant::now();
                    beat_fn()?;
                }
            }
        })();
        {
            let mut s = self.sl();
            s.waiting = s.waiting.saturating_sub(1);
            if let Some(i) = s.wait_lens.iter().position(|(k, _)| *k == id) {
                s.wait_lens.remove(i);
            }
        }
        if matches!(r, Ok(true)) {
            let mut s = self.sl();
            s.ctl_epoch += 1;
            self.slot_cv.notify_all();
        }
        r
    }

    /// _control: read the control lines of a GEN/BGEN; `step(None)` per heartbeat, `step(Some(tok))` per token.
    #[allow(clippy::too_many_arguments)]
    fn control(&self, owner: u64, cancel: &Flag, stop_when: Option<&dyn Fn() -> bool>,
               mut idle: Option<&mut dyn FnMut() -> EResult<()>>, stop_soon: Option<&Flag>,
               step: &mut dyn FnMut(Option<Tok>) -> Result<(), Stop>) -> Result<CtlEnd, Stop> {
        let mut stopped = false;
        let mut beat = Instant::now();
        let mut lp: Option<(u32, LogProb)> = None;
        loop {
            if !stopped && stop_soon.map(|f| f.is_set()).unwrap_or(false) && self.st().ctl_mode == "solo" {
                self.send("STOP")?;
                stopped = true;
            }
            let line = match self.lines_get(Some(0.5)) {
                Got::Timeout => {
                    if cancel.is_set() && !stopped {
                        self.send("STOP")?;
                        stopped = true;
                    }
                    if let Some(f) = idle.as_mut() {
                        f()?;
                    }
                    if beat.elapsed().as_secs_f64() >= 10.0 {
                        beat = Instant::now();
                        step(None)?;
                    }
                    continue;
                }
                Got::Eof => return Err(Stop::Raised(self.died())),
                Got::Line(l) => l,
            };
            beat = Instant::now();
            if line.starts_with("P ") {
                lp = logprobs::parse_line(&py::split(&line)[1..]);
                continue;
            }
            if let Some(rest) = line.strip_prefix("T ") {
                let id = py::strip(rest).parse::<i64>().unwrap_or(0) as u32;
                let t = logprobs::token(id, lp.as_ref());
                lp = None;
                // on_token, then the STOP check, then `yield False`
                if !stopped && (cancel.is_set() || stop_when.map(|f| f()).unwrap_or(false)) {
                    self.send("STOP")?;
                    stopped = true;
                }
                step(Some(t))?;
            } else if line.starts_with("PP ") {
                let f = py::split(&line);
                if f.len() >= 3 && py::isdigit(f[1]) && py::isdigit(f[2]) {
                    self.note_read(owner, &f);
                    self.st().prefill_tok_s_mean = if f.len() >= 5 { py::float_of(f[4]) } else { None };
                }
                if !stopped && cancel.is_set() {
                    self.send("STOP")?;
                    stopped = true;
                }
                step(None)?;
            } else if line.starts_with("PL ") {
                self.note_live(owner, &line);
                if !stopped && cancel.is_set() {
                    self.send("STOP")?;
                    stopped = true;
                }
                step(None)?;
            } else if line.starts_with("RESUME ") {
                self.note_resume(owner, &line);
            } else if line.starts_with("DONE") {
                self.parse_done(owner, &line);
            } else if line.starts_with("BADM ") {
                let f = py::split(&line);
                let w = f.get(2).copied();
                let r = (matches!(w, Some("1") | Some("2")), w == Some("2"));
                self.st().ctl_result = Some(r);
                return Ok(CtlEnd::Badm(r.0, r.1));
            } else if line.starts_with("YIELDED ") {
                let f = py::split(&line);
                if f.len() >= 3 && py::int_like(f[1]) && py::isdigit(f[2]) {
                    self.st().yielded = Some((f[1].parse().unwrap_or(0), f[2].parse().unwrap_or(0)));
                }
            } else if line.starts_with("ERR") {
                self.st().ctl_erred = true;
                return Err(Stop::Raised(request_err(py::strip(line.get(4..).unwrap_or("")), None)));
            }
            if line.starts_with("DONE") && self.st().ctl_mode == "solo" {
                self.st().ctl_done = true;
                return Ok(CtlEnd::Done);
            }
        }
    }

    /// Read control lines up to `until` (DONE or BADM); None if the engine ended or `born` is gone.
    fn drain_control(&self, owner: u64, until: &str, timeout: f64, born: Option<u64>) -> Option<String> {
        let end = Instant::now() + secs(timeout);
        while Instant::now() < end {
            if let Some(b) = born {
                if self.gen() != b || !self.alive() {
                    return None;
                }
            }
            let left = end.saturating_duration_since(Instant::now()).as_secs_f64().min(1.0).max(0.1);
            let line = match self.lines_get(Some(left)) {
                Got::Timeout => continue,
                Got::Eof => return None,
                Got::Line(l) => l,
            };
            if line.starts_with("DONE") {
                self.parse_done(owner, &line);
            }
            if line.starts_with(until) || line.starts_with("ERR") {
                return Some(line);
            }
        }
        None
    }

    fn hold(&self, s: &mut Slots, slot: usize, count: i64) {
        if slot < s.held.len() {
            s.held[slot] = count.max(0);
            s.epoch[slot] += 1;
        }
    }

    fn solo_kept(&self, slot: Option<usize>, run_size: i64, gen: &[Tok]) {
        let Some(slot) = slot else { return };
        let mut s = self.sl();
        let n = if gen.is_empty() { 0 } else { run_size + gen.len() as i64 - 1 };
        self.hold(&mut s, slot, n);
        if slot < s.used.len() {
            s.used[slot] = now_s();
        }
    }

    fn note_bdone(&self, slot: usize, f: &[&str]) {
        let known = f.len() >= 7 && py::isdigit(f[5]) && py::isdigit(f[6]);
        let v = (
            if known { f[5].parse().ok() } else { None },
            if known { f[6].parse().ok() } else { None },
            now_s(),
        );
        self.st().slot_drafts.insert(slot, v);
    }

    fn bdone_last(&self, owner: u64, f: &[String]) {
        if f.len() < 5 {
            return;
        }
        let prev = self.last_of(owner);
        let mut m: JsonMap = prev.map.iter()
            .filter(|(k, _)| !["drafts_accepted", "drafts_offered", "windows"].contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone())).collect();
        if f.len() >= 7 && py::isdigit(&f[5]) && py::isdigit(&f[6]) {
            m.insert("drafts_accepted".into(), json!(f[5].parse::<i64>().unwrap_or(0)));
            m.insert("drafts_offered".into(), json!(f[6].parse::<i64>().unwrap_or(0)));
        }
        if f.len() >= 8 && py::isdigit(&f[7]) {
            m.insert("windows".into(), json!(f[7].parse::<i64>().unwrap_or(0)));
        }
        m.insert("finish".into(), json!(f[3]));
        m.insert("decode_ms".into(), json!(py::float_of(&f[4]).unwrap_or(0.0)));
        let gen = if py::isdigit(&f[2]) { json!(f[2].parse::<i64>().unwrap_or(0)) } else { prev.map.get("generated").cloned().unwrap_or(Value::Null) };
        m.insert("generated".into(), gen);
        self.set_last(owner, new_last(m));
    }

    fn carry_drafts(&self, owner: u64, carry: &mut [i64; 3], fin: bool) {
        let get = |m: &JsonMap, k: &str| m.get(k).and_then(|v| v.as_i64());
        let mut c = *carry;
        self.edit_last(owner, |m| {
            let drafts = get(m, "drafts_offered").is_some() && get(m, "drafts_accepted").is_some();
            let windows = get(m, "windows").is_some();
            if fin {
                if drafts {
                    m.insert("drafts_accepted".into(), json!(get(m, "drafts_accepted").unwrap() + c[0]));
                    m.insert("drafts_offered".into(), json!(get(m, "drafts_offered").unwrap() + c[1]));
                }
                if windows {
                    m.insert("windows".into(), json!(get(m, "windows").unwrap() + c[2]));
                }
                c = [0, 0, 0];
            } else {
                if drafts {
                    c[0] += get(m, "drafts_accepted").unwrap();
                    c[1] += get(m, "drafts_offered").unwrap();
                }
                if windows {
                    c[2] += get(m, "windows").unwrap();
                }
            }
        });
        *carry = c;
    }

    fn note_first_reused(&self, owner: u64, first: &mut Vec<Value>, last0: u64) {
        let last = self.last_of(owner);
        if first.is_empty() && last.id != last0 {
            if let Some(v) = last.map.get("reused").filter(|v| !v.is_null()) {
                first.push(v.clone());
            }
        }
    }

    fn carry_run(&self, owner: u64, carry: &mut [i64; 3], first: &mut Vec<Value>, last0: u64, fin: bool) {
        self.note_first_reused(owner, first, last0);
        let last = self.last_of(owner);
        if fin && last.id != last0 {
            let rm = self.read_ms(owner);
            let f0 = first.first().cloned();
            self.edit_last(owner, |m| {
                if let Some(v) = f0 {
                    m.insert("reused".into(), v);
                }
                if rm > 0.0 {
                    m.insert("prompt_ms".into(), json!(rm));
                }
            });
            first.clear();
        }
        self.carry_drafts(owner, carry, fin);
    }

    fn resumes(slot: usize, meet: &[i64], size: i64) -> bool {
        0 < meet[slot] && meet[slot] < size
    }

    /// pick_slot (caller holds the slot lock).
    fn pick_slot(s: &Slots, meet: &[i64], size: i64, session: Option<&str>) -> Option<usize> {
        let free: Vec<usize> = s.order.iter().copied().filter(|&b| !s.busy[b]).collect();
        if free.is_empty() {
            return None;
        }
        let held_prefix = |b: usize| if 0 < meet[b] && meet[b] < size { meet[b] } else { 0 };
        let best = |pool: &[usize]| -> usize {
            // max() returns the first of equal maxima, min() likewise
            let mut top = pool[0];
            for &b in pool {
                if held_prefix(b) > held_prefix(top) {
                    top = b;
                }
            }
            if held_prefix(top) > 0 {
                return top;
            }
            let key = |b: usize| (s.held[b] > 0, s.used[b]);
            let mut lo = pool[0];
            for &b in pool {
                let (a, c) = (key(b), key(lo));
                if a.0 < c.0 || (a.0 == c.0 && a.1 < c.1) {
                    lo = b;
                }
            }
            lo
        };
        let mine: Vec<usize> = free.iter().copied()
            .filter(|b| session.is_some() && s.owner.get(b).map(|x| x.as_str()) == session).collect();
        if !mine.is_empty() {
            let top = best(&mine);
            if held_prefix(top) > 0 {
                return Some(top);
            }
        }
        let unowned: Vec<usize> = free.iter().copied().filter(|b| !s.owner.contains_key(b)).collect();
        if !unowned.is_empty() {
            return Some(best(&unowned));
        }
        Some(best(if mine.is_empty() { &free } else { &mine }))
    }

    fn claim_slot(s: &mut Slots, slot: usize, session: Option<&str>, owner: u64) {
        s.thread[slot] = Some(owner);
        if let Some(ses) = session {
            s.owner.insert(slot, ses.to_string());
        }
    }

    fn may_go_solo(&self, left: i64, times: usize, embeddings: bool) -> bool {
        let info = self.info();
        let t = |k: &str| info.get(k).map(logprobs::py_truthy).unwrap_or(false);
        if embeddings || times >= SOLO_AGAIN_MAX || left < SOLO_AGAIN_MIN_LEFT || !t("slot_cache")
            || (t("slot0_solo") && !t("solo_any_slot")) {
            return false;
        }
        let s = self.sl();
        s.busy.iter().filter(|b| **b).count() == 1 && s.waiting == 0
    }

    fn shorter_waiting(&self, plen: i64, free: usize) -> bool {
        let s = self.sl();
        s.busy.iter().filter(|b| !**b).count() >= free && s.wait_lens.iter().any(|(_, l)| l * 2 <= plen)
    }

    /// Hand the engine the prompt once; (handle, tokens).
    fn hold_prompt(&self, prompt: &Prompt, embeddings: bool) -> EResult<(i64, i64)> {
        let stopped = |me: &Self| EngineErr::Died(format!("the engine stopped unexpectedly (exit code {})",
            me.exit_code().map(|c| c.to_string()).unwrap_or_else(|| "None".into())));
        let line = {
            let _g = self.prompt_lock.lock().unwrap();
            let wrote = match prompt {
                Prompt::Text(t) => {
                    let body = t.text();
                    let spans = t.byte_spans();
                    let plain = if spans.is_empty() { String::new() } else {
                        format!(" plain={}", spans.iter().map(|(a, b)| format!("{a}:{b}")).collect::<Vec<_>>().join(","))
                    };
                    self.write_raw(&format!("PROMPT {}{}\n{}", body.len(), plain, body)).is_ok()
                }
                Prompt::Ids(ids) => {
                    let s = ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
                    self.write_raw(&format!("PROMPTI{} {}\n", if embeddings { " img" } else { "" }, s)).is_ok()
                }
            };
            if !wrote {
                return Err(stopped(self));
            }
            loop {
                let got = match self.io() {
                    Some(p) => recv(&p.prompt_q, Some(1.0)),
                    None => Got::Eof,
                };
                match got {
                    Got::Line(l) => break l,
                    Got::Eof => return Err(stopped(self)),
                    Got::Timeout => {
                        if !self.alive() {
                            return Err(stopped(self));
                        }
                    }
                }
            }
        };
        let f = py::split(&line);
        if f.get(1) == Some(&"ERR") {
            let err = py::strip(line.get("PROMPT ERR ".len()..).unwrap_or("")).to_string();
            let shown = match prompt {
                Prompt::Text(t) => Some(t.explain(&err)),
                _ => None,
            };
            return Err(request_err(&err, shown));
        }
        match (f.get(1).and_then(|x| x.parse().ok()), f.get(2).and_then(|x| x.parse().ok())) {
            (Some(h), Some(n)) => Ok((h, n)),
            _ => Err(EngineErr::Error(format!("the engine's reply to a prompt was not understood: {line}"))),
        }
    }

    fn meet(&self, owner: u64, handle: i64, born: u64) -> EResult<Vec<i64>> {
        self.send(&format!("PMATCH {handle}"))?;
        let line = self.drain_control(owner, "PMATCH", 300.0, Some(born)).ok_or_else(|| {
            EngineErr::Died(format!("the engine stopped unexpectedly (exit code {})",
                self.exit_code().map(|c| c.to_string()).unwrap_or_else(|| "None".into())))
        })?;
        if line.starts_with("ERR") {
            return Err(request_err(py::strip(line.get(4..).unwrap_or("")), None));
        }
        let batch = self.batch();
        let mut meet = vec![0i64; batch];
        let words: Vec<String> = py::split(&line).iter().skip(2).take(batch).map(|s| s.to_string()).collect();
        let mut s = self.sl();
        for (b, w) in words.iter().enumerate() {
            let parts: Vec<&str> = w.split(':').collect();
            let bad = || EngineErr::Error(format!("the engine's reply to a prompt was not understood: {line}"));
            if parts.len() != 2 {
                return Err(bad());
            }
            let (m, h) = match (parts[0].trim().parse::<i64>(), parts[1].trim().parse::<i64>()) {
                (Ok(m), Ok(h)) => (m, h),
                _ => return Err(bad()),
            };
            meet[b] = m;
            if b < s.busy.len() && !s.busy[b] {
                s.held[b] = h;
            }
        }
        Ok(meet)
    }

    /// BSTOP a slot whose consumer left and free it at BDONE.
    fn release_slot_when_done(self: &Arc<Self>, slot: usize, stream: Option<i64>, wait_s: f64) -> Option<Vec<String>> {
        let io = self.io()?;
        if slot >= io.slot_q.len() {
            return None;
        }
        let _ = self.send(&format!("BSTOP {slot}"));
        let sgen = io.gen;
        let tail = Arc::new(Mutex::new(stream.unwrap_or(0)));
        let drain = {
            let (io, tail, me) = (io.clone(), tail.clone(), self.clone());
            move |end: Instant| -> (&'static str, Option<Vec<String>>) {
                while Instant::now() < end {
                    let left = end.saturating_duration_since(Instant::now()).as_secs_f64().clamp(0.01, 5.0);
                    match recv(&io.slot_q[slot], Some(left)) {
                        Got::Timeout => continue,
                        Got::Eof => {
                            *tail.lock().unwrap() = 0;
                            return ("gone", None);
                        }
                        Got::Line(line) => {
                            if line.starts_with("BT ") {
                                *tail.lock().unwrap() += py::split(&line).len() as i64 - 2;
                            }
                            if line.starts_with("BDONE ") {
                                let f: Vec<String> = py::split(&line).iter().map(|s| s.to_string()).collect();
                                let fr: Vec<&str> = f.iter().map(|s| s.as_str()).collect();
                                me.note_bdone(slot, &fr);
                                return ("done", Some(f));
                            }
                        }
                    }
                }
                ("wait", None)
            }
        };
        let free = {
            let (tail, me) = (tail.clone(), self.clone());
            move || {
                let t = *tail.lock().unwrap();
                let mut s = me.sl();
                if s.gen == sgen {
                    let n = if stream.unwrap_or(0) != 0 && t != 0 { t - 1 } else { 0 };
                    me.hold(&mut s, slot, n);
                    s.busy[slot] = false;
                }
                me.slot_cv.notify_all();
            }
        };
        let (state, fields) = if wait_s > 0.0 { drain(Instant::now() + secs(wait_s)) } else { ("wait", None) };
        if state != "wait" {
            free();
            return fields;
        }
        std::thread::spawn(move || {
            drain(Instant::now() + secs(600.0));
            free();
        });
        None
    }

    // ---- solo generate
    #[allow(clippy::too_many_arguments)]
    fn generate_solo(self: &Arc<Self>, owner: u64, prompt: Prompt, max_new: i64, sampling: &Sampling, cancel: &Flag,
                     embeddings: Option<String>, stop_soon: Option<&Flag>, y: &Y) -> Result<(), Stop> {
        let text = match &prompt {
            Prompt::Text(t) => Some(t.clone()),
            _ => None,
        };
        {
            let mut st = self.st();
            st.progress = None;
            st.prefill_tok_s_mean = None;
        }
        self.reset_read(owner);
        let keys = sampling_keys(sampling);
        let wrote = match (&text, &prompt) {
            (Some(t), _) => {
                let body = t.text();
                let spans = t.byte_spans();
                let plain = if spans.is_empty() { String::new() } else {
                    format!(" plain={}", spans.iter().map(|(a, b)| format!("{a}:{b}")).collect::<Vec<_>>().join(","))
                };
                let clip = if t.clip() { " clip=1" } else { "" };
                self.write_raw(&format!("GENT {max_new}{keys}{clip} {}{plain}\n{body}", body.len()))
            }
            (None, Prompt::Ids(ids)) => {
                let head = match &embeddings {
                    Some(e) => format!("GENI {max_new}{keys} {e}"),
                    None => format!("GEN {max_new}{keys}"),
                };
                let s = ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
                self.write_raw(&format!("{head} {s}\n"))
            }
            _ => unreachable!(),
        };
        if wrote.is_err() {
            return Err(Stop::Raised(self.died()));
        }
        let silence = *self.silence_s.lock().unwrap();
        let plen = prompt.len() as i64;
        let mut allow = if silence > 0.0 { silence + plen.min(PP_CHUNK_MAX) as f64 / PP_FLOOR_TOK_S } else { 0.0 };
        let mut done = false;
        let mut heard = Instant::now();
        let mut read_to: i64 = 0;
        let mut beat = heard;
        let mut soft_sent = false;
        let can_stop = self.st().can_stop;
        let mut lp: Option<(u32, LogProb)> = None;
        let r: Result<(), Stop> = (|| {
            loop {
                if !soft_sent && stop_soon.map(|f| f.is_set()).unwrap_or(false) {
                    if !can_stop {
                        return Ok(());
                    }
                    let _ = self.write_raw("STOP\n");
                    soft_sent = true;
                }
                let mut wait = 10.0f64;
                if allow > 0.0 {
                    wait = wait.min(allow - heard.elapsed().as_secs_f64());
                    if wait <= 0.0 {
                        done = true;
                        let e = self.silent(&format!("the engine said nothing for {:.0} s during the request",
                                                     heard.elapsed().as_secs_f64()));
                        return Err(Stop::Raised(e));
                    }
                }
                let line = match self.lines_get(Some(wait.min(0.5))) {
                    Got::Timeout => {
                        if cancel.is_set() {
                            return Ok(());
                        }
                        if allow > 0.0 && heard.elapsed().as_secs_f64() >= allow {
                            continue;
                        }
                        if beat.elapsed().as_secs_f64() >= 10.0 {
                            beat = Instant::now();
                            y.yield_(None)?;
                        }
                        continue;
                    }
                    Got::Eof => {
                        done = true;
                        return Err(Stop::Raised(self.died()));
                    }
                    Got::Line(l) => l,
                };
                heard = Instant::now();
                beat = heard;
                if line.starts_with("P ") {
                    lp = logprobs::parse_line(&py::split(&line)[1..]);
                    continue;
                }
                if let Some(rest) = line.strip_prefix("T ") {
                    allow = silence;
                    if cancel.is_set() {
                        return Ok(());
                    }
                    let id = py::strip(rest).parse::<i64>().unwrap_or(0) as u32;
                    let t = logprobs::token(id, lp.as_ref());
                    lp = None;
                    y.yield_(Some(t))?;
                } else if line.starts_with("PT ") && text.is_some() {
                    let f = py::split(&line);
                    if f.len() >= 3 && py::isdigit(f[1]) && py::int_like(f[2]) {
                        let n: i64 = f[1].parse().unwrap_or(0);
                        text.as_ref().unwrap().set_counted(n, f[2].parse().unwrap_or(0));
                        allow = if silence > 0.0 { silence + n.min(PP_CHUNK_MAX) as f64 / PP_FLOOR_TOK_S } else { 0.0 };
                    }
                    y.yield_(None)?;
                } else if line.starts_with("PP ") {
                    let f = py::split(&line);
                    if f.len() >= 3 && py::isdigit(f[1]) && py::isdigit(f[2]) {
                        self.note_read(owner, &f);
                        let rate = if f.len() >= 5 { py::float_of(f[4]) } else { None };
                        self.st().prefill_tok_s_mean = rate;
                        let at: i64 = f[1].parse().unwrap_or(0);
                        let (rate, chunk) = (rate.unwrap_or(0.0), at - read_to);
                        read_to = at;
                        if silence > 0.0 && rate > 0.0 && chunk > 0 {
                            allow = silence.max(PP_SLACK * chunk as f64 / rate);
                        }
                    }
                    if cancel.is_set() {
                        return Ok(());
                    }
                    y.yield_(None)?;
                } else if line.starts_with("PL ") {
                    self.note_live(owner, &line);
                    if cancel.is_set() {
                        return Ok(());
                    }
                    y.yield_(None)?;
                } else if line.starts_with("RESUME ") {
                    self.note_resume(owner, &line);
                    if let Some(n) = py::split(&line).get(1).and_then(|x| x.parse::<i64>().ok()) {
                        read_to = n;
                    }
                } else if line.starts_with("DONE") {
                    self.parse_done(owner, &line);
                    done = true;
                    return Ok(());
                } else if line.starts_with("ERR") {
                    done = true;
                    let err = py::strip(line.get(4..).unwrap_or("")).to_string();
                    let shown = text.as_ref().map(|t| t.explain(&err));
                    return Err(Stop::Raised(request_err(&err, shown)));
                }
            }
        })();
        if !done {
            if can_stop {
                let _ = self.write_raw("STOP\n");
            }
            let mut heard = Instant::now();
            loop {
                let left = if allow > 0.0 { Some(allow - heard.elapsed().as_secs_f64()) } else { None };
                let got = match left {
                    Some(l) if l <= 0.0 => Got::Timeout,
                    _ => self.lines_get(left),
                };
                match got {
                    Got::Timeout => {
                        return Err(Stop::Raised(self.silent(&format!(
                            "the engine did not finish the request after it was stopped (STOP) within {:.0} s", allow))));
                    }
                    Got::Eof => break,
                    Got::Line(line) => {
                        if line.starts_with("ERR") {
                            break;
                        }
                        if line.starts_with("DONE") {
                            self.parse_done(owner, &line);
                            break;
                        }
                        if !can_stop {
                            heard = Instant::now();
                        }
                    }
                }
            }
        }
        r
    }
}

/// The batched generation's per-request state (generate_batched's locals and closures).
struct Run {
    prompt: Prompt,
    embeddings: Option<String>,
    session: Option<String>,
    n: i64,
    handle: i64,
    held_gen: u64,
    extra: Vec<u32>,
    out: Vec<Tok>,
    meet: Vec<i64>,
    epoch: Vec<u64>,
    keys: String,
}

impl Run {
    fn size(&self) -> i64 {
        self.n + self.extra.len() as i64
    }
    fn gref(&self) -> String {
        if self.extra.is_empty() {
            format!("#{}", self.handle)
        } else {
            format!("#{}+{}", self.handle, self.extra.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","))
        }
    }
    fn gen_keys(&self) -> String {
        if !self.out.is_empty() && self.keys.contains(" grammar=") {
            format!("{} gprompt={}", self.keys, self.n)
        } else {
            self.keys.clone()
        }
    }
    fn meets(&self, s: &Slots) -> Vec<i64> {
        self.meet.iter().zip(&self.epoch).enumerate()
            .map(|(b, (m, e))| if b < s.epoch.len() && *e == s.epoch[b] { *m } else { 0 }).collect()
    }
    fn note(&mut self, s: &Slots, slot: usize) {
        if slot < self.meet.len() {
            self.meet[slot] = s.held[slot];
            self.epoch[slot] = s.epoch[slot];
        }
    }
}

impl Inner {
    fn attach(&self, owner: u64, run: &mut Run) -> EResult<()> {
        if run.held_gen != self.gen() {
            let (h, _) = self.hold_prompt(&run.prompt, run.embeddings.is_some())?;
            run.handle = h;
            run.held_gen = self.gen();
            run.meet = vec![];
        }
        if run.meet.is_empty() {
            run.meet = self.meet(owner, run.handle, self.gen())?;
            run.epoch = self.sl().epoch.clone();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_batched(self: &Arc<Self>, owner: u64, prompt: Prompt, max_new: i64, sampling: &Sampling,
                        cancel: &Flag, embeddings: Option<String>, session: Option<String>,
                        stop_soon: Option<&Flag>, y: &Y) -> Result<(), Stop> {
        if !self.alive() {
            return Err(Stop::Raised(EngineErr::died("the engine is unavailable; this request was not sent")));
        }
        self.st().progress = None;
        self.reset_read(owner);
        self.reset_windows(owner);
        let (handle, n) = self.hold_prompt(&prompt, embeddings.is_some())?;
        let mut run = Run {
            prompt: prompt.clone(),
            embeddings,
            session,
            n,
            handle,
            held_gen: self.gen(),
            extra: vec![],
            out: vec![],
            meet: vec![],
            epoch: vec![],
            keys: sampling_keys(sampling),
        };
        let r = (|| -> Result<(), Stop> {
            let mut max_new = max_new;
            if let Prompt::Text(t) = &prompt {
                let ctx = self.st().max_context;
                max_new = t.settle(n, max_new, ctx).map_err(EngineErr::Value)?;
                y.yield_(None)?;
            }
            self.batched_body(owner, &mut run, max_new, cancel, stop_soon, y)
        })();
        if run.held_gen == self.gen() && self.alive() {
            let _ = self.send(&format!("PFREE {}", run.handle));
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_body(self: &Arc<Self>, owner: u64, run: &mut Run, max_new: i64, cancel: &Flag,
                    stop_soon: Option<&Flag>, y: &Y) -> Result<(), Stop> {
        let mut heartbeat = || y.yield_(None).map_err(Stop::from);
        let mut left = max_new;
        if !self.take_control(cancel, run.size(), None, &mut heartbeat)? {
            return Ok(());
        }
        // generate_batched's locals that its finally reads
        let mut holding = true;
        let mut born = self.gen();
        let mut slot: Option<usize> = None;
        let mut gen0: Option<usize> = None;
        let mut reserved: Option<usize> = None;
        let mut solo_slot: Option<usize> = None;
        let mut run0 = 0usize;
        let mut run_size = 0i64;
        let mut phase = "none";
        let mut stop_sent = false;
        let mut yields = 0usize;
        let mut solo_again = 0usize;
        let mut carry = [0i64; 3];
        let mut first: Vec<Value> = vec![];
        let last0 = self.last_of(owner).id;
        self.set_read_ms(owner, 0.0);
        let soft = || stop_soon.map(|f| f.is_set()).unwrap_or(false);

        let body = (|| -> Result<(), Stop> {
            self.attach(owner, run)?;
            loop {
                if !holding {
                    let mut hb = || y.yield_(None).map_err(Stop::from);
                    if !self.take_control(cancel, run.size(), None, &mut hb)? {
                        return Ok(());
                    }
                    holding = true;
                    born = self.gen();
                    self.attach(owner, run)?;
                }
                solo_slot = None;
                let alone = {
                    let mut s = self.sl();
                    let any_slot = self.info_flag("solo_any_slot");
                    let mut alone = !s.busy.iter().any(|b| *b) && s.waiting == 0 && (any_slot || !self.info_flag("slot0_solo"));
                    if alone && left > 1 && any_slot {
                        let m = run.meets(&s);
                        solo_slot = Self::pick_slot(&s, &m, run.size(), run.session.as_deref());
                        match solo_slot {
                            None => alone = false,
                            Some(b) => {
                                s.busy[b] = true;
                                Self::claim_slot(&mut s, b, run.session.as_deref(), owner);
                                reserved = Some(b);
                            }
                        }
                    }
                    alone
                };
                if alone && left > 1 {
                    let head = match (solo_slot, &run.embeddings) {
                        (Some(b), Some(e)) => format!("SGENI {b} {left}{} {e}", run.gen_keys()),
                        (Some(b), None) => format!("SGEN {b} {left}{}", run.gen_keys()),
                        (None, Some(e)) => format!("GENI {left}{} {e}", run.gen_keys()),
                        (None, None) => format!("GEN {left}{}", run.gen_keys()),
                    };
                    self.send(&format!("{head} {}", run.gref()))?;
                    run0 = run.out.len();
                    run_size = run.size();
                    let mut yield_sent = false;
                    if let Some(b) = solo_slot {
                        let mut s = self.sl();
                        self.hold(&mut s, b, 0);
                    }
                    phase = "solo";
                    {
                        let mut st = self.st();
                        st.ctl_mode = "solo";
                        st.ctl_result = None;
                        st.yielded = None;
                        st.ctl_erred = false;
                    }
                    let others = || self.sl().waiting > 0;
                    let mut step = |x: Option<Tok>| -> Result<(), Stop> {
                        match x {
                            Some(t) => {
                                run.out.push(t.clone());
                                y.yield_(Some(t))?;
                            }
                            None => {
                                if !yield_sent && run.out.is_empty() && run.embeddings.is_none() && yields < YIELDS_MAX
                                    && self.shorter_waiting(run.size(), if solo_slot.is_none() { 2 } else { 1 }) {
                                    if solo_slot.is_none() {
                                        let mut s = self.sl();
                                        let m = run.meets(&s);
                                        reserved = Self::pick_slot(&s, &m, run.size(), run.session.as_deref());
                                        if let Some(b) = reserved {
                                            s.busy[b] = true;
                                            Self::claim_slot(&mut s, b, run.session.as_deref(), owner);
                                        }
                                    }
                                    if let Some(b) = reserved {
                                        self.send(&format!("BYIELD {b}"))?;
                                        yield_sent = true;
                                    }
                                }
                                y.yield_(None)?;
                            }
                        }
                        Ok(())
                    };
                    self.control(owner, cancel, Some(&others), None, stop_soon, &mut step)?;
                    phase = "none";
                    self.note_first_reused(owner, &mut first, last0);
                    let yielded = self.st().yielded;
                    if yielded.is_some() && reserved.is_some() && yielded.unwrap().0 == reserved.unwrap() as i64 {
                        slot = reserved.take();
                    } else if let Some(b) = reserved {
                        self.solo_kept(solo_slot, run_size, &run.out[run0..]);
                        let mut s = self.sl();
                        if let Some(ss) = solo_slot {
                            run.note(&s, ss);
                        }
                        s.busy[b] = false;
                        self.slot_cv.notify_all();
                        reserved = None;
                    }
                    if slot.is_none() {
                        let finish = self.last_of(owner).map.get("finish").and_then(|v| v.as_str().map(|s| s.to_string()));
                        left = max_new - run.out.len() as i64;
                        let eos = run.out.last().map(|t| self.eos_ids().contains(&t.id)).unwrap_or(false);
                        if cancel.is_set() || soft() || left <= 0
                            || matches!(finish.as_deref(), Some("stop") | Some("length") | Some("error")) || eos {
                            self.carry_run(owner, &mut carry, &mut first, last0, true);
                            return Ok(());
                        }
                        self.carry_run(owner, &mut carry, &mut first, last0, false);
                        run.extra = run.out.iter().map(|t| t.id).collect();
                    }
                }
                let cont;
                loop {
                    if slot.is_none() {
                        let mut s = self.sl();
                        let ticket: Ticket = s.admission.join();
                        let got = loop {
                            let m = run.meets(&s);
                            let pick = Self::pick_slot(&s, &m, run.size(), run.session.as_deref());
                            if let Some(b) = pick {
                                let hit = Self::resumes(b, &m, run.size());
                                if s.admission.may_go(ticket, hit) {
                                    s.busy[b] = true;
                                    Self::claim_slot(&mut s, b, run.session.as_deref(), owner);
                                    break Some(b);
                                }
                            }
                            if holding {
                                self.ctl.release();
                                holding = false;
                                self.slot_cv.notify_all();
                            }
                            s = self.slot_cv.wait_timeout(s, Duration::from_secs(10)).unwrap().0;
                            if cancel.is_set() {
                                break None;
                            }
                        };
                        s.admission.leave(ticket);
                        self.slot_cv.notify_all();
                        drop(s);
                        match got {
                            Some(b) => slot = Some(b),
                            None => return Ok(()),
                        }
                    }
                    let b = slot.unwrap();
                    if !holding {
                        let mut hb = || y.yield_(None).map_err(Stop::from);
                        if !self.take_control(cancel, run.size(), None, &mut hb)? {
                            return Ok(());
                        }
                        holding = true;
                        born = self.gen();
                        self.attach(owner, run)?;
                    }
                    let yielded = self.st().yielded;
                    if let Some((_, read)) = yielded {
                        {
                            let mut s = self.sl();
                            self.hold(&mut s, b, read);
                            run.note(&s, b);
                        }
                        self.st().yielded = None;
                        yields += 1;
                        self.ctl.release();
                        holding = false;
                        let ce = self.sl().ctl_epoch;
                        let mut hb = || y.yield_(None).map_err(Stop::from);
                        if !self.take_control(cancel, run.size(), Some(ce), &mut hb)? {
                            return Ok(());
                        }
                        holding = true;
                        born = self.gen();
                        self.attach(owner, run)?;
                    }
                    let ctx = self.st().max_context;
                    left = left.min(ctx - run.size() - SLOT_MARGIN).max(1);
                    if let Some(io) = self.io() {
                        if b < io.slot_q.len() {
                            while try_recv(&io.slot_q[b]).is_some() {}
                        }
                    }
                    let head = match &run.embeddings {
                        Some(e) => format!("BGENI {b} {left}{} {e}", run.gen_keys()),
                        None => format!("BGEN {b} {left}{}", run.gen_keys()),
                    };
                    self.send(&format!("{head} {}", run.gref()))?;
                    {
                        let mut s = self.sl();
                        self.hold(&mut s, b, 0);
                    }
                    phase = "admit";
                    {
                        let mut st = self.st();
                        st.ctl_mode = "batch";
                        st.ctl_result = None;
                        st.yielded = None;
                        st.ctl_erred = false;
                    }
                    let asked = std::cell::Cell::new(false);
                    let size_now = run.size();
                    let emb = run.embeddings.is_some();
                    let ask_yield = || -> EResult<()> {
                        if !asked.get() && !emb && yields < YIELDS_MAX && self.shorter_waiting(size_now, 1) {
                            self.send(&format!("BYIELD {b}"))?;
                            asked.set(true);
                        }
                        Ok(())
                    };
                    let mut idle = || ask_yield();
                    let mut step = |x: Option<Tok>| -> Result<(), Stop> {
                        match x {
                            Some(t) => {
                                run.out.push(t.clone());
                                y.yield_(Some(t))?;
                            }
                            None => {
                                ask_yield()?;
                                y.yield_(None)?;
                            }
                        }
                        Ok(())
                    };
                    let end = self.control(owner, cancel, None, Some(&mut idle), None, &mut step)?;
                    self.note_first_reused(owner, &mut first, last0);
                    let c = matches!(end, CtlEnd::Badm(true, _));
                    phase = if c { "slot" } else { "none" };
                    let yielded = self.st().yielded;
                    if !c && yielded.map(|v| v.0 == b as i64).unwrap_or(false) && !cancel.is_set() {
                        continue;
                    }
                    cont = c;
                    break;
                }
                self.ctl.release();
                holding = false;
                if !cont {
                    return Ok(());
                }
                let b = slot.unwrap();
                let notok = matches!(self.st().ctl_result, Some((_, true)));
                gen0 = Some(if notok { run.out.len() } else { run.out.len().saturating_sub(1) });
                let mut going_solo = false;
                let mut slot_lp: Vec<Option<(u32, LogProb)>> = vec![];
                let io = self.io().ok_or_else(|| Stop::Raised(self.died()))?;
                loop {
                    if !stop_sent && !going_solo && soft() {
                        self.send(&format!("BSTOP {b}"))?;
                        stop_sent = true;
                    }
                    let line = match recv(&io.slot_q[b], Some(10.0)) {
                        Got::Timeout => {
                            if cancel.is_set() && !stop_sent {
                                self.send(&format!("BSTOP {b}"))?;
                                stop_sent = true;
                            }
                            y.yield_(None)?;
                            continue;
                        }
                        Got::Eof => {
                            phase = "none";
                            return Err(Stop::Raised(self.died()));
                        }
                        Got::Line(l) => l,
                    };
                    if line.starts_with("BP ") {
                        slot_lp.push(logprobs::parse_line(&py::split(&line)[2..]));
                        continue;
                    }
                    if line.starts_with("BT ") {
                        let f = py::split(&line);
                        self.count_window(owner, (f.len() - 2) as u64);
                        let lps = std::mem::take(&mut slot_lp);
                        for (k, w) in f[2..].iter().enumerate() {
                            let id = w.parse::<i64>().unwrap_or(0) as u32;
                            let t = logprobs::token(id, lps.get(k).and_then(|x| x.as_ref()));
                            run.out.push(t.clone());
                            if cancel.is_set() {
                                if !stop_sent {
                                    self.send(&format!("BSTOP {b}"))?;
                                    stop_sent = true;
                                }
                                continue;
                            }
                            y.yield_(Some(t))?;
                        }
                        if !going_solo && !stop_sent && !cancel.is_set()
                            && self.may_go_solo(max_new - run.out.len() as i64, solo_again, run.embeddings.is_some()) {
                            self.send(&format!("BSTOP {b}"))?;
                            going_solo = true;
                        }
                    } else if line.starts_with("BDONE ") {
                        phase = "none";
                        let f: Vec<String> = py::split(&line).iter().map(|s| s.to_string()).collect();
                        let fr: Vec<&str> = f.iter().map(|s| s.as_str()).collect();
                        self.note_bdone(b, &fr);
                        self.bdone_last(owner, &f);
                        {
                            let g0 = gen0.unwrap_or(0);
                            let mut s = self.sl();
                            let n = if run.out.len() > g0 { run.size() + run.out.len() as i64 - g0 as i64 - 1 } else { 0 };
                            self.hold(&mut s, b, n);
                            run.note(&s, b);
                        }
                        let eos = run.out.last().map(|t| self.eos_ids().contains(&t.id)).unwrap_or(false);
                        let again = going_solo && f.get(3).map(|s| s == "cancel").unwrap_or(false) && !cancel.is_set()
                            && (run.out.len() as i64) < max_new && !eos && !soft();
                        self.carry_run(owner, &mut carry, &mut first, last0, !again);
                        if again {
                            {
                                let mut s = self.sl();
                                s.used[b] = now_s();
                                s.busy[b] = false;
                                self.slot_cv.notify_all();
                            }
                            slot = None;
                            gen0 = None;
                            run.extra = run.out.iter().map(|t| t.id).collect();
                            left = max_new - run.out.len() as i64;
                            solo_again += 1;
                            break;
                        }
                        return Ok(());
                    }
                }
            }
        })();

        // finally: keep engine and server in step
        let fin = (|| -> EResult<()> {
            if (phase == "solo" || phase == "admit") && (self.gen() != born || !self.alive()) {
                phase = "none";
            }
            if (phase == "solo" || phase == "admit") && self.st().ctl_erred {
                phase = "none";
            }
            let drained = phase == "solo" || phase == "admit";
            if phase == "solo" {
                self.send("STOP")?;
                self.drain_control(owner, "DONE", 300.0, Some(born));
            } else if phase == "admit" {
                let line = self.drain_control(owner, "BADM", 300.0, Some(born));
                if let Some(l) = line {
                    let f = py::split(&l);
                    if l.starts_with("BADM ") && matches!(f.get(2).copied(), Some("1") | Some("2")) {
                        phase = "slot";
                    }
                }
            }
            if drained && phase != "slot" {
                self.carry_run(owner, &mut carry, &mut first, last0, true);
            }
            Ok(())
        })();
        if let Err(e) = fin {
            if !e.is_died() {
                return Err(Stop::Raised(e));
            }
        }
        if holding {
            self.ctl.release();
        }
        if let Some(rb) = reserved {
            if Some(rb) == solo_slot {
                self.solo_kept(solo_slot, run_size, &run.out[run0.min(run.out.len())..]);
            }
            let mut s = self.sl();
            if rb < s.busy.len() {
                s.busy[rb] = false;
            }
            self.slot_cv.notify_all();
        }
        if let Some(b) = slot {
            {
                let mut s = self.sl();
                if b < s.used.len() {
                    s.used[b] = now_s();
                }
            }
            if phase == "slot" {
                {
                    let mut s = self.sl();
                    self.hold(&mut s, b, 0);
                }
                let stream = match gen0 {
                    Some(g0) if run.out.len() > g0 => Some(run.size() + run.out.len() as i64 - g0 as i64),
                    _ => None,
                };
                let f = self.release_slot_when_done(b, stream, BDONE_WAIT_S);
                if let Some(f) = f {
                    self.bdone_last(owner, &f);
                    self.carry_run(owner, &mut carry, &mut first, last0, true);
                } else {
                    self.edit_last(owner, |m| {
                        for k in ["drafts_accepted", "drafts_offered", "windows"] {
                            m.shift_remove(k);
                        }
                    });
                }
            } else {
                let mut s = self.sl();
                if b < s.busy.len() {
                    s.busy[b] = false;
                }
                self.slot_cv.notify_all();
            }
        }
        body
    }

    fn encode(&self, command: &str, replies: usize, what: &str, timeout: Option<f64>) -> EResult<Vec<String>> {
        if !self.alive() {
            return Err(EngineErr::died("the engine is not running"));
        }
        let timeout = timeout.unwrap_or(ENC_TIMEOUT_S);
        let mut have_ctl = false;
        if self.batch() > 0 {
            let mut nop = || Ok(());
            match self.take_control(&Flag::new(), 1, None, &mut nop) {
                Ok(true) => have_ctl = true,
                Ok(false) => {}
                Err(Stop::Raised(e)) => return Err(e),
                Err(Stop::Closed) => {}
            }
        }
        let r = (|| {
            self.send(command)?;
            let deadline = Instant::now() + secs(timeout);
            let mut got = Vec::new();
            while got.len() < replies {
                let left = deadline.saturating_duration_since(Instant::now()).as_secs_f64().max(0.1);
                let line = match self.lines_get(Some(left)) {
                    Got::Timeout => return Err(EngineErr::Died(format!("the engine did not answer the {what} command"))),
                    Got::Eof => return Err(EngineErr::died("the engine ended")),
                    Got::Line(l) => py::strip(&l).to_string(),
                };
                if line.starts_with("ENC ERR") {
                    let rest = py::strip(line.get(8..).unwrap_or(""));
                    return Err(EngineErr::Value(format!("the {what} could not be read: {}",
                        if rest.is_empty() { "the engine refused it" } else { rest })));
                }
                if line.starts_with("ENC OK ") {
                    got.push(line[4..].to_string());
                } else if Instant::now() > deadline {
                    return Err(EngineErr::Died(format!("the engine did not answer the {what} command")));
                }
            }
            Ok(got)
        })();
        if have_ctl {
            self.ctl.release();
        }
        r
    }
}

impl Engine for ProcessEngine {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn generate(&self, prompt: Prompt, max_new: i64, sampling: &Sampling, cancel: Flag, embeddings: Option<String>,
                session: Option<String>, stop_soon: Option<Flag>) -> GenStream {
        let me = self.inner.clone();
        let sampling = sampling.clone();
        let owner = thread_ident();
        Gen::new(move |y: &Y| {
            if !me.alive() {
                return Err(Stop::Raised(EngineErr::died("the engine is unavailable; this request was not sent")));
            }
            let mut prompt = prompt;
            let mut max_new = max_new;
            if let Prompt::Text(t) = &prompt {
                if embeddings.is_some() || !me.info_flag("text_prompt") {
                    let ctx = me.st().max_context;
                    let (ids, m) = t.resolve(max_new, ctx).map_err(EngineErr::Value)?;
                    prompt = Prompt::Ids(ids);
                    max_new = m;
                }
            }
            if me.batch() > 0 {
                return me.generate_batched(owner, prompt, max_new, &sampling, &cancel, embeddings, session,
                                           stop_soon.as_ref(), y);
            }
            me.generate_solo(owner, prompt, max_new, &sampling, &cancel, embeddings, stop_soon.as_ref(), y)
        })
    }
    fn supports_stop_soon(&self) -> bool {
        true
    }
    fn takes_text(&self) -> bool {
        true
    }
    fn max_context(&self) -> i64 {
        self.inner.st().max_context
    }
    fn known_ctx(&self) -> i64 {
        self.inner.st().known_ctx
    }
    fn starting(&self) -> bool {
        self.inner.st().starting
    }
    fn batch(&self) -> usize {
        self.inner.batch()
    }
    fn gen(&self) -> u64 {
        self.inner.gen()
    }
    fn last(&self) -> Option<Last> {
        self.last_for(thread_ident())
    }
    fn last_for(&self, tid: u64) -> Option<Last> {
        Some(self.inner.last_of(tid))
    }
    fn prompt_progress_for(&self, tid: u64) -> Option<JsonMap> {
        Some(self.inner.tl_with(tid, |t| t.ptl.clone().unwrap_or_default()))
    }
    fn window_stats_for(&self, tid: u64) -> Option<(u64, u64)> {
        Some(self.inner.tl_with(tid, |t| (t.windows, t.tokens)))
    }
    fn progress(&self) -> Option<(i64, i64)> {
        self.inner.st().progress
    }
    fn prompt_progress(&self) -> Option<JsonMap> {
        self.prompt_progress_for(thread_ident())
    }
    fn prefill_tok_s_mean(&self) -> Option<f64> {
        self.inner.st().prefill_tok_s_mean
    }
    fn info(&self) -> Option<JsonMap> {
        Some(self.inner.info())
    }
    fn eos_ids(&self) -> Vec<u32> {
        self.inner.eos_ids()
    }
    fn alive(&self) -> bool {
        self.inner.alive()
    }
    fn has_alive(&self) -> bool {
        true
    }
    fn asleep(&self) -> bool {
        self.inner.st().asleep
    }
    fn unloaded(&self) -> bool {
        self.inner.st().unloaded
    }
    fn loading(&self) -> bool {
        self.inner.st().loading
    }
    fn load_status(&self) -> Option<JsonMap> {
        Some(self.inner.st().load_status.clone())
    }
    fn set_load_status(&self, status: JsonMap) {
        self.inner.st().load_status = status;
    }
    fn model_path(&self) -> Option<String> {
        Some(self.inner.model_path.clone())
    }
    fn log_path(&self) -> Option<String> {
        self.inner.log_path.clone()
    }
    fn spawn_args(&self) -> Vec<String> {
        self.inner.spawn.args.clone()
    }
    fn conversation(&self) -> JsonMap {
        self.inner.st().conversation.clone()
    }
    fn window_stats(&self) -> Option<(u64, u64)> {
        self.window_stats_for(thread_ident())
    }
    fn slot_view(&self) -> Option<Vec<JsonMap>> {
        if self.inner.batch() == 0 {
            return Some(vec![]);
        }
        let s = self.inner.sl();
        Some((0..s.busy.len()).map(|b| {
            let mut m = Map::new();
            m.insert("owner".into(), if s.busy[b] { s.thread[b].map(Value::from).unwrap_or(Value::Null) } else { Value::Null });
            m.insert("held".into(), json!(s.held[b]));
            m
        }).collect())
    }
    fn slot_drafts(&self) -> SlotDrafts {
        self.inner.st().slot_drafts.clone()
    }
    fn console(&self) -> Option<Arc<ConsoleLines>> {
        // Python: a fresh ConsoleLines per started process; before the first start there is none
        self.inner.io().map(|p| p.console.clone())
    }
    fn queue_depth(&self) -> Option<usize> {
        let s = self.inner.sl();
        Some(s.waiting + s.admission.len())
    }
    fn ctl(&self) -> Option<&PyLock> {
        Some(&self.inner.ctl)
    }
    fn pid(&self) -> Option<u32> {
        let st = self.inner.st();
        if st.proc_set { st.io.as_ref().map(|p| p.pid) } else { None }
    }
    fn has_process(&self) -> bool {
        true
    }
    fn death_note(&self) -> String {
        self.inner.death_note()
    }
    fn exit_code(&self) -> Option<i64> {
        self.inner.exit_code()
    }
    fn restart(&self, tries: Option<u32>) -> EResult<()> {
        self.inner.restart(tries.unwrap_or(3))
    }
    fn unload(&self) -> EResult<()> {
        self.inner.close()?;
        self.inner.st().unloaded = true;
        Ok(())
    }
    fn close(&self) -> EResult<()> {
        self.inner.close()
    }
    fn kill(&self) {
        if let Some(p) = self.inner.io() {
            p.kill();
        }
    }
    fn can_sleep(&self) -> bool {
        suspend::available()
    }
    fn sleep(&self, ram_check: Option<RamCheck>) -> EResult<JsonMap> {
        let me = &self.inner;
        let _g = me.sleep_lock.lock().unwrap();
        let out = me.vram_command("SLEEP", 120.0)?;
        let mut why = ram_check.and_then(|f| f());
        let mut held = None;
        if why.is_none() {
            let pid = self.pid().unwrap_or(0);
            held = crate::util::vram::process_by_gpu(pid, None).map(|v| v.into_iter().collect::<BTreeMap<u32, u64>>());
            why = suspend::suspend(pid);
        }
        if let Some(w) = why {
            me.vram_command("WAKE", 120.0)?;
            return Err(EngineErr::Value(w));
        }
        let mut st = me.st();
        st.parked = held;
        st.asleep = true;
        Ok(out)
    }
    fn wake(&self, vram_check: Option<VramCheck>) -> EResult<JsonMap> {
        let me = &self.inner;
        let _g = me.sleep_lock.lock().unwrap();
        let parked = me.st().parked.clone();
        if let Some(why) = vram_check.and_then(|f| f(parked.as_ref())) {
            return Err(EngineErr::WakeBlocked(why));
        }
        let t = Instant::now();
        let pid = self.pid().unwrap_or(0);
        let mut why = suspend::resume(pid);
        me.st().restore_s = t.elapsed().as_secs_f64();
        if why.is_none() {
            me.st().asleep = false;
            match me.vram_command("WAKE", 120.0) {
                Ok(v) => return Ok(v),
                Err(EngineErr::Value(e)) => why = Some(e),
                Err(e) => return Err(e),
            }
        }
        me.st().asleep = false;
        let w = why.unwrap_or_default();
        let _ = me.silent(&w);
        Err(EngineErr::Died(w))
    }
    fn restore_s(&self) -> f64 {
        self.inner.st().restore_s
    }
    fn parked(&self) -> Option<BTreeMap<u32, u64>> {
        self.inner.st().parked.clone()
    }
    fn vram(&self, reserve_mib: Option<i64>) -> EResult<JsonMap> {
        let cmd = match reserve_mib {
            None => "VRAM".to_string(),
            Some(r) => format!("VRAM {r}"),
        };
        self.inner.vram_command(&cmd, 120.0)
    }
    fn encode_images(&self, jobs: &[(String, String)], timeout: Option<f64>) -> EResult<Vec<String>> {
        let cmd = format!("ENC {}", jobs.iter().map(|(a, b)| format!("{a} {b}")).collect::<Vec<_>>().join(" "));
        self.inner.encode(&cmd, jobs.len(), "image", timeout)
    }
    fn encode_video(&self, video: &str, out: &str, words: &str, timeout: Option<f64>) -> EResult<String> {
        let cmd = format!("ENCV {video} {out} {words}").trim_end().to_string();
        Ok(self.inner.encode(&cmd, 1, "video", timeout)?.remove(0))
    }
    fn video(&self) -> Option<String> {
        self.inner.st().video.clone()
    }
}

impl ProcessEngine {
    /// ProcessEngine.silence_s (engine_silence_s; 0 waits forever)
    pub fn set_silence_s(&self, s: f64) {
        *self.inner.silence_s.lock().unwrap() = s;
    }
    pub fn info_mut(&self, f: impl FnOnce(&mut JsonMap)) {
        f(&mut self.inner.st().info);
    }
    pub fn log_ready_pos(&self) -> u64 {
        self.inner.st().log_ready_pos
    }
    pub fn log_start(&self) -> u64 {
        self.inner.st().log_start
    }
    pub fn ended(&self) -> bool {
        self.inner.st().ended
    }
}

#[allow(dead_code)]
fn _unused(_: &PromptText) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys() {
        let mut s = Sampling::default();
        s.map.insert("temperature".into(), json!(0));
        s.map.insert("top_p".into(), json!(0.95));
        s.map.insert("top_k".into(), json!(20.0));
        s.map.insert("presence_penalty".into(), json!(1.5));
        s.map.insert("seed".into(), json!(7));
        s.map.insert("relaxed_drafting".into(), json!(false));
        s.map.insert("experimental_speed_projection".into(), json!(true));
        assert_eq!(sampling_keys(&s), " top_p=0.95 top_k=20 penalty_present=1.5 penalty_last_n=64 seed=7 relax=0 cvec=1");
        s.map.insert("temperature".into(), json!(1));
        assert!(sampling_keys(&s).starts_with(" temperature=1.0 "));
    }

    #[test]
    fn exit_status() {
        assert_eq!(format_exit_status(Some(0xC0000005)), "the engine crashed (0xC0000005 STATUS_ACCESS_VIOLATION)");
        assert_eq!(format_exit_status(Some(3)), "the engine exited (code 3)");
        assert_eq!(format_exit_status(None), "the engine ended (no exit status)");
    }

    #[test]
    fn facts() {
        let f = info_facts("INFO batch_slots=2 engine=0.5.5 neg=-3\n");
        assert_eq!(f["batch_slots"], json!(2));
        assert_eq!(f["engine"], json!("0.5.5"));
        assert_eq!(f["neg"], json!(-3));
        let c = conversation_facts("CONV parked=1 bytes=2 restored_ms=1.5");
        assert_eq!(c["restored_ms"], json!(1.5));
        assert_eq!(engine_line("a\r\n"), "a\n");
    }
}
