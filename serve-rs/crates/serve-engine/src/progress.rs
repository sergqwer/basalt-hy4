//! serve/engine/progress.py: engine start narration from its log: load bars, warnings, errors and start-failure hints.
use crate::pyfmt::{self, fmt_f, ljust, rjust};
use crate::util::console::tagged;
use crate::util::loglevel::{engine_line, ERR, WARN};
use regex::Regex;
use serde_json::{Map, Value};
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const DRAFT_HEAD_FAIL: &str = "the draft head does not fit";
pub const DRAFT_HEAD_HINT: &str = "free VRAM, or run without the drafter (--spec 0)";

const GIB: f64 = (1u64 << 30) as f64;

/// Python's threading.Event.
#[derive(Default)]
pub struct Event {
    flag: Mutex<bool>,
    cv: Condvar,
}

impl Event {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self) {
        *self.flag.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.cv.notify_all();
    }

    pub fn clear(&self) {
        *self.flag.lock().unwrap_or_else(|e| e.into_inner()) = false;
    }

    pub fn is_set(&self) -> bool {
        *self.flag.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wait up to `timeout_s` seconds for the flag; returns it.
    pub fn wait(&self, timeout_s: f64) -> bool {
        let g = self.flag.lock().unwrap_or_else(|e| e.into_inner());
        let (g, _) = self.cv.wait_timeout_while(g, Duration::from_secs_f64(timeout_s.max(0.0)), |f| !*f)
            .unwrap_or_else(|e| e.into_inner());
        *g
    }
}

/// `time.time()`.
fn wall() -> f64 {
    crate::util::logbuf::now_s()
}

/// `time.monotonic()`.
fn mono() -> f64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_secs_f64() + 1000.0
}

fn read_from(log: &str, offset: u64) -> Option<String> {
    let mut f = std::fs::File::open(log).ok()?;
    f.seek(SeekFrom::Start(offset)).ok()?;
    let mut b = Vec::new();
    f.read_to_end(&mut b).ok()?;
    Some(String::from_utf8_lossy(&b).into_owned())
}

/// Advice when the start failed because the MTP draft head did not fit the free VRAM; "" for other failures.
pub fn start_failure_hint(log: Option<&str>, offset: u64) -> String {
    let Some(log) = log.filter(|l| !l.is_empty()) else { return String::new() };
    let Some(text) = read_from(log, offset) else { return String::new() };
    if !text.contains(DRAFT_HEAD_FAIL) {
        return String::new();
    }
    let said: Vec<String> = pyfmt::splitlines(&text).into_iter()
        .filter_map(|x| engine_line(pyfmt::strip(x)))
        .filter(|e| e.tag == "basalt mtp" && (e.text.contains("draft head over") || e.text.contains("hint:")))
        .map(|e| e.text).collect();
    format!(". mtp: {DRAFT_HEAD_FAIL}. {}", if said.is_empty() { format!("Hint: {DRAFT_HEAD_HINT}") } else { said.join(" ") })
}

/// The last meaningful log line of this start, without its prefix; "" when none.
pub fn start_last_line(log: Option<&str>, offset: u64) -> String {
    let Some(log) = log.filter(|l| !l.is_empty()) else { return String::new() };
    let got = (|| -> Option<(u64, String)> {
        let mut f = std::fs::File::open(log).ok()?;
        let end = f.seek(SeekFrom::End(0)).ok()?;
        let start = offset.max(end.saturating_sub(64 * 1024));     // skip earlier starts' lines
        f.seek(SeekFrom::Start(start)).ok()?;
        let mut b = Vec::new();
        f.read_to_end(&mut b).ok()?;
        Some((start, String::from_utf8_lossy(&b).into_owned()))
    })();
    let Some((start, text)) = got else { return String::new() };
    let lines = pyfmt::splitlines(&text);
    let skip = if start > offset { 1 } else { 0 };     // not a line cut in half
    for x in lines.iter().skip(skip).rev() {
        let x = pyfmt::strip(x);
        if x.is_empty() {
            continue;
        }
        match engine_line(x) {
            None => return x.to_string(),
            Some(e) if e.level != WARN => return e.text,
            _ => {}
        }
    }
    String::new()
}

/// Where the bars print: Python's `out` stream (sys.stdout by default).
pub trait Out: Send + Sync {
    fn write(&self, text: &str);
    fn flush(&self) {}
    fn isatty(&self) -> bool;
    /// The terminal's columns (`os.get_terminal_size(out.fileno())`); None when it has none.
    fn columns(&self) -> Option<i64> {
        None
    }
}

/// The process's stdout, through util::stdout.
pub struct Stdout;

impl Out for Stdout {
    fn write(&self, text: &str) {
        crate::util::stdout::write_raw_stdout(text);
    }
    fn flush(&self) {
        crate::util::stdout::flush_stdout();
    }
    fn isatty(&self) -> bool {
        crate::util::stdout::stdout_is_tty()
    }
    fn columns(&self) -> Option<i64> {
        crate::util::stdout::stdout_columns()
    }
}

/// Text kept in memory (tests): a terminal of `cols` columns when `tty`.
#[derive(Default)]
pub struct Capture {
    pub text: Mutex<String>,
    pub writes: Mutex<Vec<(f64, String)>>,
    pub tty: bool,
    pub cols: Option<i64>,
}

impl Out for Capture {
    fn write(&self, text: &str) {
        self.text.lock().unwrap().push_str(text);
        self.writes.lock().unwrap().push((mono(), text.to_string()));
    }
    fn isatty(&self) -> bool {
        self.tty
    }
    fn columns(&self) -> Option<i64> {
        self.cols
    }
}

#[derive(Debug, Clone, Default)]
struct Row {
    have: i64,
    plan: i64,
    start: Option<i64>,
    moves: Vec<(f64, i64)>,
    phase: Option<(f64, i64)>,
    moved: Option<f64>,
    cadence: Option<f64>,
}

#[derive(Debug, Clone)]
struct Share {
    done: i64,
    total: i64,
    reported: bool,
}

struct Bars {
    gpus: Vec<i64>,
    tty: bool,
    total: i64,
    into: String,
    into_pad: usize,
    shares: Vec<(String, Share)>,
    rows: Vec<(String, Row)>,
    beat_seen: Vec<(String, i64)>,
    beat_t: Option<f64>,
    started: i64,
    t0: Option<f64>,
    drawn: usize,
    shown: Vec<String>,
    kv: Vec<(String, i64, i64, String, i64)>,
}

/// The load's progress bar: one row over every byte the engine placed against the total it announced, labelled on a
/// terminal with the destination it places into now; the n-gram table lock is its own row. finish() collapses the bars
/// into one result line split by destination. Off a terminal rows print once per heartbeat while their bytes move, and
/// each destination prints one line when its last byte lands.
pub struct LoadBars {
    out: Arc<dyn Out>,
    st: Mutex<Bars>,
}

pub const WIDTH: usize = 24;
pub const WEIGHTS: &str = "weights";                // every model byte placed: the one bar
pub const TABLE: &str = "ngram";                    // the n-gram table locked into RAM: its own row, outside the total
const STILL: f64 = 1.0;                             // s without new bytes: the stretch of movement ended
const WINDOW: f64 = 1.0;                            // s floor of the window a moving row's rate covers
const MIN_SPAN: f64 = 0.5;                          // s: a shorter span gives no rate
const STALL: f64 = 2.0;                             // x the row's line interval: past it the row stood still
const CADENCE_KEEP: f64 = 0.9;                      // decay of the row's line interval: a maximum that forgets

fn label_name(target: &str) -> &str {
    match target {
        WEIGHTS => "loading model weights",
        TABLE => "loading the n-gram table into system ram",
        t => t,
    }
}

fn part(done: i64, total: i64) -> String {
    format!("{}/{}", fmt_f(done as f64 / GIB, 1), fmt_f(total as f64 / GIB, 1))
}

fn bar(done: i64, total: i64, width: usize) -> String {
    let f = if total > 0 { (done as f64 / total as f64).min(1.0) } else { 0.0 };
    let n = (f * width as f64) as usize;
    format!("{}{}{}", "=".repeat(n), if n < width { ">" } else { "" }, " ".repeat(width.saturating_sub(n + 1)))
}

fn rate_text(bytes: i64, span: f64) -> String {
    format!("  {} MB/s", fmt_f(bytes as f64 / 1e6 / span, 0))
}

fn mb_s_tail() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"  \d+ MB/s\n?\z").unwrap())
}

impl Bars {
    fn key(&self, target: &str) -> String {
        if let Some(n) = target.strip_prefix("cuda") {
            if pyfmt::is_digits(n) {
                if let Some(i) = pyfmt::py_int(n) {
                    if i >= 0 && (i as usize) < self.gpus.len() {
                        return format!("cuda{}", self.gpus[i as usize]);
                    }
                }
            }
        }
        target.to_string()
    }

    fn share(&self, t: &str) -> Option<&Share> {
        self.shares.iter().find(|(k, _)| k == t).map(|x| &x.1)
    }

    fn row(&self, t: &str) -> Option<&Row> {
        self.rows.iter().find(|(k, _)| k == t).map(|x| &x.1)
    }

    fn row_mut(&mut self, t: &str) -> &mut Row {
        if !self.rows.iter().any(|(k, _)| k == t) {
            self.rows.push((t.to_string(), Row::default()));
        }
        &mut self.rows.iter_mut().find(|(k, _)| k == t).unwrap().1
    }

    fn label(&self, target: &str, staged: bool) -> String {
        let name = label_name(target);
        if target != WEIGHTS || self.into.is_empty() || !staged {
            return name.to_string();
        }
        let head = format!("{name}: {}", ljust(&self.key(&self.into), self.into_pad));
        if self.shares.is_empty() {
            return head;
        }
        let width = self.shares.iter().map(|(_, s)| part(s.total, s.total).len()).max().unwrap_or(0);
        let text = self.share(&self.into).map(|s| part(s.done, s.total)).unwrap_or_default();
        format!("{head} {}", rjust(&text, width))
    }

    fn split(&self, done: i64) -> String {
        if self.shares.is_empty() {
            return String::new();
        }
        let parts: Vec<(&str, i64)> = self.shares.iter().map(|(t, s)| (t.as_str(), s.total)).collect();
        let tenths: Vec<f64> = parts.iter().map(|(_, b)| *b as f64 / GIB * 10.0).collect();
        let mut units: Vec<i64> = tenths.iter().map(|x| x.floor() as i64).collect();
        if parts.iter().map(|p| p.1).sum::<i64>() == done {       // largest remainders take the tenths the floors left
            let left = pyfmt::round_int(done as f64 / GIB * 10.0) - units.iter().sum::<i64>();
            let mut order: Vec<usize> = (0..parts.len()).collect();
            order.sort_by(|a, b| (units[*a] as f64 - tenths[*a]).partial_cmp(&(units[*b] as f64 - tenths[*b]))
                .unwrap_or(std::cmp::Ordering::Equal));
            for i in order.into_iter().take(left.max(0) as usize) {
                units[i] += 1;
            }
        } else {                                                  // a miscounted start: each part as it is
            units = tenths.iter().map(|x| pyfmt::round_int(*x)).collect();
        }
        parts.iter().zip(&units).map(|((t, _), u)| format!("{}={}GiB", self.key(t), fmt_f(*u as f64 / 10.0, 1)))
            .collect::<Vec<_>>().join(" ")
    }

    fn still_for(t: &Row) -> f64 {
        match t.cadence {
            None => STILL,
            Some(c) if c == 0.0 => STILL,
            Some(c) => STILL.max(STALL * c),
        }
    }

    fn sample(t: &mut Row, before: i64, now: f64) {
        let have = t.have;
        if have == before {
            return;
        }
        let gap = t.moved.map(|m| now - m).unwrap_or(0.0);
        t.moved = Some(now);
        let drought = t.cadence.is_some() && gap > Self::still_for(t);    // the row stood still between reads
        if gap >= MIN_SPAN {
            t.cadence = Some(match t.cadence {
                Some(c) if c != 0.0 => (CADENCE_KEEP * c).max(gap),
                _ => gap,
            });
        }
        if t.phase.is_none() {                     // the row's first bytes: its rate counts from here
            t.phase = Some((now, have));
            t.moves.clear();
        } else if drought {                        // moving again after standing still: the stretch is new
            t.phase = Some((now, have));
        }
        t.moves.push((now, have));
        let win = WINDOW.max(t.cadence.unwrap_or(0.0));
        while t.moves.len() > 2 && t.moves[1].0 <= now - win {
            t.moves.remove(0);
        }
    }

    fn rate(t: &Row, now: f64) -> String {
        let have = t.have;
        let Some(moved) = t.moved else { return String::new() };
        let (since, base);
        if t.plan > 0 && have >= t.plan && t.phase.is_some() {        // complete: its whole stretch's rate
            (since, base) = t.phase.unwrap();
        } else if now - moved <= Self::still_for(t) {                 // moving: the latest window, else the stretch's mean
            let first = t.moves[0];
            if moved - first.0 >= MIN_SPAN && first.1 < have {
                (since, base) = first;
            } else if t.phase.is_some_and(|p| moved > p.0) {
                (since, base) = t.phase.unwrap();
            } else {
                return String::new();
            }
        } else {
            return String::new();
        }
        let span = moved - since;
        if span < MIN_SPAN || have <= base {
            return String::new();
        }
        rate_text(have - base, span)
    }

    fn beat_rate(have: i64, now: f64, beat: Option<(f64, i64)>) -> String {
        let Some((since, base)) = beat else { return String::new() };
        if now <= since || have <= base {
            return String::new();
        }
        rate_text(have - base, now - since)
    }

    fn done(&self) -> i64 {
        self.row(WEIGHTS).map(|r| r.have).unwrap_or(0)
    }

    fn rows_text(&self, bar_w: usize, beat: Option<(f64, &[(String, i64)])>) -> Vec<(String, String)> {
        let mut started: Vec<&(String, Row)> = self.rows.iter().filter(|(_, r)| r.start.is_some()).collect();
        started.sort_by_key(|(_, r)| r.start);
        let now = mono();
        let mut rows = Vec::new();
        for (target, t) in started {
            let have = t.have;
            let plan = if target == WEIGHTS && self.total != 0 { self.total } else { t.plan };
            let mut rate = Self::rate(t, now);
            if rate.is_empty() {
                if let Some((bt, seen)) = beat {
                    let base = seen.iter().find(|(k, _)| k == target).map(|x| x.1).unwrap_or(have);
                    rate = Self::beat_rate(have, now, Some((bt, base)));
                }
            }
            rows.push((target.clone(), format!("[server] {}", self.label(target, self.tty)), have, plan, rate));
        }
        let pad = rows.iter().map(|r| pyfmt::char_len(&r.1)).max().unwrap_or(0);
        rows.into_iter().map(|(target, label, have, plan, rate)| {
            let text = if plan <= 0 {                    // no total yet: a count, not a bar
                format!("{}  {:5.1} GiB{rate}", ljust(&label, pad), have as f64 / GIB)
            } else {
                let plan = plan.max(have);
                format!("{} [{}] {:5.1}/{} GiB{rate}", ljust(&label, pad), bar(have, plan, bar_w), have as f64 / GIB,
                        fmt_f(plan as f64 / GIB, 1))
            };
            (target, text)
        }).collect()
    }

    fn lines(&self) -> Vec<String> {
        self.rows_text(WIDTH, None).into_iter().map(|x| x.1).collect()
    }

    fn fitted(&self, out: &dyn Out) -> Vec<String> {
        let rows = self.lines();
        let width = out.columns().map(|c| c - 1).unwrap_or(0);
        if width <= 0 || rows.iter().all(|r| pyfmt::char_len(r) as i64 <= width) {
            return rows;
        }
        let longest = rows.iter().map(|r| pyfmt::char_len(r) as i64).max().unwrap_or(0);
        let bar_w = 8.max(WIDTH as i64 - (longest - width)) as usize;
        self.rows_text(bar_w, None).into_iter().map(|(_, r)| {
            if pyfmt::char_len(&r) as i64 <= width {
                r
            } else {
                pyfmt::head_chars(&mb_s_tail().replace(&r, ""), width as usize).to_string()
            }
        }).collect()
    }

    fn erase(&mut self, out: &dyn Out) {
        if self.drawn > 0 {
            out.write(&"\x1b[1A\x1b[2K".repeat(self.drawn));
            self.drawn = 0;
        }
    }

    fn draw(&mut self, out: &dyn Out) {
        if self.tty {
            let rows = self.fitted(out);
            for r in &rows {
                out.write(&format!("\x1b[2K{r}\n"));
            }
            self.drawn = rows.len();
            self.shown = rows;
        }
        out.flush();
    }

    fn say(&mut self, out: &dyn Out, text: &str) {
        self.erase(out);
        out.write(&format!("{text}\n"));
        self.draw(out);
    }

    fn kv_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut targets: Vec<&str> = Vec::new();
        for x in &self.kv {
            if !targets.contains(&x.0.as_str()) {
                targets.push(&x.0);
            }
        }
        for target in targets {
            let mine: Vec<&(String, i64, i64, String, i64)> = self.kv.iter().filter(|x| x.0 == target).collect();
            let total: i64 = mine.iter().map(|x| x.1).sum();
            let peak: i64 = mine.iter().map(|x| x.4).sum();
            let solo: Vec<i64> = mine.iter().filter(|x| x.3.is_empty()).map(|x| x.2).collect();
            let slots: Vec<i64> = mine.iter().filter(|x| x.3 == "slot").map(|x| x.2).collect();
            let draft: i64 = mine.iter().filter(|x| x.3 == "draft").map(|x| x.1).sum();
            let name = self.key(target);
            let gib = |b: i64| fmt_f(b as f64 / GIB, 1);
            if peak > total {
                let mut tags = vec!["elastic".to_string()];
                if !slots.is_empty() {
                    let solo_t = solo.first().map(|s| format!("solo={s} ")).unwrap_or_default();
                    tags.push(format!("{solo_t}slots={}", slots.len()));
                }
                out.push(tagged("kv", &format!("{}/{}GiB {name} {}", gib(total), gib(peak), tags.join(" "))));
                continue;
            }
            let head = format!("{}GiB {name}", gib(total));
            if slots.is_empty() && draft == 0 {
                let toks: String = solo.iter().map(|t| format!(" tokens={t}")).collect();
                out.push(tagged("kv", &format!("{head}{toks}")));
                continue;
            }
            let mut parts: Vec<String> = solo.iter().map(|t| format!("solo={t}")).collect();
            let mut sizes: Vec<i64> = Vec::new();
            for s in &slots {
                if !sizes.contains(s) {
                    sizes.push(*s);
                }
            }
            for tok in sizes {                   // slots of one size read as `slots=2x262144`
                parts.push(format!("slots={}x{tok}", slots.iter().filter(|s| **s == tok).count()));
            }
            if draft != 0 {
                parts.push(format!("draft={}GiB", gib(draft)));
            }
            out.push(tagged("kv", &format!("{head} {}", parts.join(" "))));
        }
        out
    }
}

impl LoadBars {
    /// `out` None: the process's stdout. `gpus`: CUDA ordinal -> nvidia-smi number.
    pub fn new(out: Option<Arc<dyn Out>>, gpus: Option<Vec<i64>>) -> LoadBars {
        let out: Arc<dyn Out> = out.unwrap_or_else(|| Arc::new(Stdout));
        let tty = out.isatty();
        LoadBars { out, st: Mutex::new(Bars { gpus: gpus.unwrap_or_default(), tty, total: 0, into: String::new(), into_pad: 0,
                                              shares: Vec::new(), rows: Vec::new(), beat_seen: Vec::new(), beat_t: None,
                                              started: 0, t0: None, drawn: 0, shown: Vec::new(), kv: Vec::new() }) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Bars> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn tty(&self) -> bool {
        self.lock().tty
    }

    /// The device's name as a key in a console line: `cuda<N>` as nvidia-smi numbers it.
    pub fn key(&self, target: &str) -> String {
        self.lock().key(target)
    }

    /// A row's name: the weights' row says where the engine places now and that destination's part so far (`staged`).
    pub fn label(&self, target: &str, staged: bool) -> String {
        self.lock().label(target, staged)
    }

    /// `done` of `total` bytes of the plan land in `target` (`basalt load: share`).
    pub fn set_share(&self, target: &str, done: i64, total: i64) {
        let mut b = self.lock();
        let out = self.out.as_ref();
        if b.t0.is_none() {
            b.t0 = Some(wall());
        }
        if !b.shares.iter().any(|(k, _)| k == target) {
            b.shares.push((target.to_string(), Share { done: 0, total, reported: false }));
        }
        let key_len = pyfmt::char_len(&b.key(target));
        let (fire, tty) = {
            let tty = b.tty;
            let sh = &mut b.shares.iter_mut().find(|(k, _)| k == target).unwrap().1;
            sh.total = total;
            sh.done = done.min(total);
            let fire = total > 0 && sh.done >= total && !sh.reported;
            if fire {
                sh.reported = true;
            }
            (fire, tty)
        };
        b.into_pad = b.into_pad.max(key_len);
        if fire && !tty {
            let line = tagged("placed", &format!("{} {}GiB {}s", b.key(target), fmt_f(total as f64 / GIB, 1),
                                                 fmt_f(wall() - b.t0.unwrap_or(0.0), 1)));
            b.say(out, &line);
        }
        if b.tty && b.drawn > 0 && target == b.into {
            b.erase(out);
            b.draw(out);
        }
    }

    /// The plan's split of `done` bytes by destination: `cuda0=12.5GiB ram=64.3GiB`; "" when none was announced.
    pub fn split(&self, done: i64) -> String {
        self.lock().split(done)
    }

    /// Where the engine will place, in order (`basalt load: stages`): the label is as wide as the longest.
    pub fn set_stages(&self, targets: &[String]) {
        let mut b = self.lock();
        let m = targets.iter().map(|t| pyfmt::char_len(&b.key(t))).max().unwrap_or(0);
        b.into_pad = b.into_pad.max(m);
    }

    /// The engine places into `target` from now on (`basalt load: into`).
    pub fn set_into(&self, target: &str) {
        let mut b = self.lock();
        let out = self.out.as_ref();
        b.into = target.to_string();
        let l = pyfmt::char_len(&b.key(target));
        b.into_pad = b.into_pad.max(l);
        if b.tty && b.drawn > 0 {
            b.erase(out);
            b.draw(out);
        }
    }

    pub fn done(&self) -> i64 {
        self.lock().done()
    }

    /// The load so far: {"percent": int | None, "gib": GiB placed} over the weights.
    pub fn progress(&self) -> Map<String, Value> {
        let b = self.lock();
        let want = if b.total != 0 { b.total } else { b.row(WEIGHTS).map(|r| r.plan).unwrap_or(0) };
        let mut m = Map::new();
        m.insert("percent".into(), if want > 0 { Value::from(99.min((100.0 * b.done() as f64 / want as f64) as i64)) }
                                   else { Value::Null });
        m.insert("gib".into(), pyfmt::round_n(b.done() as f64 / GIB, 1).into());
        m
    }

    pub fn lines(&self) -> Vec<String> {
        self.lock().lines()
    }

    /// Rows for the heartbeat off a terminal: only the rows whose bytes moved since the previous call (all the first
    /// time); a moved row too young for its own window carries the mean over the beat.
    pub fn moved_lines(&self) -> Vec<String> {
        let mut b = self.lock();
        let previous = std::mem::take(&mut b.beat_seen);
        let beat_t = b.beat_t;
        b.beat_t = Some(mono());
        let rows = b.rows_text(WIDTH, beat_t.map(|t| (t, previous.as_slice())));
        let have: Vec<(String, i64)> = b.rows.iter().map(|(k, r)| (k.clone(), r.have)).collect();
        let out = rows.into_iter().filter(|(k, _)| {
            let prev = previous.iter().find(|(p, _)| p == k).map(|x| x.1);
            let now = have.iter().find(|(p, _)| p == k).map(|x| x.1);
            prev != now
        }).map(|x| x.1).collect();
        b.beat_seen = have;
        out
    }

    /// Redraw the bars when time alone changed them (a rate that ended).
    pub fn refresh(&self) {
        let mut b = self.lock();
        let out = self.out.as_ref();
        if b.tty && b.drawn > 0 && b.fitted(out) != b.shown {
            b.erase(out);
            b.draw(out);
        }
    }

    /// A line printed above the bars.
    pub fn say(&self, text: &str) {
        let mut b = self.lock();
        b.say(self.out.as_ref(), text);
    }

    pub fn set_kv(&self, target: &str, nbytes: i64, tokens: i64, kind: &str, max_bytes: i64) {
        self.lock().kv.push((target.to_string(), nbytes, tokens, kind.to_string(), max_bytes.max(nbytes)));
    }

    /// One line per device: all the session state allocated there; an elastic one reads `backed now/most it can take`.
    pub fn kv_lines(&self) -> Vec<String> {
        self.lock().kv_lines()
    }

    /// The bytes the whole start places.
    pub fn set_total(&self, total: i64) {
        self.lock().total = total;
    }

    /// `done` of `total` bytes of `target` (the weights, or the n-gram table lock).
    pub fn update(&self, target: &str, done: i64, total: i64) {
        let mut b = self.lock();
        let out = self.out.as_ref();
        if b.t0.is_none() {
            b.t0 = Some(wall());
        }
        let started = b.started;
        let mut bump = false;
        let before;
        {
            let t = b.row_mut(target);
            before = t.have;
            if t.start.is_none() {
                t.start = Some(started);
                bump = true;
            }
            t.have = if total != 0 { done.min(total) } else { done };
            t.plan = total;
        }
        if bump {
            b.started += 1;
        }
        if target == WEIGHTS {
            b.total = total;                      // the engine's latest announcement
        }
        Bars::sample(b.row_mut(target), before, mono());
        if target == TABLE && !b.tty && total > 0 && done >= total && before < total {
            let line = tagged("placed", &format!("n-gram table {}GiB {}s", fmt_f(total as f64 / GIB, 1),
                                                 fmt_f(wall() - b.t0.unwrap_or(0.0), 1)));
            b.say(out, &line);
        }
        b.erase(out);
        b.draw(out);
    }

    /// The engine is up: replace the bars with the result line.
    pub fn finish(&self) {
        let mut b = self.lock();
        let out = self.out.as_ref();
        b.erase(out);
        let done = b.done();
        let want = if b.total != 0 { b.total } else { b.row(WEIGHTS).map(|r| r.plan).unwrap_or(0) };
        if done > 0 && done >= want {                  // a failed start is reported elsewhere
            let split = b.split(done);
            let line = format!("{}GiB {}s{}", fmt_f(done as f64 / GIB, 1), fmt_f(wall() - b.t0.unwrap_or(0.0), 1),
                               if split.is_empty() { String::new() } else { format!("  {split}") });
            out.write(&(tagged("loaded", &line) + "\n"));
        }
        for line in b.kv_lines() {
            out.write(&(line + "\n"));
        }
        b.kv.clear();
        b.rows.clear();
        b.shares.clear();
        out.flush();
    }
}

static ACTIVE: Mutex<Option<Arc<LoadBars>>> = Mutex::new(None);    // the bars on screen during a start

/// Print a `[server]` line; while load bars are on screen it goes above them, so they are redrawn in place.
pub fn server_line(text: &str) {
    let bars = ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    match bars {
        Some(b) => b.say(text),
        None => crate::util::stdout::out(text),
    }
}

/// (target, done, total) from an engine `basalt load: <target> <done> <total>` line (`weights` or `ngram`), else None.
pub fn load_progress(line: &str) -> Option<(String, i64, i64)> {
    let p = pyfmt::split_ws(line);
    if p.len() != 5 || p[0] != "basalt" || p[1] != "load:" || !(p[2] == WEIGHTS || p[2] == TABLE) {
        return None;
    }
    Some((p[2].to_string(), pyfmt::py_int(p[3])?, pyfmt::py_int(p[4])?))
}

/// (target, bytes, tokens, kind, max bytes) from `basalt load: kv [slot|draft ]<target> <bytes> <tokens> [<max>]`.
pub fn load_kv(line: &str) -> Option<(String, i64, i64, String, i64)> {
    let p = pyfmt::split_ws(line);
    if p.len() < 3 || p[..3] != ["basalt", "load:", "kv"] {
        return None;
    }
    let mut rest: Vec<&str> = p[3..].to_vec();
    let kind = if rest.first().is_some_and(|x| *x == "slot" || *x == "draft") { rest.remove(0).to_string() } else { String::new() };
    if (rest.len() == 3 || rest.len() == 4) && rest[1..].iter().all(|x| pyfmt::is_digits(x)) {
        let b = pyfmt::py_int(rest[1])?;
        let max = if rest.len() == 4 { pyfmt::py_int(rest[3])? } else { b };
        return Some((rest[0].to_string(), b, pyfmt::py_int(rest[2])?, kind, max));
    }
    None
}

/// Whether an engine line is `basalt load: planning`.
pub fn load_planning(line: &str) -> bool {
    pyfmt::split_ws(line) == ["basalt", "load:", "planning"]
}

fn stage_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\A(?:ram|cuda\d+)\z").unwrap())
}

/// The targets of an engine `basalt load: stages <target>...` line, in order, else None.
pub fn load_stages(line: &str) -> Option<Vec<String>> {
    let p = pyfmt::split_ws(line);
    (p.len() >= 4 && p[..3] == ["basalt", "load:", "stages"] && p[3..].iter().all(|t| stage_re().is_match(t)))
        .then(|| p[3..].iter().map(|s| s.to_string()).collect())
}

/// (target, done, total) from an engine `basalt load: share <target> <done> <total>` line, else None.
pub fn load_share(line: &str) -> Option<(String, i64, i64)> {
    let p = pyfmt::split_ws(line);
    if p.len() == 6 && p[..3] == ["basalt", "load:", "share"] && stage_re().is_match(p[3]) && pyfmt::is_digits(p[4])
        && pyfmt::is_digits(p[5]) {
        return Some((p[3].to_string(), pyfmt::py_int(p[4])?, pyfmt::py_int(p[5])?));
    }
    None
}

/// Where the engine places now from a `basalt load: into <target>` line (`cuda<N>` or `ram`), else None.
pub fn load_into(line: &str) -> Option<String> {
    let p = pyfmt::split_ws(line);
    (p.len() == 4 && p[..3] == ["basalt", "load:", "into"] && stage_re().is_match(p[3])).then(|| p[3].to_string())
}

/// The bytes of the whole start from an engine `basalt load: total <bytes>` line, else None.
pub fn load_total(line: &str) -> Option<i64> {
    let p = pyfmt::split_ws(line);
    if p.len() == 4 && p[..3] == ["basalt", "load:", "total"] && pyfmt::is_digits(p[3]) {
        return pyfmt::py_int(p[3]);
    }
    None
}

/// The text of an engine warning line, else None.
pub fn engine_warning(line: &str) -> Option<String> {
    let e = engine_line(line)?;
    let t = pyfmt::strip(&e.text);
    (e.level == WARN && !t.is_empty()).then(|| t.to_string())
}

/// The text after `RAM floor ` of an engine `basalt generate: RAM floor ...` line, else None.
pub fn engine_ram_floor(line: &str) -> Option<String> {
    let e = engine_line(line)?;
    let rest = pyfmt::strip(pyfmt::tail_chars(&e.text, 10));
    (e.tag == "basalt generate" && e.text.starts_with("RAM floor ") && !rest.is_empty()).then(|| rest.to_string())
}

/// The text of an engine error line, else None.
pub fn engine_error(line: &str) -> Option<String> {
    let e = engine_line(line)?;
    let t = pyfmt::strip(&e.text);
    (e.level == ERR && !t.is_empty()).then(|| t.to_string())
}

/// The engine's error lines this start wrote to its log.
pub fn start_errors(log: Option<&str>, offset: u64) -> Vec<String> {
    let Some(log) = log.filter(|l| !l.is_empty()) else { return Vec::new() };
    let Some(text) = read_from(log, offset) else { return Vec::new() };
    pyfmt::splitlines(&text).into_iter().filter_map(|x| engine_error(pyfmt::strip(x))).collect()
}

pub type OnProgress = Box<dyn Fn(Map<String, Value>) + Send>;

/// Show load progress and warnings from the engine log until `done` is set, with a heartbeat while idle (blocking:
/// run it on its own thread). `on_progress` gets LoadBars::progress() after each batch of log lines.
pub fn narrate_start(log_path: &str, offset: u64, done: &Event, heartbeat: f64, out: Option<Arc<dyn Out>>,
                     gpus: Option<Vec<i64>>, on_progress: Option<OnProgress>) {
    let bars = Arc::new(LoadBars::new(out, gpus));
    *ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(bars.clone());
    let t0 = wall();
    let mut last = t0;
    let mut beat = t0;                               // last busy heartbeat
    let mut pos = offset;
    loop {
        let stop = done.wait(0.05);                  // often: a row's rate is timed by when its lines are read
        let chunk = (|| -> Option<Vec<u8>> {
            let mut f = std::fs::File::open(log_path).ok()?;
            f.seek(SeekFrom::Start(pos)).ok()?;
            let mut b = Vec::new();
            f.read_to_end(&mut b).ok()?;
            Some(b)
        })().unwrap_or_default();
        let has_lines = chunk.contains(&b'\n');
        if has_lines {
            let cut = chunk.iter().rposition(|b| *b == b'\n').unwrap() + 1;
            pos += cut as u64;
            let text = String::from_utf8_lossy(&chunk[..cut]).into_owned();
            for line in pyfmt::splitlines(&text) {
                if load_planning(line) {
                    bars.say("[server] planning weight allocation");
                    last = wall();
                } else if let Some(stages) = load_stages(line) {
                    bars.set_stages(&stages);
                } else if let Some(into) = load_into(line) {
                    bars.set_into(&into);
                } else if let Some((t, d, n)) = load_share(line) {
                    bars.set_share(&t, d, n);
                } else if let Some(n) = load_total(line) {
                    bars.set_total(n);
                    last = wall();
                } else if let Some((t, b, k, kind, m)) = load_kv(line) {
                    bars.set_kv(&t, b, k, &kind, m);
                } else if let Some((t, d, n)) = load_progress(line) {
                    bars.update(&t, d, n);
                    last = wall();
                } else if let Some(w) = engine_warning(line) {
                    bars.say(&format!("[server] warning: {w}"));
                } else if let Some(r) = engine_ram_floor(line) {
                    bars.say(&tagged("ram", &format!("floor {r}")));
                }
            }
        }
        if has_lines {
            if let Some(f) = &on_progress {
                f(bars.progress());
            }
        }
        bars.refresh();
        if stop {
            bars.finish();
            let mut a = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
            if a.as_ref().is_some_and(|x| Arc::ptr_eq(x, &bars)) {
                *a = None;
            }
            return;
        }
        if !bars.tty() && wall() - beat >= heartbeat {      // busy heartbeat: print only rows that moved
            beat = wall();
            for r in bars.moved_lines() {
                bars.say(&r);
            }
        }
        if wall() - last > heartbeat {
            last = wall();
            bars.say(&tagged("starting", &format!("{}s", fmt_f(wall() - t0, 0))));
        }
    }
}

/// narrate_start on a new thread.
pub fn spawn_narrator(log_path: String, offset: u64, done: Arc<Event>, heartbeat: f64, out: Option<Arc<dyn Out>>,
                      gpus: Option<Vec<i64>>, on_progress: Option<OnProgress>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || narrate_start(&log_path, offset, &done, heartbeat, out, gpus, on_progress))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn parsers() {
        assert_eq!(load_progress("basalt load: weights 5 10"), Some(("weights".into(), 5, 10)));
        assert_eq!(load_progress("basalt load: cuda0 5 10"), None);
        assert_eq!(load_progress("basalt load: weights x 10"), None);
        assert_eq!(load_kv("basalt load: kv slot cuda0 100 4096"), Some(("cuda0".into(), 100, 4096, "slot".into(), 100)));
        assert_eq!(load_kv("basalt load: kv cuda1 100 4096 300"), Some(("cuda1".into(), 100, 4096, "".into(), 300)));
        assert_eq!(load_kv("basalt load: kv cuda1 -1 4096"), None);
        assert!(load_planning(" basalt  load: planning "));
        assert_eq!(load_stages("basalt load: stages cuda0 ram"), Some(vec!["cuda0".into(), "ram".into()]));
        assert_eq!(load_stages("basalt load: stages cuda0 disk"), None);
        assert_eq!(load_share("basalt load: share cuda1 3 9"), Some(("cuda1".into(), 3, 9)));
        assert_eq!(load_into("basalt load: into ram"), Some("ram".into()));
        assert_eq!(load_total("basalt load: total 77"), Some(77));
        assert_eq!(engine_warning("basalt generate: warning:  low RAM "), Some("low RAM".into()));
        assert_eq!(engine_warning("basalt generate: low RAM"), None);
        assert_eq!(engine_ram_floor("basalt generate: RAM floor 12 GiB kept"), Some("12 GiB kept".into()));
        assert_eq!(engine_ram_floor("basalt serve: RAM floor 12"), None);
        assert_eq!(engine_error("basalt serve: error: boom"), Some("boom".into()));
    }

    fn bars(tty: bool, gpus: Vec<i64>) -> (Arc<Capture>, LoadBars) {
        let cap = Arc::new(Capture { tty, ..Default::default() });
        let b = LoadBars::new(Some(cap.clone() as Arc<dyn Out>), Some(gpus));
        (cap, b)
    }

    #[test]
    fn split_kv_and_finish() {
        let (cap, b) = bars(false, vec![0, 1]);
        let g = 1i64 << 30;
        b.set_share("cuda0", 0, 3 * g);
        b.set_share("ram", 0, 9 * g);
        assert_eq!(b.split(12 * g), "cuda0=3.0GiB ram=9.0GiB");
        assert_eq!(b.key("cuda1"), "cuda1");
        let (_, b2) = bars(false, vec![1, 0]);
        assert_eq!(b2.key("cuda0"), "cuda1");
        assert_eq!(b2.key("cuda5"), "cuda5");
        b.set_share("cuda0", 3 * g, 3 * g);
        assert!(cap.text.lock().unwrap().starts_with("[server] placed   cuda0 3.0GiB "));
        b.set_kv("cuda0", 2 * g, 262144, "", 0);
        b.set_kv("cuda0", g, 4096, "slot", 0);
        b.set_kv("cuda0", g, 4096, "slot", 0);
        b.set_kv("cuda0", g / 2, 0, "draft", 0);
        b.set_kv("cuda1", g, 8192, "", 0);
        b.set_kv("ram", g, 100, "slot", 4 * g);
        assert_eq!(b.kv_lines(), vec!["[server] kv       4.5GiB cuda0 solo=262144 slots=2x4096 draft=0.5GiB",
                                      "[server] kv       1.0GiB cuda1 tokens=8192",
                                      "[server] kv       1.0/4.0GiB ram elastic slots=1"]);
        b.update(WEIGHTS, 12 * g, 12 * g);
        b.finish();
        let text = cap.text.lock().unwrap().clone();
        let fin: Vec<&str> = text.lines().filter(|l| l.starts_with("[server] loaded")).collect();
        assert_eq!(fin.len(), 1);
        let re = Regex::new(r"^\[server\] loaded   12\.0GiB \d+\.\ds  cuda0=3\.0GiB ram=9\.0GiB$").unwrap();
        assert!(re.is_match(fin[0]), "{}", fin[0]);
        assert!(text.ends_with("[server] kv       1.0/4.0GiB ram elastic slots=1\n"));
    }

    #[test]
    fn rows_and_labels() {
        let (_, b) = bars(true, vec![0]);
        let g = 1i64 << 30;
        b.update(WEIGHTS, 3 * g, 12 * g);
        assert_eq!(b.lines(), vec![format!("[server] loading model weights [======>{}]   3.0/12.0 GiB", " ".repeat(17))]);
        b.set_share("cuda0", g, 3 * g);
        b.set_into("cuda0");
        assert_eq!(b.label(WEIGHTS, true), "loading model weights: cuda0 1.0/3.0");
        assert_eq!(b.label(WEIGHTS, false), "loading model weights");
        assert_eq!(b.progress()["percent"], 25);
        assert_eq!(b.progress()["gib"], 3.0);
        let (_, c) = bars(false, vec![]);
        c.update(WEIGHTS, g, 0);
        assert_eq!(c.lines(), vec!["[server] loading model weights    1.0 GiB"]);
        assert_eq!(bar(5, 10, 8), "====>   ");
        assert_eq!(bar(10, 10, 8), "========");
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("serve-progress-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.join("engine.log")
    }

    fn narrate(log: &PathBuf, out: Arc<Capture>, heartbeat: f64) -> (std::thread::JoinHandle<()>, Arc<Event>) {
        std::fs::write(log, "").unwrap();
        let done = Arc::new(Event::new());
        let h = spawn_narrator(log.to_string_lossy().into_owned(), 0, done.clone(), heartbeat,
                               Some(out as Arc<dyn Out>), Some(vec![0]), None);
        (h, done)
    }

    fn weights_rows(cap: &str) -> Vec<String> {
        let re = Regex::new(r"\d+\.\d/\d+\.\d GiB").unwrap();
        cap.lines().filter(|l| l.contains("model weights") && re.is_match(l)).map(str::to_string).collect()
    }

    #[test]
    fn off_a_terminal_moved_rows_rates_and_the_idle_heartbeat_all_print() {
        let total: i64 = 12 << 30;
        let log = tmp("busy");
        let out = Arc::new(Capture::default());
        let (h, done) = narrate(&log, out.clone(), 0.5);
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        writeln!(f, "basalt load: total {total}").unwrap();
        for i in 1..=16 {
            writeln!(f, "basalt load: weights {} {total}", total * i / 16).unwrap();
            f.flush().unwrap();
            std::thread::sleep(Duration::from_millis(250));
        }
        std::thread::sleep(Duration::from_millis(1300));
        done.set();
        h.join().unwrap();
        let cap = out.text.lock().unwrap().clone();
        let rows = weights_rows(&cap);
        assert!(rows.len() >= 3, "{cap}");
        assert!(rows.iter().any(|l| l.contains(" MB/s")), "{cap}");
        assert!(cap.lines().any(|l| l.starts_with("[server] starting")), "{cap}");
        let fin: Vec<&str> = cap.lines().filter(|l| l.starts_with("[server] loaded")).collect();
        assert_eq!(fin.len(), 1, "{cap}");
        assert!(fin[0].contains("loaded   12.0GiB"));
    }

    #[test]
    fn on_a_terminal_the_heartbeat_adds_no_rows_and_nothing_wraps() {
        let total: i64 = 8 << 30;
        let log = tmp("tty");
        let out = Arc::new(Capture { tty: true, cols: Some(60), ..Default::default() });
        let (h, done) = narrate(&log, out.clone(), 0.5);
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        writeln!(f, "basalt load: total {total}").unwrap();
        for i in 1..=8 {
            writeln!(f, "basalt load: weights {} {total}", total * i / 8).unwrap();
            f.flush().unwrap();
            std::thread::sleep(Duration::from_millis(250));
        }
        done.set();
        h.join().unwrap();
        let cap = out.text.lock().unwrap().clone();
        let bare: Vec<&str> = cap.lines().filter(|l| l.starts_with("[server]") && l.contains(" GiB")
            && !l.starts_with("\x1b[2K") && !l.starts_with("[server] loaded")).collect();
        assert!(bare.is_empty(), "{cap}");
        for w in out.writes.lock().unwrap().iter() {
            if let Some(r) = w.1.strip_prefix("\x1b[2K") {
                assert!(r.trim_end_matches('\n').chars().count() < 60, "{r}");
            }
        }
    }

    #[test]
    fn start_log_readers() {
        let log = tmp("start");
        std::fs::write(&log, "earlier\nbasalt serve: error: old\n").unwrap();
        let off = std::fs::metadata(&log).unwrap().len();
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        write!(f, "basalt mtp: error: {DRAFT_HEAD_FAIL}\nbasalt mtp: draft head over 5 GiB\nbasalt serve: error: x\n\
                   basalt serve: warning: late\n\n").unwrap();
        let p = log.to_string_lossy().into_owned();
        assert_eq!(start_errors(Some(&p), off), vec![DRAFT_HEAD_FAIL.to_string(), "x".into()]);
        assert_eq!(start_last_line(Some(&p), off), "x");
        assert_eq!(start_failure_hint(Some(&p), off), format!(". mtp: {DRAFT_HEAD_FAIL}. draft head over 5 GiB"));
        assert_eq!(start_failure_hint(Some(&p), 0), format!(". mtp: {DRAFT_HEAD_FAIL}. draft head over 5 GiB"));
        assert_eq!(start_failure_hint(None, 0), "");
        assert_eq!(start_errors(Some("no/such/file"), 0), Vec::<String>::new());
        assert_eq!(start_last_line(Some(&p), 10_000), "");
    }
}
