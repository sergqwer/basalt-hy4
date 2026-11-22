//! serve/engine/console.py: the engine's console lines (XSLOT, XMAP), sent once the server asks with STATS, and what
//! /slots and /metrics show of them; the INFO cards= and cpu_l3= facts and the hardware view built on them.
use crate::pyfmt::{self, round_int, round_n};
use regex::Regex;
use serde_json::{Map, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};

pub const LINES: [&str; 2] = ["XSLOT ", "XMAP "];
pub const HISTORY: i64 = 60;        // seconds of tokens per window kept per slot
pub const RATE_SPAN_S: f64 = 2.0;   // the lookup rates are over the lines of the last seconds
pub const STALE_S: f64 = 2.5;       // no line for this long: no window runs, the rates are 0
pub const RATE_MAP_S: f64 = 1.0;    // the expert map's lookups a second are over about this many seconds of XMAP lines
pub const DECAY: f64 = 0.85;        // a sampled expert's recent lookups fade by this much a second

pub const COUNTS: [&str; 6] = ["windows", "tokens", "hits", "lookups", "offloaded", "files"];

/// One XSLOT line's counts (in COUNTS order) and its (accepted, offered) drafts per position.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Counts {
    pub c: [i64; 6],
    pub accept: Vec<(i64, i64)>,
}

impl Counts {
    fn get(&self, k: &str) -> i64 {
        self.c[COUNTS.iter().position(|x| *x == k).unwrap()]
    }
}

/// An XSLOT line as (slot, counts); None when it is not one. (A draft pair of one number, which Python stores and then
/// fails on when the slot is read, is refused here.)
pub fn parse_xslot(line: &str) -> Option<(i64, Counts)> {
    let f = pyfmt::split_ws(line);
    if f.len() < 9 || f[0] != "XSLOT" {
        return None;
    }
    let mut c = [0i64; 6];
    for (i, x) in f[2..8].iter().enumerate() {
        c[i] = pyfmt::py_int(x)?;
    }
    let accept = if f[8] == "-" {
        Vec::new()
    } else {
        let mut v = Vec::new();
        for p in f[8].split(',') {
            let nums: Option<Vec<i64>> = p.split('/').map(pyfmt::py_int).collect();
            let nums = nums?;
            if nums.len() < 2 {
                return None;
            }
            v.push((nums[0], nums[1]));
        }
        v
    };
    Some((pyfmt::py_int(f[1])?, Counts { c, accept }))
}

/// An XMAP line as (total, homes, routed); None when it is not one.
pub fn parse_xmap(line: &str) -> Option<(i64, Vec<i64>, Vec<i64>)> {
    let f = pyfmt::split_ws(line);
    if f.len() < 4 || f[0] != "XMAP" {
        return None;
    }
    let homes: Option<Vec<i64>> = f[2].chars().map(|c| pyfmt::py_int(&c.to_string())).collect();
    let routed: Option<Vec<i64>> = f[3].split(',').map(pyfmt::py_int).collect();
    let (homes, routed) = (homes?, routed?);
    let total = pyfmt::py_int(f[1])?;
    (homes.len() == routed.len()).then_some((total, homes, routed))
}

struct MapState {
    t: f64,
    total: i64,
    homes: Vec<i64>,
    lines: VecDeque<(f64, f64, Vec<i64>)>,
    recent: Vec<f64>,
}

#[derive(Default)]
struct State {
    slots: HashMap<i64, Counts>,
    runs: HashMap<i64, VecDeque<(f64, Counts)>>,        // slot -> (start time, its latest counts) per engine run
    deltas: HashMap<i64, VecDeque<(f64, [i64; 6])>>,    // slot -> (time, d counts)
    per_s: HashMap<i64, VecDeque<(i64, i64, i64)>>,     // slot -> (second, d tokens, d windows)
    map: Option<MapState>,
}

fn push<T>(q: &mut VecDeque<T>, cap: usize, v: T) {
    if q.len() == cap {
        q.pop_front();
    }
    q.push_back(v);
}

pub type Clock = Box<dyn Fn() -> f64 + Send + Sync>;

/// The latest XSLOT counts of each slot, the recent deltas behind their rates, and the expert map.
pub struct ConsoleLines {
    clock: Clock,
    st: Mutex<State>,
}

impl Default for ConsoleLines {
    fn default() -> Self {
        Self::new()
    }
}

fn idx(k: &str) -> usize {
    COUNTS.iter().position(|x| *x == k).unwrap()
}

impl ConsoleLines {
    pub fn new() -> Self {
        Self::with_clock(Box::new(crate::util::logbuf::now_s))
    }

    pub fn with_clock(clock: Clock) -> Self {
        ConsoleLines { clock, st: Mutex::new(State::default()) }
    }

    /// Take an XSLOT or XMAP line; false for any other line.
    pub fn feed(&self, line: &str) -> bool {
        if !LINES.iter().any(|p| line.starts_with(p)) {
            return false;
        }
        let now = (self.clock)();
        if line.starts_with("XSLOT ") {
            if let Some((b, c)) = parse_xslot(line) {
                self.slot_line(b, c, now);
            }
        } else if let Some((total, homes, routed)) = parse_xmap(line) {
            self.map_line(total, homes, routed, now);
        }
        true
    }

    fn slot_line(&self, b: i64, c: Counts, now: f64) {
        let mut st = self.st.lock().unwrap();
        let prev = st.slots.get(&b).cloned();
        let runs = st.runs.entry(b).or_default();
        // an engine run starts from zero: its first line counts from there
        let prev = match prev {
            Some(p) if c.c[0] >= p.c[0] => {
                let start = runs.back().map(|r| r.0).unwrap_or(now);
                if let Some(last) = runs.back_mut() {
                    *last = (start, c.clone());
                }
                p.c
            }
            _ => {
                push(runs, 16, (now, c.clone()));
                [0; 6]
            }
        };
        let mut d = [0i64; 6];
        for i in 0..6 {
            d[i] = c.c[i] - prev[i];
        }
        st.slots.insert(b, c);
        push(st.deltas.entry(b).or_default(), 16, (now, d));
        if d[0] > 0 {
            let q = st.per_s.entry(b).or_default();
            let sec = now.trunc() as i64;
            match q.back_mut() {
                Some(last) if last.0 == sec => *last = (sec, last.1 + d[1], last.2 + d[0]),
                _ => push(q, (HISTORY * 3) as usize, (sec, d[1], d[0])),
            }
        }
    }

    fn map_line(&self, total: i64, homes: Vec<i64>, routed: Vec<i64>, now: f64) {
        let mut st = self.st.lock().unwrap();
        let same = st.map.as_ref().is_some_and(|m| m.recent.len() == routed.len());
        let (dt, recent, mut lines) = match st.map.take() {
            Some(m) if same => {
                let dt = (now - m.t).max(0.05).min(5.0);
                let k = DECAY.powf(dt);
                (dt, m.recent.iter().map(|r| r * k).collect::<Vec<f64>>(), m.lines)
            }
            _ => (1.0, vec![0.0; routed.len()], VecDeque::new()),
        };
        let recent = recent.iter().zip(&routed).map(|(r, n)| r + *n as f64).collect();
        push(&mut lines, 16, (now, dt, routed));
        st.map = Some(MapState { t: now, total, homes, lines, recent });
    }

    /// Slot `b`'s console fields over the request that started at `since` (its engine runs added up; None: the last
    /// run): acceptance by draft position, tokens per window a second and the expert split with its lookup rates.
    pub fn slot(&self, b: i64, since: Option<f64>) -> Map<String, Value> {
        let now = (self.clock)();
        let (runs, deltas, per_s) = {
            let st = self.st.lock().unwrap();
            (st.runs.get(&b).map(|q| q.iter().cloned().collect::<Vec<_>>()).unwrap_or_default(),
             st.deltas.get(&b).map(|q| q.iter().cloned().collect::<Vec<_>>()).unwrap_or_default(),
             st.per_s.get(&b).map(|q| q.iter().cloned().collect::<Vec<_>>()).unwrap_or_default())
        };
        // a request with no run of its own yet has no figures: not the slot's last request's
        let mine: Vec<&Counts> = match since {
            Some(s) => runs.iter().filter(|(t, _)| *t >= s).map(|(_, r)| r).collect(),
            None => runs.last().map(|(_, r)| r).into_iter().collect(),
        };
        let mut out = Map::new();
        if mine.is_empty() {
            for k in ["accept_pos", "accept_hist", "hit"] {
                out.insert(k.into(), Value::Null);
            }
            return out;
        }
        let mut c = Counts::default();
        for i in 0..6 {
            c.c[i] = mine.iter().map(|r| r.c[i]).sum();
        }
        let n = mine.iter().map(|r| r.accept.len()).max().unwrap_or(0);
        c.accept = (0..n).map(|j| {
            let have: Vec<&(i64, i64)> = mine.iter().filter_map(|r| r.accept.get(j)).collect();
            (have.iter().map(|p| p.0).sum(), have.iter().map(|p| p.1).sum())
        }).collect();
        let accept: Vec<Value> = c.accept.iter().filter(|(_, o)| *o > 0)
            .map(|(a, o)| Value::from(round_n(*a as f64 / *o as f64, 3))).collect();
        let mut by_sec: HashMap<i64, f64> = HashMap::new();
        for (s, t, w) in &per_s {
            if *w > 0 {
                by_sec.insert(*s, *t as f64 / *w as f64);
            }
        }
        let last = now.floor() as i64;
        let hist: Vec<Value> = (last - HISTORY + 1..=last)
            .map(|s| by_sec.get(&s).map(|v| Value::from(round_n(*v, 2))).unwrap_or(Value::Null)).collect();
        out.insert("accept_pos".into(), if accept.is_empty() { Value::Null } else { Value::Array(accept) });
        out.insert("accept_hist".into(), if hist.iter().any(|v| !v.is_null()) { Value::Array(hist) } else { Value::Null });
        out.insert("hit".into(), Self::hit(&c, &deltas, now).map(Value::Object).unwrap_or(Value::Null));
        if !c.accept.is_empty() {       // the drafts a window offers count at each position they reach
            out.insert("drafts_accepted".into(), c.accept.iter().map(|p| p.0).sum::<i64>().into());
            out.insert("drafts_offered".into(), c.accept.iter().map(|p| p.1).sum::<i64>().into());
        }
        out
    }

    fn hit(c: &Counts, deltas: &[(f64, [i64; 6])], now: f64) -> Option<Map<String, Value>> {
        let (hits, lookups, offloaded) = (c.get("hits"), c.get("lookups"), c.get("offloaded"));
        let routed = lookups + offloaded;
        if routed <= 0 {
            return None;                        // a dense model, or no window yet
        }
        let cpu = lookups - hits;
        let r = routed as f64;
        let mut out = Map::new();
        out.insert("vram".into(), round_n(hits as f64 / r, 3).into());
        out.insert("peer".into(), round_n(offloaded as f64 / r, 3).into());
        out.insert("cpu".into(), round_n(cpu as f64 / r, 3).into());
        for k in ["lookups_s", "peer_s", "ram_s"] {
            out.insert(k.into(), Value::Null);
        }
        let recent: Vec<&(f64, [i64; 6])> = deltas.iter().filter(|(t, _)| now - t <= RATE_SPAN_S).collect();
        if deltas.last().is_some_and(|d| now - d.0 > STALE_S) {
            for k in ["lookups_s", "peer_s", "ram_s"] {
                out.insert(k.into(), 0.into());     // the slot's windows stopped
            }
        } else if recent.len() >= 2 {
            let span = recent[recent.len() - 1].0 - recent[0].0;
            if span > 0.0 {
                let sum = |k: &str| recent[1..].iter().map(|x| x.1[idx(k)]).sum::<i64>() as f64;
                let (h, l, o) = (sum("hits"), sum("lookups"), sum("offloaded"));
                out.insert("lookups_s".into(), round_int((l + o) / span).into());
                out.insert("peer_s".into(), round_int(o / span).into());
                out.insert("ram_s".into(), round_int((l - h) / span).into());
            }
        }
        Some(out)
    }

    /// The map's sample: each expert's home, its lookups a second lately and a fading count of its recent lookups;
    /// None before the first XMAP line.
    pub fn expert_map(&self) -> Option<Map<String, Value>> {
        let now = (self.clock)();
        let st = self.st.lock().unwrap();
        let m = st.map.as_ref()?;
        let age = (now - m.t).max(0.0);
        let k = DECAY.powf(age);
        // lookups a second over the lines of about the last second
        let (mut span, mut sums) = (0.0f64, vec![0i64; m.recent.len()]);
        if age <= STALE_S {
            for (_, dt, routed) in m.lines.iter().rev() {
                if span >= RATE_MAP_S {
                    break;
                }
                span += dt;
                for (a, n) in sums.iter_mut().zip(routed) {
                    *a += n;
                }
            }
        }
        let mut out = Map::new();
        out.insert("total".into(), m.total.into());
        out.insert("window_s".into(), 1.into());
        out.insert("tier".into(), m.homes.clone().into());
        out.insert("routed".into(), sums.iter()
            .map(|n| if span > 0.0 { Value::from(round_n(*n as f64 / span, 2)) } else { Value::from(0) }).collect());
        out.insert("recent".into(), m.recent.iter().map(|r| Value::from(round_n(r * k, 2))).collect());
        Some(out)
    }
}

/// `str(v or "")` for a fact read from a JSON map.
fn words_of(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => if *b { "True".into() } else { String::new() },
        Some(Value::Number(n)) => {
            if n.as_f64() == Some(0.0) {
                String::new()
            } else if n.is_f64() {
                pyfmt::repr_f64(n.as_f64().unwrap_or(0.0))
            } else {
                n.to_string()
            }
        }
        Some(Value::Array(a)) if a.is_empty() => String::new(),
        Some(Value::Object(o)) if o.is_empty() => String::new(),
        Some(other) => other.to_string(),
    }
}

/// A PCI address as domain:bus:device.function in plain hex, so NVML's and CUDA's spellings compare equal.
pub fn norm_bus(bus: Option<&Value>) -> String {
    static R: OnceLock<Regex> = OnceLock::new();
    let r = R.get_or_init(|| Regex::new(r"\A([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])\z").unwrap());
    let s = words_of(bus);
    let Some(m) = r.captures(pyfmt::strip(&s)) else { return String::new() };
    let hex = |x: &str| {
        let t = x.trim_start_matches('0').to_ascii_lowercase();
        if t.is_empty() { "0".to_string() } else { t }
    };
    format!("{}:{}:{}.{}", hex(&m[1]), hex(&m[2]), hex(&m[3]), &m[4])
}

/// INFO cards=<bus>/<expert slots>/<GB/s or ->/<expert MiB>/<KV MiB or ->;... as
/// {bus: {expert_slots, h2d_gbps, experts, kv}} (experts and kv in bytes; kv None when the engine does not say).
pub fn parse_cards(words: Option<&Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for part in words_of(words).split(';') {
        let f: Vec<&str> = part.split('/').collect();
        if f.len() != 5 {
            continue;
        }
        let card = (|| -> Option<Map<String, Value>> {
            let mut m = Map::new();
            m.insert("expert_slots".into(), pyfmt::py_int(f[1])?.into());
            m.insert("h2d_gbps".into(), if f[2] == "-" { Value::Null } else { Value::from(pyfmt::py_float(f[2])?) });
            m.insert("experts".into(), (pyfmt::py_int(f[3])? << 20).into());
            m.insert("kv".into(), if f[4] == "-" { Value::Null } else { (pyfmt::py_int(f[4])? << 20).into() });
            Some(m)
        })();
        if let Some(m) = card {
            out.insert(norm_bus(Some(&Value::from(f[0]))), Value::Object(m));
        }
    }
    out
}

/// INFO cpu_l3=0-7,16-23;8-15,24-31 as the logical processors of each L3 cache.
pub fn parse_l3(words: Option<&Value>) -> Vec<Vec<i64>> {
    let mut groups = Vec::new();
    for part in words_of(words).split(';') {
        let mut g = Vec::new();
        for r in part.split(',') {
            let (lo, hi) = r.split_once('-').unwrap_or((r, ""));
            if pyfmt::is_digits(lo) && (hi.is_empty() || pyfmt::is_digits(hi)) {
                let (Some(a), Some(b)) = (pyfmt::py_int(lo), pyfmt::py_int(if hi.is_empty() { lo } else { hi })) else { continue };
                g.extend(a..=b);
            }
        }
        if !g.is_empty() {
            groups.push(g);
        }
    }
    groups
}

/// Takes the per-card and CPU words out of the engine's INFO facts; with cards, the engine's expert slots are their sum.
pub fn engine_facts(info: &mut Map<String, Value>) -> (Map<String, Value>, Vec<Vec<i64>>) {
    let cards = parse_cards(info.shift_remove("cards").as_ref());
    let l3 = parse_l3(info.shift_remove("cpu_l3").as_ref());
    if !cards.is_empty() {
        let sum: i64 = cards.values().filter_map(|c| c.get("expert_slots").and_then(Value::as_i64)).sum();
        info.insert("expert_slots".into(), sum.into());
    }
    (cards, l3)
}

fn is_int(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Number(n)) if !n.is_f64() => n.as_i64(),
        Some(Value::Bool(b)) => Some(*b as i64),
        _ => None,
    }
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// The sampler's reading with what needs the engine: each card's experts, link and VRAM split, the load of each L3
/// group (a CCD), and RAM split into the engine's expert tier, the page cache and the rest.
pub fn hardware_view(now: &Map<String, Value>, static_: &Map<String, Value>, info: &Map<String, Value>,
                     cards: &Map<String, Value>, l3: &[Vec<i64>]) -> Map<String, Value> {
    let mut hw = now.clone();
    let mut bus: Vec<(Value, String)> = Vec::new();
    if let Some(Value::Array(gs)) = static_.get("gpus") {
        for g in gs {
            let k = g.get("index").cloned().unwrap_or(Value::Null);
            let v = norm_bus(g.get("bus"));
            match bus.iter_mut().find(|(x, _)| *x == k) {
                Some(e) => e.1 = v,
                None => bus.push((k, v)),
            }
        }
    }
    let empty = Map::new();
    let mut gpus = Vec::new();
    let src: Vec<Value> = match hw.get("gpus") {
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    };
    for g in src {
        let gm = g.as_object().cloned().unwrap_or_default();
        let index = gm.get("index").cloned().unwrap_or(Value::Null);
        let c = bus.iter().find(|(k, _)| *k == index).and_then(|(_, b)| cards.get(b)).and_then(Value::as_object)
            .unwrap_or(&empty);
        let mut parts = Value::Null;
        if let (false, Some(used)) = (c.is_empty(), is_int(gm.get("mem_used"))) {
            let experts = c.get("experts").and_then(Value::as_i64).unwrap_or(0);
            let kv = c.get("kv").and_then(Value::as_i64).unwrap_or(0);
            let mut p = Map::new();
            p.insert("experts".into(), experts.into());
            p.insert("kv".into(), c.get("kv").cloned().unwrap_or(Value::Null));
            p.insert("other".into(), (used - experts - kv).max(0).into());
            parts = Value::Object(p);
        }
        let mut out = gm;
        out.insert("expert_slots".into(), c.get("expert_slots").cloned().unwrap_or(Value::Null));
        out.insert("h2d_gbps".into(), c.get("h2d_gbps").cloned().unwrap_or(Value::Null));
        out.insert("vram_parts".into(), parts);
        gpus.push(Value::Object(out));
    }
    hw.insert("gpus".into(), Value::Array(gpus));
    let cores = hw.shift_remove("cpu_cores");
    hw.insert("cpu_ccd".into(), Value::Null);
    if let (true, Some(Value::Array(cores))) = (truthy(cores.as_ref()) && l3.len() > 1, cores.as_ref()) {
        let ccd: Vec<Value> = l3.iter().map(|g| {
            let v: Vec<f64> = g.iter().filter(|i| **i >= 0 && (**i as usize) < cores.len())
                .filter_map(|i| cores[*i as usize].as_f64()).collect();
            if v.is_empty() {
                Value::Null
            } else {
                Value::from(round_n(pyfmt::py_sum(v.iter().copied()) / v.len() as f64, 1))
            }
        }).collect();
        hw.insert("cpu_ccd".into(), Value::Array(ccd));
    }
    let cached = hw.shift_remove("ram_cached");
    let used = is_int(hw.get("ram_used"));
    hw.insert("ram_parts".into(), Value::Null);
    if let (Some(cached), Some(used)) = (is_int(cached.as_ref()), used) {
        let mib = match info.get("ram_experts_mib") {
            Some(Value::String(s)) if !s.is_empty() => pyfmt::py_int(s).unwrap_or(0),
            Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)).unwrap_or(0),
            Some(Value::Bool(b)) => *b as i64,
            _ => 0,
        };
        let experts = used.min(mib << 20);
        let mut p = Map::new();
        p.insert("experts".into(), experts.into());
        p.insert("page_cache".into(), cached.into());
        p.insert("other".into(), (used - experts).into());
        hw.insert("ram_parts".into(), Value::Object(p));
    }
    hw
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    #[derive(Clone)]
    struct Clk(Arc<Mutex<f64>>);
    impl Clk {
        fn new() -> Self {
            Clk(Arc::new(Mutex::new(1000.0)))
        }
        fn add(&self, d: f64) {
            *self.0.lock().unwrap() += d;
        }
        fn get(&self) -> f64 {
            *self.0.lock().unwrap()
        }
        fn lines(&self) -> ConsoleLines {
            let c = self.clone();
            ConsoleLines::with_clock(Box::new(move || c.get()))
        }
    }

    fn v(m: Map<String, Value>) -> Value {
        Value::Object(m)
    }

    #[test]
    fn parse() {
        let (b, c) = parse_xslot("XSLOT 2 10 25 80 100 20 3 9/10,5/9,1/4").unwrap();
        assert_eq!((b, c.c, c.accept), (2, [10, 25, 80, 100, 20, 3], vec![(9, 10), (5, 9), (1, 4)]));
        assert_eq!(parse_xslot("XSLOT 0 3 3 0 0 0 0 -").unwrap().1.accept, vec![]);
        assert!(parse_xslot("XSLOT 0 x").is_none());
        assert_eq!(parse_xmap("XMAP 20480 0123 4,0,1,2"), Some((20480, vec![0, 1, 2, 3], vec![4, 0, 1, 2])));
        assert!(parse_xmap("XMAP 20480 012 4,0").is_none());
    }

    #[test]
    fn a_slots_acceptance_split_and_rates() {
        let clk = Clk::new();
        let c = clk.lines();
        assert!(!c.feed("T 5"));
        assert_eq!(v(c.slot(0, None)), json!({"accept_pos": null, "accept_hist": null, "hit": null}));
        c.feed("XSLOT 0 10 30 60 90 10 0 8/10,4/8");
        clk.add(1.0);
        c.feed("XSLOT 0 20 60 120 180 20 0 16/20,8/16");
        let s = v(c.slot(0, None));
        assert_eq!(s["accept_pos"], json!([0.8, 0.5]));
        assert_eq!(s["hit"], json!({"vram": 0.6, "peer": 0.1, "cpu": 0.3, "lookups_s": 100, "peer_s": 10, "ram_s": 30}));
        assert_eq!(s["accept_hist"][59], json!(3.0));
        assert_eq!(s["accept_hist"].as_array().unwrap().len(), 60);
        clk.add(5.0);
        let s = v(c.slot(0, None));
        assert_eq!((s["hit"]["lookups_s"].clone(), s["hit"]["ram_s"].clone()), (json!(0), json!(0)));
        assert_eq!(s["hit"]["vram"], json!(0.6));
    }

    #[test]
    fn a_new_request_counts_from_zero() {
        let clk = Clk::new();
        let c = clk.lines();
        c.feed("XSLOT 1 50 100 10 10 0 0 -");
        clk.add(1.0);
        c.feed("XSLOT 1 0 0 0 0 0 0 -");
        clk.add(1.0);
        c.feed("XSLOT 1 4 12 0 0 0 0 8/8");
        let s = v(c.slot(1, None));
        assert_eq!(s["accept_hist"][59], json!(3.0));
        assert_eq!(s["accept_pos"], json!([1.0]));
        assert_eq!(s["hit"], Value::Null);
    }

    #[test]
    fn a_request_in_several_engine_runs_adds_them_up() {
        let clk = Clk::new();
        let c = clk.lines();
        c.feed("XSLOT 0 9 9 9 10 0 0 4/8");
        clk.add(10.0);
        let started = clk.get();
        clk.add(0.1);
        c.feed("XSLOT 0 0 0 0 0 0 0 -");
        c.feed("XSLOT 0 10 30 60 90 10 0 8/10,4/8");
        clk.add(0.5);
        c.feed("XSLOT 0 0 0 0 0 0 0 -");
        c.feed("XSLOT 0 5 10 30 40 0 0 2/4");
        let s = v(c.slot(0, Some(started)));
        assert_eq!(s["accept_pos"], json!([round_n(10.0 / 14.0, 3), 0.5]));
        assert_eq!((s["hit"]["vram"].clone(), s["hit"]["cpu"].clone()),
                   (json!(round_n(90.0 / 140.0, 3)), json!(round_n(40.0 / 140.0, 3))));
        assert_eq!((s["drafts_accepted"].clone(), s["drafts_offered"].clone()), (json!(14), json!(22)));
        assert_eq!(v(c.slot(0, None))["accept_pos"], json!([0.5]));
        clk.add(5.0);
        assert_eq!(v(c.slot(0, Some(clk.get() - 1.0))), json!({"accept_pos": null, "accept_hist": null, "hit": null}));
    }

    #[test]
    fn the_expert_map_fades() {
        let clk = Clk::new();
        let c = clk.lines();
        assert!(c.expert_map().is_none());
        c.feed("XMAP 8 0123 2,0,0,4");
        let m = v(c.expert_map().unwrap());
        assert_eq!((m["total"].clone(), m["tier"].clone(), m["routed"].clone(), m["recent"].clone()),
                   (json!(8), json!([0, 1, 2, 3]), json!([2.0, 0.0, 0.0, 4.0]), json!([2.0, 0.0, 0.0, 4.0])));
        clk.add(0.5);
        c.feed("XMAP 8 0123 1,0,0,0");
        let m = v(c.expert_map().unwrap());
        assert_eq!(m["routed"], json!([2.0, 0.0, 0.0, 2.67]));
        assert!((m["recent"][0].as_f64().unwrap() - (2.0 * 0.85f64.powf(0.5) + 1.0)).abs() < 0.005);
        clk.add(10.0);
        let m = v(c.expert_map().unwrap());
        assert_eq!(m["routed"], json!([0, 0, 0, 0]));
        assert!(m["recent"][3].as_f64().unwrap() < 1.0);
        assert_eq!(m.as_object().unwrap().keys().collect::<Vec<_>>(), ["total", "window_s", "tier", "routed", "recent"]);
    }

    #[test]
    fn cards_and_groups() {
        let cards = parse_cards(Some(&json!("0000:01:00.0/9052/52.1/23869/512;0000:08:00.0/5497/-/14500/0;bad/1/2")));
        assert_eq!(v(cards), json!({"0:1:0.0": {"expert_slots": 9052, "h2d_gbps": 52.1, "experts": 23869i64 << 20, "kv": 512 << 20},
                                    "0:8:0.0": {"expert_slots": 5497, "h2d_gbps": null, "experts": 14500i64 << 20, "kv": 0}}));
        assert_eq!(parse_l3(Some(&json!("0-2,8;3-4"))), vec![vec![0, 1, 2, 8], vec![3, 4]]);
        let mut info = json!({"cards": "0000:01:00.0/9052/52.1/20480/-;0000:08:00.0/5497/-/10240/0", "cpu_l3": "0-1;2-3",
                              "expert_slots": 388}).as_object().unwrap().clone();
        let (cards, l3) = engine_facts(&mut info);
        assert_eq!(v(info), json!({"expert_slots": 14549}));
        let g = 1i64 << 30;
        let now = json!({"gpus": [{"index": 0, "mem_used": 30 * g}, {"index": 1, "mem_used": 12 * g}],
                         "cpu_cores": [10.0, 30.0, 50.0, 50.0], "ram_used": 100 * g, "ram_cached": 20 * g});
        let st = json!({"gpus": [{"index": 0, "bus": "0000:01:00.0"}, {"index": 1, "bus": "00000000:08:00.0"}]});
        let info2 = json!({"ram_experts_mib": 60 << 10});
        let hw = v(hardware_view(now.as_object().unwrap(), st.as_object().unwrap(), info2.as_object().unwrap(), &cards, &l3));
        assert_eq!(hw["gpus"][0]["expert_slots"], json!(9052));
        assert_eq!(hw["gpus"][1]["h2d_gbps"], Value::Null);
        assert_eq!(hw["gpus"][0]["vram_parts"], json!({"experts": 20 * g, "kv": null, "other": 10 * g}));
        assert_eq!(hw["gpus"][1]["vram_parts"], json!({"experts": 10 * g, "kv": 0, "other": 2 * g}));
        assert_eq!(hw["cpu_ccd"], json!([20.0, 50.0]));
        assert_eq!(hw["ram_parts"], json!({"experts": 60 * g, "page_cache": 20 * g, "other": 40 * g}));
        assert!(hw.get("cpu_cores").is_none());
        assert_eq!(hw.as_object().unwrap().keys().collect::<Vec<_>>(), ["gpus", "ram_used", "cpu_ccd", "ram_parts"]);
        let bare = v(hardware_view(json!({"gpus": [{"index": 0}], "ram_used": 1, "ram_cached": null}).as_object().unwrap(),
                                   &Map::new(), &Map::new(), &Map::new(), &[]));
        assert_eq!((bare["cpu_ccd"].clone(), bare["ram_parts"].clone(), bare["gpus"][0]["expert_slots"].clone()),
                   (Value::Null, Value::Null, Value::Null));
    }
}
