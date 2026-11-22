//! Service.run and the per-request bookkeeping it updates (_note, _slot_of, live_line, _progress).

use crate::api::logprobs::{self, Channels};
use crate::error::ApiError;
use crate::frontend::{Event, OutputParse, TextPlace, EXAMPLE_TURN};
use crate::py::{self, Obj};
use crate::service::*;
use serde_json::{json, Value};
use serve_engine::errors::EngineErr;
use serve_engine::gen::{Gen, Stop, Yielder};
use serve_engine::iface::{thread_ident, Flag, Last, Prompt, Sampling};
use serve_engine::live::LiveRequest;
use serve_engine::util::stdout::out;
use serve_text::pyjson::{py_fixed, py_round};
use serve_text::text::{stop_strings, Detokenizer, StopMatcher};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

/// What run() takes: Python's arguments plus the request context (path, model), the embeddings file of prepare()
/// and the handle that learns the run thread's identity.
pub struct RunArgs {
    pub ids: Prompt,
    pub thinking: bool,
    pub tools: Option<Value>,
    pub max_new: i64,
    /// the request dict (sampling fields, `_grammar`, `_logprobs`) and the checked logit bias
    pub sampling: Sampling,
    pub cancel: Flag,
    pub force: Option<String>,
    pub session: Option<String>,
    pub live: Option<Arc<LiveRequest>>,
    pub path: Option<String>,
    pub model: Option<String>,
    pub embeddings: Option<String>,
    pub thread: RunThread,
}

fn api_err(e: ApiError) -> EngineErr {
    match e {
        ApiError::Engine(e) => e,
        ApiError::Responses(r) => EngineErr::Value(r.message),
        ApiError::Internal(m) => EngineErr::Internal(m),
    }
}

fn mono() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

fn ival(v: Option<&Value>) -> i64 {
    v.and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64))).unwrap_or(0)
}

fn fval(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| x.as_f64())
}

fn nn(v: Option<&Value>) -> Option<&Value> {
    v.filter(|x| !x.is_null())
}

fn ids_of(p: &Prompt) -> Vec<u32> {
    match p {
        Prompt::Ids(v) => v.clone(),
        Prompt::Text(t) => t.ids(),
    }
}

struct Run {
    svc: Arc<Service>,
    ids: Prompt,
    thinking: bool,
    max_new: i64,
    sampling: Sampling,
    budget_req: Obj,
    cancel: Flag,
    force: Option<String>,
    session: Option<String>,
    live: Option<Arc<LiveRequest>>,
    path: Option<String>,
    model: String,
    emb: Option<String>,
    budget: Option<i64>,
    budget_ready: bool,
    parser: Box<dyn OutputParse>,
    detok: Detokenizer,
    n: i64,
    finish: &'static str,
    lp: Option<Channels>,
    thinking_n: i64,
    stops: Option<StopMatcher>,
    one_call: bool,
    opening: Vec<Event>,
    tail: String,
    reused0: Option<i64>,
    prompt_ms0: Option<Value>,
    timings: Option<Obj>,
    before: Option<u64>,
    raw_ids: Vec<u32>,
    last0: Option<u64>,
    counted_last: Option<u64>,
    timed_last: Option<u64>,
    par: bool,
    which: Which2,
    wait_key: u64,
    /// stop token as text: where the reply's text has got to, stop ids written as text, the text's last piece the
    /// model writes itself, turn starts written as examples (still open)
    place: TextPlace,
    literals: i64,
    owed: Option<Vec<u32>>,
    examples: i64,
}

/// serve/service.py STOP_TEXT_MAX: stop ids written as text per reply, at most
const STOP_TEXT_MAX: i64 = 64;

#[derive(Clone, Copy)]
enum Which2 {
    Main,
    Live(u64),
}

static NEXT_KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl StatusAll {
    fn st2(&mut self, w: Which2) -> &mut Obj {
        match w {
            Which2::Main => &mut self.status,
            Which2::Live(id) => match self.live_reqs.iter().position(|(k, _, _)| *k == id) {
                Some(i) => &mut self.live_reqs[i].1,
                None => &mut self.status,
            },
        }
    }
    fn rate2(&mut self, w: Which2) -> &mut Rate {
        match w {
            Which2::Main => &mut self.rate,
            Which2::Live(id) => match self.live_reqs.iter().position(|(k, _, _)| *k == id) {
                Some(i) => &mut self.live_reqs[i].2,
                None => &mut self.rate,
            },
        }
    }
}

impl Run {
    fn st_get(&self, k: &str) -> Value {
        let mut s = self.svc.status_lock.lock().unwrap();
        s.st2(self.which).get(k).cloned().unwrap_or(Value::Null)
    }

    fn cut(&mut self, evs: Vec<Event>) -> Vec<Event> {
        let Some(stops) = self.stops.as_mut() else { return evs };
        let mut outv = Vec::new();
        for ev in evs {
            if stops.hit.is_some() {
                break;
            }
            if ev.kind == "content" {
                let text = stops.push(&ev.text);
                if !text.is_empty() {
                    outv.push(Event::text("content", text));
                }
                continue;
            }
            let held = stops.flush();
            if !held.is_empty() {
                outv.push(Event::text("content", held));
            }
            outv.push(ev);
        }
        outv
    }

    fn clean(&self) -> bool {
        self.parser.buf().is_empty() && !self.detok.pending()
    }

    fn dequeue(&self) {
        let mut s = self.svc.status_lock.lock().unwrap();
        let q = ival(s.status.get("queued")) - 1;
        s.status.insert("queued".into(), Value::from(q));
        s.waiting.retain(|(k, _)| *k != self.wait_key);
    }

    fn done_cancel(&self, y: &Yielder<RunItem>) -> Result<(), Stop> {
        let d = json!({"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": null, "reasoning_tokens": 0});
        y.yield_(RunItem::Done(d.as_object().unwrap().clone()))?;
        Ok(())
    }

    fn feed(&mut self, piece: &str) -> Result<Vec<Event>, EngineErr> {
        let evs = self.parser.feed(piece).map_err(api_err)?;
        for ev in &evs {
            if ev.kind == "reasoning" || ev.kind == "content" {
                self.place.feed(ev.kind, &ev.text);
            }
        }
        Ok(self.cut(evs))
    }

    /// stop_as_text: stop id `t` came where no turn ends -> (the ids of the token's text, where it was); None: it
    /// ends the reply. `room`: the output tokens left in place of `t`.
    fn stop_as_text(&self, t: u32, room: i64) -> Option<(Vec<u32>, &'static str)> {
        let svc = &self.svc;
        if !svc.cfg.read().unwrap().stop_token_text || self.literals >= STOP_TEXT_MAX || self.detok.pending() {
            return None;
        }
        let state = self.parser.state();
        let why = if state == "call" {
            if self.parser.in_value() { Some("in a tool call's argument") } else { None }
        } else if !py::strip(self.parser.buf()).is_empty() {
            None
        } else {
            self.place.with_held(state, self.parser.buf()).literal_context(
                state,
                self.examples > 0,
                svc.dialect.trims_messages(),
            )
        }?;
        let text = svc.tok.decode(&[t]).unwrap_or_default();
        let ids = svc.tok.encode(&text, false, &[]);
        if ids.is_empty() || ids.len() as i64 + 1 > room || ids.iter().any(|i| svc.stop_ids.contains(i)) {
            return None;
        }
        Some((ids, why))
    }

    /// written: ids the server writes into the reply as output (stop token as text); the parser, the stream and
    /// the count see them as the model's.
    fn written(&mut self, ids: &[u32], y: &Yielder<RunItem>) -> Result<(), Stop> {
        for &x in ids {
            self.n += 1;
            self.raw_ids.push(x);
            self.thinking_n += (self.parser.state() == "reasoning") as i64;
            let piece = self.detok.push(x);
            self.tail = py::tail(&format!("{}{}", self.tail, piece), 2).to_string();
            let state0 = self.parser.state();
            let evs = self.feed(&piece)?;
            if self.lp.is_some() {
                let state1 = self.parser.state();
                let items = self.lp.as_mut().unwrap().token(None, state0, state1, &evs);
                for item in items {
                    y.yield_(RunItem::Logprobs(item))?;
                }
            }
            self.svc.note(self.n, &evs, self.which_main(), self.wait_key_opt());
            for ev in evs {
                y.yield_(RunItem::Event(ev))?;
            }
        }
        Ok(())
    }

    fn body(&mut self, y: &Yielder<RunItem>) -> Result<(), Stop> {
        let r = self.queued_and_turn(y);
        if let Some(e) = self.emb.take() {
            let _ = std::fs::remove_file(e);
        }
        r?;
        if self.finish == "__cancelled_in_queue" {
            return Ok(());
        }
        let evs = self.parser.finish().map_err(api_err)?;
        let last_evs = self.cut(evs);
        if let Some(ch) = self.lp.as_mut() {
            if self.stops.as_ref().map(|s| s.hit.is_some()).unwrap_or(false) {
                ch.pending.clear();
            }
            for item in ch.finish(&last_evs) {
                y.yield_(RunItem::Logprobs(item))?;
            }
        }
        for ev in last_evs {
            y.yield_(RunItem::Event(ev))?;
        }
        if let Some(stops) = self.stops.as_mut() {
            if stops.hit.is_none() && !stops.held.is_empty() {
                let held = stops.flush();
                y.yield_(RunItem::Event(Event::text("content", held)))?;
            }
        }
        let mut done = Obj::new();
        done.insert("finish".into(), Value::from(self.finish));
        done.insert("completion_tokens".into(), Value::from(self.n));
        done.insert(
            "reused".into(),
            Value::from(self.timings.as_ref().map(|t| ival(t.get("cache_n"))).unwrap_or(0)),
        );
        done.insert("timings".into(), self.timings.clone().map(Value::Object).unwrap_or(Value::Null));
        done.insert("reasoning_tokens".into(), Value::from(self.thinking_n));
        done.insert("prompt_tokens".into(), Value::from(self.ids.len()));
        if let Some(hit) = self.stops.as_ref().and_then(|s| s.hit.clone()) {
            done.insert("stop_sequence".into(), Value::from(hit));
        }
        y.yield_(RunItem::Done(done))?;
        Ok(())
    }

    fn queued_and_turn(&mut self, y: &Yielder<RunItem>) -> Result<(), Stop> {
        let svc = self.svc.clone();
        // a queued VRAM resize goes first
        let open_gate = match svc_vram_gate(&svc, y, &self.cancel) {
            Ok(g) => g,
            Err(e) => {
                self.dequeue();
                return Err(e);
            }
        };
        if !open_gate {
            self.dequeue();
            self.done_cancel(y)?;
            self.finish = "__cancelled_in_queue";
            return Ok(());
        }
        let mut have_fifo = false;
        if svc.engine.batch() == 0 {
            loop {
                if svc.fifo.acquire(Some(Duration::from_secs_f64(QUEUE_BEAT_S))) {
                    have_fifo = true;
                    break;
                }
                if self.cancel.is_set() {
                    break;
                }
                if let Err(e) = y.yield_(RunItem::Ping) {
                    self.dequeue();
                    return Err(e.into());
                }
            }
            if !have_fifo {
                self.dequeue();
                self.done_cancel(y)?;
                self.finish = "__cancelled_in_queue";
                return Ok(());
            }
        }
        let r = self.turn(y);
        let r = match r {
            Err(Stop::Closed) => {
                self.finish = "disconnect";
                Err(Stop::Closed)
            }
            other => other,
        };
        self.settle();
        if have_fifo {
            svc.fifo.release();
        }
        r
    }

    fn turn(&mut self, y: &Yielder<RunItem>) -> Result<(), Stop> {
        let svc = self.svc.clone();
        self.dequeue();
        if self.par {
            svc.fifo.acquire(None);
            let r = svc.ensure_loaded();
            svc.fifo.release();
            r?;
        } else {
            svc.ensure_loaded()?;
        }
        let started = now();
        {
            let mut s = svc.status_lock.lock().unwrap();
            let api = if self.path.as_deref() == Some("/v1/messages") { "anthropic" } else { "openai" };
            let upd = json!({"busy": true, "phase": "reading the prompt", "prompt_tokens": self.ids.len(),
                "generated": 0, "started": fv(started), "first_token": null, "tool": null, "tail": "",
                "path": self.path, "prompt_ms": null, "decode_ms": null, "decoded": null, "passes": null,
                "windows": null, "window_tokens": null, "max_tokens": self.max_new, "slot": null, "reused": null,
                "thread": thread_ident(), "api": api, "model": self.model});
            if self.par {
                let key = self.wait_key;
                let st = upd.as_object().unwrap().clone();
                s.live_reqs.push((key, st.clone(), VecDeque::new()));
                for (k, v) in st {
                    s.status.insert(k, v);
                }
                self.which = Which2::Live(key);
            } else {
                for (k, v) in upd.as_object().unwrap() {
                    s.status.insert(k.clone(), v.clone());
                }
            }
            s.last_request_at = Some(now());
            s.rate2(self.which).clear();
        }
        self.before = svc.engine.last().map(|l| l.id);
        let mut last_print = now();
        if let Some(l) = &self.live {
            l.set_state("prompt");
        }
        let mut prompt = self.ids.clone();
        let mut thought: i64 = 0;
        let soft_stops = self.live.is_some() && svc.engine.supports_stop_soon();
        let mut closed_early = false;
        let mut pt_settled = false;
        let mut run_id: Option<u32> = None;
        let mut run_n: i64 = 0;
        let is_text = matches!(self.ids, Prompt::Text(_));
        for ev in std::mem::take(&mut self.opening) {
            y.yield_(RunItem::Event(ev))?;
        }
        let loop_tokens = svc.cfg.read().unwrap().thinking_loop_tokens;
        let start_ids = svc.tok.encode(&svc.dialect.turn_start(), true, &[]);
        let turn_start_id = if start_ids.len() == 1 { Some(start_ids[0]) } else { None };
        loop {
            let soft = if soft_stops { Some(Flag::new()) } else { None };
            if self.thinking && !self.budget_ready && !is_text {
                self.budget = svc.answer_budget(Some(&self.budget_req), self.max_new, self.budget).map_err(api_err)?;
                self.budget_ready = true;
            }
            let mut cap = self.max_new - self.n;
            let budget_on = self.budget.map(|b| b != 0).unwrap_or(false);
            let capped = budget_on && self.parser.state() == "reasoning" && self.budget.unwrap() - thought < cap;
            if capped {
                cap = (self.budget.unwrap() - thought).max(1);
            }
            let mut gen = svc.engine.generate(
                prompt.clone(),
                cap,
                &self.sampling,
                self.cancel.clone(),
                self.emb.clone(),
                self.session.clone(),
                soft.clone(),
            );
            let n_pass0 = self.n;
            let mut seg: Vec<u32> = Vec::new();
            let (mut wrap, mut leaving, mut looped, mut asked, mut drained, mut resumed, mut ended_thinking, mut opens) =
                (false, false, false, false, false, false, false, false);
            let mut relit = false; // this pass ended on a stop id written as text
            let mut err: Option<Stop> = None;
            let mut last_out = mono();
            'pass: loop {
                let item = match gen.next() {
                    None => {
                        drained = true;
                        break 'pass;
                    }
                    Some(Ok(x)) => x,
                    Some(Err(e)) => {
                        self.engine_failed(&e);
                        err = Some(Stop::Raised(e));
                        break 'pass;
                    }
                };
                if let Prompt::Text(pt) = &self.ids {
                    if pt.n().is_some() && !pt_settled {
                        pt_settled = true;
                        if let Some(m) = pt.max_new().filter(|m| *m != 0) {
                            if m < cap {
                                self.max_new = self.max_new.min(n_pass0 + m);
                            }
                        }
                    }
                }
                let counted = match &self.ids {
                    Prompt::Text(pt) => pt.n().is_some(),
                    _ => true,
                };
                if self.thinking && !self.budget_ready && counted {
                    match svc.answer_budget(Some(&self.budget_req), self.max_new, self.budget) {
                        Ok(b) => self.budget = b,
                        Err(e) => {
                            let e = api_err(e);
                            self.engine_failed(&e);
                            err = Some(Stop::Raised(e));
                            break 'pass;
                        }
                    }
                    self.budget_ready = true;
                }
                let Some(t) = item else {
                    if let Prompt::Text(pt) = &self.ids {
                        if let Some(n) = pt.n() {
                            let mut s = svc.status_lock.lock().unwrap();
                            let st = s.st2(self.which);
                            if st.get("prompt_tokens") != Some(&Value::from(n)) {
                                st.insert("prompt_tokens".into(), Value::from(n));
                                st.insert("max_tokens".into(), Value::from(self.max_new));
                            }
                        }
                    }
                    last_print = self.progress(last_print, 1.0);
                    last_out = mono();
                    if y.yield_(RunItem::Ping).is_err() {
                        leaving = true;
                        err = Some(Stop::Closed);
                        break 'pass;
                    }
                    continue;
                };
                self.n += 1;
                if svc.stop_ids.contains(&t.id) {
                    if let Some(owed) = self.owed.take() {
                        // it ends right after the text it began: the text's last piece first
                        self.n -= 1;
                        if let Err(e) = self.written(&owed, y) {
                            leaving = matches!(e, Stop::Closed);
                            err = Some(e);
                            break 'pass;
                        }
                        seg.extend(&owed);
                        self.n += 1;
                    }
                    if let Some((lit, why)) = self.stop_as_text(t.id, self.max_new - self.n + 1) {
                        // the token's text in place of its id, and the reply goes on from there; its last piece is
                        // left to the model, which can write the closing quote, backtick or comma with it
                        let (lit, owed) = if lit.len() > 1 {
                            (lit[..lit.len() - 1].to_vec(), Some(lit[lit.len() - 1..].to_vec()))
                        } else {
                            (lit, None)
                        };
                        self.owed = owed;
                        self.literals += 1;
                        if why == EXAMPLE_TURN {
                            self.examples -= 1;
                        }
                        out(&format!(
                            "[server] the model wrote {}'s id {why}: kept as its text and the reply goes on (--no-stop-token-text turns this off)",
                            svc.tok.decode(&[t.id]).unwrap_or_default()
                        ));
                        self.n -= 1;
                        if let Err(e) = self.written(&lit, y) {
                            leaving = matches!(e, Stop::Closed);
                            err = Some(e);
                            break 'pass;
                        }
                        seg.extend(&lit);
                        if self.stops.as_ref().map(|s| s.hit.is_some()).unwrap_or(false) {
                            self.finish = "stop";
                        } else {
                            relit = true;
                        }
                        break 'pass;
                    }
                    if self.parser.state() == "reasoning" && !closed_early && self.clean() {
                        self.n -= 1;
                        ended_thinking = true;
                        break 'pass;
                    }
                    self.finish = "stop";
                    self.raw_ids.push(t.id);
                    break 'pass;
                }
                self.owed = None;
                if Some(t.id) == turn_start_id && py::strip(self.parser.buf()).is_empty() {
                    let state = self.parser.state();
                    if self.place.with_held(state, self.parser.buf()).example_place(state) {
                        self.examples += 1;
                    }
                }
                self.raw_ids.push(t.id);
                seg.push(t.id);
                if self.parser.state() == "reasoning" {
                    run_n = if run_id == Some(t.id) { run_n + 1 } else { 1 };
                    run_id = Some(t.id);
                } else {
                    run_n = 0;
                }
                self.thinking_n += (self.parser.state() == "reasoning") as i64;
                let piece = self.detok.push(t.id);
                self.tail = py::tail(&format!("{}{}", self.tail, piece), 2).to_string();
                let state0 = self.parser.state();
                let evs = match self.feed(&piece) {
                    Ok(e) => e,
                    Err(e) => {
                        self.engine_failed(&e);
                        err = Some(Stop::Raised(e));
                        break 'pass;
                    }
                };
                if self.lp.is_some() {
                    let entry = t.lp.as_ref().map(|_| logprobs::entry(svc.tok.as_ref(), &t));
                    let state1 = self.parser.state();
                    let items = self.lp.as_mut().unwrap().token(entry, state0, state1, &evs);
                    for item in items {
                        if y.yield_(RunItem::Logprobs(item)).is_err() {
                            leaving = true;
                            err = Some(Stop::Closed);
                            break 'pass;
                        }
                    }
                }
                svc.note(self.n, &evs, self.which_main(), self.wait_key_opt());
                if self.st_get("reused").is_null() {
                    let run_last = svc.engine.last();
                    if let Some(rl) = &run_last {
                        if Some(rl.id) != self.last0 {
                            if let Some(r) = nn(rl.map.get("reused")) {
                                let v = ival(Some(r)).min(self.ids.len() as i64);
                                let mut s = svc.status_lock.lock().unwrap();
                                s.st2(self.which).insert("reused".into(), Value::from(v));
                            }
                        }
                    }
                }
                last_print = self.progress(last_print, 1.0);
                let had = !evs.is_empty();
                let tool_call = evs.iter().any(|e| e.kind == "tool_call");
                let tool_any = evs.iter().any(|e| e.kind == "tool_start" || e.kind == "tool_call");
                for ev in evs {
                    if y.yield_(RunItem::Event(ev)).is_err() {
                        leaving = true;
                        err = Some(Stop::Closed);
                        break 'pass;
                    }
                }
                if had {
                    last_out = mono();
                } else if mono() - last_out >= KEEPALIVE_S {
                    last_out = mono();
                    if y.yield_(RunItem::Ping).is_err() {
                        leaving = true;
                        err = Some(Stop::Closed);
                        break 'pass;
                    }
                }
                if self.stops.as_ref().map(|s| s.hit.is_some()).unwrap_or(false) {
                    self.finish = "stop";
                    break 'pass;
                }
                if loop_tokens != 0 && run_n >= loop_tokens && self.parser.state() == "reasoning" && self.clean() {
                    wrap = true;
                    looped = true;
                    break 'pass;
                }
                if self.one_call && tool_call {
                    self.finish = "stop";
                    break 'pass;
                }
                if self.budget.map(|b| b != 0).unwrap_or(false) && self.parser.state() == "reasoning" {
                    thought += 1;
                    if thought >= self.budget.unwrap() && self.clean() {
                        wrap = true;
                        break 'pass;
                    }
                }
                if let Some(l) = &self.live {
                    l.set_state(if self.parser.state() == "reasoning" { "thinking" } else { "answering" });
                    if l.answer_now.is_set() && self.parser.state() == "reasoning" {
                        if let Some(s) = &soft {
                            s.set();
                        } else if self.clean() {
                            wrap = true;
                            asked = true;
                            break 'pass;
                        }
                    }
                }
                if self.force.is_some() {
                    if tool_any {
                        self.force = None;
                    } else if self.parser.state() == "content" && self.clean() {
                        opens = true;
                        break 'pass;
                    }
                }
            }
            // finally: gen.close() STOPs and drains to this request's DONE
            if let Err(e) = gen.close() {
                if matches!(e, EngineErr::Silent(_)) {
                    self.finish = "error";
                    svc.say_died_pub(&e);
                    if !leaving && !self.cancel.is_set() {
                        err = Some(Stop::Raised(e));
                    }
                } else {
                    err = Some(Stop::Raised(e));
                }
            }
            if let Some(e) = err {
                return Err(e);
            }
            let run_last: Option<Last> = svc.engine.last();
            let rl_id = run_last.as_ref().map(|l| l.id);
            {
                let mut s = svc.status_lock.lock().unwrap();
                let st = s.st2(self.which);
                st.insert("run_windows".into(), Value::from(0));
                st.insert("run_tokens".into(), Value::from(0));
                if let Some(rl) = &run_last {
                    if rl_id != self.counted_last && py::truthy(rl.map.get("windows").unwrap_or(&Value::Null)) {
                        let w = ival(st.get("windows")) + ival(rl.map.get("windows"));
                        st.insert("windows".into(), Value::from(w));
                        let wt = ival(st.get("window_tokens")) + self.n - n_pass0;
                        st.insert("window_tokens".into(), Value::from(wt));
                        self.counted_last = rl_id;
                    }
                    if rl_id != self.timed_last && py::truthy(rl.map.get("decode_ms").unwrap_or(&Value::Null)) {
                        let dm = fval(st.get("decode_ms")).unwrap_or(0.0) + fval(rl.map.get("decode_ms")).unwrap_or(0.0);
                        st.insert("decode_ms".into(), fv(dm));
                        let dc = ival(st.get("decoded")) + ival(rl.map.get("generated"));
                        st.insert("decoded".into(), Value::from(dc));
                        let ps = ival(st.get("passes")) + 1;
                        st.insert("passes".into(), Value::from(ps));
                        self.timed_last = rl_id;
                    }
                }
            }
            if let Some(rl) = &run_last {
                if rl_id != self.last0 && rl.map.get("finish").and_then(|f| f.as_str()) == Some("error") {
                    self.finish = "error";
                    let e = "the grammar allowed no token to continue the reply; the engine log has the state";
                    out(&format!("[server] error: engine reported {e}"));
                    return Err(Stop::Raised(EngineErr::Error(e.into())));
                }
                if self.reused0.is_none() && rl_id != self.last0 {
                    if let Some(r) = nn(rl.map.get("reused")) {
                        self.reused0 = Some(ival(Some(r)).min(self.ids.len() as i64));
                        self.prompt_ms0 = nn(rl.map.get("prompt_ms")).cloned();
                        if let Some(pm) = &self.prompt_ms0 {
                            let mut s = svc.status_lock.lock().unwrap();
                            s.st2(self.which).insert("prompt_ms".into(), pm.clone());
                        }
                    }
                }
            }
            let reasoning = self.parser.state() == "reasoning";
            if drained && capped && !wrap && self.n - n_pass0 >= cap && self.n < self.max_new && !self.cancel.is_set() {
                if reasoning && thought >= self.budget.unwrap_or(0) && self.clean() {
                    wrap = true;
                } else {
                    resumed = true;
                }
            }
            if drained && soft.as_ref().map(|s| s.is_set()).unwrap_or(false) && self.n < self.max_new && !self.cancel.is_set() {
                if reasoning && self.clean() {
                    wrap = true;
                    asked = true;
                } else {
                    resumed = true;
                }
            }
            if !(wrap || opens || ended_thinking || resumed || relit) || self.cancel.is_set() {
                break;
            }
            let extra: Vec<u32>;
            let close = format!("{}{}", svc.dialect.reasoning_close, self.force.clone().unwrap_or_default());
            if ended_thinking {
                closed_early = true;
                extra = svc.tok.encode(&close, true, &[]);
                self.force = None;
                if self.max_new - self.n - (extra.len() as i64) < 1 {
                    self.finish = "stop";
                    break;
                }
                out("[server] the model ended while thinking, closing it for an answer");
            } else if wrap {
                self.budget = None;
                extra = svc.tok.encode(&close, true, &[]);
                self.force = None;
                if self.max_new - self.n - (extra.len() as i64) < 1 {
                    break;
                }
                if asked {
                    out(&format!("[server] answer now: closing the thinking after {} tokens", self.thinking_n));
                } else if looped {
                    out(&format!("[server] thinking repeated one token {run_n} times, ending it"));
                } else {
                    out(&format!("[server] thinking budget reached ({thought} tokens), wrapping up"));
                }
            } else if resumed || relit {
                extra = vec![];
            } else {
                let trailing = self.tail.len() - self.tail.trim_end_matches('\n').len();
                let count = svc.dialect.call_newlines.saturating_sub(trailing);
                let text = format!("{}{}", "\n".repeat(count), self.force.clone().unwrap_or_default());
                self.force = None;
                extra = svc.tok.encode(&text, true, &[]);
                if self.max_new - self.n - (extra.len() as i64) < 1 {
                    break;
                }
            }
            for &t in &extra {
                self.n += 1;
                self.raw_ids.push(t);
                self.thinking_n += (self.parser.state() == "reasoning") as i64;
                let state0 = self.parser.state();
                let piece = self.detok.push(t);
                let evs = self.feed(&piece)?;
                if self.lp.is_some() {
                    let state1 = self.parser.state();
                    let items = self.lp.as_mut().unwrap().token(None, state0, state1, &evs);
                    for item in items {
                        y.yield_(RunItem::Logprobs(item))?;
                    }
                }
                svc.note(self.n, &evs, self.which_main(), self.wait_key_opt());
                for ev in evs {
                    y.yield_(RunItem::Event(ev))?;
                }
            }
            if let Some(l) = &self.live {
                l.set_state(if self.parser.state() == "reasoning" { "thinking" } else { "answering" });
            }
            if self.stops.as_ref().map(|s| s.hit.is_some()).unwrap_or(false) {
                self.finish = "stop";
                break;
            }
            let mut next = ids_of(&prompt);
            next.extend(&seg);
            next.extend(&extra);
            prompt = Prompt::Ids(next);
        }
        if self.cancel.is_set() {
            self.finish = "cancel";
        } else if self.finish == "length" && self.n < self.max_new && svc.stopping.is_set() {
            self.finish = "error";
            return Err(Stop::Raised(EngineErr::Died("the server is stopping".into())));
        }
        Ok(())
    }

    fn which_main(&self) -> bool {
        matches!(self.which, Which2::Main)
    }

    fn wait_key_opt(&self) -> Option<u64> {
        match self.which {
            Which2::Live(k) => Some(k),
            Which2::Main => None,
        }
    }

    fn engine_failed(&mut self, e: &EngineErr) {
        if e.is_died() {
            self.finish = "error";
            self.svc.say_died_pub(e);
        } else if e.is_value() {
            self.finish = "error";
            out(&format!("[server] error: engine reported {}", e.message()));
        }
    }

    fn progress(&self, last_print: f64, every: f64) -> f64 {
        self.svc.progress_line(last_print, every, self.wait_key_opt())
    }

    /// The finally of the turn: status and history settled before the next request sets its own.
    fn settle(&mut self) {
        let svc = self.svc.clone();
        let fresh = svc.engine.last();
        let mut guard = svc.status_lock.lock().unwrap();
        let s = &mut *guard;
        let w = self.which;
        let busy = py::truthy(s.st2(w).get("busy").unwrap_or(&Value::Null));
        if busy {
            let mut last: Obj = match &fresh {
                Some(l) if Some(l.id) != self.last0 => l.map.clone(),
                _ => Obj::new(),
            };
            let st = s.st2(w).clone();
            if ival(st.get("passes")) > 1 && nn(last.get("decode_ms")).is_some() {
                last.insert("decode_ms".into(), st.get("decode_ms").cloned().unwrap_or(Value::Null));
                last.insert("generated".into(), st.get("decoded").cloned().unwrap_or(Value::Null));
            }
            if let Some(r0) = self.reused0 {
                if nn(last.get("reused")).is_some() {
                    last.insert("reused".into(), Value::from(r0));
                    if let Some(pm) = &self.prompt_ms0 {
                        last.insert("prompt_ms".into(), pm.clone());
                    }
                }
            }
            let started = fval(st.get("started")).unwrap_or_else(now);
            let lookups = ival(last.get("lookups"));
            let hit_rate = if lookups != 0 { Some(py_round(ival(last.get("hits")) as f64 / lookups as f64, 3)) } else { None };
            let routed = lookups + ival(last.get("offloaded"));
            let pcie = if nn(last.get("offloaded")).is_some() && routed != 0 {
                Some(py_round(ival(last.get("offloaded")) as f64 / routed as f64, 3))
            } else {
                None
            };
            let ssd = if ival(last.get("file_reads")) != 0 && routed != 0 {
                Some(py_round(ival(last.get("file_reads")) as f64 / routed as f64, 3))
            } else {
                None
            };
            let seen = prompt_tokens_seen(self.ids.len() as i64, &last);
            s.requests_done += 1;
            let n = self.n;
            for (key, add) in [
                ("prompt", Value::from(seen)),
                ("reused", Value::from(ival(last.get("reused")))),
                ("written", Value::from(n)),
                ("prompt_ms", fv(fval(nn(last.get("prompt_ms"))).unwrap_or(0.0))),
                ("decode_ms", fv(fval(nn(last.get("decode_ms"))).unwrap_or(0.0))),
            ] {
                let cur = s.totals.get(key).cloned().unwrap_or(Value::from(0));
                let sum = if cur.is_f64() || add.is_f64() {
                    fv(cur.as_f64().unwrap_or(0.0) + add.as_f64().unwrap_or(0.0))
                } else {
                    Value::from(cur.as_i64().unwrap_or(0) + add.as_i64().unwrap_or(0))
                };
                s.totals.insert(key.into(), sum);
            }
            if let Some(slot) = nn(st.get("slot")).and_then(|v| v.as_i64()) {
                s.slot_last.insert(
                    slot,
                    json!({"ctx": seen + n, "offered": last.get("drafts_offered").cloned(),
                           "accepted": last.get("drafts_accepted").cloned(), "at": fv(now())})
                    .as_object()
                    .unwrap()
                    .clone(),
                );
            }
            let ft = fval(nn(st.get("first_token")));
            let ttft = match (ft, fval(nn(st.get("started")))) {
                (Some(f), Some(st0)) if f != 0.0 && st0 != 0.0 => Some(f - st0),
                _ => None,
            };
            let windows = ival(st.get("windows"));
            let wtok = ival(st.get("window_tokens"));
            let dl = if windows != 0 { Some(wtok as f64 / windows as f64) } else { None };
            let slot = svc.slot_of_locked(s, self.wait_key_opt());
            let id = format!("r_{:04}", s.requests_done);
            let decode_tok_s = if n != 0 && ival(last.get("generated")) != 0 && fval(last.get("decode_ms")).map(|d| d != 0.0).unwrap_or(false)
            {
                fv(py_round(ival(last.get("generated")) as f64 / (fval(last.get("decode_ms")).unwrap() / 1000.0), 1))
            } else {
                Value::Null
            };
            let opt = |x: Option<f64>| x.map(fv).unwrap_or(Value::Null);
            let rec = json!({
                "id": id, "path": self.path, "slot": slot, "time": fv(started),
                "duration_s": fv(py_round(now() - started, 1)), "finish": self.finish,
                "prompt_tokens": seen, "reused": last.get("reused").cloned(), "output_tokens": n,
                "prompt_ms": last.get("prompt_ms").cloned(), "decode_tok_s": decode_tok_s,
                "hit_rate": opt(hit_rate), "pcie_share": opt(pcie), "ssd_share": opt(ssd),
                "drafts_offered": last.get("drafts_offered").cloned(),
                "drafts_accepted": last.get("drafts_accepted").cloned(),
                "ttft_s": opt(ttft.map(|t| py_round(t, 2))), "draft_len": opt(dl.map(|d| py_round(d, 2)))});
            let rec = rec.as_object().unwrap().clone();
            s.history.push_back(rec.clone());
            while s.history.len() > 500 {
                s.history.pop_front();
            }
            svc.note_best_locked(s, &rec);
            if let Some(f) = &fresh {
                if Some(f.id) != self.before {
                    self.timings = request_timings(seen, n, &last);
                    s.last_timings = self.timings.clone().map(|mut t| {
                        t.insert("at".into(), Value::from(now() as i64));
                        t
                    });
                }
            }
            s.last_request_at = Some(now());
            let rate = match ft {
                Some(f) if f != 0.0 => n as f64 / (now() - f).max(1e-6),
                _ => 0.0,
            };
            out(&done_line(
                slot,
                seen,
                n,
                self.finish,
                self.cancel.is_set(),
                self.timings.as_ref(),
                ttft,
                rate,
                dl,
                hit_rate,
                pcie,
                ssd,
            ));
            if self.finish == "length" && self.parser.state() == "reasoning" {
                out("[server] warning: max tokens reached while thinking, no answer (set reasoning_budget_tokens, or start with --reasoning-budget-tokens)");
            }
            if DEBUG.load(Ordering::Relaxed) && !self.raw_ids.is_empty() {
                let raw = svc.tok.decode(&self.raw_ids).unwrap_or_default();
                out(&format!("[server] raw: {}", serve_text::pyjson::py_str_repr(&raw)));
            }
        }
        if !(self.par && matches!(w, Which2::Main)) {
            let st = s.st2(w);
            st.insert("busy".into(), Value::Bool(false));
            st.shift_remove("tail");
            st.shift_remove("tool");
        }
        if self.par {
            if let Which2::Live(k) = w {
                s.live_reqs.retain(|(key, _, _)| *key != k);
            }
            if let Some((_, newest, _)) = s.live_reqs.last().cloned() {
                for (k, v) in newest {
                    s.status.insert(k, v);
                }
            } else {
                s.status.insert("busy".into(), Value::Bool(false));
                s.status.shift_remove("tail");
                s.status.shift_remove("tool");
            }
        }
    }
}

fn svc_vram_gate(svc: &Arc<Service>, y: &Yielder<RunItem>, cancel: &Flag) -> Result<bool, Stop> {
    svc.vram_gate_pub(y, cancel)
}

impl Service {
    /// run: yield events as text arrives, pings as heartbeat, logprobs entries, then the done record.
    pub fn run(self: &Arc<Self>, a: RunArgs) -> RunGen {
        let svc = self.clone();
        Gen::new(move |y: &Yielder<RunItem>| {
            a.thread.0.store(thread_ident(), Ordering::SeqCst);
            if let Some(l) = &a.live {
                l.set_thinking(Some(a.thinking));
            }
            let budget = if a.thinking { svc.reasoning_budget_of(Some(&a.sampling.map), true).map_err(api_err)? } else { None };
            let budget_req = a.sampling.map.clone();
            let mut sampling = a.sampling.clone();
            if !svc.sampling_defaults.is_empty() {
                let mut m = svc.sampling_defaults.clone();
                for (k, v) in &a.sampling.map {
                    if !v.is_null() {
                        m.insert(k.clone(), v.clone());
                    }
                }
                sampling.map = m;
            }
            let mut parser = svc.dialect.parser(a.thinking, a.tools.as_ref(), true);
            let lp = if nn(sampling.map.get(logprobs::REQUEST_KEY)).is_some() { Some(Channels::new(a.thinking)) } else { None };
            let stop_list = stop_strings(&sampling.map).map_err(EngineErr::Value)?;
            let stops = if stop_list.is_empty() { None } else { Some(StopMatcher::new(stop_list)) };
            let mut force = a.force.clone().filter(|f| !f.is_empty());
            let one_call = svc.dialect.forced_call_ends_reply
                && force.as_ref().map(|f| py::rstrip(f).ends_with('>')).unwrap_or(false);
            let mut opening = vec![];
            if let Some(f) = &force {
                if !a.thinking {
                    opening = parser.feed(&format!("{}{}", svc.dialect.nothink_lead, f)).map_err(api_err)?;
                    force = None;
                }
            }
            let par = svc.engine.batch() > 0;
            let wait_key = NEXT_KEY.fetch_add(1, Ordering::SeqCst);
            let last0 = svc.engine.last().map(|l| l.id);
            let model = a.model.clone().unwrap_or_else(|| svc.model.clone());
            {
                let mut s = svc.status_lock.lock().unwrap();
                let q = ival(s.status.get("queued")) + 1;
                s.status.insert("queued".into(), Value::from(q));
                if !par {
                    let api = if a.path.as_deref() == Some("/v1/messages") { "anthropic" } else { "openai" };
                    s.waiting.push((
                        wait_key,
                        json!({"api": api, "model": model, "prompt_tokens": a.ids.len(), "started": fv(now())})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ));
                }
            }
            let mut run = Run {
                svc: svc.clone(),
                ids: a.ids.clone(),
                thinking: a.thinking,
                max_new: a.max_new,
                sampling,
                budget_req,
                cancel: a.cancel.clone(),
                force,
                session: a.session.clone(),
                live: a.live.clone(),
                path: a.path.clone(),
                model,
                emb: a.embeddings.clone(),
                budget,
                budget_ready: false,
                detok: Detokenizer::new(svc.tok.clone()),
                parser,
                n: 0,
                finish: "length",
                lp,
                thinking_n: 0,
                stops,
                one_call,
                opening,
                tail: String::new(),
                reused0: None,
                prompt_ms0: None,
                timings: None,
                before: None,
                raw_ids: vec![],
                last0,
                counted_last: last0,
                timed_last: last0,
                par,
                which: Which2::Main,
                wait_key,
                place: TextPlace::default(),
                literals: 0,
                owed: None,
                examples: 0,
            };
            run.body(y)
        })
    }

    pub(crate) fn say_died_pub(&self, e: &EngineErr) {
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

    /// _note: the request's generated count, rate sample, phase and tail.
    pub(crate) fn note(&self, n: i64, evs: &[Event], main: bool, key: Option<u64>) {
        let w = if main { Which2::Main } else { Which2::Live(key.unwrap_or(0)) };
        let ws = self.engine.window_stats();
        let pp = self.engine.prompt_progress();
        let mut s = self.status_lock.lock().unwrap();
        let st = s.st2(w);
        st.insert("generated".into(), Value::from(n));
        if let Some((a, b)) = ws {
            st.insert("run_windows".into(), Value::from(a));
            st.insert("run_tokens".into(), Value::from(b));
        }
        if nn(st.get("first_token")).is_none() {
            st.insert("first_token".into(), fv(now()));
            let pp = pp.unwrap_or_default();
            if let Some(ms) = pp.get("ms").and_then(|m| m.as_f64()) {
                st.insert("prompt_ms".into(), fv(py_round(ms, 1)));
            }
            if nn(st.get("reused")).is_none() {
                if let Some(c) = pp.get("cache").and_then(|c| if crate::py::is_int(c) { c.as_i64() } else { None }) {
                    let pt = ival(st.get("prompt_tokens"));
                    let cap = if pt != 0 { pt } else { c };
                    st.insert("reused".into(), Value::from(c.min(cap)));
                }
            }
        }
        for ev in evs {
            match ev.kind {
                "reasoning" => {
                    st.insert("phase".into(), Value::from("thinking"));
                }
                "content" => {
                    st.insert("phase".into(), Value::from("answering"));
                }
                "tool_start" => {
                    let name = ev.call.as_ref().map(|c| c.name.clone()).unwrap_or_default();
                    st.insert("phase".into(), Value::from(format!("writing a tool call: {name}")));
                    st.insert("tool".into(), Value::from(name));
                }
                "tool_call" => {
                    st.insert("phase".into(), Value::from("tool call complete"));
                }
                _ => {}
            }
            let t = format!("{}{}", st.get("tail").and_then(|t| t.as_str()).unwrap_or(""), ev.text);
            st.insert("tail".into(), Value::from(py::tail(&t, 600)));
        }
        let r = s.rate2(w);
        r.push_back((now(), n));
        while r.len() > RATE_SAMPLES {
            r.pop_front();
        }
    }

    /// _slot_of with status_lock held.
    pub(crate) fn slot_of_locked(&self, s: &mut StatusAll, key: Option<u64>) -> Option<i64> {
        let total = self.engine.batch().max(1) as i64;
        let w = match key {
            Some(k) => Which2::Live(k),
            None => Which2::Main,
        };
        let cur = s.st2(w).get("slot").and_then(|v| v.as_i64());
        if cur.map(|c| c >= total).unwrap_or(true) {
            let taken: Vec<i64> = s
                .live_reqs
                .iter()
                .filter(|(k, _, _)| Some(*k) != key)
                .filter_map(|(_, o, _)| o.get("slot").and_then(|v| v.as_i64()))
                .filter(|&x| x < total)
                .collect();
            let slot = (0..total).find(|i| !taken.contains(i));
            s.st2(w).insert("slot".into(), slot.map(Value::from).unwrap_or(Value::Null));
            return slot;
        }
        cur
    }

    /// live_line: one line for every request that is generating.
    pub fn live_line(&self, t: f64) -> Option<String> {
        let mut rows: Vec<(i64, Obj)> = Vec::new();
        {
            let mut s = self.status_lock.lock().unwrap();
            let mut keys: Vec<Option<u64>> = s.live_reqs.iter().map(|(k, _, _)| Some(*k)).collect();
            if keys.is_empty() && py::truthy(s.status.get("busy").unwrap_or(&Value::Null)) {
                keys.push(None);
            }
            let mut reqs: Vec<(Option<u64>, f64)> = keys
                .iter()
                .map(|k| {
                    let w = k.map(Which2::Live).unwrap_or(Which2::Main);
                    (*k, fval(s.st2(w).get("started")).unwrap_or(0.0))
                })
                .collect();
            reqs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (k, _) in reqs {
                let w = k.map(Which2::Live).unwrap_or(Which2::Main);
                let st = s.st2(w).clone();
                if !py::truthy(st.get("busy").unwrap_or(&Value::Null)) || nn(st.get("first_token")).is_none() {
                    continue;
                }
                let slot = self.slot_of_locked(&mut s, k);
                let st = s.st2(w).clone();
                rows.push((slot.unwrap_or(-1), st));
            }
        }
        if rows.is_empty() {
            return None;
        }
        rows.sort_by_key(|r| r.0);
        let mut parts = Vec::new();
        let mut total = 0.0;
        for (slot, st) in rows {
            let seen = t - fval(st.get("first_token")).unwrap_or(t);
            let generated = ival(st.get("generated"));
            let rate = if seen >= LIVE_MIN_S { Some(generated as f64 / seen) } else { None };
            total += rate.unwrap_or(0.0);
            let mt = st.get("max_tokens").map(py::py_str).unwrap_or_else(|| "None".into());
            parts.push(format!(
                "{} tg={generated}/{mt}{}",
                slot_tag(if slot < 0 { None } else { Some(slot) }),
                rate.map(|r| format!(" {}t/s", py_fixed(r, 1))).unwrap_or_default()
            ));
        }
        Some(format!("[server] live | {} | total {}t/s", parts.join(" | "), py_fixed(total, 0)))
    }

    /// _progress: the prompt read lines, then the shared live line.
    pub(crate) fn progress_line(&self, last_print: f64, every: f64, key: Option<u64>) -> f64 {
        let t = now();
        let w = key.map(Which2::Live).unwrap_or(Which2::Main);
        let s = { self.status_lock.lock().unwrap().st2(w).clone() };
        let reading = nn(s.get("first_token")).is_none();
        if t - last_print < if reading { READ_PRINT_EVERY } else { every } {
            return last_print;
        }
        if reading {
            let pr = self.engine.progress();
            let done = match pr {
                Some((a, b)) if b != 0 => format!("{a}/{b}"),
                _ => format!("{}", ival(s.get("prompt_tokens"))),
            };
            let pp = if pr.map(|p| p.1 != 0).unwrap_or(false) { self.engine.prefill_tok_s_mean() } else { None };
            let speed = pp.filter(|p| *p > 0.0).map(|p| format!(" {}t/s", py_fixed(p, 0))).unwrap_or_default();
            let slot = {
                let mut g = self.status_lock.lock().unwrap();
                self.slot_of_locked(&mut g, key)
            };
            let started = fval(s.get("started")).unwrap_or(t);
            out(&format!("[server] {} read  pp={done}{speed} {}s", slot_tag(slot), py_fixed(t - started, 0)));
        } else {
            let due = {
                let mut la = self.live_at_lock();
                let due = t - *la >= LIVE_EVERY_S;
                if due {
                    *la = t;
                }
                due
            };
            if due {
                if let Some(line) = self.live_line(t) {
                    out(&line);
                }
            }
        }
        t
    }
}
