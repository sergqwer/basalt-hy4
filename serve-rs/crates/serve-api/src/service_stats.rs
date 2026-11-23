//! The Service figures: live rates, the slot table, /metrics, /v1/status and llama.cpp's prompt_progress.

use crate::py::{self, Obj};
use crate::service::*;
use serde_json::{json, Map, Value};
use serve_text::pyjson::py_round;

fn ival(v: Option<&Value>) -> i64 {
    v.and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64))).unwrap_or(0)
}

fn fval(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| x.as_f64())
}

fn nn(v: Option<&Value>) -> Option<&Value> {
    v.filter(|x| !x.is_null())
}

fn truthy(v: Option<&Value>) -> bool {
    v.map(py::truthy).unwrap_or(false)
}

/// Python's `dict(**a, **b)`-style update: keys of `b` replace in place or append.
fn merge(a: &mut Obj, b: &Obj) {
    for (k, v) in b {
        a.insert(k.clone(), v.clone());
    }
}

impl Service {
    /// telemetry_extra: this model's speeds, for the hardware sampler.
    pub fn telemetry_extra_v(&self) -> Value {
        let slots: Vec<Value> = self.slot_table_v().0.iter().map(|r| r.get("tok_s").cloned().unwrap_or(fv(0.0))).collect();
        json!({"tok_s": fv(self.tok_s()), "prefill_tok_s_mean": self.prefill_tok_s_mean().map(fv).unwrap_or(Value::Null),
               "slot_tok_s": slots})
    }

    /// _tok_s: tok/s over the last RATE_WINDOW_S (summed over parallel requests).
    pub fn tok_s(&self) -> f64 {
        let s = self.status_lock.lock().unwrap();
        if !s.live_reqs.is_empty() {
            let many: Vec<(Obj, Vec<(f64, i64)>)> =
                s.live_reqs.iter().map(|(_, st, r)| (st.clone(), r.iter().cloned().collect())).collect();
            drop(s);
            return many.iter().map(|(st, r)| rate_of(st, r)).sum();
        }
        let st = s.status.clone();
        let r: Vec<(f64, i64)> = s.rate.iter().cloned().collect();
        drop(s);
        rate_of(&st, &r)
    }

    /// _tok_s_mean: mean tok/s since the first token.
    pub fn tok_s_mean(&self) -> f64 {
        let s = self.status_lock.lock().unwrap();
        let many: Vec<Obj> = if s.live_reqs.is_empty() {
            vec![s.status.clone()]
        } else {
            s.live_reqs.iter().map(|(_, st, _)| st.clone()).collect()
        };
        drop(s);
        many.iter()
            .filter(|st| truthy(st.get("busy")) && truthy(st.get("first_token")))
            .map(|st| ival(st.get("generated")) as f64 / (now() - fval(st.get("first_token")).unwrap()).max(1e-6))
            .sum()
    }

    /// _prefill_tok_s_mean: the engine's mean over newly read tokens while reading, else 0.0.
    pub fn prefill_tok_s_mean(&self) -> Option<f64> {
        let reading = {
            let s = self.status_lock.lock().unwrap();
            truthy(s.status.get("busy")) && nn(s.status.get("first_token")).is_none()
        };
        if reading {
            self.engine.prefill_tok_s_mean()
        } else {
            Some(0.0)
        }
    }

    /// slot_stats: request slots, how many run, how many wait.
    pub fn slot_stats_v(&self) -> Value {
        let total = self.engine.batch() as i64;
        let (running, queued) = {
            let s = self.status_lock.lock().unwrap();
            let running = if total != 0 { s.live_reqs.len() as i64 } else { truthy(s.status.get("busy")) as i64 };
            (running, ival(s.status.get("queued")))
        };
        let waiting = if total != 0 { self.engine.queue_depth().unwrap_or(0) as i64 } else { 0 };
        let t = total.max(1);
        json!({"total": t, "active": (running - waiting).clamp(0, t).max(0).min(t), "queue_depth": queued + waiting})
    }

    fn engine_slots(&self, total: i64) -> Option<Vec<Obj>> {
        self.engine.slot_view().filter(|v| v.len() as i64 == total)
    }

    /// slot_table: per-slot rows and the waiting requests.
    pub fn slot_table_v(&self) -> (Vec<Obj>, Vec<Value>) {
        let base = self.slot_stats_v();
        let total = ival(base.get("total"));
        let active = ival(base.get("active"));
        let t = now();
        let view = self.engine_slots(total);
        let mut owners: Vec<(Value, i64)> = Vec::new();
        if let Some(v) = &view {
            for (b, row) in v.iter().enumerate() {
                if let Some(o) = nn(row.get("owner")) {
                    owners.push((o.clone(), b as i64));
                }
            }
        }
        let owner_of = |th: Option<&Value>| -> Option<i64> {
            let th = th?;
            owners.iter().find(|(o, _)| py::py_eq(o, th)).map(|(_, b)| *b)
        };
        let (by_slot, queue, last) = {
            let mut s = self.status_lock.lock().unwrap();
            // (key: None = the main status, Some(k) = a live request)
            let mut live: Vec<(Option<u64>, f64)> = if !s.live_reqs.is_empty() {
                s.live_reqs.iter().map(|(k, st, _)| (Some(*k), fval(st.get("started")).unwrap_or(0.0))).collect()
            } else if truthy(s.status.get("busy")) {
                vec![(None, fval(s.status.get("started")).unwrap_or(0.0))]
            } else {
                vec![]
            };
            live.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            fn get_st(s: &mut StatusAll, k: Option<u64>) -> &mut Obj {
                match k {
                    None => &mut s.status,
                    Some(k) => {
                        let i = s.live_reqs.iter().position(|(x, _, _)| *x == k).unwrap();
                        &mut s.live_reqs[i].1
                    }
                }
            }
            let (running, waiting): (Vec<Option<u64>>, Vec<Option<u64>>) = if !owners.is_empty() {
                let mut r = vec![];
                let mut w = vec![];
                for (k, _) in &live {
                    let th = get_st(&mut s, *k).get("thread").cloned();
                    match owner_of(th.as_ref()) {
                        Some(b) => {
                            get_st(&mut s, *k).insert("slot".into(), Value::from(b));
                            r.push(*k);
                        }
                        None => w.push(*k),
                    }
                }
                (r, w)
            } else {
                let a = (active.max(0) as usize).min(live.len());
                (live[..a].iter().map(|x| x.0).collect(), live[a..].iter().map(|x| x.0).collect())
            };
            let mut taken: Vec<i64> = running
                .iter()
                .filter_map(|k| get_st(&mut s, *k).get("slot").and_then(|v| v.as_i64()))
                .filter(|&x| x < total)
                .collect();
            for k in &running {
                let cur = get_st(&mut s, *k).get("slot").and_then(|v| v.as_i64());
                if cur.map(|c| c >= total).unwrap_or(true) {
                    let slot = (0..total).find(|i| !taken.contains(i));
                    get_st(&mut s, *k).insert("slot".into(), slot.map(Value::from).unwrap_or(Value::Null));
                    if let Some(x) = slot {
                        taken.push(x);
                    }
                }
            }
            let mut by_slot: Vec<(i64, Obj, Vec<(f64, i64)>)> = Vec::new();
            for k in &running {
                let st = get_st(&mut s, *k).clone();
                let rate: Vec<(f64, i64)> = match k {
                    None => s.rate.iter().cloned().collect(),
                    Some(k) => s.live_reqs.iter().find(|(x, _, _)| x == k).map(|(_, _, r)| r.iter().cloned().collect()).unwrap_or_default(),
                };
                let slot = st.get("slot").and_then(|v| v.as_i64()).unwrap_or(-1);
                by_slot.retain(|(b, _, _)| *b != slot);
                by_slot.push((slot, st, rate));
            }
            let mut queue = Vec::new();
            let mut wsts: Vec<Obj> = waiting.iter().map(|k| get_st(&mut s, *k).clone()).collect();
            let mut queued: Vec<Obj> = s.waiting.iter().map(|(_, w)| w.clone()).collect();
            queued.sort_by(|a, b| {
                fval(a.get("started")).unwrap_or(0.0).partial_cmp(&fval(b.get("started")).unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal)
            });
            wsts.extend(queued);
            for st in wsts {
                let started = fval(nn(st.get("started"))).unwrap_or(t);
                queue.push(json!({"api": st.get("api").cloned(), "model": st.get("model").cloned(),
                                  "prompt_tokens": st.get("prompt_tokens").cloned(),
                                  "waited_s": fv(py_round(t - started, 1))}));
            }
            let last = s.slot_last.clone();
            (by_slot, queue, last)
        };
        let reading: Vec<i64> =
            by_slot.iter().filter(|(_, st, _)| nn(st.get("first_token")).is_none()).map(|(b, _, _)| *b).collect();
        let progress = self.engine.progress();
        let console = self.engine.console();
        let last_start: std::collections::BTreeMap<i64, Value> = {
            let s = self.status_lock.lock().unwrap();
            let mut m = std::collections::BTreeMap::new();
            for r in &s.history {
                if let Some(sl) = nn(r.get("slot")).and_then(|v| v.as_i64()) {
                    m.insert(sl, r.get("time").cloned().unwrap_or(Value::Null));
                }
            }
            m
        };
        let drafts = self.engine.slot_drafts();
        let ctx = self.engine.max_context();
        let mut rows = Vec::new();
        for i in 0..total {
            let mut held: Obj = last.get(&i).cloned().unwrap_or_default();
            if let Some(bd) = drafts.get(&(i as usize)) {
                if bd.2 >= fval(held.get("at")).unwrap_or(0.0) {
                    held.insert("accepted".into(), bd.0.map(Value::from).unwrap_or(Value::Null));
                    held.insert("offered".into(), bd.1.map(Value::from).unwrap_or(Value::Null));
                }
            }
            let mine = by_slot.iter().find(|(b, _, _)| *b == i);
            let cons: Obj = match &console {
                Some(c) => {
                    let since = match mine {
                        Some((_, st, _)) => fval(st.get("started")),
                        None => last_start.get(&i).and_then(|v| v.as_f64()),
                    };
                    c.slot(i, since)
                }
                None => json!({"accept_pos": null, "accept_hist": null, "hit": null}).as_object().unwrap().clone(),
            };
            let mut row = json!({"id": i, "n_ctx": ctx, "is_processing": mine.is_some(), "state": "idle",
                "api": null, "model": null, "tok_s": 0.0, "generated": 0, "max_tokens": null, "prompt_tokens": 0,
                "prefill_done": 0, "reused": null, "ttft_s": null, "draft_len": null,
                "ctx_used": held.get("ctx").cloned().unwrap_or(Value::from(0)),
                "drafts_offered": held.get("offered").cloned(), "drafts_accepted": held.get("accepted").cloned(),
                "started": null, "path": null, "prompt_ms": null})
            .as_object()
            .unwrap()
            .clone();
            merge(&mut row, &cons);
            if let Some(v) = &view {
                row.insert("ctx_used".into(), v[i as usize].get("held").cloned().unwrap_or(Value::Null));
            }
            if let Some((_, st, rate)) = mine {
                let prompt = ival(st.get("prompt_tokens"));
                let gen = ival(st.get("generated"));
                let decoding = nn(st.get("first_token")).is_some();
                let done = if decoding {
                    Value::from(prompt)
                } else if let (Some(p), true) = (progress, reading == vec![i]) {
                    Value::from(p.0.min(prompt))
                } else {
                    Value::Null
                };
                let windows = ival(st.get("windows")) + ival(st.get("run_windows"));
                let wt = ival(st.get("window_tokens")) + ival(st.get("run_tokens"));
                let ttft = if decoding && truthy(st.get("started")) {
                    fv(py_round(fval(st.get("first_token")).unwrap() - fval(st.get("started")).unwrap(), 2))
                } else {
                    Value::Null
                };
                let has = |k: &str| console.is_none() || cons.contains_key(k);
                let dof = if has("drafts_offered") { row.get("drafts_offered").cloned().unwrap_or(Value::Null) } else { Value::Null };
                let dac = if has("drafts_accepted") { row.get("drafts_accepted").cloned().unwrap_or(Value::Null) } else { Value::Null };
                let upd = json!({"state": if decoding { "decoding" } else { "prefill" }, "api": st.get("api").cloned(),
                    "model": st.get("model").cloned(), "tok_s": fv(py_round(rate_of(st, rate), 1)), "generated": gen,
                    "max_tokens": st.get("max_tokens").cloned(), "prompt_tokens": prompt, "prefill_done": done,
                    "ctx_used": prompt + gen, "reused": st.get("reused").cloned(), "ttft_s": ttft,
                    "draft_len": if windows != 0 { fv(py_round(wt as f64 / windows as f64, 2)) } else { Value::Null },
                    "started": st.get("started").cloned(), "path": st.get("path").cloned(),
                    "prompt_ms": st.get("prompt_ms").cloned(), "drafts_offered": dof, "drafts_accepted": dac});
                merge(&mut row, upd.as_object().unwrap());
            }
            rows.push(row);
        }
        (rows, queue)
    }

    /// prompt_progress: llama.cpp's {total, cache, processed, time_ms} for the run on thread `tid`.
    pub fn prompt_progress(&self, n_prompt: Option<i64>, tid: u64) -> Option<Value> {
        let s = self.engine.prompt_progress_for(tid)?;
        if s.is_empty() {
            return None;
        }
        let total = match nn(s.get("total")) {
            Some(t) => ival(Some(t)),
            None => n_prompt?,
        };
        let cache = ival(s.get("cache")).min(total);
        let processed = match nn(s.get("read")) {
            Some(r) => ival(Some(r)).max(cache).min(total),
            None => cache,
        };
        Some(json!({"total": total, "cache": cache, "processed": processed, "time_ms": ival(s.get("ms"))}))
    }

    /// metrics: GET /metrics.
    pub fn metrics_v(&self) -> Value {
        let (s, decoding, hist, totals) = {
            let st = self.status_lock.lock().unwrap();
            let mut s = st.status.clone();
            if let Some((_, newest, _)) = st.live_reqs.last() {
                let q = s.get("queued").cloned().unwrap_or(Value::from(0));
                merge(&mut s, newest);
                s.insert("queued".into(), q);
            }
            let decoding = st.live_reqs.iter().any(|(_, x, _)| nn(x.get("first_token")).is_some());
            let hist: Vec<Obj> = st.history.iter().cloned().collect();
            let mut totals = json!({"since": self.started_at as i64, "requests": st.requests_done}).as_object().unwrap().clone();
            merge(&mut totals, &st.totals);
            (s, decoding, hist, totals)
        };
        let busy = truthy(s.get("busy"));
        let state = if self.is_loading() {
            "loading"
        } else if busy && nn(s.get("first_token")).is_none() && !decoding {
            "reading"
        } else if busy {
            "generating"
        } else if !self.loaded() {
            "unloaded"
        } else if self.engine.asleep() {
            "asleep"
        } else {
            "idle"
        };
        let slots = self.slot_stats_v();
        let live = json!({"state": state, "load": self.load_progress_v(), "queued": slots.get("queue_depth").cloned(),
            "slots": slots, "queue": self.slot_table_v().1,
            "prompt_tokens": if busy { s.get("prompt_tokens").cloned().unwrap_or(Value::Null) } else { Value::Null },
            "generated": if busy { s.get("generated").cloned().unwrap_or(Value::Null) } else { Value::Null },
            "elapsed_s": if busy && truthy(s.get("started")) { fv(py_round(now() - fval(s.get("started")).unwrap(), 1)) } else { Value::Null },
            "tok_s": if state == "generating" { fv(py_round(self.tok_s(), 1)) } else { Value::Null },
            "phase": if busy { s.get("phase").cloned().unwrap_or(Value::Null) } else { Value::Null },
            "prefill_tok_s_mean": if busy { self.engine.prefill_tok_s_mean().map(fv).unwrap_or(Value::Null) } else { Value::Null }});
        let mut info = self.engine.info().unwrap_or_default();
        let experts_total = info.get("spill_total").cloned();
        let spill = spill_of(&mut info);
        let (cards, l3) = serve_engine::console::engine_facts(&mut info);
        let ctx = self.engine.max_context();
        let mut engine = json!({"model": self.model,
                                "max_context": if ctx != 0 { ctx } else { self.engine.known_ctx() }})
            .as_object()
            .unwrap()
            .clone();
        merge(&mut engine, &info);
        engine.insert("spill".into(), spill.unwrap_or(Value::Null));
        engine.insert(
            "expert_total".into(),
            experts_total.filter(py::is_int).unwrap_or(Value::Null),
        );
        let tel = self.telemetry.read().unwrap().as_ref().map(|t| t.snapshot());
        let tel = tel.unwrap_or_else(|| {
            json!({"now": {"gpus": []}, "history": {}, "static": {"gpus": []}}).as_object().unwrap().clone()
        });
        let best = {
            let mut st = self.status_lock.lock().unwrap();
            self.drop_stale_best_locked(&mut st);
            st.best.clone()
        };
        let console = self.engine.console();
        let empty = Map::new();
        let tnow = tel.get("now").and_then(|v| v.as_object()).unwrap_or(&empty);
        let tstatic = tel.get("static").and_then(|v| v.as_object()).unwrap_or(&empty);
        let hw = serve_engine::console::hardware_view(tnow, tstatic, &info, &cards, &l3);
        let requests: Vec<Value> = hist.iter().rev().take(REQUESTS_SHOWN).cloned().map(Value::Object).collect();
        json!({"engine": engine, "live": live, "requests": requests, "best": best,
               "experts_map": console.and_then(|c| c.expert_map()).map(Value::Object).unwrap_or(Value::Null),
               "totals": totals, "conversation": self.engine.conversation(),
               "hardware": hw, "hardware_static": tel.get("static").cloned().unwrap_or(Value::Null),
               "history": tel.get("history").cloned().unwrap_or(Value::Null)})
    }

    pub(crate) fn drop_stale_best_locked(&self, s: &mut StatusAll) {
        let gen = self.engine.gen();
        if gen != s.best_gen {
            s.best_gen = gen;
            s.best = Obj::new();
        }
    }

    pub(crate) fn note_best_locked(&self, s: &mut StatusAll, r: &Obj) {
        self.drop_stale_best_locked(s);
        let out_tokens = ival(r.get("output_tokens"));
        let dts = fval(nn(r.get("decode_tok_s")));
        let cur_d = s.best.get("decode").and_then(|d| fval(d.get("tok_s"))).unwrap_or(0.0);
        if out_tokens >= BEST_MIN_OUT {
            if let Some(d) = dts.filter(|d| *d != 0.0 && *d > cur_d) {
                s.best.insert(
                    "decode".into(),
                    json!({"tok_s": fv(d), "id": r.get("id").cloned(), "time": r.get("time").cloned(),
                           "output_tokens": r.get("output_tokens").cloned()}),
                );
            }
        }
        let read = ival(r.get("prompt_tokens")) - ival(r.get("reused"));
        if let Some(pm) = fval(nn(r.get("prompt_ms"))).filter(|p| *p != 0.0) {
            if read >= BEST_MIN_READ {
                let rate = py_round(read as f64 / (pm / 1000.0), 1);
                let cur = s.best.get("prefill").and_then(|d| fval(d.get("tok_s"))).unwrap_or(0.0);
                if rate <= BEST_MAX_READ_TOK_S && rate > cur {
                    s.best.insert(
                        "prefill".into(),
                        json!({"tok_s": fv(rate), "id": r.get("id").cloned(), "time": r.get("time").cloned(),
                               "read_tokens": read}),
                    );
                }
            }
        }
    }

    /// v1_status: GET /v1/status.
    pub fn v1_status_v(&self) -> Value {
        let (s, last_t, last_at, done) = {
            let st = self.status_lock.lock().unwrap();
            (st.status.clone(), st.last_timings.clone(), st.last_request_at, st.requests_done)
        };
        let tel = self.telemetry.read().unwrap().as_ref().map(|t| t.snapshot()).unwrap_or_else(|| {
            json!({"now": {}, "static": {}}).as_object().unwrap().clone()
        });
        let empty = Map::new();
        let tnow = tel.get("now").and_then(|v| v.as_object()).unwrap_or(&empty);
        let tstatic = tel.get("static").and_then(|v| v.as_object()).unwrap_or(&empty);
        let gpus: Vec<Value> = tnow.get("gpus").and_then(|g| g.as_array()).cloned().unwrap_or_default();
        let gpus_static: Vec<Value> = tstatic.get("gpus").and_then(|g| g.as_array()).cloned().unwrap_or_default();
        let hw = gpus.first().cloned().unwrap_or(json!({}));
        let name = gpus_static.first().and_then(|g| g.get("name")).cloned().filter(|n| py::truthy(n));
        let scaled = |v: Option<&Value>, unit: f64, digits: usize| -> Value {
            match v.and_then(|x| x.as_f64()) {
                Some(f) if v.map(|x| x.is_number()).unwrap_or(false) => {
                    if digits == 0 {
                        Value::from(py_round(f / unit, 0) as i64).as_i64().map(|_| fv(py_round(f / unit, 0))).unwrap()
                    } else {
                        fv(py_round(f / unit, digits))
                    }
                }
                _ => Value::Null,
            }
        };
        let busy = truthy(s.get("busy"));
        let ctx = self.engine.max_context();
        let images = self.vision.is_some();
        let info = self.engine.info().unwrap_or_default();
        let mut vram = info.get("vram").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        vram.insert("elastic".into(), Value::Bool(truthy(info.get("vram_elastic"))));
        merge(&mut vram, self.vram_state().as_object().unwrap());
        let gpu = if let Some(n) = &name {
            json!({"name": n, "used_mib": round0(scaled(hw.get("mem_used"), (1u64 << 20) as f64, 0)),
                   "total_mib": round0(scaled(hw.get("mem_total"), (1u64 << 20) as f64, 0)), "util_pct": hw.get("util").cloned(),
                   "temp_c": hw.get("temp").cloned(), "power_w": hw.get("power").cloned()})
        } else {
            Value::Null
        };
        let ram = if truthy(tnow.get("ram_total")) {
            json!({"used_gib": scaled(tnow.get("ram_used"), (1u64 << 30) as f64, 1),
                   "total_gib": scaled(tnow.get("ram_total"), (1u64 << 30) as f64, 1)})
        } else {
            Value::Null
        };
        let serving = (self.engine.batch() as i64).max(1);
        json!({
            "service": "basalt", "model": self.model,
            "loaded": self.loaded(), "loading": self.is_loading(), "load": self.load_progress_v(), "asleep": self.engine.asleep(),
            "auto_load": self.engine.has_process(),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "grammar",
                                  "constrained_decoding": self.grammar_engine, "stream_buffered": false},
            "engine": info.get("version").cloned(),
            "started": self.started_at as i64, "uptime_s": (now() - self.started_at) as i64,
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx, "elastic_kv_ceiling": info.get("elastic_kv_ceiling").cloned()},
            "concurrency": {"serving": serving, "requested": serving},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": null},
            "activity": {"requests": done + busy as i64, "in_flight": busy as i64 + ival(s.get("queued")),
                         "last_request_at": last_at.filter(|t| *t != 0.0).map(|t| t as i64)},
            "last_timings": last_t,
            "vram": vram,
            "machine": {"at": now() as i64, "gpu": gpu, "ram": ram}})
    }

    /// GET /status: the status dict with the live rates, started and first_token dropped.
    pub fn status_view_v(&self) -> Value {
        let mut s = self.status_lock.lock().unwrap().status.clone();
        let t = now();
        if truthy(s.get("busy")) {
            let started = fval(s.get("started")).unwrap_or(t);
            s.insert("elapsed_s".into(), fv(py_round(t - started, 1)));
            if truthy(s.get("first_token")) {
                s.insert("tokens_per_s".into(), fv(py_round(self.tok_s(), 1)));
                s.insert("tokens_per_s_mean".into(), fv(py_round(self.tok_s_mean(), 1)));
            }
        }
        s.shift_remove("started");
        s.shift_remove("first_token");
        Value::Object(s)
    }
}

/// round(v / unit) with no digits is an int in Python.
fn round0(v: Value) -> Value {
    match v.as_f64() {
        Some(f) => Value::from(f as i64),
        None => v,
    }
}

/// _rate_of: a request's windowed tok/s.
pub fn rate_of(s: &Obj, rate: &[(f64, i64)]) -> f64 {
    if !truthy(s.get("busy")) || !truthy(s.get("first_token")) {
        return 0.0;
    }
    let t = now();
    let newest = rate.last();
    let oldest = rate.iter().find(|(ts, _)| t - ts <= RATE_WINDOW_S);
    if let (Some(n), Some(o)) = (newest, oldest) {
        if n.0 - o.0 >= RATE_MIN_SPAN_S {
            return ((n.1 - o.1) as f64 / (n.0 - o.0)).max(0.0);
        }
    }
    ival(s.get("generated")) as f64 / RATE_MIN_SPAN_S.max(t - fval(s.get("first_token")).unwrap())
}

/// GET /v1/stats: how many recent requests by default, and at most.
pub const STATS_REQUESTS_DEFAULT: usize = 50;
pub const STATS_REQUESTS_MAX: usize = 500;

/// Large pages as the engine's own memory lines say: {"state": "yes" | "no" | null, "line": the line it came from}.
/// The newest line that decides wins; no such line means the engine did not say (null), never a guess.
pub fn large_pages_of(facts: &[Obj]) -> Value {
    for f in facts.iter().rev() {
        let m = f.get("msg").and_then(|v| v.as_str()).unwrap_or("");
        let state = if m.contains("large pages refused")
            || m.contains("using 4 KB pages")
            || m.contains("MAP_HUGETLB unavailable")
            || m.contains("no large-page minimum")
        {
            Some("no")
        } else if m.contains("large pages (") || m.contains("hugetlb 2 MB pages") {
            Some("yes")
        } else {
            None
        };
        if let Some(s) = state {
            return json!({"state": s, "line": m});
        }
    }
    json!({"state": null, "line": null})
}

/// One finished request as /v1/stats shows it: the history record plus the rates it implies.
pub fn stats_request(r: &Obj) -> Value {
    let prompt = ival(r.get("prompt_tokens"));
    let reused = nn(r.get("reused")).map(|v| ival(Some(v)));
    let read = prompt - reused.unwrap_or(0);
    let prefill = match fval(nn(r.get("prompt_ms"))) {
        Some(ms) if ms > 0.0 && read > 0 => fv(py_round(read as f64 / (ms / 1000.0), 1)),
        _ => Value::Null,
    };
    let offered = ival(nn(r.get("drafts_offered")));
    let accept = if offered > 0 {
        fv(py_round(ival(nn(r.get("drafts_accepted"))) as f64 / offered as f64, 3))
    } else {
        Value::Null
    };
    let g = |k: &str| r.get(k).cloned().unwrap_or(Value::Null);
    json!({"id": g("id"), "time": g("time"), "path": g("path"), "slot": g("slot"), "duration_s": g("duration_s"),
           "end_reason": g("finish"), "prompt_tokens": prompt, "reused": g("reused"), "read_tokens": read,
           "output_tokens": g("output_tokens"), "prompt_ms": g("prompt_ms"), "prefill_tok_s": prefill,
           "decode_tok_s": g("decode_tok_s"), "ttft_s": g("ttft_s"), "drafts_offered": g("drafts_offered"),
           "drafts_accepted": g("drafts_accepted"), "draft_acceptance": accept, "draft_len": g("draft_len"),
           "expert_hit_rate": g("hit_rate"), "pcie_share": g("pcie_share"), "ssd_share": g("ssd_share")})
}

impl Service {
    /// stats: GET /v1/stats - the engine's state and its last `n` requests, newest first, from what the engine
    /// reports (its INFO/VRAM/CONV lines, the request records, the memory lines of its log). A figure the engine
    /// does not report is null. Rust server only.
    pub fn stats_v(&self, n: usize, facts: &[Obj]) -> Value {
        let m = self.metrics_v();
        let v1 = self.v1_status_v();
        let (rows, queue) = self.slot_table_v();
        let hist: Vec<Obj> = {
            let st = self.status_lock.lock().unwrap();
            st.history.iter().rev().take(n.min(STATS_REQUESTS_MAX)).cloned().collect()
        };
        let empty = Map::new();
        let engine = m.get("engine").and_then(|v| v.as_object()).unwrap_or(&empty);
        let live = m.get("live").and_then(|v| v.as_object()).unwrap_or(&empty);
        let e = |k: &str| engine.get(k).cloned().unwrap_or(Value::Null);
        let ctx = self.engine.max_context();
        let max_ctx = if ctx != 0 { ctx } else { self.engine.known_ctx() };
        let used: Vec<i64> = rows.iter().map(|r| ival(r.get("ctx_used"))).collect();
        let slots: Vec<Value> = rows
            .iter()
            .map(|r| {
                let g = |k: &str| r.get(k).cloned().unwrap_or(Value::Null);
                json!({"id": g("id"), "state": g("state"), "ctx_used": g("ctx_used"), "n_ctx": g("n_ctx"),
                       "tok_s": g("tok_s"), "generated": g("generated"), "prompt_tokens": g("prompt_tokens"),
                       "prefill_done": g("prefill_done"), "reused": g("reused"),
                       "drafts_offered": g("drafts_offered"), "drafts_accepted": g("drafts_accepted")})
            })
            .collect();
        let memory: Vec<Value> =
            facts.iter().map(|f| json!({"t": f.get("t").cloned(), "msg": f.get("msg").cloned()})).collect();
        let last_load = self.last_load.lock().unwrap().clone();
        json!({
            "service": "basalt", "model": self.model, "at": fv(py_round(now(), 3)),
            "state": live.get("state").cloned(), "loaded": self.loaded(), "loading": self.is_loading(),
            "asleep": self.engine.asleep(),
            "engine_version": e("version"), "quant": e("quant"), "gpus": e("gpus"), "kv": e("kv"),
            "uptime_s": v1.get("uptime_s").cloned(), "started": v1.get("started").cloned(),
            "context": {"max": max_ctx, "in_use": used.iter().sum::<i64>(), "largest_slot": used.iter().max().cloned()},
            "slots": {"summary": live.get("slots").cloned(), "rows": slots, "queue": queue},
            "vram": v1.get("vram").cloned(),
            "experts": {"slots": e("expert_slots"), "slots_primary": e("expert_slots_primary"),
                        "ram_mib": e("ram_experts_mib"), "total": e("expert_total"), "spill": e("spill")},
            "large_pages": large_pages_of(facts),
            "conversation": m.get("conversation").cloned(),
            "load": {"in_progress": live.get("load").cloned(), "last": last_load, "memory_lines": memory},
            "live": {"tok_s": live.get("tok_s").cloned(), "prompt_tokens": live.get("prompt_tokens").cloned(),
                     "generated": live.get("generated").cloned(), "elapsed_s": live.get("elapsed_s").cloned(),
                     "phase": live.get("phase").cloned(), "prefill_tok_s_mean": live.get("prefill_tok_s_mean").cloned()},
            "totals": m.get("totals").cloned(), "best": m.get("best").cloned(),
            "machine": v1.get("machine").cloned(),
            "requests": hist.iter().map(stats_request).collect::<Vec<_>>()})
    }
}

#[cfg(test)]
mod stats_tests {
    use super::*;

    fn obj(v: Value) -> Obj {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn large_pages_read_from_the_newest_deciding_line() {
        assert_eq!(large_pages_of(&[]), json!({"state": null, "line": null}));
        let yes = obj(json!({"msg": "FileExpertSource: ... ready; large pages (2097152 B)"}));
        let no = obj(json!({"msg": "large pages refused for 4194304 B (GetLargePageMinimum=2097152, VirtualAlloc error 1314); using 4 KB pages"}));
        let other = obj(json!({"msg": "the cards' share is read (20.00 GiB in 4.1 s); RAM next"}));
        assert_eq!(large_pages_of(&[yes.clone(), other.clone()])["state"], "yes");
        assert_eq!(large_pages_of(&[yes.clone(), no.clone()])["state"], "no");
        assert_eq!(large_pages_of(&[no, yes])["state"], "yes");
        assert_eq!(large_pages_of(&[other])["state"], Value::Null);
    }

    #[test]
    fn a_request_row_derives_only_what_its_record_supports() {
        let r = obj(json!({"id": "r_0001", "finish": "stop", "prompt_tokens": 10000, "reused": 8000, "output_tokens": 300,
                           "prompt_ms": 500.0, "decode_tok_s": 150.5, "hit_rate": 0.912, "pcie_share": 0.05,
                           "drafts_offered": 200, "drafts_accepted": 150}));
        let s = stats_request(&r);
        assert_eq!(s["read_tokens"], 2000);
        assert_eq!(s["prefill_tok_s"], json!(4000.0));
        assert_eq!(s["draft_acceptance"], json!(0.75));
        assert_eq!(s["end_reason"], "stop");
        assert_eq!(s["expert_hit_rate"], json!(0.912));
        // no clock, no drafts: null, not zero
        let bare = obj(json!({"prompt_tokens": 5, "reused": null, "prompt_ms": null}));
        let s = stats_request(&bare);
        assert_eq!(s["prefill_tok_s"], Value::Null);
        assert_eq!(s["draft_acceptance"], Value::Null);
        assert_eq!(s["read_tokens"], 5);
        assert_eq!(s["ssd_share"], Value::Null);
    }
}
