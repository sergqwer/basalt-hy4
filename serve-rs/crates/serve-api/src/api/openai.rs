//! api/openai.py: OpenAI chat completion chunks, and their non-streaming collection.

use crate::api::logprobs::{logprobs_part, REQUEST_KEY as LOGPROBS_KEY};
use crate::error::{AResult, ApiError};
use crate::py::{uuid_hex, Obj};
use crate::service::{RunItem, Service};
use serde_json::{json, Map, Value};
use serve_engine::gen::{Gen, Stop, Yielder};
use serve_engine::iface::{Flag, Prompt, Sampling};
use serve_engine::live::LiveRequest;
use serve_text::pyjson::{dumps, py_repr, truthy};
use std::collections::BTreeMap;
use std::sync::Arc;

/// checked_return_progress: llama.cpp's `return_progress`; Err (a 400 before streaming) for a non-boolean.
pub fn checked_return_progress(req: &Obj) -> AResult<bool> {
    let value = req.get("return_progress").filter(|v| !v.is_null());
    if let Some(v) = value {
        if !v.is_boolean() {
            return Err(ApiError::value(format!("return_progress: expected a boolean, not {}", py_repr(v))));
        }
    }
    Ok(value.map(truthy).unwrap_or(false) && req.get("stream").map(truthy).unwrap_or(false))
}

pub fn now_s() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

struct Lps {
    want: bool,
    content: Vec<Value>,
    reasoning: Vec<Value>,
    tools: BTreeMap<i64, Vec<Value>>,
}

/// openai_chunks: the stream's chunks; None is a heartbeat.
#[allow(clippy::too_many_arguments)]
pub fn openai_chunks(
    svc: &Arc<Service>,
    req: &Obj,
    bias: Vec<(u32, f64)>,
    ids: &Prompt,
    thinking: bool,
    tools: Option<Value>,
    max_new: i64,
    cancel: Flag,
    force: Option<String>,
    session: Option<String>,
    live: Option<Arc<LiveRequest>>,
) -> Gen<Option<Value>> {
    let cid = match &live {
        Some(l) => l.id.clone(),
        None => format!("chatcmpl-{}", &uuid_hex()[..24]),
    };
    let created = now_s();
    let model = svc.model.clone();
    let progress = checked_return_progress(req).unwrap_or(false);
    let mut run = svc.run_args(ids.clone(), thinking, tools, max_new, Sampling { map: req.clone(), logit_bias: if bias.is_empty() { None } else { Some(bias) } }, cancel);
    run.force = force;
    run.session = session;
    run.live = live;
    let thread = run.thread.clone();
    let svc = svc.clone();
    let ids = ids.clone();
    let want = req.get(LOGPROBS_KEY).map(|v| !v.is_null()).unwrap_or(false);
    Gen::new(move |y: &Yielder<Option<Value>>| {
        let mut lps = Lps { want, content: vec![], reasoning: vec![], tools: BTreeMap::new() };
        let chunk = |lps: &mut Lps, delta: Value, finish: Option<&str>| -> Map<String, Value> {
            let mut choice = Map::new();
            choice.insert("index".into(), json!(0));
            let has_content = delta.get("content").map(truthy).unwrap_or(false);
            choice.insert("delta".into(), delta);
            choice.insert("finish_reason".into(), finish.map(Value::from).unwrap_or(Value::Null));
            if lps.want && (!lps.content.is_empty() || !lps.reasoning.is_empty() || !lps.tools.is_empty() || has_content || finish.is_some())
            {
                choice.insert("logprobs".into(), logprobs_part(&lps.content, &lps.reasoning, &lps.tools));
                lps.content.clear();
                lps.reasoning.clear();
                lps.tools.clear();
            }
            let mut c = Map::new();
            c.insert("id".into(), Value::from(cid.clone()));
            c.insert("object".into(), json!("chat.completion.chunk"));
            c.insert("created".into(), json!(created));
            c.insert("model".into(), Value::from(model.clone()));
            c.insert("choices".into(), json!([choice]));
            c
        };
        let mut sent: Option<Value> = None;
        let mut progress_chunk = |lps: &mut Lps, final_: bool| -> Option<Value> {
            let mut p = svc.prompt_progress(Some(ids.len() as i64), thread.get())?;
            if final_ && p["processed"] == p["total"] && sent.as_ref() == Some(&p) {
                return None;
            }
            if final_ {
                let t = p["total"].clone();
                p["processed"] = t;
            }
            if sent.as_ref() == Some(&p) {
                return None;
            }
            sent = Some(p.clone());
            let mut c = chunk(lps, json!({"role": "assistant", "content": null}), None);
            c.insert("prompt_progress".into(), p);
            Some(Value::Object(c))
        };
        y.yield_(Some(Value::Object(chunk(&mut lps, json!({"role": "assistant", "content": ""}), None))))?;
        let mut reading = progress;
        let mut calls: i64 = 0;
        let mut streamed: Vec<(String, i64)> = vec![];
        let mut finished: Vec<String> = vec![];
        let mut gen = svc.run(run);
        let r = (|| -> Result<(), Stop> {
            while let Some(item) = gen.next() {
                let item = item?;
                if reading && !matches!(item, RunItem::Ping) {
                    reading = false;
                    if let Some(c) = progress_chunk(&mut lps, true) {
                        y.yield_(Some(c))?;
                    }
                }
                match item {
                    RunItem::Ping => {
                        if reading {
                            if let Some(c) = progress_chunk(&mut lps, false) {
                                y.yield_(Some(c))?;
                            }
                        }
                        y.yield_(None)?;
                    }
                    RunItem::Logprobs((channel, entry, index)) => match channel {
                        "content" => lps.content.push(entry),
                        "reasoning" => lps.reasoning.push(entry),
                        _ => lps.tools.entry(index.unwrap_or(0)).or_default().push(entry),
                    },
                    RunItem::Event(ev) => {
                        let idx = |id: &str, s: &Vec<(String, i64)>| s.iter().find(|(k, _)| k == id).map(|(_, i)| *i);
                        if ev.kind == "reasoning" && !ev.text.is_empty() {
                            y.yield_(Some(Value::Object(chunk(&mut lps, json!({"reasoning_content": ev.text}), None))))?;
                        } else if ev.kind == "content" && !ev.text.is_empty() {
                            y.yield_(Some(Value::Object(chunk(&mut lps, json!({"content": ev.text}), None))))?;
                        } else if ev.kind == "tool_start" {
                            let call = ev.call.as_ref().unwrap();
                            streamed.retain(|(k, _)| k != &call.id);
                            streamed.push((call.id.clone(), calls));
                            calls += 1;
                            let d = json!({"tool_calls": [{"index": idx(&call.id, &streamed), "id": call.id, "type": "function",
                                            "function": {"name": call.name, "arguments": ""}}]});
                            y.yield_(Some(Value::Object(chunk(&mut lps, d, None))))?;
                        } else if ev.kind == "tool_args" {
                            let call = ev.call.as_ref().unwrap();
                            let Some(i) = idx(&call.id, &streamed) else {
                                return Err(Stop::Raised(serve_engine::errors::EngineErr::Internal(format!(
                                    "KeyError: {}",
                                    serve_text::pyjson::py_str_repr(&call.id)
                                ))));
                            };
                            let d = json!({"tool_calls": [{"index": i, "function": {"arguments": ev.text}}]});
                            y.yield_(Some(Value::Object(chunk(&mut lps, d, None))))?;
                        } else if ev.kind == "tool_call" {
                            let call = ev.call.as_ref().unwrap();
                            if idx(&call.id, &streamed).is_some() {
                                if !finished.contains(&call.id) {
                                    finished.push(call.id.clone());
                                }
                            } else {
                                let d = json!({"tool_calls": [{"index": calls, "id": call.id, "type": "function",
                                    "function": {"name": call.name, "arguments": dumps(&Value::Object(call.arguments.clone()), false)}}]});
                                y.yield_(Some(Value::Object(chunk(&mut lps, d, None))))?;
                                calls += 1;
                            }
                        }
                    }
                    RunItem::Done(x) => {
                        if !lps.content.is_empty() || !lps.reasoning.is_empty() || !lps.tools.is_empty() {
                            y.yield_(Some(Value::Object(chunk(&mut lps, json!({}), None))))?;
                        }
                        let whole = calls != 0 && streamed.iter().all(|(k, _)| finished.contains(k));
                        let fin = x.get("finish").and_then(|f| f.as_str()).unwrap_or("").to_string();
                        let finish = if whole && fin == "stop" {
                            "tool_calls".to_string()
                        } else if fin == "cancel" {
                            "stop".to_string()
                        } else {
                            fin
                        };
                        let mut last = chunk(&mut lps, json!({}), Some(&finish));
                        let n = ids.len() as i64;
                        let ct = x.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                        let reused = x.get("reused").and_then(|v| v.as_i64()).unwrap_or(0);
                        last.insert(
                            "usage".into(),
                            json!({"prompt_tokens": n, "completion_tokens": ct, "total_tokens": n + ct,
                                   "prompt_tokens_details": {"cached_tokens": reused.min(n)}}),
                        );
                        if let Some(t) = x.get("timings").filter(|t| truthy(t)) {
                            last.insert("timings".into(), t.clone());
                        }
                        y.yield_(Some(Value::Object(last)))?;
                    }
                }
            }
            Ok(())
        })();
        crate::service::close_inner(gen, r)
    })
}

fn is_json(text: &str) -> bool {
    serve_text::pyjson::loads(text).is_ok()
}

/// openai_collect: the non-streaming answer from the chunks.
pub fn openai_collect(chunks: &mut Gen<Option<Value>>) -> AResult<Value> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut by_index: BTreeMap<i64, Value> = BTreeMap::new();
    let mut last: Option<Value> = None;
    let mut lps: Option<Vec<Value>> = None;
    let mut lps_reasoning: Vec<Value> = vec![];
    let mut lps_tools: BTreeMap<i64, Vec<Value>> = BTreeMap::new();
    for c in chunks.by_ref() {
        let Some(c) = c? else { continue };
        let ch = &c["choices"][0];
        if let Some(part) = ch.get("logprobs") {
            let l = lps.get_or_insert_with(Vec::new);
            l.extend(part["content"].as_array().cloned().unwrap_or_default());
            lps_reasoning.extend(part.get("reasoning_content").and_then(|v| v.as_array()).cloned().unwrap_or_default());
            for t in part.get("tool_calls").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
                lps_tools
                    .entry(t["index"].as_i64().unwrap_or(0))
                    .or_default()
                    .extend(t["content"].as_array().cloned().unwrap_or_default());
            }
        }
        let d = &ch["delta"];
        content.push_str(d.get("content").and_then(|v| v.as_str()).unwrap_or(""));
        reasoning.push_str(d.get("reasoning_content").and_then(|v| v.as_str()).unwrap_or(""));
        for tc in d.get("tool_calls").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
            let i = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(by_index.len() as i64);
            let cur = by_index
                .entry(i)
                .or_insert_with(|| json!({"id": null, "type": "function", "function": {"name": "", "arguments": ""}}));
            if let Some(id) = tc.get("id").filter(|v| truthy(v)) {
                cur["id"] = id.clone();
            }
            let empty = json!({});
            let f = tc.get("function").filter(|v| truthy(v)).unwrap_or(&empty);
            let name = format!("{}{}", cur["function"]["name"].as_str().unwrap_or(""), f.get("name").and_then(|v| v.as_str()).unwrap_or(""));
            let args = format!(
                "{}{}",
                cur["function"]["arguments"].as_str().unwrap_or(""),
                f.get("arguments").and_then(|v| v.as_str()).unwrap_or("")
            );
            cur["function"]["name"] = Value::from(name);
            cur["function"]["arguments"] = Value::from(args);
        }
        last = Some(c);
    }
    let last = last.ok_or_else(|| ApiError::internal("TypeError: 'NoneType' object is not subscriptable"))?;
    let finish = last["choices"][0]["finish_reason"].clone();
    let mut calls: Vec<Value> = by_index.into_values().collect();
    if finish != "tool_calls" {
        calls.retain(|c| is_json(c["function"]["arguments"].as_str().unwrap_or("")));
    }
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    msg.insert("content".into(), if content.is_empty() { Value::Null } else { Value::from(content) });
    if !reasoning.is_empty() {
        msg.insert("reasoning_content".into(), Value::from(reasoning));
    }
    if !calls.is_empty() {
        msg.insert("tool_calls".into(), Value::Array(calls));
    }
    let mut choice = Map::new();
    choice.insert("index".into(), json!(0));
    choice.insert("message".into(), Value::Object(msg));
    choice.insert("finish_reason".into(), finish);
    if let Some(l) = lps {
        choice.insert("logprobs".into(), logprobs_part(&l, &lps_reasoning, &lps_tools));
    }
    let mut out = Map::new();
    out.insert("id".into(), last["id"].clone());
    out.insert("object".into(), json!("chat.completion"));
    out.insert("created".into(), last["created"].clone());
    out.insert("model".into(), last["model"].clone());
    out.insert("choices".into(), json!([choice]));
    out.insert("usage".into(), last.get("usage").cloned().unwrap_or(Value::Null));
    if let Some(t) = last.get("timings").filter(|t| truthy(t)) {
        out.insert("timings".into(), t.clone());
    }
    Ok(Value::Object(out))
}
