//! api/anthropic.py: Anthropic messages events and their non-streaming collection.

use crate::api::logprobs::{logprobs_part, REQUEST_KEY as LOGPROBS_KEY};
use crate::error::{AResult, ApiError};
use crate::prompt::tokens_known;
use crate::py::{uuid_hex, Obj};
use crate::service::{RunItem, Service};
use serde_json::{json, Map, Value};
use serve_engine::gen::{Gen, Stop, Yielder};
use serve_engine::iface::{Flag, Prompt, Sampling};
use serve_engine::live::LiveRequest;
use serve_text::pyjson::dumps;
use std::collections::BTreeMap;
use std::sync::Arc;

pub type Item = Option<(String, Value)>;

/// anthropic_events: (event name, data) pairs; None is a heartbeat.
#[allow(clippy::too_many_arguments)]
pub fn anthropic_events(
    svc: &Arc<Service>,
    req: &Obj,
    ids: &Prompt,
    thinking: bool,
    tools: Option<Value>,
    max_new: i64,
    cancel: Flag,
    force: Option<String>,
    session: Option<String>,
    live: Option<Arc<LiveRequest>>,
) -> Gen<Item> {
    let mid = match &live {
        Some(l) => l.id.clone(),
        None => format!("msg_{}", &uuid_hex()[..24]),
    };
    let mut run = svc.run_args(ids.clone(), thinking, tools, max_new, Sampling { map: req.clone(), logit_bias: None }, cancel);
    run.force = force;
    run.session = session;
    run.live = live;
    let svc = svc.clone();
    let ids = ids.clone();
    let want_lp = req.get(LOGPROBS_KEY).map(|v| !v.is_null()).unwrap_or(false);
    Gen::new(move |y: &Yielder<Item>| {
        let start = || -> Item {
            Some((
                "message_start".into(),
                json!({"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": svc.model, "content": [],
                    "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": ids.len(), "output_tokens": 0}}}),
            ))
        };
        let mut started = tokens_known(&ids);
        if started {
            y.yield_(start())?;
        }
        let mut index: i64 = -1;
        let mut open_kind: Option<&'static str> = None;
        let mut used_tool = false;
        let mut lp_content: Vec<Value> = vec![];
        let mut lp_reasoning: Vec<Value> = vec![];
        let mut lp_tool: BTreeMap<i64, Vec<Value>> = BTreeMap::new();
        let mut calls: i64 = -1;
        let mut streamed: Vec<String> = vec![];
        let mut finished: Vec<String> = vec![];
        let mut gen = svc.run(run);
        let r = (|| -> Result<(), Stop> {
            macro_rules! delta {
                ($d:expr, $ch:expr) => {{
                    let mut e = Map::new();
                    e.insert("type".into(), json!("content_block_delta"));
                    e.insert("index".into(), json!(index));
                    e.insert("delta".into(), $d);
                    if want_lp {
                        let l = match $ch {
                            "tool" => lp_tool.remove(&calls).unwrap_or_default(),
                            "reasoning" => std::mem::take(&mut lp_reasoning),
                            _ => std::mem::take(&mut lp_content),
                        };
                        e.insert("logprobs".into(), Value::Array(l));
                    }
                    Some(("content_block_delta".to_string(), Value::Object(e)))
                }};
            }
            let close = |index: i64| -> Item {
                Some(("content_block_stop".into(), json!({"type": "content_block_stop", "index": index})))
            };
            while let Some(item) = gen.next() {
                let item = item?;
                if !started && (!matches!(item, RunItem::Ping) || tokens_known(&ids)) {
                    started = true;
                    y.yield_(start())?;
                }
                match item {
                    RunItem::Ping => {
                        y.yield_(None)?;
                    }
                    RunItem::Logprobs((channel, entry, call)) => match channel {
                        "tool" => lp_tool.entry(call.unwrap_or(0)).or_default().push(entry),
                        "reasoning" => lp_reasoning.push(entry),
                        _ => lp_content.push(entry),
                    },
                    RunItem::Event(ev) => {
                        if ev.kind == "tool_args" {
                            y.yield_(delta!(json!({"type": "input_json_delta", "partial_json": ev.text}), "tool"))?;
                            continue;
                        }
                        let cid = ev.call.as_ref().map(|c| c.id.clone()).unwrap_or_default();
                        if ev.kind == "tool_call" && streamed.contains(&cid) {
                            if !finished.contains(&cid) {
                                finished.push(cid);
                            }
                            continue;
                        }
                        let want: &'static str = match ev.kind {
                            "reasoning" => "thinking",
                            "content" => "text",
                            _ => "tool_use",
                        };
                        if ev.kind != "tool_call" && ev.kind != "tool_start" && ev.text.is_empty() {
                            continue;
                        }
                        if ev.kind == "tool_start" {
                            if !streamed.contains(&cid) {
                                streamed.push(cid.clone());
                            }
                            used_tool = true;
                        }
                        if open_kind != Some(want) || want == "tool_use" {
                            if open_kind.is_some() {
                                y.yield_(close(index))?;
                            }
                            index += 1;
                            open_kind = Some(want);
                            let block = match want {
                                "thinking" => json!({"type": "thinking", "thinking": "", "signature": ""}),
                                "text" => json!({"type": "text", "text": ""}),
                                _ => json!({"type": "tool_use", "id": ev.call.as_ref().map(|c| c.id.clone()).unwrap_or_default(),
                                            "name": ev.call.as_ref().map(|c| c.name.clone()).unwrap_or_default(), "input": {}}),
                            };
                            calls += (want == "tool_use") as i64;
                            y.yield_(Some((
                                "content_block_start".into(),
                                json!({"type": "content_block_start", "index": index, "content_block": block}),
                            )))?;
                        }
                        if want == "thinking" {
                            y.yield_(delta!(json!({"type": "thinking_delta", "thinking": ev.text}), "reasoning"))?;
                        } else if want == "text" {
                            y.yield_(delta!(json!({"type": "text_delta", "text": ev.text}), "content"))?;
                        } else if ev.kind == "tool_start" {
                        } else {
                            used_tool = true;
                            let args = dumps(&Value::Object(ev.call.as_ref().unwrap().arguments.clone()), false);
                            y.yield_(delta!(json!({"type": "input_json_delta", "partial_json": args}), "tool"))?;
                        }
                    }
                    RunItem::Done(x) => {
                        if open_kind.is_some() {
                            y.yield_(close(index))?;
                        }
                        let fin = x.get("finish").and_then(|f| f.as_str()).unwrap_or("");
                        let seq = x.get("stop_sequence").cloned().unwrap_or(Value::Null);
                        let stop = if !seq.is_null() {
                            "stop_sequence"
                        } else if used_tool && streamed.iter().all(|s| finished.contains(s)) && fin == "stop" {
                            "tool_use"
                        } else {
                            match fin {
                                "stop" | "cancel" => "end_turn",
                                "length" => "max_tokens",
                                other => {
                                    return Err(Stop::Raised(serve_engine::errors::EngineErr::Internal(format!(
                                        "KeyError: {}",
                                        serve_text::pyjson::py_str_repr(other)
                                    ))))
                                }
                            }
                        };
                        let n = ids.len() as i64;
                        let reused = x.get("reused").and_then(|v| v.as_i64()).unwrap_or(0).min(n);
                        let mut last = Map::new();
                        last.insert("type".into(), json!("message_delta"));
                        last.insert("delta".into(), json!({"stop_reason": stop, "stop_sequence": seq}));
                        last.insert(
                            "usage".into(),
                            json!({"input_tokens": n - reused, "cache_read_input_tokens": reused,
                                   "output_tokens": x.get("completion_tokens").cloned().unwrap_or(json!(0))}),
                        );
                        if want_lp {
                            last.insert("logprobs".into(), logprobs_part(&lp_content, &lp_reasoning, &lp_tool));
                        }
                        y.yield_(Some(("message_delta".into(), Value::Object(last))))?;
                        y.yield_(Some(("message_stop".into(), json!({"type": "message_stop"}))))?;
                    }
                }
            }
            Ok(())
        })();
        crate::service::close_inner(gen, r)
    })
}

/// anthropic_collect: the non-streaming message from the events.
pub fn anthropic_collect(events: &mut Gen<Item>) -> AResult<Value> {
    let mut msg: Option<Map<String, Value>> = None;
    let mut blocks: Vec<Map<String, Value>> = vec![];
    let mut lps: Option<Vec<Value>> = None;
    let mut lps_reasoning: Vec<Value> = vec![];
    let mut lps_tools: BTreeMap<i64, Vec<Value>> = BTreeMap::new();
    let mut tool_n: i64 = -1;
    let missing = || ApiError::internal("TypeError: 'NoneType' object is not subscriptable");
    for item in events.by_ref() {
        let Some((name, e)) = item? else { continue };
        if name == "content_block_start" && e["content_block"]["type"] == "tool_use" {
            tool_n += 1;
        }
        if let Some(l) = e.get("logprobs") {
            let acc = lps.get_or_insert_with(Vec::new);
            let arr = |v: Option<&Value>| v.and_then(|x| x.as_array()).cloned().unwrap_or_default();
            if name == "message_delta" {
                acc.extend(arr(l.get("content")));
                lps_reasoning.extend(arr(l.get("reasoning_content")));
                for t in arr(l.get("tool_calls")) {
                    lps_tools.entry(t["index"].as_i64().unwrap_or(0)).or_default().extend(arr(t.get("content")));
                }
            } else if e["delta"]["type"] == "text_delta" {
                acc.extend(arr(Some(l)));
            } else if e["delta"]["type"] == "thinking_delta" {
                lps_reasoning.extend(arr(Some(l)));
            } else {
                lps_tools.entry(tool_n).or_default().extend(arr(Some(l)));
            }
        }
        match name.as_str() {
            "message_start" => msg = e["message"].as_object().cloned(),
            "content_block_start" => blocks.push(e["content_block"].as_object().cloned().unwrap_or_default()),
            "content_block_delta" => {
                let d = &e["delta"];
                let b = blocks.last_mut().ok_or_else(|| ApiError::internal("IndexError: list index out of range"))?;
                let cat = |b: &mut Map<String, Value>, k: &str, add: &str| {
                    let s = format!("{}{}", b.get(k).and_then(|v| v.as_str()).unwrap_or(""), add);
                    b.insert(k.into(), Value::from(s));
                };
                if d["type"] == "text_delta" {
                    cat(b, "text", d["text"].as_str().unwrap_or(""));
                } else if d["type"] == "thinking_delta" {
                    cat(b, "thinking", d["thinking"].as_str().unwrap_or(""));
                } else {
                    cat(b, "_json", d["partial_json"].as_str().unwrap_or(""));
                }
            }
            "content_block_stop" if blocks.last().map(|b| b.contains_key("_json")).unwrap_or(false) => {
                let b = blocks.last_mut().unwrap();
                let j = b.shift_remove("_json").and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
                let j = if j.is_empty() { "{}".to_string() } else { j };
                match serve_text::pyjson::loads(&j) {
                    Ok(v) => {
                        b.insert("input".into(), v);
                    }
                    Err(_) => {
                        blocks.pop();
                    }
                }
            }
            "message_delta" => {
                let m = msg.as_mut().ok_or_else(missing)?;
                m.insert("stop_reason".into(), e["delta"]["stop_reason"].clone());
                m.insert("stop_sequence".into(), e["delta"].get("stop_sequence").cloned().unwrap_or(Value::Null));
                if let (Some(u), Some(add)) = (m.get_mut("usage").and_then(|u| u.as_object_mut()), e["usage"].as_object()) {
                    for (k, v) in add {
                        u.insert(k.clone(), v.clone());
                    }
                }
            }
            _ => {}
        }
    }
    let mut msg = msg.ok_or_else(missing)?;
    msg.insert("content".into(), Value::Array(blocks.into_iter().map(Value::Object).collect()));
    if let Some(l) = lps {
        msg.insert("logprobs".into(), logprobs_part(&l, &lps_reasoning, &lps_tools));
    }
    Ok(Value::Object(msg))
}
