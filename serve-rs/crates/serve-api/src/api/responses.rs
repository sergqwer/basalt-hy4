//! api/responses.py: OpenAI's Responses API (POST /v1/responses), stateless, on the chat path.

use crate::api::logprobs::{MAX_TOP as MAX_TOP_LOGPROBS, REQUEST_KEY as LOGPROBS_KEY};
use crate::error::{AResult, ApiError, ResponsesError};
use crate::frontend::{effort_kwargs, is_image_part, is_video_part, late_system_to_user, parts_of, stable_tool_order, Event};
use crate::py::{as_int, is_int, truthy, uuid_hex, Obj};
use crate::service::{RunItem, Service};
use base64::Engine as _;
use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use serve_engine::gen::{Gen, Stop, Yielder};
use serve_engine::iface::{Flag, Prompt, Sampling};
use serve_engine::live::LiveRequest;
use serve_text::pyjson::{dumps, py_repr, py_str};
use std::sync::{Arc, Mutex};

pub const ENCRYPTED_PREFIX: &str = "basalt.r1:";
pub const HOSTED_TOOLS: [&str; 10] = [
    "web_search", "web_search_preview", "file_search", "computer_use_preview", "computer_use", "code_interpreter",
    "image_generation", "local_shell", "mcp", "tool_search",
];

pub fn error_body(message: &str, kind: &str, param: Option<&str>, code: Option<&str>) -> Value {
    json!({"error": {"message": message, "type": kind, "param": param, "code": code}})
}

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid_hex())
}

fn rerr(msg: impl Into<String>, param: &str) -> ApiError {
    ApiError::Responses(ResponsesError::new(msg, Some(param), None))
}

fn rerr_code(msg: impl Into<String>, param: &str, code: &str) -> ApiError {
    ApiError::Responses(ResponsesError::new(msg, Some(param), Some(code)))
}

fn out(line: &str) {
    serve_engine::util::stdout::out(line);
}

// -------------------------------------------------------------------------------------------- request -> chat
fn content(c: Option<&Value>, param: &str) -> AResult<Value> {
    let c = match c {
        None | Some(Value::Null) => return Ok(Value::from("")),
        Some(Value::String(s)) => return Ok(Value::from(s.clone())),
        Some(Value::Array(a)) => a,
        Some(_) => return Err(rerr("expected a string or an array of content parts", param)),
    };
    let mut parts = vec![];
    for (j, part) in c.iter().enumerate() {
        if let Value::String(s) = part {
            parts.push(json!({"type": "text", "text": s}));
            continue;
        }
        let Some(p) = part.as_object() else {
            return Err(rerr("expected a content part object", &format!("{param}[{j}]")));
        };
        let kind = p.get("type").cloned().unwrap_or(Value::Null);
        let k = kind.as_str().unwrap_or("\0");
        let text_of = |key: &str| p.get(key).filter(|v| truthy(v)).cloned().unwrap_or(Value::from(""));
        if ["input_text", "output_text", "text", "summary_text", "reasoning_text"].contains(&k) {
            parts.push(json!({"type": "text", "text": text_of("text")}));
        } else if k == "refusal" {
            parts.push(json!({"type": "text", "text": text_of("refusal")}));
        } else if k == "input_file" {
            let has = |key: &str| p.get(key).map(truthy).unwrap_or(false);
            if !(has("file_data") || has("file_url")) {
                let field = if has("file_id") { "file_id" } else { "file_data" };
                return Err(rerr_code(
                    "only files given as file_data (a data: URL with the file's bytes) or file_url are read; this server keeps no files",
                    &format!("{param}[{j}].{field}"),
                    "unsupported_parameter",
                ));
            }
            parts.push(part.clone());
        } else if is_image_part(&kind) {
            if !p.get("image_url").map(truthy).unwrap_or(false) {
                out("[server] error: an image part has no image_url");
                return Err(rerr_code(
                    "only images given as image_url (a data: or http(s) URL) are supported; this server keeps no files",
                    &format!("{param}[{j}]"),
                    "unsupported_parameter",
                ));
            }
            parts.push(json!({"type": "input_image", "image_url": p["image_url"]}));
        } else if is_video_part(&kind) {
            let ok = p.get("video_url").and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false);
            if !ok {
                return Err(rerr_code(
                    "only videos given as video_url (a data: or http(s) URL) are supported; this server keeps no files",
                    &format!("{param}[{j}]"),
                    "unsupported_parameter",
                ));
            }
            let mut m = Map::new();
            for key in ["fps", "max_frames"] {
                if let Some(v) = p.get(key) {
                    m.insert(key.into(), v.clone());
                }
            }
            m.insert("type".into(), json!("input_video"));
            m.insert("video_url".into(), p["video_url"].clone());
            parts.push(Value::Object(m));
        } else {
            return Err(rerr_code(
                format!(
                    "content parts of type {} are not supported (text, images, videos and files are)",
                    py_repr(&kind)
                ),
                &format!("{param}[{j}].type"),
                "unsupported_parameter",
            ));
        }
    }
    parts_of(&Value::Array(parts))
}

fn reasoning_text(item: &Map<String, Value>, param: &str) -> AResult<String> {
    let texts: Vec<String> = item
        .get("content")
        .filter(|v| truthy(v))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|p| p.as_object())
                .filter(|p| matches!(p.get("type").and_then(|t| t.as_str()), Some("reasoning_text") | Some("text")))
                .map(|p| p.get("text").filter(|v| truthy(v)).map(py_str).unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    if !texts.is_empty() {
        return Ok(texts.concat());
    }
    if let Some(enc) = item.get("encrypted_content").and_then(|v| v.as_str()) {
        if let Some(rest) = enc.strip_prefix(ENCRYPTED_PREFIX) {
            let damaged =
                || rerr_code("this reasoning item's encrypted_content is damaged", &format!("{param}.encrypted_content"), "invalid_encrypted_content");
            let raw = crate::image_source::b64decode_validate(rest).ok_or_else(damaged)?;
            let s = String::from_utf8(raw).map_err(|_| damaged())?;
            let v = serve_text::pyjson::loads(&s).map_err(|_| damaged())?;
            return match v.get("text") {
                Some(Value::String(t)) => Ok(t.clone()),
                // json.loads(...)["text"] of a non-dict or a missing key: TypeError / KeyError -> damaged
                Some(other) if v.is_object() => Ok(py_str(other)),
                _ => Err(damaged()),
            };
        }
    }
    Ok(String::new())
}

fn arguments(raw: Option<&Value>, param: &str) -> AResult<Value> {
    match raw {
        Some(Value::Object(_)) => Ok(raw.unwrap().clone()),
        None | Some(Value::Null) => Ok(json!({})),
        Some(Value::String(s)) if crate::py::strip(s).is_empty() => Ok(json!({})),
        Some(Value::String(s)) => match serve_text::pyjson::loads(s) {
            Err(_) => Ok(json!({"arguments": s})),
            Ok(v @ Value::Object(_)) => Ok(v),
            Ok(v) => Ok(json!({"arguments": v})),
        },
        Some(_) => Err(rerr("function_call arguments must be a JSON string", param)),
    }
}

fn tool_output(item: &Map<String, Value>, param: &str) -> AResult<Value> {
    let o = item.get("output");
    if let Some(Value::Object(m)) = o {
        let t = m.get("type").cloned().unwrap_or(Value::Null);
        if is_image_part(&t) || t == "input_file" {
            return content(Some(&json!([o.unwrap()])), &format!("{param}.output"));
        }
        return Ok(Value::from(dumps(o.unwrap(), false)));
    }
    match o {
        Some(Value::String(s)) => Ok(Value::from(s.clone())),
        other => content(other, &format!("{param}.output")),
    }
}

/// input_messages: `instructions` + `input` -> the chat template's messages.
pub fn input_messages(req: &Obj) -> AResult<Vec<Value>> {
    let mut messages: Vec<Map<String, Value>> = vec![];
    if let Some(ins) = req.get("instructions").filter(|v| !v.is_null() && *v != "") {
        let c = if ins.is_string() { ins.clone() } else { content(Some(ins), "instructions")? };
        messages.push(json!({"role": "system", "content": c}).as_object().unwrap().clone());
    }
    let items = match req.get("input") {
        None | Some(Value::Null) => {
            return Err(rerr_code(
                "input is required: a string or an array of input items",
                "input",
                "missing_required_parameter",
            ))
        }
        Some(Value::String(s)) => vec![json!({"type": "message", "role": "user", "content": s})],
        Some(Value::Array(a)) => a.clone(),
        Some(_) => return Err(rerr("input must be a string or an array of input items", "input")),
    };
    let mut thinking: Vec<String> = vec![];
    let mut open_turn: Option<usize> = None;
    let turn = |messages: &mut Vec<Map<String, Value>>, thinking: &mut Vec<String>, c: Value| -> usize {
        let mut t = Map::new();
        t.insert("role".into(), json!("assistant"));
        t.insert("content".into(), c);
        if !thinking.is_empty() {
            t.insert("reasoning_content".into(), Value::from(thinking.concat()));
            thinking.clear();
        }
        messages.push(t);
        messages.len() - 1
    };
    for (i, item) in items.iter().enumerate() {
        let param = format!("input[{i}]");
        let Some(item) = item.as_object() else { return Err(rerr("expected an input item object", &param)) };
        let kind = match item.get("type").filter(|v| truthy(v)) {
            Some(k) => k.clone(),
            None if item.contains_key("role") => json!("message"),
            None => Value::Null,
        };
        let k = kind.as_str().unwrap_or("\0");
        if k == "message" {
            let role = item.get("role").cloned().unwrap_or(Value::Null);
            let r = role.as_str().unwrap_or("\0");
            if !["user", "assistant", "system", "developer"].contains(&r) {
                return Err(rerr(format!("unknown message role {}", py_repr(&role)), &format!("{param}.role")));
            }
            let c = content(item.get("content"), &format!("{param}.content"))?;
            if r == "assistant" {
                open_turn = Some(turn(&mut messages, &mut thinking, c));
            } else {
                open_turn = None;
                thinking.clear();
                let role = if r == "developer" { "system" } else { r };
                messages.push(json!({"role": role, "content": c}).as_object().unwrap().clone());
            }
        } else if k == "reasoning" {
            let text = reasoning_text(item, &param)?;
            if !text.is_empty() {
                if let Some(o) = open_turn {
                    let m = &messages[o];
                    if m.get("tool_calls").map(truthy).unwrap_or(false) || m.get("content").map(truthy).unwrap_or(false) {
                        open_turn = None;
                    }
                }
                thinking.push(text);
            }
        } else if k == "function_call" || k == "custom_tool_call" {
            let name = item.get("name");
            let Some(name) = name.and_then(|n| n.as_str()).filter(|n| !n.is_empty()) else {
                return Err(rerr("a function_call needs a name", &format!("{param}.name")));
            };
            let name = match item.get("namespace").filter(|v| truthy(v)) {
                Some(ns) => format!("{}.{name}", py_str(ns)),
                None => name.to_string(),
            };
            let args = if k == "function_call" {
                arguments(item.get("arguments"), &format!("{param}.arguments"))?
            } else {
                json!({"input": item.get("input").filter(|v| v.is_string()).cloned().unwrap_or(json!(""))})
            };
            let target = match open_turn {
                Some(o) if thinking.is_empty() => o,
                _ => {
                    let t = turn(&mut messages, &mut thinking, json!(""));
                    open_turn = Some(t);
                    t
                }
            };
            let tc = messages[target].entry("tool_calls").or_insert_with(|| json!([]));
            tc.as_array_mut().unwrap().push(json!({"function": {"name": name, "arguments": args},
                                                   "_call_id": item.get("call_id").cloned().unwrap_or(Value::Null)}));
        } else if k == "function_call_output" || k == "custom_tool_call_output" {
            let c = tool_output(item, &param)?;
            messages.push(
                json!({"role": "tool", "content": c, "_call_id": item.get("call_id").cloned().unwrap_or(Value::Null)})
                    .as_object()
                    .unwrap()
                    .clone(),
            );
            open_turn = None;
            thinking.clear();
        } else if k == "item_reference" {
            return Err(rerr_code(
                "item references need stored responses, and this server keeps none: send the items themselves",
                &param,
                "unsupported_parameter",
            ));
        } else {
            return Err(rerr_code(
                format!("input items of type {} are not supported", py_repr(&kind)),
                &format!("{param}.type"),
                "unsupported_parameter",
            ));
        }
    }
    order_tool_results(&mut messages);
    let mut lead = 0;
    while lead < messages.len() && messages[lead]["role"] == "system" && messages[lead]["content"].is_string() {
        lead += 1;
    }
    if lead > 1 {
        let joined = messages[..lead].iter().map(|m| m["content"].as_str().unwrap().to_string()).collect::<Vec<_>>().join("\n\n");
        messages.splice(..lead, [json!({"role": "system", "content": joined}).as_object().unwrap().clone()]);
    }
    Ok(late_system_to_user(messages.into_iter().map(Value::Object).collect()))
}

fn order_tool_results(messages: &mut [Map<String, Value>]) {
    let mut i = 0;
    while i < messages.len() {
        if messages[i]["role"] != "tool" {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < messages.len() && messages[j]["role"] == "tool" {
            j += 1;
        }
        let order: Vec<Value> = if i > 0 {
            messages[i - 1]
                .get("tool_calls")
                .filter(|v| truthy(v))
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|c| c.get("_call_id").cloned().unwrap_or(Value::Null)).collect())
                .unwrap_or_default()
        } else {
            vec![]
        };
        // rank: the last position of each truthy id (a dict comprehension keeps the last)
        let rank = |cid: &Value| -> usize {
            let mut r = None;
            for (k, c) in order.iter().enumerate() {
                if truthy(c) && serve_text::pyjson::py_eq(c, cid) {
                    r = Some(k);
                }
            }
            r.unwrap_or(order.len())
        };
        let mut run: Vec<Map<String, Value>> = messages[i..j].to_vec();
        run.sort_by_key(|m| rank(m.get("_call_id").unwrap_or(&Value::Null)));
        for (k, m) in run.into_iter().enumerate() {
            messages[i + k] = m;
        }
        i = j;
    }
    for m in messages.iter_mut() {
        m.shift_remove("_call_id");
        if let Some(Value::Array(calls)) = m.get_mut("tool_calls") {
            for c in calls {
                if let Some(o) = c.as_object_mut() {
                    o.shift_remove("_call_id");
                }
            }
        }
    }
}

/// {template name: (namespace, name, kind)}
pub type Names = IndexMap<String, (Option<Value>, String, String)>;

/// request_tools: (template tools or None, names, hosted tools left out).
pub fn request_tools(req: &Obj) -> AResult<(Option<Value>, Names, Vec<String>)> {
    let mut tools: Vec<Value> = vec![];
    let mut names = Names::new();
    let mut skipped = vec![];
    let given = match req.get("tools").filter(|v| truthy(v)) {
        None => vec![],
        Some(Value::Array(a)) => a.clone(),
        Some(_) => return Err(rerr("tools must be an array", "tools")),
    };
    let mut add = |tool: &Map<String, Value>, param: &str, namespace: Option<&Value>, ns_desc: &str| -> AResult<()> {
        let kind = tool.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let Some(name) = tool.get("name").and_then(|v| v.as_str()).filter(|n| !n.is_empty()) else {
            return Err(rerr("a tool needs a name", &format!("{param}.name")));
        };
        let ns = namespace.filter(|v| truthy(v));
        let flat = match ns {
            Some(n) => format!("{}.{name}", py_str(n)),
            None => name.to_string(),
        };
        let mut description = tool.get("description").filter(|v| truthy(v)).map(py_str).unwrap_or_default();
        if !ns_desc.is_empty() {
            description = if description.is_empty() { ns_desc.to_string() } else { format!("{ns_desc}\n\n{description}") };
        }
        let params = if kind == "function" {
            let p = tool.get("parameters").filter(|v| !v.is_null());
            if let Some(p) = p {
                if !p.is_object() {
                    return Err(rerr(
                        format!("tool \"{name}\": \"parameters\" must be an object (its JSON schema)"),
                        &format!("{param}.parameters"),
                    ));
                }
            }
            match p.filter(|v| truthy(v)) {
                Some(p) => p.clone(),
                None => json!({"type": "object", "properties": {}}),
            }
        } else {
            let fmt = tool.get("format").filter(|v| truthy(v)).cloned().unwrap_or(json!({}));
            if let Some(f) = fmt.as_object() {
                if f.get("type").and_then(|v| v.as_str()) == Some("grammar") && f.get("definition").map(truthy).unwrap_or(false) {
                    let syntax = f.get("syntax").map(py_str).unwrap_or_default();
                    description += &format!("\n\nThe input must follow this {syntax} grammar:\n{}", py_str(&f["definition"]));
                }
            }
            json!({"type": "object", "properties": {"input": {"type": "string", "description": "the raw input"}},
                   "required": ["input"]})
        };
        tools.push(json!({"name": flat, "description": description, "parameters": params}));
        names.insert(flat, (namespace.cloned().filter(|v| !v.is_null()), name.to_string(), kind));
        Ok(())
    };
    for (i, tool) in given.iter().enumerate() {
        let param = format!("tools[{i}]");
        let Some(t) = tool.as_object() else { return Err(rerr("expected a tool object", &param)) };
        let kind = t.get("type");
        match kind.and_then(|k| k.as_str()) {
            Some("function") | Some("custom") => add(t, &param, None, "")?,
            Some("namespace") => {
                let members = t.get("tools").filter(|v| truthy(v)).and_then(|v| v.as_array()).cloned().unwrap_or_default();
                for (j, m) in members.iter().enumerate() {
                    let ok = m
                        .as_object()
                        .map(|o| matches!(o.get("type").map(|x| x.as_str()), None | Some(Some("function")) | Some(Some("custom"))))
                        .unwrap_or(false);
                    if !ok {
                        return Err(rerr("a namespace holds function tools", &format!("{param}.tools[{j}]")));
                    }
                    let mut member = Map::new();
                    member.insert("type".into(), json!("function"));
                    for (k, v) in m.as_object().unwrap() {
                        member.insert(k.clone(), v.clone());
                    }
                    let nsd = t.get("description").filter(|v| truthy(v)).map(py_str).unwrap_or_default();
                    add(&member, &format!("{param}.tools[{j}]"), t.get("name"), &nsd)?;
                }
            }
            Some(k) => skipped.push(k.to_string()),
            None => return Err(rerr("a tool needs a type", &format!("{param}.type"))),
        }
    }
    if req.get("tool_choice").map(|c| c == "none").unwrap_or(false) {
        return Ok((None, names, skipped));
    }
    let t = if tools.is_empty() {
        None
    } else {
        Some(Value::Array(stable_tool_order(tools, |t| t["name"].as_str().unwrap_or("").to_string())))
    };
    Ok((t, names, skipped))
}

/// text_format: `text.format` -> the chat path's response_format (None: plain text).
pub fn text_format(req: &Obj) -> AResult<Option<Value>> {
    let Some(text) = req.get("text").filter(|v| !v.is_null()) else { return Ok(None) };
    let Some(t) = text.as_object() else { return Err(rerr("text must be an object", "text")) };
    let Some(fmt) = t.get("format").filter(|v| !v.is_null()) else { return Ok(None) };
    let Some(f) = fmt.as_object() else { return Err(rerr("text.format must be an object", "text.format")) };
    match f.get("type") {
        None | Some(Value::Null) => Ok(None),
        Some(k) if k == "text" => Ok(None),
        Some(k) if k == "json_object" => Ok(Some(json!({"type": "json_object"}))),
        Some(k) if k == "json_schema" => {
            let mut spec = Map::new();
            for key in ["name", "schema", "strict", "description"] {
                if let Some(v) = f.get(key) {
                    spec.insert(key.into(), v.clone());
                }
            }
            Ok(Some(json!({"type": "json_schema", "json_schema": spec})))
        }
        Some(_) => Err(rerr("text.format.type must be text, json_object or json_schema", "text.format.type")),
    }
}

/// check_request: what this stateless server cannot do, refused before anything runs.
pub fn check_request(req: &Obj) -> AResult<()> {
    let set = |k: &str| req.get(k).map(truthy).unwrap_or(false);
    if set("previous_response_id") {
        return Err(rerr_code(
            "this server keeps no responses (stateless): send the whole conversation in input instead of previous_response_id",
            "previous_response_id",
            "unsupported_parameter",
        ));
    }
    if set("conversation") {
        return Err(rerr_code(
            "this server keeps no conversations (stateless): send the whole conversation in input",
            "conversation",
            "unsupported_parameter",
        ));
    }
    if set("background") {
        return Err(rerr_code(
            "background responses are not supported: use stream or wait for the answer",
            "background",
            "unsupported_parameter",
        ));
    }
    if set("prompt") {
        return Err(rerr_code(
            "stored prompt templates are not supported: send instructions and input",
            "prompt",
            "unsupported_parameter",
        ));
    }
    if let Some(n) = req.get("max_output_tokens").filter(|v| !v.is_null()) {
        if !is_int(n) || as_int(n).map(|x| x < 1).unwrap_or(true) {
            return Err(rerr("max_output_tokens must be a positive whole number", "max_output_tokens"));
        }
    }
    if let Some(r) = req.get("reasoning").filter(|v| !v.is_null()) {
        if !r.is_object() {
            return Err(rerr("reasoning must be an object", "reasoning"));
        }
    }
    Ok(())
}

/// requested_logprobs: top_logprobs when `include` names message.output_text.logprobs, else None.
pub fn requested_logprobs(req: &Obj) -> AResult<Option<i64>> {
    let top = req.get("top_logprobs").filter(|v| !v.is_null());
    if let Some(t) = top {
        let ok = is_int(t) && as_int(t).map(|n| (0..=MAX_TOP_LOGPROBS).contains(&n)).unwrap_or(false);
        if !ok {
            return Err(rerr(format!("top_logprobs must be a whole number from 0 to {MAX_TOP_LOGPROBS}"), "top_logprobs"));
        }
    }
    let include = req.get("include").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    if include.iter().any(|v| v == "message.output_text.logprobs") {
        Ok(Some(top.and_then(as_int).unwrap_or(0)))
    } else {
        Ok(None)
    }
}

pub fn template_kwargs(req: &Obj) -> AResult<Obj> {
    let empty = json!({});
    let r = req.get("reasoning").filter(|v| truthy(v)).unwrap_or(&empty);
    let effort = r.get("effort").cloned().unwrap_or(Value::Null);
    effort_kwargs(&effort).map_err(|e| rerr(e.message(), "reasoning.effort"))
}

// -------------------------------------------------------------------------------------------- output
/// Builds one response's output items and typed events from Service.run's events.
pub struct Assembler {
    names: Names,
    encrypted: bool,
    prompt_tokens: i64,
    seq: i64,
    /// the open output item: its index in response.output
    item: Option<usize>,
    lps: Vec<Value>,
    want_lp: bool,
    lps_reasoning: Vec<Value>,
    lps_tools: std::collections::BTreeMap<i64, Vec<Value>>,
    /// the call items' output indices: tool call i is calls[i]
    calls: Vec<usize>,
    pub response: Map<String, Value>,
}

fn now_s() -> i64 {
    crate::api::openai::now_s()
}

impl Assembler {
    pub fn new(req: &Obj, model: &str, prompt_tokens: usize, names: Names, encrypted: bool, rid: &str) -> Assembler {
        let reasoning = req.get("reasoning").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let g = |k: &str| req.get(k).cloned().unwrap_or(Value::Null);
        let rid = if rid.is_empty() { new_id("resp") } else { rid.to_string() };
        let response = json!({
            "id": rid, "object": "response", "created_at": now_s(), "status": "in_progress",
            "background": false, "error": null, "incomplete_details": null,
            "instructions": g("instructions"), "max_output_tokens": g("max_output_tokens"),
            "model": model, "output": [], "parallel_tool_calls": req.get("parallel_tool_calls").cloned().unwrap_or(json!(true)),
            "previous_response_id": null,
            "reasoning": {"effort": reasoning.get("effort").cloned().unwrap_or(Value::Null),
                          "summary": reasoning.get("summary").cloned().unwrap_or(Value::Null)},
            "store": false, "temperature": g("temperature"),
            "text": if req.get("text").map(|t| t.is_object()).unwrap_or(false) { g("text") } else { json!({"format": {"type": "text"}}) },
            "tool_choice": req.get("tool_choice").cloned().unwrap_or(json!("auto")),
            "tools": req.get("tools").filter(|v| truthy(v)).cloned().unwrap_or(json!([])),
            "top_p": g("top_p"), "top_logprobs": g("top_logprobs"), "truncation": "disabled", "usage": null, "user": null,
            "metadata": req.get("metadata").filter(|v| truthy(v)).cloned().unwrap_or(json!({}))});
        Assembler {
            names,
            encrypted,
            prompt_tokens: prompt_tokens as i64,
            seq: 0,
            item: None,
            lps: vec![],
            want_lp: req.get(LOGPROBS_KEY).map(|v| !v.is_null()).unwrap_or(false),
            lps_reasoning: vec![],
            lps_tools: Default::default(),
            calls: vec![],
            response: response.as_object().unwrap().clone(),
        }
    }

    fn event(&mut self, kind: &str, data: Vec<(&str, Value)>) -> Value {
        let mut e = Map::new();
        e.insert("type".into(), Value::from(kind));
        e.insert("sequence_number".into(), Value::from(self.seq));
        for (k, v) in data {
            e.insert(k.into(), v);
        }
        self.seq += 1;
        Value::Object(e)
    }

    fn snapshot(&self) -> Value {
        Value::Object(self.response.clone())
    }

    fn output(&mut self) -> &mut Vec<Value> {
        self.response.get_mut("output").unwrap().as_array_mut().unwrap()
    }

    fn at(&mut self, i: usize) -> &mut Map<String, Value> {
        self.output()[i].as_object_mut().unwrap()
    }

    pub fn start(&mut self) -> Vec<Value> {
        let s = self.snapshot();
        let a = self.event("response.created", vec![("response", s)]);
        let s = self.snapshot();
        let b = self.event("response.in_progress", vec![("response", s)]);
        vec![a, b]
    }

    pub fn in_progress(&mut self) -> Value {
        let s = self.snapshot();
        self.event("response.in_progress", vec![("response", s)])
    }

    fn index(&self) -> i64 {
        self.response["output"].as_array().unwrap().len() as i64 - 1
    }

    fn open(&mut self, mut item: Map<String, Value>) -> Vec<Value> {
        let ty = item["type"].as_str().unwrap_or("").to_string();
        if self.want_lp && ["reasoning", "function_call", "custom_tool_call"].contains(&ty.as_str()) {
            item.insert("logprobs".into(), json!([]));
        }
        self.output().push(Value::Object(item.clone()));
        let at = self.output().len() - 1;
        self.item = Some(at);
        if self.want_lp && (ty == "function_call" || ty == "custom_tool_call") {
            self.calls.push(at);
        }
        let idx = self.index();
        let mut out = vec![self.event("response.output_item.added", vec![("output_index", json!(idx)), ("item", Value::Object(item.clone()))])];
        if ty == "message" {
            let part = json!({"type": "output_text", "text": "", "annotations": [], "logprobs": []});
            self.at(at)["content"].as_array_mut().unwrap().push(part.clone());
            let id = item["id"].clone();
            out.push(self.event(
                "response.content_part.added",
                vec![("item_id", id), ("output_index", json!(idx)), ("content_index", json!(0)), ("part", part)],
            ));
        }
        out
    }

    pub fn close(&mut self, status: &str) -> Vec<Value> {
        let Some(at) = self.item.take() else { return vec![] };
        let mut out = vec![];
        self.at(at).insert("status".into(), Value::from(status));
        if self.want_lp {
            if let Some(k) = self.calls.iter().position(|c| *c == at) {
                let got = self.lps_tools.remove(&(k as i64)).unwrap_or_default();
                self.at(at)["logprobs"].as_array_mut().unwrap().extend(got);
            }
        }
        let item = self.at(at).clone();
        let id = item.get("id").cloned().unwrap_or(Value::Null);
        let a = json!(at);
        match item["type"].as_str().unwrap_or("") {
            "message" => {
                let part = item["content"][0].clone();
                out.push(self.event(
                    "response.output_text.done",
                    vec![("item_id", id.clone()), ("output_index", a.clone()), ("content_index", json!(0)),
                         ("text", part["text"].clone()), ("logprobs", part["logprobs"].clone())],
                ));
                out.push(self.event(
                    "response.content_part.done",
                    vec![("item_id", id), ("output_index", a.clone()), ("content_index", json!(0)), ("part", part)],
                ));
            }
            "reasoning" => {
                let text = item["content"][0]["text"].clone();
                out.push(self.event(
                    "response.reasoning_text.done",
                    vec![("item_id", id), ("output_index", a.clone()), ("content_index", json!(0)), ("text", text.clone())],
                ));
                if self.encrypted {
                    let raw = dumps(&json!({"text": text}), false);
                    let enc = format!("{ENCRYPTED_PREFIX}{}", base64::engine::general_purpose::STANDARD.encode(raw.as_bytes()));
                    self.at(at).insert("encrypted_content".into(), Value::from(enc));
                }
            }
            "function_call" => {
                out.push(self.event(
                    "response.function_call_arguments.done",
                    vec![("item_id", id), ("output_index", a.clone()), ("name", item["name"].clone()), ("arguments", item["arguments"].clone())],
                ));
            }
            "custom_tool_call" => {
                out.push(self.event(
                    "response.custom_tool_call_input.done",
                    vec![("item_id", id), ("output_index", a.clone()), ("input", item["input"].clone())],
                ));
            }
            _ => {}
        }
        let item = Value::Object(self.at(at).clone());
        out.push(self.event("response.output_item.done", vec![("output_index", a), ("item", item)]));
        out
    }

    /// logprob: a generated token's logprobs (Service.run's "logprobs" item).
    pub fn logprob(&mut self, channel: &str, entry: Value, index: Option<i64>) {
        match channel {
            "content" => self.lps.push(entry),
            "reasoning" => self.lps_reasoning.push(entry),
            _ => self.lps_tools.entry(index.unwrap_or(0)).or_default().push(entry),
        }
    }

    fn item_type(&self) -> Option<String> {
        self.item.map(|i| self.response["output"][i]["type"].as_str().unwrap_or("").to_string())
    }

    pub fn feed(&mut self, ev: &Event) -> Vec<Value> {
        let mut out = vec![];
        match ev.kind {
            "reasoning" => {
                if ev.text.is_empty() {
                    return out;
                }
                if self.item_type().as_deref() != Some("reasoning") {
                    out.extend(self.close("completed"));
                    let it = json!({"id": new_id("rs"), "type": "reasoning", "status": "in_progress", "summary": [],
                                    "content": [{"type": "reasoning_text", "text": ""}]});
                    out.extend(self.open(it.as_object().unwrap().clone()));
                }
                let at = self.item.unwrap();
                let t = format!("{}{}", self.at(at)["content"][0]["text"].as_str().unwrap_or(""), ev.text);
                self.at(at)["content"][0]["text"] = Value::from(t);
                let mut data = vec![];
                let lp = if self.want_lp {
                    let l = std::mem::take(&mut self.lps_reasoning);
                    self.at(at)["logprobs"].as_array_mut().unwrap().extend(l.clone());
                    Some(l)
                } else {
                    None
                };
                let id = self.at(at)["id"].clone();
                data.push(("item_id", id));
                data.push(("output_index", json!(self.index())));
                data.push(("content_index", json!(0)));
                data.push(("delta", Value::from(ev.text.clone())));
                if let Some(l) = lp {
                    data.push(("logprobs", Value::Array(l)));
                }
                out.push(self.event("response.reasoning_text.delta", data));
            }
            "content" => {
                if ev.text.is_empty() {
                    return out;
                }
                if self.item_type().as_deref() != Some("message") {
                    out.extend(self.close("completed"));
                    let it = json!({"id": new_id("msg"), "type": "message", "status": "in_progress", "role": "assistant", "content": []});
                    out.extend(self.open(it.as_object().unwrap().clone()));
                }
                let at = self.item.unwrap();
                let t = format!("{}{}", self.at(at)["content"][0]["text"].as_str().unwrap_or(""), ev.text);
                self.at(at)["content"][0]["text"] = Value::from(t);
                let l = std::mem::take(&mut self.lps);
                self.at(at)["content"][0]["logprobs"].as_array_mut().unwrap().extend(l.clone());
                let id = self.at(at)["id"].clone();
                let idx = self.index();
                out.push(self.event(
                    "response.output_text.delta",
                    vec![("item_id", id), ("output_index", json!(idx)), ("content_index", json!(0)),
                         ("delta", Value::from(ev.text.clone())), ("logprobs", Value::Array(l))],
                ));
            }
            "tool_start" => {
                out.extend(self.close("completed"));
                let it = self.call_item(ev);
                out.extend(self.open(it));
            }
            "tool_args" => {
                let cid = ev.call.as_ref().map(|c| c.id.clone()).unwrap_or_default();
                let Some(at) = self.item else { return out };
                if self.at(at).get("call_id").and_then(|v| v.as_str()) != Some(cid.as_str()) {
                    return out;
                }
                if self.at(at)["type"] == "function_call" {
                    let a = format!("{}{}", self.at(at)["arguments"].as_str().unwrap_or(""), ev.text);
                    self.at(at).insert("arguments".into(), Value::from(a));
                    let mut extra = None;
                    if self.want_lp {
                        if let Some(k) = self.calls.iter().position(|c| *c == at) {
                            let got = self.lps_tools.remove(&(k as i64)).unwrap_or_default();
                            self.at(at)["logprobs"].as_array_mut().unwrap().extend(got.clone());
                            extra = Some(got);
                        }
                    }
                    let id = self.at(at)["id"].clone();
                    let mut data = vec![("item_id", id), ("output_index", json!(self.index())), ("delta", Value::from(ev.text.clone()))];
                    if let Some(g) = extra {
                        data.push(("logprobs", Value::Array(g)));
                    }
                    out.push(self.event("response.function_call_arguments.delta", data));
                }
            }
            "tool_call" => {
                let call = ev.call.as_ref().unwrap();
                let same = self.item.map(|at| self.response["output"][at].get("call_id").and_then(|v| v.as_str()) == Some(call.id.as_str())).unwrap_or(false);
                if !same {
                    out.extend(self.close("completed"));
                    let it = self.call_item(ev);
                    out.extend(self.open(it));
                    let at = self.item.unwrap();
                    if self.at(at)["type"] == "function_call" {
                        let a = dumps(&Value::Object(call.arguments.clone()), false);
                        self.at(at).insert("arguments".into(), Value::from(a.clone()));
                        let id = self.at(at)["id"].clone();
                        out.push(self.event(
                            "response.function_call_arguments.delta",
                            vec![("item_id", id), ("output_index", json!(self.index())), ("delta", Value::from(a))],
                        ));
                    }
                }
                let at = self.item.unwrap();
                if self.at(at)["type"] == "custom_tool_call" {
                    let input = match call.arguments.get("input") {
                        Some(Value::String(s)) => s.clone(),
                        _ => dumps(&Value::Object(call.arguments.clone()), false),
                    };
                    self.at(at).insert("input".into(), Value::from(input.clone()));
                    let id = self.at(at)["id"].clone();
                    out.push(self.event(
                        "response.custom_tool_call_input.delta",
                        vec![("item_id", id), ("output_index", json!(self.index())), ("delta", Value::from(input))],
                    ));
                }
                out.extend(self.close("completed"));
            }
            _ => {}
        }
        out
    }

    fn call_item(&self, ev: &Event) -> Map<String, Value> {
        let call = ev.call.as_ref().unwrap();
        let (namespace, name, kind) =
            self.names.get(&call.name).cloned().unwrap_or((None, call.name.clone(), "function".into()));
        let mut item = if kind == "custom" {
            json!({"id": new_id("ctc"), "type": "custom_tool_call", "status": "in_progress", "call_id": call.id,
                   "name": name, "input": ""})
        } else {
            json!({"id": new_id("fc"), "type": "function_call", "status": "in_progress", "call_id": call.id,
                   "name": name, "arguments": ""})
        };
        if let Some(ns) = namespace.filter(|v| truthy(v)) {
            item.as_object_mut().unwrap().insert("namespace".into(), ns);
        }
        item.as_object().unwrap().clone()
    }

    /// finish: close open items, then response.completed (or .incomplete on length).
    pub fn finish(&mut self, done: &Obj) -> Vec<Value> {
        let length = done.get("finish").and_then(|f| f.as_str()) == Some("length");
        let unfinished = matches!(self.item_type().as_deref(), Some("function_call") | Some("custom_tool_call"));
        let mut out = self.close(if length || unfinished { "incomplete" } else { "completed" });
        let tools = std::mem::take(&mut self.lps_tools);
        for (k, entries) in tools {
            if k >= 0 && (k as usize) < self.calls.len() {
                let at = self.calls[k as usize];
                self.at(at)["logprobs"].as_array_mut().unwrap().extend(entries);
            }
        }
        let n = done.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
        let prompt = done.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(self.prompt_tokens);
        let reused = done.get("reused").and_then(|v| v.as_i64()).unwrap_or(0);
        self.response.insert(
            "usage".into(),
            json!({"input_tokens": prompt, "input_tokens_details": {"cached_tokens": reused.min(prompt)},
                   "output_tokens": n, "output_tokens_details": {"reasoning_tokens": done.get("reasoning_tokens").cloned().unwrap_or(json!(0))},
                   "total_tokens": prompt + n}),
        );
        if length {
            self.response.insert("status".into(), json!("incomplete"));
            self.response.insert("incomplete_details".into(), json!({"reason": "max_output_tokens"}));
            let s = self.snapshot();
            out.push(self.event("response.incomplete", vec![("response", s)]));
            return out;
        }
        self.response.insert("status".into(), json!("completed"));
        self.response.insert("completed_at".into(), json!(now_s()));
        let s = self.snapshot();
        out.push(self.event("response.completed", vec![("response", s)]));
        out
    }

    /// failed: response.failed, an error after the stream started.
    pub fn failed(&mut self, message: &str, code: &str) -> Value {
        if let Some(at) = self.item.take() {
            self.at(at).insert("status".into(), json!("incomplete"));
        }
        self.response.insert("status".into(), json!("failed"));
        self.response.insert("error".into(), json!({"code": code, "message": message}));
        let s = self.snapshot();
        self.event("response.failed", vec![("response", s)])
    }
}

/// response_events: handler.py's inner events() generator.
#[allow(clippy::too_many_arguments)]
pub fn response_events(
    svc: &Arc<Service>,
    asm: &Arc<Mutex<Assembler>>,
    ids: &Prompt,
    thinking: bool,
    tools: Option<Value>,
    max_new: i64,
    req: &Obj,
    cancel: Flag,
    force: Option<String>,
    session: Option<String>,
    live: Option<Arc<LiveRequest>>,
) -> Gen<Option<Value>> {
    let mut run = svc.run_args(ids.clone(), thinking, tools, max_new, Sampling { map: req.clone(), logit_bias: None }, cancel.clone());
    run.force = force;
    run.session = session;
    run.live = live;
    let svc = svc.clone();
    let asm = asm.clone();
    Gen::new(move |y: &Yielder<Option<Value>>| {
        let start = asm.lock().unwrap().start();
        for e in start {
            y.yield_(Some(e))?;
        }
        let mut done: Option<Obj> = None;
        let mut gen = svc.run(run);
        let r = (|| -> Result<(), Stop> {
            while let Some(item) = gen.next() {
                match item? {
                    RunItem::Ping => y.yield_(None)?,
                    RunItem::Event(ev) => {
                        let evs = asm.lock().unwrap().feed(&ev);
                        for e in evs {
                            y.yield_(Some(e))?;
                        }
                    }
                    RunItem::Logprobs((ch, entry, idx)) => asm.lock().unwrap().logprob(ch, entry, idx),
                    RunItem::Done(d) => done = Some(d),
                }
            }
            Ok(())
        })();
        // `with closing(...)`: the run is closed and its cleanup error raised
        let c = gen.close();
        match (r, c) {
            (Err(e), _) => return Err(e),
            (Ok(()), Err(e)) => return Err(Stop::Raised(e)),
            _ => {}
        }
        if let Some(d) = done {
            if d.get("finish").and_then(|f| f.as_str()) != Some("cancel") && !cancel.is_set() {
                let evs = asm.lock().unwrap().finish(&d);
                for e in evs {
                    y.yield_(Some(e))?;
                }
            }
        }
        Ok(())
    })
}

/// collect: the non-streaming answer, the final response object.
pub fn collect(events: &mut Gen<Option<Value>>) -> AResult<Value> {
    let mut last: Option<Value> = None;
    for e in events.by_ref() {
        if let Some(e) = e? {
            last = Some(e);
        }
    }
    let last = last.ok_or_else(|| ApiError::internal("TypeError: 'NoneType' object is not subscriptable"))?;
    Ok(last.get("response").cloned().unwrap_or(Value::Null))
}
