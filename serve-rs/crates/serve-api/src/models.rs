//! serve/models: per-model chat dialects (Qwen, the mock engine's; Hy4-flash).

use crate::error::{AResult, ApiError};
use crate::frontend::{self, hold_back, text_kinds, decode_param, tool_properties, Event, OutputParse, OutputParser, Tags,
                      ToolCall, CALL_START};
use crate::py::{getv, py_str, Obj};
use crate::template::ChatTemplate;
use serde_json::{json, Map, Value};
use serve_text::pyjson;

pub const OFFICIAL_SUFFIX: &str = ":opensource";
const MARK0: u32 = 0xF2100;

/// The Hy4 template's special tokens for one suffix.
#[derive(Clone, Debug)]
pub struct Hy4Tags {
    pub suffix: String,
    pub start: String,
    pub middle: String,
    pub end: String,
    pub image_start: String,
    pub image_pad: String,
    pub image_end: String,
    pub video_start: String,
    pub video_pad: String,
    pub video_end: String,
    pub reasoning_mode: String,
    pub think: String,
    pub think_end: String,
    pub calls: String,
    pub calls_end: String,
    pub call: String,
    pub call_end: String,
    pub key: String,
    pub key_end: String,
    pub value: String,
    pub value_end: String,
    pub response: String,
    pub response_end: String,
}

impl Hy4Tags {
    pub fn new(suffix: &str) -> Self {
        let f = |n: &str| format!("<｜hy_{n}{suffix}｜>");
        let img = |n: &str| format!("<｜hy_image▁{n}{suffix}｜>");
        let vid = |n: &str| format!("<｜hy_video▁{n}{suffix}｜>");
        let t = |n: &str| format!("<{n}{suffix}>");
        Hy4Tags {
            suffix: suffix.to_string(),
            start: f("start"),
            middle: f("middle"),
            end: f("end"),
            image_start: img("start"),
            image_pad: img("pad"),
            image_end: img("end"),
            video_start: vid("start"),
            video_pad: vid("pad"),
            video_end: vid("end"),
            reasoning_mode: format!("<｜reasoning_mode{suffix}｜>"),
            think: t("think"),
            think_end: t("/think"),
            calls: t("tool_calls"),
            calls_end: t("/tool_calls"),
            call: t("tool_call"),
            call_end: t("/tool_call"),
            key: t("arg_key"),
            key_end: t("/arg_key"),
            value: t("arg_value"),
            value_end: t("/arg_value"),
            response: t("tool_response"),
            response_end: t("/tool_response"),
        }
    }

    /// Tags.all(): every token, in attribute order.
    pub fn all(&self) -> Vec<String> {
        vec![
            self.start.clone(),
            self.middle.clone(),
            self.end.clone(),
            self.image_start.clone(),
            self.image_pad.clone(),
            self.image_end.clone(),
            self.video_start.clone(),
            self.video_pad.clone(),
            self.video_end.clone(),
            self.reasoning_mode.clone(),
            self.think.clone(),
            self.think_end.clone(),
            self.calls.clone(),
            self.calls_end.clone(),
            self.call.clone(),
            self.call_end.clone(),
            self.key.clone(),
            self.key_end.clone(),
            self.value.clone(),
            self.value_end.clone(),
            self.response.clone(),
            self.response_end.clone(),
        ]
    }
}

/// suffix_of: the token suffix a template names (its HYTK variable).
pub fn suffix_of(source: &str) -> AResult<String> {
    // Python: (['"])(.*?) - the earliest match of either quote kind
    let sq = regex::Regex::new(r#"\{%-?\s*set\s+HYTK\s*=\s*'(.*?)'"#).unwrap();
    let dq = regex::Regex::new(r#"\{%-?\s*set\s+HYTK\s*=\s*"(.*?)""#).unwrap();
    let best = [sq.captures(source), dq.captures(source)]
        .into_iter()
        .flatten()
        .min_by_key(|c| c.get(0).unwrap().start());
    match best {
        Some(c) => Ok(c.get(1).unwrap().as_str().to_string()),
        None => Err(ApiError::value("this chat template has no HYTK token suffix: not a Hy4-flash template")),
    }
}

#[derive(Clone, Debug)]
pub enum Family {
    Qwen,
    Hy4(Hy4Tags),
}

#[derive(Clone, Debug)]
pub struct Dialect {
    pub name: &'static str,
    pub stop_texts: Vec<String>,
    pub reasoning_close: String,
    pub forced_call_ends_reply: bool,
    pub nothink_lead: String,
    pub call_newlines: usize,
    pub image_pad: String,
    pub image_start: String,
    pub video_pad: String,
    pub video_marker: String,
    pub in_engine_vision: bool,
    pub effort_turn: String,
    /// llama.cpp's /props chat_template_caps
    pub caps: Value,
    pub family: Family,
}

fn all_caps() -> Value {
    json!({"supports_tool_calls": true, "supports_parallel_tool_calls": true, "supports_reasoning_effort": true,
           "supports_system_role": true, "supports_preserve_reasoning": true})
}

/// dialect_for: the dialect of a model file's architecture; None is the mock engine's Qwen form.
pub fn dialect_for(arch: Option<&str>) -> AResult<Dialect> {
    match arch {
        Some("hy4flash") | Some("hy_v4_flash") => Ok(Dialect::hy4(OFFICIAL_SUFFIX)),
        None => Ok(Dialect::qwen()),
        Some(a) => Err(ApiError::value(format!("no dialect for architecture {}", pyjson::py_str_repr(a)))),
    }
}

impl Dialect {
    pub fn qwen() -> Self {
        Dialect {
            name: "qwen",
            stop_texts: vec![serve_text::text::IM_END.into(), "<|endoftext|>".into()],
            reasoning_close: "\n</think>\n\n".into(),
            forced_call_ends_reply: false,
            nothink_lead: String::new(),
            call_newlines: 2,
            image_pad: "<|image_pad|>".into(),
            image_start: "<|vision_start|>".into(),
            video_pad: "<|video_pad|>".into(),
            video_marker: "<|vision_start|><|video_pad|><|vision_end|>".into(),
            in_engine_vision: false,
            effort_turn: "<|im_start|>system\n{}<|im_end|>\n".into(),
            caps: all_caps(),
            family: Family::Qwen,
        }
    }

    pub fn hy4(suffix: &str) -> Self {
        let tags = Hy4Tags::new(suffix);
        let mut d = Dialect {
            name: "hy4flash",
            stop_texts: vec![],
            reasoning_close: String::new(),
            forced_call_ends_reply: false,
            nothink_lead: String::new(),
            call_newlines: 0,
            image_pad: String::new(),
            image_start: String::new(),
            video_pad: String::new(),
            video_marker: String::new(),
            in_engine_vision: true,
            effort_turn: String::new(),
            caps: all_caps(),
            family: Family::Hy4(tags),
        };
        d.tokens();
        d
    }

    fn tokens(&mut self) {
        if let Family::Hy4(t) = &self.family {
            self.stop_texts = vec![t.end.clone()];
            self.reasoning_close = t.think_end.clone();
            self.image_pad = t.image_pad.clone();
            self.image_start = t.image_start.clone();
        }
    }

    /// the template trims every message: no turn ever ends after whitespace
    pub fn trims_messages(&self) -> bool {
        matches!(self.family, Family::Qwen)
    }

    /// the special token a turn of the chat format starts with
    pub fn turn_start(&self) -> String {
        match &self.family {
            Family::Qwen => "<|im_start|>".into(),
            Family::Hy4(t) => t.start.clone(),
        }
    }

    pub fn tags(&self) -> Option<&Hy4Tags> {
        match &self.family {
            Family::Hy4(t) => Some(t),
            Family::Qwen => None,
        }
    }

    /// template_kwargs: the frontend's kwargs as this model's template names them.
    pub fn template_kwargs(&self, kwargs: &Obj) -> Obj {
        match &self.family {
            Family::Qwen => {
                if !kwargs.contains_key("preserve_reasoning") {
                    return kwargs.clone();
                }
                let mut out = kwargs.clone();
                let v = out.shift_remove("preserve_reasoning").unwrap();
                out.insert("preserve_thinking".into(), v);
                out
            }
            Family::Hy4(_) => {
                let mut out = kwargs.clone();
                let et = out.shift_remove("enable_thinking");
                if et == Some(Value::Bool(false)) {
                    out.insert("reasoning_effort".into(), Value::from("no_think"));
                }
                if let Some(v) = out.shift_remove("preserve_reasoning") {
                    out.insert("preserved_thinking".into(), v);
                }
                out
            }
        }
    }

    /// effort_level: the frontend's level as this model's template takes it.
    pub fn effort_level(&self, level: &Value) -> AResult<Value> {
        if crate::py::is_int(level) {
            return Err(ApiError::value(format!(
                "unknown reasoning effort {}: use none, low, medium, high or max",
                pyjson::py_repr(level)
            )));
        }
        let is = |s: &str| level == s;
        Ok(match self.family {
            Family::Qwen => {
                if is("high") || is("max") {
                    Value::from("xhigh")
                } else {
                    level.clone()
                }
            }
            Family::Hy4(_) => Value::from(if is("max") { "max" } else { "high" }),
        })
    }

    /// template: the prompt renderer (a Hy4 template sets the token suffix).
    pub fn template(&mut self, source: &str) -> AResult<ChatTemplate> {
        if let Family::Hy4(_) = self.family {
            let suffix = suffix_of(source)?;
            self.family = Family::Hy4(Hy4Tags::new(&suffix));
            self.tokens();
        }
        ChatTemplate::new(source).map_err(ApiError::internal)
    }

    pub fn parser(&self, thinking: bool, tools: Option<&Value>, stream_tools: bool) -> Box<dyn OutputParse> {
        match &self.family {
            Family::Qwen => Box::new(OutputParser::new(thinking, tools, stream_tools)),
            Family::Hy4(t) => Box::new(Hy4Parser::new(t.clone(), thinking, tools, stream_tools)),
        }
    }

    pub fn literal_tags(&self, controls: &[String]) -> Tags {
        match &self.family {
            Family::Qwen => frontend::literal_tags(controls),
            Family::Hy4(t) => {
                let own: Tags = t
                    .all()
                    .into_iter()
                    .enumerate()
                    .map(|(k, tok)| (tok, char::from_u32(MARK0 + k as u32).unwrap().to_string()))
                    .collect();
                let mut tags = frontend::literal_tags(controls);
                frontend::merge_tags(&mut tags, own);
                frontend::sort_tags(tags)
            }
        }
    }

    pub fn video_text(&self, times: &[f64], rows: usize) -> String {
        match self.family {
            Family::Qwen => times
                .iter()
                .map(|t| {
                    format!("<{} seconds><|vision_start|>{}<|vision_end|>", format!("{:.1}", t), self.video_pad.repeat(rows))
                })
                .collect(),
            Family::Hy4(_) => String::new(),
        }
    }

    pub fn forced_opening(&self, name: Option<&str>) -> String {
        match &self.family {
            Family::Qwen => match name {
                Some(n) => format!("{CALL_START}\n<function={n}>\n"),
                None => format!("{CALL_START}\n<function="),
            },
            Family::Hy4(t) => format!("{}{}{}", t.calls, t.call, name.unwrap_or("")),
        }
    }
}

// ------------------------------------------------------------------------------------------------ Hy4 parser
pub struct Hy4Parser {
    t: Hy4Tags,
    pub state: &'static str,
    pub buf: String,
    stream_tools: bool,
    props: Map<String, Value>,
    sub: &'static str,
    call: Option<ToolCall>,
    args: Map<String, Value>,
    first: bool,
    key: String,
    schema: Option<Value>,
    text_value: bool,
    raw: String,
    calls_seen: bool,
}

impl Hy4Parser {
    pub fn new(t: Hy4Tags, thinking: bool, tools: Option<&Value>, stream_tools: bool) -> Self {
        let mut props = Map::new();
        if let Some(Value::Array(ts)) = tools {
            for tool in ts {
                props.insert(py_str(getv(tool, "name")), Value::Object(tool_properties(Some(tool))));
            }
        }
        Hy4Parser {
            t,
            state: if thinking { "reasoning" } else { "content" },
            buf: String::new(),
            stream_tools,
            props,
            sub: "between",
            call: None,
            args: Map::new(),
            first: true,
            key: String::new(),
            schema: None,
            text_value: false,
            raw: String::new(),
            calls_seen: false,
        }
    }

    fn push_args(&self, out: &mut Vec<Event>, piece: String) {
        if !piece.is_empty() && self.stream_tools {
            out.push(Event::args(piece, self.call.as_ref().unwrap()));
        }
    }

    fn end_call(&mut self, out: &mut Vec<Event>) {
        self.push_args(out, "}".into());
        let c = self.call.take().unwrap();
        out.push(Event::call("tool_call", &ToolCall { name: c.name, arguments: self.args.clone(), id: c.id }));
    }

    fn reasoning(&mut self, out: &mut Vec<Event>) -> bool {
        let (te, tc) = (self.t.think_end.clone(), self.t.calls.clone());
        let end = self.buf.find(&te);
        let calls = self.buf.find(&tc);
        if let Some(e) = end {
            if calls.map(|c| e < c).unwrap_or(true) {
                if e > 0 {
                    out.push(Event::text("reasoning", &self.buf[..e]));
                }
                self.buf = self.buf[e + te.len()..].to_string();
                self.state = "content";
                return true;
            }
        }
        if let Some(c) = calls {
            if c > 0 {
                out.push(Event::text("reasoning", &self.buf[..c]));
            }
            self.buf = self.buf[c + tc.len()..].to_string();
            self.state = "call";
            return true;
        }
        let keep = hold_back(&self.buf, &[&te, &tc]);
        if self.buf.len() > keep {
            let cut = self.buf.len() - keep;
            out.push(Event::text("reasoning", &self.buf[..cut]));
            self.buf = self.buf[cut..].to_string();
        }
        false
    }

    fn content(&mut self, out: &mut Vec<Event>) -> bool {
        let tc = self.t.calls.clone();
        if let Some(c) = self.buf.find(&tc) {
            if c > 0 {
                out.push(Event::text("content", &self.buf[..c]));
            }
            self.buf = self.buf[c + tc.len()..].to_string();
            self.state = "call";
            return true;
        }
        let keep = hold_back(&self.buf, &[&tc]);
        if self.buf.len() > keep {
            let cut = self.buf.len() - keep;
            out.push(Event::text("content", &self.buf[..cut]));
            self.buf = self.buf[cut..].to_string();
        }
        false
    }

    fn call_step(&mut self, out: &mut Vec<Event>) -> bool {
        let t = self.t.clone();
        match self.sub {
            "between" => {
                let s = crate::py::lstrip(&self.buf).to_string();
                if s.starts_with(&t.calls_end) {
                    self.buf = s[t.calls_end.len()..].to_string();
                    self.state = "content";
                    self.calls_seen = true;
                    return true;
                }
                if s.starts_with(&t.call) {
                    self.buf = s[t.call.len()..].to_string();
                    self.sub = "name";
                    return true;
                }
                if !s.is_empty() && !(t.calls_end.starts_with(&s) || t.call.starts_with(&s)) {
                    self.buf = format!("{}{}", t.calls, self.buf);
                    self.state = "content";
                    self.calls_seen = true;
                    out.push(Event::text("content", self.buf.clone()));
                    self.buf.clear();
                }
                false
            }
            "name" => {
                let ends: Vec<usize> = [self.buf.find(&t.key), self.buf.find(&t.call_end)].into_iter().flatten().collect();
                let Some(&end) = ends.iter().min() else { return false };
                let call = ToolCall::new(crate::py::strip(&self.buf[..end]).to_string(), Map::new());
                self.args = Map::new();
                self.first = true;
                self.sub = "args";
                self.buf = self.buf[end..].to_string();
                if self.stream_tools {
                    out.push(Event::call("tool_start", &call));
                }
                self.call = Some(call);
                self.push_args(out, "{".into());
                true
            }
            "args" => {
                let s = crate::py::lstrip(&self.buf).to_string();
                if s.starts_with(&t.call_end) {
                    self.buf = s[t.call_end.len()..].to_string();
                    self.sub = "between";
                    self.end_call(out);
                    return true;
                }
                if s.starts_with(&t.key) {
                    self.buf = s[t.key.len()..].to_string();
                    self.sub = "key";
                    self.key.clear();
                    return true;
                }
                false
            }
            "key" => {
                let Some(end) = self.buf.find(&t.key_end) else { return false };
                self.key = self.buf[..end].to_string();
                self.buf = self.buf[end + t.key_end.len()..].to_string();
                self.sub = "open";
                true
            }
            "open" => {
                let s = crate::py::lstrip(&self.buf).to_string();
                if !s.starts_with(&t.value) {
                    if !s.is_empty() && !t.value.starts_with(&s) {
                        self.buf = s;
                        self.sub = "args";
                        return true;
                    }
                    return false;
                }
                self.buf = s[t.value.len()..].to_string();
                self.sub = "value";
                self.raw.clear();
                let name = self.call.as_ref().unwrap().name.clone();
                self.schema = self.props.get(&name).and_then(|p| p.get(&self.key)).cloned();
                let (text, null) = text_kinds(self.schema.as_ref());
                self.text_value = text && !null;
                let piece =
                    format!("{}{}:", if self.first { "" } else { "," }, pyjson::dumps_str(&self.key, false));
                self.push_args(out, piece);
                self.first = false;
                if self.text_value {
                    self.push_args(out, "\"".into());
                }
                true
            }
            _ => {
                // a value
                if let Some(end) = self.buf.find(&t.value_end) {
                    let chunk = self.buf[..end].to_string();
                    self.raw.push_str(&chunk);
                    let v = decode_param(&self.raw, self.schema.as_ref());
                    self.args.insert(self.key.clone(), v.clone());
                    if self.text_value {
                        let j = pyjson::dumps_str(&chunk, false);
                        self.push_args(out, format!("{}\"", &j[1..j.len() - 1]));
                    } else {
                        self.push_args(out, pyjson::dumps(&v, false));
                    }
                    self.buf = self.buf[end + t.value_end.len()..].to_string();
                    self.sub = "args";
                    return true;
                }
                let keep = hold_back(&self.buf, &[&t.value_end]);
                let cut = self.buf.len() - keep;
                let text = self.buf[..cut].to_string();
                self.buf = self.buf[cut..].to_string();
                self.raw.push_str(&text);
                if self.text_value {
                    let j = pyjson::dumps_str(&text, false);
                    self.push_args(out, j[1..j.len() - 1].to_string());
                }
                false
            }
        }
    }
}

impl OutputParse for Hy4Parser {
    fn feed(&mut self, delta: &str) -> AResult<Vec<Event>> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        loop {
            let more = match self.state {
                "reasoning" => self.reasoning(&mut out),
                "content" => self.content(&mut out),
                _ => self.call_step(&mut out),
            };
            if !more {
                return Ok(out);
            }
        }
    }

    fn finish(&mut self) -> AResult<Vec<Event>> {
        let mut out = Vec::new();
        if self.state == "call" {
            if self.call.is_some() {
                if self.sub == "value" {
                    let b = self.buf.clone();
                    self.raw.push_str(&b);
                    if self.text_value {
                        let j = pyjson::dumps_str(&b, false);
                        self.push_args(&mut out, format!("{}\"", &j[1..j.len() - 1]));
                    } else {
                        let v = decode_param(&self.raw, self.schema.as_ref());
                        self.push_args(&mut out, pyjson::dumps(&v, false));
                    }
                }
                self.push_args(&mut out, "}".into());
                self.call = None;
            } else if !crate::py::strip(&self.buf).is_empty() && !self.calls_seen {
                out.push(Event::text("content", format!("{}{}", self.t.calls, self.buf)));
            }
        } else if !self.buf.is_empty() {
            out.push(Event::text(if self.state == "reasoning" { "reasoning" } else { "content" }, self.buf.clone()));
        }
        self.buf.clear();
        Ok(out)
    }

    fn state(&self) -> &'static str {
        self.state
    }

    fn buf(&self) -> &str {
        &self.buf
    }

    fn in_value(&self) -> bool {
        self.state == "call" && self.sub == "value"
    }
}
