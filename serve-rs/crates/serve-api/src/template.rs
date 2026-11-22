//! frontend.ChatTemplate: the model's Jinja chat template rendered with the settings of transformers'
//! apply_chat_template (trim_blocks, lstrip_blocks, loopcontrols, `{% generation %}`, a json.dumps `tojson`,
//! `raise_exception`), on minijinja with the Python behaviours the templates rely on: str() of values
//! (None, True, lists and dicts as Python prints them), str methods (`format`, `startswith`, `strip`, ...), dict
//! methods (`items`, `get`, ...) and `trim` as str.strip().

use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::{Map, Value as Json};
use serve_text::pyjson;
use std::cell::RefCell;

/// A render failure: the template's own refusal (a 400 with its message) or anything else (Python lets that
/// exception escape the request handler).
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateErr {
    Request(String),
    Other(String),
}

impl std::fmt::Display for TemplateErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateErr::Request(m) | TemplateErr::Other(m) => f.write_str(m),
        }
    }
}

thread_local! {
    static RAISED: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// A JSON value as a template value, map order kept.
pub fn from_json(v: &Json) -> Value {
    match v {
        Json::Null => Value::from(()),
        Json::Bool(b) => Value::from(*b),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::from(i)
            } else if let Some(u) = n.as_u64() {
                Value::from(u)
            } else {
                Value::from(n.as_f64().unwrap_or(0.0))
            }
        }
        Json::String(s) => Value::from(s.as_str()),
        Json::Array(a) => a.iter().map(from_json).collect(),
        Json::Object(m) => Value::from_pairs(m.iter().map(|(k, v)| (Value::from(k.as_str()), from_json(v)))),
    }
}

pub struct ChatTemplate {
    pub source: String,
    env: Environment<'static>,
}

/// Python's str.isspace for one character.
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// str.strip() with Python's whitespace.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// str() of a template value, as Jinja2 prints it.
pub fn py_str(v: &Value) -> String {
    match v.kind() {
        ValueKind::Undefined => String::new(),
        ValueKind::String => v.as_str().unwrap_or_default().to_string(),
        _ => py_repr(v),
    }
}

/// repr() of a template value.
pub fn py_repr(v: &Value) -> String {
    match v.kind() {
        ValueKind::Undefined => String::new(),
        ValueKind::None => "None".into(),
        ValueKind::Bool => if v.is_true() { "True" } else { "False" }.into(),
        ValueKind::Number => {
            if let Ok(i) = i64::try_from(v.clone()) {
                if !is_float(v) {
                    return i.to_string();
                }
            }
            match f64::try_from(v.clone()) {
                Ok(f) if is_float(v) => pyjson::py_float_repr(f),
                _ => v.to_string(),
            }
        }
        ValueKind::String => pyjson::py_str_repr(v.as_str().unwrap_or_default()),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<String> = v.try_iter().map(|it| it.map(|x| py_repr(&x)).collect()).unwrap_or_default();
            format!("[{}]", items.join(", "))
        }
        ValueKind::Map => {
            let mut parts = Vec::new();
            if let Ok(keys) = v.try_iter() {
                for k in keys {
                    let x = v.get_item(&k).unwrap_or_default();
                    parts.push(format!("{}: {}", py_repr(&k), py_repr(&x)));
                }
            }
            format!("{{{}}}", parts.join(", "))
        }
        _ => v.to_string(),
    }
}

fn is_float(v: &Value) -> bool {
    // minijinja keeps the float-ness of a number: its Display has a '.' or an exponent, or is inf/nan
    let s = v.to_string();
    s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") || s.contains("nan")
}

/// A template value as JSON (for tojson); Err for what json.dumps cannot take.
pub fn to_json(v: &Value) -> Result<Json, Error> {
    match v.kind() {
        ValueKind::Undefined => Err(Error::new(ErrorKind::BadSerialization, "Object of type Undefined is not JSON serializable")),
        ValueKind::None => Ok(Json::Null),
        ValueKind::Bool => Ok(Json::Bool(v.is_true())),
        ValueKind::Number => {
            if !is_float(v) {
                if let Ok(i) = i64::try_from(v.clone()) {
                    return Ok(Json::from(i));
                }
            }
            let f = f64::try_from(v.clone()).map_err(|e| Error::new(ErrorKind::BadSerialization, e.to_string()))?;
            serde_json::Number::from_f64(f)
                .map(Json::Number)
                .ok_or_else(|| Error::new(ErrorKind::BadSerialization, "NaN"))
        }
        ValueKind::String => Ok(Json::String(v.as_str().unwrap_or_default().to_string())),
        ValueKind::Seq | ValueKind::Iterable => {
            let mut out = Vec::new();
            for x in v.try_iter()? {
                out.push(to_json(&x)?);
            }
            Ok(Json::Array(out))
        }
        ValueKind::Map => {
            let mut m = Map::new();
            for k in v.try_iter()? {
                let key = match k.kind() {
                    ValueKind::String => k.as_str().unwrap().to_string(),
                    ValueKind::None => "null".into(),
                    ValueKind::Bool => if k.is_true() { "true" } else { "false" }.into(),
                    ValueKind::Number => py_repr(&k),
                    _ => {
                        return Err(Error::new(
                            ErrorKind::BadSerialization,
                            "keys must be str, int, float, bool or None",
                        ))
                    }
                };
                m.insert(key, to_json(&v.get_item(&k)?)?);
            }
            Ok(Json::Object(m))
        }
        _ => Err(Error::new(ErrorKind::BadSerialization, "not JSON serializable")),
    }
}

fn tojson(v: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let mut j = to_json(v)?;
    if sort_keys == Some(true) {
        j = sorted(j);
    }
    let ascii = ensure_ascii.unwrap_or(false);
    let (item, key) = match &separators {
        Some(s) if !s.is_none() => {
            let parts: Vec<String> = s.try_iter()?.map(|x| py_str(&x)).collect();
            (parts.first().cloned().unwrap_or(", ".into()), parts.get(1).cloned().unwrap_or(": ".into()))
        }
        _ => {
            if indent.as_ref().map(|i| !i.is_none()).unwrap_or(false) {
                (",".to_string(), ": ".to_string())
            } else {
                (", ".to_string(), ": ".to_string())
            }
        }
    };
    let out = match indent.filter(|i| !i.is_none()) {
        Some(i) => {
            let pad = match i.kind() {
                ValueKind::String => i.as_str().unwrap().to_string(),
                _ => " ".repeat(i64::try_from(i).unwrap_or(0).max(0) as usize),
            };
            indented(&j, ascii, &pad, &item, &key, 0)
        }
        None => pyjson::dumps_with(&j, ascii, &item, &key),
    };
    Ok(Value::from(out))
}

fn sorted(j: Json) -> Json {
    match j {
        Json::Object(m) => {
            let mut items: Vec<(String, Json)> = m.into_iter().collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            Json::Object(items.into_iter().map(|(k, v)| (k, sorted(v))).collect())
        }
        Json::Array(a) => Json::Array(a.into_iter().map(sorted).collect()),
        other => other,
    }
}

fn indented(j: &Json, ascii: bool, pad: &str, item: &str, key: &str, level: usize) -> String {
    let nl = |l: usize| format!("\n{}", pad.repeat(l));
    match j {
        Json::Array(a) if !a.is_empty() => {
            let parts: Vec<String> = a.iter().map(|x| indented(x, ascii, pad, item, key, level + 1)).collect();
            format!("[{}{}{}]", nl(level + 1), parts.join(&format!("{}{}", item, nl(level + 1))), nl(level))
        }
        Json::Object(m) if !m.is_empty() => {
            let parts: Vec<String> = m
                .iter()
                .map(|(k, x)| {
                    format!("{}{}{}", pyjson::dumps_str(k, ascii), key, indented(x, ascii, pad, item, key, level + 1))
                })
                .collect();
            format!("{{{}{}{}}}", nl(level + 1), parts.join(&format!("{}{}", item, nl(level + 1))), nl(level))
        }
        other => pyjson::dumps(other, ascii),
    }
}

fn raise_exception(message: Value) -> Result<Value, Error> {
    let text = py_str(&message);
    RAISED.with(|r| *r.borrow_mut() = Some(text.clone()));
    Err(Error::new(ErrorKind::InvalidOperation, text))
}

/// `{% generation %}...{% endgeneration %}` renders its body as it is: the same block tags as an `if true`.
fn without_generation(source: &str) -> String {
    let re = regex::Regex::new(r"\{%(-?)\s*generation\s*(-?)%\}").unwrap();
    let s = re.replace_all(source, "{%${1} if true ${2}%}");
    let re = regex::Regex::new(r"\{%(-?)\s*endgeneration\s*(-?)%\}").unwrap();
    re.replace_all(&s, "{%${1} endif ${2}%}").into_owned()
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<Self, String> {
        let mut env = Environment::new();
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| e.to_string())?,
        );
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        env.set_formatter(|out, _state, value| {
            out.write_str(&py_str(value)).map_err(|_| Error::from(ErrorKind::WriteFailure))
        });
        env.set_unknown_method_callback(|state, value, method, args| {
            if value.kind() == ValueKind::String {
                let s = value.as_str().unwrap_or_default();
                let none_arg = args.first().map(|a| a.is_none()).unwrap_or(true);
                if matches!(method, "strip" | "lstrip" | "rstrip") && none_arg && args.len() <= 1 {
                    let out = match method {
                        "strip" => s.trim_matches(py_isspace),
                        "lstrip" => s.trim_start_matches(py_isspace),
                        _ => s.trim_end_matches(py_isspace),
                    };
                    return Ok(Value::from(out));
                }
            }
            minijinja_contrib::pycompat::unknown_method_callback(state, value, method, args)
        });
        env.add_filter("tojson", tojson);
        env.add_filter("string", |v: Value| Value::from(py_str(&v)));
        env.add_filter("trim", |v: Value, chars: Option<String>| -> Value {
            let s = py_str(&v);
            match chars {
                Some(c) => Value::from(s.trim_matches(|x| c.contains(x)).to_string()),
                None => Value::from(py_strip(&s).to_string()),
            }
        });
        env.add_function("raise_exception", raise_exception);
        let source_owned = source.to_string();
        env.add_template_owned("chat", without_generation(source)).map_err(|e| e.to_string())?;
        Ok(ChatTemplate { source: source_owned, env })
    }

    /// render(messages, tools=, add_generation_prompt=, **kwargs).
    pub fn render(
        &self,
        messages: &Json,
        tools: Option<&Json>,
        add_generation_prompt: bool,
        kwargs: &Map<String, Json>,
    ) -> Result<String, TemplateErr> {
        let mut ctx = std::collections::BTreeMap::new();
        ctx.insert("messages".to_string(), from_json(messages));
        ctx.insert("tools".to_string(), tools.map(from_json).unwrap_or(Value::from(())));
        ctx.insert("add_generation_prompt".to_string(), Value::from(add_generation_prompt));
        for (k, v) in kwargs {
            ctx.insert(k.clone(), from_json(v));
        }
        RAISED.with(|r| *r.borrow_mut() = None);
        let tpl = self.env.get_template("chat").map_err(|e| TemplateErr::Other(e.to_string()))?;
        match tpl.render(Value::from(ctx)) {
            Ok(s) => Ok(s),
            Err(e) => match RAISED.with(|r| r.borrow_mut().take()) {
                Some(m) => Err(TemplateErr::Request(m)),
                None => Err(TemplateErr::Other(format!("{e:#}"))),
            },
        }
    }
}

