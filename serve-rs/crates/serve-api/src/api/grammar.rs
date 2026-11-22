//! api/grammar.py: the `grammar` key of an engine request.

use crate::error::{AResult, ApiError};
use crate::frontend::tool_choice_of;
use crate::py::{get, getv, truthy};
use serde_json::{json, Value};

pub const REQUEST_KEY: &str = "_grammar";

fn strict(value: &Value) -> bool {
    match value {
        Value::Object(_) => {
            get(value, "strict") == Some(&Value::Bool(true)) || strict(getv(value, "function")) || strict(getv(value, "tools"))
        }
        Value::Array(a) => a.iter().any(strict),
        _ => false,
    }
}

fn as_function(tool: &Value) -> Value {
    if getv(tool, "type") == "function" && getv(tool, "function").is_object() {
        tool.clone()
    } else {
        json!({"type": "function", "function": tool})
    }
}

/// grammar_spec: the `grammar` value for a request, or None.
pub fn grammar_spec(
    enabled: bool,
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    raw_tools: Option<&Value>,
    response_format: Option<&Value>,
) -> AResult<Option<Value>> {
    let mut answer: Option<Value> = None;
    let rf = response_format.filter(|r| r.is_object());
    let kind = rf.map(|r| getv(r, "type").clone()).unwrap_or(Value::Null);
    if kind == "json_schema" {
        let empty = json!({});
        let spec = rf.and_then(|r| get(r, "json_schema")).filter(|s| truthy(s)).unwrap_or(&empty);
        if enabled || get(spec, "strict") == Some(&Value::Bool(true)) {
            let schema = get(spec, "schema").cloned().ok_or_else(|| ApiError::internal("'schema'"))?;
            answer = Some(json!({"kind": "json_schema", "schema": schema}));
        }
    } else if kind == "json_object" && enabled {
        answer = Some(json!({"kind": "json_object"}));
    }
    let tools = match tools.and_then(|t| t.as_array()) {
        Some(t) if !t.is_empty() => t,
        _ => return Ok(answer),
    };
    let calls: Vec<Value> = tools.iter().map(as_function).collect();
    if let Some(a) = &answer {
        if tool_choice_of(tool_choice)?.0 == "auto" {
            let mut d = a.as_object().unwrap().clone();
            d.insert("tools".into(), Value::Array(calls));
            return Ok(Some(Value::Object(d)));
        }
    }
    if enabled || raw_tools.map(strict).unwrap_or(false) {
        let tc = match tool_choice {
            None | Some(Value::Null) => Value::from("auto"),
            Some(v) => v.clone(),
        };
        return Ok(Some(json!({"kind": "tools", "tools": calls, "tool_choice": tc})));
    }
    Ok(None)
}

/// unenforced: response_format asks for JSON output but the request would get no grammar key.
pub fn unenforced(enabled: bool, response_format: Option<&Value>) -> AResult<bool> {
    let kind = response_format.filter(|r| r.is_object()).map(|r| getv(r, "type").clone()).unwrap_or(Value::Null);
    Ok((kind == "json_schema" || kind == "json_object") && grammar_spec(enabled, None, None, None, response_format)?.is_none())
}
