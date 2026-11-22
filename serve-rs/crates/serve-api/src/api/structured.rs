//! api/structured.py: JSON response formats are checked here; the engine enforces the schema with a grammar.

use crate::error::{AResult, ApiError};
use crate::py::{get, getv, Obj};
use serde_json::{json, Value};

fn check_refs(node: &Value) -> AResult<()> {
    match node {
        Value::Object(m) => {
            for (key, value) in m {
                if (key == "$ref" || key == "$dynamicRef") && value.as_str().map(|s| !s.starts_with('#')).unwrap_or(false)
                {
                    return Err(ApiError::value("response_format supports only local schema references (#...)"));
                }
                check_refs(value)?;
            }
        }
        Value::Array(a) => {
            for v in a {
                check_refs(v)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// check_format: true for a JSON response format, false for plain text; ValueError for what the server cannot take.
pub fn check_format(response_format: Option<&Value>) -> AResult<bool> {
    let rf = match response_format {
        None | Some(Value::Null) => return Ok(false),
        Some(r) => r,
    };
    if !rf.is_object() {
        return Err(ApiError::value("response_format must be an object"));
    }
    let kind = getv(rf, "type");
    if kind == "text" {
        return Ok(false);
    }
    if kind == "json_object" {
        return Ok(true);
    }
    if kind != "json_schema" {
        return Err(ApiError::value("response_format.type must be text, json_object or json_schema"));
    }
    let spec = getv(rf, "json_schema");
    if !spec.is_object() || !getv(spec, "schema").is_object() {
        return Err(ApiError::value("response_format.json_schema needs a schema object"));
    }
    if !getv(spec, "name").as_str().map(|n| !n.is_empty()).unwrap_or(false) {
        return Err(ApiError::value("response_format.json_schema needs a name"));
    }
    if let Some(s) = get(spec, "strict") {
        if !s.is_boolean() {
            return Err(ApiError::value("response_format.json_schema.strict must be boolean"));
        }
    }
    if getv(getv(spec, "schema"), "type") != "object" {
        return Err(ApiError::value("response_format schema must have type object at its root"));
    }
    check_refs(getv(spec, "schema"))?;
    Ok(true)
}

/// anthropic_format: the chat path's response_format for output_config.format (or output_format), or None.
pub fn anthropic_format(req: &Obj) -> AResult<Option<Value>> {
    let mut fmt = match req.get("output_config") {
        Some(o @ Value::Object(_)) => get(o, "format").filter(|f| !f.is_null()),
        _ => None,
    };
    let mut place = "output_config.format";
    if fmt.is_none() {
        if let Some(f) = req.get("output_format").filter(|f| !f.is_null()) {
            fmt = Some(f);
            place = "output_format";
        }
    }
    let Some(fmt) = fmt else { return Ok(None) };
    if !fmt.is_object() || getv(fmt, "type") != "json_schema" {
        return Err(ApiError::value(format!("{place}.type must be json_schema")));
    }
    let rf = json!({"type": "json_schema", "json_schema": {"name": "output", "schema": getv(fmt, "schema").clone()}});
    if let Err(e) = check_format(Some(&rf)) {
        let m = e.message().replace("response_format.json_schema", place).replace("response_format", place);
        return Err(ApiError::value(m));
    }
    Ok(Some(rf))
}
