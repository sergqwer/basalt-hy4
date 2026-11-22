//! Small Python semantics the ported code leans on: dict.get on JSON, truthiness, int(), str slicing by characters.

use serde_json::{Map, Value};
pub use serve_text::pyjson::{py_eq, py_repr, py_str, truthy, type_name};

pub type Obj = Map<String, Value>;

pub static NULL: Value = Value::Null;

/// d.get(k) for a JSON value that may not be a dict (None then).
pub fn get<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    v.as_object().and_then(|m| m.get(k))
}

/// d.get(k) with None for absent or null (`is not None` checks).
pub fn get_some<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    get(v, k).filter(|x| !x.is_null())
}

/// d.get(k) or a null reference.
pub fn getv<'a>(v: &'a Value, k: &str) -> &'a Value {
    get(v, k).unwrap_or(&NULL)
}

pub fn get_str<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    get(v, k).and_then(|x| x.as_str())
}

pub fn is_dict(v: &Value) -> bool {
    v.is_object()
}

/// isinstance(v, int) and not isinstance(v, bool): a JSON integer.
pub fn is_int(v: &Value) -> bool {
    matches!(v, Value::Number(n) if !n.is_f64())
}

pub fn as_int(v: &Value) -> Option<i64> {
    if is_int(v) {
        v.as_i64()
    } else {
        None
    }
}

/// isinstance(v, (int, float)) and not bool.
pub fn is_number(v: &Value) -> bool {
    v.is_number()
}

/// isinstance(v, float) and v.is_integer() -> int, as several fields accept 5.0 for 5.
pub fn whole(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) if !n.is_f64() => n.as_i64(),
        Value::Number(n) => {
            let f = n.as_f64()?;
            if f.fract() == 0.0 && f.is_finite() {
                Some(f as i64)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// int(v) as Python's int() takes it (str of digits with whitespace, float truncated, bool); None where int() raises.
pub fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(*b as i64),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)
            }
        }
        Value::String(s) => {
            let t = s.trim_matches(crate::template::py_isspace).replace('_', "");
            if s.contains("__") || s.trim().starts_with('_') || s.trim().ends_with('_') {
                return None;
            }
            t.parse::<i64>().ok()
        }
        _ => None,
    }
}

/// The first `n` characters of `s`.
pub fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// The last `n` characters of `s`.
pub fn tail(s: &str, n: usize) -> &str {
    if n == 0 {
        return "";
    }
    match s.char_indices().rev().nth(n - 1) {
        Some((i, _)) => &s[i..],
        None => s,
    }
}

/// s[-2:-1]: the character before the last one ("" when there is none).
pub fn second_last(s: &str) -> &str {
    let t = tail(s, 2);
    if t.chars().count() < 2 {
        return "";
    }
    head(t, 1)
}

pub fn first_char(s: &str) -> Option<char> {
    s.chars().next()
}

/// str.lstrip() / rstrip() / strip() with Python's whitespace.
pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(crate::template::py_isspace)
}

pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(crate::template::py_isspace)
}

pub fn strip(s: &str) -> &str {
    s.trim_matches(crate::template::py_isspace)
}

/// Python's `x in list` (== membership).
pub fn contains(list: &[Value], x: &Value) -> bool {
    list.iter().any(|y| py_eq(x, y))
}

/// A new random hex id (uuid4().hex).
pub fn uuid_hex() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

pub fn obj(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    Value::Object(m)
}

#[macro_export]
macro_rules! jobj {
    ($($k:expr => $v:expr),* $(,)?) => {{
        let mut m = serde_json::Map::new();
        $( m.insert(($k).to_string(), serde_json::Value::from($v)); )*
        serde_json::Value::Object(m)
    }};
}

/// `int(v)` with Python's exceptions: ValueError (a 400) for a string that is no integer, TypeError / OverflowError
/// (uncaught in the handler) for other types and infinities.
pub fn int_call(v: &Value) -> crate::error::AResult<i64> {
    if let Some(n) = py_int(v) {
        return Ok(n);
    }
    match v {
        Value::String(s) => Err(crate::error::ApiError::value(format!(
            "invalid literal for int() with base 10: {}",
            serve_text::pyjson::py_str_repr(s)
        ))),
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(0.0);
            Err(crate::error::ApiError::internal(if f.is_nan() {
                "ValueError: cannot convert float NaN to integer".to_string()
            } else {
                "OverflowError: cannot convert float infinity to integer".to_string()
            }))
        }
        other => Err(crate::error::ApiError::internal(format!(
            "TypeError: int() argument must be a string, a bytes-like object or a real number, not '{}'",
            serve_text::pyjson::type_name(other)
        ))),
    }
}

/// `int(req.get(a) or req.get(b) or ... or 0)`: the first truthy value through int().
pub fn int_of_first(req: &Obj, keys: &[&str]) -> crate::error::AResult<i64> {
    for k in keys {
        if let Some(v) = req.get(*k).filter(|v| truthy(v)) {
            return int_call(v);
        }
    }
    Ok(0)
}
