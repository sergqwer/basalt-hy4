//! api/logit_bias.py: OpenAI `logit_bias`, checked before any streaming.

use crate::error::{AResult, ApiError};
use crate::py::{is_int, Obj};
use serde_json::Value;
use serve_engine::iface::Engine;
use serve_text::pyjson::py_str_repr;

pub const REQUEST_KEY: &str = "_logit_bias";

/// normalize: the request's logit_bias as (token id, bias) in the order sent.
pub fn normalize(value: Option<&Value>, vocab_size: Option<i64>) -> AResult<Vec<(u32, f64)>> {
    let value = match value {
        None | Some(Value::Null) => return Ok(vec![]),
        Some(v) => v,
    };
    let Some(m) = value.as_object() else {
        return Err(ApiError::value("logit_bias: expected an object mapping token ids to biases"));
    };
    let mut out: Vec<(u32, f64)> = Vec::new();
    for (key, bias) in m {
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ApiError::value(format!(
                "logit_bias: {} is not a token id (a non-negative whole number)",
                py_str_repr(key)
            )));
        }
        let digits = key.trim_start_matches('0');
        let shown = if digits.is_empty() { "0".to_string() } else { digits.to_string() };
        let token: u128 = if digits.is_empty() {
            0
        } else if digits.len() > 30 {
            u128::MAX
        } else {
            digits.parse().unwrap()
        };
        if token > (1u128 << 31) - 1 || vocab_size.map(|v| token >= v as u128).unwrap_or(false) {
            return Err(ApiError::value(format!("logit_bias: token id {shown} is outside the model's vocabulary")));
        }
        let token = token as u32;
        if out.iter().any(|(t, _)| *t == token) {
            return Err(ApiError::value(format!("logit_bias: token id {token} appears twice")));
        }
        let b = match bias {
            Value::Number(n) => n.as_f64().filter(|f| f.is_finite() && (-100.0..=100.0).contains(f)),
            _ => None,
        };
        let Some(b) = b else {
            return Err(ApiError::value(format!(
                "logit_bias: the bias of token {token} must be a number from -100 to 100"
            )));
        };
        out.push((token, b));
    }
    if let Some(v) = vocab_size {
        if out.iter().filter(|(_, b)| *b == -100.0).count() as i64 >= v {
            return Err(ApiError::value("logit_bias: it bans every token"));
        }
    }
    Ok(out)
}

/// checked: the request's bias, refused when this engine cannot apply it.
pub fn checked(req: &Obj, engine: &dyn Engine) -> AResult<Vec<(u32, f64)>> {
    let info = engine.info();
    let started = info.as_ref().map(|i| i.contains_key("context")).unwrap_or(false);
    let vocab = if started { info.as_ref().and_then(|i| i.get("logit_bias")).cloned() } else { None };
    let known = vocab.as_ref().map(|v| is_int(v) && v.as_i64().unwrap_or(0) > 1).unwrap_or(false);
    let bias = normalize(req.get("logit_bias"), if known { vocab.and_then(|v| v.as_i64()) } else { None })?;
    if !bias.is_empty() && started && !known {
        return Err(ApiError::value("logit_bias: this engine build cannot apply token biases"));
    }
    if !bias.is_empty() && engine.batch() > 0 {
        return Err(ApiError::value("logit_bias: not supported with parallel slots (--parallel)"));
    }
    Ok(bias)
}
