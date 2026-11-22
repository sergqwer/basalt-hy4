//! config.py's request-side rules: SAMPLING_KEYS, checked_sampling and GRAMMAR_DEFAULT.
use serde_json::Value;
use serve_text::pyjson::py_repr;

/// The `grammar` option when neither the file nor the command line sets it.
pub const GRAMMAR_DEFAULT: bool = true;
/// The engine's penalty window cap (kPenaltyWindowCap).
pub const PENALTY_WINDOW_MAX: i64 = 4096;
/// SAMPLING_KEYS in its order.
pub const SAMPLING_ORDER: [&str; 9] = ["temperature", "top_p", "min_p", "top_k", "presence_penalty", "frequency_penalty",
    "repetition_penalty", "penalty_last_n", "seed"];

/// `_num(v)`: an int or float, not a bool, finite.
pub fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64().filter(|x| x.is_finite()),
        _ => None,
    }
}

/// `_whole(v)`: a number equal to its int.
pub fn whole(v: &Value) -> Option<f64> {
    num(v).filter(|x| *x == x.trunc())
}

fn temperature(v: &Value) -> bool { num(v).map(|x| x >= 0.0).unwrap_or(false) }
fn top_p(v: &Value) -> bool { num(v).map(|x| 0.0 < x && x <= 1.0).unwrap_or(false) }
fn min_p(v: &Value) -> bool { num(v).map(|x| (0.0..=1.0).contains(&x)).unwrap_or(false) }
fn top_k(v: &Value) -> bool { whole(v).map(|x| x >= -1.0).unwrap_or(false) }
fn any_num(v: &Value) -> bool { num(v).is_some() }
fn repetition(v: &Value) -> bool { num(v).map(|x| x > 0.0).unwrap_or(false) }
fn last_n(v: &Value) -> bool { whole(v).map(|x| 0.0 <= x && x <= PENALTY_WINDOW_MAX as f64).unwrap_or(false) }
fn seed(v: &Value) -> bool {
    match v {
        Value::Number(n) if n.is_i64() || n.is_u64() => true,
        _ => whole(v).map(|x| x >= -(2f64.powi(63)) && x < 2f64.powi(64)).unwrap_or(false),
    }
}

/// key -> (converts to int, test, what is expected)
pub fn sampling_rule(key: &str) -> Option<(bool, fn(&Value) -> bool, String)> {
    Some(match key {
        "temperature" => (false, temperature as fn(&Value) -> bool, "a number >= 0 (0 = greedy)".into()),
        "top_p" => (false, top_p, "0 < top_p <= 1".into()),
        "min_p" => (false, min_p, "0 <= min_p <= 1".into()),
        "top_k" => (true, top_k, "an integer >= 0 (0 or -1 = off: every token)".into()),
        "presence_penalty" => (false, any_num, "a number".into()),
        "frequency_penalty" => (false, any_num, "a number".into()),
        "repetition_penalty" => (false, repetition, "a number > 0 (1 = off)".into()),
        "penalty_last_n" => (true, last_n, format!("an integer 0..{PENALTY_WINDOW_MAX} (the engine's penalty window)")),
        "seed" => (true, seed, "an integer".into()),
        _ => return None,
    })
}

/// A request's sampling fields against SAMPLING_KEYS: Err naming the first one out of range.
pub fn checked_sampling(req: &serde_json::Map<String, Value>) -> Result<(), String> {
    for key in SAMPLING_ORDER {
        let (_, ok, expected) = sampling_rule(key).unwrap();
        if let Some(v) = req.get(key) {
            if !v.is_null() && !ok(v) {
                return Err(format!("{key}: expected {expected}, not {}", py_repr(v)));
            }
        }
    }
    if let Some(rd) = req.get("relaxed_drafting") {
        let fine = rd.is_null() || *rd == Value::Bool(false) || num(rd).map(|x| (0.0..=1.0).contains(&x)).unwrap_or(false);
        if !fine {
            return Err(format!("relaxed_drafting: expected false or a number 0..1, not {}", py_repr(rd)));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn checks() {
        let m = |v: Value| v.as_object().unwrap().clone();
        assert!(checked_sampling(&m(json!({"temperature": 0.7, "top_k": 40}))).is_ok());
        assert_eq!(checked_sampling(&m(json!({"top_p": 0}))).unwrap_err(), "top_p: expected 0 < top_p <= 1, not 0");
        assert_eq!(checked_sampling(&m(json!({"temperature": true}))).unwrap_err(),
                   "temperature: expected a number >= 0 (0 = greedy), not True");
        assert!(checked_sampling(&m(json!({"relaxed_drafting": false}))).is_ok());
        assert!(checked_sampling(&m(json!({"relaxed_drafting": true}))).is_err());
    }
}
