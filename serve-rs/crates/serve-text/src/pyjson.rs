//! Python's `json` module and the repr()/format rules the server's bytes depend on.
//!
//! * `loads` / `loads_bytes`: json.loads, with its exact error messages ("Expecting value: line 1 column 1 (char 0)").
//! * `dumps`: json.dumps(obj, ensure_ascii=...) with the default separators (", ", ": ").
//! * `dumps_compact`: separators=(",", ":"), ensure_ascii=True, allow_nan=False.
//! * `py_float_repr`, `py_format_g`, `py_round`, `py_repr`, `py_ascii`: repr(), `{x:g}`, round(), repr()/ascii() of
//!   a JSON value.
//!
//! Values are serde_json::Value with preserve_order (Python dict order). Known gaps: integers beyond i64/u64 become
//! floats, NaN/Infinity literals are refused (serde_json cannot hold them).

use serde_json::{Map, Number, Value};
use std::fmt::Write as _;

/// A json.loads failure: Python's JSONDecodeError (or the UnicodeDecodeError of a byte body) message.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonError(pub String);

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for JsonError {}

// ------------------------------------------------------------------------------------------------ loads
struct Parser<'a> {
    s: &'a [char],
}

#[derive(Debug)]
enum Fail {
    Stop(usize),          // StopIteration(idx): "Expecting value" at idx
    Msg(&'static str, usize),
    Text(String),
}

fn errmsg(doc: &[char], msg: &str, pos: usize) -> String {
    let lineno = doc[..pos].iter().filter(|&&c| c == '\n').count() + 1;
    let colno = match doc[..pos].iter().rposition(|&c| c == '\n') {
        Some(i) => pos - i,
        None => pos + 1,
    };
    format!("{msg}: line {lineno} column {colno} (char {pos})")
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

impl<'a> Parser<'a> {
    fn ws(&self, mut i: usize) -> usize {
        while i < self.s.len() && is_ws(self.s[i]) {
            i += 1;
        }
        i
    }

    fn starts(&self, i: usize, lit: &str) -> bool {
        let mut k = i;
        for c in lit.chars() {
            if k >= self.s.len() || self.s[k] != c {
                return false;
            }
            k += 1;
        }
        true
    }

    fn scan(&self, idx: usize) -> Result<(Value, usize), Fail> {
        if idx >= self.s.len() {
            return Err(Fail::Stop(idx));
        }
        match self.s[idx] {
            '"' => {
                let (s, end) = self.string(idx + 1)?;
                Ok((Value::String(s), end))
            }
            '{' => self.object(idx + 1),
            '[' => self.array(idx + 1),
            'n' if self.starts(idx, "null") => Ok((Value::Null, idx + 4)),
            't' if self.starts(idx, "true") => Ok((Value::Bool(true), idx + 4)),
            'f' if self.starts(idx, "false") => Ok((Value::Bool(false), idx + 5)),
            'N' if self.starts(idx, "NaN") => Err(Fail::Text("NaN is not supported by this server".into())),
            'I' if self.starts(idx, "Infinity") => Err(Fail::Text("Infinity is not supported by this server".into())),
            '-' if self.starts(idx, "-Infinity") => Err(Fail::Text("-Infinity is not supported by this server".into())),
            _ => self.number(idx),
        }
    }

    fn number(&self, idx: usize) -> Result<(Value, usize), Fail> {
        let s = self.s;
        let n = s.len();
        let mut i = idx;
        if i < n && s[i] == '-' {
            i += 1;
        }
        if i < n && s[i] == '0' {
            i += 1;
        } else if i < n && ('1'..='9').contains(&s[i]) {
            while i < n && s[i].is_ascii_digit() {
                i += 1;
            }
        } else {
            return Err(Fail::Stop(idx));
        }
        let int_end = i;
        let mut is_float = false;
        if i + 1 < n && s[i] == '.' && s[i + 1].is_ascii_digit() {
            i += 2;
            while i < n && s[i].is_ascii_digit() {
                i += 1;
            }
            is_float = true;
        }
        if i < n && (s[i] == 'e' || s[i] == 'E') {
            let mut j = i + 1;
            if j < n && (s[j] == '-' || s[j] == '+') {
                j += 1;
            }
            if j < n && s[j].is_ascii_digit() {
                while j < n && s[j].is_ascii_digit() {
                    j += 1;
                }
                i = j;
                is_float = true;
            }
        }
        let text: String = s[idx..i].iter().collect();
        let _ = int_end;
        if !is_float {
            if let Ok(v) = text.parse::<i64>() {
                return Ok((Value::Number(v.into()), i));
            }
            if let Ok(v) = text.parse::<u64>() {
                return Ok((Value::Number(v.into()), i));
            }
        }
        let f: f64 = text.parse().map_err(|_| Fail::Stop(idx))?;
        match Number::from_f64(f) {
            Some(num) => Ok((Value::Number(num), i)),
            None => Err(Fail::Text("a number too large for this server".into())),
        }
    }

    /// `end` is the index after the opening quote.
    fn string(&self, end: usize) -> Result<(String, usize), Fail> {
        let s = self.s;
        let n = s.len();
        let begin = end - 1;
        let mut out = String::new();
        let mut next = end;
        loop {
            // the run of ordinary characters
            let mut c = '\0';
            while next < n {
                c = s[next];
                if c == '"' || c == '\\' {
                    break;
                }
                if (c as u32) <= 0x1f {
                    return Err(Fail::Msg("Invalid control character at", next));
                }
                out.push(c);
                next += 1;
            }
            if next >= n {
                return Err(Fail::Msg("Unterminated string starting at", begin));
            }
            if c == '"' {
                return Ok((out, next + 1));
            }
            // a backslash
            next += 1;
            if next >= n {
                return Err(Fail::Msg("Unterminated string starting at", begin));
            }
            let e = s[next];
            if e != 'u' {
                let ch = match e {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return Err(Fail::Msg("Invalid \\escape", next - 1)),
                };
                out.push(ch);
                next += 1;
                continue;
            }
            // \uXXXX
            next += 1;
            let end4 = next + 4;
            if end4 >= n {
                return Err(Fail::Msg("Invalid \\uXXXX escape", next - 1));
            }
            let hex = |from: usize| -> Option<u32> {
                let mut v = 0u32;
                for k in from..from + 4 {
                    v = v * 16 + s.get(k)?.to_digit(16)?;
                }
                Some(v)
            };
            let Some(mut u) = hex(next) else {
                return Err(Fail::Msg("Invalid \\uXXXX escape", next - 1));
            };
            next = end4;
            if (0xd800..=0xdbff).contains(&u) && next + 1 < n && s[next] == '\\' && s[next + 1] == 'u' {
                if next + 6 < n + 1 {
                    if let Some(u2) = hex(next + 2) {
                        if (0xdc00..=0xdfff).contains(&u2) {
                            u = 0x10000 + (((u - 0xd800) << 10) | (u2 - 0xdc00));
                            next += 6;
                        }
                    }
                }
            }
            // a lone surrogate cannot live in a Rust String: U+FFFD stands in (Python keeps it)
            out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
        }
    }

    fn object(&self, idx0: usize) -> Result<(Value, usize), Fail> {
        let s = self.s;
        let mut map = Map::new();
        let mut idx = self.ws(idx0);
        if idx < s.len() && s[idx] == '}' {
            return Ok((Value::Object(map), idx + 1));
        }
        if idx >= s.len() || s[idx] != '"' {
            return Err(Fail::Msg("Expecting property name enclosed in double quotes", idx));
        }
        loop {
            let (key, after) = self.string(idx + 1)?;
            idx = self.ws(after);
            if idx >= s.len() || s[idx] != ':' {
                return Err(Fail::Msg("Expecting ':' delimiter", idx));
            }
            idx = self.ws(idx + 1);
            let (val, after) = self.scan(idx)?;
            map.insert(key, val);
            idx = self.ws(after);
            if idx < s.len() && s[idx] == '}' {
                return Ok((Value::Object(map), idx + 1));
            }
            if idx >= s.len() || s[idx] != ',' {
                return Err(Fail::Msg("Expecting ',' delimiter", idx));
            }
            idx = self.ws(idx + 1);
            if idx >= s.len() || s[idx] != '"' {
                return Err(Fail::Msg("Expecting property name enclosed in double quotes", idx));
            }
        }
    }

    fn array(&self, idx0: usize) -> Result<(Value, usize), Fail> {
        let s = self.s;
        let mut out = Vec::new();
        let mut idx = self.ws(idx0);
        if idx < s.len() && s[idx] == ']' {
            return Ok((Value::Array(out), idx + 1));
        }
        loop {
            let (val, after) = self.scan(idx)?;
            out.push(val);
            idx = self.ws(after);
            if idx < s.len() && s[idx] == ']' {
                return Ok((Value::Array(out), idx + 1));
            }
            if idx >= s.len() || s[idx] != ',' {
                return Err(Fail::Msg("Expecting ',' delimiter", idx));
            }
            idx = self.ws(idx + 1);
        }
    }
}

/// json.loads of a str.
pub fn loads(text: &str) -> Result<Value, JsonError> {
    let chars: Vec<char> = text.chars().collect();
    if chars.first() == Some(&'\u{feff}') {
        return Err(JsonError(errmsg(&chars, "Unexpected UTF-8 BOM (decode using utf-8-sig)", 0)));
    }
    let p = Parser { s: &chars };
    let start = p.ws(0);
    let fail = |f: Fail| -> JsonError {
        match f {
            Fail::Stop(i) => JsonError(errmsg(&chars, "Expecting value", i)),
            Fail::Msg(m, i) => JsonError(errmsg(&chars, m, i)),
            Fail::Text(t) => JsonError(t),
        }
    };
    let (v, end) = p.scan(start).map_err(fail)?;
    let end = p.ws(end);
    if end != chars.len() {
        return Err(JsonError(errmsg(&chars, "Extra data", end)));
    }
    Ok(v)
}

/// json.loads of bytes: the encoding detected as json.detect_encoding does (UTF-8, with or without BOM; UTF-16/32
/// by their BOM or zero bytes), decoded strictly.
pub fn loads_bytes(b: &[u8]) -> Result<Value, JsonError> {
    let enc = detect_encoding(b);
    let text = match enc {
        "utf-8-sig" => decode_utf8(&b[3..], 3)?,
        "utf-8" => decode_utf8(b, 0)?,
        other => decode_wide(b, other)?,
    };
    loads(&text)
}

fn decode_utf8(b: &[u8], offset: usize) -> Result<String, JsonError> {
    match std::str::from_utf8(b) {
        Ok(s) => Ok(s.to_string()),
        Err(e) => {
            let pos = e.valid_up_to();
            let byte = b[pos];
            let (end, reason) = match e.error_len() {
                None => (b.len(), "unexpected end of data"),
                Some(len) => {
                    let reason = if byte < 0x80 || (0x80..0xc2).contains(&byte) || byte > 0xf4 {
                        "invalid start byte"
                    } else {
                        "invalid continuation byte"
                    };
                    (pos + len, reason)
                }
            };
            let codec = if offset == 3 { "utf-8-sig" } else { "utf-8" };
            if end - pos > 1 && e.error_len().is_none() {
                return Err(JsonError(format!(
                    "'{codec}' codec can't decode bytes in position {}-{}: {reason}",
                    pos,
                    end - 1
                )));
            }
            Err(JsonError(format!("'{codec}' codec can't decode byte 0x{byte:02x} in position {pos}: {reason}")))
        }
    }
}

fn decode_wide(b: &[u8], enc: &str) -> Result<String, JsonError> {
    let bad = || JsonError(format!("'{enc}' codec can't decode the request body"));
    let (unit, le, skip) = match enc {
        "utf-16" => (2, b.first() == Some(&0xff), 2),
        "utf-16-le" => (2, true, 0),
        "utf-16-be" => (2, false, 0),
        "utf-32" => (4, b.first() == Some(&0xff), 4),
        "utf-32-le" => (4, true, 0),
        _ => (4, false, 0),
    };
    let body = &b[skip.min(b.len())..];
    if body.len() % unit != 0 {
        return Err(bad());
    }
    if unit == 2 {
        let units: Vec<u16> = body
            .chunks(2)
            .map(|c| if le { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) })
            .collect();
        String::from_utf16(&units).map_err(|_| bad())
    } else {
        body.chunks(4)
            .map(|c| {
                let v = if le {
                    u32::from_le_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    u32::from_be_bytes([c[0], c[1], c[2], c[3]])
                };
                char::from_u32(v).ok_or_else(bad)
            })
            .collect()
    }
}

fn detect_encoding(b: &[u8]) -> &'static str {
    if b.starts_with(&[0xef, 0xbb, 0xbf]) {
        return "utf-8-sig";
    }
    if b.starts_with(&[0xff, 0xfe, 0, 0]) || b.starts_with(&[0, 0, 0xfe, 0xff]) {
        return "utf-32";
    }
    if b.starts_with(&[0xff, 0xfe]) || b.starts_with(&[0xfe, 0xff]) {
        return "utf-16";
    }
    if b.len() >= 4 {
        if b[0] == 0 {
            return if b[1] != 0 { "utf-16-be" } else { "utf-32-be" };
        }
        if b[1] == 0 {
            return if b[2] != 0 || b[3] != 0 { "utf-16-le" } else { "utf-32-le" };
        }
    } else if b.len() == 2 {
        if b[0] == 0 {
            return "utf-16-be";
        }
        if b[1] == 0 {
            return "utf-16-le";
        }
    }
    "utf-8"
}

// ------------------------------------------------------------------------------------------------ dumps
/// json.dumps(v, ensure_ascii=ensure_ascii) with the default separators.
pub fn dumps(v: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, v, ensure_ascii, ", ", ": ");
    out
}

/// json.dumps(v, ensure_ascii=True, allow_nan=False, separators=(",", ":")).
pub fn dumps_compact(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v, true, ",", ":");
    out
}

/// json.dumps with explicit separators.
pub fn dumps_with(v: &Value, ensure_ascii: bool, item_sep: &str, key_sep: &str) -> String {
    let mut out = String::new();
    write_value(&mut out, v, ensure_ascii, item_sep, key_sep);
    out
}

/// json.dumps of one string.
pub fn dumps_str(s: &str, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_str(&mut out, s, ensure_ascii);
    out
}

fn write_value(out: &mut String, v: &Value, ascii: bool, item: &str, key: &str) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number_repr(n)),
        Value::String(s) => write_str(out, s, ascii),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                write_value(out, x, ascii, item, key);
            }
            out.push(']');
        }
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                write_str(out, k, ascii);
                out.push_str(key);
                write_value(out, x, ascii, item, key);
            }
            out.push('}');
        }
    }
}

/// A JSON number as Python prints it: an int as digits, a float by repr().
pub fn number_repr(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    py_float_repr(n.as_f64().unwrap_or(0.0))
}

fn write_str(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ascii && (c as u32) > 0x7e => {
                let v = c as u32;
                if v >= 0x10000 {
                    let v = v - 0x10000;
                    let _ = write!(out, "\\u{:04x}\\u{:04x}", 0xd800 + (v >> 10), 0xdc00 + (v & 0x3ff));
                } else {
                    let _ = write!(out, "\\u{:04x}", v);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ------------------------------------------------------------------------------------------------ floats
/// The shortest round-trip digits and the decimal exponent: value = 0.d1d2d3... * 10^exp10 written as
/// (digits, point) where point is the position of the decimal point after the first `point` digits.
fn shortest_digits(x: f64) -> (String, i32) {
    // Rust's {:e} gives the shortest round-trip digits: "1.2345e6", "5e-324"
    let s = format!("{:e}", x.abs());
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0').to_string();
    let digits = if digits.is_empty() { "0".to_string() } else { digits };
    (digits, exp + 1)
}

/// repr(float).
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let (digits, decpt) = shortest_digits(x);
    // float_repr_style 'short' with repr's rule: exponent form when decpt <= -4 or decpt > 16
    if decpt > -4 && decpt <= 16 {
        let n = digits.len() as i32;
        let body = if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt >= n {
            format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
        } else {
            format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
        };
        format!("{sign}{body}")
    } else {
        let e = decpt - 1;
        let mant = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        let es = if e < 0 { format!("-{:02}", -e) } else { format!("+{:02}", e) };
        format!("{sign}{mant}e{es}")
    }
}

/// format(x, "g"): 6 significant digits, trailing zeros dropped, exponent form for exp < -4 or exp >= 6.
pub fn py_format_g(x: f64) -> String {
    py_format_g_prec(x, 6)
}

/// format(x, ".{p}g").
pub fn py_format_g_prec(x: f64, p: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let p = p.max(1);
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    // the exponent after rounding to p significant digits
    let sci = format!("{:.*e}", p - 1, x);
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if exp < -4 || exp >= p as i32 {
        let mant = if mant.contains('.') { mant.trim_end_matches('0').trim_end_matches('.') } else { mant };
        let es = if exp < 0 { format!("-{:02}", -exp) } else { format!("+{:02}", exp) };
        format!("{mant}e{es}")
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, x);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

/// round(x, n) for a float: correctly rounded, half to even on the exact binary value (as CPython).
pub fn py_round(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.*}", n, x).parse().unwrap_or(x)
}

/// f"{x:.{n}f}".
pub fn py_fixed(x: f64, n: usize) -> String {
    format!("{:.*}", n, x)
}

// ------------------------------------------------------------------------------------------------ repr
fn is_printable(c: char) -> bool {
    use unicode_general_category::{get_general_category, GeneralCategory as G};
    if c == ' ' {
        return true;
    }
    !matches!(
        get_general_category(c),
        G::Control
            | G::Format
            | G::Surrogate
            | G::PrivateUse
            | G::Unassigned
            | G::LineSeparator
            | G::ParagraphSeparator
            | G::SpaceSeparator
    )
}

/// repr(str).
pub fn py_str_repr(s: &str) -> String {
    str_repr(s, false)
}

/// ascii(str).
pub fn py_str_ascii(s: &str) -> String {
    str_repr(s, true)
}

fn str_repr(s: &str, ascii: bool) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c if (c as u32) < 0x7f => out.push(c),
            c if !ascii && is_printable(c) => out.push(c),
            c => {
                let v = c as u32;
                if v <= 0xff {
                    let _ = write!(out, "\\x{:02x}", v);
                } else if v <= 0xffff {
                    let _ = write!(out, "\\u{:04x}", v);
                } else {
                    let _ = write!(out, "\\U{:08x}", v);
                }
            }
        }
    }
    out.push(quote);
    out
}

/// repr() of a JSON value as Python holds it (dict, list, str, int, float, True/False/None).
pub fn py_repr(v: &Value) -> String {
    repr_inner(v, false)
}

/// ascii() of a JSON value.
pub fn py_ascii(v: &Value) -> String {
    repr_inner(v, true)
}

fn repr_inner(v: &Value, ascii: bool) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => number_repr(n),
        Value::String(s) => str_repr(s, ascii),
        Value::Array(a) => {
            let parts: Vec<String> = a.iter().map(|x| repr_inner(x, ascii)).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(m) => {
            let parts: Vec<String> =
                m.iter().map(|(k, x)| format!("{}: {}", str_repr(k, ascii), repr_inner(x, ascii))).collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// str() of a JSON value: a str is itself, everything else its repr.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python truthiness of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}

/// type(v).__name__ for a JSON value.
pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Python equality of two JSON values (1 == 1.0, True == 1).
pub fn py_eq(a: &Value, b: &Value) -> bool {
    fn num(v: &Value) -> Option<f64> {
        match v {
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Number(n) => n.as_f64(),
            _ => None,
        }
    }
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, p)| y.get(k).map(|q| py_eq(p, q)).unwrap_or(false))
        }
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Null, Value::Null) => true,
        _ => match (num(a), num(b)) {
            (Some(p), Some(q)) => {
                // exact for ints beyond f64's precision
                match (a.as_i64(), b.as_i64()) {
                    (Some(i), Some(j)) => i == j,
                    _ => p == q,
                }
            }
            _ => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats() {
        for (x, s) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (123.456, "123.456"),
            (-2.5e-7, "-2.5e-07"),
            (1.5e300, "1.5e+300"),
            (0.95, "0.95"),
        ] {
            assert_eq!(py_float_repr(x), s);
        }
        assert_eq!(py_format_g(-100.0), "-100");
        assert_eq!(py_format_g(0.5), "0.5");
        assert_eq!(py_format_g(1234567.0), "1.23457e+06");
        assert_eq!(py_format_g(1e-5), "1e-05");
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(py_round(0.125, 2), 0.12);
    }

    #[test]
    fn errors() {
        assert_eq!(loads("").unwrap_err().0, "Expecting value: line 1 column 1 (char 0)");
        assert_eq!(loads("{\"a\" 1}").unwrap_err().0, "Expecting ':' delimiter: line 1 column 6 (char 5)");
        assert_eq!(loads("[1,]").unwrap_err().0, "Expecting value: line 1 column 4 (char 3)");
        assert_eq!(loads("{\"a\":1,}").unwrap_err().0,
                   "Expecting property name enclosed in double quotes: line 1 column 8 (char 7)");
        assert_eq!(loads("{} x").unwrap_err().0, "Extra data: line 1 column 4 (char 3)");
        assert_eq!(loads("\"abc").unwrap_err().0, "Unterminated string starting at: line 1 column 1 (char 0)");
    }

    #[test]
    fn dump() {
        let v = loads(r#"{"a": [1, 2.0, "é\n"], "b": null, "c": {}}"#).unwrap();
        assert_eq!(dumps(&v, false), "{\"a\": [1, 2.0, \"é\\n\"], \"b\": null, \"c\": {}}");
        assert_eq!(dumps(&v, true), "{\"a\": [1, 2.0, \"\\u00e9\\n\"], \"b\": null, \"c\": {}}");
        assert_eq!(py_repr(&v), "{'a': [1, 2.0, 'é\\n'], 'b': None, 'c': {}}");
        assert_eq!(py_str_repr("it's"), "\"it's\"");
    }
}
