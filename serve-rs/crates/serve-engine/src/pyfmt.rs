//! Python's number formatting and str helpers, byte for byte: repr(float), format(x, 'g'), round(), sum() of floats,
//! str.strip/split/splitlines, int() and float() of text, repr(str).

/// Python `repr(x)` of a float: the shortest round-trip digits, scientific below 1e-4 and from 1e16.
pub fn repr_f64(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let (neg, digits, exp) = sci_parts(&format!("{:e}", x));
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        format!("{sign}{}", fixed_from_digits(&digits, exp, true))
    } else {
        format!("{sign}{}", sci_from_digits(&digits, exp))
    }
}

/// Python `format(x, 'g')`.
pub fn fmt_g(x: f64) -> String {
    fmt_g_prec(x, 6)
}

/// Python `f"{x:.{prec}g}"`.
pub fn fmt_g_prec(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let p = prec.max(1);
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let (neg, digits, exp) = sci_parts(&format!("{:.*e}", p - 1, x));
    let sign = if neg { "-" } else { "" };
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    if exp >= -4 && exp < p as i32 {
        format!("{sign}{}", fixed_from_digits(digits, exp, false))
    } else {
        format!("{sign}{}", sci_from_digits(digits, exp))
    }
}

/// Python `f"{x:.{prec}f}"` (Rust's fixed formatting rounds the exact binary value half-even, as Python does).
pub fn fmt_f(x: f64, prec: usize) -> String {
    format!("{:.*}", prec, x)
}

fn sci_parts(s: &str) -> (bool, String, i32) {
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let (mant, exp) = s.split_once('e').unwrap_or((s, "0"));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    (neg, digits, exp.parse().unwrap_or(0))
}

fn fixed_from_digits(digits: &str, exp: i32, point_zero: bool) -> String {
    let n = digits.len() as i32;
    let out = if exp < 0 {
        format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
    } else if exp + 1 >= n {
        let int = format!("{}{}", digits, "0".repeat((exp + 1 - n) as usize));
        if point_zero {
            format!("{int}.0")
        } else {
            int
        }
    } else {
        let (a, b) = digits.split_at((exp + 1) as usize);
        format!("{a}.{b}")
    };
    out
}

fn sci_from_digits(digits: &str, exp: i32) -> String {
    let (a, b) = digits.split_at(1);
    let m = if b.is_empty() { a.to_string() } else { format!("{a}.{b}") };
    let es = if exp < 0 { "-" } else { "+" };
    format!("{m}e{es}{:02}", exp.abs())
}

/// Python `round(x, n)` for a float: correctly rounded to n decimals, half to even on the exact value.
pub fn round_n(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    fmt_f(x, n).parse().unwrap_or(x)
}

/// Python `round(x)`: the nearest integer, half to even.
pub fn round_int(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Python 3.12+ `sum()` of floats (Neumaier-compensated, as CPython's builtin sum).
pub fn py_sum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let (mut f, mut c) = (0.0f64, 0.0f64);
    for x in xs {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// `str.isspace()` of one character.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// `s.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `s.rstrip()`.
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `s.split()`.
pub fn split_ws(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|w| !w.is_empty()).collect()
}

/// `s.splitlines()`: \n, \r, \r\n, \v, \f, \x1c-\x1e, \x85, \u2028, \u2029 end a line; no empty last line.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let brk = matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if brk {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(_, '\n')) = it.peek() {
                    it.next();
                    end += 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `s.isdigit()` for ASCII digits (the engine's lines are ASCII).
pub fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Python `int(s)`: surrounding whitespace, a sign and single underscores between digits; None where Python raises.
pub fn py_int(s: &str) -> Option<i64> {
    let t = strip(s);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body.is_empty() || body.starts_with('_') || body.ends_with('_') || body.contains("__") {
        return None;
    }
    let digits: String = body.chars().filter(|c| *c != '_').collect();
    if !is_digits(&digits) {
        return None;
    }
    let v: i64 = digits.parse().ok()?;
    Some(if neg { -v } else { v })
}

/// Python `float(s)`: whitespace, inf/nan words, underscores between digits; None where Python raises.
pub fn py_float(s: &str) -> Option<f64> {
    let t = strip(s);
    let low = t.to_ascii_lowercase();
    let body = low.trim_start_matches(['+', '-']);
    if low.len() - body.len() > 1 {
        return None;
    }
    let neg = low.starts_with('-');
    let special = match body {
        "inf" | "infinity" => Some(f64::INFINITY),
        "nan" => Some(f64::NAN),
        _ => None,
    };
    if let Some(v) = special {
        return Some(if neg { -v } else { v });
    }
    if body.starts_with('_') || body.ends_with('_') || body.contains("__") {
        return None;
    }
    let ok_unders = body.char_indices().all(|(i, c)| {
        c != '_' || {
            let b = body.as_bytes();
            b[i - 1].is_ascii_digit() && b.get(i + 1).is_some_and(|x| x.is_ascii_digit())
        }
    });
    if !ok_unders || body.is_empty() {
        return None;
    }
    let clean: String = low.chars().filter(|c| *c != '_').collect();
    if !clean.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'+' | b'-')) {
        return None;
    }
    clean.parse().ok()
}

/// Python `repr(s)` of a str (quotes chosen as Python does; printable non-ASCII kept).
pub fn repr_str(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (0x80..0xa0).contains(&(c as u32)) => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// `s[:n]` by characters.
pub fn head_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `s[n:]` by characters.
pub fn tail_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[i..],
        None => "",
    }
}

/// `len(s)` (code points).
pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// `s.ljust(n)`.
pub fn ljust(s: &str, n: usize) -> String {
    let l = char_len(s);
    if l >= n {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(n - l))
    }
}

/// `s.rjust(n)`.
pub fn rjust(s: &str, n: usize) -> String {
    let l = char_len(s);
    if l >= n {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(n - l))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr() {
        for (x, s) in [(0.125, "0.125"), (1e16, "1e+16"), (1.5e-5, "1.5e-05"), (123456789.0, "123456789.0"),
                       (0.1, "0.1"), (1.0, "1.0"), (-2.5, "-2.5"), (0.0001, "0.0001"), (1e22, "1e+22"),
                       (9999999999999998.0, "9999999999999998.0"), (1.2345e-7, "1.2345e-07")] {
            assert_eq!(repr_f64(x), s, "{x}");
        }
    }

    #[test]
    fn g() {
        for (x, s) in [(0.125, "0.125"), (1e16, "1e+16"), (1.5e-5, "1.5e-05"), (123456789.0, "1.23457e+08"),
                       (100000.0, "100000"), (1000000.0, "1e+06"), (0.0001, "0.0001"), (0.00001, "1e-05"),
                       (2.0, "2"), (0.0, "0"), (3.14159265, "3.14159")] {
            assert_eq!(fmt_g(x), s, "{x}");
        }
        assert_eq!(fmt_g_prec(0.949999988079071, 2), "0.95");
        assert_eq!(fmt_g_prec(0.699999988079071, 1), "0.7");
    }

    #[test]
    fn rounding() {
        assert_eq!(round_n(0.125, 2), 0.12);
        assert_eq!(round_n(0.375, 2), 0.38);
        assert_eq!(round_n(2.675, 2), 2.67);
        assert_eq!(round_int(2.5), 2);
        assert_eq!(round_int(3.5), 4);
        assert_eq!(fmt_f(0.25, 1), "0.2");
        assert_eq!(fmt_f(0.35, 1), "0.3");
        assert_eq!(py_sum([0.1; 10]), 1.0);
    }

    #[test]
    fn text() {
        assert_eq!(splitlines("a\nb\r\nc\rd\x0be\n"), vec!["a", "b", "c", "d", "e"]);
        assert_eq!(splitlines("a\n\nb"), vec!["a", "", "b"]);
        assert_eq!(split_ws(" a \x1c b\t"), vec!["a", "b"]);
        assert_eq!(py_int(" +1_0 "), Some(10));
        assert_eq!(py_int("1__0"), None);
        assert_eq!(py_float("52.1"), Some(52.1));
        assert_eq!(py_float("x"), None);
        assert_eq!(repr_str("C:\\x"), "'C:\\\\x'");
        assert_eq!(repr_str("it's"), "\"it's\"");
    }
}
