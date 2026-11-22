//! Small Python-semantics helpers used across the engine side (str.isdigit, str.split, float()).

/// `s.isdigit()` for ASCII text: non-empty and all digits.
pub fn isdigit(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `s.lstrip("-").isdigit()`: the test Python uses before int(s).
pub fn int_like(s: &str) -> bool {
    isdigit(s.trim_start_matches('-'))
}

/// `int(s)` for the text int_like accepts (None for "--5", which Python would raise on).
pub fn int_of(s: &str) -> Option<i64> {
    s.parse::<i64>().ok()
}

/// Python `str.split()` (whitespace runs, no empty words).
pub fn split(s: &str) -> Vec<&str> {
    s.split(|c: char| c.is_whitespace() || c == '\x1c' || c == '\x1d' || c == '\x1e' || c == '\x1f')
        .filter(|w| !w.is_empty())
        .collect()
}

/// Python `float(s)`: accepts surrounding whitespace, inf/nan words, underscores between digits.
pub fn float_of(s: &str) -> Option<f64> {
    let t = s.trim();
    let low = t.to_ascii_lowercase();
    let body = low.trim_start_matches(['+', '-']);
    let neg = low.starts_with('-');
    match body {
        "inf" | "infinity" => return Some(if neg { f64::NEG_INFINITY } else { f64::INFINITY }),
        "nan" => return Some(f64::NAN),
        _ => {}
    }
    if t.contains('_') {
        // Python allows single underscores between digits only
        let b = t.as_bytes();
        for (i, &c) in b.iter().enumerate() {
            if c == b'_' && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()) {
                return None;
            }
        }
        return t.replace('_', "").parse::<f64>().ok();
    }
    if body.is_empty() || !body.bytes().next().map(|c| c.is_ascii_digit() || c == b'.').unwrap_or(false) {
        return None;
    }
    t.parse::<f64>().ok()
}

/// `s.partition(sep)`.
pub fn partition<'a>(s: &'a str, sep: &str) -> (&'a str, bool, &'a str) {
    match s.find(sep) {
        Some(i) => (&s[..i], true, &s[i + sep.len()..]),
        None => (s, false, ""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floats() {
        assert_eq!(float_of(" 1.5 "), Some(1.5));
        assert_eq!(float_of("1e3"), Some(1000.0));
        assert_eq!(float_of("x"), None);
        assert_eq!(float_of(""), None);
        assert_eq!(float_of("-inf"), Some(f64::NEG_INFINITY));
        assert!(int_like("-12"));
        assert!(!int_like("1.2"));
        assert_eq!(split("  a  b\tc\n"), vec!["a", "b", "c"]);
    }
}

/// Python `str.splitlines()` (no line ends kept).
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

/// Python `str.strip()` (Unicode whitespace).
pub fn strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}
