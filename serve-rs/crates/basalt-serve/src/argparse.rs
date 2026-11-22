//! The part of Python's argparse (3.12) that serve/config.py's parser uses: optionals only, store / store_true /
//! append / BooleanOptionalAction / nargs="?", parse_known_args, the help and usage text and the error messages.
use serde_json::Value;

pub type TypeFn = fn(&str) -> Result<Value, String>;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Store,
    StoreTrue,
    Append,
    BoolOpt,
    Help,
}

#[derive(Clone)]
pub struct Arg {
    pub flags: Vec<String>,
    pub dest: String,
    pub kind: Kind,
    pub default: Value,
    pub help: Option<String>, // None: SUPPRESS
    pub metavar: Option<String>,
    pub ty: Option<TypeFn>,
    pub type_name: &'static str, // for "invalid <name> value"
    pub choices: Option<Vec<String>>,
    pub optional_nargs: bool, // nargs="?"
    pub constant: Option<String>,
}

impl Arg {
    pub fn new(flag: &str) -> Self {
        Arg {
            flags: vec![flag.to_string()],
            dest: flag.trim_start_matches('-').replace('-', "_"),
            kind: Kind::Store,
            default: Value::Null,
            help: Some(String::new()),
            metavar: None,
            ty: None,
            type_name: "",
            choices: None,
            optional_nargs: false,
            constant: None,
        }
    }
    pub fn kind(mut self, k: Kind) -> Self {
        self.kind = k;
        if k == Kind::StoreTrue {
            self.default = Value::Bool(false);
        }
        if k == Kind::BoolOpt {
            let f = self.flags[0].clone();
            self.flags.push(format!("--no-{}", &f[2..]));
        }
        self
    }
    pub fn default(mut self, v: Value) -> Self {
        self.default = v;
        self
    }
    pub fn help(mut self, h: &str) -> Self {
        self.help = Some(h.to_string());
        self
    }
    pub fn suppress(mut self) -> Self {
        self.help = None;
        self
    }
    pub fn metavar(mut self, m: &str) -> Self {
        self.metavar = Some(m.to_string());
        self
    }
    pub fn ty(mut self, f: TypeFn, name: &'static str) -> Self {
        self.ty = Some(f);
        self.type_name = name;
        self
    }
    pub fn choices(mut self, c: &[&str]) -> Self {
        self.choices = Some(c.iter().map(|s| s.to_string()).collect());
        self
    }
    pub fn nargs_optional(mut self, constant: &str) -> Self {
        self.optional_nargs = true;
        self.constant = Some(constant.to_string());
        self
    }
    fn takes_value(&self) -> bool {
        matches!(self.kind, Kind::Store | Kind::Append)
    }
    fn metavar_text(&self) -> String {
        if let Some(m) = &self.metavar {
            return m.clone();
        }
        if let Some(c) = &self.choices {
            return format!("{{{}}}", c.join(","));
        }
        self.dest.to_uppercase()
    }
    fn args_text(&self) -> String {
        let m = self.metavar_text();
        if self.optional_nargs {
            format!("[{m}]")
        } else {
            m
        }
    }
    fn invocation(&self) -> String {
        if !self.takes_value() {
            return self.flags.join(", ");
        }
        let a = self.args_text();
        self.flags.iter().map(|f| format!("{f} {a}")).collect::<Vec<_>>().join(", ")
    }
    fn usage_part(&self) -> String {
        match self.kind {
            Kind::BoolOpt => format!("[{}]", self.flags.join(" | ")),
            _ if self.takes_value() => format!("[{} {}]", self.flags[0], self.args_text()),
            _ => format!("[{}]", self.flags[0]),
        }
    }
}

pub struct Parser {
    pub prog: String,
    pub description: String,
    pub args: Vec<Arg>,
}

/// What a parse found: dest -> value, in the parser's order.
pub type Namespace = serde_json::Map<String, Value>;

/// argparse's exit: the text for stderr (or stdout for --help) and the status.
#[derive(Debug, Clone, PartialEq)]
pub struct Exit {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

fn negative_number(s: &str) -> bool {
    // argparse: '^-\d+$|^-\d*\.\d+$'
    let b = s.as_bytes();
    if b.len() < 2 || b[0] != b'-' {
        return false;
    }
    let r = &s[1..];
    if r.bytes().all(|c| c.is_ascii_digit()) {
        return true;
    }
    if let Some((a, c)) = r.split_once('.') {
        return a.bytes().all(|x| x.is_ascii_digit()) && !c.is_empty() && c.bytes().all(|x| x.is_ascii_digit());
    }
    false
}

/// argparse's 'O' (an option-looking word) vs 'A' (an argument).
fn looks_optional(s: &str) -> bool {
    if s.is_empty() || !s.starts_with('-') || s == "-" {
        return false;
    }
    if s == "--" {
        return true;
    }
    if negative_number(s) {
        return false;
    }
    if s.contains(' ') {
        return false;
    }
    true
}

impl Parser {
    pub fn new(prog: &str, description: &str) -> Self {
        let help = Arg { dest: "help".into(), kind: Kind::Help, help: Some("show this help message and exit".into()),
                         flags: vec!["-h".into(), "--help".into()], ..Arg::new("--help") };
        Parser { prog: prog.into(), description: description.into(), args: vec![help] }
    }
    pub fn add(&mut self, a: Arg) {
        self.args.push(a);
    }
    pub fn find(&self, flag: &str) -> Option<&Arg> {
        self.args.iter().find(|a| a.flags.iter().any(|f| f == flag))
    }

    fn width() -> usize {
        let cols = std::env::var("COLUMNS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(80);
        cols.saturating_sub(2)
    }

    pub fn format_usage(&self) -> String {
        let width = Self::width();
        let prefix = "usage: ";
        let parts: Vec<String> = self.args.iter().filter(|a| a.help.is_some()).map(|a| a.usage_part()).collect();
        let usage = format!("{} {}", self.prog, parts.join(" "));
        if prefix.len() + usage.len() <= width {
            return format!("{prefix}{usage}\n");
        }
        // wrap: the prog on the first line, the parts indented under it
        let indent = " ".repeat(prefix.len() + self.prog.len() + 1);
        let text_width = width;
        let get_lines = |parts: &[String], indent: &str, pfx: Option<&str>| -> Vec<String> {
            let mut lines = Vec::new();
            let mut line: Vec<String> = Vec::new();
            let mut line_len = match pfx {
                Some(p) => p.len() - 1,
                None => indent.len() - 1,
            };
            for part in parts {
                if line_len + 1 + part.len() > text_width && !line.is_empty() {
                    lines.push(format!("{indent}{}", line.join(" ")));
                    line.clear();
                    line_len = indent.len() - 1;
                }
                line.push(part.clone());
                line_len += part.len() + 1;
            }
            if !line.is_empty() {
                lines.push(format!("{indent}{}", line.join(" ")));
            }
            if pfx.is_some() && !lines.is_empty() {
                lines[0] = lines[0][indent.len()..].to_string();
            }
            lines
        };
        let lines = if prefix.len() + self.prog.len() <= (0.75 * text_width as f64) as usize {
            let mut l = get_lines(&parts, &indent, Some(&format!("{} ", self.prog)));
            if !l.is_empty() {
                l[0] = format!("{} {}", self.prog, l[0]);
            }
            l
        } else {
            let ind = " ".repeat(prefix.len());
            let mut l = vec![self.prog.clone()];
            l.extend(get_lines(&parts, &ind, None));
            l
        };
        format!("{prefix}{}\n", lines.join("\n"))
    }

    pub fn format_help(&self) -> String {
        let width = Self::width();
        let max_help_position = 24usize.min(width.saturating_sub(20).max(4));
        let mut out = self.format_usage();
        out.push('\n');
        if !self.description.is_empty() {
            out.push_str(&self.description);
            out.push_str("\n\n");
        }
        out.push_str("options:\n");
        let shown: Vec<&Arg> = self.args.iter().filter(|a| a.help.is_some()).collect();
        let max_len = shown.iter().map(|a| a.invocation().len() + 2).max().unwrap_or(0);
        let help_position = (max_len + 2).min(max_help_position);
        let help_width = width.saturating_sub(help_position).max(11);
        let action_width = help_position - 2 - 2;
        for a in shown {
            let header = a.invocation();
            let help = a.help.as_deref().unwrap_or("");
            let indent_first;
            if help.is_empty() {
                out.push_str(&format!("  {header}\n"));
                continue;
            } else if header.len() <= action_width {
                out.push_str(&format!("  {header:<action_width$}  "));
                indent_first = 0;
            } else {
                out.push_str(&format!("  {header}\n"));
                indent_first = help_position;
            }
            let lines = wrap(&collapse_ws(help), help_width);
            if let Some(first) = lines.first() {
                out.push_str(&format!("{}{}\n", " ".repeat(indent_first), first));
            }
            for l in lines.iter().skip(1) {
                out.push_str(&format!("{}{}\n", " ".repeat(help_position), l));
            }
        }
        out
    }

    /// parser.error(message): usage and `prog: error: message` on stderr, status 2.
    pub fn error(&self, message: &str) -> Exit {
        Exit { code: 2, stdout: String::new(), stderr: format!("{}{}: error: {}\n", self.format_usage(), self.prog, message) }
    }

    fn defaults(&self) -> Namespace {
        let mut ns = Namespace::new();
        for a in &self.args {
            if a.kind == Kind::Help {
                continue;
            }
            ns.insert(a.dest.clone(), a.default.clone());
        }
        ns
    }

    fn convert(&self, a: &Arg, text: &str) -> Result<Value, Exit> {
        let name = &a.flags.iter().find(|f| f.starts_with("--")).cloned().unwrap_or_else(|| a.flags[0].clone());
        let v = match a.ty {
            Some(f) => f(text).map_err(|e| {
                if e.starts_with('\u{0}') {
                    self.error(&format!("argument {}: invalid {} value: {}", name, a.type_name,
                                        serve_text::pyjson::py_str_repr(text)))
                } else {
                    self.error(&format!("argument {name}: {e}"))
                }
            })?,
            None => Value::String(text.to_string()),
        };
        if let Some(c) = &a.choices {
            let s = v.as_str().unwrap_or("").to_string();
            if !c.contains(&s) {
                return Err(self.error(&format!("argument {name}: invalid choice: {} (choose from {})",
                                               serve_text::pyjson::py_str_repr(&s), c.join(", "))));
            }
        }
        Ok(v)
    }

    /// parse_known_args(words): the namespace and the words it did not know, in order.
    pub fn parse_known_args(&self, words: &[String]) -> Result<(Namespace, Vec<String>), Exit> {
        let mut ns = self.defaults();
        let mut extras = Vec::new();
        let mut appended: std::collections::HashSet<String> = Default::default();
        let mut i = 0;
        while i < words.len() {
            let w = &words[i];
            if w == "--" {
                extras.extend(words[i + 1..].iter().cloned());
                break;
            }
            if !looks_optional(w) {
                extras.push(w.clone());
                i += 1;
                continue;
            }
            let (flag, attached) = match w.split_once('=') {
                Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
                _ => (w.clone(), None),
            };
            let Some(a) = self.find(&flag) else {
                extras.push(w.clone());
                i += 1;
                continue;
            };
            i += 1;
            match a.kind {
                Kind::Help => {
                    return Err(Exit { code: 0, stdout: self.format_help(), stderr: String::new() });
                }
                Kind::StoreTrue | Kind::BoolOpt => {
                    if attached.is_some() {
                        return Err(self.error(&format!("argument {}: ignored explicit argument {}",
                            a.flags.join("/"), serve_text::pyjson::py_str_repr(attached.as_deref().unwrap()))));
                    }
                    let on = !(a.kind == Kind::BoolOpt && flag.starts_with("--no-"));
                    ns.insert(a.dest.clone(), Value::Bool(on));
                }
                Kind::Store | Kind::Append => {
                    let text = if let Some(v) = attached {
                        Some(v)
                    } else if i < words.len() && !looks_optional(&words[i]) {
                        i += 1;
                        Some(words[i - 1].clone())
                    } else {
                        None
                    };
                    let text = match text {
                        Some(t) => t,
                        None if a.optional_nargs => a.constant.clone().unwrap_or_default(),
                        None => return Err(self.error(&format!("argument {}: expected one argument", a.flags.join("/")))),
                    };
                    let v = self.convert(a, &text)?;
                    if a.kind == Kind::Append {
                        if !appended.contains(&a.dest) {
                            appended.insert(a.dest.clone());
                            // argparse copies the default list and appends to it
                            let base = match ns.get(&a.dest) {
                                Some(Value::Array(x)) => x.clone(),
                                _ => vec![],
                            };
                            ns.insert(a.dest.clone(), Value::Array(base));
                        }
                        if let Some(Value::Array(x)) = ns.get_mut(&a.dest) {
                            x.push(v);
                        }
                    } else {
                        ns.insert(a.dest.clone(), v);
                    }
                }
            }
        }
        Ok((ns, extras))
    }
}

/// A type error that argparse words as "invalid <type> value: 'x'".
pub fn invalid() -> String {
    "\u{0}".to_string()
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

/// textwrap's chunks: words, the spaces between them, and hyphenated words split after their hyphens.
fn chunks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            let s = i;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            out.push(chars[s..i].iter().collect());
            continue;
        }
        let s = i;
        let mut j = i;
        loop {
            if j >= chars.len() || chars[j].is_whitespace() {
                break;
            }
            if chars[j] == '-' && j > s {
                let lt2 = j >= 2 && is_letter(chars[j - 1]) && is_letter(chars[j - 2]);
                let ltlt = j >= 3 && is_letter(chars[j - 1]) && chars[j - 2] == '-' && is_letter(chars[j - 3]);
                let ahead = j + 2 < chars.len() && is_letter(chars[j + 1])
                    && (is_letter(chars[j + 2]) || (chars[j + 2] == '-' && j + 3 < chars.len() && is_letter(chars[j + 3])));
                if (lt2 || ltlt) && ahead {
                    j += 1;
                    out.push(chars[s..j].iter().collect());
                    i = j;
                    break;
                }
            }
            j += 1;
        }
        if i == s {
            out.push(chars[s..j].iter().collect());
            i = j;
        }
    }
    out
}

/// textwrap.wrap(text, width) with the defaults argparse uses.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut ch: Vec<String> = chunks(text);
    ch.reverse();
    let mut lines = Vec::new();
    while !ch.is_empty() {
        let mut cur: Vec<String> = Vec::new();
        let mut cur_len = 0usize;
        if !lines.is_empty() && ch.last().map(|c| c.trim().is_empty()).unwrap_or(false) {
            ch.pop();
        }
        while let Some(c) = ch.last() {
            let l = c.chars().count();
            if cur_len + l <= width {
                cur_len += l;
                cur.push(ch.pop().unwrap());
            } else {
                break;
            }
        }
        if let Some(c) = ch.last() {
            if c.chars().count() > width {
                // break a long word
                let space_left = if width < 1 { 1 } else { width - cur_len };
                let c = ch.pop().unwrap();
                let head: String = c.chars().take(space_left).collect();
                let tail: String = c.chars().skip(space_left).collect();
                cur.push(head);
                ch.push(tail);
            }
        }
        if cur.last().map(|c| c.trim().is_empty()).unwrap_or(false) {
            cur.pop();
        }
        if !cur.is_empty() {
            lines.push(cur.concat());
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hyphen_chunks() {
        assert_eq!(chunks("a --max-context b"), vec!["a", " ", "--max-", "context", " ", "b"]);
        assert_eq!(chunks("slot-full-context"), vec!["slot-", "full-", "context"]);
        assert_eq!(chunks("x-y"), vec!["x-y"]);
    }
    #[test]
    fn neg() {
        assert!(negative_number("-1"));
        assert!(negative_number("-.5"));
        assert!(!looks_optional("-1"));
        assert!(looks_optional("--x"));
    }
}
