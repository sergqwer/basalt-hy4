//! A small xml.etree.ElementTree: elements with namespace-expanded tags ("{uri}local") and attribute names, `text`
//! (the text before the first child) and pre-order iteration. Enough for the parts of Office files.

#[derive(Debug, Clone, Default)]
pub struct Element {
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    pub text: String,
    pub tail: String,
    pub children: Vec<Element>,
}

impl Element {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    /// iter(): this element and every one below it, in document order.
    pub fn iter(&self) -> std::vec::IntoIter<&Element> {
        let mut out = vec![];
        fn walk<'a>(e: &'a Element, out: &mut Vec<&'a Element>) {
            out.push(e);
            for c in &e.children {
                walk(c, out);
            }
        }
        walk(self, &mut out);
        out.into_iter()
    }
}

struct P<'a> {
    s: &'a str,
    i: usize,
}

fn err(msg: &str, s: &str, i: usize) -> String {
    let line = s[..i].matches('\n').count() + 1;
    let col = i - s[..i].rfind('\n').map(|x| x + 1).unwrap_or(0);
    format!("{msg}: line {line}, column {col}")
}

fn decode_entities(raw: &str, s: &str, at: usize) -> Result<String, String> {
    if !raw.contains('&') {
        return Ok(raw.to_string());
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let Some(end) = after.find(';') else { return Err(err("not well-formed (invalid token)", s, at)) };
        let name = &after[..end];
        let ch = match name {
            "lt" => Some('<'),
            "gt" => Some('>'),
            "amp" => Some('&'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            n if n.starts_with("#x") => u32::from_str_radix(&n[2..], 16).ok().and_then(char::from_u32),
            n if n.starts_with('#') => n[1..].parse::<u32>().ok().and_then(char::from_u32),
            _ => return Err(err("undefined entity", s, at)),
        };
        match ch {
            Some(c) => out.push(c),
            None => return Err(err("reference to invalid character number", s, at)),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

impl<'a> P<'a> {
    fn starts(&self, t: &str) -> bool {
        self.s[self.i..].starts_with(t)
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.s[self.i..].chars().next() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn skip_misc(&mut self) -> Result<(), String> {
        loop {
            self.skip_ws();
            if self.starts("<?") {
                let e = self.s[self.i..].find("?>").ok_or_else(|| err("unclosed token", self.s, self.i))?;
                self.i += e + 2;
            } else if self.starts("<!--") {
                let e = self.s[self.i..].find("-->").ok_or_else(|| err("unclosed token", self.s, self.i))?;
                self.i += e + 3;
            } else if self.starts("<!DOCTYPE") {
                let e = self.s[self.i..].find('>').ok_or_else(|| err("unclosed token", self.s, self.i))?;
                self.i += e + 1;
            } else {
                return Ok(());
            }
        }
    }

    fn name(&mut self) -> Result<&'a str, String> {
        let start = self.i;
        while let Some(c) = self.s[self.i..].chars().next() {
            if c.is_whitespace() || "/>=".contains(c) {
                break;
            }
            self.i += c.len_utf8();
        }
        if self.i == start {
            return Err(err("not well-formed (invalid token)", self.s, self.i));
        }
        Ok(&self.s[start..self.i])
    }

    fn element(&mut self, ns: &[(String, String)]) -> Result<Element, String> {
        // at '<'
        self.i += 1;
        let qname = self.name()?;
        let mut raw_attrs: Vec<(&str, String)> = vec![];
        loop {
            self.skip_ws();
            if self.starts("/>") || self.starts(">") {
                break;
            }
            let an = self.name()?;
            self.skip_ws();
            if !self.starts("=") {
                return Err(err("not well-formed (invalid token)", self.s, self.i));
            }
            self.i += 1;
            self.skip_ws();
            let q = self.s[self.i..].chars().next().ok_or_else(|| err("unclosed token", self.s, self.i))?;
            if q != '"' && q != '\'' {
                return Err(err("not well-formed (invalid token)", self.s, self.i));
            }
            self.i += 1;
            let e = self.s[self.i..].find(q).ok_or_else(|| err("unclosed token", self.s, self.i))?;
            let v = decode_entities(&self.s[self.i..self.i + e], self.s, self.i)?;
            // attribute values normalize whitespace characters to spaces
            let v = v.replace(['\n', '\t', '\r'], " ");
            raw_attrs.push((an, v));
            self.i += e + 1;
        }
        let mut scope: Vec<(String, String)> = ns.to_vec();
        for (k, v) in &raw_attrs {
            if *k == "xmlns" {
                scope.push((String::new(), v.clone()));
            } else if let Some(p) = k.strip_prefix("xmlns:") {
                scope.push((p.to_string(), v.clone()));
            }
        }
        let expand = |q: &str, attr: bool| -> Result<String, String> {
            let (prefix, local) = match q.split_once(':') {
                Some((p, l)) => (p, l),
                None => ("", q),
            };
            if prefix.is_empty() && attr {
                return Ok(local.to_string());
            }
            if prefix == "xml" {
                return Ok(format!("{{http://www.w3.org/XML/1998/namespace}}{local}"));
            }
            match scope.iter().rev().find(|(p, _)| p == prefix) {
                Some((_, uri)) if !uri.is_empty() => Ok(format!("{{{uri}}}{local}")),
                Some(_) => Ok(local.to_string()),
                None if prefix.is_empty() => Ok(local.to_string()),
                None => Err(err("unbound prefix", self.s, self.i)),
            }
        };
        let mut el = Element { tag: expand(qname, false)?, ..Default::default() };
        for (k, v) in raw_attrs {
            if k == "xmlns" || k.starts_with("xmlns:") {
                continue;
            }
            el.attrs.push((expand(k, true)?, v));
        }
        if self.starts("/>") {
            self.i += 2;
            return Ok(el);
        }
        self.i += 1;
        let mut text = String::new();
        loop {
            if self.i >= self.s.len() {
                return Err(err("no element found", self.s, self.i));
            }
            if self.starts("</") {
                self.i += 2;
                let n = self.name()?;
                if n != qname {
                    return Err(err("mismatched tag", self.s, self.i));
                }
                self.skip_ws();
                if !self.starts(">") {
                    return Err(err("not well-formed (invalid token)", self.s, self.i));
                }
                self.i += 1;
                match el.children.last_mut() {
                    Some(c) => c.tail.push_str(&text),
                    None => el.text.push_str(&text),
                }
                return Ok(el);
            }
            if self.starts("<!--") {
                let e = self.s[self.i..].find("-->").ok_or_else(|| err("unclosed token", self.s, self.i))?;
                self.i += e + 3;
                continue;
            }
            if self.starts("<![CDATA[") {
                let e = self.s[self.i..].find("]]>").ok_or_else(|| err("unclosed CDATA section", self.s, self.i))?;
                text.push_str(&self.s[self.i + 9..self.i + e]);
                self.i += e + 3;
                continue;
            }
            if self.starts("<?") {
                let e = self.s[self.i..].find("?>").ok_or_else(|| err("unclosed token", self.s, self.i))?;
                self.i += e + 2;
                continue;
            }
            if self.starts("<") {
                match el.children.last_mut() {
                    Some(c) => c.tail.push_str(&text),
                    None => el.text.push_str(&text),
                }
                text.clear();
                let child = self.element(&scope)?;
                el.children.push(child);
                continue;
            }
            let e = self.s[self.i..].find('<').unwrap_or(self.s.len() - self.i);
            let raw = &self.s[self.i..self.i + e];
            let at = self.i;
            text.push_str(&decode_entities(&raw.replace("\r\n", "\n").replace('\r', "\n"), self.s, at)?);
            self.i += e;
        }
    }
}

/// ElementTree.fromstring: the root element; Err with expat's kind of message.
pub fn parse(raw: &[u8]) -> Result<Element, String> {
    let raw = raw.strip_prefix(b"\xef\xbb\xbf").unwrap_or(raw);
    let s = std::str::from_utf8(raw).map_err(|_| "not well-formed (invalid token): line 1, column 0".to_string())?;
    let mut p = P { s, i: 0 };
    p.skip_misc()?;
    if !p.starts("<") {
        return Err(err(if s.trim().is_empty() { "no element found" } else { "syntax error" }, s, p.i));
    }
    let root = p.element(&[])?;
    p.skip_misc()?;
    if p.i < s.len() {
        return Err(err("junk after document element", s, p.i));
    }
    Ok(root)
}
