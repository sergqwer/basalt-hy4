//! The PDF reading documents.py does with pypdf: page count, each page's text layer and its largest embedded image.
//! Not pypdf: the object model is read the way pypdf's non-strict reader recovers it (every `N G obj` in the file,
//! object streams, the last trailer), and text comes from the content stream's text operators with pypdf's plain
//! layout rules (a new line when the baseline moves by more than the font's height, a space for a gap). Simple PDFs
//! (one font, ToUnicode or standard encodings, Flate streams) give pypdf's text; unusual layouts can differ in where
//! spaces and line ends fall.

use base64::Engine as _;
use std::collections::HashMap;
use std::io::Read;

#[derive(Debug, Clone, PartialEq)]
pub enum Obj {
    Null,
    Bool(bool),
    Num(f64),
    Name(String),
    Str(Vec<u8>),
    Arr(Vec<Obj>),
    Dict(Vec<(String, Obj)>),
    Ref(u32, u16),
    Stream(Vec<(String, Obj)>, Vec<u8>),
}

impl Obj {
    fn dict(&self) -> Option<&Vec<(String, Obj)>> {
        match self {
            Obj::Dict(d) | Obj::Stream(d, _) => Some(d),
            _ => None,
        }
    }
    fn get(&self, k: &str) -> Option<&Obj> {
        self.dict()?.iter().find(|(x, _)| x == k).map(|(_, v)| v)
    }
    fn num(&self) -> Option<f64> {
        match self {
            Obj::Num(n) => Some(*n),
            _ => None,
        }
    }
    fn name(&self) -> Option<&str> {
        match self {
            Obj::Name(n) => Some(n),
            _ => None,
        }
    }
}

pub enum PdfErr {
    Encrypted,
    Bad(String),
}

// -------------------------------------------------------------------------------------------- lexer
struct Lex<'a> {
    b: &'a [u8],
    i: usize,
}

fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | b'\x0c' | b'\0')
}

fn is_delim(c: u8) -> bool {
    matches!(c, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

enum Tok {
    Obj(Obj),
    Op(String),
    ArrEnd,
    DictEnd,
    Eof,
}

impl<'a> Lex<'a> {
    fn skip(&mut self) {
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if is_ws(c) {
                self.i += 1;
            } else if c == b'%' {
                while self.i < self.b.len() && self.b[self.i] != b'\n' && self.b[self.i] != b'\r' {
                    self.i += 1;
                }
            } else {
                break;
            }
        }
    }

    fn word(&mut self) -> &'a [u8] {
        let s = self.i;
        while self.i < self.b.len() && !is_ws(self.b[self.i]) && !is_delim(self.b[self.i]) {
            self.i += 1;
        }
        &self.b[s..self.i]
    }

    fn literal(&mut self) -> Vec<u8> {
        // at '(' already consumed
        let mut out = vec![];
        let mut depth = 1;
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'(' => {
                    depth += 1;
                    out.push(c);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    out.push(c);
                }
                b'\\' => {
                    if self.i >= self.b.len() {
                        break;
                    }
                    let e = self.b[self.i];
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'\r' => {
                            if self.b.get(self.i) == Some(&b'\n') {
                                self.i += 1;
                            }
                        }
                        b'\n' => {}
                        b'0'..=b'7' => {
                            let mut v = (e - b'0') as u32;
                            for _ in 0..2 {
                                match self.b.get(self.i) {
                                    Some(d @ b'0'..=b'7') => {
                                        v = v * 8 + (d - b'0') as u32;
                                        self.i += 1;
                                    }
                                    _ => break,
                                }
                            }
                            out.push(v as u8);
                        }
                        other => out.push(other),
                    }
                }
                _ => out.push(c),
            }
        }
        out
    }

    fn hex(&mut self) -> Vec<u8> {
        let mut digits = vec![];
        while self.i < self.b.len() && self.b[self.i] != b'>' {
            let c = self.b[self.i];
            if c.is_ascii_hexdigit() {
                digits.push(c);
            }
            self.i += 1;
        }
        self.i += 1;
        if digits.len() % 2 == 1 {
            digits.push(b'0');
        }
        digits.chunks(2).map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap()).collect()
    }

    fn tok(&mut self) -> Tok {
        self.skip();
        if self.i >= self.b.len() {
            return Tok::Eof;
        }
        let c = self.b[self.i];
        match c {
            b'/' => {
                self.i += 1;
                let w = self.word();
                let mut n = vec![];
                let mut k = 0;
                while k < w.len() {
                    if w[k] == b'#' && k + 2 < w.len() + 0 && k + 2 <= w.len() - 1 {
                        if let Ok(v) = u8::from_str_radix(std::str::from_utf8(&w[k + 1..k + 3]).unwrap_or("zz"), 16) {
                            n.push(v);
                            k += 3;
                            continue;
                        }
                    }
                    n.push(w[k]);
                    k += 1;
                }
                Tok::Obj(Obj::Name(String::from_utf8_lossy(&n).into_owned()))
            }
            b'(' => {
                self.i += 1;
                Tok::Obj(Obj::Str(self.literal()))
            }
            b'<' if self.b.get(self.i + 1) == Some(&b'<') => {
                self.i += 2;
                let mut d = vec![];
                loop {
                    match self.tok() {
                        Tok::DictEnd | Tok::Eof => break,
                        Tok::Obj(Obj::Name(k)) => {
                            let v = match self.value() {
                                Some(v) => v,
                                None => break,
                            };
                            d.push((k, v));
                        }
                        _ => {}
                    }
                }
                Tok::Obj(Obj::Dict(d))
            }
            b'<' => {
                self.i += 1;
                Tok::Obj(Obj::Str(self.hex()))
            }
            b'>' if self.b.get(self.i + 1) == Some(&b'>') => {
                self.i += 2;
                Tok::DictEnd
            }
            b'[' => {
                self.i += 1;
                let mut a = vec![];
                loop {
                    match self.tok() {
                        Tok::ArrEnd | Tok::Eof => break,
                        Tok::Obj(o) => a.push(o),
                        Tok::Op(op) if op == "R" => {
                            if a.len() >= 2 {
                                let g = a.pop().unwrap();
                                let n = a.pop().unwrap();
                                if let (Obj::Num(n), Obj::Num(g)) = (n, g) {
                                    a.push(Obj::Ref(n as u32, g as u16));
                                }
                            }
                        }
                        Tok::Op(op) => match op.as_str() {
                            "true" => a.push(Obj::Bool(true)),
                            "false" => a.push(Obj::Bool(false)),
                            "null" => a.push(Obj::Null),
                            _ => {}
                        },
                        Tok::DictEnd => {}
                    }
                }
                Tok::Obj(Obj::Arr(a))
            }
            b']' => {
                self.i += 1;
                Tok::ArrEnd
            }
            b'{' | b'}' | b')' | b'>' => {
                self.i += 1;
                self.tok()
            }
            _ => {
                let w = self.word();
                if w.is_empty() {
                    self.i += 1;
                    return self.tok();
                }
                let s = String::from_utf8_lossy(w).into_owned();
                if let Ok(n) = s.parse::<f64>() {
                    Tok::Obj(Obj::Num(n))
                } else if s.starts_with('.') || s.starts_with("-.") || s.starts_with("+") {
                    Tok::Obj(Obj::Num(s.trim_start_matches('+').parse().unwrap_or(0.0)))
                } else {
                    Tok::Op(s)
                }
            }
        }
    }

    /// One value, with `n g R` references and booleans read.
    fn value(&mut self) -> Option<Obj> {
        match self.tok() {
            Tok::Obj(Obj::Num(n)) => {
                let save = self.i;
                if let Tok::Obj(Obj::Num(g)) = self.tok() {
                    let save2 = self.i;
                    if let Tok::Op(op) = self.tok() {
                        if op == "R" {
                            return Some(Obj::Ref(n as u32, g as u16));
                        }
                    }
                    self.i = save2;
                }
                self.i = save;
                Some(Obj::Num(n))
            }
            Tok::Obj(o) => Some(o),
            Tok::Op(op) => Some(match op.as_str() {
                "true" => Obj::Bool(true),
                "false" => Obj::Bool(false),
                _ => Obj::Null,
            }),
            _ => None,
        }
    }
}

// -------------------------------------------------------------------------------------------- the document
pub struct Pdf {
    data: Vec<u8>,
    offsets: HashMap<u32, usize>,
    in_streams: HashMap<u32, (u32, usize)>,
    cache: std::cell::RefCell<HashMap<u32, Obj>>,
    pages: Vec<Obj>,
}

fn find_all(hay: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut out = vec![];
    if needle.is_empty() || hay.len() < needle.len() {
        return out;
    }
    let mut i = 0;
    while i + needle.len() <= hay.len() {
        if &hay[i..i + needle.len()] == needle {
            out.push(i);
        }
        i += 1;
    }
    out
}

pub fn inflate(data: &[u8]) -> Vec<u8> {
    let mut out = vec![];
    let mut d = flate2::read::ZlibDecoder::new(data);
    if d.read_to_end(&mut out).is_ok() {
        return out;
    }
    // a damaged stream: what decoded
    let mut out2 = vec![];
    let mut d = flate2::read::ZlibDecoder::new(data);
    let mut buf = [0u8; 4096];
    while let Ok(n) = d.read(&mut buf) {
        if n == 0 {
            break;
        }
        out2.extend_from_slice(&buf[..n]);
    }
    out2
}

impl Pdf {
    pub fn open(data: &[u8]) -> Result<Pdf, PdfErr> {
        let mut pdf = Pdf {
            data: data.to_vec(),
            offsets: HashMap::new(),
            in_streams: HashMap::new(),
            cache: Default::default(),
            pages: vec![],
        };
        // every `N G obj`, the last one of a number winning (incremental updates)
        for at in find_all(data, b"obj") {
            if at > 0 && !is_ws(data[at - 1]) {
                continue;
            }
            if data.get(at + 3).map(|c| !is_ws(*c) && !is_delim(*c)).unwrap_or(false) {
                continue;
            }
            let mut j = at;
            while j > 0 && is_ws(data[j - 1]) {
                j -= 1;
            }
            let ge = j;
            while j > 0 && data[j - 1].is_ascii_digit() {
                j -= 1;
            }
            if j == ge {
                continue;
            }
            let gs = j;
            while j > 0 && is_ws(data[j - 1]) {
                j -= 1;
            }
            let ne = j;
            while j > 0 && data[j - 1].is_ascii_digit() {
                j -= 1;
            }
            if j == ne {
                continue;
            }
            let n: u32 = std::str::from_utf8(&data[j..ne]).ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let _g = &data[gs..ge];
            pdf.offsets.insert(n, at + 3);
        }
        if pdf.offsets.is_empty() {
            return Err(PdfErr::Bad("Could not read malformed PDF file".into()));
        }
        // object streams
        let nums: Vec<u32> = pdf.offsets.keys().copied().collect();
        for n in nums {
            if let Some(Obj::Stream(d, _)) = pdf.raw(n) {
                let s = Obj::Stream(d.clone(), vec![]);
                if s.get("Type").and_then(|t| t.name()) == Some("ObjStm") {
                    let count = s.get("N").and_then(|x| x.num()).unwrap_or(0.0) as usize;
                    let body = pdf.stream_data(n);
                    let mut lx = Lex { b: &body, i: 0 };
                    for k in 0..count {
                        let (Some(Obj::Num(on)), Some(Obj::Num(_))) = (lx.value_num(), lx.value_num()) else { break };
                        pdf.in_streams.entry(on as u32).or_insert((n, k));
                    }
                }
            }
        }
        let trailer = pdf.trailer();
        if trailer.as_ref().and_then(|t| t.get("Encrypt")).is_some() {
            return Err(PdfErr::Encrypted);
        }
        let root = trailer
            .as_ref()
            .and_then(|t| t.get("Root").cloned())
            .map(|r| pdf.resolve(&r))
            .or_else(|| pdf.find_catalog())
            .ok_or_else(|| PdfErr::Bad("Cannot find \"/Root\" key in trailer".into()))?;
        let pages = root.get("Pages").map(|p| pdf.resolve(p)).ok_or_else(|| PdfErr::Bad("Cannot find \"/Pages\"".into()))?;
        let mut out = vec![];
        pdf.walk_pages(&pages, &mut out, 0);
        pdf.pages = out;
        Ok(pdf)
    }

    fn find_catalog(&self) -> Option<Obj> {
        let mut nums: Vec<u32> = self.offsets.keys().chain(self.in_streams.keys()).copied().collect();
        nums.sort();
        nums.into_iter().map(|n| self.object(n)).find(|o| o.get("Type").and_then(|t| t.name()) == Some("Catalog"))
    }

    fn trailer(&self) -> Option<Obj> {
        let mut best: Option<Obj> = None;
        for at in find_all(&self.data, b"trailer") {
            let mut lx = Lex { b: &self.data, i: at + 7 };
            if let Some(o @ Obj::Dict(_)) = lx.value() {
                if o.get("Root").is_some() {
                    best = Some(o);
                }
            }
        }
        if best.is_some() {
            return best;
        }
        // an XRef stream's dictionary
        let mut nums: Vec<u32> = self.offsets.keys().copied().collect();
        nums.sort();
        for n in nums.into_iter().rev() {
            if let Some(o) = self.raw(n) {
                if o.get("Type").and_then(|t| t.name()) == Some("XRef") && o.get("Root").is_some() {
                    return Some(Obj::Dict(o.dict().unwrap().clone()));
                }
            }
        }
        None
    }

    fn raw(&self, n: u32) -> Option<Obj> {
        let at = *self.offsets.get(&n)?;
        let mut lx = Lex { b: &self.data, i: at };
        let v = lx.value()?;
        let d = match &v {
            Obj::Dict(d) => d.clone(),
            _ => return Some(v),
        };
        lx.skip();
        if self.data[lx.i..].starts_with(b"stream") {
            let mut s = lx.i + 6;
            if self.data.get(s) == Some(&b'\r') {
                s += 1;
            }
            if self.data.get(s) == Some(&b'\n') {
                s += 1;
            }
            let len = match d.iter().find(|(k, _)| k == "Length").map(|(_, v)| v) {
                Some(Obj::Num(n)) => Some(*n as usize),
                Some(Obj::Ref(r, _)) if *r != n => self.raw(*r).and_then(|o| o.num()).map(|x| x as usize),
                _ => None,
            };
            let end = match len {
                Some(l) if s + l <= self.data.len() && self.data[s + l..].iter().take(32).collect::<Vec<_>>().windows(1).count() > 0
                    && find_all(&self.data[s + l..(s + l + 32).min(self.data.len())], b"endstream").first().is_some() =>
                {
                    s + l
                }
                _ => {
                    let e = find_all(&self.data[s..], b"endstream").first().map(|e| s + e).unwrap_or(self.data.len());
                    let mut e2 = e;
                    while e2 > s && (self.data[e2 - 1] == b'\n' || self.data[e2 - 1] == b'\r') {
                        e2 -= 1;
                    }
                    e2
                }
            };
            return Some(Obj::Stream(d, self.data[s..end].to_vec()));
        }
        Some(Obj::Dict(d))
    }

    pub fn object(&self, n: u32) -> Obj {
        if let Some(o) = self.cache.borrow().get(&n) {
            return o.clone();
        }
        let o = if let Some(&(sn, k)) = self.in_streams.get(&n).filter(|_| !self.offsets.contains_key(&n) || true) {
            if self.offsets.contains_key(&n) && self.in_streams.get(&n).is_some() {
                // a plain object written later overrides the compressed one
                self.raw(n).unwrap_or(Obj::Null)
            } else {
                self.from_stream(sn, k).unwrap_or(Obj::Null)
            }
        } else {
            self.raw(n).unwrap_or(Obj::Null)
        };
        self.cache.borrow_mut().insert(n, o.clone());
        o
    }

    fn from_stream(&self, sn: u32, k: usize) -> Option<Obj> {
        let s = self.raw(sn)?;
        let first = s.get("First").and_then(|x| x.num())? as usize;
        let body = self.stream_data(sn);
        let mut lx = Lex { b: &body, i: 0 };
        let mut off = 0;
        for j in 0..=k {
            let _ = lx.value_num();
            let o = lx.value_num();
            if j == k {
                off = o.and_then(|x| x.num()).unwrap_or(0.0) as usize;
            }
        }
        let mut lx = Lex { b: &body, i: first + off };
        lx.value()
    }

    fn resolve(&self, o: &Obj) -> Obj {
        let mut cur = o.clone();
        for _ in 0..32 {
            match cur {
                Obj::Ref(n, _) => cur = self.object(n),
                other => return other,
            }
        }
        Obj::Null
    }

    fn stream_data(&self, n: u32) -> Vec<u8> {
        match self.object_or_raw(n) {
            Obj::Stream(d, raw) => decode(&Obj::Stream(d, vec![]), &raw, self),
            _ => vec![],
        }
    }

    fn object_or_raw(&self, n: u32) -> Obj {
        self.raw(n).unwrap_or(Obj::Null)
    }

    fn walk_pages(&self, node: &Obj, out: &mut Vec<Obj>, depth: usize) {
        if depth > 64 {
            return;
        }
        let kids = node.get("Kids").map(|k| self.resolve(k));
        match kids {
            Some(Obj::Arr(a)) => {
                for k in a {
                    let page = self.resolve(&k);
                    let mut inherited = page.clone();
                    if let Obj::Dict(d) = &mut inherited {
                        for key in ["Resources", "MediaBox"] {
                            if !d.iter().any(|(x, _)| x == key) {
                                if let Some(v) = node.get(key) {
                                    d.push((key.into(), v.clone()));
                                }
                            }
                        }
                    }
                    if inherited.get("Type").and_then(|t| t.name()) == Some("Pages") || inherited.get("Kids").is_some() {
                        self.walk_pages(&inherited, out, depth + 1);
                    } else {
                        out.push(inherited);
                    }
                }
            }
            _ => {
                if node.get("Type").and_then(|t| t.name()) == Some("Page") {
                    out.push(node.clone());
                }
            }
        }
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    fn contents(&self, page: &Obj) -> Vec<u8> {
        let mut out = vec![];
        let c = page.get("Contents").cloned().unwrap_or(Obj::Null);
        let items = match self.resolve(&c) {
            Obj::Arr(a) => a,
            _ => vec![c],
        };
        for it in items {
            let n = match it {
                Obj::Ref(n, _) => n,
                _ => continue,
            };
            out.extend(self.stream_data(n));
            out.push(b'\n');
        }
        out
    }

    /// page.extract_text()
    pub fn page_text(&self, k: usize) -> Result<String, String> {
        let page = &self.pages[k];
        let res = page.get("Resources").map(|r| self.resolve(r)).unwrap_or(Obj::Null);
        let body = self.contents(page);
        Ok(extract_text(&res, &body, self, 0))
    }

    /// _scan_of: the largest embedded image of a page as a data: URL, or None below `min_pixels`.
    pub fn scan_of(&self, k: usize, min_pixels: u64) -> Option<String> {
        let page = &self.pages[k];
        let res = page.get("Resources").map(|r| self.resolve(r))?;
        let xo = res.get("XObject").map(|x| self.resolve(x))?;
        let mut best: Option<(u32, u64)> = None;
        for (_, v) in xo.dict()? {
            let Obj::Ref(n, _) = v else { continue };
            let o = self.object(*n);
            if o.get("Subtype").and_then(|s| s.name()) != Some("Image") {
                continue;
            }
            let w = o.get("Width").map(|x| self.resolve(x)).and_then(|x| x.num()).unwrap_or(0.0) as u64;
            let h = o.get("Height").map(|x| self.resolve(x)).and_then(|x| x.num()).unwrap_or(0.0) as u64;
            if w * h > best.map(|b| b.1).unwrap_or(0) {
                best = Some((*n, w * h));
            }
        }
        let (n, size) = best?;
        if size < min_pixels {
            return None;
        }
        let o = self.object(n);
        let Obj::Stream(d, raw) = &o else { return None };
        let filters = filter_names(&Obj::Stream(d.clone(), vec![]), self);
        let (bytes, media) = if filters.last().map(|f| f == "DCTDecode").unwrap_or(false) {
            let pre = decode_upto(&Obj::Stream(d.clone(), vec![]), raw, self, filters.len() - 1);
            if pre.starts_with(b"\xff\xd8") {
                (pre, "image/jpeg")
            } else {
                return None;
            }
        } else {
            let px = decode(&o, raw, self);
            let w = o.get("Width").and_then(|x| x.num())? as u32;
            let h = o.get("Height").and_then(|x| x.num())? as u32;
            let bpc = o.get("BitsPerComponent").and_then(|x| x.num()).unwrap_or(8.0) as u32;
            let cs = o.get("ColorSpace").map(|c| self.resolve(c)).and_then(|c| c.name().map(String::from)).unwrap_or_default();
            if bpc != 8 {
                return None;
            }
            let img = match cs.as_str() {
                "DeviceRGB" => image::RgbImage::from_raw(w, h, px.get(..(w * h * 3) as usize)?.to_vec()).map(image::DynamicImage::ImageRgb8),
                "DeviceGray" => image::GrayImage::from_raw(w, h, px.get(..(w * h) as usize)?.to_vec()).map(image::DynamicImage::ImageLuma8),
                _ => None,
            }?;
            let mut out = vec![];
            img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png).ok()?;
            (out, "image/png")
        };
        Some(format!("data:{media};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)))
    }
}

impl<'a> Lex<'a> {
    fn value_num(&mut self) -> Option<Obj> {
        match self.tok() {
            Tok::Obj(o) => Some(o),
            _ => None,
        }
    }
}

fn filter_names(s: &Obj, pdf: &Pdf) -> Vec<String> {
    match s.get("Filter").map(|f| pdf.resolve(f)) {
        Some(Obj::Name(n)) => vec![n],
        Some(Obj::Arr(a)) => a.iter().filter_map(|x| x.name().map(String::from)).collect(),
        _ => vec![],
    }
}

fn decode_upto(s: &Obj, raw: &[u8], pdf: &Pdf, upto: usize) -> Vec<u8> {
    let mut data = raw.to_vec();
    for f in filter_names(s, pdf).iter().take(upto) {
        data = match f.as_str() {
            "FlateDecode" | "Fl" => inflate(&data),
            "ASCIIHexDecode" | "AHx" => {
                let mut lx = Lex { b: &data, i: 0 };
                lx.hex()
            }
            _ => data,
        };
    }
    data
}

fn decode(s: &Obj, raw: &[u8], pdf: &Pdf) -> Vec<u8> {
    let n = filter_names(s, pdf).len();
    let mut data = decode_upto(s, raw, pdf, n);
    // PNG predictors after Flate
    if let Some(p) = s.get("DecodeParms").map(|p| pdf.resolve(p)) {
        let pred = p.get("Predictor").and_then(|x| x.num()).unwrap_or(1.0) as u32;
        if pred >= 10 {
            let cols = p.get("Columns").and_then(|x| x.num()).unwrap_or(1.0) as usize;
            let colors = p.get("Colors").and_then(|x| x.num()).unwrap_or(1.0) as usize;
            let bpc = p.get("BitsPerComponent").and_then(|x| x.num()).unwrap_or(8.0) as usize;
            let bpp = (colors * bpc).div_ceil(8).max(1);
            let row = (cols * colors * bpc).div_ceil(8);
            data = unpredict(&data, row, bpp);
        }
    }
    data
}

fn unpredict(data: &[u8], row: usize, bpp: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut prev = vec![0u8; row];
    for chunk in data.chunks(row + 1) {
        if chunk.len() < row + 1 {
            break;
        }
        let t = chunk[0];
        let mut cur = chunk[1..].to_vec();
        for i in 0..row {
            let a = if i >= bpp { cur[i - bpp] as i32 } else { 0 };
            let b = prev[i] as i32;
            let c = if i >= bpp { prev[i - bpp] as i32 } else { 0 };
            let x = cur[i] as i32;
            cur[i] = match t {
                1 => x + a,
                2 => x + b,
                3 => x + (a + b) / 2,
                4 => {
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    x + if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }
                }
                _ => x,
            } as u8;
        }
        out.extend_from_slice(&cur);
        prev = cur;
    }
    out
}

include!("pdf_text.rs");
