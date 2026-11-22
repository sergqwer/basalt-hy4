// pypdf's plain text extraction (pypdf 7: _page._extract_text, _text_extraction.TextExtraction, crlf_space_check,
// get_text_operands, generic._font.Font.from_font_resource, _cmap.get_encoding): the same state machine, encodings,
// ToUnicode maps, width tables (pdf_tables.json, exported from pypdf) and space rules. Not ported: font programs
// (a Type1 font file's own encoding when there is no ToUnicode), right-to-left runs, the layout mode.

struct Tables {
    encodings: HashMap<String, Vec<String>>,
    glyphs: HashMap<String, String>,
    core: HashMap<String, (i64, HashMap<String, f64>)>,
}

fn tables() -> &'static Tables {
    static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let v: serde_json::Value = serde_json::from_str(include_str!("pdf_tables.json")).unwrap();
        let encodings = v["encodings"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, a)| (k.clone(), a.as_array().unwrap().iter().map(|x| x.as_str().unwrap_or("").to_string()).collect()))
            .collect();
        let glyphs = v["glyphs"].as_object().unwrap().iter().map(|(k, x)| (k.clone(), x.as_str().unwrap_or("").to_string())).collect();
        let core = v["core"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, m)| {
                let w = m["widths"].as_object().unwrap().iter().map(|(c, w)| (c.clone(), w.as_f64().unwrap_or(0.0))).collect();
                (k.clone(), (m["flags"].as_i64().unwrap_or(32), w))
            })
            .collect();
        Tables { encodings, glyphs, core }
    })
}

#[derive(Clone)]
enum Enc {
    Dict(HashMap<u32, String>),
    /// "utf-16-be" (Identity and the like)
    Utf16,
}

#[derive(Clone)]
struct Font {
    encoding: Enc,
    char_map: HashMap<String, String>,
    widths: HashMap<String, f64>,
    space_width: f64,
    space_char: String,
}

fn charset(name: &str) -> Option<HashMap<u32, String>> {
    tables().encodings.get(name).map(|v| v.iter().enumerate().map(|(i, c)| (i as u32, c.clone())).collect())
}

fn glyph_name_to_unicode(name: &str) -> Option<String> {
    if let Some(g) = tables().glyphs.get(name) {
        return Some(g.clone());
    }
    let cp = if let Some(r) = name.strip_prefix("uni") { r } else { name.strip_prefix('u')? };
    u32::from_str_radix(cp, 16).ok().and_then(char::from_u32).map(|c| c.to_string())
}

fn latin1(b: &[u8]) -> String {
    b.iter().map(|x| *x as char).collect()
}

fn utf16(b: &[u8]) -> String {
    let u: Vec<u16> = b.chunks(2).map(|c| ((c[0] as u16) << 8) | *c.get(1).unwrap_or(&0) as u16).collect();
    String::from_utf16_lossy(&u)
}

fn unhex(s: &[u8]) -> Option<Vec<u8>> {
    if s.len() % 2 == 1 {
        return None;
    }
    s.chunks(2).map(|p| std::str::from_utf8(p).ok().and_then(|x| u8::from_str_radix(x, 16).ok())).collect()
}

/// _cmap.prepare_cm + process_cm_line: the ToUnicode map (decoded source -> text) and its source codes.
fn parse_to_unicode(cm: &[u8]) -> (HashMap<String, String>, Vec<u32>) {
    let mut s = cm.to_vec();
    while s.first().map(|c| c.is_ascii_whitespace()).unwrap_or(false) {
        s.remove(0);
    }
    while s.last().map(|c| c.is_ascii_whitespace()).unwrap_or(false) {
        s.pop();
    }
    let rep = |s: Vec<u8>, a: &[u8], b: &[u8]| -> Vec<u8> {
        let mut out = vec![];
        let mut i = 0;
        while i < s.len() {
            if s[i..].starts_with(a) {
                out.extend_from_slice(b);
                i += a.len();
            } else {
                out.push(s[i]);
                i += 1;
            }
        }
        out
    };
    for (a, b) in [
        (&b"beginbfchar"[..], &b"\nbeginbfchar\n"[..]),
        (b"endbfchar", b"\nendbfchar\n"),
        (b"beginbfrange", b"\nbeginbfrange\n"),
        (b"endbfrange", b"\nendbfrange\n"),
        (b"<<", b"\n{\n"),
        (b">>", b"\n}\n"),
    ] {
        s = rep(s, a, b);
    }
    let parts: Vec<&[u8]> = s.split(|c| *c == b'<').collect();
    let mut ll: Vec<Vec<u8>> = vec![];
    for p in parts {
        match p.iter().position(|c| *c == b'>') {
            Some(0) => ll.push([&b". "[..], &p[1..]].concat()),
            Some(j) => {
                let content: Vec<u8> = p[..j].iter().copied().filter(|c| *c != b' ').collect();
                ll.push([&content[..], b" ", &p[j + 1..]].concat());
            }
            None => ll.push(p.to_vec()),
        }
    }
    let mut joined = ll.join(&b' ');
    joined = rep(joined, b"[", b" [ ");
    joined = rep(joined, b"]", b" ]\n ");
    joined = rep(joined, b"\r", b"\n");
    let mut map: HashMap<String, String> = HashMap::new();
    let mut ints: Vec<u32> = vec![];
    let mut nbytes: usize = 1;
    let (mut rg, mut ch) = (false, false);
    let mut multi: Option<(u32, u32)> = None;
    let key = |code: u32, n: usize| -> String {
        let b: Vec<u8> = (0..n).rev().map(|k| (code >> (8 * k)) as u8).collect();
        if n == 1 {
            latin1(&b)
        } else {
            utf16(&b)
        }
    };
    for raw in joined.split(|c| *c == b'\n') {
        let line: Vec<u8> = {
            let t: &[u8] = raw;
            let st = t.iter().position(|c| *c != b' ' && *c != b'\t').unwrap_or(t.len());
            let en = t.iter().rposition(|c| *c != b' ' && *c != b'\t').map(|x| x + 1).unwrap_or(st);
            t[st..en.max(st)].iter().map(|c| if *c == b'\t' { b' ' } else { *c }).collect()
        };
        if line.is_empty() || line[0] == b'%' {
            continue;
        }
        let has = |w: &[u8]| line.windows(w.len()).any(|x| x == w);
        if has(b"beginbfrange") {
            rg = true;
        } else if has(b"endbfrange") {
            rg = false;
        } else if has(b"beginbfchar") {
            ch = true;
        } else if has(b"endbfchar") {
            ch = false;
        } else if rg {
            let lst: Vec<&[u8]> = line.split(|c| c.is_ascii_whitespace()).filter(|x| !x.is_empty()).collect();
            if let Some((mut a, _b)) = multi {
                let mut closed = false;
                for sq in &lst {
                    if *sq == b"]" {
                        closed = true;
                        break;
                    }
                    if let Some(d) = unhex(sq) {
                        map.insert(key(a, nbytes), utf16(&d));
                    }
                    ints.push(a);
                    a += 1;
                }
                multi = if closed { None } else { Some((a, _b)) };
                continue;
            }
            if lst.len() < 3 {
                continue;
            }
            let (Ok(a), Ok(b)) = (
                u32::from_str_radix(std::str::from_utf8(lst[0]).unwrap_or("x"), 16),
                u32::from_str_radix(std::str::from_utf8(lst[1]).unwrap_or("x"), 16),
            ) else {
                continue;
            };
            nbytes = (lst[0].len().max(lst[1].len()) + 1) / 2;
            let mut a = a;
            if lst[2] == b"[" {
                let mut closed = false;
                for sq in &lst[3..] {
                    if *sq == b"]" {
                        closed = true;
                        break;
                    }
                    if let Some(d) = unhex(sq) {
                        map.insert(key(a, nbytes), utf16(&d));
                    }
                    ints.push(a);
                    a += 1;
                }
                multi = if closed { None } else { Some((a, b)) };
            } else {
                let Ok(mut c) = u32::from_str_radix(std::str::from_utf8(lst[2]).unwrap_or("x"), 16) else { continue };
                let width = lst[2].len().max(4);
                while a <= b && a - a.min(b) < 70000 {
                    let hex = format!("{c:0width$X}");
                    if let Some(d) = unhex(hex.as_bytes()) {
                        map.insert(key(a, nbytes), utf16(&d));
                    }
                    ints.push(a);
                    a += 1;
                    c += 1;
                }
            }
        } else if ch {
            let lst: Vec<&[u8]> = line.split(|c| *c == b' ').filter(|x| !x.is_empty()).collect();
            if lst.is_empty() {
                continue;
            }
            nbytes = lst[0].len() / 2;
            let mut i = 0;
            while i + 1 < lst.len() {
                let (src, dst) = (lst[i], lst[i + 1]);
                let to = if dst == b"." {
                    String::new()
                } else {
                    match unhex(dst) {
                        Some(d) if dst.len() < 4 => latin1(&d),
                        Some(d) => utf16(&d),
                        None => String::new(),
                    }
                };
                if let Some(sb) = unhex(src) {
                    let k = if nbytes == 1 { latin1(&sb) } else { utf16(&sb) };
                    map.insert(k, to);
                }
                if let Ok(code) = u32::from_str_radix(std::str::from_utf8(src).unwrap_or("x"), 16) {
                    ints.push(code);
                }
                i += 2;
            }
        }
    }
    (map, ints)
}

impl Font {
    fn unknown() -> Font {
        Font {
            encoding: Enc::Dict((0..256).map(|i| (i, "\u{fffd}".to_string())).collect()),
            char_map: HashMap::new(),
            widths: [("default".to_string(), 500.0)].into_iter().collect(),
            space_width: 250.0,
            space_char: " ".into(),
        }
    }

    /// the extractor's initial font: WinAnsi, default widths
    fn initial() -> Font {
        Font {
            encoding: Enc::Dict(charset("WinAnsiEncoding").unwrap()),
            char_map: HashMap::new(),
            widths: [("default".to_string(), 500.0)].into_iter().collect(),
            space_width: 250.0,
            space_char: " ".into(),
        }
    }

    fn parse_encoding(f: &Obj, pdf: &Pdf) -> Enc {
        let base = f.get("BaseFont").and_then(|b| b.name()).unwrap_or("");
        let Some(e) = f.get("Encoding") else {
            if let Some(c) = charset(base) {
                return Enc::Dict(c);
            }
            return Enc::Dict(charset("StandardEncoding").unwrap());
        };
        let e = pdf.resolve(e);
        let mut enc = match &e {
            Obj::Name(n) => {
                if let Some(c) = charset(n) {
                    Enc::Dict(c)
                } else if n.contains("Identity") || n.contains("UTF16") || n.contains("-UCS2-") {
                    Enc::Utf16
                } else if ["GB-EUC-H", "GB-EUC-V", "GBpc-EUC-H", "GBpc-EUC-V", "GBK-EUC-H", "GBK-EUC-V", "GBK2K-H", "GBK2K-V",
                    "ETen-B5-H", "ETen-B5-V", "ETenms-B5-H", "ETenms-B5-V", "90ms-RKSJ-H", "90ms-RKSJ-V"]
                    .contains(&n.as_str())
                {
                    Enc::Utf16
                } else if f.get("Subtype").and_then(|s| s.name()) == Some("Type0") {
                    Enc::Utf16
                } else {
                    Enc::Dict(charset("StandardEncoding").unwrap())
                }
            }
            Obj::Dict(_) if e.get("BaseEncoding").is_some() => {
                Enc::Dict(e.get("BaseEncoding").and_then(|b| b.name()).and_then(charset).unwrap_or_else(|| charset("StandardEncoding").unwrap()))
            }
            _ => Enc::Dict(charset("StandardEncoding").unwrap()),
        };
        if let (Enc::Dict(m), Some(Obj::Arr(diff))) = (&mut enc, e.get("Differences").map(|d| pdf.resolve(d))) {
            let mut x: u32 = 0;
            for o in diff {
                match o {
                    Obj::Num(n) => x = n as u32,
                    Obj::Name(g) => {
                        if x < 256 {
                            m.insert(x, glyph_name_to_unicode(&g).unwrap_or(format!("/{g}")));
                        }
                        x += 1;
                    }
                    _ => {}
                }
            }
        }
        enc
    }

    fn load(f: &Obj, pdf: &Pdf) -> Font {
        let name = f.get("BaseFont").and_then(|b| b.name()).unwrap_or("Unknown").to_string();
        let sub = f.get("Subtype").and_then(|s| s.name()).unwrap_or("Unknown").to_string();
        let mut encoding = Font::parse_encoding(f, pdf);
        let (char_map, ints) = match f.get("ToUnicode") {
            Some(Obj::Ref(n, _)) => match pdf.object(*n) {
                Obj::Stream(..) => parse_to_unicode(&pdf.stream_data(*n)),
                _ => parse_to_unicode(b"beginbfrange\n<0000> <0001> <0000>\nendbfrange"),
            },
            Some(_) => parse_to_unicode(b"beginbfrange\n<0000> <0001> <0000>\nendbfrange"),
            None => (HashMap::new(), vec![]),
        };
        if let Enc::Dict(m) = &mut encoding {
            for x in &ints {
                if *x <= 255 {
                    m.insert(*x, char::from_u32(*x).unwrap().to_string());
                }
            }
        }
        let mut widths: HashMap<String, f64> = HashMap::new();
        let mut flags: i64 = 32;
        let mut have_desc = false;
        if ["Type1", "MMType1", "TrueType", "Type3"].contains(&sub.as_str()) {
            if let Some(w) = f.get("Widths").map(|w| pdf.resolve(w)) {
                if let Obj::Arr(a) = w {
                    let first = f.get("FirstChar").map(|x| pdf.resolve(x)).and_then(|x| x.num()).unwrap_or(0.0) as i64;
                    if first >= 0 {
                        for (k, x) in a.iter().enumerate() {
                            let code = first + k as i64;
                            if code > 255 {
                                break;
                            }
                            if let Some(v) = pdf.resolve(x).num() {
                                widths.insert(char::from_u32(code as u32).unwrap().to_string(), v.trunc());
                            }
                        }
                    }
                }
            } else if let Some((fl, cw)) = tables().core.get(&name) {
                flags = *fl;
                have_desc = true;
                if let Enc::Dict(m) = &encoding {
                    for (code, ch) in m {
                        if let Some(w) = cw.get(ch) {
                            widths.insert(char::from_u32(*code).unwrap().to_string(), *w);
                        }
                    }
                }
            }
            if let Some(d) = f.get("FontDescriptor").map(|d| pdf.resolve(d)) {
                if let Some(mw) = d.get("MissingWidth").map(|x| pdf.resolve(x)).and_then(|x| x.num()) {
                    widths.insert("default".into(), mw);
                }
                flags = d.get("Flags").map(|x| pdf.resolve(x)).and_then(|x| x.num()).map(|x| x as i64).unwrap_or(32);
                have_desc = true;
            }
        } else if let Some(Obj::Arr(ds)) = f.get("DescendantFonts").map(|d| pdf.resolve(d)) {
            for d in ds {
                let d = pdf.resolve(&d);
                if let Some(Obj::Arr(w)) = d.get("W").map(|w| pdf.resolve(w)) {
                    let mut i = 0;
                    while i < w.len() {
                        let Some(start) = pdf.resolve(&w[i]).num() else {
                            i += 1;
                            continue;
                        };
                        let next = w.get(i + 1).map(|x| pdf.resolve(x));
                        match next {
                            Some(Obj::Arr(list)) => {
                                for (k, x) in list.iter().enumerate() {
                                    if let (Some(c), Some(v)) = (char::from_u32(start as u32 + k as u32), pdf.resolve(x).num()) {
                                        widths.insert(c.to_string(), v);
                                    }
                                }
                                i += 2;
                            }
                            Some(Obj::Num(stop)) if w.get(i + 2).map(|x| pdf.resolve(x).num().is_some()).unwrap_or(false) => {
                                let v = pdf.resolve(&w[i + 2]).num().unwrap();
                                for c in (start as u32)..=(stop as u32).min(start as u32 + 65536) {
                                    if let Some(c) = char::from_u32(c) {
                                        widths.insert(c.to_string(), v);
                                    }
                                }
                                i += 3;
                            }
                            _ => i += 1,
                        }
                    }
                }
                if let Some(dw) = d.get("DW").map(|x| pdf.resolve(x)).and_then(|x| x.num()) {
                    widths.insert("default".into(), dw);
                }
                let fd = d.get("FontDescriptor").map(|x| pdf.resolve(x));
                flags = fd.and_then(|fd| fd.get("Flags").map(|x| pdf.resolve(x)).and_then(|x| x.num())).map(|x| x as i64).unwrap_or(32);
                have_desc = true;
            }
        }
        let _ = have_desc;
        // _get_space_char: the map's source for a space, else the encoding's code for one
        let mut space_char = " ".to_string();
        let mut found = false;
        // dict order: the map in ToUnicode order is not kept here; any source for a space serves
        let mut cands: Vec<&String> = char_map.iter().filter(|(_, v)| v.as_str() == " ").map(|(k, _)| k).collect();
        cands.sort();
        if let Some(k) = cands.first() {
            space_char = (*k).clone();
            found = true;
        }
        if !found {
            if let Enc::Dict(m) = &encoding {
                let mut codes: Vec<u32> = m.iter().filter(|(_, v)| v.as_str() == " ").map(|(k, _)| *k).collect();
                codes.sort();
                if let Some(c) = codes.first() {
                    space_char = char::from_u32(*c).unwrap().to_string();
                }
            }
        }
        let fixed = flags & 1 == 1;
        if widths.get("default").copied().unwrap_or(0.0) == 0.0 {
            if widths.is_empty() {
                widths.insert("default".into(), 500.0);
            } else if widths.get(&space_char).copied().unwrap_or(0.0) != 0.0 {
                let sp = widths[&space_char];
                widths.insert("default".into(), if fixed { sp } else { (2.0 * sp).trunc() });
            } else {
                let valid: Vec<f64> = widths.values().copied().filter(|w| *w > 0.0).collect();
                let d = if valid.is_empty() { 500.0 } else { (valid.iter().sum::<f64>() / valid.len() as f64).floor() };
                widths.insert("default".into(), d);
            }
        }
        let sp = widths.get(&space_char).copied().unwrap_or(0.0);
        let mut space_width = if sp != 0.0 {
            sp
        } else if fixed {
            widths["default"]
        } else {
            (widths["default"] / 2.0).floor()
        };
        // _extract_text: a font without a space width gets the call's space_width (200)
        if widths.get(&space_char).copied().unwrap_or(0.0) == 0.0 {
            space_width = 200.0;
        }
        Font { encoding, char_map, widths, space_width, space_char }
    }

    fn width(&self, c: &str) -> f64 {
        self.widths.get(c).copied().unwrap_or_else(|| self.widths.get("default").copied().unwrap_or(500.0))
    }
}

type M = [f64; 6];

fn mult(m: &M, n: &M) -> M {
    [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]
}

const ID: M = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn orient(m: &M) -> i32 {
    if m[3] > 1e-6 {
        0
    } else if m[3] < -1e-6 {
        180
    } else if m[1] > 0.0 {
        90
    } else {
        270
    }
}

struct Ex<'a> {
    cm: M,
    tm: M,
    stack: Vec<(M, Option<String>, f64, f64, f64)>,
    cm_prev: M,
    tm_prev: M,
    memo_cm: M,
    memo_tm: M,
    space_width: f64,
    str_widths: f64,
    str_height: f64,
    tl: f64,
    font_size: f64,
    line_span: Option<(usize, f64, f64)>,
    text: String,
    output: String,
    font: Option<String>,
    fonts: &'a HashMap<String, Font>,
    initial: Font,
    unknown: Font,
}

fn last_char(s: &str) -> &str {
    match s.char_indices().last() {
        Some((i, _)) => &s[i..],
        None => "",
    }
}

impl<'a> Ex<'a> {
    fn font(&self) -> &Font {
        match &self.font {
            None => &self.initial,
            Some(n) => self.fonts.get(n).unwrap_or(&self.unknown),
        }
    }

    fn flush(&mut self) {
        self.output.push_str(&self.text);
        self.text.clear();
        self.memo_cm = self.cm;
        self.memo_tm = self.tm;
    }

    fn post(&mut self, str_widths: f64) {
        let was_empty = self.text.is_empty();
        let spacewidth = self.font_size * self.space_width / 1000.0;
        let m_prev = mult(&self.tm_prev, &self.cm_prev);
        let m = mult(&self.tm, &self.cm);
        let o = orient(&m);
        let dx = m[4] - m_prev[4];
        let dy = m[5] - m_prev[5];
        let scale_prev_x = (m_prev[0] * m_prev[0] + m_prev[1] * m_prev[1]).sqrt();
        let scale_prev_y = (m_prev[2] * m_prev[2] + m_prev[3] * m_prev[3]).sqrt();
        let scale_y = (m[2] * m[2] + m[3] * m[3]).sqrt();
        let (moved_h, moved_w) = if o == 0 || o == 180 { (dy, dx) } else { (dx, dy) };
        let axis_index = if o == 0 || o == 180 { 5 } else { 4 };
        let axis = m[axis_index];
        let distance = match self.line_span {
            Some((ai, lo, hi)) if ai == axis_index => {
                if lo <= axis && axis <= hi {
                    0.0
                } else {
                    (axis - lo).abs().min((axis - hi).abs())
                }
            }
            _ => moved_h.abs(),
        };
        let both = format!("{}{}", self.output, self.text);
        let last = last_char(&both).to_string();
        let threshold = 0.8 * (self.str_height * scale_prev_y).min(self.font_size * scale_y);
        if distance > threshold {
            if !(last.is_empty() || last == "\n") {
                self.output.push_str(&self.text);
                self.output.push('\n');
                self.text.clear();
            }
            self.line_span = Some((axis_index, axis, axis));
        } else {
            let other = moved_h.abs() > threshold;
            if (other || moved_w >= (spacewidth + str_widths) * scale_prev_x) && !(last.is_empty() || last == " ") {
                self.text.push(' ');
            }
            self.line_span = match self.line_span {
                Some((ai, lo, hi)) if !(last.is_empty() || last == "\n") && ai == axis_index => Some((ai, lo.min(axis), hi.max(axis))),
                _ => Some((axis_index, axis, axis)),
            };
        }
        self.tm_prev = self.tm;
        self.cm_prev = self.cm;
        if was_empty || self.text.is_empty() {
            self.memo_cm = self.cm;
            self.memo_tm = self.tm;
        }
    }

    fn take_widths(&mut self) -> f64 {
        let w = self.str_widths / 1000.0;
        self.str_widths = 0.0;
        w
    }

    /// Tj with a string operand (bytes) or the literal " " TJ inserts
    fn tj(&mut self, raw: Option<&[u8]>, literal: Option<&str>) {
        let font = self.font().clone();
        let mut widths = 0.0;
        let text: String;
        let mut chars_for_width: Vec<String> = vec![];
        match (raw, literal) {
            (_, Some(l)) => {
                text = l.to_string();
                self.text.push_str(&text);
                for c in text.chars() {
                    let c = c.to_string();
                    widths += if c == font.space_char { font.space_width } else { font.width(&c) };
                }
                self.str_widths += widths * self.font_size;
                self.str_height = self.font_size;
                return;
            }
            (Some(b), None) => {
                match &font.encoding {
                    Enc::Utf16 => {
                        text = utf16(b);
                    }
                    Enc::Dict(m) => {
                        text = b.iter().map(|x| m.get(&(*x as u32)).cloned().unwrap_or_else(|| (*x as char).to_string())).collect();
                        chars_for_width = b.iter().map(|x| (*x as char).to_string()).collect();
                    }
                }
            }
            _ => return,
        }
        let wchars: Vec<String> = if chars_for_width.is_empty() { text.chars().map(|c| c.to_string()).collect() } else { chars_for_width };
        for c in &wchars {
            widths += if *c == font.space_char { font.space_width } else { font.width(c) };
        }
        for c in text.chars() {
            let k = c.to_string();
            match font.char_map.get(&k) {
                Some(x) => self.text.push_str(x),
                None => self.text.push(c),
            }
        }
        self.str_widths += widths * self.font_size;
        self.str_height = self.font_size;
    }
}

fn nums(ops: &[Obj], k: usize, default: f64) -> f64 {
    ops.get(k).and_then(|o| o.num()).unwrap_or(default)
}

fn font_map(res: &Obj, pdf: &Pdf) -> HashMap<String, Font> {
    let mut fmap = HashMap::new();
    if let Some(d) = res.get("Font").map(|f| pdf.resolve(f)).as_ref().and_then(|f| f.dict().cloned()) {
        for (name, f) in d {
            let fo = pdf.resolve(&f);
            if fo.dict().is_some() {
                fmap.insert(name, Font::load(&fo, pdf));
            }
        }
    }
    fmap
}

/// PageObject._extract_text for a page or a form XObject.
fn extract_text(res: &Obj, body: &[u8], pdf: &Pdf, depth: usize) -> String {
    if res.dict().map(|d| d.is_empty()).unwrap_or(true) {
        return String::new();
    }
    let fonts = font_map(res, pdf);
    let mut ex = Ex {
        cm: ID,
        tm: ID,
        stack: vec![],
        cm_prev: ID,
        tm_prev: ID,
        memo_cm: ID,
        memo_tm: ID,
        space_width: 500.0,
        str_widths: 0.0,
        str_height: 0.0,
        tl: 0.0,
        font_size: 12.0,
        line_span: None,
        text: String::new(),
        output: String::new(),
        font: None,
        fonts: &fonts,
        initial: Font::initial(),
        unknown: Font::unknown(),
    };
    let mut lx = Lex { b: body, i: 0 };
    let mut ops: Vec<Obj> = vec![];
    loop {
        let t = lx.tok();
        let op = match t {
            Tok::Eof => break,
            Tok::Obj(o) => {
                ops.push(o);
                continue;
            }
            Tok::ArrEnd | Tok::DictEnd => continue,
            Tok::Op(op) => op,
        };
        match op.as_str() {
            "BT" => {
                ex.tm = ID;
                ex.flush();
            }
            "ET" => ex.flush(),
            "q" => ex.stack.push((ex.cm, ex.font.clone(), ex.font_size, 0.0, ex.tl)),
            "Q" => match ex.stack.pop() {
                Some((cm, f, fs, _, tl)) => {
                    ex.cm = cm;
                    ex.font = f;
                    ex.font_size = fs;
                    ex.tl = tl;
                }
                None => ex.cm = ID,
            },
            "cm" => {
                ex.output.push_str(&ex.text);
                ex.text.clear();
                let m: Option<Vec<f64>> = ops.iter().take(6).map(|o| o.num()).collect();
                ex.cm = match m {
                    Some(v) if v.len() == 6 => mult(&[v[0], v[1], v[2], v[3], v[4], v[5]], &ex.cm),
                    _ => ID,
                };
                ex.memo_cm = ex.cm;
                ex.memo_tm = ex.tm;
            }
            "TL" => ex.tl = nums(&ops, 0, 0.0),
            "Tf" => {
                if !ex.text.is_empty() {
                    ex.output.push_str(&ex.text);
                }
                ex.text.clear();
                ex.memo_cm = ex.cm;
                ex.memo_tm = ex.tm;
                ex.font = Some(ops.first().and_then(|o| o.name()).unwrap_or("\0missing").to_string());
                ex.space_width = ex.font().space_width / 2.0;
                if let Some(s) = ops.get(1).and_then(|o| o.num()) {
                    ex.font_size = s;
                }
            }
            "Td" | "TD" => {
                if op == "TD" {
                    if let Some(ty) = ops.get(1).and_then(|o| o.num()) {
                        ex.tl = -ty;
                    }
                }
                let (tx, ty) = (nums(&ops, 0, 0.0), nums(&ops, 1, 0.0));
                ex.tm[4] += tx * ex.tm[0] + ty * ex.tm[2];
                ex.tm[5] += tx * ex.tm[1] + ty * ex.tm[3];
                let w = ex.take_widths();
                ex.post(w);
            }
            "Tm" => {
                let m: Option<Vec<f64>> = ops.iter().take(6).map(|o| o.num()).collect();
                ex.tm = match m {
                    Some(v) if v.len() == 6 => [v[0], v[1], v[2], v[3], v[4], v[5]],
                    _ => ID,
                };
                let w = ex.take_widths();
                ex.post(w);
            }
            "T*" | "'" | "\"" => {
                if op == "\"" && ops.len() < 3 {
                    ops.clear();
                    continue;
                }
                ex.tm[4] -= ex.tl * ex.tm[2];
                ex.tm[5] -= ex.tl * ex.tm[3];
                let w = ex.take_widths();
                ex.post(w);
                if op != "T*" {
                    let s = if op == "'" { ops.first() } else { ops.get(2) };
                    if let Some(Obj::Str(b)) = s {
                        let b = b.clone();
                        ex.tj(Some(&b), None);
                        ex.post(0.0);
                    }
                }
            }
            "Tj" => {
                if let Some(Obj::Str(b)) = ops.first() {
                    let b = b.clone();
                    ex.tj(Some(&b), None);
                    ex.post(0.0);
                }
            }
            "TJ" => {
                let confirm = ex.space_width * 0.95;
                if let Some(Obj::Arr(a)) = ops.first() {
                    for x in a.clone() {
                        match x {
                            Obj::Str(b) => {
                                ex.tj(Some(&b), None);
                                ex.post(0.0);
                            }
                            Obj::Num(n) => {
                                if n.abs() >= confirm && !ex.text.is_empty() && !ex.text.ends_with(' ') {
                                    ex.tj(None, Some(" "));
                                    ex.post(0.0);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "Do" => {
                ex.output.push_str(&ex.text);
                if !ex.output.is_empty() && !ex.output.ends_with('\n') {
                    ex.output.push('\n');
                }
                if depth < 8 {
                    if let Some(name) = ops.first().and_then(|o| o.name()) {
                        let xo = res.get("XObject").map(|x| pdf.resolve(x));
                        if let Some(Obj::Ref(n, _)) = xo.as_ref().and_then(|x| x.get(name).cloned()) {
                            let o = pdf.object(n);
                            if o.get("Subtype").and_then(|s| s.name()) == Some("Form") {
                                let inner = pdf.stream_data(n);
                                let r2 = o.get("Resources").map(|r| pdf.resolve(r)).unwrap_or(Obj::Null);
                                ex.output.push_str(&extract_text(&r2, &inner, pdf, depth + 1));
                            }
                        }
                    }
                }
                ex.text.clear();
                ex.memo_cm = ex.cm;
                ex.memo_tm = ex.tm;
            }
            "BI" => {
                if let Some(e) = find_all(&body[lx.i..], b"EI").first() {
                    lx.i += e + 2;
                }
            }
            _ => {}
        }
        ops.clear();
    }
    ex.output.push_str(&ex.text);
    ex.output
}
