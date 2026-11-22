//! api/documents.py: Anthropic `document` blocks, Chat Completions `file` parts and Responses `input_file` parts become
//! the text and image items every API's content is made of.

use crate::error::{AResult, ApiError};
use crate::image_source::{load_image, ImageSources, IMAGE};
use regex::Regex;
use serde_json::{json, Map, Value};
use serve_engine::util::stdout::out;
use serve_text::pyjson::{py_repr, py_str_repr};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::io::Read;
use std::sync::{LazyLock, Mutex};

pub const DOCUMENT_PARTS: [&str; 3] = ["document", "file", "input_file"];
pub const MAX_DOCUMENT_BYTES: usize = 64 << 20;
pub const MAX_PAGES: usize = 600;
pub const MIN_SCAN_PIXELS: u64 = 64 * 64;
pub const SPARSE_TEXT: usize = 200;
pub const MIN_PICTURE_PIXELS: u64 = 512 * 512;
const CACHE_ENTRIES: usize = 32;
pub const MAX_PART_BYTES: u64 = 64 << 20;
const TEXT_TYPES: [&str; 7] = [
    "application/json", "application/xml", "application/x-yaml", "application/yaml", "application/csv",
    "application/javascript", "application/x-sh",
];

fn office_type(media: &str) -> Option<&'static str> {
    match media {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Some("Word (.docx)"),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => Some("PowerPoint (.pptx)"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Some("Excel (.xlsx)"),
        _ => None,
    }
}

type Items = Vec<Map<String, Value>>;

static CACHE: LazyLock<Mutex<indexmap::IndexMap<String, (usize, Option<String>, Items)>>> =
    LazyLock::new(|| Mutex::new(indexmap::IndexMap::new()));
static PARSES: Mutex<u32> = Mutex::new(0);
static PARSES_CV: std::sync::Condvar = std::sync::Condvar::new();

thread_local! {
    static REQUEST: RefCell<Option<ImageSources>> = const { RefCell::new(None) };
}

fn verr(m: impl Into<String>) -> ApiError {
    ApiError::value(m)
}

pub fn is_document(part: &Value) -> bool {
    part.get("type").and_then(|t| t.as_str()).map(|t| DOCUMENT_PARTS.contains(&t)).unwrap_or(false) && part.is_object()
}

pub fn has_document(content: &Value) -> bool {
    content.as_array().map(|a| a.iter().any(is_document)).unwrap_or(false)
}

/// expand_documents: `content` with each document part replaced by its text and image items.
pub fn expand_documents(content: &[Value]) -> AResult<Vec<Value>> {
    let mut outv = vec![];
    for (k, part) in content.iter().enumerate() {
        if !is_document(part) {
            outv.push(part.clone());
            continue;
        }
        let mut items = document_items(part.as_object().unwrap())?;
        if let Some(first) = items.first_mut() {
            if first["type"] == "text" && k > 0 {
                let t = format!("\n\n{}", first["text"].as_str().unwrap_or(""));
                first.insert("text".into(), Value::from(t));
            }
        }
        if let Some(last) = items.last_mut() {
            if last["type"] == "text" && k < content.len() - 1 {
                let t = format!("{}\n\n", last["text"].as_str().unwrap_or(""));
                last.insert("text".into(), Value::from(t));
            }
        }
        outv.extend(items.into_iter().map(Value::Object));
    }
    Ok(outv)
}

/// The scope of `sources(allowed)`: documents named by URL or path are read under `allowed` until it drops.
pub struct SourcesGuard(());

impl Drop for SourcesGuard {
    fn drop(&mut self) {
        REQUEST.with(|r| *r.borrow_mut() = None);
    }
}

pub fn sources(allowed: ImageSources) -> SourcesGuard {
    REQUEST.with(|r| *r.borrow_mut() = Some(allowed));
    SourcesGuard(())
}

fn fetch(source: &str, name: &str) -> AResult<Vec<u8>> {
    let allowed = REQUEST.with(|r| r.borrow().clone());
    match load_image(&Value::from(source), allowed.as_ref(), IMAGE) {
        Ok(d) => Ok(d),
        Err(e) => {
            if let Some(l) = &e.log_line {
                out(&format!("[server] error: document {}: {l}", quoted(name)));
            }
            Err(verr(format!("document {}: {}", py_str_repr(name), e.message)))
        }
    }
}

fn given(v: Option<&Value>) -> String {
    v.and_then(|x| x.as_str()).unwrap_or("").to_string()
}

static SCHEME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-zA-Z][a-zA-Z0-9+.-]+://").unwrap());

fn url_name(source: &str) -> String {
    let mut path = source.to_string();
    if SCHEME.is_match(source) {
        // urlsplit(...).path, percent-decoded
        let rest = &source[source.find("://").unwrap() + 3..];
        let start = rest.find('/').unwrap_or(rest.len());
        let p = &rest[start..];
        let end = p.find(['?', '#']).unwrap_or(p.len());
        path = percent_decode(p[..end].as_bytes());
    }
    let trimmed = path.trim_end_matches(['\\', '/']);
    let last = trimmed.rsplit(['\\', '/']).next().unwrap_or("");
    if last.is_empty() {
        "document".into()
    } else {
        last.to_string()
    }
}

fn percent_decode_bytes(b: &[u8]) -> Vec<u8> {
    let mut outv = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() + 1 && i + 2 <= b.len() - 1 {
            if let Some(v) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|x| u8::from_str_radix(x, 16).ok()) {
                outv.push(v);
                i += 3;
                continue;
            }
        }
        outv.push(b[i]);
        i += 1;
    }
    outv
}

fn percent_decode(b: &[u8]) -> String {
    String::from_utf8_lossy(&percent_decode_bytes(b)).into_owned()
}

// -------------------------------------------------------------------------------------------- the part shapes
fn document_items(part: &Map<String, Value>) -> AResult<Items> {
    let kind = part["type"].as_str().unwrap().to_string();
    if kind == "document" {
        return anthropic(part);
    }
    let spec = if kind == "file" { part.get("file").cloned() } else { Some(Value::Object(part.clone())) };
    let Some(Value::Object(spec)) = spec else {
        return Err(verr(format!("a {kind} part needs a \"file\" object with file_data")));
    };
    let mut name = given(spec.get("filename"));
    if name.is_empty() {
        name = "document".into();
    }
    if spec.get("file_data").map(crate::py::truthy).unwrap_or(false) {
        let (media, data) = data_url(&spec["file_data"], &kind)?;
        return read(&data, &media, &name, None, spec.get("pages"));
    }
    if spec.get("file_id").map(crate::py::truthy).unwrap_or(false) {
        return Err(verr(format!(
            "{kind}: this server keeps no files, so a file_id cannot be read; send the file's bytes as file_data (a data: URL)"
        )));
    }
    if let Some(u) = spec.get("file_url").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
        let mut name = given(spec.get("filename"));
        if name.is_empty() {
            name = url_name(u);
        }
        let data = fetch(u, &name)?;
        return read(&data, "", &name, None, spec.get("pages"));
    }
    Err(verr(format!("a {kind} part needs file_data (a data: URL with the file's bytes) or file_url")))
}

fn anthropic(part: &Map<String, Value>) -> AResult<Items> {
    let Some(Value::Object(src)) = part.get("source") else {
        return Err(verr("a document block needs a source object"));
    };
    let mut name = given(part.get("title"));
    if name.is_empty() {
        name = "document".into();
    }
    let context = part.get("context").and_then(|v| v.as_str()).map(String::from);
    let ctx = context.as_deref();
    let kind = src.get("type").cloned().unwrap_or(Value::Null);
    match kind.as_str() {
        Some("base64") => {
            let data = b64(src.get("data"), "document source.data")?;
            let media = src.get("media_type").filter(|v| crate::py::truthy(v)).map(serve_text::pyjson::py_str).unwrap_or_default();
            read(&data, &media, &name, ctx, part.get("pages"))
        }
        Some("text") => {
            let Some(d) = src.get("data").and_then(|v| v.as_str()) else {
                return Err(verr("a text document's source.data must be a string"));
            };
            Ok(vec![text_item(&wrap(&name, None, ctx, &clean(d), ""))])
        }
        Some("content") => match src.get("content") {
            Some(Value::String(body)) => Ok(vec![text_item(&wrap(&name, None, ctx, body, ""))]),
            Some(Value::Array(body)) => {
                let w = wrap(&name, None, ctx, "\0", "");
                let (head, tail) = w.split_once('\0').unwrap();
                let mut items = vec![text_item(head)];
                for block in body {
                    match block.get("type").and_then(|t| t.as_str()) {
                        Some("text") if block.is_object() => {
                            items.push(text_item(&block.get("text").filter(|v| crate::py::truthy(v)).map(serve_text::pyjson::py_str).unwrap_or_default()))
                        }
                        Some("image") if block.is_object() => items.push(block.as_object().unwrap().clone()),
                        _ => return Err(verr("a content document holds text and image blocks only")),
                    }
                }
                items.push(text_item(tail));
                Ok(items)
            }
            _ => Err(verr("a content document's source.content must be a string or a list of blocks")),
        },
        Some("file") => Err(verr(
            "document: this server keeps no files, so a file_id cannot be read; send the file's bytes as a base64 source",
        )),
        Some("url") => {
            let Some(u) = src.get("url").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) else {
                return Err(verr("a url document needs source.url"));
            };
            let mut name = given(part.get("title"));
            if name.is_empty() {
                name = url_name(u);
            }
            let data = fetch(u, &name)?;
            read(&data, "", &name, ctx, part.get("pages"))
        }
        _ => Err(verr(format!(
            "document source type {} is not supported (base64, text and content are)",
            py_repr(&kind)
        ))),
    }
}

static DATA_URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^data:([^;,]*)(;[^,]*)?,").unwrap());

fn data_url(value: &Value, kind: &str) -> AResult<(String, Vec<u8>)> {
    let Some(v) = value.as_str() else { return Err(verr(format!("{kind}: file_data must be a string"))) };
    let Some(m) = DATA_URL.captures(v) else {
        return Ok((String::new(), b64(Some(value), &format!("{kind} file_data"))?));
    };
    let body = &v[m.get(0).unwrap().end()..];
    let media = m.get(1).unwrap().as_str().to_string();
    if !m.get(2).map(|g| g.as_str()).unwrap_or("").contains(";base64") {
        return Ok((media, percent_decode_bytes(body.as_bytes())));
    }
    Ok((media, b64(Some(&Value::from(body)), &format!("{kind} file_data"))?))
}

fn b64(value: Option<&Value>, what: &str) -> AResult<Vec<u8>> {
    let Some(v) = value.and_then(|x| x.as_str()) else { return Err(verr(format!("{what} must be a base64 string"))) };
    if v.chars().count() > MAX_DOCUMENT_BYTES * 4 / 3 + 4 {
        return Err(verr(format!("{what}: a document is limited to {} MiB", MAX_DOCUMENT_BYTES >> 20)));
    }
    crate::image_source::b64decode_validate(v).ok_or_else(|| verr(format!("{what} is not valid base64")))
}

// -------------------------------------------------------------------------------------------- reading
fn read(data: &[u8], media: &str, name: &str, context: Option<&str>, want: Option<&Value>) -> AResult<Items> {
    let media = crate::py::strip(media.split(';').next().unwrap_or("")).to_lowercase();
    let head = &data[..data.len().min(1024)];
    let lead = head.iter().position(|b| !b" \t\n\r\x0b\x0c".contains(b)).unwrap_or(head.len());
    let pdf = head[lead..].starts_with(b"%PDF-");
    let generic = media.is_empty() || media == "application/octet-stream";
    let want = want.filter(|v| !v.is_null());
    if media == "application/pdf" || (pdf && generic) {
        if !pdf {
            return Err(verr(format!("document {} is not a PDF (it does not start with %PDF-)", py_str_repr(name))));
        }
        let digest: String = Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect();
        let key = format!("pdf:{digest}:{}", want.map(py_repr).unwrap_or_else(|| "None".into()));
        let hit = {
            let mut c = CACHE.lock().unwrap();
            let h = c.get(&key).cloned();
            if h.is_some() {
                let v = c.shift_remove(&key).unwrap();
                c.insert(key.clone(), v);
            }
            h
        };
        let hit = match hit {
            Some(h) => h,
            None => {
                {
                    let mut n = PARSES.lock().unwrap();
                    while *n >= 2 {
                        n = PARSES_CV.wait(n).unwrap();
                    }
                    *n += 1;
                }
                let r = read_pdf(data, name, want);
                *PARSES.lock().unwrap() -= 1;
                PARSES_CV.notify_one();
                let h = r?;
                let mut c = CACHE.lock().unwrap();
                c.insert(key, h.clone());
                while c.len() > CACHE_ENTRIES {
                    c.shift_remove_index(0);
                }
                h
            }
        };
        let (pages, readv, items) = hit;
        return Ok(framed(name, pages, readv.as_deref(), context, items));
    }
    if want.is_some() {
        return Err(verr(format!("document {}: pages are named in PDF documents only", py_str_repr(name))));
    }
    let office = if office_type(&media).is_some() || generic { office_kind(data) } else { None };
    if let Some(kind) = office {
        let (body, count) = read_office(data, name, kind)?;
        return Ok(vec![text_item(&wrap(name, None, context, &body, &count))]);
    }
    if let Some(t) = office_type(&media) {
        return Err(verr(format!("document {} is not a {t} file", py_str_repr(name))));
    }
    if media.starts_with("text/") || TEXT_TYPES.contains(&media.as_str()) || (generic && !pdf) {
        let d = data.strip_prefix(b"\xef\xbb\xbf").unwrap_or(data);
        let Ok(text) = std::str::from_utf8(d) else {
            return Err(verr(format!("document {} is not UTF-8 text", py_str_repr(name))));
        };
        return Ok(vec![text_item(&wrap(name, None, context, &clean(text), ""))]);
    }
    Err(verr(format!(
        "document {}: media type {} is not read (PDF, Word, PowerPoint, Excel and text documents are)",
        py_str_repr(name),
        py_str_repr(&media)
    )))
}

fn read_pdf(data: &[u8], name: &str, want: Option<&Value>) -> AResult<(usize, Option<String>, Items)> {
    let t0 = std::time::Instant::now();
    let doc = crate::api::pdf::Pdf::open(data).map_err(|e| match e {
        crate::api::pdf::PdfErr::Encrypted => verr(format!("document {} is a password-protected PDF", py_str_repr(name))),
        crate::api::pdf::PdfErr::Bad(m) => verr(format!("document {} could not be read as a PDF: {m}", py_str_repr(name))),
    })?;
    let n = doc.page_count();
    let (chosen, readv) = page_list(want, n, name)?;
    if chosen.len() > MAX_PAGES {
        return Err(verr(format!(
            "document {}: {} pages; at most {MAX_PAGES} are read per document (name the ones to read with \"pages\")",
            py_str_repr(name),
            chosen.len()
        )));
    }
    let mut items: Items = vec![];
    let mut text: Vec<String> = vec![];
    let (mut scans, mut chars) = (0, 0usize);
    for k in chosen {
        let body = doc
            .page_text(k)
            .map(|t| clean(&t))
            .map_err(|e| verr(format!("document {}: page {} could not be read: {e}", py_str_repr(name), k + 1)))?;
        let len = body.chars().count();
        chars += len;
        let image = if len >= SPARSE_TEXT {
            None
        } else {
            doc.scan_of(k, if body.is_empty() { MIN_SCAN_PIXELS } else { MIN_PICTURE_PIXELS })
        };
        let Some(image) = image else {
            text.push(if !body.is_empty() { format!("[page {}]\n{body}\n", k + 1) } else { format!("[page {}: no text]\n", k + 1) });
            continue;
        };
        scans += 1;
        text.push(if !body.is_empty() {
            format!("[page {}, with its image]\n{body}\n", k + 1)
        } else {
            format!("[page {}: scanned, as an image]\n", k + 1)
        });
        items.push(text_item(&text.join("\n")));
        items.push(json!({"type": "image_url", "image_url": image}).as_object().unwrap().clone());
        text.clear();
    }
    if !text.is_empty() {
        items.push(text_item(&text.join("\n")));
    }
    out(&format!(
        "[server] document: {} {n} pages{}, {scans} as images, {chars} chars of text, {:.2}s",
        quoted(name),
        readv.as_ref().map(|r| format!(", read {r}")).unwrap_or_default(),
        t0.elapsed().as_secs_f64()
    ));
    Ok((n, readv, items))
}

static PIECE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$").unwrap());

fn page_list(want: Option<&Value>, n: usize, name: &str) -> AResult<(Vec<usize>, Option<String>)> {
    let Some(want) = want else { return Ok(((0..n).collect(), None)) };
    let want = if crate::py::is_int(want) { Value::from(serve_text::pyjson::py_str(want)) } else { want.clone() };
    let usage = || verr(format!("document {}: pages must name pages and ranges such as \"1-5,8\", not {}", py_str_repr(name), py_repr(&want)));
    let Some(w) = want.as_str().filter(|s| !crate::py::strip(s).is_empty()) else { return Err(usage()) };
    let mut chosen = std::collections::BTreeSet::new();
    for piece in w.split(',') {
        let Some(m) = PIECE.captures(piece).filter(|_| !piece.contains('\n') || true) else { return Err(usage()) };
        let big = |s: &str| -> u128 { s.parse().unwrap_or(u128::MAX) };
        let first = big(m.get(1).unwrap().as_str());
        let last = m.get(2).map(|g| big(g.as_str())).unwrap_or(first);
        if first < 1 || last < first {
            return Err(usage());
        }
        if last > n as u128 {
            return Err(verr(format!("document {}: page {last} is past its end ({n} pages)", py_str_repr(name))));
        }
        for p in (first - 1)..last {
            chosen.insert(p as usize);
        }
    }
    let pages: Vec<usize> = chosen.into_iter().collect();
    let mut runs = vec![];
    let mut start = pages[0];
    for (i, &a) in pages.iter().enumerate() {
        let b = pages.get(i + 1).copied();
        if b != Some(a + 1) {
            runs.push(if start == a { format!("{}", start + 1) } else { format!("{}-{}", start + 1, a + 1) });
            if let Some(b) = b {
                start = b;
            }
        }
    }
    Ok((pages, Some(runs.join(","))))
}

// -------------------------------------------------------------------------------------------- Office files
const W: &str = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}";
const A: &str = "{http://schemas.openxmlformats.org/drawingml/2006/main}";
const S: &str = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}";
const R: &str = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}";
const PR: &str = "{http://schemas.openxmlformats.org/package/2006/relationships}";

fn office_kind(data: &[u8]) -> Option<&'static str> {
    if !data.starts_with(b"PK\x03\x04") {
        return None;
    }
    let z = zip::ZipArchive::new(std::io::Cursor::new(data)).ok()?;
    let names: Vec<&str> = z.file_names().collect();
    for (part, kind) in [("word/document.xml", "docx"), ("ppt/presentation.xml", "pptx"), ("xl/workbook.xml", "xlsx")] {
        if names.contains(&part) {
            return Some(kind);
        }
    }
    None
}

enum OfficeErr {
    Value(String),
    Other(String),
}

fn read_office(data: &[u8], name: &str, kind: &str) -> AResult<(String, String)> {
    let r = (|| -> Result<(String, String), OfficeErr> {
        let mut z = zip::ZipArchive::new(std::io::Cursor::new(data)).map_err(|e| OfficeErr::Other(e.to_string()))?;
        let mut part = |path: &str| -> Result<crate::xml::Element, OfficeErr> { xml_part(&mut z, path) };
        if kind == "docx" {
            return Ok((docx(&part("word/document.xml")?), String::new()));
        }
        if kind == "pptx" {
            let slides = targets(&mut part, "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "ppt/", "sldIdLst")?;
            let mut bodies = vec![];
            for (k, (_, path)) in slides.iter().enumerate() {
                bodies.push(format!("[slide {}]\n{}", k + 1, paragraphs(&part(path)?, &format!("{A}p"), &format!("{A}t"))));
            }
            return Ok((bodies.join("\n\n"), format!("slides=\"{}\"", slides.len())));
        }
        let sheets = targets(&mut part, "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/", "sheets")?;
        let has_shared = z.file_names().any(|n| n == "xl/sharedStrings.xml");
        let mut part = |path: &str| -> Result<crate::xml::Element, OfficeErr> { xml_part(&mut z, path) };
        let shared: Vec<String> = if has_shared {
            let root = part("xl/sharedStrings.xml")?;
            root.iter().into_iter().filter(|e| e.tag == format!("{S}si")).map(|si| text_of_node(si, &format!("{S}t"))).collect()
        } else {
            vec![]
        };
        let mut bodies = vec![];
        for (title, path) in &sheets {
            bodies.push(format!("[sheet {title}]\n{}", sheet(&part(path)?, &shared)));
        }
        Ok((bodies.join("\n\n"), format!("sheets=\"{}\"", sheets.len())))
    })();
    r.map_err(|e| match e {
        OfficeErr::Value(m) => verr(m),
        OfficeErr::Other(m) => verr(format!("document {} could not be read as a {kind} file: {m}", py_str_repr(name))),
    })
}

fn xml_part<R2: std::io::Read + std::io::Seek>(z: &mut zip::ZipArchive<R2>, path: &str) -> Result<crate::xml::Element, OfficeErr> {
    let f = z.by_name(path).map_err(|_| OfficeErr::Other(format!("\"There is no item named {} in the archive\"", py_str_repr(path))))?;
    let too = || OfficeErr::Value(format!("an Office part is larger than {} MiB unpacked", MAX_PART_BYTES >> 20));
    if f.size() > MAX_PART_BYTES {
        return Err(too());
    }
    let mut raw = vec![];
    f.take(MAX_PART_BYTES + 1).read_to_end(&mut raw).map_err(|e| OfficeErr::Other(e.to_string()))?;
    if raw.len() as u64 > MAX_PART_BYTES {
        return Err(too());
    }
    crate::xml::parse(&raw).map_err(OfficeErr::Other)
}

fn targets(
    part: &mut dyn FnMut(&str) -> Result<crate::xml::Element, OfficeErr>,
    main: &str,
    rels: &str,
    base: &str,
    list_tag: &str,
) -> Result<Vec<(String, String)>, OfficeErr> {
    let rel_root = part(rels)?;
    let mut links: std::collections::HashMap<String, String> = Default::default();
    for r in rel_root.iter().into_iter().filter(|e| e.tag == format!("{PR}Relationship")) {
        links.insert(r.get("Id").unwrap_or("").to_string(), r.get("Target").unwrap_or("").to_string());
    }
    let main_root = part(main)?;
    let suffix = format!("}}{list_tag}");
    let mut outv = vec![];
    if let Some(list) = main_root.iter().into_iter().find(|n| n.tag.ends_with(&suffix)) {
        for node in &list.children {
            let target = node.get(&format!("{R}id")).and_then(|id| links.get(id)).cloned().unwrap_or_default();
            if !target.is_empty() {
                let path = if let Some(t) = target.strip_prefix('/') { t.trim_start_matches('/').to_string() } else { format!("{base}{target}") };
                outv.push((node.get("name").unwrap_or("").to_string(), path));
            }
        }
    }
    Ok(outv)
}

fn text_of_node(node: &crate::xml::Element, tag: &str) -> String {
    node.iter().into_iter().filter(|t| t.tag == tag).map(|t| t.text.clone()).collect()
}

fn paragraphs(root: &crate::xml::Element, p_tag: &str, t_tag: &str) -> String {
    root.iter()
        .into_iter()
        .filter(|p| p.tag == p_tag)
        .map(|p| text_of_node(p, t_tag))
        .filter(|l| !crate::py::strip(l).is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn docx(root: &crate::xml::Element) -> String {
    let para = |p: &crate::xml::Element| -> String {
        let mut o = String::new();
        for node in p.iter() {
            if node.tag == format!("{W}t") {
                o.push_str(&node.text);
            } else if node.tag == format!("{W}tab") {
                o.push('\t');
            } else if node.tag == format!("{W}br") || node.tag == format!("{W}cr") {
                o.push('\n');
            }
        }
        o
    };
    let mut lines = vec![];
    if let Some(body) = root.children.iter().find(|c| c.tag == format!("{W}body")) {
        for block in &body.children {
            if block.tag == format!("{W}p") {
                lines.push(para(block));
            } else if block.tag == format!("{W}tbl") {
                for row in block.iter().into_iter().filter(|e| e.tag == format!("{W}tr")) {
                    let cells: Vec<String> = row
                        .iter()
                        .filter(|e| e.tag == format!("{W}tc"))
                        .map(|cell| {
                            crate::py::strip(
                                &cell.iter().into_iter().filter(|e| e.tag == format!("{W}p")).map(para).collect::<Vec<_>>().join(" "),
                            )
                            .to_string()
                        })
                        .collect();
                    lines.push(cells.join(" | "));
                }
            }
        }
    }
    clean(&lines.join("\n"))
}

fn sheet(root: &crate::xml::Element, shared: &[String]) -> String {
    let mut rows = vec![];
    for row in root.iter().into_iter().filter(|e| e.tag == format!("{S}row")) {
        let mut cells = vec![];
        for c in row.iter().into_iter().filter(|e| e.tag == format!("{S}c")) {
            let kind = c.get("t");
            let v = c.children.iter().find(|e| e.tag == format!("{S}v"));
            let vt = v.map(|v| v.text.clone()).unwrap_or_default();
            let digits = !vt.is_empty() && vt.chars().all(|ch| ch.is_ascii_digit());
            if kind == Some("s") && v.is_some() && digits && vt.parse::<usize>().map(|i| i < shared.len()).unwrap_or(false) {
                cells.push(shared[vt.parse::<usize>().unwrap()].clone());
            } else if kind == Some("inlineStr") {
                cells.push(text_of_node(c, &format!("{S}t")));
            } else if kind == Some("b") && v.is_some() {
                cells.push(if vt == "1" { "TRUE".into() } else { "FALSE".into() });
            } else {
                cells.push(vt);
            }
        }
        if cells.iter().any(|c| !c.is_empty()) {
            rows.push(cells.join(" | "));
        }
    }
    rows.join("\n")
}

// -------------------------------------------------------------------------------------------- text
static NL3: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());
static CTRL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\x00-\x1f\x7f]+").unwrap());

pub fn clean(text: &str) -> String {
    let t = text.replace("\r\n", "\n").replace('\r', "\n").replace('\0', "");
    let t = t.split('\n').map(crate::py::rstrip).collect::<Vec<_>>().join("\n");
    crate::py::strip(&NL3.replace_all(&t, "\n\n")).to_string()
}

fn quoted(name: &str) -> String {
    format!("\"{}\"", CTRL.replace_all(name, " ").replace('"', "'"))
}

fn header(name: &str, pages: Option<usize>, context: Option<&str>, readv: Option<&str>) -> String {
    let mut h = format!("<document name={}", quoted(name));
    if let Some(p) = pages {
        h += &format!(" pages=\"{p}\"");
    }
    if let Some(r) = readv.filter(|r| !r.is_empty()) {
        h += &format!(" read=\"{r}\"");
    }
    h += ">\n";
    if let Some(c) = context.filter(|c| !c.is_empty()) {
        h += &format!("<context>{c}</context>\n");
    }
    h
}

fn wrap(name: &str, pages: Option<usize>, context: Option<&str>, body: &str, count: &str) -> String {
    let mut head = header(name, pages, context, None);
    if !count.is_empty() {
        head = head.replacen(">\n", &format!(" {count}>\n"), 1);
    }
    format!("{head}{body}\n</document>")
}

fn framed(name: &str, pages: usize, readv: Option<&str>, context: Option<&str>, mut items: Items) -> Items {
    let head = header(name, Some(pages), context, readv);
    if items.first().map(|i| i["type"] == "text").unwrap_or(false) {
        let t = format!("{head}{}", items[0]["text"].as_str().unwrap_or(""));
        items[0].insert("text".into(), Value::from(t));
    } else {
        items.insert(0, text_item(&head));
    }
    if items.last().unwrap()["type"] == "text" {
        let l = items.len() - 1;
        let t = format!("{}\n</document>", items[l]["text"].as_str().unwrap_or("").trim_end_matches('\n'));
        items[l].insert("text".into(), Value::from(t));
    } else {
        items.push(text_item("</document>"));
    }
    items
}

fn text_item(text: &str) -> Map<String, Value> {
    json!({"type": "text", "text": text}).as_object().unwrap().clone()
}
