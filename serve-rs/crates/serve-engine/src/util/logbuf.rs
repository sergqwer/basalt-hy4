//! serve/util/logbuf.py: recent server and engine output for GET /api/logs: LogBuffer, TeeStream, ColorStream and
//! tail_file.
use super::loglevel::{LevelReader, ERR, WARN};
use crate::pyfmt;
use regex::Regex;
use serde_json::{Map, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const CAPACITY: usize = 2000;
pub const MAX_LINE: usize = 2000;
pub const SERVER_PREFIX: &str = "[server] ";

/// The ANSI escape pattern the buffer and the colorer strip.
pub fn ansi() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"\x1b(?:\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x40-\x5a\x5c-\x5f])").unwrap()
    })
}

/// `time.time()`.
pub fn now_s() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[derive(Debug, Clone)]
struct Line {
    id: u64,
    t: f64,
    src: String,
    lvl: &'static str,
    msg: String,
}

impl Line {
    fn to_map(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("id".into(), self.id.into());
        m.insert("t".into(), self.t.into());
        m.insert("src".into(), self.src.clone().into());
        m.insert("lvl".into(), self.lvl.into());
        m.insert("msg".into(), self.msg.clone().into());
        m
    }
}

struct Ring {
    lines: VecDeque<Line>,
    last: u64,
    readers: HashMap<String, LevelReader>,      // per source: a traceback spans lines
}

/// A thread-safe ring of the last CAPACITY lines with rising ids.
pub struct LogBuffer {
    capacity: usize,
    ring: Mutex<Ring>,
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl LogBuffer {
    pub fn new() -> Self {
        Self::with_capacity(CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        LogBuffer { capacity, ring: Mutex::new(Ring { lines: VecDeque::new(), last: 0, readers: HashMap::new() }) }
    }

    /// Add each non-empty line of `text` (ANSI removed, long lines cut) from `src`; the level comes from its marker and
    /// a `[server] ` prefix is dropped after that.
    pub fn add(&self, src: &str, text: &str) {
        let now = pyfmt::round_n(now_s(), 3);
        let clean = ansi().replace_all(text, "");
        for raw in pyfmt::splitlines(&clean) {
            let msg = pyfmt::head_chars(pyfmt::rstrip(raw), MAX_LINE);
            if pyfmt::strip(msg).is_empty() {
                continue;
            }
            let mut r = self.ring.lock().unwrap();
            let lvl = r.readers.entry(src.to_string()).or_default().read(msg);
            r.last += 1;
            let id = r.last;
            if self.capacity == 0 {
                continue;
            }
            if r.lines.len() == self.capacity {
                r.lines.pop_front();
            }
            r.lines.push_back(Line { id, t: now, src: src.to_string(), lvl,
                                     msg: msg.strip_prefix(SERVER_PREFIX).unwrap_or(msg).to_string() });
        }
    }

    /// (the newest `limit` lines above `after_id`, oldest first; the newest id).
    pub fn since(&self, after_id: i64, limit: i64) -> (Vec<Map<String, Value>>, u64) {
        let r = self.ring.lock().unwrap();
        let fresh: Vec<&Line> = r.lines.iter().filter(|x| (x.id as i128) > after_id as i128).collect();
        let last = r.last;
        if limit <= 0 {
            return (Vec::new(), last);
        }
        let skip = fresh.len().saturating_sub(limit as usize);
        (fresh[skip..].iter().map(|x| x.to_map()).collect(), last)
    }
}

/// A text stream as Python's sys.stdout is to these wrappers: write, flush, isatty.
pub trait TextWrite: Send + Sync {
    fn write(&self, text: &str);
    fn flush(&self) {}
    fn isatty(&self) -> bool {
        false
    }
}

/// Passes everything through and adds each complete line to the buffer as `src`.
pub struct TeeStream {
    stream: Box<dyn TextWrite>,
    buffer: Arc<LogBuffer>,
    src: String,
    partial: Mutex<String>,
}

impl TeeStream {
    pub fn new(stream: Box<dyn TextWrite>, buffer: Arc<LogBuffer>, src: &str) -> Self {
        TeeStream { stream, buffer, src: src.to_string(), partial: Mutex::new(String::new()) }
    }
}

impl TextWrite for TeeStream {
    fn write(&self, text: &str) {
        self.stream.write(text);
        let done: Vec<String> = {
            let mut p = self.partial.lock().unwrap();
            p.push_str(text);
            let mut parts: Vec<String> = p.split('\n').map(str::to_string).collect();
            *p = parts.pop().unwrap_or_default();
            parts
        };
        for line in done {
            self.buffer.add(&self.src, &line);
        }
    }

    fn flush(&self) {
        self.stream.flush();
    }

    fn isatty(&self) -> bool {
        self.stream.isatty()
    }
}

/// Wraps a terminal stream: error lines red, warning lines yellow, by their markers; cursor-moving text (escape codes,
/// carriage returns) passes at once so in-place redraws keep working.
pub struct ColorStream {
    stream: Box<dyn TextWrite>,
    state: Mutex<(String, LevelReader)>,
}

pub const RED: &str = "\x1b[31m";
pub const YELLOW: &str = "\x1b[33m";
pub const RESET: &str = "\x1b[0m";

impl ColorStream {
    /// The stream wrapped on a terminal without a non-empty NO_COLOR; else the stream itself.
    pub fn wrap(stream: Box<dyn TextWrite>) -> Box<dyn TextWrite> {
        let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
        if stream.isatty() && !no_color {
            Box::new(ColorStream::new(stream))
        } else {
            stream
        }
    }

    pub fn new(stream: Box<dyn TextWrite>) -> Self {
        ColorStream { stream, state: Mutex::new((String::new(), LevelReader::new())) }
    }

    fn paint(reader: &mut LevelReader, line: &str) -> String {
        let lvl = reader.read(&ansi().replace_all(line, ""));
        let color = if lvl == ERR { RED } else if lvl == WARN { YELLOW } else { "" };
        if color.is_empty() {
            line.to_string()
        } else {
            format!("{color}{line}{RESET}")
        }
    }
}

impl TextWrite for ColorStream {
    fn write(&self, text: &str) {
        let mut st = self.state.lock().unwrap();
        let (partial, reader) = &mut *st;
        let joined = format!("{partial}{text}");
        let mut done: Vec<&str> = joined.split('\n').collect();
        let rest = done.pop().unwrap_or("");
        partial.clear();
        let out = if rest.contains('\x1b') || rest.contains('\r') {     // a redraw in progress: no waiting for its newline
            done.push(rest);
            let n = done.len();
            let parts: Vec<String> = done.iter().enumerate()
                .map(|(i, x)| if i < n - 1 { Self::paint(reader, x) } else { x.to_string() })
                .collect();
            parts.join("\n")
        } else {
            *partial = rest.to_string();
            done.iter().map(|x| Self::paint(reader, x) + "\n").collect()
        };
        if !out.is_empty() {
            self.stream.write(&out);
        }
    }

    fn flush(&self) {
        {
            let mut st = self.state.lock().unwrap();
            if !st.0.is_empty() {
                self.stream.write(&st.0);
                st.0.clear();
            }
        }
        self.stream.flush();
    }

    fn isatty(&self) -> bool {
        self.stream.isatty()
    }
}

/// Follow a file from its end, adding complete lines to the buffer; waits for a missing file and rereads a shrunk one
/// from the start. Runs until `stop` is set (Python's never returns; None runs forever). `on_line(text, end offset)`.
pub fn tail_file(buffer: &LogBuffer, path: &Path, src: &str, poll_s: f64, on_line: Option<&dyn Fn(&str, u64)>,
                 skip: Option<&dyn Fn(&str) -> bool>, stop: Option<&AtomicBool>) {
    let size_of = |p: &Path| std::fs::metadata(p).map(|m| m.len());
    let mut pos = size_of(path).unwrap_or(0);
    let mut partial: Vec<u8> = Vec::new();
    loop {
        if stop.is_some_and(|s| s.load(Ordering::SeqCst)) {
            return;
        }
        if let Ok(size) = size_of(path) {
            if size < pos {
                pos = 0;
                partial.clear();
            }
            if size > pos {
                let data = (|| -> std::io::Result<Vec<u8>> {
                    let mut f = std::fs::File::open(path)?;
                    f.seek(SeekFrom::Start(pos))?;
                    let mut buf = Vec::new();
                    f.take((size - pos).min(1 << 20)).read_to_end(&mut buf)?;
                    Ok(buf)
                })();
                if let Ok(data) = data {
                    let mut line_end = pos - partial.len() as u64;
                    pos += data.len() as u64;
                    partial.extend_from_slice(&data);
                    let mut lines: Vec<Vec<u8>> = partial.split(|b| *b == b'\n').map(<[u8]>::to_vec).collect();
                    partial = lines.pop().unwrap_or_default();
                    for line in lines {
                        line_end += line.len() as u64 + 1;
                        let text = String::from_utf8_lossy(&line);
                        if skip.is_some_and(|s| s(&text)) {
                            continue;
                        }
                        buffer.add(src, &text);
                        if let Some(cb) = on_line {
                            cb(&text, line_end);
                        }
                    }
                    continue;
                }
            }
        }
        std::thread::sleep(Duration::from_secs_f64(poll_s.max(0.0)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn msgs(b: &LogBuffer) -> Vec<(String, String)> {
        b.since(0, 500).0.iter().map(|x| (x["lvl"].as_str().unwrap().to_string(), x["msg"].as_str().unwrap().to_string()))
            .collect()
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn levels_per_source() {
        let b = LogBuffer::new();
        b.add("eng", "basalt generate: error: no GPU\nbasalt generate: no error here\nFATAL: not a marker");
        b.add("srv", "Traceback (most recent call last):");
        b.add("eng", "basalt load: cuda0 1 2");
        b.add("srv", "  File \"a.py\", line 1\nOSError: x\n[server] warning: low RAM");
        assert_eq!(msgs(&b), pairs(&[("err", "basalt generate: error: no GPU"), ("info", "basalt generate: no error here"),
                                     ("info", "FATAL: not a marker"), ("err", "Traceback (most recent call last):"),
                                     ("info", "basalt load: cuda0 1 2"), ("err", "  File \"a.py\", line 1"),
                                     ("err", "OSError: x"), ("warn", "warning: low RAM")]));
        let b = LogBuffer::new();
        b.add("srv", "[server] warning  no api key, open to the network\n[server] warnings=0 ok\n[server] error: x");
        assert_eq!(msgs(&b), pairs(&[("warn", "warning  no api key, open to the network"), ("info", "warnings=0 ok"),
                                     ("err", "error: x")]));
    }

    #[test]
    fn ids_lines_and_fields() {
        let b = LogBuffer::new();
        b.add("srv", "one\ntwo\n\n   \nthree");
        let (lines, last) = b.since(0, 500);
        assert_eq!(lines.iter().map(|x| x["msg"].as_str().unwrap()).collect::<Vec<_>>(), ["one", "two", "three"]);
        assert_eq!(lines.iter().map(|x| x["id"].as_u64().unwrap()).collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(last, 3);
        assert_eq!(lines[0].keys().cloned().collect::<Vec<_>>(), ["id", "t", "src", "lvl", "msg"]);
        let t = lines[0]["t"].as_f64().unwrap();
        assert_eq!(t, pyfmt::round_n(t, 3));
    }

    #[test]
    fn after_limit_ring_ansi() {
        let b = LogBuffer::new();
        for i in 0..10 {
            b.add("eng", &format!("line {i}"));
        }
        let ids = |v: Vec<Map<String, Value>>| v.iter().map(|x| x["id"].as_u64().unwrap()).collect::<Vec<_>>();
        assert_eq!(ids(b.since(7, 500).0), [8, 9, 10]);
        assert_eq!(b.since(10, 500), (vec![], 10));
        assert_eq!(ids(b.since(0, 4).0), [7, 8, 9, 10]);
        assert_eq!(LogBuffer::new().since(0, 500), (vec![], 0));
        let b = LogBuffer::new();
        b.add("eng", "\x1b[31mred\x1b[0m and \x1b[1;32mgreen\x1b[0m");
        b.add("eng", &"x".repeat(MAX_LINE + 50));
        let l = b.since(0, 500).0;
        assert_eq!(l[0]["msg"], "red and green");
        assert_eq!(l[1]["msg"].as_str().unwrap().len(), MAX_LINE);
        let b = LogBuffer::new();
        for i in 0..CAPACITY + 5 {
            b.add("srv", &format!("n{i}"));
        }
        let (l, last) = b.since(0, (CAPACITY * 2) as i64);
        assert_eq!((l.len(), last), (CAPACITY, (CAPACITY + 5) as u64));
        assert_eq!(l[0]["id"], 6);
    }

    #[test]
    fn threads_get_distinct_ids() {
        let b = Arc::new(LogBuffer::new());
        let hs: Vec<_> = (0..8).map(|_| {
            let b = b.clone();
            std::thread::spawn(move || (0..100).for_each(|_| b.add("srv", "x")))
        }).collect();
        hs.into_iter().for_each(|h| h.join().unwrap());
        let ids: Vec<u64> = b.since(0, 1000).0.iter().map(|x| x["id"].as_u64().unwrap()).collect();
        assert_eq!(ids, (1..=800).collect::<Vec<u64>>());
    }

    #[derive(Clone, Default)]
    struct Cap(Arc<Mutex<String>>, bool);
    impl TextWrite for Cap {
        fn write(&self, text: &str) {
            self.0.lock().unwrap().push_str(text);
        }
        fn isatty(&self) -> bool {
            self.1
        }
    }

    #[test]
    fn tee_keeps_whole_lines() {
        let (out, b) = (Cap::default(), Arc::new(LogBuffer::new()));
        let tee = TeeStream::new(Box::new(out.clone()), b.clone(), "srv");
        tee.write("[server] model unloaded");
        tee.write("\n");
        tee.write("part");
        assert_eq!(msgs(&b), pairs(&[("info", "model unloaded")]));
        tee.write("ial\nnext: failed\n");
        assert_eq!(msgs(&b)[1..], pairs(&[("info", "partial"), ("info", "next: failed")]));
        assert_eq!(*out.0.lock().unwrap(), "[server] model unloaded\npartial\nnext: failed\n");
    }

    const LINES: &str = "[server] warning: model 76.9 GiB\nbasalt generate: error: boom\n[server] error: refused Host 'x'\n\
                         [server] listening\nbasalt serve: warning: low\nbasalt serve: no error here\n";

    #[test]
    fn color() {
        // off a terminal: unwrapped (the stream's own bytes)
        let out = Cap::default();
        let w = ColorStream::wrap(Box::new(out.clone()));
        w.write(LINES);
        assert_eq!(*out.0.lock().unwrap(), LINES);
        let out = Cap(Default::default(), true);
        let w = ColorStream::new(Box::new(out.clone()));
        for ch in LINES.chars() {
            w.write(&ch.to_string());
        }
        w.flush();
        let (y, r, z) = (YELLOW, RED, RESET);
        assert_eq!(*out.0.lock().unwrap(),
                   format!("{y}[server] warning: model 76.9 GiB{z}\n{r}basalt generate: error: boom{z}\n\
                            {r}[server] error: refused Host 'x'{z}\n[server] listening\n{y}basalt serve: warning: low{z}\n\
                            basalt serve: no error here\n"));
        let out = Cap(Default::default(), true);
        let w = ColorStream::new(Box::new(out.clone()));
        w.write("\x1b[1A\x1b[2K");
        w.write("bar 50%\r");
        assert_eq!(*out.0.lock().unwrap(), "\x1b[1A\x1b[2Kbar 50%\r");
        let b = Arc::new(LogBuffer::new());
        let t = TeeStream::new(Box::new(ColorStream::new(Box::new(Cap(Default::default(), true)))), b.clone(), "srv");
        t.write("[server] warning: x\n");
        assert_eq!(b.since(0, 9).0[0]["msg"], "warning: x");
        assert!(t.isatty());
    }

    fn wait_for(b: &LogBuffer, n: usize) {
        let end = Instant::now() + Duration::from_secs(5);
        while Instant::now() < end && b.since(0, 500).0.len() < n {
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("serve-logbuf-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn append(p: &Path, s: &str) {
        use std::io::Write;
        std::fs::OpenOptions::new().append(true).create(true).open(p).unwrap().write_all(s.as_bytes()).unwrap();
    }

    #[test]
    fn tail_follows_and_survives_truncation() {
        let d = tmp("follow");
        let log = d.join("engine.log");
        std::fs::write(&log, "an earlier run\n").unwrap();
        let b = Arc::new(LogBuffer::new());
        let stop = Arc::new(AtomicBool::new(false));
        let seen: Arc<Mutex<Vec<(String, u64)>>> = Default::default();
        let h = {
            let (b, stop, log, seen) = (b.clone(), stop.clone(), log.clone(), seen.clone());
            std::thread::spawn(move || {
                let cb = |t: &str, p: u64| seen.lock().unwrap().push((t.to_string(), p));
                let skip = |l: &str| l.starts_with("basalt load: cuda0 ");
                tail_file(&b, &log, "eng", 0.02, Some(&cb), Some(&skip), Some(&stop))
            })
        };
        std::thread::sleep(Duration::from_millis(200));
        append(&log, "basalt: ready\nbasalt load: cuda0 1 2\nhalf a li");
        wait_for(&b, 1);
        assert_eq!(msgs(&b), pairs(&[("info", "basalt: ready")]));
        assert_eq!(seen.lock().unwrap()[0], ("basalt: ready".to_string(), "an earlier run\nbasalt: ready\n".len() as u64));
        append(&log, "ne\n");
        wait_for(&b, 2);
        assert_eq!(b.since(0, 9).0[1]["msg"], "half a line");
        std::fs::write(&log, "new run\n").unwrap();
        wait_for(&b, 3);
        let l = b.since(0, 9).0;
        assert_eq!((l[2]["msg"].as_str(), l[2]["src"].as_str()), (Some("new run"), Some("eng")));
        stop.store(true, Ordering::SeqCst);
        h.join().unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn tail_waits_for_a_file() {
        let d = tmp("later");
        let log = d.join("later.log");
        let b = Arc::new(LogBuffer::new());
        let stop = Arc::new(AtomicBool::new(false));
        let h = {
            let (b, stop, log) = (b.clone(), stop.clone(), log.clone());
            std::thread::spawn(move || tail_file(&b, &log, "eng", 0.02, None, None, Some(&stop)))
        };
        std::thread::sleep(Duration::from_millis(100));
        std::fs::write(&log, "hello\n").unwrap();
        wait_for(&b, 1);
        assert_eq!(b.since(0, 9).0[0]["msg"], "hello");
        stop.store(true, Ordering::SeqCst);
        h.join().unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }
}
