//! serve/engine/vision.py: the image and video encoder (a sidecar process, or inside the engine), its cache, and the
//! GPU helpers the engine and encoder environments are built from.

use crate::image_source::{load_image, Bytes, ImageSources, SourceErr, IMAGE, VIDEO};
use serde_json::{Map, Value};
use serve_engine::iface::Engine;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const VISION_READY_S: f64 = 300.0;
pub const VISION_ENCODE_S: f64 = 300.0;
pub const VISION_CACHE_BYTES: u64 = 1 << 30;
pub const IMAGE_MAX_PIXELS: u64 = 64 << 20;
pub const IMAGE_REQUEST_BYTES: u64 = 1 << 30;
pub const IMAGE_WAITING_BYTES: u64 = 2 << 30;
pub const VIDEO_REQUEST_BYTES: u64 = 1 << 30;
pub const VIDEO_MAX_TOKENS: i64 = 8192;
pub const VISION_MODES: [&str; 3] = ["off", "dynamic", "cpu"];
pub const VISION_USAGE: &str = "vision: use one of off, dynamic, cpu or a GPU number, as nvidia-smi numbers them";

static DECODES: Mutex<u32> = Mutex::new(0);
static DECODES_CV: std::sync::Condvar = std::sync::Condvar::new();

fn unlink_quietly(p: &Path) {
    let _ = std::fs::remove_file(p);
}

fn hex12() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..12].to_string()
}

fn hex8() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..8].to_string()
}

fn sha32(parts: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    let d = h.finalize();
    d.iter().map(|b| format!("{b:02x}")).collect::<String>()[..32].to_string()
}

/// An encoded video: `groups` records of `rows` rows each, and each group's time in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoRows {
    pub groups: i64,
    pub rows: i64,
    pub times: Vec<f64>,
}

/// video_rows: a video reply's `OK <groups> <nx> <ny> <ms> <time,...>`.
pub fn video_rows(line: &str) -> Result<VideoRows, String> {
    let w: Vec<&str> = line.split_whitespace().collect();
    let num = |i: usize| -> Result<i64, String> {
        w.get(i).and_then(|x| x.parse().ok()).ok_or_else(|| format!("bad video reply: {line}"))
    };
    let times = w
        .get(5)
        .ok_or_else(|| format!("bad video reply: {line}"))?
        .split(',')
        .map(|t| t.parse::<f64>().map_err(|_| format!("bad video reply: {line}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(VideoRows { groups: num(1)?, rows: num(2)? * num(3)?, times })
}

/// Per item of a request's media: an image's token count or a video's rows.
#[derive(Clone, Debug)]
pub enum MediaResult {
    Image(i64),
    Video(VideoRows),
}

/// One encode job, in prompt order.
pub enum Job {
    Image(Bytes),
    Video(Bytes, Map<String, Value>),
}

struct Cache {
    entries: indexmap::IndexMap<String, (PathBuf, i64, u64)>,
    bytes: u64,
    pinned: HashMap<String, i64>,
    videos: HashMap<String, VideoRows>,
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    lines: mpsc::Receiver<String>,
}

/// The sidecar's spawn: argv, log file, environment.
#[derive(Clone)]
pub struct Spawn {
    pub args: Vec<String>,
    pub log: Option<PathBuf>,
    pub env: Option<Vec<(String, String)>>,
}

/// The encoder: the resident `basalt-vision` sidecar, or the engine's own (`--vision dynamic`).
pub struct Vision {
    pub dir: PathBuf,
    spawn: Option<Spawn>,
    proc: Mutex<Option<Proc>>,
    stopped: std::sync::atomic::AtomicBool,
    pub ready_s: Mutex<f64>,
    video: Mutex<Option<String>>,
    lock: Mutex<Cache>,
    engine: Mutex<Option<Arc<dyn Engine>>>,
    in_engine: bool,
}

/// The sidecar config: cfg["vision_bin"], cfg["model"], cfg.get("gpu").
pub struct VisionCfg {
    pub vision_bin: String,
    pub model: String,
    pub gpu: bool,
}

fn mkdtemp() -> PathBuf {
    let base = std::env::temp_dir();
    loop {
        let d = base.join(format!("basalt-vision-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]));
        if std::fs::create_dir(&d).is_ok() {
            return d;
        }
    }
}

impl Vision {
    fn base(spawn: Option<Spawn>, in_engine: bool) -> Vision {
        Vision {
            dir: mkdtemp(),
            spawn,
            proc: Mutex::new(None),
            stopped: std::sync::atomic::AtomicBool::new(false),
            ready_s: Mutex::new(0.0),
            video: Mutex::new(None),
            lock: Mutex::new(Cache { entries: Default::default(), bytes: 0, pinned: HashMap::new(), videos: HashMap::new() }),
            engine: Mutex::new(None),
            in_engine,
        }
    }

    /// Vision(cfg, log, env, start): the sidecar; `start` false leaves it down until the service restarts it.
    pub fn sidecar(cfg: &VisionCfg, log: Option<PathBuf>, env: Option<Vec<(String, String)>>, start: bool) -> Result<Vision, String> {
        let mut args = vec![cfg.vision_bin.clone(), "--model".into(), cfg.model.clone()];
        if cfg.gpu {
            args.push("--gpu".into());
        }
        let v = Vision::base(Some(Spawn { args, log, env }), false);
        if start {
            v.start()?;
        } else {
            v.set_stopped(true);
        }
        Ok(v)
    }

    /// EngineVision(cfg): the encoder inside the engine process.
    pub fn engine_vision() -> Vision {
        Vision::base(None, true)
    }

    pub fn attach(&self, engine: Arc<dyn Engine>) {
        *self.engine.lock().unwrap() = Some(engine);
    }

    pub fn in_engine(&self) -> bool {
        self.in_engine
    }

    fn set_stopped(&self, v: bool) {
        self.stopped.store(v, std::sync::atomic::Ordering::SeqCst);
    }

    fn start(&self) -> Result<(), String> {
        let sp = self.spawn.clone().unwrap();
        let t0 = std::time::Instant::now();
        let mut cmd = Command::new(&sp.args[0]);
        cmd.args(&sp.args[1..]).stdin(Stdio::piped()).stdout(Stdio::piped());
        match &sp.log {
            Some(p) => {
                let f = std::fs::OpenOptions::new().create(true).append(true).open(p).map_err(|e| e.to_string())?;
                cmd.stderr(f);
            }
            None => {
                cmd.stderr(Stdio::null());
            }
        }
        if let Some(env) = &sp.env {
            cmd.env_clear();
            for (k, v) in env {
                cmd.env(k, v);
            }
        }
        let mut child = serve_engine::lifeline::spawn(cmd).map_err(|e| e.to_string())?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut r = BufReader::new(stdout);
            loop {
                let mut line = Vec::new();
                match r.read_until(b'\n', &mut line) {
                    Ok(0) | Err(_) => {
                        let _ = tx.send(String::new());
                        break;
                    }
                    Ok(_) => {
                        if tx.send(String::from_utf8_lossy(&line).replace("\r\n", "\n")).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        *self.proc.lock().unwrap() = Some(Proc { child, stdin, lines: rx });
        let what = "the vision encoder did not start";
        let mut line = match self.readline(VISION_READY_S, what) {
            Ok(l) => l,
            Err(e) => {
                self.set_stopped(true);
                return Err(e);
            }
        };
        if let Some(v) = line.strip_prefix("VIDEO ") {
            *self.video.lock().unwrap() = Some(crate::py::strip(v).to_string());
            line = self.readline(VISION_READY_S, what)?;
        }
        if !line.starts_with("READY") {
            return Err(format!("{what}: {}", crate::py::strip(&line)));
        }
        *self.ready_s.lock().unwrap() = t0.elapsed().as_secs_f64();
        self.set_stopped(false);
        Ok(())
    }

    /// _readline: one line from the encoder, at most `timeout` s; on a timeout the encoder is killed.
    fn readline(&self, timeout: f64, what: &str) -> Result<String, String> {
        let mut g = self.proc.lock().unwrap();
        let p = g.as_mut().ok_or_else(|| format!("{what}: it is not running"))?;
        match p.lines.recv_timeout(Duration::from_secs_f64(timeout)) {
            Ok(l) => Ok(l),
            Err(mpsc::RecvTimeoutError::Disconnected) => Ok(String::new()),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = p.child.kill();
                Err(format!("{what}: it said nothing for {timeout:.0} s"))
            }
        }
    }

    pub fn alive(&self) -> bool {
        if self.in_engine {
            return true;
        }
        if self.stopped.load(std::sync::atomic::Ordering::SeqCst) {
            return false;
        }
        let mut g = self.proc.lock().unwrap();
        match g.as_mut() {
            Some(p) => matches!(p.child.try_wait(), Ok(None)),
            None => false,
        }
    }

    /// video_state: "on ..." / "off <reason>"; None before the encoder has said.
    pub fn video_state(&self) -> Option<String> {
        if self.in_engine {
            return self.engine.lock().unwrap().as_ref().and_then(|e| e.video());
        }
        self.video.lock().unwrap().clone()
    }

    pub fn read_video(&self, source: &Value, allowed: &ImageSources) -> Result<Vec<u8>, SourceErr> {
        load_image(source, Some(allowed), VIDEO)
    }

    pub fn unload(&self) {
        if self.in_engine {
            return;
        }
        self.quit();
        self.set_stopped(true);
    }

    pub fn restart(&self) -> Result<(), String> {
        if self.in_engine {
            return Ok(());
        }
        if let Some(p) = self.proc.lock().unwrap().as_mut() {
            let _ = p.child.kill();
        }
        self.start()
    }

    /// read: the request's images read under its policy and normalized, each source once.
    pub fn read(&self, sources: &[Value], allowed: &ImageSources) -> Result<Vec<Bytes>, SourceErr> {
        let mut seen: Vec<(String, Bytes)> = vec![];
        let mut out = vec![];
        let mut total = 0u64;
        for source in sources {
            let key = source.as_str();
            if let Some((_, d)) = key.and_then(|k| seen.iter().find(|(s, _)| s == k)) {
                out.push(d.clone());
                continue;
            }
            let data = Arc::new(normalize(load_image(source, Some(allowed), IMAGE)?).map_err(SourceErr::value)?);
            total += data.len() as u64;
            if total > IMAGE_REQUEST_BYTES {
                return Err(SourceErr::value(format!(
                    "the request's images are larger than {} MiB",
                    IMAGE_REQUEST_BYTES >> 20
                )));
            }
            if let Some(k) = key {
                seen.push((k.to_string(), data.clone()));
            }
            out.push(data);
        }
        Ok(out)
    }

    fn pin(c: &mut Cache, key: &str, pins: &mut Vec<String>) {
        *c.pinned.entry(key.to_string()).or_insert(0) += 1;
        pins.push(key.to_string());
        if let Some(v) = c.entries.shift_remove(key) {
            c.entries.insert(key.to_string(), v);
        }
    }

    fn unpin(&self, pins: &[String]) {
        let mut c = self.lock.lock().unwrap();
        for k in pins {
            if let Some(n) = c.pinned.get_mut(k) {
                *n -= 1;
                if *n == 0 {
                    c.pinned.remove(k);
                }
            }
        }
        Self::evict(&mut c);
    }

    fn evict(c: &mut Cache) {
        let keys: Vec<String> = c.entries.keys().filter(|k| !c.pinned.contains_key(*k)).cloned().collect();
        for k in keys {
            if c.bytes <= VISION_CACHE_BYTES {
                break;
            }
            if let Some((path, _, size)) = c.entries.shift_remove(&k) {
                let _ = std::fs::remove_file(path);
                c.bytes -= size;
                c.videos.remove(&k);
            }
        }
    }

    fn remember(c: &mut Cache, key: &str, out: PathBuf, n: i64) -> (PathBuf, i64, u64) {
        let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
        c.entries.insert(key.to_string(), (out, n, size));
        c.bytes += size;
        Self::evict(c);
        c.entries[key].clone()
    }

    fn enc(&self, img: &Path, out: &Path) -> Result<String, String> {
        self.send(&format!("ENC {} {}\n", img.display(), out.display()))?;
        match self.readline(VISION_ENCODE_S, "the image could not be read") {
            Ok(l) => Ok(crate::py::strip(&l).to_string()),
            Err(e) => {
                self.set_stopped(true);
                Err(e)
            }
        }
    }

    fn send(&self, line: &str) -> Result<(), String> {
        let mut g = self.proc.lock().unwrap();
        let p = g.as_mut().ok_or("the vision encoder stopped")?;
        p.stdin.write_all(line.as_bytes()).and_then(|_| p.stdin.flush()).map_err(|e| e.to_string())
    }

    fn encv(&self, video: &Path, out: &Path, words: &str) -> Result<String, String> {
        if self.in_engine {
            let e = self.engine.lock().unwrap().clone().ok_or("the vision encoder stopped")?;
            return e
                .encode_video(&video.display().to_string(), &out.display().to_string(), words, None)
                .map_err(|e| e.message());
        }
        let line = format!("ENCV {} {} {words}", video.display(), out.display());
        self.send(&format!("{}\n", line.trim_end()))?;
        match self.readline(VISION_ENCODE_S, "the video could not be read") {
            Ok(l) => Ok(crate::py::strip(&l).to_string()),
            Err(e) => {
                self.set_stopped(true);
                Err(e)
            }
        }
    }

    fn encode_one(&self, data: &[u8], pins: &mut Vec<String>) -> Result<(PathBuf, i64, u64), String> {
        let key = sha32(&[data]);
        let mut c = self.lock.lock().unwrap();
        Self::pin(&mut c, &key, pins);
        if let Some(v) = c.entries.get(&key) {
            return Ok(v.clone());
        }
        let tag = hex8();
        let img = self.dir.join(format!("{key}-{tag}.img"));
        let part = self.dir.join(format!("{key}-{tag}.part"));
        let out = self.dir.join(format!("{key}.sve"));
        std::fs::write(&img, data).map_err(|e| e.to_string())?;
        let line = self.enc(&img, &part);
        unlink_quietly(&img);
        let line = line?;
        if !line.starts_with("OK") {
            unlink_quietly(&part);
            return Err(format!(
                "the image could not be read: {}",
                if let Some(r) = line.strip_prefix("ERR") { r.chars().skip(1).collect::<String>() } else { "the vision encoder stopped".into() }
            ));
        }
        std::fs::rename(&part, &out).map_err(|e| e.to_string())?;
        let n = line.split_whitespace().nth(1).and_then(|x| x.parse().ok()).ok_or("bad encoder reply")?;
        Ok(Self::remember(&mut c, &key, out, n))
    }

    fn encode_pinned(&self, images: &[Bytes], pins: &mut Vec<String>) -> Result<Vec<(PathBuf, i64, u64)>, String> {
        if !self.in_engine {
            return images.iter().map(|d| self.encode_one(d, pins)).collect();
        }
        let keys: Vec<String> = images.iter().map(|d| sha32(&[d])).collect();
        let mut c = self.lock.lock().unwrap();
        for k in &keys {
            Self::pin(&mut c, k, pins);
        }
        let mut todo: Vec<(String, Bytes)> = vec![];
        for (k, d) in keys.iter().zip(images) {
            if !c.entries.contains_key(k) && !todo.iter().any(|(x, _)| x == k) {
                todo.push((k.clone(), d.clone()));
            }
        }
        let tag = hex8();
        let jobs: Vec<(PathBuf, PathBuf)> =
            todo.iter().map(|(k, _)| (self.dir.join(format!("{k}-{tag}.img")), self.dir.join(format!("{k}-{tag}.part")))).collect();
        let r = (|| -> Result<(), String> {
            for ((img, _), (_, d)) in jobs.iter().zip(&todo) {
                std::fs::write(img, &d[..]).map_err(|e| e.to_string())?;
            }
            if !jobs.is_empty() {
                let e = self.engine.lock().unwrap().clone().ok_or("the vision encoder stopped")?;
                let pairs: Vec<(String, String)> =
                    jobs.iter().map(|(a, b)| (a.display().to_string(), b.display().to_string())).collect();
                let lines = e.encode_images(&pairs, None).map_err(|e| e.message())?;
                for (((k, _), (_, part)), line) in todo.iter().zip(&jobs).zip(lines) {
                    let n: i64 = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|x| x.parse().ok())
                        .ok_or_else(|| format!("invalid literal for int() with base 10: {}", serve_text::pyjson::py_str_repr(line.split_whitespace().nth(1).unwrap_or(""))))?;
                    let out = self.dir.join(format!("{k}.sve"));
                    std::fs::rename(part, &out).map_err(|e| e.to_string())?;
                    Self::remember(&mut c, k, out, n);
                }
            }
            Ok(())
        })();
        for (img, part) in &jobs {
            unlink_quietly(img);
            unlink_quietly(part);
        }
        r?;
        Ok(keys.iter().map(|k| c.entries[k].clone()).collect())
    }

    fn encode_video(&self, data: &[u8], opts: &Map<String, Value>, pins: &mut Vec<String>) -> Result<(PathBuf, VideoRows), String> {
        let mut keys: Vec<&String> = opts.keys().collect();
        keys.sort();
        let words = keys
            .iter()
            .map(|k| {
                let v = &opts[*k];
                if v.is_f64() {
                    format!("{k}={}", serve_text::pyjson::py_float_repr(v.as_f64().unwrap()))
                } else {
                    format!("{k}={}", serve_text::pyjson::py_str(v))
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        let key = sha32(&[data, b"\0", words.as_bytes()]);
        let mut c = self.lock.lock().unwrap();
        Self::pin(&mut c, &key, pins);
        if let (Some(e), Some(v)) = (c.entries.get(&key), c.videos.get(&key)) {
            return Ok((e.0.clone(), v.clone()));
        }
        let tag = hex8();
        let vid = self.dir.join(format!("{key}-{tag}.vid"));
        let part = self.dir.join(format!("{key}-{tag}.part"));
        let out = self.dir.join(format!("{key}.sve"));
        std::fs::write(&vid, data).map_err(|e| e.to_string())?;
        let line = self.encv(&vid, &part, &words);
        unlink_quietly(&vid);
        let line = line?;
        if !line.starts_with("OK") {
            unlink_quietly(&part);
            return Err(format!(
                "the video could not be read: {}",
                if line.starts_with("ERR") { line.chars().skip(4).collect::<String>() } else { "the vision encoder stopped".into() }
            ));
        }
        std::fs::rename(&part, &out).map_err(|e| e.to_string())?;
        let rows = video_rows(&line)?;
        c.videos.insert(key.clone(), rows.clone());
        let p = Self::remember(&mut c, &key, out, rows.groups * rows.rows).0;
        Ok((p, rows))
    }

    /// encode_media: images and videos in prompt order into one embeddings file; per item its result.
    pub fn encode_media(&self, jobs: &[Job]) -> Result<(String, Vec<MediaResult>), String> {
        let mut pins = vec![];
        let r = (|| {
            let imgs: Vec<Bytes> = jobs
                .iter()
                .filter_map(|j| match j {
                    Job::Image(d) => Some(d.clone()),
                    _ => None,
                })
                .collect();
            let images = self.encode_pinned(&imgs, &mut pins)?;
            let mut parts: Vec<(PathBuf, MediaResult)> = vec![];
            let mut k = 0;
            for j in jobs {
                match j {
                    Job::Image(_) => {
                        parts.push((images[k].0.clone(), MediaResult::Image(images[k].1)));
                        k += 1;
                    }
                    Job::Video(d, opts) => {
                        let (p, rows) = self.encode_video(d, opts, &mut pins)?;
                        parts.push((p, MediaResult::Video(rows)));
                    }
                }
            }
            let combined = self.dir.join(format!("req-{}.sve", hex12()));
            let w = (|| -> std::io::Result<()> {
                let mut f = std::fs::File::create(&combined)?;
                for (p, _) in &parts {
                    f.write_all(&std::fs::read(p)?)?;
                }
                Ok(())
            })();
            if let Err(e) = w {
                let _ = std::fs::remove_file(&combined);
                return Err(e.to_string());
            }
            Ok((combined.display().to_string(), parts.into_iter().map(|(_, r)| r).collect()))
        })();
        self.unpin(&pins);
        r
    }

    fn quit(&self) {
        let mut g = self.proc.lock().unwrap();
        let Some(p) = g.as_mut() else { return };
        let ok = p.stdin.write_all(b"QUIT\n").and_then(|_| p.stdin.flush()).is_ok();
        if ok {
            let t = std::time::Instant::now();
            while t.elapsed() < Duration::from_secs(10) {
                if let Ok(Some(_)) = p.child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        let _ = p.child.kill();
    }

    /// close: the encoder quits, and its work folder goes with the embeddings cached in it.
    pub fn close(&self) {
        if !self.in_engine {
            self.quit();
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

// -------------------------------------------------------------------------------------------- normalize
const PILLOW_FORMATS: &str = "JPEG, PNG, BMP, GIF, WEBP, TIFF, TGA, PPM, QOI";

/// The EXIF Orientation of a JPEG, when it is there and not 1.
fn exif_rotated(data: &[u8]) -> bool {
    matches!(jpeg_orientation(data), Some(o) if o != 0 && o != 1)
}

/// The Orientation tag (0x0112) of a JPEG's APP1 Exif segment.
pub fn jpeg_orientation(data: &[u8]) -> Option<u16> {
    let mut i = 2;
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            return None;
        }
        let marker = data[i + 1];
        if marker == 0xD9 || marker == 0xDA {
            return None;
        }
        let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        let seg = data.get(i + 4..i + 2 + len)?;
        if marker == 0xE1 && seg.starts_with(b"Exif\0\0") {
            return tiff_orientation(&seg[6..]);
        }
        i += 2 + len;
    }
    None
}

fn tiff_orientation(t: &[u8]) -> Option<u16> {
    let le = match t.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b = t.get(o..o + 2)?;
        Some(if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b = t.get(o..o + 4)?;
        Some(if le { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) } else { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) })
    };
    let ifd = u32_at(4)? as usize;
    let n = u16_at(ifd)? as usize;
    for k in 0..n {
        let e = ifd + 2 + 12 * k;
        if u16_at(e)? == 0x0112 {
            return u16_at(e + 8);
        }
    }
    None
}

/// normalize: formats the decoder reads pass through; others, and a turned JPEG, become a PNG.
pub fn normalize(data: Vec<u8>) -> Result<Vec<u8>, String> {
    let jpeg = data.starts_with(b"\xff\xd8\xff");
    if (jpeg && !exif_rotated(&data))
        || data.starts_with(b"\x89PNG\r\n\x1a\n")
        || data.starts_with(b"BM")
        || data.starts_with(b"GIF87a")
        || data.starts_with(b"GIF89a")
    {
        return Ok(data);
    }
    {
        let mut n = DECODES.lock().unwrap();
        while *n >= 4 {
            n = DECODES_CV.wait(n).unwrap();
        }
        *n += 1;
    }
    let r = converted(&data);
    *DECODES.lock().unwrap() -= 1;
    DECODES_CV.notify_one();
    r
}

fn converted(data: &[u8]) -> Result<Vec<u8>, String> {
    use image::ImageFormat as F;
    let fmt = image::guess_format(data).ok().filter(|f| {
        matches!(f, F::Jpeg | F::Png | F::Bmp | F::Gif | F::WebP | F::Tiff | F::Tga | F::Pnm | F::Qoi)
    });
    let Some(fmt) = fmt else {
        return Err("the image could not be read: not a format this server decodes".into());
    };
    let reader = image::ImageReader::with_format(std::io::Cursor::new(data), fmt);
    let (w, h) = reader.into_dimensions().map_err(|e| format!("the image could not be read ({e})"))?;
    if w as u64 * h as u64 > IMAGE_MAX_PIXELS {
        return Err(format!("the image is {w}x{h} pixels; the limit is {} megapixels", IMAGE_MAX_PIXELS >> 20));
    }
    let mut dec = image::ImageReader::with_format(std::io::Cursor::new(data), fmt);
    dec.no_limits();
    let mut im = dec.decode().map_err(|e| format!("the image could not be read ({e})"))?;
    let orient = if fmt == F::Jpeg { jpeg_orientation(data) } else { None };
    if let Some(o) = orient {
        im = match o {
            2 => im.fliph(),
            3 => im.rotate180(),
            4 => im.flipv(),
            5 => im.rotate90().fliph(),
            6 => im.rotate90(),
            7 => im.rotate270().fliph(),
            8 => im.rotate270(),
            _ => im,
        };
    }
    let rgb = if im.color().has_alpha() {
        // transparency becomes white
        let rgba = im.to_rgba8();
        let mut out = image::RgbImage::new(rgba.width(), rgba.height());
        for (x, y, p) in rgba.enumerate_pixels() {
            let a = p[3] as u32;
            let blend = |c: u8| ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
            out.put_pixel(x, y, image::Rgb([blend(p[0]), blend(p[1]), blend(p[2])]));
        }
        out
    } else {
        im.to_rgb8()
    };
    let mut out = Vec::new();
    image::DynamicImage::ImageRgb8(rgb)
        .write_to(&mut std::io::Cursor::new(&mut out), F::Png)
        .map_err(|e| format!("the image could not be read ({e})"))?;
    Ok(out)
}

#[allow(dead_code)]
const _FORMATS: &str = PILLOW_FORMATS;

// -------------------------------------------------------------------------------------------- GPU helpers
/// gpu_text: "" (unset), "auto" or "A,B,..."; a config may give a number or a list.
pub fn gpu_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Array(a)) => a.iter().map(|x| crate::py::strip(&serve_text::pyjson::py_str(x)).to_string()).collect::<Vec<_>>().join(","),
        Some(v) => crate::py::strip(&serve_text::pyjson::py_str(v)).to_string(),
    }
}

/// check_gpu: Err(text) unless the gpu setting reads as auto or distinct GPU numbers.
pub fn check_gpu(value: Option<&Value>) -> Result<(), String> {
    let text = gpu_text(value);
    if text.is_empty() || text == "auto" {
        return Ok(());
    }
    let items: Vec<&str> = text.split(',').map(crate::py::strip).collect();
    let digits = items.iter().all(|x| !x.is_empty() && x.chars().all(|c| c.is_ascii_digit()));
    if !digits {
        return Err(text);
    }
    let mut seen: Vec<u128> = vec![];
    for x in &items {
        let n: u128 = x.parse().unwrap_or(u128::MAX);
        if seen.contains(&n) {
            return Err(text);
        }
        seen.push(n);
    }
    Ok(())
}

/// A card as CUDA_VISIBLE_DEVICES names it: a number, or a UUID.
#[derive(Clone, Debug, PartialEq)]
pub enum Card {
    Index(i64),
    Name(String),
}

impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Card::Index(i) => write!(f, "{i}"),
            Card::Name(s) => f.write_str(s),
        }
    }
}

/// gpu_list: the cards the engine runs on, main first.
pub fn gpu_list(cfg: &Map<String, Value>) -> Vec<Card> {
    let g = gpu_text(cfg.get("gpu"));
    if g.is_empty() || g == "auto" {
        return vec![];
    }
    g.split(',')
        .map(crate::py::strip)
        .filter(|x| !x.is_empty())
        .map(|x| if x.chars().all(|c| c.is_ascii_digit()) { Card::Index(x.parse().unwrap_or(0)) } else { Card::Name(x.to_string()) })
        .collect()
}

/// gpus_in_play: the nvidia-smi numbers of the GPUs the engine uses, main first; [0] if none.
pub fn gpus_in_play(cfg: &Map<String, Value>) -> Vec<i64> {
    let smi: Vec<i64> = cfg
        .get("gpu_smi")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter(|x| crate::py::is_int(x)).filter_map(|x| x.as_i64()).filter(|i| *i >= 0).collect())
        .unwrap_or_default();
    if !smi.is_empty() {
        return smi;
    }
    let l: Vec<i64> = gpu_list(cfg)
        .into_iter()
        .filter_map(|c| match c {
            Card::Index(i) => Some(i),
            _ => None,
        })
        .collect();
    if l.is_empty() {
        vec![0]
    } else {
        l
    }
}

pub fn engine_args(cfg: &Map<String, Value>) -> Vec<String> {
    cfg.get("args").and_then(|a| a.as_array()).map(|a| a.iter().map(serve_text::pyjson::py_str).collect()).unwrap_or_default()
}

/// child_env: the engine environment, only the --gpu cards visible.
pub fn child_env(cfg: &Map<String, Value>) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = std::env::vars().collect();
    let l = gpu_list(cfg);
    if !l.is_empty() {
        set_env(&mut env, "CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        set_env(&mut env, "CUDA_VISIBLE_DEVICES", &l.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","));
    }
    env
}

fn set_env(env: &mut Vec<(String, String)>, k: &str, v: &str) {
    let win = cfg!(windows);
    if let Some(e) = env.iter_mut().find(|(x, _)| if win { x.eq_ignore_ascii_case(k) } else { x == k }) {
        e.1 = v.to_string();
    } else {
        env.push((k.to_string(), v.to_string()));
    }
}

/// vision_mode: one of VISION_MODES (None: "off") or a GPU number as a string.
pub fn vision_mode(value: Option<&Value>) -> Result<String, String> {
    let Some(v) = value.filter(|v| !v.is_null()) else { return Ok("off".into()) };
    let text = crate::py::strip(&serve_text::pyjson::py_str(v)).to_lowercase();
    if VISION_MODES.contains(&text.as_str()) {
        return Ok(text);
    }
    if !text.is_empty() && text.chars().all(|c| c.is_ascii_digit()) {
        let t = text.trim_start_matches('0');
        return Ok(if t.is_empty() { "0".into() } else { t.to_string() });
    }
    Err(format!("{VISION_USAGE} (got {})", serve_text::pyjson::py_repr(v)))
}

/// vision_card: the GPU number a vision mode names; None for off, dynamic and cpu.
pub fn vision_card(mode: &str) -> Option<i64> {
    let t = crate::py::strip(mode);
    if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
        t.parse().ok()
    } else {
        None
    }
}

/// vision_env: a GPU-numbered vision mode runs the sidecar on that card alone.
pub fn vision_env(env: &[(String, String)], mode: &str) -> Vec<(String, String)> {
    let mut env = env.to_vec();
    if let Some(card) = vision_card(mode) {
        set_env(&mut env, "CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        set_env(&mut env, "CUDA_VISIBLE_DEVICES", &card.to_string());
    }
    env
}

/// gpu_card_name: a card's name from one nvidia-smi query.
pub fn gpu_card_name(index: i64) -> Option<String> {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args(["-i", &index.to_string(), "--query-gpu=name", "--format=csv,noheader"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let name = if out.status.success() && !s.trim().is_empty() { s.trim().lines().next().unwrap_or("").trim().to_string() } else { String::new() };
    let name = name.strip_prefix("NVIDIA GeForce ").or_else(|| name.strip_prefix("NVIDIA ")).unwrap_or(&name).to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// vision_item: the startup item for image input; None when it is off.
pub fn vision_item(vision: Option<&Map<String, Value>>, cfg: &Map<String, Value>) -> Option<String> {
    let v = vision.filter(|v| !v.is_empty())?;
    let mode = v.get("mode").filter(|m| crate::py::truthy(m)).map(serve_text::pyjson::py_str).unwrap_or_else(|| "cpu".into());
    if let Some(g) = vision_card(&mode) {
        return Some(format!("vision=gpu{g}"));
    }
    if v.get("mode").and_then(|m| m.as_str()) == Some("dynamic") {
        let l = gpu_list(cfg);
        return Some(format!("vision=gpu{}", l.first().map(|c| c.to_string()).unwrap_or_else(|| "0".into())));
    }
    Some(vision_item_rest(v))
}

fn vision_item_rest(_v: &Map<String, Value>) -> String {
    "vision=cpu".into()
}

/// model_plan: the engine's choice for the gpu setting (`basalt --gpu-plan`); Err with the engine's reason.
pub fn model_plan(engine_bin: &str, path: &str, value: Option<&Value>, words: &[String]) -> Result<Map<String, Value>, String> {
    let text = gpu_text(value);
    let mut cmd = Command::new(engine_bin);
    cmd.args(["--model", path]).args(words).arg("--gpu-plan");
    if !text.is_empty() {
        cmd.args(["--gpu", &text]);
    }
    let fail = || format!("{engine_bin} could not check this model's flags");
    let out = run_timeout(cmd, 60).ok_or_else(fail)?;
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let errors: Vec<String> = stderr
        .lines()
        .filter_map(|l| l.split_once("error: ").map(|(_, r)| r.trim().to_string()))
        .collect();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let plan = stdout
        .trim()
        .lines()
        .last()
        .and_then(|l| serve_text::pyjson::loads(l).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    if let Some(e) = plan.get("error").filter(|e| crate::py::truthy(e)) {
        return Err(serve_text::pyjson::py_str(e));
    }
    if !out.status.success() || plan.is_empty() {
        if !errors.is_empty() {
            return Err(errors.join("; "));
        }
        return Err(format!(
            "{engine_bin} could not check this model's flags (exit code {})",
            out.status.code().map(|c| c.to_string()).unwrap_or_else(|| "None".into())
        ));
    }
    Ok(plan)
}

/// flag_refused: whether the model in `path` has no such flag; `group` is the flag and its values.
pub fn flag_refused(engine_bin: &str, path: &str, group: &[String]) -> bool {
    let mut cmd = Command::new(engine_bin);
    cmd.args(["--model", path]).args(group).arg("--gpu-plan");
    match run_timeout(cmd, 60) {
        Some(out) => String::from_utf8_lossy(&out.stderr).contains(&format!("unknown argument: {}", group[0])),
        None => false,
    }
}

fn run_timeout(mut cmd: Command, secs: u64) -> Option<std::process::Output> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let child = cmd.spawn().ok()?;
    let (tx, rx) = mpsc::channel();
    let id = child.id();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(r) => r.ok(),
        Err(_) => {
            kill_pid(id);
            None
        }
    }
}

fn kill_pid(pid: u32) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill").args(["/F", "/PID", &pid.to_string()]).output();
    }
    #[cfg(unix)]
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

#[allow(dead_code)]
fn _t(_: BTreeMap<u8, u8>) {}
