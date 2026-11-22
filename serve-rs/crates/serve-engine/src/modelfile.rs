//! serve/engine/modelfile.py: engine settings, tokenizer and vision inputs for a .basalt model file. Nothing is
//! unpacked: the tokenizer is read from its sections into memory, the vision programs open the projector in the file.
//!
//! Deviations, by crate layout: Python asks `serve.models.dialect_for(arch).in_engine_vision`; serve-models lives in
//! serve-api, above this crate, so model_config takes that answer as a function of the arch. The config's
//! "tokenizer" (file name -> bytes) is not JSON: it comes back beside the map, in ModelConfig::tokenizer.
use crate::model_file::{hex, os_error, MfError, ModelFile};
use crate::pyfmt;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

pub const VISION_EXE: &str = "basalt-vision";
/// llama.cpp's sampling keys in the GGUF metadata -> the request field each one defaults.
pub const GGUF_SAMPLING: [(&str, &str); 6] = [("general.sampling.temp", "temperature"), ("general.sampling.top_p", "top_p"),
                                               ("general.sampling.top_k", "top_k"), ("general.sampling.min_p", "min_p"),
                                               ("general.sampling.penalty_repeat", "repetition_penalty"),
                                               ("general.sampling.penalty_last_n", "penalty_last_n")];
pub const TOKENIZER_SECTIONS: [&str; 5] = ["vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja"];

/// `Path.home()`.
pub fn home() -> PathBuf {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty());
    if cfg!(windows) {
        if let Some(p) = var("USERPROFILE") {
            return PathBuf::from(p);
        }
        let drive = var("HOMEDRIVE").map(PathBuf::from).unwrap_or_default();
        return drive.join(var("HOMEPATH").unwrap_or_default());
    }
    var("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

pub fn cache_root() -> PathBuf {
    home().join(".cache").join("basalt")
}

/// One cache directory per model, named by the meta section's sha256 prefix.
pub fn cache_dir_in(root: &Path, mf: &ModelFile) -> Result<PathBuf, MfError> {
    let (_, _, digest) = mf.entry("meta")?;
    Ok(root.join(&hex(&digest)[..16]))
}

pub fn cache_dir(mf: &ModelFile) -> Result<PathBuf, MfError> {
    cache_dir_in(&cache_root(), mf)
}

fn flag_value(words: &[String], flag: &str) -> Option<String> {
    let i = words.iter().rposition(|w| w == flag)?;
    words.get(i + 1).cloned()
}

/// `words` plus the default conversation directory (in the model cache folder `cdir`) when the disk budget is above
/// zero and no directory is named.
pub fn with_conversation_dir(words: &[String], cdir: &Path) -> Vec<String> {
    let budget = flag_value(words, "--conversation-cache-disk-mib");
    let on = budget.as_deref().is_some_and(|b| pyfmt::is_digits(b) && b.trim_start_matches('0') != "");
    let mut out = words.to_vec();
    if on && flag_value(words, "--conversation-cache-dir").filter(|d| !d.is_empty()).is_none() {
        out.push("--conversation-cache-dir".into());
        out.push(path_str(&cdir.join("conversations")));
    }
    out
}

/// A GGUF float32 as the shortest decimal that is the same float32 (0.949999988079071 -> 0.95).
fn short(v: &Value) -> Value {
    let Some(x) = v.as_f64().filter(|_| v.is_f64()) else { return v.clone() };
    let bits = (x as f32).to_bits();
    for digits in 1..10 {
        if let Ok(s) = pyfmt::fmt_g_prec(x, digits).parse::<f64>() {
            if (s as f32).to_bits() == bits {
                return Value::from(s);
            }
        }
    }
    v.clone()
}

/// The model's sampling defaults: the general.sampling.* keys of meta.gguf_kv, by request field.
pub fn sampling_defaults(meta: &Map<String, Value>) -> Map<String, Value> {
    let empty = Map::new();
    let kv = meta.get("gguf_kv").and_then(Value::as_object).unwrap_or(&empty);
    let mut out = Map::new();
    for (key, field) in GGUF_SAMPLING {
        if let Some(v) = kv.get(key).filter(|v| !v.is_null()) {
            out.insert(field.into(), short(v));
        }
    }
    out
}

/// The `tokenizer/*` sections the server reads, by file name (those a pack lacks are absent).
pub fn tokenizer_files(mf: &ModelFile) -> Result<Vec<(String, Vec<u8>)>, MfError> {
    if !mf.has("tokenizer/vocab.json") {
        return Err(MfError::Value(format!("{}: no tokenizer sections", mf.path)));
    }
    let mut out = Vec::new();
    for n in TOKENIZER_SECTIONS {
        let s = format!("tokenizer/{n}");
        if mf.has(&s) {
            out.push((n.to_string(), mf.read(&s)?));
        }
    }
    Ok(out)
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// `os.path.abspath(p)`.
pub fn abspath(p: &str) -> String {
    #[cfg(windows)]
    {
        std::path::absolute(p).map(|x| path_str(&x)).unwrap_or_else(|_| p.to_string())
    }
    #[cfg(not(windows))]
    {
        let joined = if p.starts_with('/') {
            PathBuf::from(p)
        } else {
            std::env::current_dir().unwrap_or_default().join(p)
        };
        let mut parts: Vec<String> = Vec::new();
        for c in path_str(&joined).split('/') {
            match c {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                c => parts.push(c.to_string()),
            }
        }
        let lead = if path_str(&joined).starts_with("//") && !path_str(&joined).starts_with("///") { "//" } else { "/" };
        format!("{lead}{}", parts.join("/"))
    }
}

fn strip_verbatim(p: PathBuf) -> PathBuf {
    let s = path_str(&p);
    if let Some(r) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{r}"));
    }
    if let Some(r) = s.strip_prefix(r"\\?\") {
        return PathBuf::from(r);
    }
    p
}

/// `Path(p).resolve()`: links followed as far as the path exists, the rest appended.
pub fn resolve(p: &str) -> PathBuf {
    let abs = PathBuf::from(abspath(p));
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = abs.clone();
    loop {
        if let Ok(c) = std::fs::canonicalize(&cur) {
            let mut out = strip_verbatim(c);
            for t in tail.iter().rev() {
                out.push(t);
            }
            return out;
        }
        match (cur.file_name().map(|x| x.to_os_string()), cur.parent().map(Path::to_path_buf)) {
            (Some(n), Some(parent)) => {
                tail.push(n);
                cur = parent;
            }
            _ => return abs,
        }
    }
}

/// The vision sidecar's config, which names the model file (the sidecar opens `vision/mmproj` inside it); a Value error
/// names what is missing.
pub fn vision_config(mf: &ModelFile, engine_bin: &str, sidecar: bool) -> Result<Map<String, Value>, MfError> {
    let engine = resolve(engine_bin);
    let suffix = engine.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    let side = engine.parent().map(Path::to_path_buf).unwrap_or_default().join(format!("{VISION_EXE}{suffix}"));
    if !mf.has("vision/mmproj") {
        return Err(MfError::Value("this model file has no vision projector (vision/mmproj); drop --vision".into()));
    }
    if sidecar && !side.is_file() {
        return Err(MfError::Value(format!("{} not found (build the vision sidecar), or drop --vision", path_str(&side))));
    }
    let mut m = Map::new();
    m.insert("vision_bin".into(), path_str(&side).into());
    m.insert("model".into(), abspath(&mf.path).into());
    Ok(m)
}

/// --vision as given: off (False), on (True) or a mode: a GPU number, "cpu" or "dynamic".
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum VisionArg {
    #[default]
    Off,
    On,
    Mode(String),
}

impl VisionArg {
    fn truthy(&self) -> bool {
        match self {
            VisionArg::Off => false,
            VisionArg::On => true,
            VisionArg::Mode(s) => !s.is_empty(),
        }
    }

    fn is(&self, s: &str) -> bool {
        matches!(self, VisionArg::Mode(m) if m == s)
    }

    fn to_value(&self) -> Value {
        match self {
            VisionArg::Off => false.into(),
            VisionArg::On => true.into(),
            VisionArg::Mode(s) => s.clone().into(),
        }
    }

    fn text(&self) -> String {
        match self {
            VisionArg::Off => "False".into(),
            VisionArg::On => "True".into(),
            VisionArg::Mode(s) => s.clone(),
        }
    }
}

/// One model the server runs. `engine_words` are its engine flags as given; `gpu` as --gpu.
#[derive(Debug, Clone, Default)]
pub struct ModelSpec {
    pub path: String,
    pub alias: Option<String>,
    pub gpu: Value,
    pub vision: VisionArg,
    pub engine_words: Vec<String>,
}

/// model_config's dict (key order as Python's, without "tokenizer") and the tokenizer files it read.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub cfg: Map<String, Value>,
    pub tokenizer: Vec<(String, Vec<u8>)>,
}

/// The engine settings a model implies (see model_config_in), under the user's cache root.
pub fn model_config(spec: &ModelSpec, engine_bin: &str, in_engine_vision: &dyn Fn(Option<&str>) -> bool)
                    -> Result<ModelConfig, MfError> {
    model_config_in(spec, engine_bin, in_engine_vision, &cache_root())
}

/// The engine settings a model implies; its `engine_words` follow its arguments as given, its `vision` adds the
/// sidecar or, "dynamic", the in-engine encoder. `in_engine_vision(arch)` is dialect_for(arch).in_engine_vision.
pub fn model_config_in(spec: &ModelSpec, engine_bin: &str, in_engine_vision: &dyn Fn(Option<&str>) -> bool, root: &Path)
                       -> Result<ModelConfig, MfError> {
    let vision = &spec.vision;
    let mf = ModelFile::open(&spec.path)?;
    let raw = mf.read("meta")?;
    let meta_v: Value = serde_json::from_slice(&raw).map_err(|e| MfError::Value(e.to_string()))?;
    let meta = meta_v.as_object().cloned().ok_or_else(|| MfError::Value("meta is not a JSON object".into()))?;
    let path = abspath(&spec.path);
    let cdir = cache_dir_in(root, &mf)?;
    let words = with_conversation_dir(&spec.engine_words, &cdir);
    let mut args = vec!["--model".to_string(), path];   // settings come from the engine defaults and the server
    args.extend(words);
    std::fs::create_dir_all(&cdir).map_err(|e| os_error(&e, &path_str(&cdir)))?;
    let name = match (&spec.alias, meta.get("name")) {
        (Some(a), _) if !a.is_empty() => Value::from(a.clone()),
        (_, Some(n)) if truthy(n) => n.clone(),
        _ => "basalt".into(),
    };
    let arch = meta.get("gguf_kv").and_then(|k| k.get("general.architecture")).cloned().unwrap_or(Value::Null);
    let mut cfg = Map::new();
    cfg.insert("engine_bin".into(), abspath(engine_bin).into());
    cfg.insert("args".into(), Value::Null);
    cfg.insert("model_name".into(), name);
    cfg.insert("sampling".into(), Value::Object(sampling_defaults(&meta)));
    cfg.insert("arch".into(), arch.clone());
    cfg.insert("log".into(), path_str(&cdir.join("engine.log")).into());
    let tokenizer = tokenizer_files(&mf)?;
    let finish = |mut cfg: Map<String, Value>, args: Vec<String>| {
        cfg.insert("args".into(), args.into());
        Ok(ModelConfig { cfg, tokenizer: tokenizer.clone() })
    };
    if vision.truthy() && in_engine_vision(arch.as_str()) {
        if !vision.is("dynamic") {
            return Err(MfError::Value(format!("this model's images are encoded in the engine: --vision dynamic, not {}",
                                              vision.text())));
        }
        if !mf.has("vision/mmproj") {
            return Err(MfError::Value("this model file has no vision projector (vision/mmproj); drop --vision".into()));
        }
        args.push("--vision".into());
        let mut v = Map::new();
        v.insert("mode".into(), "dynamic".into());
        v.insert("gpu".into(), true.into());
        cfg.insert("vision".into(), Value::Object(v));
        return finish(cfg, args);
    }
    if vision.truthy() {
        let mut v = vision_config(&mf, engine_bin, !vision.is("dynamic"))?;
        v.insert("mode".into(), vision.to_value());
        v.insert("gpu".into(), (!vision.is("cpu")).into());
        cfg.insert("vision".into(), Value::Object(v));
        // dynamic: the engine encodes, opening the projector in the model file; else the sidecar's embeddings come in
        if vision.is("dynamic") && !args.iter().any(|a| a == "--vision") {
            args.push("--vision".into());
        } else if !vision.is("dynamic") && !args.iter().any(|a| a == "--vision-external") {
            args.push("--vision-external".into());
        }
    }
    finish(cfg, args)
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_file::ModelWriter;
    use serde_json::json;

    const TOKENIZER: [(&str, &[u8]); 4] = [("vocab.json", b"{\"a\": 0}"), ("merges.txt", b"a a"), ("token_type.json", b"[1]"),
                                           ("chat_template.jinja", b"{{ messages }}")];

    fn meta() -> Value {
        json!({"name": "tiny-basalt",
               "gguf_kv": {"general.sampling.temp": 0.699999988079071, "general.sampling.top_p": 0.8999999761581421,
                           "general.sampling.top_k": 20, "general.sampling.xtc_threshold": 0.1}})
    }

    struct Tmp {
        dir: PathBuf,
        engine_bin: PathBuf,
    }

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let dir = std::env::temp_dir().join(format!("serve-modelfile-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("bin")).unwrap();
            let engine_bin = dir.join("bin").join("basalt");
            std::fs::write(&engine_bin, "").unwrap();
            Tmp { dir, engine_bin }
        }

        fn build(&self, vision: bool, meta: &Value) -> String {
            let p = path_str(&self.dir.join("tiny.basalt"));
            let mut w = ModelWriter::create(&p).unwrap();
            w.add_bytes("meta", meta.to_string().as_bytes()).unwrap();
            for (n, b) in TOKENIZER {
                w.add_bytes(&format!("tokenizer/{n}"), b).unwrap();
            }
            if vision {
                w.add_bytes("vision/mmproj", &b"MMPROJ".repeat(100)).unwrap();
                w.add_bytes("vision/text_vocab", b"VOCAB").unwrap();
            }
            w.close().unwrap();
            p
        }

        fn cfg(&self, spec: ModelSpec) -> Result<ModelConfig, MfError> {
            self.cfg_bin(spec, &path_str(&self.engine_bin))
        }

        fn cfg_bin(&self, spec: ModelSpec, bin: &str) -> Result<ModelConfig, MfError> {
            model_config_in(&spec, bin, &|_| false, &self.dir.join("cache"))
        }

        fn spec(&self, path: &str) -> ModelSpec {
            ModelSpec { path: path.to_string(), ..Default::default() }
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn args(c: &ModelConfig) -> Vec<String> {
        serde_json::from_value(c.cfg["args"].clone()).unwrap()
    }

    #[test]
    fn config() {
        let t = Tmp::new("config");
        let path = t.build(false, &meta());
        let c = t.cfg(t.spec(&path)).unwrap();
        assert_eq!(c.cfg["engine_bin"], path_str(&t.engine_bin));
        assert_eq!(args(&c), ["--model", &path]);
        assert_eq!(c.cfg["model_name"], "tiny-basalt");
        assert_eq!(c.cfg["sampling"], json!({"temperature": 0.7, "top_p": 0.9, "top_k": 20}));
        assert_eq!(c.cfg.keys().collect::<Vec<_>>(), ["engine_bin", "args", "model_name", "sampling", "arch", "log"]);
        let key = &hex(&ModelFile::open(&path).unwrap().entry("meta").unwrap().2)[..16];
        assert_eq!(PathBuf::from(c.cfg["log"].as_str().unwrap()), t.dir.join("cache").join(key).join("engine.log"));
        assert_eq!(c.tokenizer, TOKENIZER.iter().map(|(n, b)| (n.to_string(), b.to_vec())).collect::<Vec<_>>());
        assert_eq!(std::fs::read_dir(t.dir.join("cache").join(key)).unwrap().count(), 0);
        let conv = path_str(&t.dir.join("cache").join(key).join("conversations"));
        let a = |w: &[&str]| args(&t.cfg(ModelSpec { engine_words: w.iter().map(|s| s.to_string()).collect(),
                                                     ..t.spec(&path) }).unwrap());
        assert_eq!(a(&["--conversation-cache-disk-mib", "0"]), ["--model", &path, "--conversation-cache-disk-mib", "0"]);
        assert_eq!(a(&["--conversation-cache-dir", "/x"]), ["--model", &path, "--conversation-cache-dir", "/x"]);
        assert_eq!(a(&["--conversation-cache-disk-mib", "32768"]),
                   ["--model", &path, "--conversation-cache-disk-mib", "32768", "--conversation-cache-dir", &conv]);
        assert_eq!(a(&["--conversation-cache-disk-mib", "5", "--conversation-cache-dir", "/x"]),
                   ["--model", &path, "--conversation-cache-disk-mib", "5", "--conversation-cache-dir", "/x"]);
        assert_eq!(a(&["--spec", "0", "--peer-exclusive"]), ["--model", &path, "--spec", "0", "--peer-exclusive"]);
        let c = t.cfg(ModelSpec { alias: Some("mine".into()), ..t.spec(&path) }).unwrap();
        assert_eq!(c.cfg["model_name"], "mine");
    }

    #[test]
    fn bad_files() {
        let t = Tmp::new("bad");
        let bare = path_str(&t.dir.join("bare.basalt"));
        let mut w = ModelWriter::create(&bare).unwrap();
        w.add_bytes("meta", b"{}").unwrap();
        w.close().unwrap();
        assert!(matches!(t.cfg(t.spec(&bare)), Err(MfError::Value(m)) if m.ends_with(": no tokenizer sections")));
        std::fs::write(t.dir.join("junk"), vec![b'x'; 5000]).unwrap();
        assert!(matches!(t.cfg(t.spec(&path_str(&t.dir.join("junk")))), Err(MfError::Value(_))));
        let m2 = json!({"name": "x", "engine_args": ["--spec", "7"], "gguf_kv": meta()["gguf_kv"]});
        let path = t.build(false, &m2);
        assert_eq!(args(&t.cfg(t.spec(&path)).unwrap()), ["--model", &path]);
    }

    #[test]
    fn vision() {
        let t = Tmp::new("vision");
        let path = t.build(true, &meta());
        let side = t.engine_bin.parent().unwrap().join("basalt-vision");
        let v = |mode: VisionArg| t.cfg(ModelSpec { vision: mode, ..t.spec(&path) });
        assert!(matches!(v(VisionArg::Mode("1".into())), Err(MfError::Value(m)) if m.contains("basalt-vision")));
        let c = v(VisionArg::Mode("dynamic".into())).unwrap();
        assert_eq!(args(&c), ["--model", &path, "--vision"]);
        std::fs::write(&side, "").unwrap();
        let c = v(VisionArg::On).unwrap();
        assert!(args(&c).contains(&"--vision-external".to_string()) && !args(&c).contains(&"--vision".to_string()));
        assert_eq!(c.cfg["vision"]["vision_bin"], path_str(&resolve(&path_str(&side))));
        assert_eq!(c.cfg["vision"]["model"], path);
        assert_eq!(c.cfg["vision"]["mode"], true);
        assert_eq!(c.cfg["vision"].as_object().unwrap().keys().collect::<Vec<_>>(), ["vision_bin", "model", "mode", "gpu"]);
        assert_eq!(v(VisionArg::Mode("1".into())).unwrap().cfg["vision"]["gpu"], true);
        assert_eq!(v(VisionArg::Mode("cpu".into())).unwrap().cfg["vision"]["gpu"], false);
        assert_eq!(v(VisionArg::Mode("0".into())).unwrap().cfg["vision"]["mode"], "0");
        let c = v(VisionArg::Off).unwrap();
        assert!(c.cfg.get("vision").is_none() && !args(&c).iter().any(|a| a.starts_with("--vision")));
        // the engine bin's suffix
        let exe = t.engine_bin.with_file_name("basalt.exe");
        std::fs::write(&exe, "").unwrap();
        std::fs::write(t.engine_bin.parent().unwrap().join("basalt-vision.exe"), "").unwrap();
        let c = t.cfg_bin(ModelSpec { vision: VisionArg::Mode("cpu".into()), ..t.spec(&path) }, &path_str(&exe)).unwrap();
        assert!(c.cfg["vision"]["vision_bin"].as_str().unwrap().ends_with("basalt-vision.exe"));
        // an in-engine encoder takes only dynamic
        let ie = |mode: VisionArg| model_config_in(&ModelSpec { vision: mode, ..t.spec(&path) }, &path_str(&t.engine_bin),
                                                   &|_| true, &t.dir.join("cache"));
        assert_eq!(ie(VisionArg::Mode("1".into())).unwrap_err().to_string(),
                   "this model's images are encoded in the engine: --vision dynamic, not 1");
        let c = ie(VisionArg::Mode("dynamic".into())).unwrap();
        assert_eq!(c.cfg["vision"], json!({"mode": "dynamic", "gpu": true}));
        assert_eq!(args(&c), ["--model", &path, "--vision"]);
        let plain = t.build(false, &meta());
        std::fs::write(&side, "").unwrap();
        let e = t.cfg(ModelSpec { vision: VisionArg::Mode("cpu".into()), ..t.spec(&plain) }).unwrap_err();
        assert!(e.to_string().contains("projector"));
    }

    #[test]
    fn shortest_float32() {
        assert_eq!(short(&json!(0.949999988079071)), json!(0.95));
        assert_eq!(short(&json!(20)), json!(20));
    }
}
