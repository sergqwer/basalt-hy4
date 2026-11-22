//! serve/server.py: OpenAI, Anthropic and Responses endpoints over the engine, one model loaded at a time.
//!     basalt serve --model FILE.basalt [--alias NAME] [--host H] [--port P] [--gpu 0,1] [--api-key KEY] ... --server rust
//!     basalt-serve --config config.json | --mock-engine --port 8080
mod argparse;
mod config;
mod signals;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use serde_json::{json, Map, Value};
use serve_api::service::Service;
use serve_engine::errors::EngineErr;
use serve_engine::flags::flag_value;
use serve_engine::iface::Engine;
use serve_engine::mock::MockEngine;
use serve_engine::process::{self as engine_process, ProcessEngine};
use serve_engine::util::console::{tagged, LEGEND};
use serve_engine::util::logbuf::LogBuffer;
use serve_engine::util::loglevel::{LevelReader, ERR, WARN};
use serve_engine::util::stdout;
use serve_http::handler::{Handler, ServerConf};
use serve_http::http;
use serve_http::router::Router;
use serve_http::security::{allowed_hosts_of, api_key_from_file, api_key_of, exposed_without_key, host_names_for,
                           lan_addresses, origins_of, LOOPBACK_NAMES};
use serve_text::{ByteTokenizer, Tokenizer};

use config::{Args, Setting};

const SWAP_START_TRIES: u32 = 2;

/// Print a startup error and exit with status 2.
fn fail(msg: &str) -> ! {
    stdout::err(&format!("[server] error: {msg}"));
    std::process::exit(2)
}

fn exit_with(e: argparse::Exit) -> ! {
    if !e.stdout.is_empty() {
        stdout::write_raw_stdout(&e.stdout);
        stdout::flush_stdout();
    }
    if !e.stderr.is_empty() {
        stdout::write_raw_stderr(&e.stderr);
    }
    std::process::exit(e.code)
}

fn total_ram_bytes() -> Option<u64> {
    #[cfg(windows)]
    unsafe {
        #[repr(C)]
        struct MemStatus {
            len: u32,
            load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page: u64,
            avail_page: u64,
            total_virt: u64,
            avail_virt: u64,
            avail_ext: u64,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GlobalMemoryStatusEx(b: *mut MemStatus) -> i32;
        }
        let mut m: MemStatus = std::mem::zeroed();
        m.len = std::mem::size_of::<MemStatus>() as u32;
        if GlobalMemoryStatusEx(&mut m) != 0 {
            return Some(m.total_phys);
        }
        None
    }
    #[cfg(unix)]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}

/// Warn at start when less than ~6 GB of RAM is left beside the expert arena.
fn warn_tight_ram(arena_mib: Option<&Value>) {
    let Some(arena) = arena_mib.and_then(|v| v.as_i64()).filter(|v| *v > 0) else { return };
    let Some(total) = total_ram_bytes() else { return };
    let gib = total as f64 / (1u64 << 30) as f64;
    let left = gib - arena as f64 / 1024.0;
    if left < 6.0 {
        stdout::out(&tagged("warning", &format!("low RAM: {} of {} GB left after experts",
                                                serve_text::pyjson::py_fixed(left, 1), serve_text::pyjson::py_fixed(gib, 0))));
    }
}

fn grammar_entry(on: bool, cli: &[String], used: &[String]) -> Option<(bool, String)> {
    let names = ["--grammar", "--no-grammar"];
    let origin = if cli.iter().any(|w| names.contains(&w.as_str())) {
        "cli"
    } else if used.iter().any(|w| names.contains(&w.as_str())) {
        "config"
    } else {
        ""
    };
    if origin.is_empty() { None } else { Some((on, origin.into())) }
}

fn model_file_problems(settings: &[Setting]) -> Vec<String> {
    let mut out = Vec::new();
    for (i, (m, ..)) in settings.iter().enumerate() {
        let (Some(model), Some(_bin)) = (m.st("model"), m.st("engine_bin")) else { continue };
        if m.b("mock_engine") && m.st("config").is_some() {
            continue;
        }
        let (who, where_) = if let Some(cfgp) = m.st("config") {
            let who = match m.st("alias") {
                Some(a) => format!("model '{a}'"),
                None => format!("models[{i}]"),
            };
            let name = Path::new(&cfgp).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            (who, format!(" (the \"model\" key in {name})"))
        } else {
            ("model".to_string(), " (--model)".to_string())
        };
        let p = Path::new(&model);
        let why = if p.is_dir() {
            format!("{model} is a directory, not a model file")
        } else if !p.exists() {
            format!("file not found: {model}")
        } else {
            match std::fs::File::open(p) {
                Ok(_) => continue,
                Err(e) => format!("can't read {model}: {}", strerror(&e)),
            }
        };
        out.push(format!("{who}: {why}{where_}"));
    }
    out
}

fn strerror(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::PermissionDenied => "Permission denied".into(),
        std::io::ErrorKind::NotFound => "No such file or directory".into(),
        _ => e.to_string(),
    }
}

/// What one configured model's file and flags imply (check_model).
fn check_model(a: &mut Args, file_engine: &[String], cli_engine: &mut Vec<String>, used_argv: &[String],
               known: Option<&BTreeSet<String>>) -> Map<String, Value> {
    let mut cfg = Map::new();
    let mut lines: Vec<String> = Vec::new();
    if serve_api::vision::check_gpu(a.get("gpu")).is_err() {
        fail(&format!("--gpu {}: expected numbers like 0 or 0,1, or auto", serve_api::vision::gpu_text(a.get("gpu"))));
    }
    let mock = a.b("mock_engine");
    if !mock && !(a.st("model").is_some() && a.st("engine_bin").is_some()) {
        fail("the server needs --model and the engine binary: run it as basalt serve --config FILE (or --mock-engine)");
    }
    if (!file_engine.is_empty() || !cli_engine.is_empty()) && a.st("model").is_none() {
        fail("engine flags need --model");
    }
    let pack = a.st("model").is_some() && !(mock && a.st("config").is_some());
    let mut tokenizer: Option<Vec<(String, Vec<u8>)>> = None;
    if pack {
        let Some(engine_bin) = a.st("engine_bin") else { fail("--model needs --engine-bin BINARY") };
        let model = a.st("model").unwrap();
        if let Err(e) = config::check_engine_flags(file_engine, known, &format!("--config {}", a.s("config").unwrap_or_else(|| "None".into())))
            .and_then(|_| config::check_engine_flags(cli_engine, known, "command line")) {
            fail(&e);
        }
        let mode = a.s("vision").unwrap_or_else(|| "off".into());
        let vision = if mock || mode == "off" { serve_engine::modelfile::VisionArg::Off } else { serve_engine::modelfile::VisionArg::Mode(mode.clone()) };
        let words: Vec<String> = file_engine.iter().chain(cli_engine.iter()).cloned().collect();
        let spec = serve_engine::modelfile::ModelSpec { path: model.clone(), alias: a.st("alias"), gpu: a.get("gpu").clone(),
                                                        vision, engine_words: words.clone() };
        match serve_engine::modelfile::model_config(&spec, &engine_bin, &|arch| serve_api::models::dialect_for(arch).in_engine_vision()) {
            Ok(mc) => {
                for (k, v) in mc.cfg {
                    cfg.insert(k, v);
                }
                tokenizer = Some(mc.tokenizer);
            }
            Err(e) => fail(&format!("{model}: {e}")),
        }
        if !mock {
            let plan = match serve_api::vision::model_plan(&engine_bin, &model, spec.gpu.clone(), &words) {
                Ok(p) => p,
                Err(e) => fail(&format!("{model}: {e}")),
            };
            if let Some(card) = serve_api::vision::vision_card(&mode) {
                if serve_api::vision::gpu_card_name(card).is_none() {
                    fail(&format!("vision: GPU {card} is not found by nvidia-smi"));
                }
            }
            if let Some(Value::Array(notes)) = plan.get("notes") {
                for n in notes {
                    lines.push(format!("[server] {}", serve_text::pyjson::py_str(n)));
                }
            }
            if let Some(w) = plan.get("warning").filter(|w| serve_text::pyjson::truthy(w)) {
                lines.push(format!("[server] warning: {}", serve_text::pyjson::py_str(w)));
            }
            cfg.insert("gpu".into(), plan.get("devices").cloned().unwrap_or(Value::Null));
            cfg.insert("spec".into(), plan.get("spec").cloned().unwrap_or(Value::Null));
            if a.split_default {
                if let Some(mc) = plan.get("max_context").and_then(|v| v.as_i64()).filter(|v| *v != 0) {
                    a.shared_context = Some(mc);
                    let part = vec!["--max-context".to_string(), config::slot_share(mc, a.i("parallel").unwrap_or(1)).to_string()];
                    if let Some(Value::Array(args)) = cfg.get_mut("args") {
                        args.extend(part.iter().map(|s| json!(s)));
                    }
                    cli_engine.extend(part);
                }
            }
            let cards = match plan.get("cards") {
                Some(Value::Array(c)) => c.clone(),
                _ => vec![],
            };
            cfg.insert("gpu_smi".into(), Value::Array(cards.iter().map(|c| c.get("smi").cloned().unwrap_or(json!(-1))).collect()));
            if let Some(Value::Array(args)) = cfg.get_mut("args") {
                args.push(json!("--gpu"));
                args.push(json!((0..cards.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(",")));
            }
        }
    }
    for flag in &a.skipped {
        lines.push(tagged("engine", &format!("{} from the top level: this model has no such flag, skipped", &flag[2..])));
    }
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let items = if pack && !mock {
        let shared = a.shared_context.map(|s| (s, a.i("parallel").unwrap_or(0)));
        config::start_items(file_engine, cli_engine, shared, grammar_entry(a.b("grammar"), &argv, used_argv), a.b("debug"))
    } else {
        vec![]
    };
    let spec = cfg.get("spec").and_then(|v| v.as_i64());
    let items = config::with_spec(items, spec);
    cfg.insert("items".into(), json!(items));
    cfg.insert("start_lines".into(), json!(lines));
    if let (Some(alias), false) = (a.st("alias"), pack) {
        cfg.insert("model_name".into(), json!(alias));
    }
    if !cfg.get("model_name").map(serve_text::pyjson::truthy).unwrap_or(false) {
        if let Some(m) = a.st("model") {
            let name = Path::new(&m).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or(m);
            cfg.insert("model_name".into(), json!(name));
        }
    }
    if let Some(t) = tokenizer {
        let mut tm = Map::new();
        for (k, v) in t {
            tm.insert(k, Value::Array(v.into_iter().map(Value::from).collect()));
        }
        cfg.insert("__tokenizer".into(), Value::Object(tm));
    }
    cfg
}

fn drop_sidecar_vision(cfg: &mut Map<String, Value>) {
    let mode = cfg.get("vision").and_then(|v| v.get("mode")).and_then(|m| m.as_str()).map(|s| s.to_string());
    if !cfg.get("vision").map(serve_text::pyjson::truthy).unwrap_or(false) || mode.as_deref() == Some("dynamic") {
        return;
    }
    stdout::out(&tagged("warning", "lazy loading drops a sidecar image encoder, vision off; --vision dynamic stays on"));
    cfg.shift_remove("vision");
}

/// The tail callback that repeats the engine log's warnings and errors once the engine is ready.
fn engine_echo(engine: Arc<ProcessEngine>) -> impl Fn(&str, u64) + Send + Sync {
    let state = Mutex::new((LevelReader::new(), false));
    move |text: &str, end_pos: u64| {
        let mut g = state.lock().unwrap();
        let lvl = g.0.read(text);
        let echo;
        if lvl == ERR {
            g.1 = true;
            echo = true;
        } else if g.1 && (text.starts_with("  ") || text.starts_with('\t')) && !text.trim().is_empty() {
            echo = true;
        } else {
            g.1 = false;
            echo = lvl == WARN;
        }
        if echo && engine.known_ctx() > 0 && !engine.loading() && end_pos > engine.log_ready_pos() {
            stdout::write_raw_stdout(&format!("{}\n", text.trim_end()));
            stdout::flush_stdout();
        }
    }
}

fn tokenizer_of(files: &Map<String, Value>) -> Option<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    for (k, v) in files {
        let Value::Array(a) = v else { return None };
        out.push((k.clone(), a.iter().map(|x| x.as_u64().unwrap_or(0) as u8).collect()));
    }
    Some(out)
}

fn file_of<'a>(files: &'a [(String, Vec<u8>)], name: &str) -> Option<&'a Vec<u8>> {
    files.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

/// The Service of one checked model, unloaded.
fn build_model(a: &Args, cfg: &mut Map<String, Value>, logs: &Arc<LogBuffer>, multi: bool, known: Option<&BTreeSet<String>>,
               root: &Path) -> Arc<Service> {
    let files: Vec<(String, Vec<u8>)> = match cfg.get("__tokenizer").and_then(|v| v.as_object()).and_then(tokenizer_of) {
        Some(f) => f,
        None => {
            let tdir = PathBuf::from(a.s("tokenizer").unwrap_or_default());
            serve_engine::modelfile::TOKENIZER_SECTIONS.iter()
                .filter_map(|n| std::fs::read(tdir.join(n)).ok().map(|b| (n.to_string(), b)))
                .collect()
        }
    };
    let mock = a.b("mock_engine");
    if !mock && file_of(&files, "vocab.json").is_none() {
        fail(&format!("tokenizer missing: {}/vocab.json", a.s("tokenizer").unwrap_or_default()));
    }
    let tok: Arc<dyn Tokenizer> = if file_of(&files, "vocab.json").is_some() {
        match serve_text::BpeTokenizer::from_files(&files) {
            Ok(t) => Arc::new(t),
            Err(e) => fail(&e),
        }
    } else {
        Arc::new(ByteTokenizer)
    };
    let mut has_grammar = false;
    let engine: Arc<dyn Engine>;
    let mut vision = None;
    let sampling_defaults: Map<String, Value>;
    if !mock {
        let env = serve_api::vision::child_env(cfg);
        let mut sd = match config::sampling_defaults_from_config(cfg) {
            Ok(s) => s,
            Err(text) => {
                stdout::err(&text);
                std::process::exit(1);
            }
        };
        for (k, v) in &a.sampling_launch {
            sd.insert(k.clone(), v.clone());
        }
        sampling_defaults = sd;
        if a.b("lazy") && !multi {
            drop_sidecar_vision(cfg);
        }
        let shared: Vec<String> = cfg.get("shared_items").and_then(|v| v.as_array()).map(|v| v.iter().map(serve_text::pyjson::py_str).collect()).unwrap_or_default();
        let mut shown: Vec<String> = cfg.get("items").and_then(|v| v.as_array()).map(|v| v.iter().map(serve_text::pyjson::py_str).collect::<Vec<_>>()).unwrap_or_default()
            .into_iter().filter(|x| !shared.contains(x)).collect();
        shown.extend(config::sampling_items(&sampling_defaults));
        if let Some(vi) = serve_api::vision::vision_item(cfg.get("vision"), cfg) {
            shown.push(vi);
        }
        let name = cfg.get("model_name").map(serve_text::pyjson::py_str).unwrap_or_default();
        let width = cfg.get("name_width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        stdout::out(&config::model_line(&name, width, &shown));
        if let Some(Value::Array(ls)) = cfg.get("start_lines") {
            for l in ls {
                stdout::out(&serve_text::pyjson::py_str(l));
            }
        }
        if let Some(v) = cfg.get("vision").filter(|v| serve_text::pyjson::truthy(v)).cloned() {
            let mode = v.get("mode").and_then(|m| m.as_str()).unwrap_or("");
            vision = Some(if mode == "dynamic" {
                serve_api::vision::VisionAny::engine(&v)
            } else {
                serve_api::vision::VisionAny::sidecar(&v, cfg.get("log").and_then(|l| l.as_str()), serve_api::vision::vision_env(&env, mode), !multi)
            });
        }
        if a.b("lazy") && !multi {
            stdout::out(&tagged("model", "unloaded, the first request loads it"));
        }
        has_grammar = known.map(|k| k.contains("--grammar")).unwrap_or(false);
        if a.b("grammar") && !has_grammar {
            stdout::out(&tagged("warning", "grammar is on but this engine has no --grammar; requests go without one"));
        }
        let gwords = config::grammar_words(known);
        let eargs = serve_api::vision::engine_args(cfg);
        let cdir = flag_value(&eargs, "--conversation-cache-dir");
        let cmib = flag_value(&eargs, "--conversation-cache-disk-mib").unwrap_or_else(|| "0".into());
        if let Some(cd) = &cdir {
            if cmib != "0" && a.b("debug") {
                stdout::out(&tagged("disk", &format!("conversations={cd} disk-mib={cmib}")));
            }
        }
        let bin = cfg.get("engine_bin").map(serve_text::pyjson::py_str).unwrap_or_default();
        let log = cfg.get("log").and_then(|l| l.as_str()).map(|s| s.to_string());
        let mut all_args = eargs.clone();
        all_args.extend(gwords);
        let pe = match ProcessEngine::new(&bin, all_args, None, log.clone(), env, true) {
            Ok(e) => Arc::new(e),
            Err(e) => fail(&e.message()),
        };
        if let Some(l) = &log {
            stdout::out(&format!("[server] engine log: {l}"));
            let echo = engine_echo(pe.clone());
            let (logs2, path) = (logs.clone(), PathBuf::from(l));
            std::thread::spawn(move || {
                serve_engine::util::logbuf::tail_file(&logs2, &path, "eng", 0.25, Some(&echo),
                    Some(&|line: &str| serve_engine::progress::load_progress(line).is_some()), None);
            });
        } else {
            let logs2 = logs.clone();
            *engine_process::STDERR_SINK.lock().unwrap() = Some(Arc::new(move |line: &str| logs2.add("eng", line)));
        }
        if let Some(v) = &vision {
            v.attach(pe.clone());
        }
        engine = pe;
    } else {
        let scripts = {
            let s = a.list("script");
            if s.is_empty() { vec!["Thinking about it.</think>\n\nHello from the mock engine.".to_string()] } else { s }
        };
        engine = Arc::new(MockEngine::with_scripts(tok.clone(), scripts, a.i("parallel").unwrap_or(0).max(0) as usize));
        sampling_defaults = a.sampling_launch.clone();
    }
    let arch = cfg.get("arch").and_then(|v| v.as_str()).map(|s| s.to_string());
    let dialect = serve_api::models::dialect_for(arch.as_deref());
    let template = match file_of(&files, "chat_template.jinja") {
        Some(b) => String::from_utf8_lossy(b).into_owned(),
        None => std::fs::read_to_string(root.join("serve").join("chat_template.jinja")).unwrap_or_default(),
    };
    let model_name = cfg.get("model_name").map(serve_text::pyjson::py_str).unwrap_or_else(|| "basalt".into());
    let svc = Service::new(engine.clone(), tok, dialect.template(&template), &model_name, vision, sampling_defaults,
                           a.b("fit_max_tokens"), a.b("grammar"), mock || has_grammar, dialect,
                           a.i("video_max_tokens").unwrap_or(config::VIDEO_MAX_TOKENS));
    svc.set_logs(logs.clone());
    svc.set_idle_unload_s(a.f("idle_unload").unwrap_or(0.0));
    svc.set_idle_sleep_s(a.f("idle_sleep").unwrap_or(0.0));
    svc.set_min_free_vram_mib(a.i("min_free_vram_mib").unwrap_or(0));
    svc.set_reasoning_budget_tokens(a.i("reasoning_budget_tokens").unwrap_or(0));
    svc.set_answer_room(a.get("answer_room").clone());
    svc.set_reasoning_effort(a.s("reasoning_effort"));
    svc.set_thinking_loop_tokens(a.i("thinking_loop_tokens").unwrap_or(256));
    let gargs = if mock { vec![] } else { serve_api::vision::engine_args(cfg) };
    let gi = serve_api::vision::gpus_in_play(cfg, &gargs);
    svc.set_gpu_indices(gi);
    if multi {
        svc.set_start_tries(SWAP_START_TRIES);
    }
    svc
}

fn check_models(services: &[Arc<Service>]) {
    let mut seen: Vec<(String, usize)> = Vec::new();
    let mut folders: Vec<(String, String)> = Vec::new();
    for (i, svc) in services.iter().enumerate() {
        for name in svc.served_names() {
            match seen.iter().find(|(n, _)| *n == name) {
                Some((_, j)) if *j != i => fail(&format!("--config: {} names two models: give each its own alias",
                                                         serve_text::pyjson::py_str_repr(&name))),
                Some(_) => {}
                None => seen.push((name, i)),
            }
        }
        let args = svc.engine.spawn_args();
        let where_ = flag_value(&args, "--conversation-cache-dir");
        if let Some(w) = where_ {
            if flag_value(&args, "--conversation-cache-disk-mib").unwrap_or_else(|| "0".into()) != "0" {
                match folders.iter().find(|(f, _)| *f == w) {
                    Some((_, m)) if *m != svc.model => fail(&format!("--config: {} and {} share the conversation folder {}: give each its own", svc.model, m, w)),
                    Some(_) => {}
                    None => folders.push((w, svc.model.clone())),
                }
            }
        }
    }
}

fn open_browser(url: &str) {
    #[cfg(windows)]
    let _ = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
}

fn main() {
    let ap = config::build_parser();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let engine_bin = match ap.parse_known_args(&argv) {
        Ok((ns, _)) => ns.get("engine_bin").and_then(|v| v.as_str()).map(|s| s.to_string()),
        Err(e) => exit_with(e),
    };
    let refused_fn = engine_bin.clone().filter(|s| !s.is_empty()).map(|bin| {
        move |path: &str, group: &[String]| serve_api::vision::flag_refused(&bin, path, group)
    });
    let refused: Option<&dyn Fn(&str, &[String]) -> bool> = refused_fn.as_ref().map(|f| f as &dyn Fn(&str, &[String]) -> bool);
    let mut settings = match config::model_settings(&ap, &argv, refused) {
        Ok(s) => s,
        Err(config::ConfigErr::Exit(e)) => exit_with(e),
        Err(config::ConfigErr::Value(m)) => fail(&format!("--config: {m}")),
        Err(config::ConfigErr::System(m)) => {
            stdout::err(&m);
            std::process::exit(1)
        }
    };
    let a0 = settings[0].0.clone();
    let multi = settings.len() > 1;
    serve_api::service::set_debug(a0.b("debug"));
    engine_process::REQUEST_LINES.store(a0.b("request_lines"), std::sync::atomic::Ordering::SeqCst);
    let logs = Arc::new(LogBuffer::new());
    stdout::install(logs.clone());
    stdout::out(&tagged("legend", LEGEND));
    let problems = model_file_problems(&settings);
    for line in &problems {
        stdout::err(&format!("[server] error: {line}"));
    }
    if !problems.is_empty() {
        std::process::exit(2);
    }
    let known = a0.st("engine_bin").and_then(|b| config::engine_flags_of(&b));
    let mut cfgs: Vec<Map<String, Value>> = Vec::new();
    for (m, used, file_engine, cli_engine) in settings.iter_mut() {
        let c = check_model(m, file_engine, cli_engine, used, known.as_ref());
        cfgs.push(c);
    }
    let a = settings[0].0.clone();
    let hosts = a.list("image_url_host");
    let dirs = a.list("image_dir");
    if (!hosts.is_empty() || !dirs.is_empty()) && !a.b("mock_engine")
        && settings.iter().all(|m| m.0.s("vision").as_deref() == Some("off")) {
        fail("--image-url-host and --image-dir need --vision (image input is off)");
    }
    let image_sources = match serve_api::image_source::image_sources_of(&hosts, &dirs, a.b("image_remote_only")) {
        Ok(s) => s,
        Err(e) => fail(&e),
    };
    let host = a.s("host").unwrap_or_else(|| "127.0.0.1".into());
    let port = a.i("port").unwrap_or(8080);
    // fail fast if the port is taken
    match http::bind(&host, port as u16) {
        Ok(l) => drop(l),
        Err(_) => fail(&format!("port {port} is in use (--port N)")),
    }
    let width = cfgs.iter().map(|c| c.get("model_name").and_then(|v| v.as_str()).map(|s| s.chars().count()).unwrap_or(0)).max().unwrap_or(0) + 1;
    let items_of = |c: &Map<String, Value>| -> Vec<String> {
        c.get("items").and_then(|v| v.as_array()).map(|v| v.iter().map(serve_text::pyjson::py_str).collect()).unwrap_or_default()
    };
    let shared: Vec<String> = if multi {
        items_of(&cfgs[0]).into_iter().filter(|x| cfgs[1..].iter().all(|c| items_of(c).contains(x))).collect()
    } else {
        vec![]
    };
    let sleeps: Vec<Option<f64>> = settings.iter().map(|m| m.0.f("idle_sleep")).collect();
    let (line, shared_sleep) = config::shared_line(&shared, &sleeps);
    if !line.is_empty() {
        stdout::out(&tagged("engine", &line));
    }
    for c in cfgs.iter_mut() {
        c.insert("shared_items".into(), json!(shared));
        c.insert("name_width".into(), json!(width));
    }
    let root = config::root();
    let services: Vec<Arc<Service>> = settings.iter().zip(cfgs.iter_mut())
        .map(|(m, cfg)| build_model(&m.0, cfg, &logs, multi, known.as_ref(), &root)).collect();
    check_models(&services);
    let router = Router::new(services.clone());
    let svc = router.front.clone();
    if !image_sources.url_hosts_text().is_empty() || !image_sources.dirs.is_empty() || image_sources.remote_only {
        stdout::out(&tagged("images", &format!("from other machines{}: url-hosts={} dirs={}",
            if image_sources.remote_only { " and this one" } else { "" },
            if hosts.is_empty() { "-".to_string() } else { hosts.join(",") },
            if image_sources.dirs.is_empty() { "-".to_string() } else { image_sources.dirs.join(",") })));
    }
    let used_argv = &settings[0].1;
    let empty_key = used_argv.iter().enumerate().any(|(i, x)| {
        (x == "--api-key" && i + 1 < used_argv.len() && used_argv[i + 1].trim().is_empty()) || x.trim() == "--api-key="
    });
    if empty_key {
        stdout::err("[server] error: --api-key is empty (give a key, or leave it out)");
        std::process::exit(2);
    }
    let mpm = a.i("max_payload_mib").unwrap_or(256);
    if mpm < 1 {
        fail(&format!("--max-payload-mib {mpm} must be at least 1"));
    }
    let mut api_key_raw = a.s("api_key").unwrap_or_default();
    if let Some(f) = a.st("api_key_file") {
        if !api_key_raw.is_empty() {
            stdout::err("[server] error: give --api-key or --api-key-file, not both");
            std::process::exit(2);
        }
        match api_key_from_file(&f) {
            Ok(k) => api_key_raw = k,
            Err(e) => {
                stdout::err(&format!("[server] error: {e}"));
                std::process::exit(2);
            }
        }
    }
    let api_key = match api_key_of(&api_key_raw) {
        Ok(k) => k,
        Err(e) => {
            stdout::err(&format!("[server] error: {e}: give a key, or leave --api-key out"));
            std::process::exit(2);
        }
    };
    let cors = match origins_of(&a.list("cors_origin"), "--cors-origin", true) {
        Ok(c) => c,
        Err(e) => fail(e.strip_prefix("[server] error: ").unwrap_or(&e)),
    };
    let allowed = match allowed_hosts_of(&a.list("allowed_host")) {
        Ok(h) => h,
        Err(e) => fail(&e),
    };
    if !allowed.is_empty() {
        if allowed.iter().any(|h| h == "*") {
            stdout::out(&tagged("warning", "Host check off, any name is accepted (--allowed-host *)"));
        } else {
            stdout::out(&tagged("hosts", &allowed.join(",")));
        }
    }
    if !cors.is_empty() {
        stdout::out(&tagged("cors", &cors.join(",")));
        if cors.iter().any(|c| c == "*") && api_key.is_empty() {
            stdout::out(&tagged("warning", "any web page can use the model (--api-key KEY)"));
        }
    }
    let trusted: Vec<String> = vec![];
    let conf = Arc::new(RwLock::new(ServerConf {
        api_key: api_key.clone(),
        cors_origins: cors,
        trusted_origins: trusted.clone(),
        host_names: host_names_for(&host, &allowed, &trusted),
        allowed_hosts: allowed,
        max_payload: (mpm as usize) << 20,
        image_sources,
        logs: logs.clone(),
        root: root.clone(),
    }));
    // serve(): telemetry, then the listener
    router.start_telemetry();
    let server = match http::Server::new(&host, port as u16) {
        Ok(s) => s,
        Err(_) => fail(&format!("port {port} is in use (--port N)")),
    };
    let app: Arc<dyn http::App> = Arc::new(Handler { router: router.clone(), conf: conf.clone() });
    let _http = server.start(app).expect("http thread");
    let here = if ["0.0.0.0", "", "::"].contains(&host.as_str()) { "127.0.0.1".to_string() } else { host.clone() };
    for (s, (m, ..)) in services.iter().zip(settings.iter()) {
        s.start_idle_unload();
        s.start_idle_sleep(!shared_sleep, if multi { Some(s.model.clone()) } else { None });
        s.set_on_loaded(Some(announcer(s.clone(), m.clone(), multi, a.b("mock_engine"))));
    }
    stdout::out(&tagged("ready", &format!("http://{here}:{port}/v1  openai+anthropic  stats http://{here}:{port}/{}",
                                          if api_key.is_empty() { "" } else { " api-key=on" })));
    if a.b("mock_engine") {
        for s in &services {
            s.on_loaded();
        }
    }
    if !LOOPBACK_NAMES.contains(&host.as_str()) {
        let ips = lan_addresses();
        for ip in &ips {
            stdout::out(&tagged("network", &format!("http://{ip}:{port}/v1")));
        }
        if ips.is_empty() {
            stdout::out(&tagged("network", &format!("port={port}")));
        }
        if exposed_without_key(&host, &api_key) {
            stdout::out(&tagged("warning", "no api key, open to the network (--api-key KEY)"));
        }
    }
    if !a.b("mock_engine") && !a.b("lazy") {
        if multi {
            router.preload(&svc);
        } else {
            svc.start_load();
        }
    }
    if a.b("open") {
        let url = format!("http://{}:{port}/", if host == "0.0.0.0" || host.is_empty() { "127.0.0.1" } else { &host });
        if a.b("mock_engine") || a.b("lazy") {
            open_browser(&url);
        } else {
            let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let prev = svc.take_on_loaded();
            svc.set_on_loaded(Some(Box::new(move || {
                if let Some(p) = &prev {
                    p();
                }
                if !done.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    open_browser(&url);
                }
            })));
        }
    }
    signals::install();
    let mut rc = 0;
    loop {
        std::thread::sleep(Duration::from_secs(1));
        if signals::count() > 0 {
            break;
        }
        if svc.boot_error().is_some() {
            std::thread::sleep(Duration::from_secs(2));
            rc = 1;
            break;
        }
    }
    if rc == 0 {
        stdout::out("[server] stopping (Ctrl+C again to force)");
    }
    let first = signals::count();
    let killer = {
        let services = services.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(100));
            if signals::count() > first.max(if rc == 0 { 1 } else { 0 }) {
                for s in &services {
                    s.engine.kill();
                }
                return;
            }
        })
    };
    let _ = killer;
    router.stop();
    server.shutdown();
    router.stop_telemetry();
    for s in &services {
        let _ = s.engine.close();
        s.close_vision();
    }
    stdout::out("[server] stopped");
    std::process::exit(rc);
}

fn announcer(s: Arc<Service>, m: Args, multi: bool, mock: bool) -> Box<dyn Fn() + Send + Sync> {
    let first_load = Mutex::new(true);
    let said_video = Mutex::new(false);
    let weak = Arc::downgrade(&s);
    drop(s);
    Box::new(move || {
        let Some(s) = weak.upgrade() else { return };
        let engine = &s.engine;
        {
            let mut f = first_load.lock().unwrap();
            if *f {
                *f = false;
                if !mock {
                    warn_tight_ram(engine.info().as_ref().and_then(|i| i.get("arena_mib")));
                }
                if engine.batch() > 0 {
                    let shared = if m.b("slot_full_context") { None } else { m.shared_context };
                    stdout::out(&tagged("slots", &format!("{}, others queue", config::slots_text(engine.batch(), engine.max_context(), shared))));
                }
            }
        }
        stdout::out(&tagged("weights", &format!("loaded{}, ctx={} images={}",
            if multi { format!(" {}", s.model) } else { String::new() }, engine.max_context(), if s.has_vision() { "on" } else { "off" })));
        if let Some(video) = s.vision_video_state() {
            let mut said = said_video.lock().unwrap();
            if !*said {
                *said = true;
                let (state, why) = video.split_once(' ').unwrap_or((&video, ""));
                let text = if s.dialect_video_pad() { format!("{state}, {why}") } else { "off, this model reads no video".into() };
                stdout::out(&tagged("video", &text));
            }
        }
    })
}

#[allow(dead_code)]
fn _unused(_: EngineErr) {}
