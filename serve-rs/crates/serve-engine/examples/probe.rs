//! The Rust side of the engine-protocol probe (tests/probe_py.py is the reference): ProcessEngine over
//! tests/fake_engine.cmd, one JSON line per step. Usage: probe <record file> [batch]
use std::sync::Arc;

use serde_json::{json, Value};
use serve_engine::iface::{Engine, Flag, GenStream, Prompt, Sampling};
use serve_engine::process::{ProcessEngine, REQUEST_LINES};
use serve_text::{ByteTokenizer, PromptText};

fn out(v: Value) {
    println!("{}", serve_text::pyjson::dumps(&v, true));
}

fn tokens(mut g: GenStream, keep: Option<usize>) -> (Vec<Value>, Value) {
    let mut got = Vec::new();
    let mut err = Value::Null;
    while let Some(item) = g.next() {
        match item {
            Ok(None) => continue,
            Ok(Some(t)) => {
                let lp = t.lp.map(|l| json!([l.lp, l.alts.iter().map(|(i, v)| json!([i, v])).collect::<Vec<_>>()])).unwrap_or(Value::Null);
                got.push(json!([t.id, lp]));
                if keep.map(|k| got.len() >= k).unwrap_or(false) {
                    break;
                }
            }
            Err(e) => {
                err = json!(format!("{}: {}", e.class_name(), e.message()));
                break;
            }
        }
    }
    let _ = g.close();
    (got, err)
}

fn samp(v: Value) -> Sampling {
    Sampling { map: v.as_object().unwrap().clone(), logit_bias: None }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let record = a[1].clone();
    let batch = a.get(2).cloned().unwrap_or_else(|| "0".into());
    let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut env: Vec<(String, String)> = std::env::vars().collect();
    env.push(("FAKE_RECORD".into(), record));
    env.push(("FAKE_BATCH".into(), batch.clone()));
    env.push(("FAKE_DELAY".into(), "0.05".into()));
    env.push(("FAKE_ERR".into(), "prompt (5 tokens) is too long: ok".into()));
    REQUEST_LINES.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut args = vec!["--model".to_string(), "x.basalt".to_string()];
    if batch != "0" {
        args.extend(["--batch".to_string(), batch.clone()]);
    }
    let e = ProcessEngine::new(here.join("fake_engine.cmd").to_str().unwrap(), args, None, None, Some(env), false).expect("start");
    let info = e.info().unwrap();
    out(json!({"step": "start", "info": info, "ctx": e.max_context(), "batch": e.batch()}));
    let s = samp(json!({"temperature": 0.7, "top_k": 20}));
    let (toks, err) = tokens(e.generate(Prompt::Ids(vec![1, 2, 3]), 4, &s, Flag::new(), None, None, None), None);
    out(json!({"step": "err", "toks": toks, "err": err}));
    let s = samp(json!({"temperature": 0.7, "top_k": 20, "seed": 3}));
    let (toks, err) = tokens(e.generate(Prompt::Ids(vec![1, 2, 3]), 4, &s, Flag::new(), None, None, None), None);
    out(json!({"step": "solo", "toks": toks, "err": err, "last": e.last().unwrap().map}));
    let (toks, err) = tokens(e.generate(Prompt::Ids(vec![5; 10]), 50, &samp(json!({})), Flag::new(), None, None, None), Some(2));
    out(json!({"step": "early", "toks": toks, "err": err, "last": e.last().unwrap().map}));
    let pt = Arc::new(PromptText::new(Arc::new(ByteTokenizer), "hello <|im_start|>world".into(), vec![(0, 2)], "", false));
    let (toks, err) = tokens(e.generate(Prompt::Text(pt.clone()), 30, &samp(json!({"logprobs": null, "_logprobs": 2})), Flag::new(), None, None, None), None);
    out(json!({"step": "text", "toks": toks, "err": err, "last": e.last().unwrap().map, "n": pt.n()}));
    if e.batch() > 0 {
        let e = Arc::new(e);
        let run = |e: Arc<ProcessEngine>, ids: Vec<u32>, name: &str| {
            let name = name.to_string();
            std::thread::spawn(move || tokens(e.generate(Prompt::Ids(ids), 6, &samp(json!({})), Flag::new(), None, Some(name), None), None))
        };
        let t1 = run(e.clone(), vec![7; 40], "a");
        std::thread::sleep(std::time::Duration::from_millis(120));
        let t2 = run(e.clone(), vec![8; 3], "b");
        let (a, b) = (t1.join().unwrap(), t2.join().unwrap());
        out(json!({"step": "pair", "a": a.0.len(), "b": b.0.len(), "errs": [a.1, b.1], "slots": e.slot_view().unwrap()}));
        finish(&e);
        return;
    }
    finish(&e);
}

fn finish(e: &ProcessEngine) {
    out(json!({"step": "vram", "vram": e.vram(None).unwrap()}));
    out(json!({"step": "progress", "progress": e.prompt_progress().unwrap(), "conv": e.conversation()}));
    e.close().unwrap();
    out(json!({"step": "closed", "alive": e.alive()}));
}
