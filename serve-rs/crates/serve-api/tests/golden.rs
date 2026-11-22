//! The Python serve/ functions' outputs (tests/golden/golden.json, written by tests/golden/gen.py) against this
//! crate's: request conversion, rendered prompts and their plain spans, prompt ids, errors, and parser events.
//! The Hy4 half needs the model's tokenizer sections: HY4TOK=<dir with vocab.json, merges.txt, token_type.json,
//! tokenizer.json, chat_template.jinja>; without it only the Qwen half runs.

use serde_json::{json, Value};
use serve_api::api::responses as rs;
use serve_api::error::ApiError;
use serve_api::frontend;
use serve_api::models::dialect_for;
use serve_api::service::Service;
use serve_engine::mock::MockEngine;
use serve_text::tokenizer::{BpeTokenizer, ByteTokenizer, Tokenizer};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn hy4_dir() -> Option<PathBuf> {
    let d = std::env::var("HY4TOK").ok().map(PathBuf::from)?;
    d.join("vocab.json").exists().then_some(d)
}

fn service(kind: &str, hy4: Option<&Path>) -> Arc<Service> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let (tok, mut d, src): (Arc<dyn Tokenizer>, _, String) = if kind == "qwen" {
        (
            Arc::new(ByteTokenizer),
            dialect_for(None).unwrap(),
            std::fs::read_to_string(root.join("serve/chat_template.jinja")).unwrap(),
        )
    } else {
        let h = hy4.unwrap();
        let r = |n: &str| std::fs::read(h.join(n)).unwrap();
        let tj = r("tokenizer.json");
        let t = BpeTokenizer::from_files(&r("vocab.json"), &r("merges.txt"), &r("token_type.json"), Some(&tj)).unwrap();
        (Arc::new(t), dialect_for(Some("hy4flash")).unwrap(), std::fs::read_to_string(h.join("chat_template.jinja")).unwrap())
    };
    let tmpl = d.template(&src).unwrap();
    let engine = Arc::new(MockEngine::new(tok.clone(), vec!["x".into()], 32768, 0.0, 0, None));
    Arc::new(Service::new(engine, tok, tmpl, "basalt", None, None, false, true, true, Some(d), None))
}

fn err_json(e: &ApiError) -> Value {
    json!({"message": e.message()})
}

/// (converted fields, out) for one chat case, as gen.py records them.
fn chat_case(svc: &Service, case: &Value) -> Value {
    let req = case["req"].as_object().unwrap();
    let api = case["api"].as_str().unwrap();
    let mut got = serde_json::Map::new();
    let r = (|| -> Result<(), ApiError> {
        let (messages, tools, kw) = match api {
            "openai" | "anthropic" => {
                let (m, t, kw) = if api == "openai" {
                    frontend::openai_to_messages(req)?
                } else {
                    frontend::anthropic_to_messages(req, true)?
                };
                let (t, force) = frontend::forced_call(req.get("tool_choice"), t, &|n| svc.forced_opening(n))?;
                got.insert("force".into(), force.map(Value::from).unwrap_or(Value::Null));
                (m, t, kw)
            }
            _ => {
                rs::check_request(req)?;
                let m = rs::input_messages(req)?;
                let (t, names, skipped) = rs::request_tools(req)?;
                let kw = rs::template_kwargs(req)?;
                let names: serde_json::Map<String, Value> = names
                    .into_iter()
                    .map(|(k, (ns, n, kind))| (k, json!([ns, n, kind])))
                    .collect();
                got.insert("names".into(), Value::Object(names));
                got.insert("skipped".into(), json!(skipped));
                got.insert("text_format".into(), rs::text_format(req)?.unwrap_or(Value::Null));
                (m, t, kw)
            }
        };
        got.insert("messages".into(), Value::Array(messages.clone()));
        got.insert("tools".into(), tools.clone().unwrap_or(Value::Null));
        got.insert("kwargs".into(), Value::Object(kw.clone()));
        let out = match svc.thinking_kwargs(&kw).and_then(|k| svc.render_text(&messages, tools.as_ref(), &k)) {
            Ok((text, plain)) => {
                let ids = svc.tok.encode(&text, true, &plain);
                // gen.py's spans are character offsets; ours are bytes
                let ch = |b: usize| text[..b].chars().count();
                json!({"prompt": text, "plain": plain.iter().map(|(a, b)| json!([ch(*a), ch(*b)])).collect::<Vec<_>>(), "ids": ids})
            }
            Err(e) => err_json(&e),
        };
        got.insert("out".into(), out);
        Ok(())
    })();
    if let Err(e) = r {
        got.insert("out".into(), err_json(&e));
    }
    Value::Object(got)
}

fn expected(case: &Value) -> Value {
    let mut want = serde_json::Map::new();
    for k in ["force", "names", "skipped", "text_format", "messages", "tools", "kwargs"] {
        if let Some(v) = case.get(k) {
            want.insert(k.into(), v.clone());
        }
    }
    let out = &case["out"];
    want.insert(
        "out".into(),
        if out.get("error").is_some() { json!({"message": out["message"]}) } else { out.clone() },
    );
    Value::Object(want)
}

fn parser_case(svc: &Service, case: &Value) -> Value {
    let tools = json!([{"name": "get_weather", "description": "Weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}]);
    let mut p = svc.dialect().parser(case["thinking"].as_bool().unwrap(), Some(&tools), true);
    let mut evs = vec![];
    let ev = |e: &frontend::Event| {
        json!({"kind": e.kind, "text": e.text,
               "call": e.call.as_ref().map(|c| json!({"name": c.name, "arguments": c.arguments}))})
    };
    let mut error = None;
    for c in case["chunks"].as_array().unwrap() {
        match p.feed(c.as_str().unwrap()) {
            Ok(v) => evs.extend(v.iter().map(ev)),
            Err(e) => {
                error = Some(e.message());
                break;
            }
        }
    }
    if error.is_none() {
        match p.finish() {
            Ok(v) => evs.extend(v.iter().map(ev)),
            Err(e) => error = Some(e.message()),
        }
    }
    let mut out = json!({"events": evs});
    if let Some(m) = error {
        out["message"] = Value::from(m);
    }
    out
}

fn parser_expected(case: &Value) -> Value {
    let mut out = json!({"events": case["events"]});
    if case.get("error").is_some() {
        out["message"] = case["message"].clone();
    }
    out
}

#[test]
fn golden() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/golden.json");
    let gold: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let hy4 = hy4_dir();
    let (mut pass, mut fail) = (0, 0);
    let mut report = String::new();
    for kind in ["qwen", "hy4"] {
        if kind == "hy4" && hy4.is_none() {
            eprintln!("HY4TOK not set: the Hy4 half is skipped");
            continue;
        }
        let svc = service(kind, hy4.as_deref());
        frontend::reset_tool_orders();
        for (i, case) in gold[kind]["chat"].as_array().unwrap().iter().enumerate() {
            let got = chat_case(&svc, case);
            let want = expected(case);
            if got == want {
                pass += 1;
            } else {
                fail += 1;
                if fail <= 12 {
                    report += &format!(
                        "\n--- {kind} chat #{i} ({})\nwant: {}\n got: {}\n",
                        case["api"],
                        serde_json::to_string(&want).unwrap().chars().take(1500).collect::<String>(),
                        serde_json::to_string(&got).unwrap().chars().take(1500).collect::<String>()
                    );
                }
            }
        }
        for (i, case) in gold[kind]["parser"].as_array().unwrap().iter().enumerate() {
            let got = parser_case(&svc, case);
            let want = parser_expected(case);
            if got == want {
                pass += 1;
            } else {
                fail += 1;
                if fail <= 12 {
                    report += &format!("\n--- {kind} parser #{i}\nwant: {want}\n got: {got}\n");
                }
            }
        }
    }
    eprintln!("golden: {pass} pass, {fail} fail{report}");
    assert_eq!(fail, 0, "golden mismatches");
}

/// serve/tests/fixtures/hy4_flash_golden.json: transformers' own renders of the Hy4 template (official and a pack's
/// suffix), through ChatTemplate as test_hy4flash_dialect.Reference checks them.
#[test]
fn hy4_template_matches_transformers() {
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../serve/tests/fixtures");
    let gold: Value = serde_json::from_slice(&std::fs::read(fx.join("hy4_flash_golden.json")).unwrap()).unwrap();
    let official = std::fs::read_to_string(fx.join("hy4_flash_chat_template.jinja")).unwrap();
    let pack = official
        .replace(":opensource", ":6124c78e")
        .replace("['max', 'high', 'no_think']", "['max', 'high', 'low', 'no_think']")
        .replace("should be no_think/high/max", "should be no_think/low/high/max");
    let (mut pass, mut fail) = (0, 0);
    for (key, src) in [("official", official), ("pack", pack)] {
        let mut d = dialect_for(Some("hy4flash")).unwrap();
        let t = d.template(&src).unwrap();
        for case in gold["cases"].as_array().unwrap() {
            let kw = case["kwargs"].as_object().cloned().unwrap_or_default();
            let mut kw2 = kw.clone();
            let agp = kw2.shift_remove("add_generation_prompt").map(|v| v == true).unwrap_or(true);
            let tools = case.get("tools").filter(|v| !v.is_null());
            let r = t.render(&case["messages"], tools, agp, &kw2);
            let ok = match (&case[key], &r) {
                (Value::String(want), Ok(got)) => want == got,
                (Value::Object(_), Err(serve_api::template::TemplateErr::Request(m))) => m.contains("reasoning_effort error"),
                _ => false,
            };
            if ok {
                pass += 1;
            } else {
                fail += 1;
                eprintln!("--- {key} {}: want {:?}\n got {:?}", case["name"], case[key], r);
            }
        }
    }
    eprintln!("hy4 transformers reference: {pass} pass, {fail} fail");
    assert_eq!(fail, 0);
}

/// Documents (gen.py document_cases): PDFs built by tests/golden/pdfs.py through pypdf, Office files, text, errors.
#[test]
fn documents_match_python() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/golden.json");
    let gold: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let (mut pass, mut fail) = (0, 0);
    for case in gold["documents"].as_array().unwrap() {
        let content = vec![json!({"type": "text", "text": "before"}), case["part"].clone()];
        let got = match serve_api::api::documents::expand_documents(&content) {
            Ok(items) => json!({"items": items}),
            Err(e) => json!({"message": e.message()}),
        };
        let want = if case.get("items").is_some() { json!({"items": case["items"]}) } else { json!({"message": case["message"]}) };
        if same_but_png(&got, &want) {
            pass += 1;
        } else {
            fail += 1;
            eprintln!("--- {}\nwant {}\n got {}", case["part"]["file"]["filename"], want.to_string().chars().take(800).collect::<String>(),
                      got.to_string().chars().take(800).collect::<String>());
        }
    }
    eprintln!("documents: {pass} pass, {fail} fail");
    assert_eq!(fail, 0);
}

/// Equal, with data:image/png URLs compared by their decoded pixels (Pillow's zlib stream is not reproduced).
fn same_but_png(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(x), Value::String(y)) if x.starts_with("data:image/png;base64,") && y.starts_with("data:image/png;base64,") => {
            use base64::Engine as _;
            let px = |s: &str| {
                let raw = base64::engine::general_purpose::STANDARD.decode(&s[22..]).ok()?;
                image::load_from_memory(&raw).ok().map(|i| (i.width(), i.height(), i.to_rgba8().into_raw()))
            };
            px(x).is_some() && px(x) == px(y)
        }
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_but_png(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|((k1, v1), (k2, v2))| k1 == k2 && same_but_png(v1, v2))
        }
        _ => a == b,
    }
}
