//! image_source against serve/engine/image_source.py (tests/golden/sources.json from tests/golden/sources.py).

use base64::Engine as _;
use serde_json::{json, Value};
use serve_api::image_source as S;

fn res<T: Into<Value>>(r: Result<T, String>) -> Value {
    match r {
        Ok(v) => json!({ "ok": v.into() }),
        Err(e) => json!({"error": e, "log": null}),
    }
}

#[test]
fn image_sources_match_python() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/sources.json");
    let g: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut fails = vec![];
    let mut n = 0;
    let mut check = |what: String, got: Value, want: &Value| {
        n += 1;
        if &got != want {
            fails.push(format!("{what}: got {got} want {want}"));
        }
    };
    for c in g["hosts"].as_array().unwrap() {
        let h = c[0].as_str().unwrap();
        check(format!("host {h:?}"), res(S::parse_url_host(h).map(|(h, p)| json!([h, p]))), &c[1]);
    }
    for c in g["addrs"].as_array().unwrap() {
        let a = c[0].as_str().unwrap();
        check(format!("addr {a:?}"), json!(S::is_public_address(a)), &c[1]);
    }
    for c in g["wpaths"].as_array().unwrap() {
        let p = c[0].as_str().unwrap();
        check(format!("wpath {p:?}"), json!(S::windows_path_refused(p)), &c[1]);
    }
    for c in g["furls"].as_array().unwrap() {
        let u = c[0].as_str().unwrap();
        check(format!("furl {u:?}"), res(S::file_url_path(u, true, false, true)), &c[1]);
        check(format!("furl rel {u:?}"), res(S::file_url_path(u, true, true, true)), &c[2]);
    }
    for c in g["loads"].as_array().unwrap() {
        let p = &c[1];
        let pol = S::ImageSources {
            url_hosts: p["url_hosts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|h| (h[0].as_str().unwrap().to_string(), h[1].as_u64().map(|x| x as u16)))
                .collect(),
            why: p["why"].as_str().unwrap().into(),
            peer: p["peer"].as_str().unwrap().into(),
            ..Default::default()
        };
        let got = match S::load_image(&c[0], Some(&pol), S::IMAGE) {
            Ok(b) => json!({"ok": base64::engine::general_purpose::STANDARD.encode(b)}),
            Err(e) => json!({"error": e.message, "log": e.log_line}),
        };
        check(format!("load {}", c[0]), got, &c[2]);
    }
    for f in &fails {
        eprintln!("{f}");
    }
    eprintln!("sources: {} pass, {} fail", n - fails.len(), fails.len());
    assert!(fails.is_empty());
}
