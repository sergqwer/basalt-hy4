//! vision::normalize against Vision.normalize (tests/golden/images.json from tests/golden/images.py): pass-through
//! formats byte for byte, converted ones pixel for pixel (a JPEG's decode may differ by a level: reported).

use base64::Engine as _;
use serde_json::Value;

#[test]
fn normalize_matches_pillow() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/images.json");
    let gold: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let b64 = |s: &Value| base64::engine::general_purpose::STANDARD.decode(s.as_str().unwrap()).unwrap();
    let (mut exact, mut near, mut fail) = (0, 0, 0);
    for c in gold.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let got = serve_api::vision::normalize(b64(&c["in"]));
        match (got, c.get("out")) {
            (Ok(g), Some(w)) => {
                let w = b64(w);
                if g == w {
                    exact += 1;
                    continue;
                }
                let px = |d: &[u8]| image::load_from_memory(d).map(|i| (i.width(), i.height(), i.to_rgb8().into_raw()));
                match (px(&g), px(&w)) {
                    (Ok(a), Ok(b)) if a.0 == b.0 && a.1 == b.1 => {
                        let maxd = a.2.iter().zip(&b.2).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap_or(0);
                        if maxd == 0 {
                            exact += 1;
                        } else if maxd <= 3 && name.starts_with("jpeg") {
                            near += 1;
                            eprintln!("{name}: pixels within {maxd} of Pillow's (JPEG decoder)");
                        } else {
                            fail += 1;
                            eprintln!("{name}: max pixel diff {maxd}");
                        }
                    }
                    (a, b) => {
                        fail += 1;
                        eprintln!("{name}: sizes {:?} vs {:?}", a.map(|x| (x.0, x.1)), b.map(|x| (x.0, x.1)));
                    }
                }
            }
            (Err(e), None) => {
                if e == c["error"].as_str().unwrap() {
                    exact += 1;
                } else {
                    fail += 1;
                    eprintln!("{name}: error {e:?} vs {:?}", c["error"]);
                }
            }
            (g, w) => {
                fail += 1;
                eprintln!("{name}: got {:?} want {:?}", g.map(|x| x.len()), w.is_some());
            }
        }
    }
    eprintln!("images: {exact} exact, {near} within 3 levels (JPEG), {fail} fail");
    assert_eq!(fail, 0);
}
