//! Embeds the Basalt icon (assets/basalt.ico, from tools/make_icon.py) into basalt-serve.exe.
//! A no-op for every target other than Windows.

use std::path::PathBuf;

fn main() {
    let ico = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../assets/basalt.ico");
    println!("cargo:rerun-if-changed={}", ico.display());
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let ico = ico.canonicalize().expect("assets/basalt.ico is missing");
    // windres reads backslashes in a string as escapes; forward slashes work for both rc.exe and windres.
    let path = ico.to_string_lossy().trim_start_matches(r"\\?\").replace('\\', "/");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("basalt-serve.rc");
    std::fs::write(&out, format!("IDI_ICON1 ICON \"{path}\"\n")).unwrap();
    embed_resource::compile(&out, embed_resource::NONE)
        .manifest_optional()
        .unwrap();
}
