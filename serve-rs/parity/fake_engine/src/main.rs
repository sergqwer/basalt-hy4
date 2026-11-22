//! A scripted stand-in for the `basalt` engine binary, so the Python and Rust servers can drive a real ProcessEngine
//! without a GPU. Usable as `--engine-bin`.
//!
//! * `--gpu-plan`: prints the scenario's plan as one JSON line and exits (serve/engine/vision.py model_plan).
//!   An unknown `--flag` is refused the way the engine refuses it: `error: unknown argument: --flag`, exit 2.
//! * `--serve`: prints the scenario's start lines (INFO ..., READY <ctx> stop), then answers stdin commands:
//!   GEN/GENI/GENT/SGEN/... take the next scripted reply (the last one repeats), STATS/VRAM their lines,
//!   QUIT (or EOF) exits.
//! * Env FAKE_ENGINE_SCRIPT: a JSON scenario (see DEFAULT_SCENARIO); unset = the default.
//!   Env FAKE_ENGINE_INPUT: every byte read on stdin is appended to this file.
//!   Env FAKE_ENGINE_ARGS: one line per start, its arguments joined by spaces, appended to this file.
//!
//! Reply lines may hold {prompt_n} (the ids in a GEN, the bytes of a GENT) and {max_new}.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};

// The flags `engine_flags_of` finds by scanning this binary: NUL-separated, so no flag touches printable bytes.
const FLAGS: &str = "\0--serve\0--model\0--gpu\0--gpu-plan\0--max-context\0--batch\0--slots\0--grammar\0\
--vision\0--vision-external\0--conversation-cache-dir\0--conversation-cache-disk-mib\0--seed\0--temperature\0\
--top-p\0--top-k\0--min-p\0--help\0";

const DEFAULT_SCENARIO: &str = r#"{
  "plan": {"devices": "0", "cards": [{"smi": 0}], "notes": [], "max_context": 4096},
  "start": ["INFO engine=fake-0.1 arena_mib=0", "READY 4096 stop"],
  "replies": [["T 263", "T 10", "T 72", "T 105", "DONE 4 {prompt_n} 1.0 1.0 stop"]],
  "stats": [],
  "vram": ["VRAM reserve_mib=0 cache_mib=0"]
}"#;

fn flags() -> Vec<&'static str> {
    FLAGS.split('\0').filter(|s| !s.is_empty()).collect()
}

fn lines_of(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

fn append(env: &str, data: &[u8]) {
    if let Ok(path) = std::env::var(env) {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(data);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    append("FAKE_ENGINE_ARGS", format!("{}\n", args.join(" ")).as_bytes());
    let text = match std::env::var("FAKE_ENGINE_SCRIPT") {
        Ok(p) => std::fs::read_to_string(&p).unwrap_or_else(|e| {
            eprintln!("error: FAKE_ENGINE_SCRIPT {p}: {e}");
            std::process::exit(2)
        }),
        Err(_) => DEFAULT_SCENARIO.to_string(),
    };
    let sc: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("error: scenario: {e}");
        std::process::exit(2)
    });
    let known = flags();
    if args.iter().any(|a| a == "--help") {
        for f in &known {
            println!("{f}");
        }
        return;
    }
    for a in &args {
        if a.starts_with("--") && !known.contains(&a.as_str()) {
            eprintln!("error: unknown argument: {a}");
            std::process::exit(2);
        }
    }
    let out = std::io::stdout();
    let mut out = out.lock();
    if args.iter().any(|a| a == "--gpu-plan") {
        let _ = writeln!(out, "{}", sc["plan"]);
        return;
    }
    if !args.iter().any(|a| a == "--serve") {
        eprintln!("error: fake engine: give --serve or --gpu-plan");
        std::process::exit(2);
    }
    for l in lines_of(&sc["start"]) {
        let _ = writeln!(out, "{l}");
    }
    let _ = out.flush();
    let replies: Vec<Vec<String>> = sc["replies"]
        .as_array()
        .map(|a| a.iter().map(lines_of).collect())
        .unwrap_or_default();
    let mut turn = 0usize;
    let stdin = std::io::stdin();
    let mut inp = BufReader::new(stdin.lock());
    let mut line = Vec::new();
    loop {
        line.clear();
        match inp.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        append("FAKE_ENGINE_INPUT", &line);
        let s = String::from_utf8_lossy(&line).trim_end().to_string();
        let words: Vec<&str> = s.split_whitespace().collect();
        let cmd = words.first().copied().unwrap_or("");
        let mut send: Vec<String> = Vec::new();
        match cmd {
            "QUIT" => return,
            "STATS" => send = lines_of(&sc["stats"]),
            "VRAM" => send = lines_of(&sc["vram"]),
            "GEN" | "GENI" | "GENT" | "SGEN" | "SGENI" | "SGENT" => {
                let at = if cmd.starts_with('S') { 1 } else { 0 };
                let max_new = words.get(1 + at).copied().unwrap_or("0").to_string();
                let mut prompt_n = 0usize;
                if cmd.ends_with('T') {
                    // GENT <max_new> [k=v ...] <nbytes> [plain=...]: the text follows the line
                    let n: usize = words.iter().skip(2 + at).find_map(|w| w.parse().ok()).unwrap_or(0);
                    let mut body = vec![0u8; n];
                    if inp.read_exact(&mut body).is_err() {
                        return;
                    }
                    append("FAKE_ENGINE_INPUT", &body);
                    prompt_n = n;
                    send.push(format!("PT {n} {max_new}"));
                } else if let Some(ids) = words.last() {
                    prompt_n = ids.split(',').filter(|x| !x.is_empty()).count();
                }
                if !replies.is_empty() {
                    let r = &replies[turn.min(replies.len() - 1)];
                    turn += 1;
                    for l in r {
                        send.push(l.replace("{prompt_n}", &prompt_n.to_string()).replace("{max_new}", &max_new));
                    }
                }
            }
            _ => {}
        }
        for l in send {
            let _ = writeln!(out, "{l}");
        }
        let _ = out.flush();
    }
}
