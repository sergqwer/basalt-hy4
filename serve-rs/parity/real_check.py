"""One real-engine check: the Python server, then the Rust one, over the same pack; a few greedy requests each; identical
text and usage expected. Also the server process's own memory (the engine is a child and not counted).

    python real_check.py PACK [--engine-bin BIN] [--alias NAME] [--out FILE]

Run it under the GPU lock. Each server is stopped with a real Ctrl+C; its engine must be gone afterwards."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run import Proc  # noqa: E402

PY = r"D:\Projects\basalt-055\.venv\Scripts\python.exe"
RS = str(ROOT / "serve-rs" / "target" / "release" / "basalt-serve.exe")
GIB = 1 << 30

REQUESTS = [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
                              "temperature": 0, "max_tokens": 300}),
    ("/v1/chat/completions", {"messages": [{"role": "system", "content": "You are terse."},
                                           {"role": "user", "content": "Name three prime numbers above 100."}],
                              "temperature": 0, "max_tokens": 300, "reasoning_effort": "low"}),
    ("/v1/messages", {"messages": [{"role": "user", "content": "Write one sentence about the sea."}],
                      "temperature": 0, "max_tokens": 300}),
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello in French."}],
                              "temperature": 0, "max_tokens": 200, "stream": True,
                              "stream_options": {"include_usage": True}}),
]


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
    if body.get("stream"):
        text, reasoning, usage, finish = "", "", None, None
        for line in raw.split("\n"):
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            for c in ev.get("choices") or []:
                d = c.get("delta") or {}
                text += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
                finish = c.get("finish_reason") or finish
            usage = ev.get("usage") or usage
        return {"text": text, "reasoning": reasoning, "usage": usage, "finish": finish}
    v = json.loads(raw)
    if path == "/v1/messages":
        return {"content": v["content"], "usage": v["usage"], "finish": v["stop_reason"]}
    m = v["choices"][0]["message"]
    return {"text": m.get("content"), "reasoning": m.get("reasoning_content"), "tool_calls": m.get("tool_calls"),
            "usage": v["usage"], "finish": v["choices"][0]["finish_reason"]}


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def server_proc(proc: Proc):
    """The process that serves HTTP: the venv launcher's python child, or basalt-serve itself."""
    tree = proc.tree()
    servers = [p for p in tree if p.name().lower() in ("python.exe", "basalt-serve.exe")]
    return servers[-1] if servers else None


def mem(p):
    try:
        i = p.memory_info()
        return {"working_set_mib": round(i.rss / (1 << 20), 1), "private_mib": round(i.private / (1 << 20), 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def engine_pids(proc: Proc):
    return [p.pid for p in proc.tree() if p.name().lower() == "basalt.exe"]


def one(name, argv, port, log):
    base = f"http://127.0.0.1:{port}"
    t0 = time.monotonic()
    proc = Proc(argv, dict(**__import__("os").environ), str(ROOT))
    out = {"side": name, "argv": argv}
    try:
        if not proc.wait_ready(120):
            out["error"] = "not ready"
            return out
        while time.monotonic() - t0 < 1200:
            h = get(base, "/health")
            if h.get("loaded"):
                break
            if proc.p.poll() is not None:
                out["error"] = "server ended while loading"
                return out
            time.sleep(2)
        out["load_s"] = round(time.monotonic() - t0, 1)
        sp = server_proc(proc)
        out["engine_pids"] = engine_pids(proc)
        out["mem_loaded"] = mem(sp)
        out["answers"] = []
        for path, body in REQUESTS:
            b = dict(body, model=get(base, "/v1/models")["data"][0]["id"])
            t = time.monotonic()
            out["answers"].append(post(base, path, b))
            out["answers"][-1]["wall_s"] = round(time.monotonic() - t, 2)
        out["mem_after"] = mem(sp)
        out["models"] = get(base, "/v1/models")
    finally:
        out["stop"] = proc.stop()
        time.sleep(3)
        out["engines_left"] = [p for p in out.get("engine_pids", []) if psutil.pid_exists(p)]
        out["stdout_tail"] = b"".join(proc.out).decode("utf-8", "replace").splitlines()[-12:]
        out["stderr_tail"] = b"".join(proc.err).decode("utf-8", "replace").splitlines()[-12:]
        log.write(json.dumps(out, ensure_ascii=False) + "\n")
        log.flush()
    return out


def wait_ram(target_gib, limit_s=300):
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit_s:
        if psutil.virtual_memory().available / GIB >= target_gib:
            return True
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--engine-bin", default=str(ROOT / "bin" / "basalt.exe"))
    ap.add_argument("--alias", default="Hy4-flash NVFP4")
    ap.add_argument("--out", default=str(HERE / "real_check.jsonl"))
    a = ap.parse_args()
    free0 = psutil.virtual_memory().available / GIB
    vram = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])
    print(f"free RAM {free0:.1f} GiB, VRAM used {vram} MiB", flush=True)
    if free0 < 95 or vram >= 2048:
        print("preconditions not met", flush=True)
        return 3
    common = ["--model", a.pack, "--engine-bin", a.engine_bin, "--alias", a.alias]
    res = []
    with open(a.out, "w", encoding="utf-8") as log:
        res.append(one("python", [PY, "-m", "serve.server", *common, "--port", "18850"], 18850, log))
        print(json.dumps({k: res[-1].get(k) for k in ("load_s", "mem_loaded", "mem_after", "stop", "engines_left", "error")}),
              flush=True)
        if not wait_ram(free0 - 4):
            print(f"RAM not back: {psutil.virtual_memory().available / GIB:.1f} GiB", flush=True)
        res.append(one("rust", [RS, *common, "--port", "18851"], 18851, log))
        print(json.dumps({k: res[-1].get(k) for k in ("load_s", "mem_loaded", "mem_after", "stop", "engines_left", "error")}),
              flush=True)
    wait_ram(free0 - 4)
    pa, ra = res[0].get("answers") or [], res[1].get("answers") or []
    strip = lambda x: {k: v for k, v in x.items() if k != "wall_s"}  # noqa: E731
    for i, (p, r) in enumerate(zip(pa, ra)):
        same = strip(p) == strip(r)
        print(f"request {i} {REQUESTS[i][0]}: {'IDENTICAL' if same else 'DIFFERENT'}  py {p['wall_s']} s  rs {r['wall_s']} s",
              flush=True)
        if not same:
            print("  py:", json.dumps(strip(p), ensure_ascii=False)[:600])
            print("  rs:", json.dumps(strip(r), ensure_ascii=False)[:600])
    left = [p.pid for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() == "basalt.exe"]
    vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.split()[0]
    print(f"after: basalt.exe left {left}, VRAM used {vram} MiB, free RAM {psutil.virtual_memory().available / GIB:.1f} GiB")
    return 0 if len(pa) == len(ra) == len(REQUESTS) and all(strip(p) == strip(r) for p, r in zip(pa, ra)) else 1


if __name__ == "__main__":
    sys.exit(main())
