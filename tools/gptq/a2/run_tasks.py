"""run_tasks.py [--only ID,..] [--eval] [--tag NAME] [--port 8320]
Runs each A2 task: a fresh copy of its seed under sandbox/<tag>-<task>, the claude CLI pointed through capture_proxy.py
(bodies only) at my Basalt server, acceptEdits + a sandbox allowlist (settings.json), never bypassPermissions."""
import argparse, json, os, pathlib, shutil, subprocess, sys, time

G = pathlib.Path(r"D:\Projects\hy4-data\gptq")
A2 = G / "a2"
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--eval", action="store_true", help="run the eval tasks instead of the calibration ones")
ap.add_argument("--tag", default="cal")
ap.add_argument("--port", type=int, default=8320)
ap.add_argument("--timeout", type=int, default=900)
ap.add_argument("--max-turns", type=int, default=40)
a = ap.parse_args()

tasks = json.load(open(A2 / "tasks.json", encoding="utf-8"))
tasks = [t for t in tasks if bool(t.get("eval")) == a.eval]
if a.only:
    keep = set(a.only.split(","))
    tasks = [t for t in tasks if t["id"] in keep]

env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC") or k == "AI_AGENT")}
model = "hy4-flash-nvfp4"
env.update({
    "CLAUDE_CONFIG_DIR": str(A2 / "ccconfig"),
    "ANTHROPIC_AUTH_TOKEN": "local-no-key",
    "ANTHROPIC_MODEL": model,
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
    "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_SMALL_FAST_MODEL": model,
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_WORKFLOWS": "1", "CLAUDE_CODE_DISABLE_ARTIFACT": "1", "CLAUDE_CODE_DISABLE_CRON": "1",
    "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": "2",
    "DISABLE_AUTOUPDATER": "1",
})
claude = shutil.which("claude")
SUFFIX = {"en": "\n\nWork only inside the current folder; do not read or change anything outside it.",
          "uk": "\n\nПрацюй лише в поточній теці; нічого поза нею не читай і не змінюй."}
for t in tasks:
    tid = "%s-%s" % (a.tag, t["id"])
    box = G / "sandbox" / tid
    out = A2 / "runs" / (tid + ".json")
    if out.exists():
        print("skip", tid, flush=True)
        continue
    if box.exists():
        shutil.rmtree(box)
    shutil.copytree(A2 / "seeds" / t["seed"], box)
    cap = A2 / "capture" / (tid + ".jsonl")
    lport = a.port + 1
    proxy = subprocess.Popen([sys.executable, str(G / "tools" / "capture_proxy.py"), str(cap), "--listen", str(lport),
                              "--upstream", "127.0.0.1:%d" % a.port])
    time.sleep(1.0)
    e = dict(env, ANTHROPIC_BASE_URL="http://127.0.0.1:%d" % lport)
    cmd = [claude, "-p", "--output-format", "json", "--settings", str(A2 / "settings.json"),
           "--permission-mode", "acceptEdits", "--disallowed-tools", "WebFetch,WebSearch,PowerShell",
           "--model", model, "--max-turns", str(a.max_turns)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, input=t["prompt"] + SUFFIX[t["lang"]], text=True, encoding="utf-8", capture_output=True, cwd=box, env=e,
                           timeout=a.timeout)
        res = {"rc": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-4000:]}
    except subprocess.TimeoutExpired as x:
        res = {"rc": "timeout", "stdout": (x.stdout or "")[-20000:] if isinstance(x.stdout, str) else "", "stderr": ""}
    res.update({"task": t, "seconds": round(time.time() - t0, 1)})
    proxy.terminate()
    proxy.wait()
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        j = json.loads(res["stdout"])
        summ = "turns %s cost %s err %s" % (j.get("num_turns"), j.get("total_cost_usd"), j.get("is_error"))
    except Exception:
        summ = "rc %s" % res["rc"]
    print("%s %s %.0fs %s" % (time.strftime("%H:%M"), tid, res["seconds"], summ), flush=True)
