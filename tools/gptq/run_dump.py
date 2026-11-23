"""run_dump.py CORPUS.jsonl LOG.jsonl [--port 8320] - every sample posted to my server (started with BASALT_GPTQ_DUMP)
exactly as a client sends it (OpenAI rows to /v1/chat/completions, Claude Code rows to /v1/messages), max_tokens 1,
temperature 0, not streamed, so the server's own frontend renders it and the engine prefills it. Resumable: ids already
in LOG are skipped. LOG rows: id, bucket, split, prompt tokens, cached (prefix reused, not re-prefilled), seconds."""
import argparse, json, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("corpus")
ap.add_argument("log")
ap.add_argument("--port", type=int, default=8320)
a = ap.parse_args()
done = set()
try:
    for l in open(a.log, encoding="utf-8"):
        done.add(json.loads(l)["id"])
except FileNotFoundError:
    pass
out = open(a.log, "a", encoding="utf-8")
tot = 0
for line in open(a.corpus, encoding="utf-8"):
    r = json.loads(line)
    if r["id"] in done:
        continue
    body = dict(r["body"])
    body["model"] = "hy4-flash-nvfp4"
    body["max_tokens"] = 1
    body["temperature"] = 0
    body["stream"] = False
    path = "/v1/messages" if r["api"] == "anthropic" else "/v1/chat/completions"
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (a.port, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-api-key": "local", "anthropic-version": "2023-06-01"})
    t0 = time.time()
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=3600))
        u = resp.get("usage", {})
        if r["api"] == "anthropic":
            cached = u.get("cache_read_input_tokens", 0) or 0
            prompt = (u.get("input_tokens", 0) or 0) + cached
        else:
            prompt = u.get("prompt_tokens", 0)
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        err = None
    except urllib.error.HTTPError as e:
        prompt = cached = 0
        err = "%s %s" % (e.code, e.read()[:300].decode("utf-8", "replace"))
    rec = {"id": r["id"], "bucket": r["bucket"], "lang": r["lang"], "split": r["split"], "prompt": prompt, "cached": cached,
           "s": round(time.time() - t0, 1), "err": err}
    out.write(json.dumps(rec) + "\n")
    out.flush()
    tot += prompt - cached
    print("%s %-14s %-3s prompt %6d cached %6d %.1fs%s" % (time.strftime("%H:%M:%S"), r["id"][:14], r["bucket"], prompt, cached,
                                                         rec["s"], " ERR " + err if err else ""), flush=True)
print("prefilled tokens this run:", tot)
