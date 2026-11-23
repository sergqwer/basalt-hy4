"""live_metrics.py TAG... - live Claude Code eval sessions per arm (a2/capture/TAG-*.jsonl, a2/runs/TAG-*.json):
tasks finished without error, turns, tool calls, tool_result errors, failed Edits ("String to replace not found" and
other Edit errors over Edit calls), and Han characters in the replies / thinking of Ukrainian tasks (CJK drift)."""
import glob, json, os, re, sys
G = r"D:\Projects\hy4-data\gptq\a2"
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def blocks(sse):
    out, cur = [], {}
    for line in sse.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:])
        except Exception:
            continue
        if ev.get("type") == "content_block_start":
            cur[ev["index"]] = dict(ev["content_block"])
        elif ev.get("type") == "content_block_delta":
            b, d = cur.get(ev["index"]), ev["delta"]
            if b is not None:
                for k in ("text", "thinking"):
                    if k in d:
                        b[k] = b.get(k, "") + d[k]
    return [cur[i] for i in sorted(cur)]


for tag in sys.argv[1:]:
    tasks = {t["id"]: t for t in json.load(open(G + r"\tasks.json", encoding="utf-8"))}
    s = {"tasks": 0, "finished": 0, "turns": 0, "tool_calls": 0, "edit_calls": 0, "edit_failed": 0, "result_errors": 0,
         "uk_replies": 0, "uk_replies_han": 0}
    for cap in sorted(glob.glob(G + r"\capture\%s-*.jsonl" % tag)):
        tid = os.path.basename(cap)[:-6]
        base = tid.split("-", 1)[1]
        run = G + r"\runs\%s.json" % tid
        if not os.path.exists(run):
            continue
        s["tasks"] += 1
        r = json.load(open(run, encoding="utf-8"))
        try:
            j = json.loads(r["stdout"])
            s["finished"] += int(r["rc"] == 0 and not j.get("is_error"))
        except Exception:
            pass
        uk = tasks.get(base, {}).get("lang") == "uk"
        rows = [json.loads(l) for l in open(cap, encoding="utf-8")]
        ids = {}
        for q in rows:
            if not (q.get("req") or {}).get("tools") or q.get("status") != 200:
                continue
            s["turns"] += 1
            bl = blocks(q["resp"])
            for b in bl:
                if b.get("type") == "tool_use":
                    s["tool_calls"] += 1
                    ids[b["id"]] = b["name"]
                    s["edit_calls"] += b["name"] == "Edit"
            if uk:
                txt = "".join(b.get("text", "") + b.get("thinking", "") for b in bl)
                if txt.strip():
                    s["uk_replies"] += 1
                    s["uk_replies_han"] += bool(HAN.search(txt))
        # tool results as the client reported them (the last request of the session holds every one)
        last = max((q for q in rows if (q.get("req") or {}).get("tools")), key=lambda q: len(q["req"]["messages"]), default=None)
        if last:
            for m in last["req"]["messages"]:
                if isinstance(m.get("content"), list):
                    for c in m["content"]:
                        if c.get("type") == "tool_result" and c.get("is_error"):
                            s["result_errors"] += 1
                            if ids.get(c.get("tool_use_id")) == "Edit":
                                s["edit_failed"] += 1
    s["edit_fail_rate"] = round(s["edit_failed"] / max(1, s["edit_calls"]), 4)
    s["uk_han_share"] = round(s["uk_replies_han"] / max(1, s["uk_replies"]), 4)
    print(tag, json.dumps(s))
