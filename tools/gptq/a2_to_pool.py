"""a2_to_pool.py - captured Claude Code request bodies -> corpus/pools/A2.jsonl (Anthropic /v1/messages bodies).

Per session (one capture file per task): the requests that end a conversation (the last one, and any whose successor
does not extend it, e.g. after a compaction), with tools only (the CLI's side requests carry none), plus the model's
final answer parsed from that request's SSE response, appended as the closing assistant turn.
Key scan first: the tray key by value (read here, never printed), hf_, ghp_, sk-, Bearer, PEM; a hit drops the
request. Only counts are printed. Eval tasks (tasks.json "eval": true) and the eval split are kept apart by session.
"""
import glob, hashlib, json, os, re, sys

G = r"D:\Projects\hy4-data\gptq"
PAT = re.compile(r"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._~+/-]{12,}|"
                 r"-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def tray_key():
    try:
        txt = open(r"D:\Projects\tray_llm\dist\config.toml", encoding="utf-8").read()
    except OSError:
        return None
    sec = None
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("["):
            sec = s
        elif sec == "[app]" and s.startswith("api_key"):
            v = s.split("=", 1)[1].strip().strip('"').strip("'")
            return v or None
    return None


def final_message(sse: str):
    """Anthropic SSE -> the assistant message's content blocks."""
    blocks = {}
    for line in sse.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except Exception:
            continue
        t = ev.get("type")
        if t == "content_block_start":
            b = dict(ev["content_block"])
            if b.get("type") == "tool_use":
                b["_json"] = ""
            blocks[ev["index"]] = b
        elif t == "content_block_delta":
            b, d = blocks.get(ev["index"]), ev["delta"]
            if b is None:
                continue
            if d.get("type") == "text_delta":
                b["text"] = b.get("text", "") + d["text"]
            elif d.get("type") == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d["thinking"]
            elif d.get("type") == "input_json_delta":
                b["_json"] += d.get("partial_json", "")
    out = []
    for i in sorted(blocks):
        b = blocks[i]
        if b.get("type") == "tool_use":
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except Exception:
                return None
        out.append(b)
    return out


_R = None


def ntok(q):
    """Tokens of the request as the fork's frontend renders it (make_nll_base's renderer)."""
    global _R
    if _R is None:
        sys.path.insert(0, r"D:\Projects\basalt-hy4-gptq")
        from serve import frontend as FE
        from tokenizers import Tokenizer
        _R = (FE, Tokenizer.from_file(r"C:\LLM_models\hy4-bf16\tokenizer.json"),
              FE.ChatTemplate.from_file(r"C:\LLM_models\hy4-bf16\chat_template.jinja"))
    FE, tok, tpl = _R
    msgs, tools, kw = FE.anthropic_to_messages(q)
    kw = {k: v for k, v in kw.items() if k in ("reasoning_effort", "preserved_thinking")}
    if kw.get("reasoning_effort") not in (None, "no_think", "high"):
        kw["reasoning_effort"] = "high"
    full = len(tok.encode(tpl.render(msgs, tools=tools, add_generation_prompt=False, **kw), add_special_tokens=False).ids)
    pre = len(tok.encode(tpl.render(msgs[:1], tools=tools, add_generation_prompt=False, **kw), add_special_tokens=False).ids)
    return full, pre


def main():
    key = tray_key()
    tasks = {t["id"]: t for t in json.load(open(G + r"\a2\tasks.json", encoding="utf-8"))}
    cnt = {"requests": 0, "with_tools": 0, "key_hits_dropped": 0, "pattern_hits_dropped": 0, "kept": 0, "no_final": 0}
    rows = []
    for cap in sorted(glob.glob(G + r"\a2\capture\*.jsonl")):
        tid = os.path.basename(cap)[:-6]
        tag, base = tid.split("-", 1)
        if tag not in ("cal", "evl") or not os.path.exists(os.path.join(G, "a2", "runs", tid + ".json")):
            continue          # a session that never finished (runner stopped) stays out
        reqs = []
        for line in open(cap, encoding="utf-8"):
            r = json.loads(line)
            cnt["requests"] += 1
            q = r.get("req") or {}
            if r.get("status") != 200 or not q.get("tools"):
                continue
            cnt["with_tools"] += 1
            s = json.dumps(q, ensure_ascii=False) + r.get("resp", "")
            if key and key in s:
                cnt["key_hits_dropped"] += 1
                continue
            if PAT.search(s):
                cnt["pattern_hits_dropped"] += 1
                continue
            reqs.append(r)
        ends = [i for i in range(len(reqs)) if i == len(reqs) - 1 or len(reqs[i + 1]["req"]["messages"]) <= len(reqs[i]["req"]["messages"])]
        t = tasks.get(base, {})
        split = "eval" if t.get("eval") or int(hashlib.sha1(base.encode()).hexdigest(), 16) % 10 == 0 else "train"
        for k, i in enumerate(ends):
            q = json.loads(json.dumps(reqs[i]["req"]))
            fin = final_message(reqs[i]["resp"])
            if not fin:
                cnt["no_final"] += 1
                continue
            q["messages"].append({"role": "assistant", "content": fin})
            est, pre = ntok(q)
            rows.append({"id": "A2-%s-%d" % (tid, k), "bucket": "A2", "lang": t.get("lang", "en"), "source": tid, "split": split,
                         "est_tokens": est, "api": "anthropic", "body": q, "n_requests": len(reqs),
                         "prefix_est": pre})
            cnt["kept"] += 1
    with open(G + r"\corpus\pools\A2.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("tray key found for the scan:", key is not None)
    print(json.dumps(cnt))
    print("A2 rows", len(rows), "est tokens train", sum(r["est_tokens"] for r in rows if r["split"] == "train"),
          "eval", sum(r["est_tokens"] for r in rows if r["split"] == "eval"))


if __name__ == "__main__":
    main()
