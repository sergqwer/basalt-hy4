"""assemble.py - pools -> corpus/train.jsonl, corpus/eval.jsonl, corpus/manifest.json.
Bucket A = A2 (all of it) + A3 up to A_QUOTA; other buckets as pooled. Train rows grouped by bucket (A2 first, so the
server's prefix reuse dedups the Claude Code system prompt + tool schemas), shuffled inside a bucket with seed 1234.
A1 (the user's transcripts) is not included: no consent yet; a pools/A1.jsonl would be picked up here."""
import json, os, random

G = r"D:\Projects\hy4-data\gptq\corpus"
A_QUOTA = 525_000
rng = random.Random(1234)
pools = {}
for b in ["A1", "A2", "A3", "B", "C1", "C2", "D", "E", "F", "G", "H"]:
    p = os.path.join(G, "pools", b + ".jsonl")
    if os.path.exists(p):
        pools[b] = [json.loads(l) for l in open(p, encoding="utf-8")]
train, ev = [], []
# A2 rows share the Claude Code system prompt + tool schemas; the dump pass reuses that prefix (prefix cache), so only
# the first row counts it
a2 = [r for r in pools.get("A2", []) if r["split"] == "train"]
a_tok = sum(r["est_tokens"] for r in pools.get("A1", []) if r["split"] == "train") +     sum(r["est_tokens"] - (r.get("prefix_est", 0) if i else 0) for i, r in enumerate(a2))
for b, rows in pools.items():
    tr = [r for r in rows if r["split"] == "train"]
    rng.shuffle(tr)
    if b == "A3":
        keep, t = [], a_tok
        for r in tr:
            if t >= A_QUOTA:
                break
            keep.append(r); t += r["est_tokens"]
        tr = keep
    train += tr
    ev += [r for r in rows if r["split"] == "eval"]
order = {b: i for i, b in enumerate(["A2", "A1", "A3", "B", "C1", "C2", "D", "E", "F", "G", "H"])}
train.sort(key=lambda r: order[r["bucket"]])
with open(os.path.join(G, "train.jsonl"), "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(G, "eval.jsonl"), "w", encoding="utf-8") as f:
    for r in ev:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
man = {"seed": 1234, "note": "est_tokens = tokenizer count of the content (A2: bytes/3); the dump log has the server's counts",
       "train": [{k: r[k] for k in ("id", "bucket", "lang", "source", "est_tokens")} for r in train],
       "eval": [{k: r[k] for k in ("id", "bucket", "lang", "source", "est_tokens")} for r in ev]}
json.dump(man, open(os.path.join(G, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
tot = {}
for r in train:
    tot.setdefault(r["bucket"], [0, 0])
    tot[r["bucket"]][0] += 1; tot[r["bucket"]][1] += r["est_tokens"]
for b, (n, t) in sorted(tot.items(), key=lambda x: order[x[0]]):
    print("%-3s %4d rows %8d tok" % (b, n, t))
print("train total", sum(t for _, t in tot.values()), "eval rows", len(ev), "eval tok", sum(r["est_tokens"] for r in ev))
