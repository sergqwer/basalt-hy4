"""mk_keep.py DUMPDIR DUMPLOG CORPUS [--keep-prefixes 8] - the row mask GPTQ applies to every layer's dump.

The Claude Code system prompt + tool schemas (~8.4K tokens) open every A2 request, and the server's prefix cache does
not carry it across sessions (the cwd text differs), so it was prefilled once per session. The recipe caps identical
prefixes at ~5% of the corpus: the first --keep-prefixes A2 sessions keep it, the others drop their first
prefix_est - 512 rows. Rows map to requests in dump order: request i prefilled (prompt - cached) rows, from position
`cached`. Writes DUMPDIR/keep.npy (bool, one per dumped row) and prints the row accounting.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("log")
ap.add_argument("corpus")
ap.add_argument("--keep-prefixes", type=int, default=8)
a = ap.parse_args()
pre = {}
for l in open(a.corpus, encoding="utf-8"):
    r = json.loads(l)
    pre[r["id"]] = r.get("prefix_est", 0)
reqs = [json.loads(l) for l in open(a.log, encoding="utf-8")]
# rows actually in the dump (layer 1's record headers)
raw = np.memmap(os.path.join(a.dump, "moe_l01.bin"), dtype=np.uint8, mode="r")
o, total = 0, 0
while o < raw.size:
    T, N, K, ver = np.frombuffer(raw[o:o + 32].tobytes(), dtype=np.int64)
    total += int(T)
    o += 32 + 2 * T * N + 8 * T * K
# the last prompt token runs through the decode path (max_tokens 1), not moe(): each request dumps prompt - cached - 1 rows
LAST = 1
want = sum(max(0, r["prompt"] - r["cached"] - LAST) for r in reqs if not r.get("err"))
print("rows in dump", total, "rows the log says were prefilled", want)
assert total == want, "dump rows and request log disagree"
keep = np.ones(total, dtype=bool)
at, n_a2, dropped = 0, 0, 0
for r in reqs:
    if r.get("err"):
        continue
    n = max(0, r["prompt"] - r["cached"] - LAST)
    if r["bucket"] == "A2":
        n_a2 += 1
        if n_a2 > a.keep_prefixes:
            cut = max(0, pre.get(r["id"], 0) - 512 - r["cached"])
            cut = min(cut, n)
            keep[at:at + cut] = False
            dropped += cut
    at += n
np.save(os.path.join(a.dump, "keep.npy"), keep)
print("A2 requests", n_a2, "prefix rows dropped", dropped, "kept rows", int(keep.sum()))
