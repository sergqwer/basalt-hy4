"""gate_compare.py BASE_DIR ARM_DIR... - quality-gate rows (rows.tsv anywhere under each dir; columns chunk pos target nll
nll_base kld same_top ...) compared against the first dir: mean KLD, top-1, NLL per arm, and each arm minus the base with
a 95% paired bootstrap CI over chunks (seed 1234, 10,000 resamples). With a sparse (zero-logit) base only NLL is
meaningful (--nll-only)."""
import glob, os, sys
import numpy as np

args = [x for x in sys.argv[1:] if not x.startswith("--")]
nll_only = "--nll-only" in sys.argv


def load(d):
    f = sorted(glob.glob(os.path.join(d, "**", "rows.tsv"), recursive=True))
    assert f, "no rows.tsv under " + d
    rows = {}
    for line in open(f[-1], encoding="utf-8"):
        p = line.split()
        if len(p) < 7 or not p[0].lstrip("-").isdigit():
            continue
        rows[(int(p[0]), int(p[1]))] = (float(p[3]), float(p[5]), float(p[6]))
    return rows


base = load(args[0])
keys = sorted(base)
rng = np.random.default_rng(1234)


def per_chunk(rows, ks):
    ch = {}
    for k in ks:
        ch.setdefault(k[0], []).append(rows[k])
    ids = sorted(ch)
    return ids, np.array([np.mean(ch[c], axis=0) for c in ids]), np.array([len(ch[c]) for c in ids])


cols = ["nll"] if nll_only else ["nll", "kld", "top1"]
idx = {"nll": 0, "kld": 1, "top1": 2}
print("%-28s %6s " % ("arm", "rows") + " ".join("%10s" % c for c in cols))
for d in args:
    r = load(d)
    ks = [k for k in keys if k in r]
    _, m, n = per_chunk(r, ks)
    w = n / n.sum()
    print("%-28s %6d " % (os.path.basename(d.rstrip("/\\")), len(ks)) + " ".join("%10.5f" % float((m[:, idx[c]] * w).sum()) for c in cols))
print("\ndelta vs %s (95%% paired bootstrap over chunks)" % os.path.basename(args[0].rstrip("/\\")))
for d in args[1:]:
    r = load(d)
    ks = [k for k in keys if k in r]
    _, mb, n = per_chunk(base, ks)
    _, ma, _ = per_chunk(r, ks)
    diff = ma - mb
    out = []
    for c in cols:
        x = diff[:, idx[c]]
        bs = []
        for _ in range(10000):
            s = rng.integers(0, len(x), len(x))
            bs.append((x[s] * n[s]).sum() / n[s].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append("%s %+.5f [%+.5f, %+.5f]" % (c, (x * n).sum() / n.sum(), lo, hi))
    print("%-28s " % os.path.basename(d.rstrip("/\\")) + "   ".join(out))
