"""nll_buckets.py BASE_ARM_DIR ARM_DIR... - own held-out NLL per corpus bucket (chunk -> bucket from
eval/own-nll-base.kld.json), per arm, and each arm minus the first."""
import glob, json, os, sys
import numpy as np

man = json.load(open(r"D:\Projects\hy4-data\gptq\eval\own-nll-base.kld.json"))
bucket = {c["chunk"]: c["bucket"] for c in man["chunks"]}


def load(d):
    f = sorted(glob.glob(os.path.join(d, "**", "rows.tsv"), recursive=True))[-1]
    out = {}
    for line in open(f, encoding="utf-8"):
        p = line.split("\t")
        if len(p) >= 7 and p[0].isdigit():
            out[(int(p[0]), int(p[1]))] = float(p[3])
    return out


arms = [(os.path.basename(d.rstrip("/\\")), load(d)) for d in sys.argv[1:]]
keys = sorted(set.intersection(*[set(a) for _, a in arms]))
bs = sorted(set(bucket[k[0]] for k in keys))
print("%-22s" % "arm" + "".join("%9s" % b for b in bs) + "%9s" % "all")
for name, a in arms:
    row = [np.mean([a[k] for k in keys if bucket[k[0]] == b]) for b in bs] + [np.mean([a[k] for k in keys])]
    print("%-22s" % name + "".join("%9.4f" % x for x in row))
base = arms[0][1]
for name, a in arms[1:]:
    row = [np.mean([a[k] - base[k] for k in keys if bucket[k[0]] == b]) for b in bs] + [np.mean([a[k] - base[k] for k in keys])]
    print("%-22s" % ("d " + name) + "".join("%+9.4f" % x for x in row))
