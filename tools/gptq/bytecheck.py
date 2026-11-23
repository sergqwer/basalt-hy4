"""Byte check: HF BF16 experts of a layer -> Basalt RTN blobs (converter's fp4q_rows), compare with the pack."""
import json, sys, numpy as np
sys.path.insert(0, r"D:\Projects\basalt-hy4-gptq\tools\convert"); sys.path.insert(0, r"D:\Projects\basalt-hy4-gptq\tools")
import fp4_round as Q, checkpoint as C
from model_file import ModelFile
from safetensors import safe_open
PACK = r"D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt"
BF = r"C:\LLM_models\hy4-bf16"
l = int(sys.argv[1]); exps = [int(x) for x in sys.argv[2].split(",")]
idx = json.load(open(BF + r"\model.safetensors.index.json"))["weight_map"]
m = ModelFile(PACK); eoff, _ = m.section("experts")
lay = [ln.split() for ln in m.read("experts.layout").decode().splitlines()[1:]]
row = [r for r in lay if int(r[0]) == l][0]; loff, blob = int(row[3]), int(row[4])
def get(name):
    k = "model.layers.%d.mlp.experts.%s" % (l, name)
    with safe_open(BF + "\\" + idx[k], "np" if False else "pt") as f:
        return f.get_slice(k)
gu, dn = get("gate_up_proj"), get("down_proj")
import torch
for e in exps:
    g = gu[e:e+1].float().numpy()[0]; d = dn[e:e+1].float().numpy()[0]
    print("shapes", g.shape, d.shape)
    with open(PACK, "rb") as f:
        f.seek(eoff + loff + e * blob); ref = np.frombuffer(f.read(blob), np.uint8)
    for variant, mats in {"gu[:1536]=gate,rows=out": (g[:1536], g[1536:], d),
                          "interleaved": (g[0::2], g[1::2], d),
                          "transposed": (g.T[:1536] if g.shape[0] != 3072 else None, None, d.T)}.items():
        if mats[0] is None or mats[1] is None and variant != "transposed": continue
        out, tail = [], []
        ok = True
        for x in mats:
            if x is None: ok = False; break
            x = np.ascontiguousarray(x, np.float32)
            gs = Q.fp4q_global(float(np.abs(x).max())); c, s = Q.fp4q_rows(x, gs)
            out.append(C.nvfp4_rows(c, s).tobytes()); tail.append(gs)
        if not ok: continue
        b = np.frombuffer(b"".join(out) + np.array(tail + [0.0], "<f4").tobytes(), np.uint8)
        n = 1536 * 1536 // 64 * 36
        parts = [(b[i*n:(i+1)*n] == ref[i*n:(i+1)*n]).mean() for i in range(3)]
        print(l, e, variant, "match gate/up/down:", ["%.4f" % p for p in parts], "tail", np.frombuffer(ref[-16:].tobytes(), "<f4"), np.array(tail))
