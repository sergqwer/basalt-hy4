"""dense_sensitivity.py --dump DIR [--out table.jsonl] - phase 2: per dense tensor class, the output error of
Q8_0, NVFP4 RTN (Basalt's rule) and NVFP4 GPTQ, each class alone, on the engine's own inputs.

DIR/dense/<gguf name>.bin from session.cpp's gptq_dump (every STRIDE-th prompt row of the layers dumped). Weights come
from the BF16 checkpoint (HF name mapped below; the shape is checked against the pack's dense index). For each tensor:
rows split 3/4 fit (GPTQ's Hessian) / 1/4 test; error = sum ||x W^T - x Wq^T||^2 / sum ||x W^T||^2 over test rows.
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, r"D:\Projects\hy4-data\gptq\tools")
import gptq_hy4 as Gq  # noqa: E402

HF = {  # gguf suffix -> HF suffix (under model.layers.L.)
    "attn_q_a.weight": "self_attn.q_a_proj.weight", "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight", "attn_gate.weight": "self_attn.linear_gate.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "indexer.attn_q_b.weight": "self_attn.compressed_indexer.query_projection.weight",
    "indexer.attn_k.weight": "self_attn.compressed_indexer.token_key_projection.weight",
    "indexer_compressor_value.weight": "self_attn.compressed_indexer.key_value_projection.weight",
    "indexer_compressor_gate.weight": "self_attn.compressed_indexer.key_gate_projection.weight",
    "ffn_routed_down.weight": "mlp.routed_expert_down_proj.weight", "ffn_routed_up.weight": "mlp.routed_expert_up_proj.weight",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight", "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight",
}
CLASS = {"attn_q_a": "q_a", "attn_q_b": "q_b", "attn_kv_a_mqa": "kv_a", "attn_gate": "attn_gate", "attn_output": "o",
         "indexer.attn_q_b": "indexer_q", "indexer.attn_k": "indexer_k", "indexer_compressor_value": "indexer_kv",
         "indexer_compressor_gate": "indexer_gate", "ffn_routed_down": "latent_down", "ffn_routed_up": "latent_up",
         "ffn_gate_shexp": "shared_gate", "ffn_up_shexp": "shared_up", "ffn_down_shexp": "shared_down",
         "ffn_gate": "l0_gate", "ffn_up": "l0_up", "ffn_down": "l0_down", "engram_kv": "engram_kv", "output": "head"}


def load_rows(p, cap=120_000):
    raw = np.fromfile(p, dtype=np.uint8)
    xs, o = [], 0
    while o < raw.size:
        T, N, K, ver = np.frombuffer(raw[o:o + 32].tobytes(), dtype=np.int64)
        o += 32
        xs.append(raw[o:o + 2 * T * N].view(np.float16).reshape(T, N)); o += 2 * T * N
    X = np.concatenate(xs)
    if X.shape[0] > cap:
        X = X[np.random.RandomState(0).choice(X.shape[0], cap, replace=False)]
    return torch.from_numpy(X.copy())


def q8_deq(W):
    b = W.view(W.shape[0], -1, 32)
    d = (b.abs().amax(-1, keepdim=True) / 127.0).half().float()
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = torch.sign(b * inv) * torch.floor((b * inv).abs() + 0.5)
    return (q.clamp(-127, 127) * d).view_as(W)


def rel_err(X, W, Wq):
    y = X @ W.T
    return float(((X @ Wq.T - y) ** 2).sum() / (y ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=r"D:\Projects\hy4-data\gptq\eval\dense_sensitivity.jsonl")
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    dev = a.dev
    if a.threads:
        torch.set_num_threads(a.threads)
    tab = Gq.ue4m3_table(dev)
    idx = json.load(open(Gq.BF16_DIR + r"\model.safetensors.index.json"))["weight_map"]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["tensor"] for l in open(a.out, encoding="utf-8")}
    for p in sorted(glob.glob(os.path.join(a.dump, "dense", "*.bin"))):
        name = os.path.basename(p)[:-4]
        if name in done:
            continue
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        if not m or m.group(2) not in HF:
            print("skip (no HF map)", name)
            continue
        l, suf = int(m.group(1)), m.group(2)
        k = "model.layers.%d.%s" % (l, HF[suf])
        if k not in idx:
            print("skip (not in checkpoint)", name, k)
            continue
        with safe_open(os.path.join(Gq.BF16_DIR, idx[k]), "pt") as f:
            W = f.get_tensor(k).to(dev).float()
        X = load_rows(p).to(dev).float()
        if W.shape[1] != X.shape[1]:
            print("skip (shape)", name, tuple(W.shape), tuple(X.shape))
            continue
        rows, K = W.shape
        n = X.shape[0]
        fit, test = X[: n * 3 // 4], X[n * 3 // 4:]
        H = (fit.T @ fit) / fit.shape[0]
        gs = (W.abs().amax() / 2688.0).expand(1, rows).contiguous()
        res = {"tensor": name, "class": CLASS.get(suf.replace(".weight", ""), suf), "layer": l, "shape": [rows, K], "rows_used": n}
        res["q8_0"] = rel_err(test, W, q8_deq(W))
        if K % 16 == 0:
            _, _, Qr = Gq.quant_rtn(W.unsqueeze(0), gs, tab)
            res["nvfp4_rtn"] = rel_err(test, W, Qr[0])
            _, _, Qg = Gq.quant_gptq(W.unsqueeze(0), H.unsqueeze(0), gs, tab)
            res["nvfp4_gptq"] = rel_err(test, W, Qg[0])
        res["planes_ok"] = rows % 128 == 0 and K % 256 == 0
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)
        del W, X, H
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
