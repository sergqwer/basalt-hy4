"""gptq_hy4.py - NVFP4 GPTQ of Hy4-flash's routed experts from the BF16 checkpoint, on Basalt's dumped expert inputs.

  quant   --dump DIR --layers 1-41 --out BLOBDIR [--method gptq|rtn]   -> BLOBDIR/lNN.bin ([512, 3981328] uint8) + lNN.json
  write   --blobs BLOBDIR --base RTN.basalt --pack NEW.basalt            copy RTN pack once, swap the experts per layer,
                                                                        update the experts section's SHA in the table
  analyze --dump EVALDIR --layers .. --pack A.basalt [--pack B.basalt]    routed-weighted relative error of each pack's
                                                                        experts vs BF16 on held-out rows (decoded from
                                                                        the packs' own bytes)

Dump records (session.cpp gptq_dump, v3): int64 {T, 1536, 16, 3}, T x 1536 fp16 inputs (latent after routed_down),
T x 16 int32 ids, T x 16 f32 routing weights. A row counts for an expert times its routing weight squared.

NVFP4 as Basalt's converter rounds it (tools/convert/fp4_round.py, byte-identical in --method rtn): per matrix
gs = amax / (6 * 448) (gate, up and down each their own), per 16 values the UE4M3 code of least squared error among
c0-8 .. c0+16 around amax / (6 gs), E2M1 ties to even. GPTQ: the same scale search, weighted by diag(H), on the
weights as GPTQ has updated them, at the start of each 16-column group; columns left to right (no act-order), the
rounding error spread through the Cholesky of H^-1; damping 1% of mean diag. gate and up share H (same input); down's H
comes from the hidden the *quantized* gate/up produce: h = silu(min(g, 10)) * clamp(u, -10, 10).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

BF16_DIR = r"C:\LLM_models\hy4-bf16"
L, F, K, NE = 1536, 1536, 16, 512
LIMIT = 10.0
GU_BYTES = 2 * F * (L // 64) * 36          # gate rows then up rows, ggml block_nvfp4
D_BYTES = L * (F // 64) * 36
BLOB = GU_BYTES + D_BYTES + 16
PRIOR_TOKENS = 256
MIN_ROWS = 2 * L
FP8_MAX = 448.0
STEPS = list(range(-8, 17))
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ---------------------------------------------------------------------------------------------- number formats
def ue4m3_table(dev):
    """The 127 finite non-negative E4M3 values, ascending (index = code)."""
    b = torch.arange(127)
    e, m = b >> 3, b & 7
    v = torch.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.double() - 7))
    return v.float().to(dev)


def ue4m3_round(x, tab):
    """Non-negative float -> UE4M3 code, nearest, ties to the even code, saturating (fp4_round.ue4m3_round)."""
    x = torch.clamp(x, max=448.0)
    i = torch.searchsorted(tab, x.contiguous()).clamp(1, 126)
    lo, hi = tab[i - 1], tab[i]
    up = (x - lo > hi - x) | ((x - lo == hi - x) & (i % 2 == 0))
    return torch.where(up, i, i - 1)


def e2m1_codes(y):
    """Scaled values -> E2M1 codes 0..15 (bit 3 sign), nearest, ties to even, saturating (fp4_round.e2m1_round)."""
    a = y.abs()
    c = ((a > 0.25).to(torch.int8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0)).to(torch.int8)
    return c | torch.where((y < 0) & (c > 0), 8, 0).to(torch.int8)


def e2m1_value(c, dev):
    t = E2M1.to(dev)
    return torch.cat([t, -t])[c.long()]


def best_scale(blk, gs, tab, h=None):
    """blk [..., 16] float32, gs broadcastable to blk[..., 0], h [..., 16] weights or None.
    Returns (scale codes [...], dequantized block [..., 16]) - the least (weighted) squared error over STEPS around c0;
    the first of equal errors in the order |d| ascending (as fp4q_rows scans sorted(STEPS, key=abs))."""
    c0 = ue4m3_round(blk.abs().amax(-1) / (6.0 * gs), tab)
    best_e = best_c = best_q = None
    for d in sorted(STEPS, key=abs):
        c = (c0 + d).clamp(0, 126)
        sd = tab[c] * gs
        inv = torch.where(sd > 0, 1.0 / torch.where(sd > 0, sd, torch.ones_like(sd)), torch.zeros_like(sd))
        q = e2m1_value(e2m1_codes(blk * inv.unsqueeze(-1)), blk.device) * sd.unsqueeze(-1)
        err = (q - blk) ** 2
        if h is not None:
            err = err * h
        # numpy's pairwise order for 16 contiguous float32 values (8 accumulators), so RTN ties match fp4q_rows
        r = err[..., :8] + err[..., 8:]
        err = ((r[..., 0] + r[..., 1]) + (r[..., 2] + r[..., 3])) + ((r[..., 4] + r[..., 5]) + (r[..., 6] + r[..., 7]))
        if best_e is None:
            best_e, best_c, best_q = err, c, q
        else:
            b = err < best_e
            best_e = torch.where(b, err, best_e)
            best_c = torch.where(b, c, best_c)
            best_q = torch.where(b.unsqueeze(-1), q, best_q)
    return best_c, best_q


def codes_of(w, sc, gs, tab):
    """Final codes of w [..., cols] under scale codes sc [..., cols/16]."""
    sh = w.shape
    blk = w.reshape(*sh[:-1], sh[-1] // 16, 16)
    sd = tab[sc.long()] * gs.unsqueeze(-1)
    inv = torch.where(sd > 0, 1.0 / torch.where(sd > 0, sd, torch.ones_like(sd)), torch.zeros_like(sd))
    return e2m1_codes(blk * inv.unsqueeze(-1)).reshape(sh)


def to_ggml(codes, sc):
    """codes [E, rows, cols] int8, scale codes [E, rows, cols/16] -> [E, rows * cols/64 * 36] uint8 (checkpoint.nvfp4_rows)."""
    E, rows, cols = codes.shape
    v = codes.to(torch.uint8).view(E, rows, cols // 64, 4, 16)
    qs = v[..., :8] | (v[..., 8:] << 4)
    d = sc.to(torch.uint8).view(E, rows, cols // 64, 4)
    return torch.cat([d, qs.reshape(E, rows, cols // 64, 32)], dim=-1).reshape(E, -1)


def from_ggml(raw, rows, cols, gs, tab):
    """[E, rows*cols/64*36] uint8 + gs [E, rows] -> dequantized float [E, rows, cols]."""
    E = raw.shape[0]
    b = raw.view(E, rows, cols // 64, 36)
    sc = b[..., :4].long()
    qs = b[..., 4:].view(E, rows, cols // 64, 4, 8)
    codes = torch.cat([qs & 15, qs >> 4], dim=-1)                        # [E, rows, nb, 4, 16]
    val = e2m1_value(codes, raw.device)
    sd = tab[sc] * gs.view(E, rows, 1, 1)
    return (val * sd.unsqueeze(-1)).reshape(E, rows, cols)


# ------------------------------------------------------------------------------------------------- quantizers
def quant_rtn(W, gs, tab):
    """W [E, rows, cols], gs [E, rows] -> (codes, scale codes, dequantized)."""
    E, rows, cols = W.shape
    blk = W.view(E, rows, cols // 16, 16)
    sc, q = best_scale(blk, gs.view(E, rows, 1), tab)
    return codes_of(W, sc, gs, tab), sc, q.view(E, rows, cols)


def quant_gptq(W, H, gs, tab, percdamp=0.01, blocksize=128):
    """GPTQ onto Basalt's NVFP4 grid, batched over experts. W [E, rows, cols], H [E, cols, cols], gs [E, rows]."""
    E, rows, cols = W.shape
    W = W.clone()
    H = H.clone()
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag <= 0
    diag[dead] = 1.0
    W.masked_fill_(dead.unsqueeze(1), 0.0)
    damp = percdamp * diag.mean(-1)
    H += damp.view(E, 1, 1) * torch.eye(cols, device=H.device).unsqueeze(0)
    hd = torch.diagonal(H, dim1=1, dim2=2).clone()
    Lc, info = torch.linalg.cholesky_ex(H)
    if bool((info != 0).any()):          # a failed factorization: more damping for those experts
        bad = info != 0
        H[bad] += (0.1 * hd[bad].mean(-1)).view(-1, 1, 1) * torch.eye(cols, device=H.device)
        Lc = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Lc)
    U = torch.linalg.cholesky(Hinv, upper=True)
    del Hinv, Lc, H
    sc_all = torch.zeros((E, rows, cols // 16), dtype=torch.int64, device=W.device)
    Q = torch.zeros_like(W)
    gsv = gs.view(E, rows, 1)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, :, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        U1 = U[:, i1:i2, i1:i2]
        qblk = None
        for i in range(i2 - i1):
            col = i1 + i
            if col % 16 == 0:
                g = col // 16
                hg = hd[:, col:col + 16].unsqueeze(1)
                sc, _ = best_scale(W1[:, :, i:i + 16], gs, tab, hg)
                sc_all[:, :, g] = sc
                sd = (tab[sc] * gs)                                           # [E, rows]
                inv = torch.where(sd > 0, 1.0 / torch.where(sd > 0, sd, torch.ones_like(sd)), torch.zeros_like(sd))
            x = W1[:, :, i]
            q = e2m1_value(e2m1_codes(x * inv), W.device) * sd
            Q[:, :, col] = q
            err = (x - q) / U1[:, i, i].unsqueeze(1)
            W1[:, :, i:] -= err.unsqueeze(2) * U1[:, i, i:].unsqueeze(1)
            Err1[:, :, i] = err
        W[:, :, i2:] -= torch.bmm(Err1, U[:, i1:i2, i2:])
    codes = codes_of(Q, sc_all, gs, tab)
    return codes, sc_all, Q


# ------------------------------------------------------------------------------------------------------ inputs
class Bf16:
    def __init__(self, d=BF16_DIR):
        self.d = pathlib.Path(d)
        self.idx = json.load(open(self.d / "model.safetensors.index.json"))["weight_map"]

    def get(self, l, name, e0=0, e1=NE):
        k = "model.layers.%d.mlp.experts.%s" % (l, name)
        with safe_open(self.d / self.idx[k], "pt") as f:
            return f.get_slice(k)[e0:e1]


def load_dump(d, l, dev):
    """X [T, 1536] fp16 (on dev), ids [T, 16] int64, w [T, 16] f32 (on dev)."""
    p = os.path.join(d, "moe_l%02d.bin" % l)
    raw = np.fromfile(p, dtype=np.uint8)
    xs, ids, ws, o = [], [], [], 0
    while o < raw.size:
        T, N, KK, ver = np.frombuffer(raw[o:o + 32].tobytes(), dtype=np.int64)
        assert ver == 3 and N == L and KK == K, (T, N, KK, ver)
        o += 32
        xs.append(raw[o:o + 2 * T * N].view(np.float16).reshape(T, N)); o += 2 * T * N
        ids.append(raw[o:o + 4 * T * K].view(np.int32).reshape(T, K)); o += 4 * T * K
        ws.append(raw[o:o + 4 * T * K].view(np.float32).reshape(T, K)); o += 4 * T * K
    X, I, Wt = np.concatenate(xs), np.concatenate(ids).astype(np.int64), np.concatenate(ws)
    del raw, xs
    kp = os.path.join(d, "keep.npy")          # mk_keep.py: repeated Claude Code prefixes capped
    if os.path.exists(kp):
        keep = np.load(kp)
        assert keep.size == X.shape[0], (keep.size, X.shape)
        X, I, Wt = X[keep], I[keep], Wt[keep]
    return torch.from_numpy(X).to(dev), torch.from_numpy(I).to(dev), torch.from_numpy(Wt).to(dev)


def routed(I, Wt):
    """Per expert: (row indices, routing weights), on the ids' device."""
    T = I.shape[0]
    tok = torch.arange(T, device=I.device).unsqueeze(1).expand(T, K).reshape(-1)
    e, w = I.reshape(-1), Wt.reshape(-1)
    order = torch.argsort(e, stable=True)
    e, t, w = e[order], tok[order], w[order]
    counts = torch.bincount(e, minlength=NE)
    starts = torch.cumsum(counts, 0) - counts
    cl, sl = counts.tolist(), starts.tolist()
    return [(t[s:s + c], w[s:s + c]) for s, c in zip(sl, cl)], cl


def moments(X, rows, w, f=None, chunk=65536):
    """sum w^2 f(x) f(x)^T over an expert's rows (in chunks: the busiest experts have ~10^5 rows), and the mean of f(x)^2."""
    S, sq, n = None, 0.0, 0
    for c0 in range(0, rows.numel(), chunk):
        x = X[rows[c0:c0 + chunk]].float()
        if f is not None:
            x = f(x)
        sq += float((x ** 2).sum()); n += x.numel()
        xw = x * w[c0:c0 + chunk].unsqueeze(1)
        S = xw.T @ xw if S is None else S + xw.T @ xw
    return S, (sq / n if n else 1.0)


def hessian_of(X, rows, w, prior, n0, f=None, dim=L):
    """The expert's normalized Hessian; prior None = isotropic at the mean square of f(x)."""
    S, msq = moments(X, rows, w, f) if rows.numel() else (None, 1.0)
    if S is None:
        S = torch.zeros((dim, dim), device=X.device)
    if prior is None:
        prior = torch.eye(dim, device=S.device) * msq
    return (S + n0 * prior) / (float((w ** 2).sum()) + n0)


def hessian(x, w, prior, n0):
    """sum w^2 x x^T, mixed with prior (a mean second moment) at n0 pseudo-weight, normalized."""
    xw = x.float() * w.unsqueeze(1)
    S = xw.T @ xw
    return (S + n0 * prior) / (float((w ** 2).sum()) + n0)


def swiglu(gu):
    g, u = gu[..., :F], gu[..., F:]
    return torch.nn.functional.silu(torch.clamp(g, max=LIMIT)) * torch.clamp(u, -LIMIT, LIMIT)


# ------------------------------------------------------------------------------------------------------ quant
def quant_layer(bf, l, dump, method, dev, batch, tab, n_exp=NE):
    t0 = time.time()
    gu_all = bf.get(l, "gate_up_proj")            # [512, 3072, 1536] bf16, CPU
    d_all = bf.get(l, "down_proj")                # [512, 1536, 1536]
    stats = {"layer": l, "method": method}
    if method == "gptq":
        X, I, Wt = load_dump(dump, l, dev)
        lists, counts = routed(I, Wt)
        Xs = X[torch.randperm(X.shape[0], device=dev)[:200_000]].float()
        prior = (Xs.T @ Xs) / Xs.shape[0]
        del Xs
        n0 = PRIOR_TOKENS * float((Wt ** 2).mean())
        cs = sorted(counts)
        stats.update({"tokens": int(X.shape[0]), "rows_min": cs[0], "rows_median": cs[len(cs) // 2], "rows_max": cs[-1],
                      "under_2x1536": [e for e in range(NE) if counts[e] < MIN_ROWS]})
    out = torch.empty((NE, BLOB), dtype=torch.uint8)
    for b0 in range(0, n_exp, batch):
        es = list(range(b0, min(b0 + batch, n_exp)))
        B = len(es)
        Wgu = gu_all[b0:b0 + B].to(dev).float()
        Wd = d_all[b0:b0 + B].to(dev).float()
        gs_gu = torch.cat([(Wgu[:, :F].abs().amax((1, 2)) / (6 * FP8_MAX)).unsqueeze(1).expand(B, F),
                           (Wgu[:, F:].abs().amax((1, 2)) / (6 * FP8_MAX)).unsqueeze(1).expand(B, F)], 1).contiguous()
        gs_d = (Wd.abs().amax((1, 2)) / (6 * FP8_MAX)).unsqueeze(1).expand(B, L).contiguous()
        gs_gu = torch.where(gs_gu > 0, gs_gu, torch.ones_like(gs_gu))
        gs_d = torch.where(gs_d > 0, gs_d, torch.ones_like(gs_d))
        if method == "rtn":
            c_gu, s_gu, _ = quant_rtn(Wgu, gs_gu, tab)
            c_d, s_d, _ = quant_rtn(Wd, gs_d, tab)
        else:
            Hgu = torch.stack([hessian_of(X, lists[e][0], lists[e][1], prior, n0) for e in es])
            c_gu, s_gu, Qgu = quant_gptq(Wgu, Hgu, gs_gu, tab)
            del Hgu
            Hd = []
            for j, e in enumerate(es):
                rows, w = lists[e]
                Wj = Qgu[j]
                # down's input is the hidden the QUANTIZED gate/up produce; its prior is isotropic at that hidden's mean square
                Hd.append(hessian_of(X, rows, w, None, n0, f=lambda x, Wj=Wj: swiglu(x @ Wj.T), dim=F))
            Hd = torch.stack(Hd)
            c_d, s_d, _ = quant_gptq(Wd, Hd, gs_d, tab)
            del Hd, Qgu
        tail = torch.stack([gs_gu[:, 0], gs_gu[:, F], gs_d[:, 0], torch.zeros(B, device=dev)], 1).float()
        blob = torch.cat([to_ggml(c_gu, s_gu), to_ggml(c_d, s_d), tail.contiguous().view(torch.uint8).view(B, 16)], 1)
        out[b0:b0 + B] = blob.cpu()
        del Wgu, Wd, c_gu, s_gu, c_d, s_d, blob
        torch.cuda.empty_cache()
    stats["seconds"] = round(time.time() - t0, 1)
    return out, stats


def cmd_quant(a):
    dev = a.dev
    tab = ue4m3_table(dev)
    bf = Bf16()
    os.makedirs(a.out, exist_ok=True)
    lo, hi = (int(x) for x in a.layers.split("-"))
    for l in range(lo, hi + 1):
        p = os.path.join(a.out, "l%02d.bin" % l)
        if os.path.exists(p) and os.path.exists(p[:-4] + ".json"):
            log("layer %d: done already" % l)
            continue
        blobs, st = quant_layer(bf, l, a.dump, a.method, dev, a.batch, tab, a.experts)
        blobs.numpy().tofile(p + ".tmp")
        os.replace(p + ".tmp", p)
        json.dump(st, open(p[:-4] + ".json", "w"))
        log("layer %d: %s rows min/med/max %s/%s/%s, under 2x1536: %d experts, %.0f s" % (
            l, a.method, st.get("rows_min"), st.get("rows_median"), st.get("rows_max"), len(st.get("under_2x1536", [])), st["seconds"]))
        if a.delete_dumps and a.method == "gptq":
            # disk: drop this layer's train dump once its blob is on disk at the right size and finite in its FP32 tail
            chk = np.fromfile(p, np.uint8).reshape(NE, BLOB)
            tail = chk[:, -16:].copy().view(np.float32)
            assert chk.shape == (NE, BLOB) and np.isfinite(tail).all() and (tail[:, :3] > 0).all(), "blob check failed"
            d = os.path.join(a.dump, "moe_l%02d.bin" % l)
            sz = os.path.getsize(d)
            os.remove(d)
            log("layer %d: blob checked, deleted %s (%.1f GB)" % (l, d, sz / 1e9))


# ------------------------------------------------------------------------------------------------------- pack
sys.path.insert(0, r"D:\Projects\basalt-hy4-gptq\tools")


def layout(pack):
    from model_file import ModelFile
    m = ModelFile(pack)
    eoff, esz = m.section("experts")
    lay = {}
    for ln in m.read("experts.layout").decode().splitlines()[1:]:
        f = ln.split()
        lay[int(f[0])] = (int(f[3]), int(f[4]))
    return m, eoff, esz, lay


def cmd_write(a):
    from model_file import ENTRY, HEADER, ModelFile
    if not os.path.exists(a.pack):
        log("copying %s -> %s" % (a.base, a.pack))
        shutil.copyfile(a.base, a.pack + ".tmp")
        os.replace(a.pack + ".tmp", a.pack)
    m, eoff, esz, lay = layout(a.pack)
    with open(a.pack, "r+b") as f:
        for l in sorted(lay):
            p = os.path.join(a.blobs, "l%02d.bin" % l)
            if not os.path.exists(p):
                log("layer %d: no blobs, RTN kept" % l)
                continue
            off, blob = lay[l]
            assert blob == BLOB, (l, blob)
            data = np.fromfile(p, dtype=np.uint8)
            assert data.size == NE * BLOB
            f.seek(eoff + off)
            f.write(data.tobytes())
        f.flush()
    # the experts section's SHA in the table
    h = hashlib.sha256()
    with open(a.pack, "rb") as f:
        f.seek(eoff)
        left = esz
        while left:
            b = f.read(min(64 << 20, left))
            h.update(b)
            left -= len(b)
    with open(a.pack, "r+b") as f:
        f.seek(0)
        hdr = f.read(65536)
        n = HEADER.unpack_from(hdr, 0)
        for i in range((65536 - HEADER.size) // ENTRY.size):
            name, off, size, sha, res = ENTRY.unpack_from(hdr, HEADER.size + i * ENTRY.size)
            if name.rstrip(b"\0") == b"experts":
                f.seek(HEADER.size + i * ENTRY.size)
                f.write(ENTRY.pack(name, off, size, h.digest(), res))
                break
    log("experts section sha256 %s" % h.hexdigest())
    m2 = ModelFile(a.pack)
    log("verify experts: %s" % m2.verify("experts"))


def cmd_analyze(a):
    dev = a.dev
    tab = ue4m3_table(dev)
    bf = Bf16()
    # a --pack entry is a .basalt pack, or a blob dir from `quant` (lNN.bin = [512, BLOB])
    packs = [(p, layout(p) if os.path.isfile(p) else None) for p in a.pack]
    lo, hi = (int(x) for x in a.layers.split("-"))
    res = []
    for l in range(lo, hi + 1, a.step):
        X, I, Wt = load_dump(a.dump, l, dev)
        lists, _ = routed(I, Wt)
        gu_all, d_all = bf.get(l, "gate_up_proj"), bf.get(l, "down_proj")
        acc = {"energy": 0.0}
        fhs = [open(p if lay_ else os.path.join(p, "l%02d.bin" % l), "rb") for p, lay_ in packs]
        for b0 in range(0, a.experts, a.batch):
            B = min(a.batch, a.experts - b0)
            Wgu = gu_all[b0:b0 + B].to(dev).float()
            Wd = d_all[b0:b0 + B].to(dev).float()
            deq = []
            for (p, lay_), fh in zip(packs, fhs):
                if lay_:
                    m, eoff, esz, lay = lay_
                    fh.seek(eoff + lay[l][0] + b0 * BLOB)
                else:
                    fh.seek(b0 * BLOB)
                raw = torch.from_numpy(np.frombuffer(fh.read(B * BLOB), np.uint8).copy()).view(B, BLOB).to(dev)
                tail = raw[:, -16:].contiguous().view(torch.float32)
                gq = torch.cat([from_ggml(raw[:, :GU_BYTES // 2].contiguous(), F, L, tail[:, 0:1].expand(B, F), tab),
                                from_ggml(raw[:, GU_BYTES // 2:GU_BYTES].contiguous(), F, L, tail[:, 1:2].expand(B, F), tab)], 1)
                dq = from_ggml(raw[:, GU_BYTES:GU_BYTES + D_BYTES].contiguous(), L, F, tail[:, 2:3].expand(B, L), tab)
                deq.append((gq, dq))
            for j in range(B):
                rows, w = lists[b0 + j]
                if rows.numel() == 0:
                    continue
                x = X[rows].float()
                w2 = (w ** 2).unsqueeze(1)
                y = swiglu(x @ Wgu[j].T) @ Wd[j].T
                acc["energy"] += float((w2 * y ** 2).sum())
                for k, (gq, dq) in enumerate(deq):
                    yq = swiglu(x @ gq[j].T) @ dq[j].T
                    acc[k] = acc.get(k, 0.0) + float((w2 * (yq - y) ** 2).sum())
        for fh in fhs:
            fh.close()
        r = {"layer": l, "tokens": int(X.shape[0]),
             "rel_mse": {os.path.basename(packs[k][0]): acc[k] / acc["energy"] for k in range(len(packs))}}
        res.append(r)
        log("layer %d: %s" % (l, "  ".join("%s %.5f" % (n, v) for n, v in r["rel_mse"].items())))
        del X, I, Wt
        if dev == "cuda":
            torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["quant", "write", "analyze"])
    ap.add_argument("--dump", default="")
    ap.add_argument("--layers", default="1-41")
    ap.add_argument("--out", default="")
    ap.add_argument("--method", default="gptq", choices=["gptq", "rtn"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--blobs", default="")
    ap.add_argument("--base", default=r"D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt")
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--delete-dumps", action="store_true", help="quant: delete each layer's dump after its blob is checked")
    ap.add_argument("--experts", type=int, default=NE, help="test: only the first N experts")
    a = ap.parse_args()
    if a.mode == "quant":
        cmd_quant(a)
    elif a.mode == "write":
        a.pack = a.pack[0]
        cmd_write(a)
    else:
        cmd_analyze(a)


if __name__ == "__main__":
    main()
