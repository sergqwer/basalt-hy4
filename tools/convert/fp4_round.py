"""The NVFP4 / E4M3 roundings the Hy4 release converter (hy4flash_bf16.py) uses: E4M3, UE4M3 and E2M1 codes to nearest
with ties to even, and the NVFP4 rows with the UE4M3 scale of least squared error per 16 values. Moved here from
checkpoint_qwen35.py (the Qwen3.8-27B converter, removed with that model)."""
from __future__ import annotations

import numpy as np

import model_sections as S

E4M3 = S.fp8_e4m3_table()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
UE4M3 = E4M3[:127].astype(np.float32)     # the finite non-negative codes, ascending
FP4Q_STEPS = range(-8, 17)    # UE4M3 codes tried around amax / 6 per group: half to four times that scale


def e4m3_round(x: np.ndarray) -> np.ndarray:
    """float64 values -> E4M3 codes, to nearest with ties to the even code, saturating at 448 (no -0)."""
    c = ue4m3_round(np.abs(x))
    return c | np.where((x < 0) & (c > 0), 0x80, 0).astype(np.uint8)


def fp4q_global(amax: float) -> np.float32:
    """The tensor scale of a matrix rounded to NVFP4: its largest |value| maps to E2M1 6 at the largest UE4M3 scale."""
    return np.float32(amax) / np.float32(6.0 * 448.0) if amax > 0 else np.float32(1)


def ue4m3_round(x: np.ndarray) -> np.ndarray:
    """Non-negative float32 values -> UE4M3 codes, to nearest with ties to the even code, saturating at 448."""
    x = np.minimum(x, np.float32(448))
    i = np.clip(np.searchsorted(UE4M3, x), 1, 126)
    lo, hi = UE4M3[i - 1], UE4M3[i]
    up = (x - lo > hi - x) | ((x - lo == hi - x) & (i % 2 == 0))
    return np.where(up, i, i - 1).astype(np.uint8)


def e2m1_round(x: np.ndarray) -> np.ndarray:
    """float32 values -> E2M1 codes, to nearest with ties to the even code, saturating (as nvfp4_planes.cu rounds)."""
    a = np.abs(x)
    c = ((a > 0.25).astype(np.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0)).astype(np.uint8)
    return c | np.where((x < 0) & (c > 0), 8, 0).astype(np.uint8)


def fp4q_rows(x: np.ndarray, gs: np.float32):
    """Rows [rows, K] (float32) -> (E2M1 codes [rows, K / 2], UE4M3 scales [rows, K / 16]): each 16-value group takes the
    scale of least squared error among the UE4M3 codes near amax / (6 gs) (the first such when tied, amax / 6 first)."""
    rows, K = x.shape
    g = x.reshape(rows, K // 16, 16)
    c0 = ue4m3_round(np.abs(g).max(-1) / (np.float32(6) * gs)).astype(np.int32)
    best_err = best_code = None
    for d in sorted(FP4Q_STEPS, key=abs):
        c = np.clip(c0 + d, 0, 126)
        sd = UE4M3[c] * gs
        inv = np.where(sd > 0, np.float32(1) / np.where(sd > 0, sd, 1), 0).astype(np.float32)
        q = e2m1_round(g * inv[..., None])
        err = ((E2M1[q].astype(np.float32) * sd[..., None] - g) ** 2).sum(-1)
        if best_err is None:
            best_err, best_code = err, c
        else:
            better = err < best_err
            best_err, best_code = np.where(better, err, best_err), np.where(better, c, best_code)
    sd = UE4M3[best_code] * gs
    inv = np.where(sd > 0, np.float32(1) / np.where(sd > 0, sd, 1), 0).astype(np.float32)
    q = e2m1_round(g * inv[..., None]).reshape(rows, K)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8), best_code.astype(np.uint8)
