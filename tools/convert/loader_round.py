"""The engine loader's BF16 roundings (src/core/loader.cpp quantize_bf16_q8_0 / quantize_bf16_q4_0) in numpy, byte for byte.

A pack that stores these bytes gives the engine what it used to make at load from the BF16 tensors (the output head, the
DFly2 drafter's projections), so it converts nothing as it loads and runs the same weights. The C++ is plain float32
arithmetic (MSVC /fp:precise, no contraction): d = amax / 127 or (Q4_0) the largest-magnitude value / -8, id = 1 / d, and
the fp16 of d to nearest even; Q8_0 codes are std::round(x * id), half away from zero; Q4_0 codes min(15, (int8) (x * id +
8.5)), truncated."""
from __future__ import annotations

import numpy as np

F32 = np.float32


def _scales(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = d.astype(F32)
    inv = np.where(d != 0, F32(1.0) / np.where(d != 0, d, F32(1.0)), F32(0.0)).astype(F32)
    return d, inv


def q8_0(x: np.ndarray) -> bytes:
    """float32 values (a multiple of 32, BF16 numbers) as Q8_0 blocks: 34 bytes per 32."""
    b = np.asarray(x, dtype=F32).reshape(-1, 32)
    d, inv = _scales(np.abs(b).max(axis=1) / F32(127.0))
    v = (b * inv[:, None]).astype(F32).astype(np.float64)   # the float32 product, exactly
    q = np.trunc(v + np.copysign(0.5, v)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def q4_0(x: np.ndarray) -> bytes:
    """float32 values (a multiple of 32) as Q4_0 blocks: 18 bytes per 32, value i in the low nibble of byte i, value
    16 + i in its high one."""
    b = np.asarray(x, dtype=F32).reshape(-1, 32)
    m = b[np.arange(b.shape[0]), np.argmax(np.abs(b), axis=1)]   # the first value of the largest magnitude
    d, inv = _scales(m / F32(-8.0))
    t = ((b * inv[:, None]).astype(F32) + F32(8.5)).astype(F32)
    q = np.minimum(15, np.trunc(t).astype(np.int64).astype(np.int8).astype(np.int64)).astype(np.uint8)
    out = np.empty((b.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()
