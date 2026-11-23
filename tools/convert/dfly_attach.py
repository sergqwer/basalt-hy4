"""--dfly FILE --into PACK.basalt: the DFly2 drafter joins a hy4flash pack as the `dfly` section.

FILE is a safetensors file holding the drafter's tensors (`model.dfly_layers.*`, 87 BF16 tensors, 1.69 GB): the last shard of
tencent/Hy4-flash, or the side file extracted from the checkpoint. It is stored whole; meta.dfly2 records where each tensor
lies (offsets from the start of the section), the drafter's settings from the pack's own meta.hf_config (config.json
`dfly_*` keys) and the shapes the engine checks. Tensors of the file that are not the drafter's are left unread.
Like --dspark: no other section moves, and --roll-back restores the pack bit for bit.

By default the section holds each tensor in the format the engine runs it in (meta.dfly2.format "runtime"): the norms and
vectors as F32, the attention's gate and output and the MLP as Q4_0, the other projections as Q8_0, kv_b as BF16, with the
engine loader's own roundings (loader_round.py), so the engine copies the drafter as stored and converts nothing at load.
--dfly-bf16 keeps the file whole (format "safetensors"), which the engine rounds as it loads. --dfly-nvfp4 stores the Q4_0
set as NVFP4 planes instead (dtype "NVFP4P": the pack's type-45 layout, kernels/nvfp4_planes.hpp, the experts' rounding with
one FP32 scale a matrix, recorded as the tensor's "scale"), which the engine runs on its FP4 window products.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

import loader_round

PREFIX = "model.dfly_layers."
SECTION = "dfly"


def die(msg: str):
    raise SystemExit(f"convert: {msg}")


def read_header(path: Path) -> tuple[dict, int]:
    """The safetensors header (without __metadata__) and the file offset where the tensor data starts."""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            die(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n > 64 << 20:
            die(f"{path}: not a safetensors file")
        try:
            head = json.loads(f.read(n))
        except ValueError:
            die(f"{path}: not a safetensors file")
    head.pop("__metadata__", None)
    return head, 8 + n


def wanted(layers: int) -> set[str]:
    """The drafter's tensor names, without the model.dfly_layers. prefix."""
    names = {f"{i}.{s}" for i in range(layers) for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight", "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight",
        "self_attn.linear_gate.weight", "self_attn.o_proj.weight", "self_attn.post_proj_norm.weight",
        "self_attn.learnable_sink_param")}
    names |= {"final_norm.weight", "mtp_fusion.eh_proj.weight", "mtp_fusion.enorm.weight", "mtp_fusion.hnorm.weight",
              "mtp_fusion.gate", "hidden_correction.gate_proj.weight", "hidden_correction.up_proj.weight",
              "hidden_correction.down_proj.weight", "hidden_correction.hidden_norm.weight",
              "hidden_correction.embed_norm.weight", "confidence_head.proj.weight", "confidence_head.proj.bias"}
    return names


# The engine's format of each drafter tensor (hy4flash/dfly_meta.cpp dfly_specs): F32 vectors, Q4_0 / Q8_0 projections, BF16 kv_b
_Q4 = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "self_attn.linear_gate.weight", "self_attn.o_proj.weight")
_Q8 = ("self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight", "mtp_fusion.eh_proj.weight",
       "hidden_correction.gate_proj.weight", "hidden_correction.up_proj.weight", "hidden_correction.down_proj.weight")
RUNTIME_ALIGN = 256


def runtime_dtype(name: str) -> str:
    base = name.split(".", 1)[1] if name.split(".", 1)[0].isdigit() else name
    if base in _Q4:
        return "Q4_0"
    if base in _Q8:
        return "Q8_0"
    if base == "self_attn.kv_b_proj.weight":
        return "BF16"
    return "F32"


def nvfp4_planes(x: np.ndarray) -> tuple[bytes, float]:
    """A matrix [rows, K] (rows a multiple of 128, K of 64) as NVFP4 planes: the codes [rows][K / 2], then the UE4M3 scales
    in 512-byte tiles of 128 rows x 4 groups (nvfp4_planes.hpp scale_index); the experts' rounding (fp4_round.fp4q_rows)
    under the matrix scale it returns."""
    import fp4_round as Q
    rows, K = x.shape
    if rows % 128 or K % 64:
        die(f"an NVFP4 drafter matrix of {rows} x {K} is not whole 128-row, 64-value tiles")
    gs = Q.fp4q_global(float(np.abs(x).max()))
    codes, sc = Q.fp4q_rows(x.astype(np.float32), gs)
    r = np.arange(rows)[:, None]
    g = np.arange(K // 16)[None, :]
    idx = ((r // 128) * (K // 64) + g // 4) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4 + (g & 3)
    planes = np.empty(rows * K // 16, np.uint8)
    planes[idx.reshape(-1)] = sc.reshape(-1)
    return codes.tobytes() + planes.tobytes(), float(gs)


def runtime_bytes(raw: bytes, dtype: str) -> bytes:
    """A tensor's BF16 bytes in its runtime format."""
    u = np.frombuffer(raw, dtype="<u2")
    if dtype == "BF16":
        return bytes(raw)
    x = (u.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F32":
        return x.astype("<f4").tobytes()
    if u.size % 32:
        die(f"a drafter projection of {u.size} values is not whole {dtype} blocks")
    return loader_round.q8_0(x) if dtype == "Q8_0" else loader_round.q4_0(x)


def write_runtime(path: Path, out: Path, layers: int, nvfp4: bool = False) -> dict:
    """The drafter's tensors of `path` in their runtime formats, one after another (each RUNTIME_ALIGN aligned), into `out`;
    returns their meta.dfly2 records (offsets from the start of `out`)."""
    head, base = read_header(path)
    tensors = {}
    at = 0
    with path.open("rb") as f, out.open("wb") as o:
        for name in sorted(wanted(layers)):
            t = head.get(PREFIX + name)
            if t is None:
                die(f"{path} lacks drafter tensor {name}")
            if t["dtype"] != "BF16":
                die(f"{path}: {PREFIX + name} is {t['dtype']}; the engine reads the drafter's BF16 tensors")
            lo, hi = t["data_offsets"]
            f.seek(base + lo)
            dt = runtime_dtype(name)
            raw = f.read(hi - lo)
            scale = None
            if nvfp4 and dt == "Q4_0":
                dt = "NVFP4P"
                x = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(t["shape"][0], -1)
                b, scale = nvfp4_planes(x)
            else:
                b = runtime_bytes(raw, dt)
            pad = -at % RUNTIME_ALIGN
            o.write(b"\0" * pad)
            at += pad
            o.write(b)
            tensors[name] = {"dtype": dt, "shape": t["shape"], "offset": at, "bytes": len(b), **({"scale": scale} if scale is not None else {})}
            at += len(b)
    return tensors


def dfly_record(path: Path, hf_config: dict | None, runtime: tuple[Path, dict] | None = None) -> dict:
    """meta.dfly2; `runtime`: the section is write_runtime's file (its path and tensors), not `path` whole."""
    rec = _dfly_record(path, hf_config)
    if runtime is not None:
        out, tensors = runtime
        if set(tensors) != set(rec["tensors"]):
            die("the runtime drafter's tensors are not the drafter's")
        rec.update(format="runtime", bytes=out.stat().st_size, tensors=tensors)
    return rec


def _dfly_record(path: Path, hf_config: dict | None) -> dict:
    """meta.dfly2 for the file: what the engine loads by, from the file's header and the checkpoint's config."""
    cfg = hf_config or {}
    text = {**cfg.get("text_config", {}), **cfg}      # the released config.json keeps the dfly_* keys at its top level
    for k in ("dfly_num_hidden_layers", "dfly_mask_token_id", "dfly_target_kv_layers", "dfly_target_index_layer"):
        if text.get(k) is None:
            die(f"the pack's meta.hf_config has no {k}: the pack was made without --hf-dir, or not from Hy4-flash")
    layers = int(text["dfly_num_hidden_layers"])
    if text.get("dfly_mtp_fusion", "slot0") != "slot0" or text.get("dfly_enable_hidden_correction", True) is not True:
        die("the engine runs slot0 fusion with hidden correction, the only drafter configuration there is")
    head, base = read_header(path)
    tensors = {}
    for name, t in head.items():
        if not name.startswith(PREFIX):
            continue
        if t["dtype"] != "BF16":
            die(f"{path}: {name} is {t['dtype']}; the engine reads the drafter's BF16 tensors")
        lo, hi = t["data_offsets"]
        tensors[name[len(PREFIX):]] = {"dtype": "BF16", "shape": t["shape"], "offset": base + lo, "bytes": hi - lo}
    if not tensors:
        die(f"{path} holds no {PREFIX}* tensors")
    want = wanted(layers)
    if missing := sorted(want - tensors.keys()):
        die(f"{path} lacks drafter tensors: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    window = text.get("dfly_context_local_window")
    return {
        "section": SECTION, "format": "safetensors", "bytes": path.stat().st_size, "layers": layers,
        "mask_token_id": int(text["dfly_mask_token_id"]),
        "target_kv_layers": [int(x) for x in text["dfly_target_kv_layers"]],
        "target_index_layers": [int(x) for x in text["dfly_target_index_layer"]],
        "target_index_topk": int(text.get("dfly_target_index_topk") or 2048),
        "context_local_window": int(window) if window else 0,
        "tensors": {k: v for k, v in tensors.items() if k in want},
    }


def add_dfly(a, source_record=None) -> None:
    """--dfly with --into: the drafter joins a pack that exists, whose other sections stay where and as they are."""
    from model_file import ModelFile, append_section
    pack = Path(a.into)
    if not pack.is_file():
        die(f"missing input: {pack}")
    src = Path(a.dfly)
    if not src.is_file():
        die(f"missing input: {src}")
    m = ModelFile(str(pack))
    meta = json.loads(m.read("meta"))
    if meta.get("arch") != "hy4flash":
        die(f"{pack} is no hy4flash pack")
    if SECTION in m.names() or meta.get("dfly2"):
        die(f"{pack} has a drafter already")
    section = src
    runtime = None
    if not getattr(a, "dfly_bf16", False):   # the runtime formats, beside the pack while it is appended
        section = pack.with_name(pack.name + ".dfly-runtime.tmp")
        layers = int(_dfly_record(src, meta.get("hf_config"))["layers"])
        runtime = (section, write_runtime(src, section, layers, bool(getattr(a, "dfly_nvfp4", False))))
    meta["dfly2"] = dfly_record(src, meta.get("hf_config"), runtime)
    if source_record is not None:
        meta.setdefault("sources", []).append(source_record(src))
    try:
        append_section(str(pack), SECTION, str(section), json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), a.undo_dir)
    except (ValueError, FileExistsError) as e:
        die(str(e))
    finally:
        if runtime is not None:
            section.unlink(missing_ok=True)
    m = ModelFile(str(pack))
    if not m.verify(SECTION) or not m.verify("meta"):
        die(f"{pack}: the new sections do not read back as written; --roll-back puts the file back")
    print(f"done: {pack} is {pack.stat().st_size} bytes, {len(m.names())} sections")
