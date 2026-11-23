"""hy4_retrofit.py --from PACK --out NEW [--head q8|keep] [--dfly runtime|nvfp4|keep]: a copy of a hy4flash pack whose GPU
tensors need no conversion at load.

  --head q8       the output head (BF16 in older packs) as the Q8_0 bytes the engine's loader made from it (loader_round.q8_0,
                  byte-identical to core/loader.cpp), so the engine reads the same head without rounding it
  --dfly runtime  the DFly2 drafter (stored as its safetensors) in its runtime formats (dfly_attach.write_runtime: F32
                  vectors, Q4_0 / Q8_0 projections, BF16 kv_b, the loader's roundings): the same weights, copied at load
  --dfly nvfp4    the same, with the Q4_0 set as NVFP4 planes (the FP4 window products; not the same weights)
  --dense-nvfp4 RE  the Q8_0 dense matrices whose names match RE (and whose shapes the FP4 kernels take: [K % 256, rows % 128])
                  as NVFP4 planes (type 45) plus their `<name>.scale`: an RTN rounding of the Q8_0 values (fp4_round's
                  least-squares scales), for testing the engine's dense NVFP4 path; a GPTQ pack rounds from BF16 instead
  --embd-bf16 DIR the token embedding as BF16 rows from the checkpoint DIR (model.embed_tokens.weight), copied as stored
                  (an option: +0.35 GB of host RAM against the Q8_0 rows; checked to round to the pack's Q8_0 bytes)

Every other section is copied byte for byte; the `gguf` section is laid out again (each tensor 4096-aligned, in the old
order) when the head changes. The source pack is only read. For a pack made by swapping sections in place (the GPTQ pack):
run this once on it to drop the load-time roundings.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dfly_attach  # noqa: E402
import loader_round  # noqa: E402
from model_file import TENSOR_ALIGN, ModelFile, ModelWriter, Stream  # noqa: E402

HEAD = "output.weight"
CHUNK_VALUES = 1 << 24   # BF16 values per head chunk (whole Q8_0 blocks)


def die(msg: str):
    raise SystemExit(f"hy4_retrofit: {msg}")


def head_chunks(path: str, off: int, n_values: int):
    """The BF16 head at `off` (n_values) as Q8_0 bytes, in chunks."""
    with open(path, "rb") as f:
        f.seek(off)
        for v in range(0, n_values, CHUNK_VALUES):
            n = min(CHUNK_VALUES, n_values - v)
            u = np.frombuffer(f.read(n * 2), dtype="<u2")
            if u.size != n:
                die("short read in the head")
            x = (u.astype(np.uint32) << 16).view(np.float32)
            if not np.isfinite(x).all():
                die("the head holds a non-finite value")
            yield loader_round.q8_0(x)


def _planes_job(args):
    """A Q8_0 matrix [rows][K] (its bytes at `off` of `path`) as NVFP4 planes and its scale."""
    path, off, K, rows = args
    with open(path, "rb") as f:
        f.seek(off)
        b = np.frombuffer(f.read(rows * K // 32 * 34), np.uint8).reshape(-1, 34)
    x = (b[:, :2].copy().view(np.float16).astype(np.float32) * b[:, 2:].view(np.int8).astype(np.float32)).reshape(rows, K)
    return dfly_attach.nvfp4_planes(x)


EMBD = "token_embd.weight"


def checkpoint_tensor(ckpt: str, name: str) -> tuple[str, int, int, list]:
    """(file, data offset, bytes, shape) of a BF16 tensor of a safetensors checkpoint directory."""
    idx = json.loads((Path(ckpt) / "model.safetensors.index.json").read_text())
    f = str(Path(ckpt) / idx["weight_map"][name])
    head, base = dfly_attach.read_header(Path(f))
    t = head[name]
    if t["dtype"] != "BF16":
        die(f"{name} is {t['dtype']} in {f}")
    lo, hi = t["data_offsets"]
    return f, base + lo, hi - lo, t["shape"]


def retrofit(src: str, out: str, head: str, dfly: str, dense_re: str | None = None, workers: int = 8,
             embd_ckpt: str | None = None, only: set | None = None, dfly_from: str | None = None) -> None:
    """`only`: write just these sections (and meta) to `out`, for in_place(). `dfly_from`: the pack whose safetensors
    drafter to use (read only), when `src`'s is already in runtime formats."""
    m = ModelFile(src)
    meta = json.loads(m.read("meta"))
    if meta.get("arch") != "hy4flash":
        die(f"{src} is not a hy4flash pack")
    if Path(out).exists():
        die(f"{out} exists")
    tens = meta["tensors"]
    gg_off, gg_size = m.section("gguf")
    h = tens.get(HEAD)
    if h is None:
        die(f"{src} has no {HEAD}")
    if head == "q8" and h["type"] == "Q8_0":
        head = "keep"
    if head == "q8" and (h["type"] != "BF16" or int(h["shape"][0]) % 32):
        die(f"{HEAD} is {h['type']} {h['shape']}: not BF16 rows of whole Q8_0 blocks")
    dsrc = dfly_from or src
    if dfly_from and dfly != "keep":
        dm = ModelFile(dfly_from)
        rec = json.loads(dm.read("meta")).get("dfly2")
        if "dfly" not in m.names():
            die(f"{src} has no dfly section to replace")
    else:
        rec = meta.get("dfly2")
    if dfly != "keep" and (not rec or rec.get("format") != "safetensors"):
        die("--dfly needs a drafter stored as its safetensors (meta.dfly2.format safetensors)")
    planes = {}   # name -> ((spill file, offset, bytes), bytes, scale): the dense matrices as NVFP4 planes, spilled to disk
    spill = None
    if dense_re:
        pat = re.compile(dense_re)
        todo = [(n, t) for n, t in tens.items() if pat.search(n) and t.get("type") == "Q8_0" and len(t["shape"]) == 2
                and int(t["shape"][0]) % 256 == 0 and int(t["shape"][1]) % 128 == 0 and t.get("section", "gguf") == "gguf"]
        if not todo:
            die(f"--dense-nvfp4 {dense_re!r} matches no Q8_0 matrix of [K % 256, rows % 128]")
        jobs = [(src, gg_off + t["offset"], int(t["shape"][0]), int(t["shape"][1])) for _, t in todo]
        spill = str(Path(out).parent / (Path(out).name + ".planes.tmp"))   # beside the output: RAM holds one matrix per worker
        with multiprocessing.Pool(workers) as pool, open(spill, "wb") as sf:
            for (n, _), (pb, sc) in zip(todo, pool.imap(_planes_job, jobs)):
                planes[n] = ((spill, sf.tell(), len(pb)), len(pb), sc)
                sf.write(pb)
        print(f"  {len(planes)} dense matrices to NVFP4 planes", flush=True)
    embd = None   # (file, offset, bytes) of the BF16 embedding
    if embd_ckpt:
        te = tens[EMBD]
        f, o, nb, shape = checkpoint_tensor(embd_ckpt, "model.embed_tokens.weight")
        if te["type"] != "Q8_0" or list(shape) != [int(te["shape"][1]), int(te["shape"][0])]:
            die(f"the checkpoint's embedding {shape} is not the pack's {te['type']} {te['shape']}")
        E = int(te["shape"][0])
        with open(f, "rb") as a, open(src, "rb") as b:   # a few rows must round to the pack's own Q8_0 rows
            for r in (0, 1, 777, int(shape[0]) - 1):
                a.seek(o + r * E * 2)
                x = (np.frombuffer(a.read(E * 2), "<u2").astype(np.uint32) << 16).view(np.float32)
                b.seek(gg_off + te["offset"] + r * E // 32 * 34)
                mine, theirs = np.frombuffer(loader_round.q8_0(x), np.uint8), np.frombuffer(b.read(E // 32 * 34), np.uint8)
                if (mine != theirs).mean() > 0.01:
                    die(f"row {r} of the checkpoint's embedding does not round to the pack's Q8_0 row")
        embd = (f, o, nb)
    relayout = head == "q8" or bool(planes) or embd is not None
    w = ModelWriter(out)
    tmp = None
    try:
        for name in m.names():
            if name == "meta":
                continue
            off, size = m.section(name)
            if name == "gguf" and relayout:
                order = sorted((t["offset"], n) for n, t in tens.items() if "offset" in t and t.get("section", "gguf") == "gguf")
                parts, names = [], []
                for at, n in order:
                    t = tens[n]
                    if at + t["bytes"] > gg_size:
                        die(f"{n} lies past the gguf section")
                    if n == HEAD and head == "q8":
                        nv = int(np.prod([int(d) for d in t["shape"]]))
                        parts.append(Stream(nv // 32 * 34, head_chunks(src, gg_off + at, nv)))
                    elif n in planes:
                        parts.append(planes[n][0])
                    elif n == EMBD and embd is not None:
                        parts.append(embd)
                    else:
                        parts.append((src, gg_off + at, t["bytes"]))
                    names.append(n)
                for n in planes:   # each planes matrix's FP32 scale, a tensor of its own
                    parts.append(np.array([planes[n][2]], "<f4").tobytes())
                    names.append(n + ".scale")
                offsets = w.add_parts("gguf", parts, TENSOR_ALIGN)
                for n, (_, nb, _) in planes.items():
                    tens[n].update(type="NVFP4_PLANES", type_id=45, bytes=nb)
                    tens[n + ".scale"] = {"type": "F32", "type_id": 0, "shape": [1], "bytes": 4}
                for n, o in zip(names, offsets):
                    tens[n]["offset"] = o
                if head == "q8":
                    nv = int(np.prod([int(d) for d in h["shape"]]))
                    h.update(type="Q8_0", type_id=8, bytes=nv // 32 * 34)
                    meta.setdefault("quantized_from", {})["head"] = "Q8_0 (the engine loader's rounding of the BF16 head)"
                if embd is not None:
                    tens[EMBD].update(type="BF16", type_id=30, bytes=embd[2])
                    meta.setdefault("quantized_from", {})["token_embd"] = "BF16 (the checkpoint's rows)"
                if planes:
                    meta.setdefault("quantized_from", {})["dense_nvfp4"] = {"pattern": dense_re, "tensors": len(planes),
                                                                            "rounding": "RTN of the Q8_0 values (test)"}
                print(f"  gguf: head {head}, {len(planes)} NVFP4 planes, {len(order)} tensors laid out again", flush=True)
            elif name == "dfly" and dfly != "keep":
                tmpd = tempfile.mkdtemp(dir=str(Path(out).parent))
                st = Path(tmpd) / "dfly.safetensors"
                d_off, d_size = (ModelFile(dfly_from).section("dfly") if dfly_from else (off, size))
                with open(dsrc, "rb") as f, open(st, "wb") as o:   # the section is the safetensors file whole
                    f.seek(d_off)
                    left = d_size
                    while left > 0:
                        b = f.read(min(64 << 20, left))
                        o.write(b)
                        left -= len(b)
                rt = Path(tmpd) / "dfly.runtime"
                tensors = dfly_attach.write_runtime(st, rt, int(rec["layers"]), dfly == "nvfp4")
                w.add_file("dfly", str(rt))
                rec.update(format="runtime", bytes=rt.stat().st_size, tensors=tensors)
                meta["dfly2"] = rec
                tmp = (tmpd, st, rt)
                print(f"  dfly: {dfly} ({rt.stat().st_size} bytes)", flush=True)
            elif only is None or name in only:
                w.add_file(name, src, off, size)
                print(f"  {name}: copied", flush=True)
        w.add_bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
    finally:
        if spill:
            Path(spill).unlink(missing_ok=True)
        if tmp:
            tmp[1].unlink(missing_ok=True)
            tmp[2].unlink(missing_ok=True)
            Path(tmp[0]).rmdir()
    if only is not None:
        return
    n = ModelFile(out)
    for s in ("gguf", "dfly", "meta"):
        if s in n.names() and not n.verify(s):
            die(f"{out}: section {s} does not read back as written")
    print(f"done: {out}, {Path(out).stat().st_size} bytes")


def in_place(pack: str, head: str, dfly: str, dense_re: str | None, workers: int, embd_ckpt: str | None,
             dfly_from: str | None = None) -> None:
    """The retrofit of `pack` itself (a copy made for it, e.g. by robocopy /J): the new gguf, dfly and meta sections go where
    the old ones were when they fit, else after the last section; the table and the file size follow. Only the changed
    sections are written, so a 159 GB pack needs about 10 GB of writes."""
    import os
    from model_file import ENTRY, HEADER, HEADER_BYTES, SECTION_ALIGN
    side = pack + ".retrofit.tmp"
    retrofit(pack, side, head, dfly, dense_re, workers, embd_ckpt, only=set(), dfly_from=dfly_from)
    new = ModelFile(side)
    old = ModelFile(pack)
    table = {k: (o, s, old.sections[k][2]) for k, (o, s, _) in old.sections.items()}
    order = list(old.sections)
    al = lambda n: (n + SECTION_ALIGN - 1) // SECTION_ALIGN * SECTION_ALIGN
    with open(pack, "r+b") as f, open(side, "rb") as s:
        end = max(o + n for k, (o, n, _) in table.items() if k != "meta")
        for k in [x for x in new.names() if x != "meta"] + ["meta"]:
            no, ns, nd = new.sections[k]
            oo, os_, _ = table[k]
            at = oo if (k != "meta" and ns <= os_) else al(end)
            s.seek(no)
            f.seek(at)
            left = ns
            while left > 0:
                b = s.read(min(64 << 20, left))
                f.write(b)
                left -= len(b)
            table[k] = (at, ns, nd)
            end = max(end, at + ns)
            print(f"  {k}: {ns} bytes at {at}{' (in place)' if at == oo else ' (moved)'}", flush=True)
        magic, version, _, _, reserved = HEADER.unpack_from(open(pack, "rb").read(HEADER.size), 0)
        meta_end = table["meta"][0] + table["meta"][1]
        head_b = HEADER.pack(magic, version, len(order), meta_end, reserved)
        head_b += b"".join(ENTRY.pack(k.encode(), table[k][0], table[k][1], table[k][2], b"\0" * 16) for k in order)
        f.seek(0)
        f.write(head_b + b"\0" * (HEADER_BYTES - len(head_b)))
        f.truncate(meta_end)
        f.flush()
        os.fsync(f.fileno())
    os.remove(side)
    m = ModelFile(pack)
    for k in ("gguf", "dfly", "meta"):
        if k in m.names() and not m.verify(k):
            die(f"{pack}: section {k} does not read back as written")
    print(f"done in place: {pack}, {Path(pack).stat().st_size} bytes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--head", choices=("q8", "keep"), default="q8")
    ap.add_argument("--dfly", choices=("runtime", "nvfp4", "keep"), default="runtime")
    ap.add_argument("--dense-nvfp4", metavar="RE", help="Q8_0 dense matrices matching RE as NVFP4 planes (a test rounding)")
    ap.add_argument("--embd-bf16", metavar="DIR", help="the token embedding as BF16 from this safetensors checkpoint")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dfly-from", help="take the safetensors drafter from this pack (read only)")
    ap.add_argument("--in-place", action="store_true", help="rewrite --from itself (a copy made for it); --out is not used")
    a = ap.parse_args()
    if a.in_place:
        in_place(a.src, a.head, a.dfly, a.dense_nvfp4, a.workers, a.embd_bf16, a.dfly_from)
    else:
        retrofit(a.src, a.out, a.head, a.dfly, a.dense_nvfp4, a.workers, a.embd_bf16, dfly_from=a.dfly_from)


if __name__ == "__main__":
    main()
