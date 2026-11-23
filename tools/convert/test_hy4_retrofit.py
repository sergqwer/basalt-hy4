"""hy4_retrofit on a synthetic hy4flash pack: the BF16 head to the loader's Q8_0, the drafter to its runtime formats, dense
Q8_0 matrices to NVFP4 planes with their scales, every other tensor and section unchanged, the source untouched."""
import json
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dfly_attach  # noqa: E402
import fp4_round as Q  # noqa: E402
import hy4_retrofit as RF  # noqa: E402
import loader_round  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402


def bf16(n, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.05, n).astype(np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()


class RetrofitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.src = self.t / "src.basalt"
        # gguf: a BF16 head [64 x 4 rows], a Q8_0 matrix [256 x 128 rows], an F32 vector; 4096-aligned
        self.head = bf16(64 * 4, 1)
        x = np.random.default_rng(2).normal(0, 0.1, 256 * 128).astype(np.float32)
        self.q8 = loader_round.q8_0(x)
        self.vec = np.arange(8, dtype="<f4").tobytes()
        parts = {"output.weight": ("BF16", 30, [64, 4], self.head), "blk.0.attn_q_b.weight": ("Q8_0", 8, [256, 128], self.q8),
                 "blk.0.attn_norm.weight": ("F32", 0, [8], self.vec)}
        gguf, tens = b"", {}
        for n, (ty, tid, sh, b) in parts.items():
            gguf += b"\0" * (-len(gguf) % 4096)
            tens[n] = {"type": ty, "type_id": tid, "shape": sh, "bytes": len(b), "offset": len(gguf)}
            gguf += b
        tens["engram.weight"] = {"type": "F8_E4M3_RAW", "type_id": 43, "shape": [4], "bytes": 4, "offset": 0, "section": "engram"}
        # the drafter as its safetensors, two layers of tiny 128 x 256 tensors
        names = sorted(dfly_attach.wanted(2))
        head, data = {}, b""
        for i, n in enumerate(names):
            raw = bf16(128 * 256, 10 + i)
            head[dfly_attach.PREFIX + n] = {"dtype": "BF16", "shape": [128, 256], "data_offsets": [len(data), len(data) + len(raw)]}
            data += raw
        hj = json.dumps(head).encode()
        self.st = struct.pack("<Q", len(hj)) + hj + data
        self.names = names
        st_path = self.t / "d.safetensors"
        st_path.write_bytes(self.st)
        cfg = {"dfly_num_hidden_layers": 2, "dfly_mask_token_id": 7, "dfly_target_kv_layers": [3, 5], "dfly_target_index_layer": [3, 5]}
        rec = dfly_attach.dfly_record(st_path, cfg)
        w = ModelWriter(str(self.src))
        w.add_bytes("dense", b"d" * 100)
        w.add_bytes("gguf", gguf)
        w.add_bytes("engram", b"\1\2\3\4")
        w.add_bytes("dfly", self.st)
        w.add_bytes("meta", json.dumps({"arch": "hy4flash", "tensors": tens, "dfly2": rec}).encode())
        w.close()
        self.before = self.src.read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def tensor(self, m, meta, n):
        t = meta["tensors"][n]
        off, _ = m.section(t.get("section", "gguf"))
        with open(m.path, "rb") as f:
            f.seek(off + t["offset"])
            return f.read(t["bytes"]), t

    def test_retrofit(self):
        out = self.t / "new.basalt"
        RF.retrofit(str(self.src), str(out), "q8", "runtime", r"attn_q_b", workers=1)
        self.assertEqual(self.src.read_bytes(), self.before)
        m = ModelFile(str(out))
        meta = json.loads(m.read("meta"))
        raw, t = self.tensor(m, meta, "output.weight")
        x = (np.frombuffer(self.head, "<u2").astype(np.uint32) << 16).view(np.float32)
        self.assertEqual((t["type"], t["type_id"], raw), ("Q8_0", 8, loader_round.q8_0(x)))
        raw, t = self.tensor(m, meta, "blk.0.attn_norm.weight")
        self.assertEqual(raw, self.vec)
        self.assertEqual(m.read("engram"), b"\1\2\3\4")
        self.assertEqual(m.read("dense"), b"d" * 100)
        raw, t = self.tensor(m, meta, "blk.0.attn_q_b.weight")
        self.assertEqual((t["type_id"], t["bytes"]), (45, 128 * 256 // 64 * 36))
        s, ts = self.tensor(m, meta, "blk.0.attn_q_b.weight.scale")
        self.assertEqual((ts["type_id"], ts["shape"]), (0, [1]))
        for o in (t["offset"], ts["offset"]):
            self.assertEqual(o % 4096, 0)
        b = np.frombuffer(self.q8, np.uint8).reshape(-1, 34)
        ref = (b[:, :2].copy().view(np.float16).astype(np.float32) * b[:, 2:].view(np.int8)).reshape(128, 256)
        want, scale = dfly_attach.nvfp4_planes(ref)
        self.assertEqual(raw, want)
        self.assertEqual(np.frombuffer(s, "<f4")[0], np.float32(scale))
        # the planes decode to the Q8_0 values within the FP4 rounding
        codes = np.frombuffer(raw[:128 * 128], np.uint8).reshape(128, 128)
        planes = np.frombuffer(raw[128 * 128:], np.uint8)
        dec = np.empty((128, 256))
        dec[:, 0::2], dec[:, 1::2] = Q.E2M1[codes & 15], Q.E2M1[codes >> 4]
        for r in range(128):
            for g in range(16):
                i = ((r // 128) * 4 + g // 4) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4 + (g & 3)
                dec[r, 16 * g:16 * g + 16] *= float(Q.UE4M3[planes[i]]) * scale
        self.assertLess(np.sqrt(((dec - ref) ** 2).mean() / (ref ** 2).mean()), 0.12)
        rec = meta["dfly2"]
        self.assertEqual(rec["format"], "runtime")
        sec = m.read("dfly")
        for n, r in rec["tensors"].items():
            src = bf16(128 * 256, 10 + self.names.index(n))
            self.assertEqual(sec[r["offset"]:r["offset"] + r["bytes"]], dfly_attach.runtime_bytes(src, r["dtype"]), n)
        for s_ in ("gguf", "dfly", "meta", "engram", "dense"):
            self.assertTrue(m.verify(s_), s_)

    def test_in_place_matches_the_copy(self):
        import shutil
        out = self.t / "new.basalt"
        RF.retrofit(str(self.src), str(out), "q8", "runtime", r"attn_q_b", workers=1)
        cp = self.t / "cp.basalt"
        shutil.copyfile(self.src, cp)
        RF.in_place(str(cp), "q8", "runtime", r"attn_q_b", 1, None)
        a, b = ModelFile(str(out)), ModelFile(str(cp))
        ma, mb = json.loads(a.read("meta")), json.loads(b.read("meta"))
        self.assertEqual(ma, mb)
        for n in ma["tensors"]:
            self.assertEqual(self.tensor(a, ma, n)[0], self.tensor(b, mb, n)[0], n)
        for s_ in a.names():
            self.assertEqual(a.read(s_), b.read(s_), s_)
            self.assertTrue(b.verify(s_), s_)
        self.assertFalse(Path(str(cp) + ".retrofit.tmp").exists())

    def test_planes_from_a_directory(self):
        import shutil
        d = self.t / "planes"
        d.mkdir()
        n = "blk.0.attn_q_b.weight"
        pb = bytes(random.Random(5).randrange(256) for _ in range(128 * 256 // 64 * 36))
        (d / (n + ".planes")).write_bytes(pb)
        (d / (n + ".json")).write_text(json.dumps({"gs": 0.0123}))
        cp = self.t / "cp.basalt"
        shutil.copyfile(self.src, cp)
        RF.PLANES_FROM = str(d)
        try:
            RF.in_place(str(cp), "keep", "keep", r"attn_q_b", 1, None)
        finally:
            RF.PLANES_FROM = None
        self.assertEqual(self.src.read_bytes(), self.before)
        m = ModelFile(str(cp))
        meta = json.loads(m.read("meta"))
        raw, t = self.tensor(m, meta, n)
        self.assertEqual((t["type_id"], raw), (45, pb))
        s, _ = self.tensor(m, meta, n + ".scale")
        self.assertEqual(np.frombuffer(s, "<f4")[0], np.float32(0.0123))
        self.assertEqual(self.tensor(m, meta, "output.weight")[0], self.head)
        self.assertEqual(m.read("dfly"), self.st)
        for s_ in m.names():
            self.assertTrue(m.verify(s_), s_)
        (d / (n + ".planes")).write_bytes(pb[:-1])   # a missing or short file stops the write
        RF.PLANES_FROM = str(d)
        try:
            with self.assertRaises(SystemExit):
                RF.retrofit(str(self.src), str(self.t / "x.basalt"), "keep", "keep", r"attn_q_b", workers=1)
        finally:
            RF.PLANES_FROM = None

    def test_dfly_from_redoes_a_runtime_drafter(self):
        import shutil
        out = self.t / "n4.basalt"
        RF.retrofit(str(self.src), str(out), "q8", "nvfp4", workers=1)
        cp = self.t / "cp2.basalt"
        shutil.copyfile(self.src, cp)
        RF.in_place(str(cp), "q8", "runtime", None, 1, None)
        RF.in_place(str(cp), "keep", "nvfp4", None, 1, None, dfly_from=str(self.src))
        a, b = ModelFile(str(out)), ModelFile(str(cp))
        self.assertEqual(json.loads(a.read("meta"))["dfly2"], json.loads(b.read("meta"))["dfly2"])
        self.assertEqual(a.read("dfly"), b.read("dfly"))
        self.assertEqual(a.read("gguf"), b.read("gguf"))


if __name__ == "__main__":
    unittest.main()
