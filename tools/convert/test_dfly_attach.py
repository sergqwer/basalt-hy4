"""--dfly --into: a synthetic safetensors drafter joins a synthetic hy4flash pack; the record, the offsets, roll-back."""
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
import convert  # noqa: E402
import dfly_attach  # noqa: E402
import loader_round  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402

CONFIG = {"text_config": {"dfly_num_hidden_layers": 2, "dfly_mask_token_id": 7, "dfly_target_kv_layers": [3, 5],
                          "dfly_target_index_layer": [3, 5], "dfly_target_index_topk": 2048,
                          "dfly_context_local_window": None, "dfly_mtp_fusion": "slot0",
                          "dfly_enable_hidden_correction": True}}


def bf16(n, seed):
    r = random.Random(seed)
    return b"".join(struct.pack("<f", r.gauss(0, 1))[2:] for _ in range(n))


def write_safetensors(path, names, extra=(), dtype="BF16", shape=(2, 3)):
    head, data = {}, b""
    for i, n in enumerate(list(names) + list(extra)):
        raw = bf16(shape[0] * shape[1], i)
        head[n] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    h = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)


class DflyAttachTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.pack = self.t / "p.basalt"
        w = ModelWriter(str(self.pack))
        w.add_bytes("dense", b"d" * 100)
        w.add_bytes("meta", json.dumps({"arch": "hy4flash", "hf_config": CONFIG, "sources": []}).encode())
        w.close()
        self.names = [dfly_attach.PREFIX + n for n in sorted(dfly_attach.wanted(2))]
        self.st = self.t / "dfly.safetensors"

    def args(self, **kw):
        d = dict(into=str(self.pack), dfly=str(self.st), undo_dir=str(self.t / "undo"), roll_back=False)
        d.update(kw)
        return type("A", (), d)()

    def test_the_file_is_stored_whole_and_each_tensor_is_found_by_its_offset(self):
        write_safetensors(self.st, self.names, extra=["model.layers.0.other"])
        before = ModelFile(str(self.pack))
        dense = before.read("dense")
        dfly_attach.add_dfly(self.args(dfly_bf16=True), convert.source_record)
        m = ModelFile(str(self.pack))
        self.assertTrue(m.verify("dfly"))
        self.assertEqual(m.read("dense"), dense)
        self.assertEqual(m.read("dfly"), self.st.read_bytes())
        meta = json.loads(m.read("meta"))
        rec = meta["dfly2"]
        self.assertEqual((rec["layers"], rec["mask_token_id"], rec["target_kv_layers"]), (2, 7, [3, 5]))
        self.assertEqual(len(rec["tensors"]), len(self.names))
        self.assertNotIn("other", " ".join(rec["tensors"]))
        sec = m.read("dfly")
        for name, t in rec["tensors"].items():
            self.assertEqual(sec[t["offset"]:t["offset"] + t["bytes"]], bf16(6, self.names.index(dfly_attach.PREFIX + name)))
        convert.add_dspark(type("A", (), dict(into=str(self.pack), dspark=None, undo_dir=str(self.t / "undo"), roll_back=True))())
        self.assertEqual(sorted(ModelFile(str(self.pack)).names()), sorted(before.names()))

    def test_the_runtime_formats_are_the_loaders_roundings(self):
        write_safetensors(self.st, self.names, shape=(2, 64))
        dfly_attach.add_dfly(self.args(), convert.source_record)
        m = ModelFile(str(self.pack))
        self.assertTrue(m.verify("dfly"))
        rec = json.loads(m.read("meta"))["dfly2"]
        self.assertEqual(rec["format"], "runtime")
        self.assertFalse(any(p.name.endswith(".tmp") for p in self.t.iterdir()))
        sec = m.read("dfly")
        seen = set()
        for name, t in rec["tensors"].items():
            raw = bf16(128, self.names.index(dfly_attach.PREFIX + name))
            x = np.frombuffer(b"".join(b"\0\0" + raw[i:i + 2] for i in range(0, len(raw), 2)), "<f4")
            want = {"F32": x.tobytes(), "BF16": raw, "Q8_0": loader_round.q8_0(x), "Q4_0": loader_round.q4_0(x)}[t["dtype"]]
            self.assertEqual(t["offset"] % dfly_attach.RUNTIME_ALIGN, 0, name)
            self.assertEqual(sec[t["offset"]:t["offset"] + t["bytes"]], want, name)
            seen.add(t["dtype"])
        self.assertEqual(seen, {"F32", "BF16", "Q8_0", "Q4_0"})
        self.assertEqual(rec["tensors"]["0.mlp.gate_proj.weight"]["dtype"], "Q4_0")
        self.assertEqual(rec["tensors"]["mtp_fusion.eh_proj.weight"]["dtype"], "Q8_0")
        self.assertEqual(rec["tensors"]["1.self_attn.kv_b_proj.weight"]["dtype"], "BF16")
        self.assertEqual(rec["tensors"]["confidence_head.proj.bias"]["dtype"], "F32")

    def test_the_nvfp4_drafter_stores_the_q4_set_as_planes(self):
        import fp4_round as Q
        write_safetensors(self.st, self.names, shape=(128, 64))
        dfly_attach.add_dfly(self.args(dfly_nvfp4=True), convert.source_record)
        m = ModelFile(str(self.pack))
        rec = json.loads(m.read("meta"))["dfly2"]
        sec = m.read("dfly")
        n4 = 0
        for name, t in rec["tensors"].items():
            want = dfly_attach.runtime_dtype(name)
            self.assertEqual(t["dtype"], "NVFP4P" if want == "Q4_0" else want, name)
            if t["dtype"] != "NVFP4P":
                continue
            n4 += 1
            raw = bf16(128 * 64, self.names.index(dfly_attach.PREFIX + name))
            x = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(128, 64)
            b = sec[t["offset"]:t["offset"] + t["bytes"]]
            self.assertEqual(len(b), 128 * 64 // 64 * 36)
            codes = np.frombuffer(b[:128 * 32], np.uint8).reshape(128, 32)
            sc = np.frombuffer(b[128 * 32:], np.uint8)
            vals = np.empty((128, 64))
            vals[:, 0::2], vals[:, 1::2] = Q.E2M1[codes & 15], Q.E2M1[codes >> 4]
            for r in (0, 31, 32, 77, 127):
                for g in range(4):
                    i = (r & 31) * 16 + ((r & 127) >> 5) * 4 + g   # one 128-row tile, K / 64 = 1
                    vals[r, 16 * g:16 * g + 16] *= float(Q.UE4M3[sc[i]]) * t["scale"]
                err = np.abs(vals[r] - x[r]).max()
                self.assertLessEqual(err, np.abs(x[r]).max() / 4 + 1e-6, (name, r))
        self.assertEqual(n4, 2 * 5)

    def test_the_released_config_keeps_the_dfly_keys_at_its_top_level(self):
        write_safetensors(self.st, self.names)
        rec = dfly_attach.dfly_record(self.st, {"text_config": {}, **CONFIG["text_config"]})
        self.assertEqual((rec["layers"], rec["mask_token_id"], rec["target_index_layers"]), (2, 7, [3, 5]))

    def test_a_missing_tensor_a_wrong_dtype_and_a_second_drafter_are_refused(self):
        write_safetensors(self.st, self.names[:-1])
        with self.assertRaises(SystemExit):
            dfly_attach.add_dfly(self.args())
        write_safetensors(self.st, self.names, dtype="F32")
        with self.assertRaises(SystemExit):
            dfly_attach.add_dfly(self.args())
        self.assertFalse((self.t / "undo").exists())
        write_safetensors(self.st, self.names, shape=(2, 64))
        dfly_attach.add_dfly(self.args())
        with self.assertRaises(SystemExit):
            dfly_attach.add_dfly(self.args(undo_dir=str(self.t / "undo2")))


if __name__ == "__main__":
    unittest.main()
