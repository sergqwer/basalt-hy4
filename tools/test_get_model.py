"""tools/get_model.py: the pack-name map, missing-file detection, the disk check, temp + rename + size/hash checks and
retries, all without the network; BASALT_NET_TESTS=1 adds one real download (the GPTQ repo's LICENSE) through the
same code path.

    python -m unittest tools.test_get_model
"""
import hashlib
import json
import os
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

from tools import get_model as gm

Usage = namedtuple("Usage", "total used free")
GPTQ = "Maximilian228/Hy4-flash-NVFP4-GPTQ-Basalt"


def fake_info(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_sha1": None}


def writer(data: bytes, fail_first: int = 0, corrupt_first: int = 0):
    """A downloader that writes `data` into the folder, after failing `fail_first` times and writing bad bytes
    `corrupt_first` times; it records how often it ran."""
    calls = []

    def dl(repo, filename, folder):
        calls.append((repo, filename))
        n = len(calls)
        if n <= fail_first:
            raise ConnectionError(f"fake network failure {n}")
        out = Path(folder) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" + data[1:] if n <= fail_first + corrupt_first else data)
        return out
    dl.calls = calls
    return dl


class Sources(unittest.TestCase):
    def test_known_pack_names(self):
        self.assertEqual(gm.source_for("models/Hy4-flash-NVFP4-GPTQ.basalt"), (GPTQ, "Hy4-flash-NVFP4-GPTQ.basalt"))
        self.assertEqual(gm.source_for("/x/Hy4-flash-NVFP4.basalt"), ("jesdga/Hy4-flash-Basalt", "Hy4-flash-NVFP4.basalt"))
        self.assertIsNone(gm.source_for("models/mine.basalt"))

    def test_entry_source_wins(self):
        self.assertEqual(gm.source_for("models/Hy4-flash-NVFP4.basalt", "me/repo/sub/f.basalt"), ("me/repo", "sub/f.basalt"))

    def test_parse_source(self):
        self.assertEqual(gm.parse_source(gm.DEFAULT), (GPTQ, "Hy4-flash-NVFP4-GPTQ.basalt"))
        self.assertEqual(gm.parse_source("Hy4-flash-NVFP4.basalt"), ("jesdga/Hy4-flash-Basalt", "Hy4-flash-NVFP4.basalt"))
        for bad in ("repo/file", "a//b", ""):
            with self.assertRaises(ValueError):
                gm.parse_source(bad)


class Disk(unittest.TestCase):
    def test_refuses_without_room(self):
        free = lambda _: Usage(0, 0, 160 * 10**9)   # noqa: E731
        with self.assertRaises(gm.DiskError) as c:
            gm.check_disk(Path("."), 158_746_168_309, 0, free)
        self.assertIn("free", str(c.exception))
        self.assertEqual(c.exception.code, 3)

    def test_counts_the_partial_download(self):
        free = lambda _: Usage(0, 0, 10 * 10**9)   # noqa: E731
        gm.check_disk(Path("."), 158_746_168_309, 155 * 10**9, free)   # 3.7 GB left + 5 GiB margin fits in 10 GB

    def test_margin_is_five_gib_or_one_percent(self):
        need = 158_746_168_309   # 1% is 1.6 GB: the 5 GiB floor applies
        gm.check_disk(Path("."), need, 0, lambda _: Usage(0, 0, need + (5 << 30)))
        with self.assertRaises(gm.DiskError):
            gm.check_disk(Path("."), need, 0, lambda _: Usage(0, 0, need + (5 << 30) - 1))
        need = 10**12            # 1% is 10 GB
        gm.check_disk(Path("."), need, 0, lambda _: Usage(0, 0, need + need // 100))
        with self.assertRaises(gm.DiskError):
            gm.check_disk(Path("."), need, 0, lambda _: Usage(0, 0, need + need // 100 - 1))


class Fetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.data = os.urandom(300_000)
        self.sleeps = []
        self.kw = dict(sleep=self.sleeps.append, free_fn=lambda _: Usage(0, 0, 10**12), progress_every=0.05)

    def tearDown(self):
        self.tmp.cleanup()

    def test_temp_then_rename_after_the_check(self):
        target = self.dir / "m.basalt"
        seen = []

        def dl(repo, filename, folder):
            out = Path(folder) / filename
            out.write_bytes(self.data)
            seen.append((out.parent.name, target.exists()))
            return out
        gm.fetch("o/r", "m.basalt", target, fake_info(self.data), dl, **self.kw)
        self.assertEqual(seen, [(gm.TMP_DIR, False)])        # written under the temp folder, the target not yet there
        self.assertEqual(target.read_bytes(), self.data)
        self.assertFalse((self.dir / gm.TMP_DIR).exists())  # cleaned up

    def test_retries_with_backoff(self):
        dl = writer(self.data, fail_first=3)
        gm.fetch("o/r", "m.basalt", self.dir / "m.basalt", fake_info(self.data), dl, **self.kw)
        self.assertEqual(len(dl.calls), 4)
        self.assertEqual(self.sleeps, [2.0, 4.0, 8.0])

    def test_a_bad_hash_is_fetched_again(self):
        dl = writer(self.data, corrupt_first=1)
        gm.fetch("o/r", "m.basalt", self.dir / "m.basalt", fake_info(self.data), dl, **self.kw)
        self.assertEqual(len(dl.calls), 2)
        self.assertEqual((self.dir / "m.basalt").read_bytes(), self.data)

    def test_gives_up_after_the_bound_and_keeps_no_target(self):
        dl = writer(self.data, fail_first=99)
        with self.assertRaises(gm.DownloadError):
            gm.fetch("o/r", "m.basalt", self.dir / "m.basalt", fake_info(self.data), dl, retries=3, **self.kw)
        self.assertEqual(len(dl.calls), 3)
        self.assertEqual(self.sleeps, [2.0, 4.0])
        self.assertFalse((self.dir / "m.basalt").exists())

    def test_size_mismatch(self):
        info = fake_info(self.data)
        info["size"] += 1
        with self.assertRaises(gm.DownloadError):
            gm.fetch("o/r", "m.basalt", self.dir / "m.basalt", info, writer(self.data), retries=1, **self.kw)

    def test_git_sha1_for_small_files(self):
        sha1 = hashlib.sha1(f"blob {len(self.data)}\0".encode() + self.data).hexdigest()
        info = {"size": len(self.data), "sha256": None, "git_sha1": sha1}
        gm.fetch("o/r", "LICENSE", self.dir / "LICENSE", info, writer(self.data), **self.kw)

    def test_disk_refusal_downloads_nothing(self):
        dl = writer(self.data)
        kw = dict(self.kw, free_fn=lambda _: Usage(0, 0, 1000))
        with self.assertRaises(gm.DiskError):
            gm.fetch("o/r", "m.basalt", self.dir / "m.basalt", fake_info(self.data), dl, **kw)
        self.assertEqual(dl.calls, [])

    def test_remote_info_reads_paths_info(self):
        answer = [{"type": "file", "path": "f.basalt", "size": 5, "oid": "g", "lfs": {"oid": "ab" * 32, "size": 158746168309}}]
        info = gm.remote_info("o/r", "f.basalt", get_json=lambda url, body: answer)
        self.assertEqual(info, {"size": 158746168309, "sha256": "ab" * 32, "git_sha1": None})
        with self.assertRaises(gm.DownloadError):
            gm.remote_info("o/r", "other", get_json=lambda url, body: answer)


class Ensure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, entries) -> str:
        p = self.dir / "c.json"
        p.write_text(json.dumps({"version": 2, "models": entries}))
        return str(p)

    def test_present_files_need_nothing(self):
        (self.dir / "m.basalt").write_bytes(b"x")
        self.assertEqual(gm.ensure(gm.config_entries(self.config([{"model": "m.basalt"}]))), 0)

    def test_missing_file_detection_and_sources(self):
        items = gm.config_entries(self.config([{"model": "models/Hy4-flash-NVFP4-GPTQ.basalt"},
                                               {"model": "x.basalt", "hf": "a/b/x.basalt", "download": False}]))
        self.assertEqual(items[0], (self.dir / "models" / "Hy4-flash-NVFP4-GPTQ.basalt", None, True))
        self.assertEqual(items[1], (self.dir / "x.basalt", "a/b/x.basalt", False))

    def test_missing_with_downloads_off(self):
        items = [(self.dir / "Hy4-flash-NVFP4-GPTQ.basalt", None, True)]
        self.assertEqual(gm.ensure(items, allow=False), 1)
        items = [(self.dir / "Hy4-flash-NVFP4-GPTQ.basalt", None, False)]
        self.assertEqual(gm.ensure(items), 1)

    def test_missing_without_source(self):
        self.assertEqual(gm.ensure([(self.dir / "mine.basalt", None, True)]), 1)

    def test_missing_known_pack_is_fetched(self):
        data = b"pack bytes"
        dl = writer(data)
        target = self.dir / "models" / "Hy4-flash-NVFP4-GPTQ.basalt"
        rc = gm.ensure([(target, None, True)], info=fake_info(data), downloader=dl, free_fn=lambda _: Usage(0, 0, 10**12))
        self.assertEqual(rc, 0)
        self.assertEqual(dl.calls, [(GPTQ, "Hy4-flash-NVFP4-GPTQ.basalt")])
        self.assertEqual(target.read_bytes(), data)


@unittest.skipUnless(os.environ.get("BASALT_NET_TESTS") == "1", "set BASALT_NET_TESTS=1 for the real download")
class RealDownload(unittest.TestCase):
    def test_gptq_repo_license(self):
        with tempfile.TemporaryDirectory() as d:
            for dl in {gm.default_downloader(), gm.http_downloader}:
                target = Path(d) / dl.__name__ / "LICENSE"
                self.assertEqual(gm.ensure([(target, f"{GPTQ}/LICENSE", True)], downloader=dl), 0)
                info = gm.remote_info(GPTQ, "LICENSE")
                self.assertEqual(target.stat().st_size, info["size"])


if __name__ == "__main__":
    unittest.main()
