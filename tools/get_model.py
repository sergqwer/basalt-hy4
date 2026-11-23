"""Fetches a Basalt model pack from Hugging Face: what `basalt serve` runs when a model file is missing, and what
`basalt download` runs to fetch one ahead.

    python tools/get_model.py download [REPO/FILE | PACK-NAME] [--dir models]   # default: the Hy4-flash GPTQ pack
    python tools/get_model.py ensure --model PATH [--hf REPO/FILE]               # fetch PATH when it is missing
    python tools/get_model.py ensure --config basalt_config.json                 # every model entry of a config

A model entry names its source with "hf": "REPO/FILE"; a known pack name (KNOWN below) needs none. "download": false in
an entry, or `basalt serve --no-download`, turns the fetch off. The file is fetched into a folder beside the target
(.basalt-download), checked against the size and hash the Hugging Face API gives, and only then renamed to its name.
huggingface_hub (pip install huggingface_hub hf_xet) is used when it is installed, with HF_XET_HIGH_PERFORMANCE=1;
otherwise a plain HTTPS download with Range resume. A failed attempt is retried with backoff, resuming where it stopped.
Exit codes: 0 every file is there, 1 a failure, 2 a usage error, 3 too little free disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
# known pack names -> the repo that holds them, so a config naming one needs no "hf"
KNOWN = {
    "Hy4-flash-NVFP4-GPTQ.basalt": "Maximilian228/Hy4-flash-NVFP4-GPTQ-Basalt",
    "Hy4-flash-NVFP4.basalt": "jesdga/Hy4-flash-Basalt",
}
DEFAULT = "Maximilian228/Hy4-flash-NVFP4-GPTQ-Basalt/Hy4-flash-NVFP4-GPTQ.basalt"
TMP_DIR = ".basalt-download"
RETRIES = 8
MARGIN_MIN = 5 << 30   # free disk kept after the download: 5 GiB or 1% of the file, whichever is more
TAG = "basalt download:"


class DownloadError(Exception):
    code = 1


class DiskError(DownloadError):
    code = 3


def log(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def parse_source(src: str) -> tuple[str, str]:
    """REPO/FILE (owner/name/path/in/repo) -> (owner/name, path/in/repo); a known pack name -> its repo."""
    if src in KNOWN:
        return KNOWN[src], src
    parts = src.strip().strip("/").split("/")
    if len(parts) < 3 or not all(parts):
        raise ValueError(f"{src!r}: expected OWNER/REPO/FILE or one of {', '.join(sorted(KNOWN))}")
    return "/".join(parts[:2]), "/".join(parts[2:])


def source_for(path: str | Path, hf: str | None = None) -> tuple[str, str] | None:
    """Where a model file comes from: the entry's "hf", else the built-in map by file name; None when unknown."""
    if hf:
        return parse_source(hf)
    name = Path(path).name
    return (KNOWN[name], name) if name in KNOWN else None


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token",):
        try:
            return p.read_text().strip() or None
        except OSError:
            pass
    return None


def _request(url: str, data: bytes | None = None, headers: dict | None = None) -> urllib.request.Request:
    h = {"User-Agent": "basalt-get-model/1"}
    tok = _token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    h.update(headers or {})
    return urllib.request.Request(url, data=data, headers=h)


def http_json(url: str, data: bytes | None = None) -> object:
    hdr = {"Content-Type": "application/x-www-form-urlencoded"} if data else {}
    with urllib.request.urlopen(_request(url, data, hdr), timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def remote_info(repo: str, filename: str, revision: str = "main", get_json=http_json) -> dict:
    """{"size": bytes, "sha256": hex or None, "git_sha1": hex or None} of a repo file, from the paths-info API (the
    sha256 of an LFS/Xet file; a small file in git has only its blob sha1)."""
    url = f"{HF}/api/models/{repo}/paths-info/{urllib.parse.quote(revision, safe='')}"
    body = urllib.parse.urlencode({"paths": filename}).encode()
    try:
        items = get_json(url, body)
    except urllib.error.HTTPError as e:
        raise DownloadError(f"{repo}: the Hugging Face API answered {e.code} ({e.reason})") from None
    except (urllib.error.URLError, OSError) as e:
        raise DownloadError(f"{repo}: cannot reach the Hugging Face API: {e}") from None
    for it in items if isinstance(items, list) else []:
        if it.get("path") == filename and it.get("type") == "file":
            lfs = it.get("lfs") or {}
            return {"size": int(lfs.get("size", it.get("size", 0))), "sha256": lfs.get("oid") or lfs.get("sha256"),
                    "git_sha1": None if lfs else it.get("oid")}
    raise DownloadError(f"{repo} has no file {filename} (yet?)")


def check_disk(folder: Path, need: int, have: int = 0, free_fn=None) -> None:
    """Refuses when the folder's disk cannot take the rest of the file plus the margin."""
    usage = (free_fn or shutil.disk_usage)(str(folder))
    margin = max(MARGIN_MIN, need // 100)
    rest = max(0, need - have)
    if usage.free < rest + margin:
        raise DiskError(f"{folder}: {gb(usage.free)} free, the download needs {gb(rest)} more plus {gb(margin)} to spare; "
                        f"free some space or put the model on another disk")


def file_hash(path: Path, algo: str, prefix: bytes = b"") -> str:
    h = hashlib.new(algo)
    h.update(prefix)
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, info: dict) -> None:
    size = path.stat().st_size
    if size != info["size"]:
        raise DownloadError(f"{path.name}: {size} bytes, the repo says {info['size']}")
    if info.get("sha256"):
        log(f"checking the sha256 of {path.name} ({gb(size)})")
        got = file_hash(path, "sha256")
        if got != info["sha256"]:
            raise DownloadError(f"{path.name}: sha256 {got}, the repo says {info['sha256']}")
    elif info.get("git_sha1"):
        got = file_hash(path, "sha1", f"blob {size}\0".encode())
        if got != info["git_sha1"]:
            raise DownloadError(f"{path.name}: git blob sha1 {got}, the repo says {info['git_sha1']}")
    else:
        raise DownloadError(f"{path.name}: the repo gives no hash to check it against")


def _dir_bytes(d: Path) -> int:
    n = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n


class Progress:
    """A line every `every` seconds with the bytes in the download folder: the console and the server's log both see it."""

    def __init__(self, folder: Path, total: int, every: float = 15.0):
        self.folder, self.total, self.every = folder, total, every
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        t0, n0 = time.monotonic(), _dir_bytes(self.folder)
        while not self.stop.wait(self.every):
            n = _dir_bytes(self.folder)
            dt = time.monotonic() - t0
            rate = (n - n0) / dt if dt > 0 else 0
            eta = f", {(self.total - n) / rate / 60:.0f} min left" if rate > 0 and n < self.total else ""
            log(f"{gb(n)} of {gb(self.total)} ({100 * n / max(1, self.total):.0f}%), {rate / 1e6:.0f} MB/s{eta}")

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join(timeout=5)


def hub_downloader(repo: str, filename: str, folder: Path, revision: str = "main") -> Path:
    """huggingface_hub's download into `folder` (it resumes its own .incomplete files)."""
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(folder)))


def http_downloader(repo: str, filename: str, folder: Path, revision: str = "main") -> Path:
    """A plain HTTPS download into folder/FILE.part with Range resume, renamed to folder/FILE when complete."""
    out = folder / filename
    part = out.with_name(out.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    url = f"{HF}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(filename)}"
    req = _request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:   # already whole
            part.replace(out)
            return out
        raise
    with r:
        mode = "ab" if have and r.status == 206 else "wb"
        with open(part, mode) as f:
            while chunk := r.read(8 << 20):
                f.write(chunk)
    part.replace(out)
    return out


def default_downloader():
    try:
        import huggingface_hub  # noqa: F401
        return hub_downloader
    except ImportError:
        log("huggingface_hub is not installed (pip install huggingface_hub hf_xet for faster downloads): plain HTTPS")
        return http_downloader


def fetch(repo: str, filename: str, target: Path, info: dict | None = None, downloader=None, retries: int = RETRIES,
          sleep=time.sleep, free_fn=None, get_json=http_json, progress_every: float = 15.0) -> Path:
    """Fetches repo's `filename` to `target`: disk check, download into target's folder/.basalt-download with retries
    (each attempt resumes), size and hash check, then a rename to `target`. The temp folder is kept on failure so a
    later run resumes."""
    target = Path(target)
    folder = target.parent
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / TMP_DIR
    tmp.mkdir(exist_ok=True)
    info = info or remote_info(repo, filename, get_json=get_json)
    check_disk(folder, info["size"], _dir_bytes(tmp), free_fn)
    downloader = downloader or default_downloader()
    log(f"{repo}/{filename} ({gb(info['size'])}) -> {target}")
    got = None
    for attempt in range(1, retries + 1):
        try:
            with Progress(tmp, info["size"], progress_every):
                got = Path(downloader(repo, filename, tmp))
            verify(got, info)
            break
        except DiskError:
            raise
        except DownloadError as e:   # a whole file that does not check out: start it again
            log(f"attempt {attempt} of {retries}: {e}")
            if got is not None and got.exists():
                got.unlink()
            got = None
        except Exception as e:  # noqa: BLE001 - a network or library failure: retry, resuming
            log(f"attempt {attempt} of {retries} failed: {type(e).__name__}: {e}")
            got = None
        if attempt < retries:
            wait = min(60.0, 2.0 ** attempt)
            log(f"retrying in {wait:.0f} s")
            sleep(wait)
    if got is None:
        raise DownloadError(f"{repo}/{filename}: no complete download after {retries} attempts; run it again to resume")
    os.replace(got, target)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"{target.name} is ready ({gb(info['size'])}, checked)")
    return target


def config_entries(path: str) -> list[tuple[Path, str | None, bool]]:
    """(model path, "hf", download allowed) of each entry of a v2 config; paths relative to the file, as the server reads
    them. A malformed file is left to the server to report."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = raw.get("models") if isinstance(raw, dict) else None
    except (OSError, ValueError):
        return []
    here = Path(path).resolve().parent
    out = []
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and isinstance(e.get("model"), str) and e["model"].strip():
            hf = e.get("hf") if isinstance(e.get("hf"), str) else None
            out.append((here / os.path.expanduser(e["model"]), hf, e.get("download", True) is not False))
    return out


def ensure(items: list[tuple[Path, str | None, bool]], allow: bool = True, **kw) -> int:
    """Fetches each missing file; 0 when every one is there afterwards."""
    for path, hf, ok in items:
        if Path(path).is_file():
            continue
        src = source_for(path, hf)
        if not (allow and ok):
            log(f"{path} is missing and downloads are off" + (f"; fetch it with: basalt download {'/'.join(src)} --dir {Path(path).parent}" if src else ""))
            return 1
        if src is None:
            log(f"{path} is missing and has no known source: name one in its config entry, \"hf\": \"OWNER/REPO/FILE\"")
            return 1
        try:
            fetch(src[0], src[1], Path(path), **kw)
        except DownloadError as e:
            log(str(e))
            return e.code
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="get_model.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch a pack (default: the Hy4-flash GPTQ pack)")
    d.add_argument("source", nargs="?", default=DEFAULT, help="OWNER/REPO/FILE or a known pack name")
    d.add_argument("--dir", default="models", help="the folder to put it in (default models)")
    e = sub.add_parser("ensure", help="fetch the model files that are missing")
    e.add_argument("--model")
    e.add_argument("--hf", help="the model's source, OWNER/REPO/FILE")
    e.add_argument("--config")
    e.add_argument("--no-download", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        try:
            repo, filename = parse_source(a.source)
        except ValueError as err:
            log(str(err))
            return 2
        target = Path(a.dir) / Path(filename).name
        if target.is_file():
            log(f"{target} is already there")
            return 0
        return ensure([(target, f"{repo}/{filename}", True)])
    if bool(a.model) == bool(a.config):
        log("ensure: --model PATH or --config FILE")
        return 2
    items = [(Path(a.model), a.hf, True)] if a.model else config_entries(a.config)
    return ensure(items, allow=not a.no_download)


if __name__ == "__main__":
    sys.exit(main())
