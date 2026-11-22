"""Parity harness: run the same cases against two servers (A, B) and diff what they answer.

    python serve-rs/parity/run.py                       # A = Python server, B = serve-rs basalt-serve.exe
    python serve-rs/parity/run.py --b "python -m serve.server"     # A vs A: checks the normalization itself
    python serve-rs/parity/run.py --only server/health --verbose

Each case starts both servers (on two free ports in 18000-18999, same flags), sends its requests over raw sockets,
records every byte that comes back plus the servers' stdout/stderr up to a Ctrl+C shutdown, normalizes only the
truly random fields (NORMALIZERS below) and compares the records. Exit code 1 on any FAIL.

A case is cases/<group>/<name>.json:
    {"name", "server_args": [...after --mock-engine --port N], "mock": true (false: --engine-bin fake_engine
     --model fixtures/tiny.basalt; A and B then run one after the other, they share the model's engine.log),
     "script": ["..."] (each --script X), "env": {}, "engine_script": "file.json" (FAKE_ENGINE_SCRIPT, relative to
     cases/), "settle_s", "ready_timeout_s",
     "requests": [{"method", "path", "headers": {name: value | null (drop a default)}, "http": "HTTP/1.1",
                   "body": <json> | "raw_body": "<str>" | "raw": "<the whole request>",
                   "stream", "delay_before_s", "disconnect_after_bytes" (hang up after N body bytes), "half_close",
                   "timeout_s"},
                  {"parallel": true, "requests": [...]} (sent at once, each recorded)]}
or one request's keys at the top level. Default headers: Host 127.0.0.1:<port>; with "body" Content-Type
application/json; with a body Content-Length. Strings may hold {port}, {host} and {repeat:N:TEXT}; server_args
{repo}, {parity}, {fixtures}, {tmp}, {port}.

Shutdown is a real Ctrl+C (the server runs in its own windowless console; a helper attaches to it and sends
CTRL_C_EVENT), so the Python server prints "[server] stopping" / "[server] stopped". A server that does not end in
STOP_TIMEOUT_S is killed with its process tree, and stdout is then compared without those two lines.
"""
from __future__ import annotations

import argparse
import ctypes
import difflib
import fnmatch
import glob
import hashlib
import json
import os
import re
import select
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VENV_PY = r"D:\Projects\basalt-055\.venv\Scripts\python.exe"
DEFAULT_A = f'"{VENV_PY}" -m serve.server'
DEFAULT_B = f'"{REPO / "serve-rs" / "target" / "debug" / "basalt-serve.exe"}"'
FAKE_ENGINE = HERE / "fake_engine" / "target" / "release" / "fake_engine.exe"
FIXTURES = HERE / "fixtures"
PORTS = range(18000, 19000)
READY_TIMEOUT_S = 60.0
STOP_TIMEOUT_S = 15.0
READ_TIMEOUT_S = 20.0
SETTLE_S = 0.3                  # after ready, and after the last request, before Ctrl+C: lets late lines arrive
BIG_TEXT = 4096                 # non-JSON bodies above this are recorded as length + sha256

if os.name == "nt":
    # This process may have inherited "ignore Ctrl+C" (a parent started with a new process group); the servers would
    # inherit it too and never see the Ctrl+C that makes them print "[server] stopping".
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)

# --------------------------------------------------------------------------------------------------------------------
# Normalization rules: the ONLY differences forgiven. Each has a name (printed with --verbose when it fired).
# --------------------------------------------------------------------------------------------------------------------
ID_RE = re.compile(r"(?<![0-9A-Za-z])(chatcmpl|msg|resp|call|toolu|fc|rs|ctc|item)([-_])([0-9A-Za-z]{8,})")
TIME_KEYS = {"created", "created_at", "time"}                       # epoch seconds (time: /status request history)
TIME_SUFFIX = "_at"                                                  # last_request_at and the like: epoch seconds
NUM_KEYS = {"tokens_per_s", "tokens_per_s_mean", "elapsed_s", "uptime_s", "started", "since", "at", "waited_s",
            "thread"}
NUM_SUFFIXES = ("_ms", "_s", "_per_s")
SUBTREES_ANY = {"timings", "telemetry", "machine"}                  # every number below these keys ("machine": /v1/status)
SUBTREES_METRICS = {"hardware", "hardware_static", "history", "gpu", "gpus", "cpu", "ram", "disk"}   # /metrics only
CONSOLE_RATE = re.compile(r"(?<![0-9.])[0-9]+(?:\.[0-9]+)?t/s")
CONSOLE_TTFT = re.compile(r"ttft=[0-9]+(?:\.[0-9]+)?s")
CONSOLE_CANCELLED = re.compile(r"(?<![0-9A-Za-z])tg=[0-9]+(?=.*end=cancelled)")       # tokens made before a hang-up was seen
CONSOLE_PP = re.compile(r"(pp=[0-9]+/[0-9]+c) [0-9]+(?:\.[0-9]+)?s")
LOG_TIME_KEY = "t"                                                  # /api/logs lines: {"id","t","src","lvl","msg"}

NORMALIZERS = [
    ("header-date", "Date header value -> <date>"),
    ("header-server", "Server header value -> <server>"),
    ("header-request-id", "X-Request-Id value -> mapped id"),
    ("content-length", "Content-Length value -> <len> when a body rule fired (body compared normalized; "
                       "presence compared; each side's value checked against its own body)"),
    ("id", "ids (chatcmpl|msg|resp|call|toolu|fc|rs|ctc|item)[-_]<8+ alnum> -> prefix<sep><idN>, consistent per side"),
    ("time-key", "numbers under keys created/created_at/time and any key ending _at -> <time>"),
    ("num-key", "numbers under keys ending _ms/_s/_per_s and " + "/".join(sorted(NUM_KEYS)) + " -> <num>"),
    ("subtree", "numbers anywhere under timings/telemetry/machine (any body) and hardware/hardware_static/history/gpu(s)/"
                "cpu/ram/disk (/metrics) -> <num>; under /metrics history, lists -> <list>"),
    ("log-time", "/api/logs line field t -> <time>; its msg gets the console-timing rule"),
    ("console-timing", "server output only: N.Nt/s -> <num>t/s, ttft=N.Ns -> ttft=<num>s, pp=A/Bc N.Ns -> pp=A/Bc <num>s, "
                       "tg=N -> tg=<num> on a line that ends end=cancelled (how far a cancelled run got is timing)"),
    ("close-after-complete", "connection end eof/reset -> closed-after-complete once a response with a matching "
                             "Content-Length has fully arrived (FIN vs RST over unread request bytes is a race)"),
    ("port", "the side's own port number (the harness put it there) -> <port>, in stdout/stderr and bodies"),
]


class Norm:
    """Per side and case: the id mapping and which rules fired."""

    def __init__(self, port: int):
        self.port = str(port)
        self.ids: dict[str, str] = {}
        self.fired: dict[str, int] = {}

    def hit(self, rule: str) -> None:
        self.fired[rule] = self.fired.get(rule, 0) + 1

    def text(self, s: str) -> str:
        def sub(m):
            self.hit("id")
            key = m.group(0)
            if key not in self.ids:
                self.ids[key] = f"<id{len(self.ids) + 1}>"
            return m.group(1) + m.group(2) + self.ids[key]
        s = ID_RE.sub(sub, s)
        s2 = re.sub(rf"(?<![0-9]){self.port}(?![0-9])", "<port>", s)
        if s2 != s:
            self.hit("port")
        return s2

    def line(self, s: str) -> str:
        """One line of server output: the text rules plus the console's clock figures."""
        s = self.text(s)
        s2 = CONSOLE_RATE.sub("<num>t/s", s)
        s2 = CONSOLE_TTFT.sub("ttft=<num>s", s2)
        s2 = CONSOLE_PP.sub(lambda m: m.group(1) + " <num>s", s2)
        s2 = CONSOLE_CANCELLED.sub("tg=<num>", s2)
        if s2 != s:
            self.hit("console-timing")
        return s2

    def json(self, v, key=None, metrics=False, under=False, in_history=False, in_logs=False):
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                sub = under or k in SUBTREES_ANY or (metrics and k in SUBTREES_METRICS)
                hist = in_history or (metrics and k == "history")
                logs = in_logs or (k == "lines" and isinstance(x, list))
                out[self.text(k)] = self.json(x, k, metrics, sub, hist, logs)
            return out
        if isinstance(v, list):
            if in_history:
                self.hit("subtree")
                return "<list>"
            return [self.json(x, key, metrics, under, in_history, in_logs) for x in v]
        if isinstance(v, str):
            if in_logs and key == "msg":
                return self.line(v)                  # a console line, served back by /api/logs
            return self.text(v)
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            if key in TIME_KEYS or (isinstance(key, str) and key.endswith(TIME_SUFFIX)):
                self.hit("time-key")
                return "<time>"
            if in_logs and key == LOG_TIME_KEY:
                self.hit("log-time")
                return "<time>"
            if isinstance(key, str) and (key in NUM_KEYS or key.endswith(NUM_SUFFIXES)):
                self.hit("num-key")
                return "<num>"
            if under:
                self.hit("subtree")
                return "<num>"
        return v


def py_json_format(raw: str, obj) -> str:
    """How the JSON text was written: "py" = json.dumps(obj, ensure_ascii=False) byte for byte (the Python server's
    form), else "other" (separators, escapes, float spelling or spacing differ)."""
    return "py" if raw == json.dumps(obj, ensure_ascii=False) else "other"


def norm_json_text(raw: str, norm: Norm, metrics: bool):
    """(normalized value, format) of a JSON text; None when it is not JSON."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    return norm.json(obj, metrics=metrics), py_json_format(raw, obj)


# --------------------------------------------------------------------------------------------------------------------
# Raw HTTP client
# --------------------------------------------------------------------------------------------------------------------
REPEAT_RE = re.compile(r"\{repeat:(\d+):([^}]*)\}")


def expand(s: str, port: int) -> str:
    """{port}, {host} (127.0.0.1:port) and {repeat:N:TEXT} in a case's raw strings; \\r\\n written as JSON escapes."""
    s = REPEAT_RE.sub(lambda m: m.group(2) * int(m.group(1)), s)
    return s.replace("{port}", str(port)).replace("{host}", f"127.0.0.1:{port}")


def build_request(req: dict, port: int) -> bytes:
    if "raw" in req:
        return expand(req["raw"], port).encode("utf-8")
    method = req.get("method", "GET")
    path = expand(req.get("path", "/"), port)
    version = req.get("http", "HTTP/1.1")
    body = None
    if "raw_body" in req:
        body = expand(req["raw_body"], port).encode("utf-8")
    elif "body" in req:
        body = json.dumps(req["body"]).encode("utf-8")
    headers: list[tuple[str, str]] = []
    given = {k.lower(): (k, v) for k, v in (req.get("headers") or {}).items()}
    defaults = [("Host", f"127.0.0.1:{port}")]
    if body is not None:
        if "body" in req:
            defaults.append(("Content-Type", "application/json"))
        defaults.append(("Content-Length", str(len(body))))
    for k, v in defaults:
        if k.lower() not in given:
            headers.append((k, v))
    for k, v in given.values():
        if v is not None:
            headers.append((k, expand(str(v), port)))
    head = f"{method} {path} {version}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers) + "\r\n"
    return head.encode("latin-1") + (body or b"")


def send_one(req: dict, port: int) -> dict:
    """Send one request and read until the server closes (or the case's byte limit, or the timeout). The request is
    sent from a thread while this one reads: a server that answers before reading the whole body and then closes
    resets the connection, and a reset discards what the client had not read yet."""
    data = build_request(req, port)
    limit = req.get("disconnect_after_bytes")
    timeout = float(req.get("timeout_s", READ_TIMEOUT_S))
    got = bytearray()
    end = "eof"
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.settimeout(None)

    def sender():
        try:
            view = memoryview(data)
            while view:
                n = s.send(view[:65536])
                view = view[n:]
            if req.get("half_close"):
                s.shutdown(socket.SHUT_WR)
        except OSError:                  # the server closed first: what it answered is the record
            pass
    t = threading.Thread(target=sender, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + timeout
        while True:
            head_end = got.find(b"\r\n\r\n")
            if limit is not None and head_end >= 0 and len(got) - head_end - 4 >= limit:
                del got[head_end + 4 + limit:]
                end = "client-disconnect"
                break
            left = deadline - time.monotonic()
            if left <= 0:
                end = "timeout"
                break
            r, _, _ = select.select([s], [], [], min(left, 0.5))
            if not r:
                continue
            try:
                chunk = s.recv(65536)
            except (ConnectionResetError, ConnectionAbortedError):
                end = "reset"
                break
            except OSError as e:
                end = f"error:{type(e).__name__}"
                break
            if not chunk:
                end = "eof"
                break
            got += chunk
    finally:
        try:
            if limit is not None:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        except OSError:
            pass
        s.close()
        t.join(5)
    return {"bytes": bytes(got), "end": end}


def parse_response(raw: bytes, end: str, norm: Norm) -> dict:
    """The record of one response: status line, headers in order, the body (JSON / SSE events / text / digest)."""
    rec: dict = {"end": end}
    head, sep, body = raw.partition(b"\r\n\r\n")
    if not sep:
        rec["incomplete_head"] = norm.text(raw.decode("latin-1"))
        return rec
    lines = head.decode("latin-1").split("\r\n")
    rec["status_line"] = lines[0]
    headers = []
    ctype = ""
    clen = None
    for line in lines[1:]:
        name, _, value = line.partition(":")
        value = value.strip()
        low = name.lower()
        if low == "date":
            norm.hit("header-date")
            value = "<date>"
        elif low == "server":
            norm.hit("header-server")
            value = "<server>"
        elif low == "x-request-id":
            norm.hit("header-request-id")
            value = norm.text(value) if ID_RE.fullmatch(value) else "<request-id>"
        elif low == "content-length":
            clen = value
        else:
            value = norm.text(value)
        if low == "content-type":
            ctype = value
        headers.append([name, value])
    before = dict(norm.fired)
    rec["body"] = body_record(body, ctype, norm, metrics=False, cut=end == "client-disconnect")
    if clen is not None:
        rec["content_length_matches_body"] = clen.isdigit() and int(clen) == len(body)
        body_rules = {k for k, v in norm.fired.items() if v != before.get(k, 0)}
        if body_rules:
            norm.hit("content-length")
            for h in headers:
                if h[0].lower() == "content-length":
                    h[1] = "<len>"
        if rec["content_length_matches_body"] and end in ("eof", "reset"):
            # the whole response arrived: whether the close was a FIN or a RST (unread request bytes) is a race
            if end == "reset":
                norm.hit("close-after-complete")
            rec["end"] = "closed-after-complete"
    rec["headers"] = headers
    return rec


def body_record(body: bytes, ctype: str, norm: Norm, metrics: bool, cut: bool = False):
    if not body:
        return ""
    if ctype.startswith("text/event-stream"):
        return {"sse": sse_events(body, norm, cut)}
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return {"len": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    if ctype.startswith("application/json"):
        got = norm_json_text(text, norm, metrics)
        if got is not None:
            return {"json": got[0], "json_format": got[1]}
    if len(body) > BIG_TEXT:
        return {"len": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    return {"text": norm.text(text)}


def sse_events(body: bytes, norm: Norm, cut: bool = False) -> list:
    text = body.decode("utf-8", "replace")
    events = []
    parts = text.split("\n\n")
    tail = parts.pop()                          # after the last blank line: "" for a complete stream
    for part in parts:
        ev = []
        for line in part.split("\n"):
            if line.startswith("data: "):
                got = norm_json_text(line[6:], norm, False)
                ev.append({"data": got[0], "json_format": got[1]} if got is not None else norm.text(line))
            else:
                ev.append(norm.text(line))
        events.append(ev)
    if tail and cut:
        events.append("<cut by the client>")            # the partial event depends on where the bytes were cut
    elif tail:
        events.append({"unterminated": norm.text(tail)})
    return events


# --------------------------------------------------------------------------------------------------------------------
# Servers
# --------------------------------------------------------------------------------------------------------------------
_port_lock = threading.Lock()
_ports_taken: set[int] = set()


def free_port() -> int:
    with _port_lock:
        for p in PORTS:
            if p in _ports_taken:
                continue
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    continue
            _ports_taken.add(p)
            return p
    raise RuntimeError("no free port in 18000-18999")


def release_port(p: int) -> None:
    with _port_lock:
        _ports_taken.discard(p)


def split_prefix(cmd: str) -> list[str]:
    return [w[1:-1] if len(w) >= 2 and w[0] == w[-1] == '"' else w for w in shlex.split(cmd, posix=False)]


CTRL_C_HELPER = r"""
import ctypes, sys
k = ctypes.windll.kernel32
k.FreeConsole()
if not k.AttachConsole(int(sys.argv[1])):
    sys.exit(3)
k.SetConsoleCtrlHandler(None, True)
sys.exit(0 if k.GenerateConsoleCtrlEvent(0, 0) else 4)
"""


def send_ctrl_c(pid: int) -> bool:
    """A real Ctrl+C to the server's own (windowless) console: Python's KeyboardInterrupt path, Rust's ctrl handler."""
    r = subprocess.run([sys.executable, "-c", CTRL_C_HELPER, str(pid)], creationflags=subprocess.CREATE_NO_WINDOW,
                       timeout=15)
    return r.returncode == 0


class Proc:
    def __init__(self, argv: list[str], env: dict, cwd: str):
        self.argv = argv
        self.p = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        self.out: list[bytes] = []
        self.err: list[bytes] = []
        self.ready = threading.Event()
        self.t_out = threading.Thread(target=self._pump, args=(self.p.stdout, self.out, True), daemon=True)
        self.t_err = threading.Thread(target=self._pump, args=(self.p.stderr, self.err, False), daemon=True)
        self.t_out.start()
        self.t_err.start()

    def _pump(self, f, sink, watch):
        for line in iter(f.readline, b""):
            sink.append(line)
            if watch and b"[server] ready" in line:
                self.ready.set()
        f.close()

    def wait_ready(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.ready.wait(0.1):
                return True
            if self.p.poll() is not None:
                return False
        return False

    def tree(self) -> list:
        try:
            import psutil
            me = psutil.Process(self.p.pid)
            return [me] + me.children(recursive=True)
        except Exception:  # noqa: BLE001
            return []

    def stop(self) -> dict:
        """Ctrl+C, then wait; kill the whole tree (only our own PIDs) if it does not end. Returns how it ended."""
        procs = self.tree()
        how = "exited-before-stop" if self.p.poll() is not None else "ctrl-c"
        if how == "ctrl-c":
            if not send_ctrl_c(self.p.pid):
                how = "ctrl-c-failed"
            try:
                self.p.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                how = "killed"
        leftovers = []
        try:
            import psutil
            _, alive = psutil.wait_procs(procs, timeout=5)     # a venv launcher's child ends a moment after it
        except Exception:  # noqa: BLE001
            alive = procs
        for q in alive:
            try:
                if q.is_running() and q.status() != "zombie":
                    leftovers.append(q.pid)
                    q.kill()
            except Exception:  # noqa: BLE001
                pass
        if self.p.poll() is None:
            self.p.kill()
            how = "killed"
        self.p.wait(10)
        self.t_out.join(5)
        self.t_err.join(5)
        return {"how": how, "rc": self.p.returncode, "leftover_pids_killed": leftovers}


def side_command(prefix: str, port: int, case: dict, tmp: str, side_env: dict) -> list[str]:
    argv = split_prefix(prefix)
    if case.get("mock", True):
        argv.append("--mock-engine")
    else:
        argv += ["--engine-bin", str(FAKE_ENGINE), "--model", str(FIXTURES / case.get("model", "tiny.basalt"))]
    argv += ["--port", str(port)]
    for w in case.get("server_args", []):
        argv.append(subst_path(w, port, tmp))
    for s in case.get("script", []):
        argv += ["--script", s]
    return argv


def subst_path(w: str, port: int, tmp: str) -> str:
    return (w.replace("{repo}", str(REPO)).replace("{parity}", str(HERE)).replace("{fixtures}", str(FIXTURES))
            .replace("{tmp}", tmp).replace("{port}", str(port)))


def request_groups(case: dict) -> list:
    if "requests" in case:
        return case["requests"]
    if "method" in case or "path" in case or "raw" in case:
        return [{k: case[k] for k in ("method", "path", "headers", "body", "raw_body", "raw", "stream", "http",
                                       "delay_before_s", "disconnect_after_bytes", "half_close", "timeout_s")
                 if k in case}]
    return []


def run_side(prefix: str, case: dict, env_extra: dict) -> dict:
    """Start one server, replay the case, stop it: the raw record (responses as bytes, output lines) and its port."""
    port = free_port()
    tmp = tempfile.mkdtemp(prefix="parity-")
    env = dict(os.environ)
    env["BASALT_ROOT"] = str(REPO)
    env.pop("NO_COLOR", None)
    for k, v in (case.get("env") or {}).items():
        env[k] = subst_path(v, port, tmp)
    engine_log = os.path.join(tmp, "engine-input.bin")
    env["FAKE_ENGINE_INPUT"] = engine_log
    args_log = os.path.join(tmp, "engine-args.txt")
    env["FAKE_ENGINE_ARGS"] = args_log
    if "engine_script" in case:
        env["FAKE_ENGINE_SCRIPT"] = str(HERE / "cases" / case["engine_script"]) if not os.path.isabs(
            case["engine_script"]) else case["engine_script"]
    env.update(env_extra)
    argv = side_command(prefix, port, case, tmp, env)
    try:
        proc = Proc(argv, env, str(REPO))
        ready = proc.wait_ready(float(case.get("ready_timeout_s", READY_TIMEOUT_S)))
        responses = []
        if ready:
            time.sleep(SETTLE_S)
            for item in request_groups(case):
                if item.get("parallel"):
                    group = item["requests"]
                    out: list = [None] * len(group)

                    def go(i, r):
                        if r.get("delay_before_s"):
                            time.sleep(r["delay_before_s"])
                        out[i] = send_one(r, port)
                    ts = [threading.Thread(target=go, args=(i, r)) for i, r in enumerate(group)]
                    for t in ts:
                        t.start()
                    for t in ts:
                        t.join()
                    responses.append({"parallel": out})
                else:
                    if item.get("delay_before_s"):
                        time.sleep(item["delay_before_s"])
                    responses.append(send_one(item, port))
            time.sleep(float(case.get("settle_s", SETTLE_S)))
        stop = proc.stop()
        engine_input = None
        if os.path.exists(engine_log):
            with open(engine_log, "rb") as f:
                engine_input = f.read()
        engine_args = None
        if os.path.exists(args_log):
            with open(args_log, "rb") as f:
                engine_args = f.read()
        return {"port": port, "ready": ready, "responses": responses, "stdout": b"".join(proc.out),
                "stderr": b"".join(proc.err), "stop": stop, "engine_input": engine_input,
                "engine_args": engine_args, "argv": argv}
    finally:
        release_port(port)


def out_lines(raw: bytes, norm: Norm) -> list[str]:
    """Output split into lines (CR LF and LF alike: the line ending is not compared), each normalized."""
    lines = raw.decode("utf-8", "replace").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [norm.line(x.rstrip("\r")) for x in lines]


def normalize_side(raw: dict, case: dict) -> tuple[dict, Norm]:
    norm = Norm(raw["port"])
    rec: dict = {"ready": raw["ready"]}
    reqs = request_groups(case)
    responses = []
    for item, r in zip(reqs, raw["responses"]):
        if "parallel" in r:
            responses.append({"parallel": [resp_record(sub_req, x, norm) for sub_req, x in
                                           zip(item["requests"], r["parallel"])]})
        else:
            responses.append(resp_record(item, r, norm))
    rec["responses"] = responses
    rec["stdout"] = out_lines(raw["stdout"], norm)
    rec["stderr"] = out_lines(raw["stderr"], norm)
    stop = raw["stop"]
    rec["exit"] = {"how": stop["how"], "rc": stop["rc"]}
    if raw["engine_input"] is not None:
        rec["engine_input"] = out_lines(raw["engine_input"], norm)
    if raw["engine_args"] is not None:
        rec["engine_args"] = out_lines(raw["engine_args"], norm)
    # the stdout comparison is cut at the last request when a side could not be stopped cleanly
    if stop["how"] in ("killed", "ctrl-c-failed"):
        rec["stdout_cut_at_stop"] = True
    return rec, norm


def resp_record(req: dict, r: dict, norm: Norm) -> dict:
    path = req.get("path", "")
    metrics = path.split("?")[0].rstrip("/") == "/metrics"
    rec = parse_response(r["bytes"], r["end"], norm)
    if metrics and isinstance(rec.get("body"), dict) and "json" in rec["body"]:
        # re-normalize the body with the /metrics subtrees
        head, _, body = r["bytes"].partition(b"\r\n\r\n")
        got = norm_json_text(body.decode("utf-8"), norm, True)
        rec["body"] = {"json": got[0], "json_format": got[1]}
    return rec


def compare(a: dict, b: dict) -> tuple[bool, list[str]]:
    if a.get("stdout_cut_at_stop") or b.get("stdout_cut_at_stop"):
        for rec in (a, b):          # compare only what came before the stop
            rec["stdout"] = [x for x in rec["stdout"] if "[server] stopping" not in x and "[server] stopped" not in x]
    ja = json.dumps(a, indent=1, ensure_ascii=False, sort_keys=False).splitlines()
    jb = json.dumps(b, indent=1, ensure_ascii=False, sort_keys=False).splitlines()
    if ja == jb:
        return True, []
    return False, list(difflib.unified_diff(ja, jb, "A", "B", lineterm="", n=3))


def need_engine() -> None:
    """Build the fake engine and the tiny model file when missing."""
    if not FAKE_ENGINE.exists():
        subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=HERE / "fake_engine", check=True)
    if not (FIXTURES / "tiny.basalt").exists():
        subprocess.run([sys.executable, str(FIXTURES / "make_tiny_basalt.py")], check=True)


def load_cases(pattern: str) -> list[tuple[str, dict]]:
    root = HERE / "cases"
    out = []
    for f in sorted(glob.glob(str(root / pattern), recursive=True)):
        p = Path(f)
        if p.suffix != ".json":
            continue
        name = p.relative_to(root).with_suffix("").as_posix()
        with open(p, encoding="utf-8") as fh:
            case = json.load(fh)
        out.append((name, case))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default=DEFAULT_A, help="server A command prefix (cwd = repo root)")
    ap.add_argument("--b", default=DEFAULT_B, help="server B command prefix (env BASALT_ROOT = repo root)")
    ap.add_argument("--cases", default="**/*.json", help="glob under parity/cases (default **/*.json)")
    ap.add_argument("--only", action="append", default=[], help="case name (group/name) or fnmatch pattern")
    ap.add_argument("--keep-going", action="store_true", help="run every case even after a FAIL")
    ap.add_argument("--verbose", action="store_true", help="unified diff of the normalized records, rules fired")
    ap.add_argument("--dump", metavar="DIR", help="write each side's normalized record as JSON here")
    ap.add_argument("--serial", action="store_true", help="run A and B one after the other, not at once")
    ap.add_argument("--rules", action="store_true", help="print the normalization rules and exit")
    args = ap.parse_args()
    if args.rules:
        for name, what in NORMALIZERS:
            print(f"{name:22} {what}")
        return 0

    cases = load_cases(args.cases)
    if args.only:
        cases = [(n, c) for n, c in cases if any(n == o or fnmatch.fnmatch(n, o) or c.get("name") == o
                                                 for o in args.only)]
    if not cases:
        print("no cases")
        return 1
    fails, passed = [], 0
    for name, case in cases:
        t0 = time.monotonic()
        raws: dict = {}

        def side(key, prefix):
            try:
                raws[key] = run_side(prefix, case, {})
            except Exception as e:  # noqa: BLE001
                raws[key] = e
        if not case.get("mock", True):
            need_engine()
        if args.serial or not case.get("mock", True):     # the engine log in ~/.cache/basalt is per model file
            side("a", args.a)
            side("b", args.b)
        else:
            ts = [threading.Thread(target=side, args=("a", args.a)), threading.Thread(target=side, args=("b", args.b))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        errs = [f"{k}: {type(v).__name__}: {v}" for k, v in raws.items() if isinstance(v, Exception)]
        if errs:
            print(f"FAIL {name}  (harness error: {'; '.join(errs)})")
            fails.append(name)
            if not args.keep_going:
                break
            continue
        ra, na = normalize_side(raws["a"], case)
        rb, nb = normalize_side(raws["b"], case)
        ok, diff = compare(ra, rb)
        for key, raw in (("a", raws["a"]), ("b", raws["b"])):
            if raw["stop"]["leftover_pids_killed"]:
                print(f"  note: {key}: killed leftover processes {raw['stop']['leftover_pids_killed']}")
        if args.dump:
            d = Path(args.dump) / name
            d.parent.mkdir(parents=True, exist_ok=True)
            for key, rec in (("a", ra), ("b", rb)):
                with open(f"{d}.{key}.json", "w", encoding="utf-8", newline="\n") as fh:
                    json.dump(rec, fh, indent=1, ensure_ascii=False)
        dt = time.monotonic() - t0
        print(f"{'PASS' if ok else 'FAIL'} {name}  ({dt:.1f}s)")
        if args.verbose:
            fired = sorted(set(na.fired) | set(nb.fired))
            if fired:
                print("  rules: " + ", ".join(f"{r} (A {na.fired.get(r, 0)}, B {nb.fired.get(r, 0)})" for r in fired))
            for line in diff:
                print("  " + line)
        if ok:
            passed += 1
        else:
            fails.append(name)
            if not args.keep_going:
                break
    print(f"\n{passed} passed, {len(fails)} failed, {len(cases) - passed - len(fails)} not run")
    for n in fails:
        print(f"  FAIL {n}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
