"""The reference side of the engine-protocol probe: Python's ProcessEngine over fake_engine.cmd, one JSON line per step.
Usage: probe_py.py <record file> [batch]   (run from the repo root)"""
import json
import os
import sys
import threading

sys.path.insert(0, os.getcwd())
from serve.engine import process  # noqa: E402
from serve.engine.process import ProcessEngine  # noqa: E402
from serve.prompt import PromptText  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
record, batch = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "0"
env = dict(os.environ, FAKE_RECORD=record, FAKE_BATCH=batch, FAKE_DELAY="0.05", FAKE_ERR="prompt (5 tokens) is too long: ok")
process.REQUEST_LINES = True
args = ["--model", "x.basalt"] + (["--batch", batch] if batch != "0" else [])
e = ProcessEngine(os.path.join(here, "fake_engine.cmd"), args, env=env)


def out(**kw):
    print(json.dumps(kw), flush=True)


def tokens(gen, keep=None):
    got = []
    try:
        for t in gen:
            if t is None:
                continue
            got.append([int(t), getattr(t, "lp", None)])
            if keep is not None and len(got) >= keep:
                break
    except ValueError as x:
        return got, f"{type(x).__name__}: {x}"
    finally:
        gen.close()
    return got, None


out(step="start", info={k: v for k, v in e.info.items()}, ctx=e.max_context, batch=e.batch)
toks, err = tokens(e.generate([1, 2, 3], 4, {"temperature": 0.7, "top_k": 20}, threading.Event()))
out(step="err", toks=toks, err=err)
toks, err = tokens(e.generate([1, 2, 3], 4, {"temperature": 0.7, "top_k": 20, "seed": 3}, threading.Event()))
out(step="solo", toks=toks, err=err, last=e.last)
toks, err = tokens(e.generate([5] * 10, 50, {}, threading.Event()), keep=2)
out(step="early", toks=toks, err=err, last=e.last)
pt = PromptText(ByteTokenizer(), "hello <|im_start|>world", plain=[(0, 2)])
toks, err = tokens(e.generate(pt, 30, {"logprobs": None, "_logprobs": 2}, threading.Event()))
out(step="text", toks=toks, err=err, last=e.last, n=pt.n)
res = {}


def worker(name, ids, mx):
    res[name] = tokens(e.generate(ids, mx, {}, threading.Event(), session=name))


if e.batch:
    t1 = threading.Thread(target=worker, args=("a", [7] * 40, 6))
    t1.start()
    import time
    time.sleep(0.12)
    t2 = threading.Thread(target=worker, args=("b", [8] * 3, 6))
    t2.start()
    t1.join(); t2.join()
    out(step="pair", a=len(res["a"][0]), b=len(res["b"][0]), errs=[res["a"][1], res["b"][1]], slots=e.slot_view())
out(step="vram", vram=e.vram(None))
out(step="progress", progress=e.prompt_progress(), conv=e.conversation)
e.close()
out(step="closed", alive=e.alive())
