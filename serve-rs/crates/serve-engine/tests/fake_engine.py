"""A scripted `basalt --serve` for the engine-protocol tests: answers GEN/GENT/SGEN/BGEN, PROMPT/PROMPTI/PMATCH/PFREE,
STOP/BSTOP/BYIELD, VRAM, STATS and QUIT, and appends every byte it reads on stdin to $FAKE_RECORD.

Env: FAKE_RECORD (file), FAKE_BATCH (slots, default 0), FAKE_TOKENS (tokens per answer, default 6),
FAKE_DELAY (seconds between tokens, default 0.01), FAKE_ERR (an ERR text the first GEN answers with)."""
import os
import queue
import sys
import threading
import time

rec = open(os.environ.get("FAKE_RECORD", os.devnull), "ab", buffering=0)
batch = int(os.environ.get("FAKE_BATCH", "0"))
ntok = int(os.environ.get("FAKE_TOKENS", "6"))
delay = float(os.environ.get("FAKE_DELAY", "0.01"))
err_once = os.environ.get("FAKE_ERR")
out_lock = threading.Lock()
inp = sys.stdin.buffer


def say(line: str) -> None:
    with out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def readline() -> bytes:
    line = inp.readline()
    rec.write(line)
    return line


def read(n: int) -> bytes:
    data = inp.read(n)
    rec.write(data)
    return data


cmds: queue.Queue = queue.Queue()
stops = set()       # "solo" or slot numbers


def reader():
    while True:
        line = readline()
        if not line:
            cmds.put(None)
            return
        text = line.decode("utf-8", "replace").rstrip("\n")
        words = text.split()
        if not words:
            continue
        if words[0] in ("GENT", "SGENT", "BGENT", "PROMPT"):
            start = {"PROMPT": 1, "GENT": 2}.get(words[0], 3)
            n = int(next(w for w in words[start:] if "=" not in w))
            body = read(n)
            cmds.put((words, body))
            continue
        if words[0] == "STOP":
            stops.add("solo")
            continue
        if words[0] == "BSTOP":
            stops.add(int(words[1]))
            continue
        cmds.put((words, b""))


say("INFO engine=9.9.9 eos_ids=10 text_prompt=1" + (f" batch_slots={batch} slot_cache=1" if batch else ""))
say("auto flags: pool-workers 4, resident 12 GiB")
say("READY 4096 stop")
threading.Thread(target=reader, daemon=True).start()

handles = {}
nexth = [0]


def answer_solo(prompt_n: int, max_new: int, text_n=None):
    global err_once
    if err_once:
        e, err_once = err_once, None
        say("ERR " + e)
        return
    if text_n is not None:
        say(f"PT {text_n} {max_new}")
    say(f"PP {prompt_n} {prompt_n} 12.5 800.0")
    gen = 0
    stops.discard("solo")
    for i in range(min(ntok, max_new)):
        if "solo" in stops:
            break
        say(f"T {100 + i}")
        gen += 1
        time.sleep(delay)
    finish = "cancel" if "solo" in stops else ("length" if gen >= max_new else "stop")
    stops.discard("solo")
    say(f"DONE {gen} {prompt_n} 12.5 {gen * 10.0} {finish} 3 5 0 7 9 0 1 0 {prompt_n} 2 {gen}")


def answer_slot(slot: int, max_new: int, prompt_n: int):
    stops.discard(slot)
    say(f"BADM {slot} 2")
    gen = 0

    def run():
        nonlocal gen
        for i in range(min(ntok, max_new)):
            if slot in stops:
                break
            say(f"BT {slot} {200 + i}")
            gen += 1
            time.sleep(delay)
        fin = "cancel" if slot in stops else "stop"
        stops.discard(slot)
        say(f"BDONE {slot} {gen} {fin} {gen * 10.0} 2 4 {gen}")
    threading.Thread(target=run, daemon=True).start()


while True:
    item = cmds.get()
    if item is None:
        break
    words, body = item
    c = words[0]
    if c == "QUIT":
        break
    if c == "STATS":
        continue
    if c == "VRAM":
        say("VRAM expert_slots=10 vram_free_mib=1000")
        continue
    if c in ("PROMPT", "PROMPTI"):
        nexth[0] += 1
        n = len(body.decode("utf-8")) // 2 + 1 if c == "PROMPT" else len(words[-1].split(","))
        handles[nexth[0]] = n
        say(f"PROMPT {nexth[0]} {n}")
        continue
    if c == "PMATCH":
        say("PMATCH " + words[1] + " " + " ".join("0:0" for _ in range(batch)))
        continue
    if c == "PFREE":
        continue
    if c in ("GEN", "GENI", "SGEN", "SGENI"):
        mx = int(words[2] if c.startswith("SGEN") else words[1])
        ref = words[-1]
        n = handles.get(int(ref[1:].split("+")[0]), 1) if ref.startswith("#") else len(ref.split(","))
        answer_solo(n, mx)
        continue
    if c == "GENT":
        mx = int(words[1])
        answer_solo(len(body.decode("utf-8")) // 2 + 1, mx, len(body.decode("utf-8")) // 2 + 1)
        continue
    if c in ("BGEN", "BGENI"):
        answer_slot(int(words[1]), int(words[2]), 1)
        continue
    if c == "BYIELD":
        continue
    say("ERR bad request: unknown command " + c)
