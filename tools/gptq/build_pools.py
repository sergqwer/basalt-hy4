"""build_pools.py - calibration + held-out pools for every bucket except A2 (captured separately) and F (answers by Hy4).

Writes corpus/pools/<bucket>.jsonl rows {id, bucket, lang, source, split, est_tokens, api, body}; body is an OpenAI
/v1/chat/completions request (messages [, tools]) that the dump runner posts to the server, so the server renders it.
split: 'eval' when hash(source id) % 10 == 0 (decided per source document / session before any rendering), else 'train'.
Quotas are token targets (train); eval gets ~10% more from the eval sources.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

G = r"D:\Projects\hy4-data\gptq"
SRC = G + r"\src"
OUT = G + r"\corpus\pools"
os.makedirs(OUT, exist_ok=True)
TOK = Tokenizer.from_file(r"C:\LLM_models\hy4-bf16\tokenizer.json")
rng = random.Random(1234)

QUOTA = {"A3": 420_000, "B": 180_000, "C1": 180_000, "C2": 120_000, "D": 165_000, "E": 105_000, "G": 60_000, "H": 45_000}


def ntok(s: str) -> int:
    return len(TOK.encode(s, add_special_tokens=False).ids)


def split_of(src: str) -> str:
    return "eval" if int(hashlib.sha1(src.encode()).hexdigest(), 16) % 10 == 0 else "train"


def target_len() -> int:
    """Most samples 2-8K, 20% 16-32K, 5% 64-128K."""
    u = rng.random()
    if u < 0.05:
        return rng.randint(64_000, 120_000)
    if u < 0.25:
        return rng.randint(16_000, 32_000)
    return rng.randint(2_000, 8_000)


# ------------------------------------------------------------------ sanitizing (user repos)
SECRET = re.compile(r"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|gh[ousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|"
                    r"Bearer\s+[A-Za-z0-9._~+/-]{16,}|api[_-]?key\s*[=:]\s*['\"]?[A-Za-z0-9._-]{12,}|"
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|password\s*[=:]\s*['\"][^'\"]{4,})", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
san = {"files_dropped_secret": 0, "emails_replaced": 0}


def clean(text: str) -> str | None:
    if SECRET.search(text):
        san["files_dropped_secret"] += 1
        return None
    n = len(EMAIL.findall(text))
    if n:
        san["emails_replaced"] += n
        text = EMAIL.sub("user@example.com", text)
    return text


class Pool:
    def __init__(self, bucket):
        self.bucket, self.rows, self.tok = bucket, [], {"train": 0, "eval": 0}
        self.n = 0

    def target(self, split):
        """eval: short only (2-8K, for per-answer NLL); train: 75% 2-8K, 20% 16-32K, 5% 64-72K (one long per bucket,
        only where long documents are natural)."""
        if split == "eval":
            return rng.randint(2_000, 8_000)
        t = target_len()
        if t >= 64_000:
            if self.bucket in ("C1", "D", "E") and not getattr(self, "longs", 0):
                self.longs = 1
                return min(t, 72_000)
            return rng.randint(16_000, 32_000)
        return t

    def full(self, split):
        q = QUOTA[self.bucket]
        return self.tok[split] >= (q if split == "train" else q // 10 + 1)

    def done(self):
        return self.full("train") and self.full("eval")

    def add(self, lang, source, split, messages, tools=None, est=None):
        if self.full(split):
            return False
        body = {"messages": messages}
        if tools:
            body["tools"] = tools
        est = est or sum(ntok(m["content"] if isinstance(m.get("content"), str) else json.dumps(m, ensure_ascii=False))
                         for m in messages) + (ntok(json.dumps(tools, ensure_ascii=False)) if tools else 0)
        self.n += 1
        self.rows.append({"id": "%s-%05d" % (self.bucket, self.n), "bucket": self.bucket, "lang": lang, "source": source,
                          "split": split, "est_tokens": est, "api": "openai", "body": body})
        self.tok[split] += est
        return True

    def write(self):
        with open(os.path.join(OUT, self.bucket + ".jsonl"), "w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("%-3s rows %5d train %7d eval %6d" % (self.bucket, len(self.rows), self.tok["train"], self.tok["eval"]), flush=True)


# ------------------------------------------------------------------ text buckets: documents packed to a target length
PROMPTS = {
    "uk": ["Прочитай текст нижче й стисло перекажи головне.", "Виділи основні тези цього тексту у вигляді списку.",
           "Ось кілька текстів. Для кожного напиши одне речення про його зміст.", "Знайди в тексті факти, які варто перевірити, і поясни чому.",
           "Переклади ключові думки цього тексту англійською.", "Склади 5 запитань для перевірки розуміння цього тексту."],
    "en": ["Read the text below and summarize the key points.", "List the main claims of these documents.",
           "Here are several articles. Write a one-sentence summary of each.", "Which statements in this text would you fact-check, and why?",
           "Explain the main idea of this text to a high-school student.", "Write five comprehension questions about this text."],
    "zh": ["请阅读下面的文本并概括要点。", "请列出这些文章的主要观点。", "下面有几篇文章，请为每篇写一句话的摘要。"],
    "ru": ["Прочитай текст ниже и кратко перескажи главное.", "Выдели основные тезисы этих текстов списком.",
           "Вот несколько статей. Для каждой напиши одно предложение о содержании."],
}


def docs_parquet(path, text_col="text", id_col="id", min_chars=400):
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=2000, columns=[text_col, id_col] if id_col else [text_col]):
        d = b.to_pydict()
        for i, t in enumerate(d[text_col]):
            if t and len(t) >= min_chars:
                yield (str(d[id_col][i]) if id_col else str(hash(t))), t


def pack_text(pool, lang, gens, sep="\n\n---\n\n"):
    """Documents in source order, each sample a user turn of a prompt + docs up to a drawn target length (a document
    that does not fit is cut at a paragraph)."""
    bufs = {"train": [], "eval": []}
    goal = {"train": pool.target("train"), "eval": pool.target("eval")}
    have = {"train": 0, "eval": 0}
    for src, docs in gens:
        for did, text in docs:
            sid = src + ":" + did
            sp = split_of(sid)
            if pool.full(sp):
                if pool.done():
                    return
                continue
            t = ntok(text)
            room = goal[sp] - have[sp]
            if t > room:
                if room < 300:
                    t = 0
                else:
                    keep = text[: int(len(text) * room / t)]
                    cut = keep.rfind("\n")
                    text = keep[:cut] if cut > len(keep) // 2 else keep
                    t = ntok(text)
            if t:
                bufs[sp].append((sid, text)); have[sp] += t
            if have[sp] >= goal[sp] - 300:
                msg = rng.choice(PROMPTS[lang]) + "\n\n" + sep.join(x[1] for x in bufs[sp])
                pool.add(lang, ";".join(x[0] for x in bufs[sp][:3]) + (";..." if len(bufs[sp]) > 3 else ""), sp,
                         [{"role": "user", "content": msg}], est=have[sp] + 30)
                bufs[sp], have[sp], goal[sp] = [], 0, pool.target(sp)


def build_text():
    p = Pool("D")
    pack_text(p, "uk", [("fineweb2-ukr", docs_parquet(SRC + r"\HuggingFaceFW__fineweb-2\data\ukr_Cyrl\test\000_00000.parquet")),
                        ("wiki-uk", docs_parquet(glob.glob(SRC + r"\wikimedia__wikipedia\20231101.uk\*.parquet")[0]))]
              if False else
              [("mix-uk", interleave(docs_parquet(SRC + r"\HuggingFaceFW__fineweb-2\data\ukr_Cyrl\test\000_00000.parquet"),
                                     docs_parquet(glob.glob(SRC + r"\wikimedia__wikipedia\20231101.uk\*.parquet")[0])))])
    p.write()
    p = Pool("E")
    pack_text(p, "en", [("fineweb-edu", docs_parquet(SRC + r"\HuggingFaceFW__fineweb-edu\sample\10BT\000_00000.parquet"))])
    p.write()
    p = Pool("G")
    pack_text(p, "zh", [("fineweb2-cmn", docs_parquet(SRC + r"\HuggingFaceFW__fineweb-2\data\cmn_Hani\test\000_00000.parquet", min_chars=200))])
    p.write()
    p = Pool("H")
    pack_text(p, "ru", [("fineweb2-rus", docs_parquet(SRC + r"\HuggingFaceFW__fineweb-2\data\rus_Cyrl\test\000_00000.parquet"))])
    p.write()


def interleave(a, b):
    """fineweb-2 and wikipedia documents alternately (ids prefixed by source)."""
    ia, ib = iter(a), iter(b)
    while True:
        got = False
        for tag, it in (("fw", ia), ("wk", ib)):
            x = next(it, None)
            if x is not None:
                got = True
                yield tag + "-" + x[0], x[1]
        if not got:
            return


# ------------------------------------------------------------------ code buckets
CODE_PROMPTS = {
    "en": ["Review this code: point out bugs, risky constructs and unclear parts, with line references.",
           "Explain what this code does, file by file, and how the pieces fit together.",
           "Suggest a refactoring plan for this code and show the most important change.",
           "Find security problems in this code, if any, and explain how to fix them.",
           "Write unit tests for the main functions in this code."],
    "uk": ["Переглянь цей код: знайди помилки, ризиковані місця й незрозумілі фрагменти, з номерами рядків.",
           "Поясни, що робить цей код, файл за файлом.", "Запропонуй план рефакторингу цього коду й покажи найважливішу зміну.",
           "Чи є в цьому коді проблеми безпеки? Поясни, як їх виправити."],
}
EXT_LANG = {".rs": "rust", ".cpp": "cpp", ".hpp": "cpp", ".h": "c", ".c": "c", ".cu": "cuda", ".cuh": "cuda", ".py": "python",
            ".ps1": "powershell", ".sh": "bash", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx", ".html": "html",
            ".css": "css", ".svelte": "svelte", ".toml": "toml", ".cmake": "cmake", ".mjs": "javascript"}


def stack_files(lang_dir, cap=4000):
    """the-stack-smol rows (JSON lines) of a language: (id, path, content)."""
    p = SRC + r"\bigcode__the-stack-smol\data\%s\data.json" % lang_dir
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= cap:
                return
            r = json.loads(line)
            c = r.get("content") or ""
            if 200 < len(c) < 60_000:
                yield "stack-%s-%d" % (lang_dir, i), r.get("path") or ("file%d" % i), c


def repo_files(repo, exts, prefix=""):
    out = subprocess.run(["git", "-C", repo, "ls-files", prefix], capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
    for rel in out:
        ext = os.path.splitext(rel)[1].lower()
        if ext not in exts:
            continue
        p = os.path.join(repo, rel)
        try:
            if os.path.getsize(p) > 120_000:
                continue
            c = open(p, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        c = clean(c)
        if c and len(c) > 200:
            yield "repo-%s:%s" % (os.path.basename(repo), rel), rel, c


def pack_code(pool, gens):
    """Files grouped per split (a repo directory decides the split of its files) into one review/explain request each,
    up to a drawn target length; files larger than the room left are skipped."""
    for src, files in gens:
        buf = {"train": [], "eval": []}
        have = {"train": 0, "eval": 0}
        goal = {"train": pool.target("train"), "eval": pool.target("eval")}
        for fid, path, content in files:
            group = fid.split(":")[0] + ":" + os.path.dirname(path) if fid.startswith("repo-") else fid
            sp = split_of(group)
            if pool.done():
                return
            if pool.full(sp):
                continue
            t = ntok(content)
            if t > goal[sp] - have[sp] + 500:
                continue
            fence = EXT_LANG.get(os.path.splitext(path)[1].lower(), "")
            buf[sp].append((fid, "File `%s`:\n```%s\n%s\n```" % (path, fence, content), t)); have[sp] += t
            if have[sp] >= goal[sp] - 500:
                lang = "uk" if rng.random() < 0.3 else "en"
                part = buf[sp]
                msg = rng.choice(CODE_PROMPTS[lang]) + "\n\n" + "\n\n".join(x[1] for x in part)
                pool.add(lang, part[0][0] + (";..." if len(part) > 1 else ""), sp, [{"role": "user", "content": msg}],
                         est=have[sp] + 40)
                buf[sp], have[sp], goal[sp] = [], 0, pool.target(sp)


def commit_pairs(repo, n=400):
    """(diff, real commit message) pairs from a repo's history: the message is the assistant's answer."""
    log = subprocess.run(["git", "-C", repo, "log", "--no-merges", "-n", str(n), "--format=%H"], capture_output=True, text=True).stdout.split()
    for h in log:
        msg = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%B", h], capture_output=True, text=True, encoding="utf-8").stdout.strip()
        diff = subprocess.run(["git", "-C", repo, "show", "--format=", "--stat", "-p", h], capture_output=True, text=True,
                              encoding="utf-8", errors="replace").stdout
        if not msg or not diff or len(diff) > 24_000:
            continue
        diff, msg = clean(diff), clean(msg)
        if diff and msg:
            yield "commit-%s-%s" % (os.path.basename(repo), h[:10]), diff, msg


def build_code():
    c1 = Pool("C1")
    # git diffs with their real commit messages (assistant turns), then repo files, then the-stack-smol
    for repo in (r"D:\Projects\basalt-hy4-gptq", r"D:\Projects\tray_llm"):
        for cid, diff, msg in commit_pairs(repo, 600 if "basalt" in repo else 50):
            sp = split_of(cid)
            if c1.tok[sp] > (QUOTA["C1"] * 0.25 if sp == "train" else QUOTA["C1"] * 0.025):
                continue
            lang = "uk" if rng.random() < 0.3 else "en"
            q = ("Write a commit message for this change:\n\n" if lang == "en" else "Напиши повідомлення коміту для цієї зміни:\n\n") + "```diff\n" + diff + "\n```"
            c1.add(lang, cid, sp, [{"role": "user", "content": q}, {"role": "assistant", "content": msg}])
    pack_code(c1, [("repo-basalt", repo_files(r"D:\Projects\basalt-hy4-gptq", {".cpp", ".hpp", ".cu", ".cuh", ".rs", ".py"}, "src")),
                   ("repo-basalt-rs", repo_files(r"D:\Projects\basalt-hy4-gptq", {".rs"}, "serve-rs")),
                   ("repo-tray", repo_files(r"D:\Projects\tray_llm", {".rs", ".toml"})),
                   ("stack-rust", stack_files("rust")), ("stack-cpp", stack_files("c++")), ("stack-c", stack_files("c")),
                   ("stack-py", stack_files("python")), ("stack-ps", stack_files("powershell")), ("stack-sh", stack_files("shell"))]
              if False else [("mix", roundrobin(
                  repo_files(r"D:\Projects\basalt-hy4-gptq", {".cpp", ".hpp", ".cu", ".cuh", ".py"}, "src"),
                  repo_files(r"D:\Projects\tray_llm", {".rs", ".toml"}),
                  repo_files(r"D:\Projects\basalt-hy4-gptq", {".rs"}, "serve-rs"),
                  stack_files("rust"), stack_files("c++"), stack_files("c"), stack_files("python"),
                  stack_files("powershell"), stack_files("shell")))])
    c1.write()
    c2 = Pool("C2")
    pack_code(c2, [("mix", roundrobin(
        repo_files(r"D:\Projects\basalt-hy4-gptq", {".js", ".html", ".css"}, "serve/web"),
        repo_files(r"D:\Projects\flightdeck", {".ts", ".tsx", ".js"}),
        repo_files(r"D:\Projects\llama.cpp-5090", {".ts", ".svelte", ".js", ".css", ".html"}, "tools/server/webui/src"),
        stack_files("javascript"), stack_files("typescript"), stack_files("html"), stack_files("css")))])
    c2.write()


def roundrobin(*gens):
    """Items from each source in turn, a few files at a time so a repo's directory stays together."""
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            for _ in range(4):
                x = next(g, None)
                if x is None:
                    gens.remove(g)
                    break
                yield x


# ------------------------------------------------------------------ B: function calling
def tool_obj(f):
    """A function description -> an OpenAI tool (parameters made a JSON schema object)."""
    f = dict(f)
    params = f.get("parameters") or {"type": "object", "properties": {}}
    if isinstance(params, dict) and params.get("type") == "dict":
        params = dict(params, type="object")
    if isinstance(params, dict) and "properties" not in params and "type" not in params:
        # xlam style {"arg": {"type": "str", "description": ...}}
        params = {"type": "object", "properties": {k: {kk: vv for kk, vv in v.items()} for k, v in params.items()}}
    return {"type": "function", "function": {"name": f["name"], "description": f.get("description", ""), "parameters": params}}


def call(name, args, i):
    return {"id": "call_%d" % i, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def hermes(path):
    for r in json.load(open(path, encoding="utf-8")):
        conv = r["conversations"]
        sysm = conv[0]["value"] if conv[0]["from"] == "system" else ""
        found = re.findall(r"<tools>\s*(.*?)\s*</tools>", sysm, re.S)
        if not found or not max(found, key=len):
            continue
        try:
            tools = [tool_obj(t.get("function", t)) for t in json.loads(max(found, key=len))]
        except Exception:
            continue
        msgs, k, ok = [], 0, True
        for c in conv[1:]:
            v = c["value"]
            if c["from"] == "human":
                msgs.append({"role": "user", "content": v})
            elif c["from"] == "gpt":
                calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", v, re.S)
                if calls:
                    tc = []
                    for x in calls:
                        try:
                            j = ast.literal_eval(x) if x.startswith("{'") else json.loads(x)
                        except Exception:
                            ok = False
                            break
                        k += 1
                        tc.append(call(j["name"], j.get("arguments", {}), k))
                    text = re.sub(r"<tool_call>.*?</tool_call>", "", v, flags=re.S).strip()
                    msgs.append({"role": "assistant", "content": text or None, "tool_calls": tc})
                else:
                    msgs.append({"role": "assistant", "content": v})
            elif c["from"] == "tool":
                outs = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", v, re.S) or [v]
                last = [m_ for m_ in msgs if m_.get("tool_calls")]
                ids = [t["id"] for t in last[-1]["tool_calls"]] if last else []
                for j, o in enumerate(outs):
                    msgs.append({"role": "tool", "tool_call_id": ids[j] if j < len(ids) else "call_x", "content": o})
        if ok and msgs:
            yield "hermes-" + r["id"], "en", msgs, tools


def glaive(path):
    for i, r in enumerate(json.load(open(path, encoding="utf-8"))):
        sysm = r.get("system", "")
        fn = []
        dec = json.JSONDecoder()
        s = sysm[sysm.find("{"):] if "{" in sysm else ""
        while s.strip():
            try:
                o, e = dec.raw_decode(s)
            except Exception:
                break
            fn.append(o)
            s = s[e:].lstrip()
            s = s[s.find("{"):] if "{" in s else ""
        if not fn:
            continue
        try:
            tools = [tool_obj(f) for f in fn]
        except Exception:
            continue
        parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", r["chat"])
        msgs, k, ok = [], 0, True
        for j in range(1, len(parts) - 1, 2):
            role, txt = parts[j], parts[j + 1].replace("<|endoftext|>", "").strip()
            if role == "USER:":
                msgs.append({"role": "user", "content": txt})
            elif role == "ASSISTANT:":
                m = re.match(r"<functioncall>\s*(\{.*\})\s*$", txt, re.S)
                if m:
                    raw = m.group(1)
                    try:
                        j2 = json.loads(re.sub(r"'(\{.*\})'", r"\1", raw, flags=re.S))
                        args = j2.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                    except Exception:
                        ok = False
                        break
                    k += 1
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [call(j2["name"], args, k)]})
                else:
                    msgs.append({"role": "assistant", "content": txt})
            else:
                msgs.append({"role": "tool", "tool_call_id": "call_%d" % k, "content": txt})
        if ok and len(msgs) >= 2 and any(m.get("tool_calls") for m in msgs):
            yield "glaive-%d" % i, "en", msgs, tools


def parse_pycalls(s):
    """ToolACE '[Name(a=1, b="x"), Other(c=2)]' -> [(name, args)]; names may contain spaces."""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    names = []

    def sub(m):
        names.append(m.group(1))
        return "f%d(" % (len(names) - 1)
    py = re.sub(r"(?:(?<=\[)|(?<=, ))([A-Za-z_][\w .\-]*?)\(", sub, s)
    try:
        tree = ast.parse(py, mode="eval")
        out = []
        for c in tree.body.elts:
            out.append((names[int(c.func.id[1:])], {kw.arg: ast.literal_eval(kw.value) for kw in c.keywords}))
        return out
    except Exception:
        return None


def toolace(path):
    for i, r in enumerate(json.load(open(path, encoding="utf-8"))):
        m = re.search(r"(\[\{.*\}\])", r["system"], re.S)
        if not m:
            continue
        try:
            tools = [tool_obj(t) for t in json.loads(m.group(1))]
        except Exception:
            continue
        msgs, k, ok = [], 0, True
        for c in r["conversations"]:
            v = c["value"]
            if c["from"] == "user":
                msgs.append({"role": "user", "content": v})
            elif c["from"] == "assistant":
                calls = parse_pycalls(v)
                if calls:
                    tc = []
                    for n_, a_ in calls:
                        k += 1
                        tc.append(call(n_, a_, k))
                    msgs.append({"role": "assistant", "content": None, "tool_calls": tc})
                else:
                    msgs.append({"role": "assistant", "content": v})
            elif c["from"] == "tool":
                last = [m_ for m_ in msgs if m_.get("tool_calls")]
                if not last:
                    ok = False
                    break
                msgs.append({"role": "tool", "tool_call_id": last[-1]["tool_calls"][0]["id"], "content": v})
        if ok and msgs:
            yield "toolace-%d" % i, "en", msgs, tools


def xlam(path):
    for r in json.load(open(path, encoding="utf-8")):
        try:
            tools = [tool_obj(t) for t in json.loads(r["tools"])]
            ans = json.loads(r["answers"])
        except Exception:
            continue
        tc = [call(a["name"], a.get("arguments", {}), j + 1) for j, a in enumerate(ans)]
        yield "xlam-%d" % r["id"], "en", [{"role": "user", "content": r["query"]}, {"role": "assistant", "content": None, "tool_calls": tc}], tools


def build_b():
    p = Pool("B")
    gens = [hermes(SRC + r"\NousResearch__hermes-function-calling-v1\func-calling.json"),
            hermes(SRC + r"\NousResearch__hermes-function-calling-v1\func-calling-singleturn.json"),
            toolace(SRC + r"\Team-ACE__ToolACE\data.json"), glaive(SRC + r"\glaiveai__glaive-function-calling-v2\glaive-function-calling-v2.json"),
            xlam(SRC + r"\Salesforce__xlam-function-calling-60k\xlam_function_calling_60k.json")]
    share = [0.30, 0.10, 0.25, 0.20, 0.15]
    for g, sh in zip(gens, share):
        got = {"train": 0, "eval": 0}
        for sid, lang, msgs, tools in g:
            sp = split_of(sid)
            lim = QUOTA["B"] * sh * (1 if sp == "train" else 0.1)
            if got[sp] >= lim:
                if got["train"] >= QUOTA["B"] * sh and got["eval"] >= QUOTA["B"] * sh * 0.1:
                    break
                continue
            before = p.tok[sp]
            if p.add(lang, sid, sp, msgs, tools):
                got[sp] += p.tok[sp] - before
    p.write()


# ------------------------------------------------------------------ A3: public agent trajectories
def build_a3():
    p = Pool("A3")
    sw = pq.read_table(SRC + r"\nebius__SWE-agent-trajectories\data\train-00000-of-00012.parquet",
                       columns=["instance_id", "model_name", "target", "trajectory"]).to_pylist()
    rng.shuffle(sw)
    sw.sort(key=lambda r: not r["target"])          # resolved trajectories first
    oh = pq.read_table(SRC + r"\SWE-Gym__OpenHands-SFT-Trajectories\data\train.success.oss-00000-of-00001.parquet").to_pylist()

    def swe():
        for r in sw:
            msgs = []
            for m in r["trajectory"]:
                role = {"system": "system", "user": "user", "ai": "assistant"}.get(m["role"])
                txt = m.get("text") or m.get("system_prompt") or ""
                if not role or not txt:
                    continue
                if msgs and msgs[-1]["role"] == role:
                    msgs[-1]["content"] += "\n\n" + txt
                else:
                    msgs.append({"role": role, "content": txt})
            yield "swe-%s-%s" % (r["instance_id"], r["model_name"]), msgs

    def ohs():
        for i, r in enumerate(oh):
            yield "openhands-%d" % i, [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
    half = QUOTA["A3"] // 2
    for g, lim in ((ohs(), half), (swe(), QUOTA["A3"])):
        for sid, msgs in g:
            sp = split_of(sid.rsplit("-", 1)[0] if sid.startswith("swe-") else sid)
            if p.done():
                break
            if sp == "train" and p.tok["train"] >= lim:
                continue
            est = sum(ntok(m["content"]) for m in msgs)
            if est > 120_000 or est < 1500:
                continue
            p.add("en", sid, sp, msgs, est=est)
    p.write()


if __name__ == "__main__":
    which = sys.argv[1:] or ["A3", "B", "C", "T"]
    if "A3" in which:
        build_a3()
    if "B" in which:
        build_b()
    if "C" in which:
        build_code()
        print("sanitize:", san)
    if "T" in which:
        build_text()
