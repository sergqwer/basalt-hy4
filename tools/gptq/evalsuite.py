"""evalsuite.py MODE ARM [--port 8320] - generation checks against my running server (one arm = one pack).

  web     held-out JS / HTML / CSS tasks, greedy, no thinking; JS: `node --check` + the task's own node test;
          HTML: parses with no unclosed tags and the ids/classes the JS/CSS reference exist; CSS: brace/colon parse
  loops   greedy thinking runs on puzzle prompts; a loop = repeat coverage >= 0.25 (loopcap's metric) or no end of turn
  replay  every turn of the held-out Claude Code sessions (a2/capture evl-* and eval-split cal-*) re-asked greedy:
          a tool_use that names a tool of the request, args valid against its schema's required keys, Read/Glob/Edit
          paths absolute and inside the session folder (existing for Read), Edit old_string found exactly once in a
          file content the session has shown, Bash commands that parse (bash -n), parallel calls where the reference
          made them
Results: eval/<mode>-<arm>.jsonl (resumable per item) and a summary line.
"""
import argparse, glob, html.parser, json, os, re, subprocess, sys, tempfile, time, urllib.request

G = r"D:\Projects\hy4-data\gptq"
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["web", "loops", "replay", "cjk"])
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("arm")
ap.add_argument("--port", type=int, default=8320)
a = ap.parse_args()
os.makedirs(G + r"\eval", exist_ok=True)
OUT = G + r"\eval\%s-%s.jsonl" % (a.mode, a.arm)
done = {}
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l)
        done[r["id"]] = r


def post(path, body, timeout=3600):
    body = dict(body, model="hy4-flash-nvfp4")
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (a.port, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-api-key": "local", "anthropic-version": "2023-06-01"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def emit(r):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done[r["id"]] = r


# ------------------------------------------------------------------------------------------------- web tasks
WEB = [
    ("js-debounce", "js", "Write a JavaScript module (CommonJS) exporting `debounce(fn, ms)` that returns a debounced function with a `.cancel()` method and passes the latest arguments and `this`.",
     "const {debounce}=require('./sol.js');let n=0,last;const d=debounce(function(x){n++;last=x},30);d(1);d(2);d(3);setTimeout(()=>{if(n!==1||last!==3)process.exit(1);const e=debounce(()=>n++,10);e();e.cancel();setTimeout(()=>{if(n!==1)process.exit(2)},40)},80);"),
    ("js-deepmerge", "js", "Write a CommonJS module exporting `deepMerge(a, b)`: returns a new object, plain objects merged recursively, arrays and other values from b replace a's, neither input mutated.",
     "const {deepMerge}=require('./sol.js');const a={x:{y:1,z:[1]},k:1},b={x:{w:2,z:[2]},k:{q:1}};const r=deepMerge(a,b);if(JSON.stringify(r)!=='{\"x\":{\"y\":1,\"z\":[2],\"w\":2},\"k\":{\"q\":1}}')process.exit(1);if(a.x.w!==undefined)process.exit(2);"),
    ("js-lru", "js", "Write a CommonJS module exporting a class `LRU` with constructor(capacity), get(key) (returns undefined when absent) and set(key, value), evicting the least recently used entry. Use a Map.",
     "const {LRU}=require('./sol.js');const c=new LRU(2);c.set('a',1);c.set('b',2);c.get('a');c.set('c',3);if(c.get('b')!==undefined||c.get('a')!==1||c.get('c')!==3)process.exit(1);"),
    ("js-parse-query", "js", "Write a CommonJS module exporting `parseQuery(str)` that parses a URL query string (with or without a leading '?') into an object; repeated keys become arrays; values are percent-decoded and '+' means space. Do not use URLSearchParams.",
     "const {parseQuery}=require('./sol.js');const r=parseQuery('?a=1&b=hello+world&a=2&c=%D0%BF%D1%80');if(JSON.stringify(r)!==JSON.stringify({a:['1','2'],b:'hello world',c:'\u043f\u0440'}))process.exit(1);"),
    ("js-pool", "js", "Write a CommonJS module exporting `async function mapLimit(items, limit, fn)` that runs async fn over items with at most `limit` in flight and resolves to results in input order; it rejects on the first error.",
     "const {mapLimit}=require('./sol.js');let cur=0,max=0;mapLimit([5,1,3,2,4],2,async x=>{cur++;max=Math.max(max,cur);await new Promise(r=>setTimeout(r,x*5));cur--;return x*2}).then(r=>{if(JSON.stringify(r)!=='[10,2,6,4,8]'||max>2)process.exit(1);return mapLimit([1,2],2,async x=>{if(x===2)throw new Error('e');return x}).then(()=>process.exit(2),()=>{})});"),
    ("js-csv", "js", "Write a CommonJS module exporting `parseCSV(text)` returning an array of rows (arrays of strings), handling quoted fields with commas, doubled quotes and newlines inside quotes, and CRLF line ends.",
     "const {parseCSV}=require('./sol.js');const r=parseCSV('a,b\\r\\n\"x,1\",\"he said \"\"hi\"\"\"\\n\"multi\\nline\",z');if(JSON.stringify(r)!==JSON.stringify([['a','b'],['x,1','he said \"hi\"'],['multi\\nline','z']]))process.exit(1);"),
    ("js-eventemitter", "js", "Write a CommonJS module exporting a class `Emitter` with on(event, fn) returning an unsubscribe function, once(event, fn), and emit(event, ...args) returning the number of listeners called.",
     "const {Emitter}=require('./sol.js');const e=new Emitter();let s=0;const off=e.on('a',x=>s+=x);e.once('a',x=>s+=10*x);if(e.emit('a',1)!==2||s!==11)process.exit(1);off();if(e.emit('a',1)!==0||s!==11)process.exit(2);"),
    ("html-form", "html", "Write a complete single-file HTML page (inline <style> and <script>) with a signup form: name, email, password with a strength meter (id=\"strength\"), and a submit button (id=\"submit\") that stays disabled until all fields are valid. Use semantic HTML and labels.", None),
    ("html-tabs", "html", "Write a complete single-file HTML page with an accessible tabs component (role=tablist/tab/tabpanel, aria-selected, arrow-key navigation) with three tabs. Inline CSS and JS.", None),
    ("html-modal", "html", "Сторінка HTML в одному файлі: кнопка «Відкрити» показує модальне вікно (id=\"modal\") з затемненням фону, закриття по Esc, по кліку на фон і кнопкою «×», фокус повертається на кнопку. Вбудовані CSS і JS.", None),
    ("html-table-sort", "html", "Write one HTML file with a table of 6 products (name, price, stock) whose column headers sort the table on click (ascending, then descending), showing an arrow in the active header. Inline CSS and JS.", None),
    ("css-card-grid", "css", "Write only a CSS stylesheet for a responsive card grid: .grid uses CSS grid with auto-fill columns min 220px, .card has a shadow, rounded corners and a hover lift transition, dark mode via prefers-color-scheme with custom properties.", None),
    ("css-navbar", "css", "Напиши лише CSS для адаптивного меню: .nav — flex-рядок, на екранах вужчих за 640px пункти .nav a стають колонкою, активний пункт .active підкреслений, плавна анімація кольору при наведенні, змінні кольорів у :root.", None),
    ("css-buttons", "css", "Write only CSS for a button system: .btn base, .btn-primary/.btn-secondary/.btn-danger variants, sizes .btn-sm/.btn-lg, disabled state, focus-visible outline, all colors as custom properties.", None),
]


class HP(html.parser.HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr", "!doctype"}

    def __init__(self):
        super().__init__()
        self.stack, self.bad, self.ids, self.classes = [], 0, set(), set()

    def handle_starttag(self, t, attrs):
        d = dict(attrs)
        if d.get("id"):
            self.ids.add(d["id"])
        for c in (d.get("class") or "").split():
            self.classes.add(c)
        if t not in self.VOID:
            self.stack.append(t)

    def handle_endtag(self, t):
        if t in self.VOID:
            return
        if t in self.stack:
            while self.stack and self.stack[-1] != t:
                self.stack.pop()
                self.bad += 1
            self.stack.pop()
        else:
            self.bad += 1


def css_ok(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    depth = 0
    for ch in css:
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            return False
    if depth:
        return False
    for body in re.findall(r"\{([^{}]*)\}", css):
        for decl in body.split(";"):
            if decl.strip() and ":" not in decl:
                return False
    return True


def code_block(text, langs):
    for m in re.finditer(r"```([a-zA-Z]*)\n(.*?)```", text, re.S):
        if m.group(1).lower() in langs:
            return m.group(2)
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def run_web():
    for tid, kind, prompt, test in WEB:
        if tid in done:
            continue
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt + "\nReply with the code in one fenced block."}],
                                         "temperature": 0, "max_tokens": 6000, "reasoning_effort": "none"})
        text = r["choices"][0]["message"].get("content") or ""
        res = {"id": tid, "kind": kind}
        with tempfile.TemporaryDirectory() as d:
            if kind == "js":
                code = code_block(text, {"js", "javascript"})
                ok = code is not None
                if ok:
                    open(os.path.join(d, "sol.js"), "w", encoding="utf-8").write(code)
                    open(os.path.join(d, "t.js"), "w", encoding="utf-8").write(test)
                    c1 = subprocess.run(["node", "--check", "sol.js"], cwd=d, capture_output=True, timeout=30).returncode == 0
                    c2 = c1 and subprocess.run(["node", "t.js"], cwd=d, capture_output=True, timeout=60).returncode == 0
                    res.update(check=c1, test=c2)
                    ok = c2
            elif kind == "html":
                code = code_block(text, {"html"})
                ok = code is not None
                if ok:
                    p = HP()
                    p.feed(code)
                    js = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", code, re.S))
                    open(os.path.join(d, "s.js"), "w", encoding="utf-8").write(js)
                    jsok = subprocess.run(["node", "--check", "s.js"], cwd=d, capture_output=True, timeout=30).returncode == 0
                    refs = set(re.findall(r"getElementById\(['\"]([\w-]+)['\"]\)", js)) | set(re.findall(r"querySelector(?:All)?\(['\"]#([\w-]+)", js))
                    missing = [x for x in refs if x not in p.ids]
                    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", code, re.S))
                    unclosed = len([t for t in p.stack if t not in ("html", "body", "head", "p", "li")])
                    res.update(unclosed=unclosed + p.bad, js_check=jsok, missing_ids=missing, css=css_ok(css))
                    ok = unclosed + p.bad == 0 and jsok and not missing and css_ok(css)
            else:
                code = code_block(text, {"css"})
                ok = code is not None and css_ok(code)
        res["pass"] = bool(ok)
        emit(res)
        print(tid, res, flush=True)
    rs = [done[t[0]] for t in WEB if t[0] in done]
    print("WEB %s: %d/%d pass" % (a.arm, sum(r["pass"] for r in rs), len(rs)))


# ------------------------------------------------------------------------------------------------- loops
LOOP_PROMPTS = [
    "How many positive integers n <= 1000 make n^2 + n + 41 composite? Think it through and give the exact count.",
    "A 3x3 grid is filled with the digits 1-9 so that every row, column and both diagonals sum to the same value. How many such grids exist, counting rotations and reflections as different? Explain.",
    "Find all real x with sqrt(x + 3 - 4 sqrt(x - 1)) + sqrt(x + 8 - 6 sqrt(x - 1)) = 1.",
    "У скриньці 5 червоних і 7 синіх куль. Беремо 4 кулі без повернення. Яка ймовірність, що червоних і синіх порівну? Дай точну відповідь дробом.",
    "Prove or disprove: every tournament on n vertices has a Hamiltonian path. Then count Hamiltonian paths in the transitive tournament on 5 vertices.",
    "What is the last nonzero digit of 100!? Show your method.",
    "Write a Python function that returns the number of distinct palindromic substrings of a string in O(n log n) or better, and explain why it is correct.",
    "A clock's hour and minute hands overlap how many times between 00:00 and 23:59, and at what exact times (to the second) in the first 12 hours?",
]


def coverage(text):   # loopcap.py / the server's reasoning_repeat_coverage
    import collections
    words = re.findall(r"\w+|[^\w\s]", text.lower())[-30000:]
    width, window = 12, 2000
    if len(words) < window:
        return 0.0
    grams = [tuple(words[i:i + width]) for i in range(len(words) - width + 1)]
    counts = collections.Counter(grams)
    start = max(0, len(words) - window)
    covered = set()
    for i in range(start, len(grams)):
        if counts[grams[i]] >= 3:
            covered.update(range(i, i + width))
    return len(covered) / (len(words) - start)


def run_loops():
    for i, p in enumerate(LOOP_PROMPTS):
        tid = "loop-%d" % i
        if tid in done:
            continue
        t0 = time.time()
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 7000})
        ch = r["choices"][0]
        reas = ch["message"].get("reasoning_content") or ""
        cov = coverage(reas + "\n" + (ch["message"].get("content") or ""))
        res = {"id": tid, "finish": ch.get("finish_reason"), "coverage": round(cov, 3), "tokens": r["usage"]["completion_tokens"],
               "loop": bool(cov >= 0.25 or ch.get("finish_reason") != "stop"), "s": round(time.time() - t0)}
        emit(res)
        print(res, flush=True)
    rs = [done["loop-%d" % i] for i in range(len(LOOP_PROMPTS)) if "loop-%d" % i in done]
    print("LOOPS %s: %d/%d looped, mean tokens %.0f" % (a.arm, sum(r["loop"] for r in rs), len(rs), sum(r["tokens"] for r in rs) / max(1, len(rs))))


# ------------------------------------------------------------------------------------------------- replay
def shown_texts(msgs):
    """Every tool_result text the session has shown so far."""
    out = []
    for m in msgs:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                    out.append(c or "")
    return out


def bash_parses(cmd):
    try:
        return subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def check_call(tu, req, ref_calls, box):
    tools = {t["name"]: t for t in req.get("tools") or []}
    name, inp = tu.get("name"), tu.get("input")
    r = {"tool": name, "known": name in tools, "json": isinstance(inp, dict)}
    if not (r["known"] and r["json"]):
        r["valid"] = False
        return r
    need = tools[name].get("input_schema", {}).get("required", [])
    r["required"] = all(k in inp for k in need)
    ok = r["required"]
    p = inp.get("file_path") or inp.get("path")
    if name in ("Read", "Edit", "Write", "Glob") and p:
        ap_ = p.replace("\\", "/")
        r["abs"] = bool(re.match(r"^([A-Za-z]:/|/)", ap_))
        r["inside"] = ap_.lower().startswith(box.lower())
        ok = ok and r["abs"]
        if name == "Read":
            r["exists"] = os.path.exists(p) or os.path.exists(os.path.join(box, os.path.basename(p)))
    if name == "Edit":
        texts = shown_texts(req["messages"])
        old = inp.get("old_string", "")
        r["edit_once"] = any(t.count(old) == 1 for t in texts) if old else False
        ok = ok and r["edit_once"]
    if name == "Bash":
        r["parses"] = bash_parses(inp.get("command", ""))
        ok = ok and r["parses"]
    r["valid"] = bool(ok)
    return r


def replay_items():
    tasks = {t["id"]: t for t in json.load(open(G + r"\a2\tasks.json", encoding="utf-8"))}
    import hashlib
    for cap in sorted(glob.glob(G + r"\a2\capture\*.jsonl")):
        tid = os.path.basename(cap)[:-6]
        tag, base = tid.split("-", 1)
        # a fixed set for every arm: the first three RTN eval sessions + the calibration sessions held out by hash
        held = (tag == "evl" and base[:3] in ("t24", "t25", "t26")) or             (tag == "cal" and int(hashlib.sha1(base.encode()).hexdigest(), 16) % 10 == 0)
        if not held:
            continue
        for k, line in enumerate(open(cap, encoding="utf-8")):
            r = json.loads(line)
            if r.get("status") != 200 or not (r.get("req") or {}).get("tools"):
                continue
            yield "%s-%03d" % (tid, k), tid, r


def ref_tool_uses(sse):
    names = re.findall(r'"content_block_start".*?"type": ?"tool_use".*?"name": ?"([^"]+)"', sse)
    return names


def run_replay():
    for iid, tid, r in replay_items():
        if iid in done:
            continue
        refs = ref_tool_uses(r["resp"])
        if not refs:          # the reference turn answered in text: not a tool-call turn
            continue
        q = json.loads(json.dumps(r["req"]))
        q.update(stream=False, temperature=0, max_tokens=min(q.get("max_tokens", 32000), 16000))
        t0 = time.time()
        try:
            out = post("/v1/messages", q)
        except Exception as e:
            emit({"id": iid, "error": str(e)[:200], "valid": False})
            continue
        tus = [b for b in out.get("content", []) if b.get("type") == "tool_use"]
        box = (G + "\\sandbox\\" + tid).replace("\\", "/")
        checks = [check_call(tu, q, refs, box) for tu in tus]
        res = {"id": iid, "ref_tools": refs, "got_tools": [c["tool"] for c in checks],
               "tool_turn": bool(tus), "valid": bool(tus) and all(c["valid"] for c in checks),
               "same_first_tool": bool(tus) and checks[0]["tool"] == refs[0],
               "parallel_ref": len(refs) > 1, "parallel_got": len(tus) > 1, "checks": checks, "s": round(time.time() - t0, 1)}
        emit(res)
        print(iid, res["valid"], res["got_tools"], refs, flush=True)
    rs = list(done.values())
    n = len(rs)
    par = [r for r in rs if r.get("parallel_ref")]
    print("REPLAY %s: %d turns, tool_use %d, valid %d, same first tool %d, parallel kept %d/%d" % (
        a.arm, n, sum(r.get("tool_turn", False) for r in rs), sum(r.get("valid", False) for r in rs),
        sum(r.get("same_first_tool", False) for r in rs), sum(r.get("parallel_got", False) for r in par), len(par)))


# ------------------------------------------------------------------------------------------------- CJK drift
CJK_PROMPTS = [
    "Поясни, як працює збирач сміття в Python: підрахунок посилань і циклічний збирач, з прикладом коду.",
    "Напиши функцію на Rust, яка парсить рядок дати у форматі РРРР-ММ-ДД, і поясни обробку помилок.",
    "Чим відрізняється TCP від UDP? Наведи приклади, коли що обирати, і поясни керування перевантаженням.",
    "Склади план міграції бази PostgreSQL з версії 13 на 16 без простою. Опиши ризики.",
    "Поясни різницю між процесом і потоком в ОС, а також що таке контекстне перемикання.",
    "Як захистити веб-застосунок від XSS і CSRF? Дай конкретні заголовки й приклади коду.",
    "Опиши, як працює алгоритм Дейкстри, і напиши його реалізацію на Python з купою.",
    "Переклади й поясни українською: 'premature optimization is the root of all evil'. Чи погоджуєшся ти?",
    "Що таке квантування нейромереж до 4 біт і чому GPTQ краще за просте округлення?",
    "Напиши bash-скрипт, що знаходить 10 найбільших файлів у теці, і поясни кожен рядок.",
    "Поясни, як працює HTTPS-рукостискання TLS 1.3 крок за кроком.",
    "Розкажи коротко історію Києво-Печерської лаври і чому вона важлива.",
    "Объясни, как работает сборщик мусора в Java (G1), простыми словами.",
    "Напиши на C++ потокобезопасную очередь с condition_variable и объясни, где может быть гонка.",
    "Чем отличается индекс B-tree от hash-индекса в базах данных? Когда какой использовать?",
    "Поясни різницю між async/await у JavaScript і Python на прикладах.",
]
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def run_cjk():
    for i, p in enumerate(CJK_PROMPTS):
        tid = "cjk-%d" % i
        if tid in done:
            continue
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 3000, "seed": a.seed})
        m = r["choices"][0]["message"]
        reas, txt = m.get("reasoning_content") or "", m.get("content") or ""
        res = {"id": tid, "han_reply": len(HAN.findall(txt)), "han_thinking": len(HAN.findall(reas)),
               "chars": len(txt) + len(reas), "finish": r["choices"][0].get("finish_reason")}
        res["drift"] = bool(res["han_reply"] or res["han_thinking"])
        emit(res)
        print(res, flush=True)
    rs = [done["cjk-%d" % i] for i in range(len(CJK_PROMPTS)) if "cjk-%d" % i in done]
    print("CJK %s: %d/%d replies with Han characters (reply %d, thinking %d chars total)" % (
        a.arm, sum(r["drift"] for r in rs), len(rs), sum(r["han_reply"] for r in rs), sum(r["han_thinking"] for r in rs)))


{"web": run_web, "loops": run_loops, "replay": run_replay, "cjk": run_cjk}[a.mode]()
