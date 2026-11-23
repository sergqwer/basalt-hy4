"""Seed projects for the A2 Claude Code sessions (each task runs in its own copy under sandbox/<task>)."""
import pathlib, textwrap, shutil, subprocess
S = pathlib.Path(r"D:\Projects\hy4-data\gptq\a2\seeds")


def w(rel, txt):
    p = S / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(textwrap.dedent(txt).lstrip("\n").encode())


# pyinv: python inventory with bugs + tests
w("pyinv/inventory.py", '''
    """Tiny inventory manager."""
    import json, sys

    class Inventory:
        def __init__(self):
            self.items = {}

        def add(self, name, qty, price):
            if qty <= 0:
                raise ValueError("qty must be positive")
            cur = self.items.get(name)
            if cur:
                cur["qty"] = qty            # BUG? overwrites instead of adding
                cur["price"] = price
            else:
                self.items[name] = {"qty": qty, "price": price}

        def remove(self, name, qty):
            cur = self.items[name]
            cur["qty"] -= qty
            if cur["qty"] < 0:
                cur["qty"] = 0

        def total_value(self):
            return sum(i["qty"] + i["price"] for i in self.items.values())

        def to_json(self):
            return json.dumps(self.items, sort_keys=True)

        @classmethod
        def from_json(cls, s):
            inv = cls(); inv.items = json.loads(s); return inv

    def main(argv):
        inv = Inventory()
        for line in open(argv[1], encoding="utf-8"):
            name, qty, price = line.split(",")
            inv.add(name.strip(), int(qty), float(price))
        print(f"{len(inv.items)} items, total {inv.total_value():.2f}")

    if __name__ == "__main__":
        main(sys.argv)
''')
w("pyinv/test_inventory.py", '''
    from inventory import Inventory

    def test_add_accumulates():
        inv = Inventory(); inv.add("bolt", 3, 0.5); inv.add("bolt", 2, 0.5)
        assert inv.items["bolt"]["qty"] == 5

    def test_total_value():
        inv = Inventory(); inv.add("bolt", 4, 0.25); inv.add("nut", 10, 0.1)
        assert abs(inv.total_value() - 2.0) < 1e-9

    def test_remove_floor():
        inv = Inventory(); inv.add("x", 1, 1.0); inv.remove("x", 5)
        assert inv.items["x"]["qty"] == 0

    def test_roundtrip():
        inv = Inventory(); inv.add("a", 1, 2.0)
        assert Inventory.from_json(inv.to_json()).items == inv.items
''')
w("pyinv/stock.csv", "bolt, 100, 0.05\nnut, 250, 0.02\nwasher, 75, 0.01\nbracket, 12, 1.75\n")

# rustq: ring buffer with a missing wrap
w("rustq/Cargo.toml", '''
    [package]
    name = "rustq"
    version = "0.1.0"
    edition = "2021"

    [dependencies]
''')
w("rustq/src/lib.rs", '''
    /// A fixed-capacity ring buffer that overwrites the oldest element when full.
    pub struct Ring<T> {
        buf: Vec<Option<T>>,
        head: usize,
        len: usize,
    }

    impl<T: Clone> Ring<T> {
        pub fn new(cap: usize) -> Self {
            assert!(cap > 0);
            Ring { buf: vec![None; cap], head: 0, len: 0 }
        }
        pub fn capacity(&self) -> usize { self.buf.len() }
        pub fn len(&self) -> usize { self.len }
        pub fn is_empty(&self) -> bool { self.len == 0 }

        pub fn push(&mut self, v: T) {
            let cap = self.capacity();
            let tail = (self.head + self.len) % cap;
            self.buf[tail] = Some(v);
            if self.len == cap {
                self.head = (self.head + 1) % cap;
            } else {
                self.len += 1;
            }
        }

        pub fn pop(&mut self) -> Option<T> {
            if self.len == 0 { return None; }
            let v = self.buf[self.head].take();
            self.head = self.head + 1;          // missing wrap
            self.len -= 1;
            v
        }

        pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
            (0..self.len).map(move |i| self.buf[(self.head + i) % self.capacity()].as_ref().unwrap())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn wraps() {
            let mut r = Ring::new(3);
            for i in 0..5 { r.push(i); }
            assert_eq!(r.iter().cloned().collect::<Vec<_>>(), vec![2, 3, 4]);
            assert_eq!(r.pop(), Some(2));
            assert_eq!(r.pop(), Some(3));
            r.push(9); r.push(10);
            assert_eq!(r.iter().cloned().collect::<Vec<_>>(), vec![4, 9, 10]);
            assert_eq!(r.pop(), Some(4));
            assert_eq!(r.pop(), Some(9));
            assert_eq!(r.pop(), Some(10));
            assert_eq!(r.pop(), None);
        }
    }
''')

# cppstat: C++ median / stddev bugs
w("cppstat/stats.cpp", '''
    // stats: reads numbers (one per line) from a file and prints count, mean, median, stddev
    #include <algorithm>
    #include <cmath>
    #include <fstream>
    #include <iostream>
    #include <vector>

    static double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        if (n % 2 == 0) return v[n / 2];            // wrong for even n
        return v[n / 2];
    }

    int main(int argc, char** argv) {
        if (argc < 2) { std::cerr << "usage: stats FILE\\n"; return 2; }
        std::ifstream f(argv[1]);
        std::vector<double> v; double x;
        while (f >> x) v.push_back(x);
        if (v.empty()) { std::cerr << "no data\\n"; return 1; }
        double s = 0; for (double a : v) s += a;
        double mean = s / v.size();
        double var = 0; for (double a : v) var += (a - mean) * (a - mean);
        var /= v.size();                            // population or sample?
        std::cout << "count " << v.size() << "\\nmean " << mean << "\\nmedian " << median(v)
                  << "\\nstddev " << std::sqrt(var) << "\\n";
    }
''')
w("cppstat/data.txt", "4\n8\n15\n16\n23\n42\n")
w("cppstat/README.md", "Expected for data.txt: count 6, mean 18, median 15.5, sample stddev 13.4907\n")

# nodecli: log parser
w("nodecli/parse.js", r'''
    // Count HTTP status codes in an access log: node parse.js access.log
    const fs = require("fs");

    function parseLine(line) {
      const m = line.match(/"(GET|POST|PUT|DELETE) ([^ ]+) HTTP\/[0-9.]+" (\d{3}) (\d+)/);
      if (!m) return null;
      return { method: m[1], path: m[2], status: m[3], bytes: m[4] };
    }

    function summarize(lines) {
      const by = {};
      for (const l of lines) {
        const r = parseLine(l);
        if (!r) continue;
        by[r.status] = (by[r.status] || 0) + 1;
      }
      return by;
    }

    if (require.main === module) {
      const lines = fs.readFileSync(process.argv[2], "utf8").split("\n");
      console.log(JSON.stringify(summarize(lines), null, 2));
    }
    module.exports = { parseLine, summarize };
''')
w("nodecli/access.log", "\n".join([
    '10.0.0.1 - - [22/Nov/2026:10:00:01 +0200] "GET /index.html HTTP/1.1" 200 5120',
    '10.0.0.2 - - [22/Nov/2026:10:00:02 +0200] "GET /missing HTTP/1.1" 404 312',
    '10.0.0.3 - - [22/Nov/2026:10:00:03 +0200] "POST /api/login HTTP/1.1" 302 0',
    '10.0.0.1 - - [22/Nov/2026:10:00:04 +0200] "GET /app.js HTTP/2.0" 200 88213',
    '10.0.0.4 - - [22/Nov/2026:10:00:05 +0200] "PATCH /api/user/7 HTTP/1.1" 204 0',
    '10.0.0.5 - - [22/Nov/2026:10:00:06 +0200] "GET /api/items?page=2 HTTP/1.1" 500 97',
    'garbage line',
    '10.0.0.2 - - [22/Nov/2026:10:00:07 +0200] "HEAD / HTTP/1.1" 200 0', ""]))

# messy: refactor target
w("messy/report.py", '''
    import csv,sys,statistics
    rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8")))
    d={}
    for r in rows:
     k=r["region"]
     if k not in d:d[k]=[]
     d[k].append(float(r["amount"]))
    out=[]
    for k in sorted(d):
     v=d[k]
     out.append((k,len(v),sum(v),statistics.mean(v),max(v)))
    print("region,count,total,mean,max")
    for o in out:print("%s,%d,%.2f,%.2f,%.2f"%o)
    best=None
    for o in out:
     if best is None or o[2]>best[2]:best=o
    print("best region:",best[0])
''')
w("messy/sales.csv", "region,amount\nnorth,120.5\nsouth,99\nnorth,80\neast,300.25\nsouth,10\nwest,55.5\neast,12\n")

# jsasync: in-flight sharing bug
w("jsasync/cache.js", '''
    // A tiny async cache around a loader function.
    class Cache {
      constructor(loader, ttlMs = 1000) { this.loader = loader; this.ttl = ttlMs; this.map = new Map(); }
      async get(key) {
        const hit = this.map.get(key);
        if (hit && Date.now() - hit.t < this.ttl) return hit.v;
        const v = this.loader(key);          // should share one in-flight load per key
        this.map.set(key, { v, t: Date.now() });
        return v;
      }
      clear() { this.map.clear(); }
    }
    module.exports = { Cache };
''')
w("jsasync/test.js", '''
    const assert = require("assert");
    const { Cache } = require("./cache");
    (async () => {
      let calls = 0;
      const c = new Cache(async (k) => { calls++; await new Promise(r => setTimeout(r, 20)); if (k === "bad") throw new Error("boom"); return k.toUpperCase(); });
      const [a, b] = await Promise.all([c.get("x"), c.get("x")]);
      assert.strictEqual(a, "X"); assert.strictEqual(b, "X");
      assert.strictEqual(calls, 1, "concurrent gets must share one load");
      await assert.rejects(c.get("bad"));
      await assert.rejects(c.get("bad"));
      assert.strictEqual(calls, 3, "a failed load must not be cached");
      console.log("ok");
    })().catch(e => { console.error(e.message); process.exit(1); });
''')

# htmlfix: broken page
w("htmlfix/index.html", '''
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Gallery</title>
      <link rel="stylesheet" href="style.css">
    </head>
    <body>
      <header><h1>Photo gallery</h1>
      <nav><a href="#" id="grid-btn">Grid</a> <a href="#" id="list-btn">List</a></nav>
      <main id="gallery" class="grid">
        <figure><div class="ph" style="background:#c33"></div><figcaption>Red</figcaption></figure>
        <figure><div class="ph" style="background:#3c3"></div><figcaption>Green</figcaption>
        <figure><div class="ph" style="background:#33c"></div><figcaption>Blue</figcaption></figure>
      </main>
      <script src="app.js"></script>
    </body>
    </html>
''')
w("htmlfix/style.css", '''
    body { font-family: system-ui; margin: 0 }
    header { background: #222; color: #fff; padding: 1rem }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; padding: 1rem }
    .list figure { display: flex; align-items: center; gap: 1rem }
    .ph { width: 100%; aspect-ratio: 4/3; border-radius: 8px
    .list .ph { width: 80px }
''')
w("htmlfix/app.js", '''
    const g = document.getElementById("galery");
    document.getElementById("grid-btn").onclick = () => { g.className = "grid" };
    document.getElementById("list-btn").addEventListener("click", () => g.classList = "list");
''')

# htmlparse: a saved page -> CSV
rows = "".join(f"<tr><td>{c}</td><td>{p}</td><td>{a}</td></tr>\n" for c, p, a in [
    ("Київ", "2 952 301", "839"), ("Харків", "1 421 125", "350"), ("Одеса", "1 010 537", "236.9"),
    ("Дніпро", "968 502", "409.7"), ("Львів", "717 273", "182")])
w("htmlparse/cities.html", "<html><head><meta charset='utf-8'><title>Міста</title></head><body>\n<h2>Найбільші міста</h2>\n"
  "<table id='t'><thead><tr><th>Місто</th><th>Населення</th><th>Площа, км²</th></tr></thead><tbody>\n" + rows +
  "</tbody></table>\n<p>Дані: 2022.</p></body></html>\n")

# secsrv: path traversal
w("secsrv/server.py", '''
    """A minimal static file server for ./public (python server.py 8000)."""
    import http.server, os, sys, urllib.parse

    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = urllib.parse.unquote(self.path.split("?")[0])
            full = os.path.join(ROOT, path.lstrip("/"))
            if os.path.isdir(full):
                full = os.path.join(full, "index.html")
            try:
                data = open(full, "rb").read()
            except OSError:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    if __name__ == "__main__":
        http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
''')
w("secsrv/public/index.html", "<h1>hello</h1>\n")

# cssvars: hard-coded colors
w("cssvars/site.css", '''
    body { background: #ffffff; color: #1d1d1f; font: 16px/1.5 Georgia, serif }
    a { color: #0a58ca }
    a:hover { color: #084298 }
    .card { background: #f6f6f8; border: 1px solid #d0d0d7; border-radius: 6px; padding: 16px }
    .card h3 { color: #1d1d1f; margin-top: 0 }
    .btn { background: #0a58ca; color: #ffffff; border: 0; padding: 8px 14px; border-radius: 4px }
    .btn:hover { background: #084298 }
    .muted { color: #6e6e73 }
    footer { border-top: 1px solid #d0d0d7; color: #6e6e73; padding: 24px 0 }
''')
w("cssvars/index.html", "<!doctype html><html><head><meta charset='utf-8'><link rel='stylesheet' href='site.css'></head><body>"
  "<div class='card'><h3>Title</h3><p class='muted'>Text</p><button class='btn'>Go</button></div><footer>(c)</footer></body></html>\n")

# sqlbug: injection + empty average
w("sqlbug/app.py", '''
    import sqlite3, sys

    def init(db):
        db.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT UNIQUE, age INTEGER)")

    def add_user(db, name, age):
        db.execute(f"INSERT INTO users(name, age) VALUES('{name}', {age})")

    def find(db, name):
        return db.execute(f"SELECT id, name, age FROM users WHERE name = '{name}'").fetchall()

    def average_age(db):
        rows = db.execute("SELECT age FROM users").fetchall()
        return sum(r[0] for r in rows) / len(rows)

    if __name__ == "__main__":
        db = sqlite3.connect(":memory:"); init(db)
        for n, a in [("ann", 31), ("bob", 25), ("o'brien", 40)]:
            add_user(db, n, a)
        print(find(db, "o'brien"), average_age(db))
''')

# cppcmake: library without a build
w("cppcmake/src/strutil.hpp", '''
    #pragma once
    #include <string>
    #include <vector>
    std::vector<std::string> split(const std::string& s, char sep);
    std::string trim(const std::string& s);
    std::string join(const std::vector<std::string>& v, const std::string& sep);
''')
w("cppcmake/src/strutil.cpp", '''
    #include "strutil.hpp"
    std::vector<std::string> split(const std::string& s, char sep) {
        std::vector<std::string> out; std::string cur;
        for (char c : s) { if (c == sep) { out.push_back(cur); cur.clear(); } else cur += c; }
        out.push_back(cur);
        return out;
    }
    std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \\t\\n"), b = s.find_last_not_of(" \\t\\n");
        return s.substr(a, b - a + 1);   // crashes on all-space input
    }
    std::string join(const std::vector<std::string>& v, const std::string& sep) {
        std::string r; for (size_t i = 0; i < v.size(); ++i) { if (i) r += sep; r += v[i]; } return r;
    }
''')

# todoapp: a working app for an "add a feature" task
w("todoapp/index.html", '''
    <!doctype html>
    <html lang="uk"><head><meta charset="utf-8"><title>Справи</title><link rel="stylesheet" href="app.css"></head>
    <body><main>
    <h1>Мої справи</h1>
    <form id="f"><input id="t" placeholder="Нова справа" required><button>Додати</button></form>
    <ul id="list"></ul>
    <p id="count"></p>
    </main><script src="app.js"></script></body></html>
''')
w("todoapp/app.css", "main{max-width:480px;margin:2rem auto;font-family:system-ui}li{display:flex;gap:.5rem;align-items:center;padding:.25rem 0}"
  "li.done span{text-decoration:line-through;color:#888}\n")
w("todoapp/app.js", '''
    const KEY = "todos";
    let todos = JSON.parse(localStorage.getItem(KEY) || "[]");
    const list = document.getElementById("list");
    function save() { localStorage.setItem(KEY, JSON.stringify(todos)); }
    function render() {
      list.innerHTML = "";
      todos.forEach((t, i) => {
        const li = document.createElement("li");
        if (t.done) li.className = "done";
        li.innerHTML = `<input type="checkbox" ${t.done ? "checked" : ""}><span>${t.text}</span><button>×</button>`;
        li.querySelector("input").onchange = () => { t.done = !t.done; save(); render(); };
        li.querySelector("button").onclick = () => { todos.splice(i, 1); save(); render(); };
        list.appendChild(li);
      });
      document.getElementById("count").textContent = `Залишилось: ${todos.filter(t => !t.done).length}`;
    }
    document.getElementById("f").onsubmit = (e) => {
      e.preventDefault();
      const inp = document.getElementById("t");
      todos.push({ text: inp.value, done: false }); inp.value = ""; save(); render();
    };
    render();
''')

(S / "empty").mkdir(parents=True, exist_ok=True)
(S / "empty" / "README.md").write_text("Empty project folder.\n", encoding="utf-8")

# gitrepo: pyinv with one commit and staged changes
g = S / "gitrepo"
if not (g / ".git").exists():
    shutil.copytree(S / "pyinv", g, dirs_exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=g)
    subprocess.run(["git", "add", "-A"], cwd=g)
    subprocess.run(["git", "-c", "user.name=seed", "-c", "user.email=seed@example.invalid", "commit", "-qm", "initial inventory"], cwd=g)
    p = g / "inventory.py"
    p.write_bytes(p.read_bytes().replace(b'cur["qty"] = qty            # BUG? overwrites instead of adding', b'cur["qty"] += qty'))
    (g / "CHANGES.md").write_text("- add() accumulates quantities\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=g)
print("seeds in", S)
