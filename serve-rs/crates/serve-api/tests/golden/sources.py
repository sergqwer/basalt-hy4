"""image_source goldens: host parsing, address tables, Windows path rules, file: URLs, load_image under policies."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from serve.engine import image_source as S  # noqa: E402


def run(f, *a, **k):
    try:
        return {"ok": f(*a, **k)}
    except ValueError as e:
        return {"error": str(e), "log": getattr(e, "log_line", None)}


hosts = ["example.com", "Example.COM:8080", "[::1]", "[::1]:99", "1.2.3.4:80", "a..b", ".x", "x.", "x:0", "x:65536",
         "[zz]", "", "  host  ", "h:99999", "*.x", "xn--d1a.ua"]
addrs = ["8.8.8.8", "10.1.1.1", "100.64.0.1", "127.0.0.1", "192.168.1.1", "172.16.5.5", "224.0.0.1", "2606:4700::1111",
         "fe80::1", "::1", "2001:db8::1", "::ffff:8.8.8.8", "not an ip", "1.2.3", "2002::1"]
wpaths = [r"C:\x\y.png", r"C:/x/y.png", r"\server\share\a.png", r"C:x.png", r"C:\a\CON.png", r"C:\a\b.png:ads", r"C:\a\b. ",
          r"C:\a\..\b.png", r"C:\a\com1.txt", r"C:\a\LPT¹", "relative.png"]
furls = ["file:///C:/x/y.png", "file://C:/x/y.png", "file://localhost/etc/x", "file://host/x", "file:/tmp/x%20y", "file:///tmp/a%25b"]
remote = S.ImageSources(why="peer", peer="10.0.0.9")
allowed = S.ImageSources(url_hosts=(("example.com", None),), why="host", peer="1.1.1.1")
loads = [("data:image/png;base64,aGVsbG8=", remote), ("data:image/png,aGVsbG8=", remote), ("data:x", remote),
         ("data:;base64,", remote), ("data:;base64,a$b", remote), ("data:;base64,aGVs bG8=\n", remote),
         ("http://example.com/a.png", remote), ("https://evil.com/a.png", allowed), ("http://user@example.com/a", allowed),
         ("ftp://x/y", remote), ("C:/secret.png", remote), ("/etc/passwd", remote), ("", remote), (5, remote),
         ("http://exa mple.com/", allowed)]
out = {"hosts": [[h, run(S.parse_url_host, h)] for h in hosts],
       "addrs": [[a, S.is_public_address(a)] for a in addrs],
       "wpaths": [[p, S.windows_path_refused(p)] for p in wpaths],
       "furls": [[u, run(S.file_url_path, u, True), run(S.file_url_path, u, True, True)] for u in furls],
       "loads": [[src, {"url_hosts": [list(h) for h in pol.url_hosts], "why": pol.why, "peer": pol.peer},
                  (lambda r: {"ok": __import__("base64").b64encode(r["ok"]).decode()} if "ok" in r else r)(run(S.load_image, src, pol))]
                 for src, pol in loads]}
Path(__file__).with_name("sources.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
print("ok")
