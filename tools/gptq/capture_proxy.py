"""capture_proxy.py OUT_JSONL [--listen 8321] [--upstream 127.0.0.1:8320]
Forwards HTTP to the Basalt server and appends {"t", "path", "req": <request JSON body>, "resp": <response body text>}
per /v1/messages call. Headers are forwarded but NEVER stored."""
import http.client, http.server, json, sys, threading, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--listen", type=int, default=8321)
ap.add_argument("--upstream", default="127.0.0.1:8320"); a = ap.parse_args()
lock = threading.Lock()
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host"}

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *x): pass
    def _go(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        up = http.client.HTTPConnection(a.upstream, timeout=3600)
        hdr = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        up.request(self.command, self.path, body=body, headers=hdr)
        r = up.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP: self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        buf = []
        while True:
            c = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not c: break
            buf.append(c)
            self.wfile.write(b"%x\r\n%s\r\n" % (len(c), c)); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        if self.path.startswith("/v1/messages") and self.command == "POST":
            try: req = json.loads(body)
            except Exception: req = None
            rec = {"t": time.time(), "path": self.path, "status": r.status, "req": req,
                   "resp": b"".join(buf).decode("utf-8", "replace")}
            with lock, open(a.out, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    do_POST = _go; do_GET = _go
http.server.ThreadingHTTPServer(("127.0.0.1", a.listen), H).serve_forever()
