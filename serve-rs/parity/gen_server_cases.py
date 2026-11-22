import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "cases" / "server"
OUT.mkdir(parents=True, exist_ok=True)
CHAT = {"model": "basalt", "messages": [{"role": "user", "content": "hi"}]}
cases = {}


def c(name, **kw):
    cases[name] = {"name": name, **kw}


def get(name, path, **kw):
    c(name, method="GET", path=path, **kw)


# static files and info routes
get("get_index", "/")
get("get_index_query", "/?x=1")
get("web_css", "/web/app.css")
get("web_js", "/web/app.js")
get("web_svg", "/web/favicon.svg")
get("web_png", "/web/favicon-32.png")
get("web_missing", "/web/missing.js")
get("web_bad_ext", "/web/index.html")
get("web_dotdot", "/web/../x")
get("web_dotdot_index", "/web/../serve/web/app.js")
get("fonts_real", "/fonts/geist-latin-wght.woff2")
get("fonts_missing", "/fonts/x.ttf")
get("fonts_wrong_ext", "/fonts/OFL.txt")
get("health", "/health")
get("api_health", "/api/health")
get("health_trailing_slash", "/health/")
get("health_query", "/health?verbose=1&x")
get("status", "/status")
get("v1_models", "/v1/models")
get("models", "/models")
get("props", "/props")
get("props_unknown_model", "/props?model=nope")
get("props_known_model", "/props?model=basalt")
get("slots", "/slots")
get("v1_status", "/v1/status")
get("metrics", "/metrics")
get("metrics_unknown_model", "/metrics?model=nope")
get("api_logs", "/api/logs")
get("api_logs_after_limit", "/api/logs?after=0&limit=5")
get("api_logs_bad_numbers", "/api/logs?after=x&limit=-3")
get("unknown_get", "/nope")
get("double_slash", "//health")
get("v1_unknown_get", "/v1/chat/completions")

# auth
KEY = ["--api-key", "k"]
get("auth_no_key", "/v1/models", server_args=KEY)
get("auth_wrong_key", "/v1/models", server_args=KEY, headers={"Authorization": "Bearer nope"})
get("auth_bearer", "/v1/models", server_args=KEY, headers={"Authorization": "Bearer k"})
get("auth_bearer_lowercase", "/v1/models", server_args=KEY, headers={"Authorization": "bearer k"})
get("auth_x_api_key", "/v1/models", server_args=KEY, headers={"x-api-key": "k"})
get("auth_health_open", "/health", server_args=KEY)
get("auth_status", "/status", server_args=KEY)
get("auth_metrics", "/metrics", server_args=KEY, headers={"x-api-key": "wrong"})
get("auth_key_file", "/v1/models", server_args=["--api-key-file", "{fixtures}/api_key.txt"],
    headers={"Authorization": "Bearer filekey"})
get("auth_key_file_wrong", "/v1/models", server_args=["--api-key-file", "{fixtures}/api_key.txt"],
    headers={"Authorization": "Bearer k"})
c("auth_post_no_key", method="POST", path="/v1/chat/completions", server_args=KEY, body=CHAT)
c("start_empty_api_key", server_args=["--api-key", " "])
c("start_key_and_key_file", server_args=["--api-key", "k", "--api-key-file", "{fixtures}/api_key.txt"])
c("start_missing_key_file", server_args=["--api-key-file", "{fixtures}/nope.txt"])
c("start_payload_zero", server_args=["--max-payload-mib", "0"])
c("start_bad_cors", server_args=["--cors-origin", "chat.example.com"])
c("start_bad_allowed_host", server_args=["--allowed-host", "a b"])

# Host checks
get("host_evil", "/health", headers={"Host": "evil.com"})
get("host_localhost_port", "/health", headers={"Host": "localhost:{port}"})
get("host_missing", "/health", headers={"Host": None})
get("host_sub_localhost", "/health", headers={"Host": "app.localhost"})
get("host_allowed_x", "/health", server_args=["--allowed-host", "x"], headers={"Host": "x"})
get("host_allowed_x_other", "/health", server_args=["--allowed-host", "x"], headers={"Host": "y"})
get("host_allowed_star", "/health", server_args=["--allowed-host", "*"], headers={"Host": "evil.com"})
get("host_evil_with_key", "/health", server_args=KEY, headers={"Host": "evil.com"})

# Origin checks on /v1 POST, CORS
c("origin_foreign", method="POST", path="/v1/chat/completions", body=CHAT, headers={"Origin": "https://evil.com"})
c("origin_own", method="POST", path="/v1/chat/completions", body=CHAT, headers={"Origin": "http://127.0.0.1:{port}"})
c("origin_own_text_plain", method="POST", path="/v1/chat/completions", body=CHAT,
  headers={"Origin": "http://localhost:{port}", "Content-Type": "text/plain"})
c("origin_null", method="POST", path="/v1/chat/completions", body=CHAT, headers={"Origin": "null"})
CORS = ["--cors-origin", "https://chat.example.com"]
c("cors_preflight", method="OPTIONS", path="/v1/chat/completions", server_args=CORS,
  headers={"Origin": "https://chat.example.com", "Access-Control-Request-Method": "POST",
           "Access-Control-Request-Headers": "authorization, x-custom"})
c("cors_preflight_default_headers", method="OPTIONS", path="/v1/models", server_args=CORS,
  headers={"Origin": "https://chat.example.com"})
c("cors_preflight_other_origin", method="OPTIONS", path="/v1/models", server_args=CORS,
  headers={"Origin": "https://other.example.com"})
c("cors_star", method="GET", path="/v1/models", server_args=["--cors-origin", "*"],
  headers={"Origin": "https://any.example.com"})
c("cors_v1_get", method="GET", path="/v1/models", server_args=CORS, headers={"Origin": "https://chat.example.com"})
c("cors_not_on_health", method="GET", path="/health", server_args=CORS, headers={"Origin": "https://chat.example.com"})
c("cors_post_listed_origin", method="POST", path="/v1/chat/completions", server_args=CORS, body=CHAT,
  headers={"Origin": "https://chat.example.com"})

# payload
def post_raw(name, raw_body, headers, **kw):
    c(name, method="POST", path="/v1/chat/completions", raw_body=raw_body, headers=headers, **kw)


JSON_CT = {"Content-Type": "application/json"}
post_raw("cl_invalid", "{}", {**JSON_CT, "Content-Length": "abc"})
post_raw("cl_negative", "{}", {**JSON_CT, "Content-Length": "-1"})
post_raw("cl_over_limit", "{}", {**JSON_CT, "Content-Length": "2097152"}, server_args=["--max-payload-mib", "1"])
post_raw("body_over_limit", '{"x": "{repeat:1100000:a}"}', JSON_CT, server_args=["--max-payload-mib", "1"])
CH = {**JSON_CT, "Content-Length": None, "Transfer-Encoding": "chunked"}
body = json.dumps(CHAT)
post_raw("chunked_valid", f"{len(body):x}\r\n{body}\r\n0\r\n\r\n", CH)
post_raw("chunked_two_chunks_ext", f"5;ext=1\r\n{body[:5]}\r\n{len(body) - 5:x}\r\n{body[5:]}\r\n0\r\nX-T: 1\r\n\r\n", CH)
post_raw("chunked_bad_size", "zz\r\n{}\r\n0\r\n\r\n", CH)
post_raw("chunked_negative_size", "-2\r\n{}\r\n0\r\n\r\n", CH)
post_raw("chunked_oversized", "200000\r\nabc", CH, server_args=["--max-payload-mib", "1"])
post_raw("chunked_truncated", "10\r\n{}", CH, half_close=True)
post_raw("chunked_no_crlf_after_data", "2\r\n{}XX0\r\n\r\n", CH)
post_raw("chunked_wins_over_cl", f"{len(body):x}\r\n{body}\r\n0\r\n\r\n", {**CH, "Content-Length": "3"})
post_raw("invalid_json", "{not json", JSON_CT)
post_raw("json_array", "[1, 2]", JSON_CT)
post_raw("json_string", '"hi"', JSON_CT)
post_raw("empty_body", "", JSON_CT)
c("no_body_no_length", method="POST", path="/v1/chat/completions", headers=JSON_CT)
post_raw("cl_short_body", "{}", {**JSON_CT, "Content-Length": "10"}, half_close=True)

# control routes
for p in ("/load", "/unload"):
    n = p.strip("/")
    c(f"{n}_json", method="POST", path=p, body={})
    c(f"{n}_no_ct", method="POST", path=p, raw_body="{}", headers={})
    c(f"{n}_foreign_origin", method="POST", path=p, body={}, headers={"Origin": "https://evil.com"})
    c(f"{n}_own_origin", method="POST", path=p, body={}, headers={"Origin": "http://127.0.0.1:{port}"})
c("load_body_over_64k", method="POST", path="/load", raw_body="{repeat:70000:a}", headers=JSON_CT)
c("load_bad_cl", method="POST", path="/load", raw_body="{}", headers={**JSON_CT, "Content-Length": "x"})
c("load_chunked", method="POST", path="/load", raw_body="2\r\n{}\r\n0\r\n\r\n", headers=CH)
c("load_chunked_over_64k", method="POST", path="/load", raw_body="20000\r\nabc", headers=CH)
c("load_incomplete", method="POST", path="/load", raw_body="{}", headers={**JSON_CT, "Content-Length": "100"},
  half_close=True)
for p in ("/v1/load", "/v1/unload"):
    n = p.strip("/").replace("/", "_")
    c(f"{n}_json", method="POST", path=p, body={})
    c(f"{n}_no_ct", method="POST", path=p, raw_body="{}", headers={})
    c(f"{n}_foreign_origin", method="POST", path=p, body={}, headers={"Origin": "https://evil.com"})
    c(f"{n}_unknown_model", method="POST", path=p, body={"model": "nope"})
c("unload_then_status", requests=[
    {"method": "POST", "path": "/v1/unload", "body": {}},
    {"method": "GET", "path": "/health"},
    {"method": "GET", "path": "/props"},
    {"method": "POST", "path": "/v1/load", "body": {}},
    {"method": "GET", "path": "/health"}])
c("responses_get_id", method="POST", path="/v1/responses/xyz", body={})
c("tokenize_good", method="POST", path="/tokenize", body={"content": "hello <|im_end|>"})
c("tokenize_no_special", method="POST", path="/tokenize", body={"content": "hi <|im_end|>", "parse_special": False})
c("tokenize_bad", method="POST", path="/tokenize", body={"content": 5})
c("tokenize_bad_parse_special", method="POST", path="/tokenize", body={"content": "x", "parse_special": "no"})
c("tokenize_unknown_model", method="POST", path="/tokenize", body={"content": "x", "model": "nope"})
c("detokenize_good", method="POST", path="/detokenize", body={"tokens": [104, 105]})
c("detokenize_bad", method="POST", path="/detokenize", body={"tokens": [1, -1]})
c("detokenize_bool", method="POST", path="/detokenize", body={"tokens": [True]})
c("detokenize_out_of_vocab", method="POST", path="/detokenize", body={"tokens": [99999999]})
c("answer_now_no_target", method="POST", path="/v1/answer_now", body={})
c("answer_now_unknown_id", method="POST", path="/v1/answer_now", body={"request_id": "chatcmpl-doesnotexist00"})
c("answer_now_bad_type", method="POST", path="/v1/answer_now", body={"request_id": 5})
c("answer_now_session_header", method="POST", path="/v1/answer_now", body={}, headers={"x-session-id": "s1"})
c("vram_bad_reserve", method="POST", path="/v1/vram", body={"reserve_mib": -1})
c("vram_bool_reserve", method="POST", path="/v1/vram", body={"reserve_mib": True})
c("vram_no_ct", method="POST", path="/v1/vram", raw_body="{}", headers={})
c("v1_unknown_post", method="POST", path="/v1/nope", body={"model": "basalt"})
c("post_unknown_model", method="POST", path="/v1/chat/completions", body={**CHAT, "model": "nope"})
c("post_messages_unknown_model", method="POST", path="/v1/messages",
  body={"model": "nope", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
c("post_query_string", method="POST", path="/v1/messages?beta=true",
  body={"model": "basalt", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})

# methods and malformed requests
c("head_root", method="HEAD", path="/")
c("put_root", method="PUT", path="/", body={})
c("delete_root", method="DELETE", path="/")
RAW = {
    "raw_get_only": "GET\r\n\r\n",
    "raw_extra_word": "GET / HTTP/1.1 extra\r\nHost: {host}\r\n\r\n",
    "raw_bad_version": "GET / HTTPX/1.0\r\nHost: {host}\r\n\r\n",
    "raw_version_2": "GET / HTTP/2.0\r\nHost: {host}\r\n\r\n",
    "raw_brew": "BREW / HTTP/1.0\r\nHost: {host}\r\n\r\n",
    "raw_http09": "GET /\r\n",
    "raw_header_no_colon": "GET /health HTTP/1.1\r\nHost: {host}\r\nnocolon\r\n\r\n",
    "raw_101_headers": "GET /health HTTP/1.1\r\nHost: {host}\r\n{repeat:101:X-A: b\r\n}\r\n",
    "raw_long_request_line": "GET /{repeat:70000:a} HTTP/1.1\r\nHost: {host}\r\n\r\n",
    "raw_long_header": "GET /health HTTP/1.1\r\nHost: {host}\r\nX-Long: {repeat:70000:b}\r\n\r\n",
    "raw_http10_no_host": "GET /health HTTP/1.0\r\n\r\n",
    "raw_lf_only": "GET /health HTTP/1.1\nHost: {host}\n\n",
    "raw_two_hosts": "GET /health HTTP/1.1\r\nHost: {host}\r\nHost: evil.com\r\n\r\n",
    "raw_absolute_uri": "GET http://127.0.0.1:{port}/health HTTP/1.1\r\nHost: {host}\r\n\r\n",
    "raw_empty": "",
}
for n, raw in RAW.items():
    c(n, raw=raw, **({"half_close": True} if n == "raw_empty" else {}))

# chat end to end with the mock engine
c("chat_nonstream", method="POST", path="/v1/chat/completions", body=CHAT)
c("chat_stream", method="POST", path="/v1/chat/completions", body={**CHAT, "stream": True}, stream=True)
c("chat_stream_usage", method="POST", path="/v1/chat/completions",
  body={**CHAT, "stream": True, "stream_options": {"include_usage": True}}, stream=True)
c("chat_scripted", method="POST", path="/v1/chat/completions", script=["Plain answer, no thinking."],
  body={**CHAT, "max_tokens": 20})
c("chat_stream_disconnect", method="POST", path="/v1/chat/completions", body={**CHAT, "stream": True},
  stream=True, disconnect_after_bytes=400, settle_s=1.5)
c("messages_stream", method="POST", path="/v1/messages", stream=True,
  body={"model": "basalt", "max_tokens": 100, "stream": True, "messages": [{"role": "user", "content": "hi"}]})
c("messages_nonstream", method="POST", path="/v1/messages",
  body={"model": "basalt", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]})
c("count_tokens", method="POST", path="/v1/messages/count_tokens",
  body={"model": "basalt", "messages": [{"role": "user", "content": "hi"}]})
c("responses_nonstream", method="POST", path="/v1/responses", body={"model": "basalt", "input": "hi"})
c("chat_then_status", requests=[
    {"method": "POST", "path": "/v1/chat/completions", "body": CHAT},
    {"method": "GET", "path": "/status"},
    {"method": "GET", "path": "/v1/status"},
    {"method": "GET", "path": "/metrics"},
    {"method": "GET", "path": "/api/logs?after=0&limit=3"}])
c("start_parallel2_mock", server_args=["--parallel", "2"])
c("concurrent_chats", requests=[
    {"method": "GET", "path": "/v1/status"},
    {"method": "GET", "path": "/slots"},
    {"parallel": True, "requests": [
        {"method": "POST", "path": "/v1/chat/completions", "body": CHAT},
        {"method": "POST", "path": "/v1/chat/completions", "body": CHAT}]},
    {"method": "GET", "path": "/slots"},
    {"method": "GET", "path": "/props"}])

for name, case in cases.items():
    (OUT / f"{name}.json").write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8", newline="\n")
print(len(cases), "cases")
