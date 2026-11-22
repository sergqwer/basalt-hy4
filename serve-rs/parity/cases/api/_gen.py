"""Writes the API parity cases (cases/api/*.json): every endpoint's options and error paths, on the mock engine.
Run: python _gen.py  (from this directory). The mock speaks the Qwen dialect with the byte tokenizer."""
import json
from pathlib import Path

HERE = Path(__file__).parent
CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nKyiv\n</parameter>\n<parameter=days>\n3\n"
        "</parameter>\n</function>\n</tool_call>")
THINK = "Let me think about it.</think>\n\nThe answer is 4."
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                           "parameters": {"type": "object", "properties": {
                                               "city": {"type": "string"}, "days": {"type": "integer"}},
                                               "required": ["city"]}}}]
ATOOLS = [{"name": "get_weather", "description": "Weather", "input_schema": TOOLS[0]["function"]["parameters"]}]
RTOOLS = [{"type": "function", "name": "get_weather", "parameters": TOOLS[0]["function"]["parameters"]}]
U = [{"role": "user", "content": "hi"}]
CASES = {}


def case(name, path, body, script=None, stream=False, **extra):
    c = {"name": name, "method": "POST", "path": path, "body": body}
    if script is not None:
        c["script"] = [script] if isinstance(script, str) else script
    if stream:
        c["stream"] = True
    c.update(extra)
    CASES[name] = c


def chat(name, script=None, stream=False, **body):
    b = {"model": "basalt", "messages": U, **body}
    if stream:
        b["stream"] = True
    case("chat_" + name, "/v1/chat/completions", b, script, stream)


def msg(name, script=None, stream=False, **body):
    b = {"model": "basalt", "max_tokens": 64, "messages": U, **body}
    if stream:
        b["stream"] = True
    case("msg_" + name, "/v1/messages", b, script, stream)


def resp(name, script=None, stream=False, **body):
    b = {"model": "basalt", "input": "hi", **body}
    if stream:
        b["stream"] = True
    case("resp_" + name, "/v1/responses", b, script, stream)


for s in (False, True):
    t = "_stream" if s else ""
    chat("think" + t, THINK, s)
    chat("tool" + t, "Checking.</think>\n\n" + CALL, s, tools=TOOLS)
    chat("two_tools" + t, "x</think>\n\nFirst " + CALL + "\n" + CALL, s, tools=TOOLS)
    chat("cut_tool" + t, "x</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nKy", s, tools=TOOLS)
    chat("nothink" + t, "Direct.", s, reasoning_effort="none")
    chat("length" + t, THINK, s, max_tokens=5)
    chat("stop" + t, THINK, s, stop=["is 4"])
    chat("stop_list" + t, THINK, s, stop=["zzz", "answer"])
    chat("logprobs" + t, THINK, s, logprobs=True, top_logprobs=2)
    chat("progress" + t, THINK, s, return_progress=True)
    chat("forced" + t, "Sure.</think>\n\n" + CALL, s, tools=TOOLS, tool_choice="required")
    chat("named_tool" + t, CALL, s, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "get_weather"}},
         reasoning_effort="none")
    chat("json_object" + t, "x</think>\n\n{\"a\": 1}", s, response_format={"type": "json_object"})
    chat("json_schema" + t, "x</think>\n\n{\"a\": 1}", s, response_format={
        "type": "json_schema", "json_schema": {"name": "n", "strict": True, "schema": {"type": "object"}}})
    chat("budget" + t, "a b c d e f g h i j k l m n o p</think>\n\nok", s, reasoning_budget_tokens=4, max_tokens=200)
    chat("stream_usage" + t, THINK, s, stream_options={"include_usage": True})
    msg("think" + t, THINK, s)
    msg("thinking_on" + t, THINK, s, thinking={"type": "enabled", "budget_tokens": 1024}, max_tokens=2048)
    msg("thinking_off" + t, "Direct.", s, thinking={"type": "disabled"})
    msg("tool" + t, "Checking.</think>\n\n" + CALL, s, tools=ATOOLS)
    msg("stop_seq" + t, THINK, s, stop_sequences=["is 4"])
    msg("length" + t, THINK, s, max_tokens=5)
    msg("effort_low" + t, THINK, s, output_config={"effort": "low"})
    msg("tool_any" + t, "x</think>\n\n" + CALL, s, tools=ATOOLS, tool_choice={"type": "any"})
    msg("format" + t, "x</think>\n\n{\"a\": 1}", s,
        output_config={"format": {"type": "json_schema", "schema": {"type": "object"}}})
    resp("think" + t, THINK, s)
    resp("tool" + t, "Checking.</think>\n\n" + CALL, s, tools=RTOOLS)
    resp("length" + t, THINK, s, max_output_tokens=5)
    resp("encrypted" + t, THINK, s, include=["reasoning.encrypted_content"])
    resp("logprobs" + t, THINK, s, include=["message.output_text.logprobs"], top_logprobs=1)
    resp("json" + t, "x</think>\n\n{\"a\": 1}", s, text={"format": {"type": "json_object"}})
    resp("custom_tool" + t, "x</think>\n\n<tool_call>\n<function=patch>\n<parameter=input>\nhello\n</parameter>\n"
         "</function>\n</tool_call>", s, tools=[{"type": "custom", "name": "patch"}])

# conversation shapes
chat("multi_turn", THINK, messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2"},
                                     {"role": "assistant", "content": "4", "reasoning_content": "add"},
                                     {"role": "user", "content": "3+3"}])
chat("tool_result", "Sunny.", tools=TOOLS, messages=[
    {"role": "user", "content": "weather"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "get_weather", "arguments": "{\"city\": \"Kyiv\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"}])
chat("literal_tags", "Fine.", messages=[{"role": "user", "content": "What do <think> and <tool_call> mean?"}])
chat("parts", "Fine.", messages=[{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
chat("unicode", "Добре ✓ 😀", messages=[{"role": "user", "content": "Ünïcødé 中文"}])
chat("document", "Read.", messages=[{"role": "user", "content": [
    {"type": "file", "file": {"filename": "a.txt", "file_data": "data:text/plain;base64,aGVsbG8gd29ybGQ="}},
    {"type": "text", "text": "summarize"}]}])
chat("sampling", "ok", temperature=0.5, top_p=0.9, top_k=20, min_p=0.05, presence_penalty=0.1, seed=7,
     repeat_penalty=1.1)
chat("logit_bias", "ok", logit_bias={"65": 5, "66": -100})
chat("max_completion", THINK, max_completion_tokens=7)
chat("fit_too_long", THINK, max_tokens=10 ** 6)
chat("preserve", "ok", chat_template_kwargs={"preserve_reasoning": True})
msg("tool_result", "Rain.", tools=ATOOLS, messages=[
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Oslo"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "rain"}]}])
msg("system_blocks", "ok", system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
msg("document", "ok", messages=[{"role": "user", "content": [
    {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x\r\ny"}, "title": "t.txt",
     "context": "ctx"}, {"type": "text", "text": "go"}]}])
resp("conversation", "ok", instructions="Terse.", input=[
    {"role": "developer", "content": "dev"}, {"role": "user", "content": "weather"},
    {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "call"}]},
    {"type": "function_call", "call_id": "a", "name": "get_weather", "arguments": "{\"city\": \"Rome\"}"},
    {"type": "function_call_output", "call_id": "a", "output": "20C"}], tools=RTOOLS)
resp("namespace", "ok", tools=[{"type": "namespace", "name": "fs", "description": "files",
                                "tools": [{"name": "read", "parameters": {"type": "object", "properties": {}}}]},
                               {"type": "web_search"}])
resp("effort", THINK, reasoning={"effort": "low", "summary": "auto"}, temperature=0.3, metadata={"k": "v"})

# errors
chat("err_no_messages", messages=[])
chat("err_messages_string", messages="hi")
chat("err_bad_role", messages=[{"role": "wizard", "content": "x"}])
chat("err_effort", reasoning_effort="extreme")
chat("err_tools", tools="nope")
chat("err_tool_choice", tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}})
chat("err_logprobs", logprobs="yes")
chat("err_top_logprobs", logprobs=True, top_logprobs=99)
chat("err_top_without", top_logprobs=2)
chat("err_progress", return_progress="yes")
chat("err_logit_bias_key", logit_bias={"x": 1})
chat("err_logit_bias_val", logit_bias={"1": 500})
chat("err_temperature", temperature="hot")
chat("err_stop", stop=5)
chat("err_budget", reasoning_budget_tokens="many")
chat("err_format", response_format={"type": "xml"})
chat("err_schema_ref", response_format={"type": "json_schema", "json_schema": {"name": "n", "schema": {"$ref": "http://x"}}})
chat("err_image", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}])
chat("err_document_b64", messages=[{"role": "user", "content": [{"type": "file", "file": {"file_data": "data:text/plain;base64,!!!"}}]}])
chat("err_document_id", messages=[{"role": "user", "content": [{"type": "file", "file": {"file_id": "f1"}}]}])
chat("err_document_url", messages=[{"role": "user", "content": [{"type": "file", "file": {"file_url": "http://example.com/a.pdf"}}]}])
chat("err_max_tokens", max_tokens="ten")
msg("err_no_max", max_tokens=None)
msg("err_messages", messages="x")
msg("err_thinking", thinking={"type": "maybe"})
msg("err_tool_choice", tools=ATOOLS, tool_choice={"type": "tool", "name": "nope"})
msg("err_format", output_config={"format": {"type": "xml"}})
resp("err_input_missing", input=None)
resp("err_input_type", input=5)
resp("err_item_ref", input=[{"type": "item_reference", "id": "x"}])
resp("err_role", input=[{"type": "message", "role": "wizard", "content": "x"}])
resp("err_previous", previous_response_id="resp_1")
resp("err_background", background=True)
resp("err_max_output", max_output_tokens=0)
resp("err_reasoning", reasoning="high")
resp("err_text_format", text={"format": {"type": "xml"}})
resp("err_top_logprobs", top_logprobs=50)
resp("err_tools", tools="x")
resp("err_tool_name", tools=[{"type": "function"}])
resp("err_part", input=[{"role": "user", "content": [{"type": "input_audio"}]}])
resp("err_encrypted", input=[{"type": "reasoning", "encrypted_content": "basalt.r1:!!!"}, {"role": "user", "content": "x"}])
resp("err_effort", reasoning={"effort": "extreme"})

# count_tokens, tokenize, detokenize
case("count_tokens_tools", "/v1/messages/count_tokens", {"model": "basalt", "messages": U, "tools": ATOOLS})
case("count_tokens_system", "/v1/messages/count_tokens", {"model": "basalt", "system": "sys", "messages": U})
case("count_tokens_bad", "/v1/messages/count_tokens", {"model": "basalt", "messages": "x"})
case("tokenize_unicode", "/tokenize", {"content": "héllo 😀 <|im_end|>"})
case("tokenize_with_pieces", "/tokenize", {"content": "ab", "with_pieces": True})
case("tokenize_plain_special", "/tokenize", {"content": "<|im_end|>", "parse_special": False})
case("detokenize_unicode", "/detokenize", {"tokens": [104, 195, 169, 240, 159, 152, 128]})
case("detokenize_partial", "/detokenize", {"tokens": [240, 159]})

# server options that change the API layer
LONG = " ".join(["w"] * 60) + "</think>\n\nDone."
LOOP = "a" * 40 + "</think>\n\nok"
def opt(name, args, path="/v1/chat/completions", body=None, script=None, stream=False):
    b = body or {"model": "basalt", "messages": U}
    if stream:
        b = {**b, "stream": True}
    case("opt_" + name, path, b, script, stream, server_args=args)
opt("fit_max_tokens", ["--fit-max-tokens"], body={"model": "basalt", "messages": U, "max_tokens": 10 ** 6}, script=THINK)
opt("no_grammar_json", ["--no-grammar"], body={"model": "basalt", "messages": U, "response_format": {"type": "json_object"}})
opt("no_grammar_strict", ["--no-grammar"], body={"model": "basalt", "messages": U, "response_format": {
    "type": "json_schema", "json_schema": {"name": "n", "strict": True, "schema": {"type": "object"}}}}, script="x</think>\n\n{}")
opt("budget_default", ["--reasoning-budget-tokens", "8"], script=LONG)
opt("budget_default_stream", ["--reasoning-budget-tokens", "8"], script=LONG, stream=True)
opt("budget_request_off", ["--reasoning-budget-tokens", "8"], body={"model": "basalt", "messages": U, "reasoning_budget_tokens": 0}, script=LONG)
opt("answer_room", ["--answer-room", "20"], body={"model": "basalt", "messages": U, "max_tokens": 40}, script=LONG)
opt("answer_room_auto", ["--answer-room", "auto"], body={"model": "basalt", "messages": U, "max_tokens": 40}, script=LONG)
opt("effort_default", ["--reasoning-effort", "low"], script=THINK)
opt("effort_none", ["--reasoning-effort", "none"], script="Direct.")
opt("thinking_loop", ["--thinking-loop-tokens", "8"], script=LOOP)
opt("thinking_loop_stream", ["--thinking-loop-tokens", "8"], script=LOOP, stream=True)
opt("alias", ["--alias", "other"], body={"model": "other", "messages": U}, script=THINK)
opt("image_no_vision_resp", [], path="/v1/responses", body={"model": "basalt", "input": [{"role": "user", "content": [
    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}]}]})
opt("image_no_vision_msg", [], path="/v1/messages", body={"model": "basalt", "max_tokens": 5, "messages": [{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}]}]})

for name, c in CASES.items():
    (HERE / f"{name}.json").write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(len(CASES), "cases")
