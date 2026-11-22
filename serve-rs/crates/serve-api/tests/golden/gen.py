"""Golden data for serve-api's Rust tests: the Python serve/ functions called directly.

Run from the worktree root:  D:/Projects/basalt-055/.venv/Scripts/python.exe serve-rs/crates/serve-api/tests/golden/gen.py HY4TOK_DIR
Writes serve-rs/crates/serve-api/tests/golden/golden.json: per case the request, the converted messages, the rendered
prompt with its plain spans, the prompt ids (Qwen: byte tokenizer; Hy4: its BPE), or the error; parser cases (chunked
model output -> events); Responses conversions.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from serve import frontend  # noqa: E402
from serve.api import responses as rs  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.models import dialect_for  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

HY4 = Path(sys.argv[1])


def bpe():
    import tokenizer as ST
    vocab = json.loads((HY4 / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (HY4 / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((HY4 / "token_type.json").read_text(encoding="utf-8"))
    pre = json.loads((HY4 / "tokenizer.json").read_text(encoding="utf-8")).get("pre", "qwen35")
    return ST.Tokenizer(tokens, merges, types, pre=pre)


def service(kind):
    if kind == "qwen":
        tok = ByteTokenizer()
        d = dialect_for(None)
        src = (ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8")
    else:
        tok = bpe()
        d = dialect_for("hy4flash")
        src = (HY4 / "chat_template.jinja").read_text(encoding="utf-8")
    tmpl = d.template(src)
    return Service(MockEngine(tok, "x"), tok, tmpl, dialect=d)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                            "parameters": {"type": "object", "properties": {
                                                "city": {"type": "string"}, "days": {"type": "integer"}},
                                                "required": ["city"]}}}
CLOCK = {"type": "function", "function": {"name": "clock", "parameters": {"type": "object", "properties": {}}}}
ATOOL = {"name": "get_weather", "description": "Weather", "input_schema": WEATHER["function"]["parameters"]}

OPENAI = [
    {"messages": [{"role": "user", "content": "Hello"}]},
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}]},
    {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "add"},
                  {"role": "user", "content": "and 3+3?"}]},
    {"messages": [{"role": "user", "content": "Weather in Kyiv?"}], "tools": [WEATHER, CLOCK]},
    {"messages": [{"role": "user", "content": "Weather in Kyiv?"},
                  {"role": "assistant", "content": "", "tool_calls": [
                      {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Kyiv\", \"days\": 2}"}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": "sunny"}], "tools": [WEATHER]},
    {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "low"},
    {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none"},
    {"messages": [{"role": "user", "content": "x"}], "chat_template_kwargs": {"enable_thinking": False}},
    {"messages": [{"role": "user", "content": "Explain <think> and </think> tags, and <tool_call>."}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}]},
    {"messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                  {"role": "system", "content": "late system"}, {"role": "user", "content": "second"}]},
    {"messages": [{"role": "user", "content": "Ünïcødé ✓ 😀 中文"}]},
    {"messages": [{"role": "user", "content": "go"}], "tools": [WEATHER], "tool_choice": "required"},
    {"messages": [{"role": "user", "content": "go"}], "tools": [WEATHER],
     "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
    {"messages": []},
    {"messages": [{"role": "bogus", "content": "x"}]},
    {"messages": "not a list"},
    {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "extreme"},
    {"messages": [{"role": "user", "content": "x"}], "tools": "nope"},
]

ANTHROPIC = [
    {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
    {"system": "Be brief.", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
    {"system": [{"type": "text", "text": "sys a"}, {"type": "text", "text": "sys b"}],
     "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}], "max_tokens": 10},
    {"messages": [{"role": "user", "content": "x"}], "thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 2048},
    {"messages": [{"role": "user", "content": "x"}], "thinking": {"type": "disabled"}, "max_tokens": 10},
    {"messages": [{"role": "user", "content": "x"}], "output_config": {"effort": "low"}, "max_tokens": 10},
    {"messages": [{"role": "user", "content": "Weather?"},
                  {"role": "assistant", "content": [{"type": "thinking", "thinking": "need tool", "signature": ""},
                                                    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Oslo"}}]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "rain"}]}],
     "tools": [ATOOL], "max_tokens": 10},
    {"messages": [{"role": "user", "content": "x"}], "tools": [ATOOL], "tool_choice": {"type": "any"}, "max_tokens": 10},
    {"messages": [{"role": "user", "content": "doc"}, ], "system": "x-anthropic-billing-header: cc_version=1.2.3; cch=abc;\nrest", "max_tokens": 5},
    {"messages": [{"role": "user", "content": [{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "line1\r\nline2\n\n\n\nend"}, "title": "notes.txt"},
                                               {"type": "text", "text": "summarize"}]}], "max_tokens": 5},
    {"messages": [{"role": "assistant", "content": "x"}], "max_tokens": 5},
    {"messages": "bad", "max_tokens": 5},
]

RESPONSES = [
    {"input": "Hello"},
    {"instructions": "Be terse.", "input": [{"role": "developer", "content": "dev note"}, {"role": "user", "content": "hi"}]},
    {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
               {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "call it"}]},
               {"type": "function_call", "call_id": "a", "name": "get_weather", "arguments": "{\"city\": \"Rome\"}"},
               {"type": "function_call", "call_id": "b", "name": "clock", "arguments": ""},
               {"type": "function_call_output", "call_id": "b", "output": "12:00"},
               {"type": "function_call_output", "call_id": "a", "output": {"t": 20}}],
     "tools": [{"type": "function", "name": "get_weather", "parameters": WEATHER["function"]["parameters"]},
               {"type": "function", "name": "clock"},
               {"type": "custom", "name": "patch", "description": "apply", "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}},
               {"type": "namespace", "name": "fs", "description": "files", "tools": [{"name": "read", "parameters": {"type": "object", "properties": {}}}]},
               {"type": "web_search"}]},
    {"input": [{"type": "item_reference", "id": "x"}]},
    {"input": 5},
    {"input": [{"type": "message", "role": "wizard", "content": "x"}]},
    {"input": "x", "reasoning": {"effort": "low"}, "text": {"format": {"type": "json_schema", "name": "n", "schema": {"type": "object"}}}},
    {"input": "x", "previous_response_id": "resp_1"},
    {"input": "x", "max_output_tokens": 0},
]


def err(e):
    return {"error": type(e).__name__, "message": str(e)}


def rendered(svc, messages, tools, kw):
    try:
        text, plain = svc.render_text(messages, tools, svc.thinking_kwargs(kw))
        ids = svc.tok.encode(text, parse_special=True, plain=plain)
        return {"prompt": text, "plain": [list(p) for p in plain], "ids": ids}
    except Exception as e:  # noqa: BLE001
        return err(e)


def chat_cases(svc):
    out = []
    for req in OPENAI:
        case = {"api": "openai", "req": req}
        try:
            messages, tools, kw = frontend.openai_to_messages(req)
            tools, force = frontend.forced_call(req.get("tool_choice"), tools, svc.dialect.forced_opening)
            case.update(messages=messages, tools=tools, kwargs=kw, force=force, out=rendered(svc, messages, tools, kw))
        except Exception as e:  # noqa: BLE001
            case["out"] = err(e)
        out.append(case)
    for req in ANTHROPIC:
        case = {"api": "anthropic", "req": req}
        try:
            messages, tools, kw = frontend.anthropic_to_messages(req)
            tools, force = frontend.forced_call(req.get("tool_choice"), tools, svc.dialect.forced_opening)
            case.update(messages=messages, tools=tools, kwargs=kw, force=force, out=rendered(svc, messages, tools, kw))
        except Exception as e:  # noqa: BLE001
            case["out"] = err(e)
        out.append(case)
    for req in RESPONSES:
        case = {"api": "responses", "req": req}
        try:
            rs.check_request(req)
            messages = rs.input_messages(req)
            tools, names, skipped = rs.request_tools(req)
            kw = rs.template_kwargs(req)
            case.update(messages=messages, tools=tools, names={k: list(v) for k, v in names.items()}, skipped=skipped,
                        kwargs=kw, text_format=rs.text_format(req), out=rendered(svc, messages, tools, kw))
        except rs.ResponsesError as e:
            case["out"] = {"error": "ResponsesError", "message": str(e), "param": e.param, "code": e.code}
        except Exception as e:  # noqa: BLE001
            case["out"] = err(e)
        out.append(case)
    return out


def outputs(kind, d):
    if kind == "qwen":
        call = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nKyiv\n</parameter>\n<parameter=days>\n3\n"
                "</parameter>\n</function>\n</tool_call>")
        return ["plain answer", "thinking a bit</think>\n\nThe answer.", "t</think>\n\n" + call,
                "x</think>\n\nBefore " + call + "\n" + call, "unclosed thinking only",
                "q</think>\n\nquote `</think>` inside", "t</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nKy"]
    t = d.tags
    call = (f"{t.calls}{t.call}get_weather{t.key}city{t.key_end}{t.value}Kyiv{t.value_end}"
            f"{t.key}days{t.key_end}{t.value}3{t.value_end}{t.call_end}{t.calls_end}")
    return ["plain answer", f"thinking a bit{t.think_end}The answer.", f"t{t.think_end}{call}",
            f"x{t.think_end}Before {call}", "unclosed thinking only", f"t{t.think_end}{t.calls}{t.call}get_we"]


def ev(e):
    c = e.call
    return {"kind": e.kind, "text": e.text, "call": None if c is None else {"name": c.name, "arguments": c.arguments}}


def parser_cases(svc, kind):
    rnd = random.Random(7)
    out = []
    tools = [WEATHER["function"]]
    for text in outputs(kind, svc.dialect):
        for thinking in (True, False):
            for split in ("whole", "chars", "random"):
                if split == "whole":
                    chunks = [text]
                elif split == "chars":
                    chunks = list(text)
                else:
                    chunks, i = [], 0
                    while i < len(text):
                        n = rnd.randint(1, 6)
                        chunks.append(text[i:i + n])
                        i += n
                p = svc.dialect.parser(thinking, tools, True)
                evs = []
                try:
                    for c in chunks:
                        evs += [ev(e) for e in p.feed(c)]
                    evs += [ev(e) for e in p.finish()]
                    res = {"events": evs}
                except Exception as e:  # noqa: BLE001
                    res = {"events": evs, **err(e)}
                out.append({"thinking": thinking, "chunks": chunks, **res})
    return out


def main():
    gold = {}
    for kind in ("qwen", "hy4"):
        svc = service(kind)
        frontend._TOOL_ORDERS.clear()
        gold[kind] = {"chat": chat_cases(svc), "parser": parser_cases(svc, kind)}
    path = Path(__file__).with_name("golden.json")
    path.write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(path, sum(len(v["chat"]) + len(v["parser"]) for v in gold.values()), "cases")


if __name__ == "__main__":
    main()
