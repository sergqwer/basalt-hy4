"""Writes the stop-token-as-text parity cases (cases/stoptext/*.json). Run: python _gen.py (from this directory).
The mock speaks the Qwen dialect with the byte tokenizer: "<|im_end|>" in a script is the stop id itself. A script
list is one script per engine pass, so a reply that goes on after the token's text reads its next script; the last
script ends with the mock's own stop id (a real end of turn)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
E = "<|im_end|>"
A = "Thinking.</think>\n\n"                 # the answer starts after the thinking
TOOLS = [{"type": "function", "function": {"name": "run", "description": "Run a command",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}}, "required": ["command"]}}}]
ATOOLS = [{"name": "run", "description": "Run a command", "input_schema": TOOLS[0]["function"]["parameters"]}]
U = [{"role": "user", "content": "hi"}]
CASES = {}


def case(name, script, api="chat", stream=False, args=None, **body):
    if api == "chat":
        path, b = "/v1/chat/completions", {"model": "basalt", "messages": U, **body}
    else:
        path, b = "/v1/messages", {"model": "basalt", "max_tokens": 200, "messages": U, **body}
    if stream:
        b["stream"] = True
    c = {"name": name, "method": "POST", "path": path, "body": b, "script": script}
    if stream:
        c["stream"] = True
    if args:
        c["server_args"] = args
    CASES[name] = c


for api in ("chat", "msg"):
    for stream in (False, True):
        sfx = f"{api}{'_stream' if stream else ''}"
        case(f"backtick_{sfx}", [A + "Stop on `" + E, ">` here."], api, stream)
        case(f"tool_arg_{sfx}", [A + "<tool_call>\n<function=run>\n<parameter=command>\ngrep -rn '" + E,
                                 ">' .\n</parameter>\n</function>\n</tool_call>"], api, stream,
             tools=TOOLS if api == "chat" else ATOOLS)
case("quote_chat", [A + 'Write "' + E, '>" there.'])
case("quote_typographic_chat", [A + "Write “" + E, ">” there."])
case("space_chat", [A + "The token is " + E, "> and more."])
case("space_reasoning_chat", ["I will write " + E, "> now.</think>\n\nDone."])
case("example_turn_chat", [A + "```\n<|im_start|>user\nhi" + E, ">\n```\nThat is the format."])
case("owed_then_end_chat", [A + "Stop on `" + E, E])                     # the model ends right after the text
case("two_literals_chat", [A + "Use `" + E, ">` or `" + E, ">`."])
case("real_end_chat", [A + "The answer is 4." + E + "never seen"])     # a stop where a turn ends still ends it
case("real_end_closed_backtick_chat", [A + "Use `x`" + E + "never seen"])
case("real_end_closing_quote_chat", [A + 'He said "yes"' + E + "never seen"])
case("off_backtick_chat", [A + "Stop on `" + E, ">` here."], args=["--no-stop-token-text"])
case("no_room_chat", ["Stop on `" + E, ">` here."], max_tokens=15,        # 9 tokens, then no room for the text
     chat_template_kwargs={"enable_thinking": False})
case("client_stop_chat", [A + "Stop on `" + E, ">` here."], stop=["<|im_end"])

for name, c in CASES.items():
    (HERE / f"{name}.json").write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(len(CASES), "cases")
