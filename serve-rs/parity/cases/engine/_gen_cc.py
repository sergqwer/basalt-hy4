"""Writes the --claude-code-prompt parity cases (cases/engine/cc_*.json). Run: python _gen_cc.py (from this directory).
The fake engine records the prompt ids it is sent, so A and B must rewrite Claude Code's system text identically.
fixtures/claude_code_2.1.318_system.json: the system blocks of a real `claude -p` request (2.1.318, memory path
neutralized)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
SDK = json.loads((HERE.parents[1] / "fixtures" / "claude_code_2.1.318_system.json").read_text(encoding="utf-8"))
HEADER = SDK[0]["text"]
CLI = HEADER + ("You are Claude Code, Anthropic's official CLI for Claude.\n"
                "\n# Environment\nYou have been invoked in the following environment:\n - Primary working directory: "
                "D:\\x\n - The most recent Claude models are the Claude 5 family.\n\n# Plan mode\nDo not edit files yet.")
U = [{"role": "user", "content": "hi"}]


def case(name, system, args=None):
    c = {"name": name, "mock": False, "method": "POST", "path": "/v1/messages",
         "body": {"model": "tiny-fake", "max_tokens": 10, "system": system, "messages": U}}
    if args:
        c["server_args"] = args
    (HERE / f"{name}.json").write_text(json.dumps(c, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


case("cc_prompt_local", SDK)
case("cc_prompt_as_sent", CLI, ["--claude-code-prompt", "as_sent"])    # the SDK prompt as sent is over tiny's 4K
case("cc_prompt_cli_identity", CLI)
case("cc_prompt_header_only", HEADER + "Some other agent's prompt.")      # a header without Claude Code's identity
case("cc_prompt_plain", "You are a helpful assistant.")
case("cc_prompt_identity_only", HEADER + "You are a Claude agent, built on Anthropic's Claude Agent SDK.")
case("cc_prompt_no_memory_path", [SDK[0], SDK[1], {"type": "text", "text": "\n# Memory\nNo path here.\n\n# Next\nx"}])
(HERE / "cc_prompt_bad_value.json").write_text(json.dumps(
    {"name": "cc_prompt_bad_value", "mock": False, "server_args": ["--claude-code-prompt", "bogus"]}) + "\n",
    encoding="utf-8")
