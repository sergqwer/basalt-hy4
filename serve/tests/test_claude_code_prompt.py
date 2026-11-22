"""Claude Code's system prompt rewritten for the local model (frontend.local_claude_code_system,
--claude-code-prompt)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from serve import frontend as fe

ROOT = Path(__file__).resolve().parents[2]
SDK = json.loads((ROOT / "serve-rs/parity/fixtures/claude_code_2.1.318_system.json").read_text(encoding="utf-8"))
HEADER = SDK[0]["text"]


def system_of(blocks, mode="local"):
    old, fe.CLAUDE_CODE_PROMPT = fe.CLAUDE_CODE_PROMPT, mode
    try:
        return fe.anthropic_to_messages({"system": blocks, "messages": [{"role": "user", "content": "hi"}]})[0][0]["content"]
    finally:
        fe.CLAUDE_CODE_PROMPT = old


class LocalClaudeCodePrompt(unittest.TestCase):
    def test_claude_only_parts_go(self):
        s = system_of(SDK)
        self.assertTrue(s.startswith(fe.LOCAL_AGENT_INTRO + "\n\n# Harness\n"))
        for gone in ("billing-header", "Claude agent", "security testing", "pronoun", "Claude 5 family", "Fast mode",
                     "# Environment", "You are an agent working", "```markdown"):
            self.assertNotIn(gone, s)
        for kept in ("# Harness\n - Text you output", "For actions that are hard to reverse", "# Session-specific",
                     "# Context management\nWhen the conversation grows long", "If you are weighing a choice"):
            self.assertIn(kept, s)
        self.assertIn("# Memory\nPersistent memory lives in `C:\\work\\memory\\`: one markdown file", s)

    def test_stable_and_shorter(self):
        # a pure function of the text; the per-session version tail and per-request stamp are gone with the header
        other = [dict(SDK[0], text=HEADER.replace("2.1.318.02d", "2.1.318.9c1")), SDK[1], SDK[2]]
        self.assertEqual(system_of(SDK), system_of(other))
        self.assertLess(len(system_of(SDK)), 0.6 * len(fe._text_of(SDK)))

    def test_as_sent_only_pins_the_stamp(self):
        s = system_of(SDK, "as_sent")
        self.assertEqual(s, fe.pin_billing_stamp(fe._text_of(SDK)))
        self.assertIn("security testing", s)

    def test_cli_identity_and_environment(self):
        env = ("\n# Environment\nYou have been invoked in the following environment:\n - Primary working directory: D:\\x\n"
               " - The most recent Claude models are the Claude 5 family.\n\n# Plan mode\nDo not edit files yet.")
        s = system_of(HEADER + "You are Claude Code, Anthropic's official CLI for Claude.\n" + env)
        self.assertNotIn("official CLI", s)
        self.assertIn("# Environment\nYou have been invoked in the following environment:\n - Primary working "
                      "directory: D:\\x", s)
        self.assertNotIn("Claude 5 family", s)
        self.assertTrue(s.endswith("# Plan mode\nDo not edit files yet."))   # a section it does not know passes

    def test_other_prompts_untouched(self):
        self.assertEqual(system_of("You are a helpful assistant."), "You are a helpful assistant.")
        other = HEADER + "Some other agent's prompt."             # a header without Claude Code's identity
        self.assertEqual(system_of(other), fe.pin_billing_stamp(other))
        word = HEADER + "You are Claude Codex."                   # the identity must end at a word boundary
        self.assertEqual(system_of(word), fe.pin_billing_stamp(word))

    def test_identity_alone(self):
        self.assertEqual(system_of(SDK[:2]), fe.LOCAL_AGENT_INTRO)


if __name__ == "__main__":
    unittest.main()
