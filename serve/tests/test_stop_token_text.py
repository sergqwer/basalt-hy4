"""A stop token the model writes where no turn can end is written as its text and the reply goes on
(Service.stop_as_text, frontend.TextPlace, a parser's in_value; --stop-token-text). The replies end to end are the
parity cases serve-rs/parity/cases/stoptext/*."""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from serve import frontend as fe
from serve.engine.mock import MockEngine
from serve.models.hy4flash import Hy4Parser, Tags
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
E = "<|im_end|>"


class Place(unittest.TestCase):
    def ctx(self, text, kind="content", example=False):
        p = fe.TextPlace()
        p.feed(kind, text)
        return p.literal_context(kind, example)

    def test_places(self):
        self.assertEqual(self.ctx("Stop on `"), "after the backtick that opens inline code")
        self.assertEqual(self.ctx('Write "'), "after an opening quotation mark")
        self.assertEqual(self.ctx("Write “"), "after an opening quotation mark")
        self.assertEqual(self.ctx("The token is "), "after a space")
        self.assertEqual(self.ctx("I will write ", "reasoning"), "after a space")
        p = fe.TextPlace()
        p.feed("content", "The token is ")
        self.assertIsNone(p.literal_context("content", False, space=False))    # Hy4's template keeps whitespace

    def test_turn_ends(self):
        for text in ("The answer is 4.", "Use `x`", 'He said "yes"', "it's", "```", "line\n"):
            self.assertIsNone(self.ctx(text), text)

    def test_examples(self):
        p = fe.TextPlace()
        p.feed("content", "```\n<|im_start|>user\nhi")
        self.assertTrue(p.example_place("content"))
        self.assertEqual(p.literal_context("content", True), fe.EXAMPLE_TURN)
        self.assertIsNone(p.literal_context("content", False))
        p.feed("content", "\n```\n")
        self.assertFalse(p.example_place("content"))
        self.assertTrue(p.example_place("reasoning"))

    def test_held_newlines(self):
        p = fe.TextPlace()
        p.feed("content", "```")
        self.assertTrue(p.with_held("content", "\n").example_place("content"))
        self.assertFalse(p.example_place("content"))       # the copy only


class InValue(unittest.TestCase):
    def test_qwen(self):
        p = fe.OutputParser(thinking=False)
        p.feed("<tool_call>\n<function=run>\n<parameter=command>\ngrep '")
        self.assertTrue(p.in_value())
        p.feed("x'\n</parameter>\n")
        self.assertFalse(p.in_value())

    def test_hy4(self):
        t = Tags(":opensource")
        p = Hy4Parser(t, thinking=False, tools=[{"name": "run", "input_schema": {
            "type": "object", "properties": {"command": {"type": "string"}}}}])
        p.feed(t.calls + t.call + "run" + t.key + "command" + t.key_end + t.value + "grep '")
        self.assertTrue(p.in_value())
        p.feed("x'" + t.value_end)
        self.assertFalse(p.in_value())


class StopAsText(unittest.TestCase):
    def svc(self, script, on=True):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script), tok, fe.ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        svc.stop_token_text = on
        return svc

    def reply(self, svc, max_new=200):
        text, finish = "", None
        for kind, ev in svc.run(svc.tok.encode("hi"), False, None, max_new, {}, threading.Event()):
            if kind == "event" and ev.kind == "content":
                text += ev.text
            elif kind == "done":
                finish = ev["finish"]
        return text, finish

    def test_goes_on(self):
        self.assertEqual(self.reply(self.svc(["Stop on `" + E, ">` here."])), ("Stop on `<|im_end|>` here.", "stop"))

    def test_off(self):
        self.assertEqual(self.reply(self.svc(["Stop on `" + E, ">` here."], on=False)), ("Stop on `", "stop"))

    def test_real_end(self):
        self.assertEqual(self.reply(self.svc(["The answer is 4." + E + "never"])), ("The answer is 4.", "stop"))

    def test_cap(self):
        svc = self.svc(["a `" + E] + [">` `" + E] * 70)
        text, _ = self.reply(svc, max_new=4000)
        self.assertEqual(text.count("<|im_end|"), 64)                 # STOP_TEXT_MAX, then the id ends the reply


if __name__ == "__main__":
    unittest.main()
