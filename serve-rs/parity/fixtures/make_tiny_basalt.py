"""Write fixtures/tiny.basalt: a .basalt container with only what the server reads before it starts an engine (meta and
a byte-level tokenizer: the 256 byte tokens, a few merges and the chat specials). No weights: the fake engine needs none.

    python serve-rs/parity/fixtures/make_tiny_basalt.py [OUT]

About 12 MB, almost all section-alignment zeros, so it is generated, not committed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools"))
from model_file import ModelWriter  # noqa: E402

CONTROL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]                     # token type 3
USER = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]   # type 4
MERGES = [("h", "e"), ("l", "l"), ("Ġ", "t")]


def byte_tokens() -> list[str]:
    """GPT-2's bytes_to_unicode order: printable bytes map to themselves, the rest to 256+n."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    table = dict(zip(bs, cs))
    return [chr(table[b]) for b in range(256)]


def build(out: Path) -> Path:
    tokens = byte_tokens() + [a + b for a, b in MERGES] + CONTROL + USER
    types = [1] * (256 + len(MERGES)) + [3] * len(CONTROL) + [4] * len(USER)
    vocab = {t: i for i, t in enumerate(tokens)}
    meta = {"name": "tiny-fake", "gguf_kv": {"general.sampling.temp": 0.6}}
    tmp = out.with_suffix(".tmp")
    w = ModelWriter(str(tmp))
    w.add_bytes("meta", json.dumps(meta).encode())
    w.add_bytes("tokenizer/vocab.json", json.dumps(vocab, ensure_ascii=False).encode("utf-8"))
    w.add_bytes("tokenizer/merges.txt", "\n".join(f"{a} {b}" for a, b in MERGES).encode("utf-8"))
    w.add_bytes("tokenizer/token_type.json", json.dumps(types).encode())
    w.close()
    tmp.replace(out)
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "tiny.basalt"))
