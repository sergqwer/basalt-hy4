"""make_nll_base.py OUT.kld [--n-ctx 2048] [--per-bucket 5] - a llama.cpp-format KL base over our held-out samples, for
teacher-forced NLL of the real next tokens (no BF16 reference exists for them: the logit rows are zero, so the gate's
KLD/top-1 columns are meaningless here and only `nll` is used).

Each eval sample is rendered with the fork's Python frontend (serve/frontend.py: openai_to_messages /
anthropic_to_messages + the pack's chat template, Hy4's kwargs mapping) and tokenized with the pack's tokenizer; a chunk
is the sample's LAST n_ctx tokens (so the scored half ends with the conversation's final answer). Short samples of one
bucket are joined (separate conversations back to back) to reach n_ctx. The file's logit rows are written sparse
(zeros). A manifest OUT.json lists chunk -> bucket, sample ids.
"""
import argparse, json, os, random, sys

sys.path.insert(0, r"D:\Projects\basalt-hy4-gptq")
from serve import frontend as FE  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

G = r"D:\Projects\hy4-data\gptq"
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n-ctx", type=int, default=2048)
ap.add_argument("--per-bucket", type=int, default=5)
ap.add_argument("--corpus", default=G + r"\corpus\eval.jsonl")
a = ap.parse_args()
tok = Tokenizer.from_file(r"C:\LLM_models\hy4-bf16\tokenizer.json")
tpl = FE.ChatTemplate.from_file(r"C:\LLM_models\hy4-bf16\chat_template.jinja")
V = 120832


def render(r):
    body = r["body"]
    if r["api"] == "anthropic":
        msgs, tools, kw = FE.anthropic_to_messages(body)
    else:
        msgs, tools, kw = FE.openai_to_messages(body)
    if kw.pop("enable_thinking", None) is False:
        kw["reasoning_effort"] = "no_think"
    kw.pop("reasoning_budget_tokens", None)
    kw = {k: v for k, v in kw.items() if k in ("reasoning_effort", "preserved_thinking")}
    if kw.get("reasoning_effort") not in (None, "no_think", "high"):   # the server maps every other level to high
        kw["reasoning_effort"] = "high"
    text = tpl.render(msgs, tools=tools, add_generation_prompt=False, **kw)
    return tok.encode(text, add_special_tokens=False).ids


rows = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
by = {}
for r in rows:
    by.setdefault(r["bucket"], []).append(r)
rng = random.Random(7)
chunks, man = [], []
for b in sorted(by):
    rs = by[b][:]
    rng.shuffle(rs)
    got, carry, carry_ids = 0, [], []
    for r in rs:
        if got >= a.per_bucket:
            break
        try:
            ids = render(r)
        except Exception as e:
            print("render failed", r["id"], str(e)[:100])
            continue
        if len(ids) >= a.n_ctx:
            chunks.append(ids[-a.n_ctx:]); man.append({"chunk": len(chunks) - 1, "bucket": b, "ids": [r["id"]]})
            got += 1
        else:
            carry_ids += ids; carry.append(r["id"])
            if len(carry_ids) >= a.n_ctx:
                chunks.append(carry_ids[-a.n_ctx:]); man.append({"chunk": len(chunks) - 1, "bucket": b, "ids": carry})
                got += 1
                carry, carry_ids = [], []
    print(b, "chunks", got)
n = len(chunks)
rows_per = a.n_ctx - 1 - a.n_ctx // 2
row_u16 = 2 * ((V + 1) // 2) + 4
size = 20 + n * a.n_ctx * 4 + n * rows_per * row_u16 * 2
with open(a.out, "wb") as f:
    f.write(b"_logits_")
    f.write(a.n_ctx.to_bytes(4, "little") + V.to_bytes(4, "little") + n.to_bytes(4, "little"))
    for c in chunks:
        f.write(b"".join(int(t).to_bytes(4, "little") for t in c))
    f.truncate(size)
json.dump({"n_ctx": a.n_ctx, "chunks": man}, open(a.out + ".json", "w"), indent=1)
print("chunks", n, "file", size >> 20, "MiB (sparse logits)")
