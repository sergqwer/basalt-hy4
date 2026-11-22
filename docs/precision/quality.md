# How quality is measured

Basalt scores a model against a full-precision reference on the same text, through the decode path that generates tokens. Hy4-flash's
numbers: [Hy4-flash](hy4-flash.md).

## The gate

`basalt-bench --quality` runs the gate. `tools/ppl_gate.py` runs it offline against a built engine ([commands](#reproduce-the-reference)).

- **Text:** wikitext-2 in 512-token chunks. The first 256 tokens of each chunk are prefilled.
- **Scored rows:** 256 to 510, through the decode path in verify windows of `--quality-t` tokens (default 3). The window tokens are the
  reference text, so the drafter's output is not used.
- **Reference:** a llama.cpp KL-divergence base file (`--kl-divergence-base`).
- **KLD** is KL(base || test) over the full vocabulary, from the fp32 log-softmax, summed in double. Base terms at log-probability -16 or
  below are skipped, a tiny underestimate.
- **Reported:** PPL, mean KLD (the tables also give median and p99) and top-1 agreement.
- **Length:** 20 chunks for a quick check. A change to numerics needs 100.
- **Coverage:** the 256-token prefill halves run the model's prompt path for that length (MMQ,
  [prompt path](hy4-flash.md#prompt-path)).
- **Determinism:** two runs of one build match row for row, with the tiers pinned (see [greedy identity](#greedy-identity)).

## The BF16 reference

Every score is against the original model in BF16:

| Model | Base | Base PPL |
| --- | --- | --- |
| Hy4-flash | Tencent's BF16 checkpoint converted to a BF16 GGUF (llama.cpp 0a054d0b0), through `llama-perplexity` with the llama.cpp port's Engram key/query norm epsilon set to vLLM's and `add_bos_token=true`, wikitext-2, 100 chunks of 512; `hy4flash-bf16-base.kld` | 18.572 (over the 25,500 scored rows) |

The bases are published as the [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases) dataset on Hugging Face,
with checksums: pass one to `basalt-bench --quality --kld-base FILE` to score a build or a self-converted pack.

A quantized base is not a reference. llama.cpp's Q8_0 GGUF adds its own rounding: upstream's Flash-Next packs read 0.003 to 0.010 KLD
higher against a Q8_0 base than against BF16 ([upstream](https://github.com/jesdga95/basalt/blob/a11e19f6/docs/precision/qwen38-flash-next.md#summary)).

Numerics changes: report PPL, mean KLD and top-1 next to the previous build. Bitwise-neutral changes: show greedy identity.

## Lossy routing flags

Two opt-in flags trade accuracy for fewer experts that no GPU holds. Each one given is named in
the start's one lossy warning, `--serve` reports it (`INFO lossy=`), `basalt-bench` marks the run LOSSY and `--community`
refuses it.

- `--route-tail-skip [R]`: in a decode window, an expert no GPU holds is neither copied nor computed when every entry of the window
  that names it ranks R or lower by router weight (0-based of the model's k routed experts; Hy4-flash routes 16). Its entries get weight 0; the other
  weights stay as the router gave them (no renormalisation). R is 1 to k-1, lower drops more; the flag alone is k-1.
- `--route-prior [LAMBDA]`: a decode window picks its top k as if every expert a GPU holds had LAMBDA added to the score the
  router ranks by: for Hy4-flash's sigmoid router, the biased score `s + bias`. The weights stay the router's own scores of the
  picked experts, normalised over the picks as the model normalises them. LAMBDA is in (0, 16]; the flag alone is 0.01.

"Held" means a main-card slot or a second card's. Both flags are off for prompts and for the drafters' own routing, and off when
every expert is held by a GPU (nothing to skip). Hy4-flash reads which experts are held at the start of each decode window,
so a copy that lands during a window counts from the next one. Routing then follows placement, and placement follows history, so
the same prompt can give different text on different runs. The final `TIER` line counts what `--route-tail-skip` dropped.
The measured cost and speed: [Hy4-flash](hy4-flash.md#lossy-routing-flags).

## Greedy identity

Pin the placement first, or experts moving between the CPU pool and the GPUs will round differently: a fixed `--expert-cache N`
and `--diag-tier-frozen` ([experts and placement](hy4-flash.md#experts-and-placement)).

Forced-window identity (the same window width must give the same rows):

```sh
python3 tools/ppl_gate.py --model PACK.basalt --engine-bin bin/basalt --base hy4flash-bf16-base.kld \
    --chunks 20 --t {1|3|6} --out DIR --expert-cache N --diag-tier-frozen
python3 tools/ppl_gate.py --diff DIR1/rows.tsv DIR6/rows.tsv
```

Drafting on and off, token by token: one engine session per configuration over `--serve`:

```sh
bin/basalt --serve --model PACK.basalt {--spec 0 | --spec 6} --expert-cache N --diag-tier-frozen \
    --vram-reserve-mib 700 --gpu 0,1 --max-context 8192
```

After `READY`, `GEN 256 temperature=0 ID,ID,...` on stdin returns `T ID` lines and ends with `DONE`; `QUIT` ends the session. Compare
per position.

## Reproduce the reference

The base is Tencent's BF16 checkpoint converted to a BF16 GGUF and run through `llama-perplexity` with
`--kl-divergence-base hy4flash-bf16-base.kld -c 512 --chunks 100` ([the BF16 reference](#the-bf16-reference) has the llama.cpp
commit and settings). Score a pack against it with the same command and `-m PACK.gguf --kl-divergence` added. Upstream's
Flash-Next command: [upstream](https://github.com/jesdga95/basalt/blob/a11e19f6/docs/precision/quality.md#reproduce-the-reference).
