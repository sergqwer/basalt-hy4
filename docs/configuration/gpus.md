# GPUs

Hy4-flash takes up to three second cards as expert tiers and nothing else, each holding the experts of its own block of
layers ([memory](../models/hy4-flash.md#memory)). The layer split, the priced choice between mechanisms and the prompt
pipeline across cards were upstream's Flash-Next and Qwen3.8-27B mechanisms and were removed with them
([upstream page](https://github.com/jesdga95/basalt/blob/a11e19f6/docs/configuration/gpus.md)).

One setting picks the cards: `gpu`. It means the same in the server (`"gpu"` in the config's `server` section, or
`--gpu`), in `basalt` itself, in `basalt-bench` and in `basalt-probe`.

| Value | Cards used | Main card |
| --- | --- | --- |
| not set | GPU 0 alone; one warning at start names the other visible cards | GPU 0 |
| `"0,1"`, `"1,0"`, ... | exactly these, in this order | the first one listed |
| `"auto"` | every visible Blackwell (sm_120) card, and when built with `-DPEER_ADA=ON`, every sm_89 card as an expert tier; any other card is skipped with one line naming it | the one with the most memory bandwidth, then the most VRAM |

Cards are numbered as `nvidia-smi` numbers them. When `CUDA_VISIBLE_DEVICES` is set, the numbers count within its list
(`CUDA_VISIBLE_DEVICES=1,0` makes `gpu 0` the card `nvidia-smi` calls 1), and its numbers are read as `nvidia-smi`'s
too. A number past the visible cards, a card listed twice, a first card that is not sm_120, or a later card that is
neither sm_120 nor (when built with `-DPEER_ADA=ON`) sm_89 stops the start with an error.

The main card holds the dense weights, the KV cache, the drafter and its own expert cache. The engine decides what the
other cards do and prints one line at start, for example `gpus: RTX 5090 (nvidia-smi 0) main, RTX 5060 Ti (nvidia-smi 1)
expert tier`:

- **Expert tier.** Each card after the main one holds the experts of its own block of layers, computes their rows in
  every decode window, and computes its experts' share of each prompt chunk ([long prompts](../models/hy4-flash.md#long-prompts)).
  The start line `expert homes of 512 per layer: ...` gives the split.
- **Idle.** The engine runs up to three expert tiers: a fifth card and later ones stay idle. So does a card that would hold no
  expert: when the main card holds every expert, or a card has no room for one, the start runs without that card.

Three or more cards are community-tested: the path is checked here on two cards, with two expert tiers sharing the
second one. Reports from machines with three or four cards are welcome (see [community](../community.md)).

## Two 16 GB cards

`gpu "0,1"` with two cards of 16 GB (two RTX 5080s, say) runs the expert tier: each card holds experts in what its VRAM has
left, and the second card also reads its experts' share of each prompt. The engine sizes the second card's tier to leave
room for the buffers of that share (about 200 MiB, the size is computed, not a setting) and `peer-reserve-mib` more
(default 600 MiB; at least 128) for everything else the card needs after its experts are in. Raise `peer-reserve-mib` when
the card shows a display or another program uses its VRAM; the second card's experts shrink by the same amount. The start line `peer GPU 1: N experts` says how many
it holds.

## An older peer card

One older architecture can be built for the peer's kernels: sm_89 (Ada), beside an sm_120 main card, enabled with the
`-DPEER_ADA=ON` build flag (off by default to keep build times fast for users without a hybrid setup). Without this flag,
or for a card of any other pre-Blackwell architecture, the card is refused at the start rather than failing at its first
kernel. What such a peer does and does not do:

- It holds experts and computes their rows in every decode window, with the standard-CUDA kernels (Q8_1/Q2_0 and the
  GGUF i-quant rows, dp4a arithmetic) compiled for its own architecture. Where an expert sits does not change its rows.
- It reads its experts' share of a prompt chunk.
- It does not run the main card's work: the dense weights, the KV cache, the drafter, TMA and the FP4 tensor-core
  expert kernels stay on the Blackwell card. An NVFP4 pack's FP4 experts are main-card only.
