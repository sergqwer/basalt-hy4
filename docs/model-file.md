# `.basalt` model file

One file holds everything the engine, the server and the vision sidecar load: weights, tokenizer and chat template, the DFly2
drafter, 8-bit hyper-connection copies, vision projector, and the GGUF metadata with default sampling. Flags and the expert profile are
not in it.

- **Layout:** a 64 KiB header (magic `BASALTMF`, version 2, up to 511 named sections with offset, size and SHA-256), then sections aligned
  to 2 MiB. Authoritative: [model_format.hpp](../include/basalt/core/model_format.hpp).
- **Versions:** the engine reads version 2 only. Version 1 files convert with
  [tools/migrations/basalt_v1_to_v2.py](../tools/migrations/README.md), without a download. What changed:
  [file format changes](basalt-file-changelog.md).

## Build a pack from a GGUF

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--compat-bf16] [--name NAME]
```

`--help` describes every input. `--compat-bf16` converts GGUF projections stored as BF16 in another type. A pack built this way
without the official tuned drafter is slower than the official file.

The other models' packs (DeepSeek, Flash-Next, Qwen3.8-27B, Inkling 2 Mini) were removed with those models: [upstream](https://github.com/jesdga95/basalt/blob/a11e19f6/docs/model-file.md).

## `NVFP4_PLANES` (tensor type 45)

The converter still has this type (it came with upstream's Qwen3.8-27B pack); Hy4-flash's routed experts are NVFP4 type 40.

An NVFP4 matrix of `rows` x `K` (rows a multiple of 128, K of 64) as two planes in one tensor, 36 bytes per 64 values like
ggml's NVFP4 blocks, the layout the dense products read with TMA (NInfer's and Infernix's `block_scale_k16_m128x4`):

- **Codes**, `rows * K / 2` bytes from the tensor's start: row-major, byte `b` of a row holds value `2b` in its low nibble and
  `2b + 1` in its high nibble (E2M1, as NVIDIA's checkpoints store them).
- **Scales**, `rows * K / 16` UE4M3 bytes right after the codes, one per 16 values: 512-byte tiles of 128 rows x 4 groups,
  row-block major, so the scale of row `r` and group `g` (`= k / 16`) sits at
  `((r / 128) * (K / 64) + g / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4`.
- The matrix's FP32 scale is a separate F32 [1] tensor `<name>.scale`; a value is `e2m1 * ue4m3 * scale`.

[nvfp4_planes.hpp](../include/basalt/kernels/nvfp4_planes.hpp) is the engine's copy of this layout.

## Hy4-flash NVFP4 pack

The published pack (`Hy4-flash-NVFP4.basalt`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt)) comes from Tencent's
BF16 checkpoint ([tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash), 148 shards, Tencent Hy License 1.0), through three
inputs: the checkpoint converted to a BF16 GGUF by llama.cpp's `convert_hf_to_gguf.py --outtype bf16` (llama.cpp 0a054d0b0, which
has `hy_v4_flash`), the vision projector as llama.cpp's `mmproj-Hy4-flash-BF16.gguf`, and the DFly2 drafter, the checkpoint's
`model.dfly_layers.*` tensors (its last shard, `model-00148-of-00148.safetensors`, or a side file with those 87 tensors).

```sh
python3 tools/convert/convert.py --gguf Hy4-flash-BF16.gguf --nvfp4 --hf-dir tencent-Hy4-flash \
    --mmproj mmproj-Hy4-flash-BF16.gguf --dfly tencent-Hy4-flash/model-00148-of-00148.safetensors \
    --out Hy4-flash-NVFP4.basalt --name "Hy4-flash NVFP4"
python3 tools/convert/convert.py --verify Hy4-flash-NVFP4.basalt --gguf Hy4-flash-BF16.gguf
```

- Routed experts (layers 1 to 41) NVFP4 (type 40): blobs `[gate | up | down]` and an FP32 tail `{s_gate, s_up, s_down, 0}`, 3,981,328
  bytes each; per matrix the tensor scale amax / (6 x 448), per 16 values the UE4M3 scale with the least squared error among half
  to four times amax / 6.
- The BF16 projections (attention, latent, shared expert, dense FFN, Engram key/value, indexer, embedding) Q8_0 in ggml's reference
  rounding; `--dense-q6k` makes the projections with whole 256-value rows Q6_K. The head stays BF16 (the engine rounds it to Q8_0 as it
  loads); norms, hc, router and biases F32 as stored.
- The Engram table: E4M3 codes (type 43), to nearest with ties to even, no scale; the exact fraction and RMS error go to
  `meta.quantized_from.engram`.
- Tokenizer and chat template are the GGUF's, refused unless they equal `--hf-dir`'s. `vision/mmproj` and `dfly` hold the side files
  whole; `--hf-dir` also gives `LICENSE` and `config.json`.
- 158.7 GB (experts 83.58, Engram 63.36, dense 7.45 + 0.97, projector 1.04, drafter 1.69). Every rounding is deterministic and runs once,
  in pieces by worker processes. `--verify` repeats each rounding against the BF16 GGUF and compares the bytes; it takes as long as
  the build. How each tensor rounds: [precision](precision/hy4-flash.md#weights).

## Hy4-flash test pack

From [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)'s Q4_K_M (`general.architecture` `hy_v4_flash`)
plus the model's Hugging Face directory ([tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash): `LICENSE` and `config.json`
only). Nothing is requantized: every tensor keeps its GGUF bytes and name.

```sh
python3 tools/convert/convert.py --gguf Hy4-flash-Q4_K_M-00001-of-00002.gguf --hf-dir tencent-Hy4-flash \
    --out Hy4-flash-Q4_K_M-test.basalt --name "Hy4-flash Q4_K_M (test pack)"
python3 tools/convert/convert.py --verify Hy4-flash-Q4_K_M-test.basalt --gguf Hy4-flash-Q4_K_M-00001-of-00002.gguf
```

- `dense`, `dense.index`, `gguf`: F32 and BF16 tensors in `dense`, quantized non-expert tensors in `gguf` (`gguf` holds every
  non-expert tensor, so float tensors are also in `dense`). Three-dimensional tensors (`attn_k_b`, `attn_v_b`) keep their GGUF shape in `meta.tensors`; their dense-index row is
  `(ne0, product of the rest)`. The projections (attention, latent, shared expert, layer 0's FFN, Engram key/value, indexer)
  and the token embedding are Q8_0, the head BF16 (742 MB), norms, hyper-connections, router and biases F32.
- `experts`, `experts.layout` (v5): one row per MoE layer by layer id, 1 to 41 (layer 0 is dense and has no row). Gate and up are
  Q4_K, down Q6_K (19 layers) or Q4_K (22), so a layer's blob is 3,981,312 or 4,589,568 bytes; 89.49 GB in all.
- `engram`: the table, 96 x 660,006,416 `F8_E4M3_RAW` (llama.cpp's type 43: one E4M3 code per value, no scale), 2 MiB aligned.
- `mtp/dense`, `mtp/dense.index`, `mtp/experts`, `mtp/experts.layout`: the GGUF's next-n block (the last block), BF16 as stored,
  laid out like the main sections (`meta.mtp.dense_shapes` has the 3-D shapes). The official checkpoint has no such layer, so
  the engine may ignore these sections.
- `meta`: `arch` `hy4flash`, `geometry`, `engram`, `rope` (MRoPE sections from `config.json`), `hf_config` (the whole
  `config.json`: vision tower and DFly2 settings), `license`. `vision` and `dfly2` are null until those sections are attached.
- `--verify` reads the pack back: every section's SHA-256, each source tensor found where the index, layout or `meta.tensors`
  puts it, with its type, shape, size and SHA-256.
