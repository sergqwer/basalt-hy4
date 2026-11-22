# Removed when this fork went Hy4-Flash only

This fork dropped every model except Tencent Hy4-Flash. The pieces below were removed with those models and may be
worth porting to Hy4 later. Paths are as they were in upstream main **a11e19f6**
(github.com/jesdga95/basalt); `git show a11e19f6:<path>` or `git log a11e19f6 -- <path>` finds them.

## Steering

- **Control vectors (cvec) on the residual stream** (Qwen3.8-Flash-Next, `qwen4exp`): the loader and CLI
  `src/models/qwen4exp/cli/control_vectors.{cpp,hpp}` (`--control-vector`, `--control-vector-scaled`,
  `--control-vector-layer-range`, `--cvec-dir`, `--cvec-mode`, flags in `src/models/qwen4exp/flags.cpp`), the kernel
  `src/models/qwen4exp/kernels/cvec.cu` + `include/basalt/models/qwen4exp/kernels/cvec.hpp` (add and the
  projection mode), its hook points in `src/models/qwen4exp/decode/layer.cpp`, `decode/verify/record_layer.cpp`,
  `record_moe.cpp`, `record_head.cpp`, `head_split.cpp`, the conversation-state identity in
  `src/models/qwen4exp/conversation/conversation_state.cpp`, and the test
  `tests/models/qwen4exp/kernels/cvec_parity.cpp`. The shared disk tier still keys on a cvec identity
  (`src/core/conversation_store.cpp`, kept).

## Prompt path

- **FP4 two-term prompt experts** (activation row = two NVFP4 terms under one token scale): the GEMM itself is
  shared and kept (`src/prefill/moe_fp4.cu`, `include/basalt/prefill/moe_fp4.hpp`, library `prefill_fp4`), but its
  only callers were Inkling 2 (`src/models/inkling2/prompt.cpp`, `src/models/inkling2/kernels/prompt_ops.cu`,
  parity `src/models/inkling2/kernels/prompt_parity.cpp`; upstream commit 14f79a6b) and the dense stack
  (`src/dense/stage.cpp`, commit ec3be895). Hy4's prompt experts still run MMQ; wiring Hy4 to `prefill_fp4` is the port.
- **Qwen prompt pipeline** `src/models/qwen4exp/prefill/*` (layer-ahead expert stream `ahead.cpp`/`expert_stream.cpp`,
  peer-card prompt experts `peer.cpp`/`peer_experts.cpp`, CPU expert share `cpu_experts.cpp`, fused IQ MoE
  `moe_fused*.cu`), and the prompt-time layer split across two cards `src/models/qwen4exp/load/prompt_split.cpp`,
  `src/models/qwen4exp/serve/prompt_split.cpp` (test `tests/core/prompt_split_state_test.cpp`).
- **Dense two-card layer split** (Qwen3.8-27B): `src/dense/split_plan.cpp`, `include/basalt/dense/split_plan.hpp`.

## KV cache

- **KV streaming / `--kv-resident`**: `src/models/qwen4exp/kernels/kv_stream.cu` +
  `include/basalt/models/qwen4exp/kernels/kv_stream.hpp`, the flag in `src/models/qwen4exp/flags.cpp` and
  `cli/options.cpp`, the arena `src/models/qwen4exp/decode/kv_arena.cpp`; the dense version in `src/dense/session.cpp`
  and `src/models/qwen35/adapter.cpp`. Test `tests/models/qwen4exp/kernels/kv_stream_parity.cpp`.
- **Quantized KV** Q4/Q8 and the hybrid form: `src/models/qwen4exp/kernels/kv_q4.cu`, `kv_q8.cu`
  (tests `kv_q4_parity.cpp`, `kv_q8_parity.cpp`, `kv_hybrid_parity.cpp` under `tests/models/qwen4exp/kernels/`).
- **Sparse attention (QSA)** with the indexer and top-k: `src/models/qwen4exp/kernels/qsa*.cu`,
  `native_qsa_indexer.cu`, `src/models/qwen4exp/prefill/qsa.cpp`.

## Drafting

- **MTP drafter and its tuning** (`--mtp-window`, `--mtp-max-t`, `--relaxed-drafting`, `--spec-corrupt`):
  `src/models/qwen4exp/decode/mtp/*` (draft, conditional branches, batching), `include/basalt/models/qwen4exp/decode/mtp.hpp`,
  verify path `src/models/qwen4exp/decode/verify/*`, docs `docs/precision/qwen38-flash-next.md#relaxed-drafting`,
  converter fetch `tools/convert/mtp_fetch.py`. The suffix drafter and the draft policy are shared and kept
  (`src/spec/*`), but only Qwen wired `--suffix-draft`.
- **DSpark drafters**: DeepSeek-V4.1 `src/models/deepseek41/dspark*.{cpp,hpp,cu}` (with `--draft-experts`,
  `--diag-drafter-gpu`) and Inkling 2 `src/models/inkling2/dspark.{cpp,hpp}`, `kernels/dspark.cu`.

## Expert tier

- **Learned expert profiles and adaptive swaps** (`--expert-profile-save`, `--adapt-every`, `--adapt-swaps`,
  `--adapt-decay`, `--peer-adapt-swaps`): `src/models/qwen4exp/serve/tier_profile.cpp`, profiles
  `data/profiles/qwen4exp-48x512.prof`, `deepseek41-40x384*.prof`, `inkling2-24x128.prof`
  (Hy4's `data/profiles/hy4flash-42x512.prof` is kept), test `tests/core/expert_profile_save_test.cpp`.
- `--vram-elastic`, `--vram-segment-mib`, `--mmap-experts`, `--resident-experts`, `--resident-cpu-experts`,
  `--shared-expert-arena`, `--ple-row-cache`, `--prefill-cpu-share`, `--no-peer-prefill`: `src/models/qwen4exp/flags.cpp`,
  `src/models/qwen4exp/load/*`.
- **Inkling 2 expert spill** `src/models/inkling2/spill.{cpp,hpp}`, expert map `expert_map.hpp`.
- **Conversation snapshots** (`src/models/qwen4exp/conversation/*`) and the batch/parallel-slot pipeline
  (`src/models/qwen4exp/serve/batch*.cpp`, `decode/verify/batch_*.cpp`), loans (`serve/loans.cpp`, `loan_moves.cpp`).

## Other

- **Engram** tables (DeepSeek): `src/models/deepseek41/kernels/engram.cu`, `engram_hash.cpp`, `engram_rows.hpp`.
- **DeepSeek vision encoder** `src/models/deepseek41/vision/*`; Inkling 2 vision `src/models/inkling2/vision*.cpp`.
- **Planning tools** `basalt-plan` / `basalt-load` (`src/models/qwen4exp/plan/plan_main.cpp`, `load_main.cpp`) and the
  Flash-Next memory plan (`include/basalt/models/qwen4exp/plan/plan.hpp`). `basalt-device` (used by `basalt-bench
  --link-rates` and the `cuda_device_selftest` test) moved to `src/program/device_main.cpp`, minus its Flash-Next plan print.
- **Converters** for the other models: `tools/convert/checkpoint_{qwen35,deepseek41,inkling2}.py`,
  `tools/convert/arch/deepseek41.py` and their tests. The NVFP4 roundings Hy4's `hy4flash_bf16.py` took from
  `checkpoint_qwen35.py` now live in `tools/convert/fp4_round.py`. Still in the tree: `convert.py`'s GGUF path keeps
  Flash-Next's rules as its default (`tools/convert/arch/qwen4exp.py`, `mtp_fetch.py`, `--dspark`), because
  `test_convert.py` is written against them; stripping that needs a converter rewrite.
- **Shared-code tests that included a model's headers** (removed): `tests/core/{conversation_snapshot,conversation_store,
  conversation_validation,kv_arena,layout_check,native_dense_ple_key,prompt_split_state}_test.cpp`,
  `tests/kernels/ple_{bf16,q5}_parity.cpp`, `tests/ngram/ple_reader_test.cpp`, `tests/program/loan_moves_test.cpp`.
  Shared-tier tests upstream built only with qwen4exp (`file_expert_*`, `expert_cache_*`, `expert_layout`,
  `expert_profile_save`, `model_file`, `ds41_peer_parity`, `prefill_mmq_iq1s`, `fp4_dense_tma`, `silu_parity`) now
  link the shared libraries and run.
- **Tokenizer**: the qwen35 pre-tokenizer left the engine's `tokenizer` library (`src/program/tokenizer/pretok_qwen35.cpp`,
  `include/basalt/program/qwen_forms.hpp` with the Qwen tool form and token names); it stays as
  `tests/program/pretok_qwen35.cpp`, the BPE tests' reference rule. The Python tokenizer's `inkling2` split is gone.
- **Bench fixtures**: `data/bench/*.ids` are Hy4's now (were Flash-Next's; Hy4's were `data/bench/hy4flash/`);
  `data/bench/{deepseek41,inkling2}/` removed.
- **CLI help**: entries of flags no remaining parser takes were removed from `src/program/cli/help.cpp` (the PLE,
  KV streaming, cvec, adaptive-tier, resident/mmap expert and measurement flags of qwen4exp).
- **Serve dialects** `serve/models/{deepseek41,deepseek41_encoding,inkling2}.py` (the Qwen dialect and
  `serve/chat_template.jinja` stay as the mock engine's plain dialect for the serve tests).
- Model docs `docs/models/{qwen38-flash-next,qwen38-27b,deepseek41,inkling2}.md` and
  `docs/precision/{qwen38-flash-next,qwen38-27b,deepseek41,inkling2}.md`.
