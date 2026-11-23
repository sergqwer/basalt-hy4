<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png"><img src="assets/basalt-lockup-light.png" alt="Basalt" width="420"></picture></p>

<h3 align="center">Basalt Hy4: a Hy4-flash fork of Basalt</h3>

> **This is a fork.** The original project is **[Basalt](https://github.com/jesdga95/basalt)** by Jesús Gulfo
> ([website](https://basalt.jgulfo.com), [releases](https://github.com/jesdga95/basalt/releases)). This fork branched from
> upstream `main` at `a11e19f6` (2026-11-22, between 0.5.5 and 0.5.6). It runs one model, Tencent's **Hy4-flash**, and adds
> the changes listed below. For other models, or for upstream's newer releases, use the original repository.

Basalt Hy4 runs **Hy4-flash** locally: a large mixture-of-experts model on one NVIDIA Blackwell GPU (sm_120). The experts
that do not fit in VRAM live in system RAM. The server speaks the OpenAI and Anthropic APIs, so Claude Code, Open WebUI and
OpenAI clients can talk to it directly.

## What this fork adds

**Prompt speed.** These changes give the same output, apart from the FP4 prompt experts, which are measured against BF16 in
the [precision notes](docs/precision/hy4-flash.md).
- Prompt routed experts run on FP4 W4A4 tensor cores by default. `BASALT_HY4_PROMPT_FP4=0` switches back to MMQ.
- One fused pass at each hyper-connection boundary of a prompt chunk: +5.7% on a 64K prompt, bit-identical output.
- The prompt expert ring's device copies run on their own stream. Host-to-GPU copies went from 29 to 54 GB/s, about the
  PCIe 5.0 x16 limit, and a 64K prompt got 4.8% faster.
- The `k_b` product writes the query rows in place, so the prompt step stops copying 0.5 GB per layer.
- The prompt MoE reads its routing from mapped pinned memory instead of a pageable copy.

**Windows.**
- `--vram-reserve` defaults to 700 MiB, and KV growth only takes free VRAM. With less headroom, WDDM silently moves
  allocations to system RAM: a 400K prompt took 586 s and now takes 120 s.
- The RAM tier's pinned pages are taken a layer ahead of the readers, which cut start-up from 46 to 17 s.
- A page-file check at start, with the steps to a fixed 64,000 MB file.
- Large pages for the RAM tier are opt-in. [`tools/windows/grant-lock-pages.bat`](tools/windows/grant-lock-pages.bat) grants
  the right.

**Server.**
- A Rust port of the server: `basalt serve ... --server rust`, parity-tested against the Python one.
- `GET /v1/stats` reports a live prompt progress: chunk, time per chunk and ETA.
- A conversation cache keeps parked conversations in RAM and gives RAM back when the system runs low. A restored turn
  continues from its own checkpoint.
- Claude Code's system prompt is rewritten for a local model (`--claude-code-prompt local`, the default): 1,373 → 720 tokens.
  It drops the Claude-only parts and keeps the tool rules.
- A reply stuck on one repeated token is ended and the engine restarted. A stray stop token in the middle of a reply is kept
  as text.

**Model pack.**
- The default pack is **[Hy4-flash-NVFP4-GPTQ](https://huggingface.co/Maximilian228/Hy4-flash-NVFP4-GPTQ-Basalt)** (158.7 GB),
  with GPTQ-calibrated NVFP4 experts. The head and the DFly2 drafter are stored pre-quantized.
- `basalt serve` downloads the pack when it is missing. The download resumes and is checked against the repo's sha256.

**Diagnostics and opt-in experiments.** Each experiment was measured and left off by default; the numbers are in the
[precision notes](docs/precision/hy4-flash.md).
- Diagnostics: `BASALT_PROMPT_PROGRESS` (per-chunk GPU timeline), `HY4_DECODE_PROFILE` (per-window decode split, drafter
  timing), `HY4_BOUNDS_CHECK=1` (names an out-of-range index instead of faulting).
- `--kv int8`: at 128K, KLD is 7-8× the tier-shift noise, with no decode gain.
- `BASALT_HY4_ROUTER_TC=1`: +3.5% on the prompt, but 4.6× noise at 128K.
- `HY4_RESEED=1`: no measurable decode gain.
- Engram int8 rows (pack type 46): 4× lower weight error than the E4M3 table, but no difference in KLD against BF16.
- Dense NVFP4 planes (pack type 45): +11% prompt and +19% decode, but KLD against BF16 rises from 0.22 to 0.38.

## Performance

Measured on an RTX 5090 32 GB (600 W, driver 621.14) with a Ryzen 9 9950X3D, 128 GB DDR5 and Windows 11. The pack is the
GPTQ pack and the build is this release's. The GPU had no display attached.

| Workload | Speed |
|---|---|
| Prompt, 64K tokens | ~4,850 tok/s |
| Prompt, 150K tokens (through the server) | 31 s (~4,830 tok/s) |
| Decode, prose (1,500-token story) | ~63 tok/s |
| Decode, structured output | ~92-95 tok/s |
| Decode at 128K context | ~40 tok/s |

Accuracy against the BF16 model (wikitext, 100 chunks): mean KLD 0.201 and top-1 agreement 86.1%. Details:
[precision](docs/precision/hy4-flash.md).

## Requirements

- An NVIDIA Blackwell GPU (compute capability 12.0, RTX 50 series), with a driver of the R580 branch or newer.
- The CUDA 13 runtime libraries on `PATH` (cudart, cuBLAS). The release archive does not include them; install the CUDA
  Toolkit 13.x.
- System RAM for the experts that do not fit in VRAM. With a 32 GB card the engine holds about 79 GB resident, so 128 GB
  is recommended. Keep a page file of 64 GB.
- 160 GB of disk for the model pack.
- Python 3.10+. Windows 10 1903+ x64, or Linux x86-64 with AVX2 when building from source.

## Quick start (Windows release)

Download `basalt-<version>-windows-x64.zip` from [Releases](https://github.com/sergqwer/basalt-hy4/releases) and unzip it,
then:

```bat
cd basalt-<version>-windows-x64
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt huggingface_hub hf_xet
```

Put the settings in `basalt_config.json`:

```json
{
  "version": 2,
  "models": [{ "model": "models/Hy4-flash-NVFP4-GPTQ.basalt" }],
  "server": { "port": 8080 }
}
```

```bat
bin\basalt.exe serve --config basalt_config.json
```

The first start downloads the 158.7 GB pack into `models/`. To use the Rust server, add `--server rust`
(`bin\basalt-serve.exe` ships beside `basalt.exe`). The server listens on 127.0.0.1. Every setting is described in
[configuration](docs/configuration/index.md), with [example configurations](docs/configuration/examples.md).

## Build from source

```sh
git clone https://github.com/sergqwer/basalt-hy4 && cd basalt-hy4
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
cargo build --release --manifest-path serve-rs/Cargo.toml -p basalt-serve   # optional: the Rust server
pip install -r requirements.txt huggingface_hub hf_xet
```

On Windows, add `-G Ninja` and build from an x64 Visual Studio 2022 prompt ([platforms](docs/platforms.md#building-on-windows)).

## Model

| | Hy4-flash |
|---|---|
| Pack | `Hy4-flash-NVFP4-GPTQ` (default), [Hugging Face](https://huggingface.co/Maximilian228/Hy4-flash-NVFP4-GPTQ-Basalt); round-to-nearest `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt) |
| Context | up to 1,048,576 |
| Drafting | DFly2, on by default |
| Images | yes |
| Documents (PDF, text) | yes |
| Guide | [guide](docs/models/hy4-flash.md) |
| Precision | [precision](docs/precision/hy4-flash.md) |

## Documentation

- **Run:** [running](docs/running.md), [configuration](docs/configuration/index.md), [sampling](docs/configuration/sampling.md),
  [image sources](docs/configuration/image-sources.md), [documents](docs/configuration/documents.md),
  [diagnostics](docs/configuration/diagnostics.md), [basalt-probe](docs/probe.md), [platforms](docs/platforms.md).
- **Model:** [Hy4-flash](docs/models/hy4-flash.md), [the `.basalt` file](docs/model-file.md),
  [file format changes](docs/basalt-file-changelog.md).
- **Quality:** [precision](docs/precision/index.md), [method](docs/precision/quality.md).
- **Build:** [building](docs/building.md). What the removed models left worth porting: [NOTES-strip.md](NOTES-strip.md).

## Credits

- **[Basalt](https://github.com/jesdga95/basalt)** by Jesús Gulfo and its
  [contributors](https://github.com/jesdga95/basalt/graphs/contributors): the engine, the server and almost everything in
  this repository. To support the original author: [GitHub Sponsors](https://github.com/sponsors/jesdga95),
  [Ko-fi](https://ko-fi.com/jesdga), [PayPal](https://paypal.me/jesdga).
- **Code Basalt builds on:**
  - [Strata](https://github.com/Niko1221/Strata) (MIT), the project Basalt forks;
  - [Infernix](https://github.com/Wallawalla47/Infernix) (Ian Ranson and contributors, Apache-2.0), whose kernels are adapted;
  - [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT);
  - ideas from [NInfer](https://github.com/Neroued/ninfer).

  Licences are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- **Models:** [Tencent](https://huggingface.co/tencent) (Hy4-flash), under the licence on its card;
  [Unsloth](https://huggingface.co/unsloth)'s GGUFs; [NVIDIA](https://huggingface.co/nvidia)'s NVFP4 checkpoints.

## License

MIT, the same as upstream; see [LICENSE](LICENSE).
