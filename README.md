<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png"><img src="assets/basalt-lockup-light.png" alt="Basalt" width="480"></picture></p>

> **Hy4-flash-only fork.** This is a fork of Basalt ([upstream](https://github.com/jesdga95/basalt), forked from main
> `a11e19f6`) that runs Tencent's Hy4-flash only. The other models (Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, Qwen3.8-27B,
> Inkling 2 Mini) were removed; [NOTES-strip.md](NOTES-strip.md) lists the pieces of them worth porting.

Basalt is a local inference engine for **Hy4-flash**, a large mixture-of-experts language model. It runs on an NVIDIA
Blackwell GPU (sm_120) on Linux and Windows x64, with what does not fit in VRAM (the experts) in system RAM, and optionally on
more GPUs. The server speaks the OpenAI and Anthropic APIs. Website: [basalt.jgulfo.com](https://basalt.jgulfo.com).

<p align="center"><img src="docs/img/frontend.gif" alt="The Basalt web UI while a reply streams" width="720"></p>

**Help wanted:** spare Blackwell compute for testing? See [Contributing](CONTRIBUTING.md#other-ways-to-help).

<p align="center">
  <a href="https://github.com/sponsors/jesdga95"><img src="assets/donate/github.svg" height="44" alt="Sponsor on GitHub"></a>
  <a href="https://ko-fi.com/jesdga"><img src="assets/donate/kofi.svg" height="44" alt="Support on Ko-fi"></a>
  <a href="https://paypal.me/jesdga"><img src="assets/donate/paypal.svg" height="44" alt="Donate on PayPal"></a>
</p>

## Requirements

- Linux x86-64 with AVX2, or Windows 10 1903+ x64 ([platforms](docs/platforms.md)).
- An NVIDIA Blackwell GPU (compute capability 12.0) with CUDA 13.
- CMake 3.24+, GCC 13+ (Linux), Python 3.10+, and enough system RAM for the experts that do not fit in VRAM.

## Quick start

Build from source (`hy4flash` is the only model, so the build needs no `-DBASALT_MODELS`):

```sh
git clone https://github.com/sergqwer/basalt-hy4 && cd basalt-hy4
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
                                        # Windows: add -G Ninja to the first line, from an x64 VS 2022 prompt ([platforms](docs/platforms.md#building-on-windows))
pip install -r requirements.txt
hf download jesdga/Hy4-flash-Basalt Hy4-flash-NVFP4.basalt --local-dir models
```

Put the settings in `basalt_config.json`:

```json
{
  "version": 2,
  "models": [{ "model": "models/Hy4-flash-NVFP4.basalt", "gpu": "0,1" }],
  "server": { "port": 8080 }
}
```

```sh
bin/basalt serve --config basalt_config.json
```

The server listens on 127.0.0.1. Every setting: [configuration](docs/configuration/index.md). More examples:
[example configurations](docs/configuration/examples.md).

## Model

| | Hy4-flash |
|---|---|
| Pack | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt) |
| Context | up to 1,048,576 |
| Drafting | DFly2, on by default |
| Images | yes |
| Video | no |
| Documents (PDF, text) | yes |
| Guide | [guide](docs/models/hy4-flash.md) |
| Precision | [precision](docs/precision/hy4-flash.md) |

What it supports, with its context limit, flags and cards: [models](docs/models/index.md).

## Documentation

- **Run:** [running](docs/running.md), [configuration](docs/configuration/index.md), [models and the router](docs/configuration/models.md),
  [sampling](docs/configuration/sampling.md), [image sources](docs/configuration/image-sources.md),
  [documents](docs/configuration/documents.md),
  [diagnostics](docs/configuration/diagnostics.md), [basalt-probe](docs/probe.md), [platforms](docs/platforms.md).
- **Model:** [models](docs/models/index.md), [Hy4-flash](docs/models/hy4-flash.md),
  [the `.basalt` file](docs/model-file.md), [file format changes](docs/basalt-file-changelog.md).
- **Quality:** [precision](docs/precision/index.md), [method](docs/precision/quality.md).
- **Build and develop:** [building](docs/building.md), [adding a model](docs/adding-a-model.md), [community runs](docs/community.md),
  [changelog](docs/changelog/index.md), [contributing](CONTRIBUTING.md).

## Credits

- **Code:** [Strata](https://github.com/Niko1221/Strata), the project Basalt forks (MIT); [Infernix](https://github.com/Wallawalla47/Infernix)
  (Ian Ranson and contributors, Apache-2.0), whose kernels are adapted; [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp)
  (MIT); and ideas from [NInfer](https://github.com/Neroued/ninfer). Licences: [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- **Models:** [Tencent](https://huggingface.co/tencent) (Hy4-flash), under the licence on its card;
  [Unsloth](https://huggingface.co/unsloth)'s GGUFs; [NVIDIA](https://huggingface.co/nvidia)'s NVFP4 checkpoints.

Thank you to everyone who has [contributed](https://github.com/jesdga95/basalt/graphs/contributors) or
[sponsored](https://github.com/sponsors/jesdga95) the project.

## License

MIT, see [LICENSE](LICENSE).
