# Configuration reference

Every setting: server options, engine flags, and the sampling fields. Effects on speed and precision: [running](../running.md),
[precision](../precision/index.md).

- [How settings are given](#how-settings-are-given) · [Automatic settings](#automatic-settings) · [Server](#server) · [Model and GPUs](#model-and-gpus)
- [Context and KV](#context-and-kv-cache) · [Expert cache](#expert-cache-and-memory) · [Drafting](#drafting)
- [Slots](#parallel-slots) · [Conversation cache](#conversation-cache) · [Vision](#vision)
- Pages: [models and the router](models.md), [sampling](sampling.md), [examples](examples.md), [diagnostics](diagnostics.md)

## How settings are given

```sh
bin/basalt serve --config basalt_config.json [options]
bin/basalt serve MODEL.basalt [--python PY] [options]
```

- **Server options** (the Python server): model, address, port, slots, vision, logging. `--name value` on the command line.
- **Engine flags** (`bin/basalt --help`): context, KV, expert cache, speculation. Given like any flag; the server passes them
  on in order and asks the engine which it accepts.

A config file is one JSON object. `models` lists the models; `engine` applies to every model that takes it; `server` is shared:

```json
{ "version": 2, "models": [{ "model": "model.basalt", "kv": "fp8" }],
  "engine": { "max-context": 65536 }, "server": { "port": 8080 } }
```

- Where a setting goes in a config file: `server` is the `server` block, `entry` a model entry or the top-level `engine`,
  `engine` (an engine flag) the same places, `model entry` a model entry only.
- Keys are flag names without dashes. Switches are `true`. A repeatable option (`allowed-host`, `cors-origin`, `image-url-host`,
  `image-dir`) takes a list. `model` paths are relative to the file.
- A flag the model does not take stops the start in a model entry or on the command line; in the top-level `engine` it is left out for that model, with a start line ([checks at start](models.md#checks-at-start)).
- Precedence: command line, then config file, then defaults; `--model` and `--alias` do not go with `--config` (its entries name the models).
  A wrong block or an unknown name stops the start. `api-key` cannot be in a file.
- The defaults are the fastest settings that need no knowledge of the machine. The keys that change output or memory (8,192 context,
  fp16 KV) are kept for you to set.

## Automatic settings

When left out, the engine picks `resident` (`resident-budget-gib`), `pool-affinity` and `pool-workers`. A given value
always wins. It prints one `[server] auto` line, where `ram-experts` is the `resident` amount (the experts held in RAM, not the whole model):

| Setting | Rule |
| --- | --- |
| `resident` | Free RAM above the safety floor, capped at the experts the GPUs do not hold |
| `pool-affinity` | `auto` |
| `pool-workers` | physical cores minus the host thread; on hybrid CPUs, P-cores minus one plus half the E-cores |

## Server

Command line `--name`; the Where column says which config block takes it.

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `model` | model entry | path | required | The `.basalt` file: weights, tokenizer, chat template, drafter, vision projector and default sampling. |
| `alias` | model entry | string | name in the file | The model name the API reports. |
| `host` | server | address | `127.0.0.1` | Listen address. `0.0.0.0` serves the network; never without `api-key`. |
| `port` | server | integer | `8080` | TCP port. |
| `api-key` | command line | string | none | Require this key on `/v1/*` (`Authorization: Bearer` or `x-api-key`). Command line only. |
| `api-key-file` | command line | path | none | Read that key from the first line of this file, which keeps it out of the process list. Not with `api-key`. Command line only. |
| `allowed-host` | server | string, repeatable | none | Also answer to this `Host` name; `"*"` turns the check off. |
| `cors-origin` | server | string, repeatable | none | Let web pages from this origin call `/v1/*`. |
| `fit-max-tokens` | entry | switch | off | Clamp `max_tokens` to the remaining context instead of a 400. |
| `open` | server | switch | off | Open the web page when ready. |
| `lazy` | server | switch | off | Start unloaded; load on the first request. Drops a sidecar encoder. |
| `idle-unload` | entry | seconds | off | Unload after this long idle. Frees VRAM for other programs. |
| `idle-sleep` | entry | seconds | off | Sleep after this long idle: VRAM freed, RAM kept, wake in seconds. Linux only. [running](../running.md#sleep---idle-sleep) |
| `min-free-vram-mib` | entry | integer | off | Load an unloaded model only when this much VRAM is free; else 503. |
| `grammar` | entry | switch (`--no-grammar`) | on | Hold tool calls and JSON formats (`json_object`, `json_schema`) to their schema inside the engine. Without it, other JSON formats are refused with 400. |
| `max-payload-mib` | server | MiB | `256` | Largest request body the server reads; a bigger one gets a 413. |
| `debug` | server | switch | off | One log line per request, and the raw model text. |
| `request-lines` | server | switch | off | Echo the per-request summary on stdout. |
| `claude-code-prompt` | server | `local` or `as_sent` | `local` | Claude Code's system prompt (a system text that starts with its billing header and identity line): `local` drops its Claude-only parts (identity, Anthropic's security policy, Claude model ids, the pronoun rule), shortens the memory how-to to its path and one paragraph, and starts with a short local-agent intro (call only the listed tools, by exact name, in the tools section's format). The same text every turn, so the prefix cache keeps hitting: Claude Code 2.1.318's `claude -p` system prompt goes from 1,373 to 720 Hy4 tokens. `as_sent` passes it as sent (billing stamp pinned). |
| `temperature`, `top-p`, `top-k`, `min-p`, `presence-penalty`, `frequency-penalty`, `repetition-penalty`, `penalty-last-n`, `seed` | entry | number (`top-k`, `penalty-last-n`, `seed`: integer) | the model file's `sampling`, else the engine's | Default for a request that leaves the field out. See [sampling](sampling.md). |
| `reasoning-effort` | entry | `none`, `low`, `medium`, `high`, `xhigh` or `max` | unset (the template's default) | Reasoning effort for requests that name no effort, thinking switch or budget. See [sampling](sampling.md). |
| `reasoning-budget-tokens` | entry | integer | `0` | Thinking budget for requests that name none. A request's own value wins. |
| `thinking-loop-tokens` | entry | integer | `256` | End thinking when one token repeats this many times in a row; `0` turns it off. |
| `answer-room` | entry | `auto` or integer | `auto` | Output tokens kept for closing the thinking and the answer: `auto`, or a fixed number (`0`: none). See [sampling](sampling.md#thinking). |
| `config` | command line | path | none | The config file. Command line only. |

## Model and GPUs

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `gpu` | entry, `basalt`, `basalt-bench`, `basalt-probe` | `"0"`, `"0,1"`, `"auto"` | GPU 0 alone | Cards, numbered as `nvidia-smi`. The first is the main card. `auto` takes every Blackwell card and makes the fastest main. [GPUs](gpus.md) |
| `peer-reserve-mib` | engine | MiB | 600 | VRAM the second card keeps free after its experts and prompt buffers. Raise it when another program uses the second card's VRAM. [GPUs](gpus.md#two-16-gb-cards) |

## Context and KV cache

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `max-context` | engine | tokens | 8192 | Longest context. Shared by `parallel` slots. The KV cache grows with it. |
| `kv` | engine | `fp16`, `fp8` | `fp16` | Cache storage of the sparse layers' rows. `fp8` stores vLLM's 656-byte FP8 row: 8.6 KB a position against `fp16`'s 14.6 ([guide](../models/hy4-flash.md#memory), [precision](../precision/hy4-flash.md#attention-caches)). The window rings and the pooled indexer keys stay fp16. Another value stops the start. |

## Expert cache and memory

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `vram-reserve-mib` | engine | MiB, at least 256 (128 on Linux) | 700 (lowered to 300 on a card with too few slots) | VRAM kept free on the main card for graphs, scratch and the head. Raised to the floor the kernels need (`--parallel 4` on Linux: at least 320). |
| `resident-budget-gib` | engine | GiB | auto | At most N GiB of experts in RAM, hottest first. |
| `expert-profile` | engine | file | `data/profiles/<arch>-<layers>x<experts>.prof` | Which experts start in VRAM. |
| `expert-cache` | engine | `auto` or a count | `auto` | Expert blobs kept in VRAM. `auto` takes what fits after weights, session and reserve. A count is honoured exactly or the start is refused with a one-line message (what fits, the free VRAM, the reserve); it is never trimmed. |
| `ram-reserve-mib` | engine | MiB | 2048 | RAM left free for the rest of the system. The expert tiers take what is above it after the engine's own host buffers, and the serving RAM guard holds it. `0` keeps nothing and turns the guard off. [memory floors](../running.md#memory-floors) |

## Drafting

With the tiers pinned, output is exactly plain decoding's ([precision](../precision/hy4-flash.md#drafting)).

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `spec` | engine | integer | 6 with the pack's DFly2 drafter, else 0 | Verify windows of at most N tokens. `0` turns it off. |
| `route-tail-skip` | engine (`[R]`) | `true` or 1 to 15 | off (bare: 15) | **Changes output.** Skips experts no GPU holds when they rank R or lower in every window entry. [precision](../precision/hy4-flash.md#lossy-routing-flags) |
| `route-prior` | engine (`[LAMBDA]`) | `true` or 0 to 16 | off (bare: 0.01) | **Changes output.** Adds LAMBDA to held experts' biased router scores. [precision](../precision/hy4-flash.md#lossy-routing-flags) |

## Parallel slots

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `parallel` | entry | 1..8 | 1 | Serve N requests at once. Sets the engine's `batch`. Not measured on Hy4-flash ([guide](../models/hy4-flash.md#parallel-slots)). |
| `slot-full-context` | entry | switch | off | Give every slot the whole `max-context`, at N times the KV. |

## Conversation cache

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `prompt-cache` | engine | integer | 6 | Conversation checkpoints kept (page-locked RAM, held from start with one more per slot). `0` reads every prompt from the start. |
| `prompt-cache-root` | engine | tokens | 2048 (`0` = never) | Checkpoint the end of the system prompt when it is at least this long. |
| `turn-token` | engine | token id | the pack's | The token that opens a chat turn; checkpoints go at the last one. |
| `tail-role-token` | engine | token id | -1 (off) | A short trailing turn that opens with this token keeps the checkpoint in front of it. |
| `conversation-cache-mib` | engine | MiB | 0 (off) | RAM budget for parked conversations. |
| `conversation-cache-slots` | engine | integer | 4 | Most parked conversations. |
| `conversation-cache-min-free-mib` | engine | MiB | 2560 | Physical RAM kept free when parking. |
| `conversation-cache-min-tokens` | engine | tokens | 0 | Park only conversations of at least this many tokens. |
| `conversation-cache-disk-mib` | engine | MiB | 0 (off) | Disk budget for parked conversations. [running](../running.md#conversations-on-disk---conversation-cache-disk-mib) |
| `conversation-cache-dir` | engine | directory | model cache folder | Another folder for the disk tier. |

## Vision

| Setting | Where | Type, values | Default | What it does |
| --- | --- | --- | --- | --- |
| `vision` | entry | `off`, `dynamic` | `off` | Image input. Hy4-flash takes `dynamic` only ([images](../models/hy4-flash.md#images)). Needs a pack with a vision projector. 32 MiB and 64 megapixels per image. [running](../running.md#vision---vision) |
| `image-url-host` | server | `HOST` or `HOST:PORT`, repeatable | none | Allow `http(s)` image URLs on this host from other machines. [image sources](image-sources.md) |
| `image-dir` | server | directory, repeatable | none | Allow image files under this directory from other machines. [image sources](image-sources.md) |
| `image-remote-only` | server | switch | off | Treat this machine's requests as remote too. [image sources](image-sources.md) |
| `video-max-tokens` | entry | 64 to 65536 | 8192 | The server still takes it, but Hy4-flash reads no video, so it has no effect. |

Diagnostic flags: [diagnostics](diagnostics.md).
