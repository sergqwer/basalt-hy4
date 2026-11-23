# Running the server

Settings: [configuration](configuration/index.md). Diagnostic flags: `bin/basalt --help`. Several models: [models and the router](configuration/models.md).

## Start-up

The page is up at once and the model loads behind it. The header reads `Loading` with the percent over a progress bar while weights load; choosing a model in the picker loads it, and the Unload button beside the model name unloads the loaded one until the next request; each reads `Loading…` or `Unloading…` until the server shows the change, and a refused one shows its error in the header. `/health` and `/metrics`
show the load: `load.percent` and `load.gib`, the GiB placed so far in one figure over every byte the model loads (a card, a second card, RAM), where it was one figure per target. A failed start-up load ends the server. Requests wait during a load and get a 503 if it fails.

## Concurrent requests: `--parallel N`

`--parallel N` (1 to 8) serves N requests at once in batch slots. Further requests wait in arrival order. `--max-context` is shared:
each slot holds `max-context / N` tokens (`--parallel 4 --max-context 262144` gives four slots of 65,536). A request longer than its slot
is refused with a 400 that names the length; the start log's `slots` line shows the context each slot gets. `--slot-full-context` gives every slot the whole `--max-context`, at N times the KV cache.

Each slot keeps two checkpoints (end of the system prompt, last turn), so requests that share a long system prompt read it once.

The engine tokenizes every prompt and matches it against what each slot holds. A reply the client sends back unchanged continues the
slot's own tokens even when the model wrote it in other pieces than the tokenizer cuts the same bytes into (not with images).

### Session affinity

With `--parallel`, a request that names its session goes to the slot that session's conversation sits in, so the next turn continues
the K/V already there. The session is the first of these headers that is set, then the request body's `session_id`: `x-llm-flow-id`,
`x-claude-code-session-id`, `x-session-id`, `x-session-affinity`, `x-grok-conv-id`, `x-grok-session-id`. A request with none picks a
slot by the longest shared prefix, then the least recently used. A session may hold several slots: a client may send one session id for a
session and its parallel subagents (Claude Code's `x-claude-code-session-id`, OpenCode's `X-Session-Id` and `x-session-affinity`, which
name the parent session), and each conversation keeps the slot its K/V is in. A request that continues none of the session's own slots
takes an unowned one first; with every slot owned, the session's own least recently used one comes before another session's.

### Prompts beside decoding slots

A prompt of up to 16,384 tokens is read at once, and the slots wait a few seconds. A longer one is read beside them, on its own thread,
while the slots keep decoding. A short request that arrives meanwhile goes first at the end of the current chunk.

## The console log

Every line starts with `[server]`. Values are `key=value`. Start-up prints a legend, a line per model, `ready`, `loaded`, `kv`, `slots`.
While a prompt is read, a slot prints `pp=<read>/<total>` with the rate. While requests generate, a `live` line appears every 2 s.
Each request ends with one `done` line:

```
[server] s1 done  pp=96/89c 0.1s  tg=1720 119.6t/s acc=65% dl=2.9  ttft=0.1s  end=stop  hit=89.7%  pcie=39.3%
```

- `pp=<prompt tokens>/<cached>c`: the rate counts newly read tokens only.
- `acc`: drafts accepted over drafts offered. `dl`: tokens per window, from the engine's own window count over the whole request (its solo and batch runs and every pass). With `--parallel` a window serves every busy slot in a fixed number of rows, so `dl` falls toward 1.0 as the slots fill. Both are absent when the request did not draft.
- `end`: `stop`, `length`, `cancelled`, `disconnect` or `error`.
- `hit`: expert cache hit rate. `pcie`: share of routed experts read over PCIe.

The engine's own `basalt ...` lines go to `~/.cache/basalt/<id>/engine.log`, one folder per model file (`<id>` is 16 hex digits
taken from the file); `[server] engine log: PATH` names it at start. A log of 10 MiB or more moves to `engine.log.1` at the
next start. An engine that dies of a fault or an unhandled exception writes a `basalt crash: error: ...` line (the fault, the stage
it was in) and the server's warning quotes it with the exit status. The folder also holds the conversation
cache. Nothing is unpacked from the model file: the server reads the tokenizer from its sections, and the engine and `basalt-vision`
read the vision projector inside it.

## The stats page

The server's address serves a page, updated every second from `/metrics`, `/slots` and `/api/logs`. The top of the page always shows:

- Decode and prefill: the live rate with a 60 s chart; when idle, the last request's decode and the last prompt read of at least 512
  new tokens (a cached follow-up has no meaningful read rate). The decode number's tooltip has the best decode and the conversation
  cache; the prefill number's has the last prompt, its prefix cache hit and the best prefill (the fastest read listed when the
  server has none yet).
- Experts: where the last request's expert lookups were served (VRAM, a peer card, CPU and RAM, the SSD), as a legend, and a map
  of 640 experts sampled evenly from every layer (`experts_map`): its colour is where the expert lives now, its brightness how often
  decode looked it up lately, an ember edge a lookup off the GPUs in the last second. A dense model shows its weights by bytes.
  Experts on the SSD turn the heading amber.
- With no model loaded, a slow field of basalt columns on its own (8 frames a second, paused while the tab is hidden, one still
  frame under reduced motion).

Under it, four tabs (keys `1` to `4`, `L` for the logs, or `#tab=req|slots|hw|logs` in the address):

- Requests: the queue, the recent requests, tokens per request, the last request, the best since the server started, the
  conversation cache (`conversation`: `parked`, `bytes`, `evictions`, `restored`, `restored_ms`) and the totals (`totals`).
- Slots: a card per slot with its current request, or for an idle slot its last one, its context and its drafting; past four
  slots, compact rows. With no model loaded the cards stay, empty.
- Hardware: each card's load, power, temperature, VRAM and PCIe traffic, and the CPU, RAM, disk reads, experts in VRAM, driver and
  uptime.
- Logs: the console lines, all or warnings only. The engine's long cache lines are shown short (`cache: restored 187k tokens in
  912 ms`); the log and `/api/logs` keep the raw line. The footer always shows the newest line and the request and token totals.

The engine's own figures, which the server asks for with `STATS` once the engine is ready:

- `/metrics` `experts_map`: `total` experts, and for each sampled one `tier` (0 VRAM, 1 peer card, 2 RAM, 3 file),
  `routed` (decode lookups a second, 0 once nothing runs) and `recent` (its lookups, fading by 0.85 a second).
- `/slots` per slot: `started` (epoch seconds), `path` (the endpoint) and `prompt_ms` (the read time) of the running request;
  for its current or last request `accept_pos` (drafts accepted over offered by draft position) and `accept_hist` (tokens per
  decode window, a sample a second over 60 s); for a model with routed experts `hit`: its windows' lookups split `vram`, `peer`
  (a bus copy or another card) and `cpu`, with `lookups_s`, `peer_s` and `ram_s` a second. A batch window several slots share
  counts its lookups for each of them; a model whose slots run through layer split stages counts none per slot.
- `/metrics` `engine`: `commit` (the build's), and `expert_slots`, the experts every card holds now (fewer while asleep).
- `/metrics` `hardware`: each card's `sm_clock` and `mem_clock` (MHz), `throttle` (what holds the clocks down, empty for nothing),
  `expert_slots`; `cpu_ccd` (the load of each group of cores that share an L3 cache), `ram_parts` (`experts` the engine holds in
  RAM, `page_cache`, `other`; Linux only) and `disk_hist` (the read rate's last 60 s).

The header word reads `Thinking`, `Answering` or `Tool call` (`live.phase`) while exactly one request runs. The conversation cache
figures come from the engine's `conversation cache:` lines and clear when the engine unloads, sleeps or restarts; a model that prints
no line when it parks shows them after its first restore, without evictions.

### `GET /v1/stats`

One JSON document for a stats window (the tray's `Statistics...`), with the same key as every other endpoint; `basalt-serve`
(the Rust server) only, the Python server answers 404. It holds only what the engine reports, and `null` where it reports
nothing:

- the engine: `state` (as `/metrics` `live.state`), `model`, `engine_version`, `quant`, `gpus`, `kv`, `uptime_s`;
  `context` (`max`, `in_use` summed over the slots, `largest_slot`); `slots` (`summary`, per-slot `rows`, `queue`);
  `vram` (as `/v1/status`); `experts` (`slots`, `slots_primary`, `ram_mib`, `total`, `spill`, from the engine's INFO line);
  `conversation` (the conversation cache, as `/metrics`); `totals`, `best` and `machine` (GPU and RAM).
- `large_pages`: `state` `"yes"` or `"no"` from the newest of the engine's own memory lines that says (large pages, or 4 KB
  pages), with that `line`; `null` when no line said.
- `load`: `in_progress` while loading, `last` (`kind` weights or wake, `seconds`, `at`: the last load the server waited
  for) and `memory_lines`, the engine's memory and load lines (large pages, the RAM copy of the experts, the cards' share),
  kept after the log ring moves past them (the last 32).
- `requests`: the last `n` finished requests, newest first (`?n=`, 50 by default, at most 500): `prompt_tokens`, `reused`,
  `read_tokens`, `output_tokens`, `prompt_ms`, `prefill_tok_s` (read tokens over `prompt_ms`), `decode_tok_s`, `ttft_s`,
  `drafts_offered`, `drafts_accepted`, `draft_acceptance` (accepted over offered), `draft_len`, `expert_hit_rate`,
  `pcie_share`, `ssd_share`, `end_reason`, `duration_s`, `slot`, `path`, `time`.

## Conversations on disk: `--conversation-cache-disk-mib`

Off at 0. With a budget, conversations of at least 8,192 tokens are written to `conversations` in the model's cache folder when they leave
RAM, a session, or at exit. Only the system prompt's and the last three turns' states are stored. Files are bound to the model and settings
they were made with, and checked by hash; a bad file is deleted. The folder is capped by the budget (least recently used first), and a
write always leaves 4 GiB of the volume free. After a restart, a prompt that starts with a stored conversation reads it back.

## Sleep: `--idle-sleep`

After the given seconds without requests, the engine frees its VRAM (the driver moves the GPU state to host memory). The process and its
RAM copy of the experts stay (a dense model has no cache to give back first: the driver moves all its GPU state, both cards' on two cards). The next request wakes it in seconds. Linux only. The server stays awake when the GPU state would not fit in
free RAM with 8 GiB to spare. `--idle-unload` ends the process instead.

## Lending VRAM: `POST /v1/vram`

With `--vram-elastic`, `POST /v1/vram` with `{"reserve_mib": N}` resizes the expert cache to keep N MiB free (`null`: the start's reserve).
A resize needs a gap between requests. Idle, it applies at once and answers `200` with the new figures. While a request runs it is
queued and answered `202 {"status": "pending"}`: it applies when the running requests have finished, and requests that arrive meanwhile
wait for it, so steady traffic cannot hold it off. A later POST replaces a queued one (`"replaced": true`). A malformed body is a
`400`. `GET /v1/status` shows `vram.pending` (the queued reserve, or `null`), `vram.applied` and `vram.error` (the engine refused
the queued one).

## Memory floors

Basalt refuses settings that would leave the machine without memory, and never shrinks what you asked for:

- `--vram-reserve-mib` below 256 is refused on Windows and WSL2 (128 on Linux): less free VRAM makes the driver page GPU memory.
- `--ram-reserve-mib` (default 2048) is the RAM left free for the rest of the system. The engine allocates its host buffers
  first, or holds their RAM when it page-locks them later; the expert tiers then take the free RAM less the reserve and less
  the driver's tracking of the pages they pin (`RAM floor ...` in the start log). When some experts stay in the file, the
  tiers also leave the OS file cache they are read through a floor: one prompt chunk's file reads and two layers' share of
  those experts, at least 4 GiB. Programs left idle beside a small reserve may be swapped out.
- While serving, the RAM guard holds the reserve: below it, it parks conversations to disk or drops them. A request waits up to 30 s
  for RAM to return. A running request stops with an error only when RAM is critical (under 1 GiB free, or below the reserve under
  memory pressure). `--ram-reserve-mib 0` turns the guard off.
- Experts on the SSD give one warning, the same for every model: `N experts on SSD (X GiB) · free X GiB RAM`. The console
  repeats it and the stats page adds an SSD tier and turns its Experts heading amber.
- The disk tier keeps its floor a GiB above the reserve.
- `--resident-budget-gib` that would leave less than the floor is refused with the largest safe value.

## Vision: `--vision`

`--vision off | dynamic | cpu | <GPU number>` (config: `"vision"` in `server`). The encoder is Basalt's own. The GPU and CPU
encoders round to bf16 at the same steps but are not bitwise identical: on Hy4-flash, a 3912 x 3936 photo (4,032 merged tokens)
gives a relative L2 difference of 4.8e-3 between the two, and each is 4.7e-3 from an FP32 pass (cosine of every row at least
0.9995; 0.002% of the values more than 11% off), so the difference is bf16 rounding. All of them read the projector in place from
the model file's `vision/mmproj` section. The text can still differ between modes: the sidecar's VRAM cost shrinks the expert
cache, and which experts run on the GPU or the CPU shifts the last bits of the logits, so a greedy answer can fork after some
tokens (`dynamic` does the same with a different cache size).

| Mode | Encoder runs | VRAM cost | Fresh-image TTFT (IQ3_S, 400 W): 1k / 3k / 4k image tokens |
| --- | --- | --- | --- |
| `off` | none | none | |
| GPU number | sidecar `bin/basalt-vision` on that card | about 2.2 GiB of its expert cache | 0.77 s / 1.4 s / 1.8 s |
| `dynamic` | inside the engine, on the main GPU | none at rest (0.85 GiB pinned RAM) | 0.77 s / 1.45 s / 1.8 s |
| `cpu` | sidecar on the CPU | none | about 6 s / 40 s / 70 s |

`dynamic` borrows the cache's last slots per image and refills them afterwards (about 70 ms), so the cache is identical after.
Hy4-flash reads images with `dynamic` only: its main-card expert slots go back for the request's pictures and fill again as after a
prompt. Which
images a request may name: [image sources](configuration/image-sources.md). An image may not sit in a system prompt.

## Video

Hy4-flash reads no video. Upstream's video input (Qwen3.8-Flash-Next and Qwen3.8-27B):
[upstream](https://github.com/jesdga95/basalt/blob/a11e19f6/docs/running.md#video).
