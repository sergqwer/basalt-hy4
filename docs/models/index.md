# Models

A model is one `.basalt` file. The server reads the architecture from the file and picks the chat format, tool syntax and
thinking switch. This fork runs one model, Hy4-flash; the others upstream Basalt runs were removed. Several packs can share
one address ([router](../configuration/models.md)). What it supports:

| | Hy4-flash |
| --- | --- |
| Architecture id | `hy4flash` |
| Packs | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt), built from Tencent's BF16 checkpoint ([model file](../model-file.md#hy4-flash-nvfp4-pack)) |
| Layers, experts | 42 (12 sparse MLA with an indexer, 30 sliding-window); layer 0 dense, then 512 per layer, top-16, plus one shared |
| Context | default 8,192; `--max-context` sets it, up to 1,048,576 |
| Drafting | DFly2 block drafter in the pack, on; `--spec` default 6 |
| Chat, tools, thinking | yes; `<tool_calls>` tools |
| Reasoning effort | `none`, `low`, `medium`, `high`, `max` (all but `none` and `max` are `high`); a number is refused |
| Images | `--vision dynamic` only |
| Video | no |
| `--kv` | fp16 (default), fp8 |
| Cards | main card plus up to three expert-tier cards |
| Layer split | no |
| Parallel slots | not measured |
| Lossy flags | `--route-prior`, `--route-tail-skip` (off) |
| Idle sleep, wake | yes (Linux) |
| `basalt-bench --quality` | yes |
| Guide | [guide](hy4-flash.md) |
| Precision | [precision](../precision/hy4-flash.md) |
