# Example configurations

Start with `bin/basalt serve --config basalt_config.json`. Model paths are relative to the file. Settings: [configuration](index.md).

## One GPU (32 GB) and RAM

```json
{
  "version": 2,
  "models": [{ "model": "Hy4-flash-NVFP4-GPTQ.basalt", "gpu": "0", "max-context": 262144, "vision": "dynamic" }],
  "engine": { "vram-reserve-mib": 700 },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

## A GPU and a smaller second card (peer expert tier)

Like the first, with `"gpu": "0,1"`: the second card holds experts of its own ([GPUs](gpus.md)).

## Long context

Like the first, with `"max-context": 1048576` and `"kv": "fp8"`, which halves the sparse layers' cache
([guide](../models/hy4-flash.md#memory)).

Several packs: [models and the router](models.md).
