# Unreleased

On `main`, not yet tagged.

## Highlights

- **The model downloads itself.** `basalt serve` fetches a missing model file from Hugging Face before the server starts:
  a known pack name (`Hy4-flash-NVFP4-GPTQ.basalt`, `Hy4-flash-NVFP4.basalt`) needs no setting, any other file names its
  source with `"hf": "OWNER/REPO/FILE"` in its config entry. `basalt download [SOURCE] [--dir models]` fetches one ahead
  (`tools/get_model.py`). The download checks the free disk first (the file plus 5 GiB), goes into
  `.basalt-download` beside the target, resumes and retries with backoff (8 attempts), and renames the file only after its
  size and sha256 match the repo's. Progress goes to the console and the server's log every 15 s.
- **The GPTQ pack is the default Hy4-flash pack.** `Maximilian228/Hy4-flash-NVFP4-GPTQ-Basalt`: the same format, size and
  speed as the round-to-nearest pack, KLD 0.177 vs 0.205 against BF16, same top-1 86.8 % vs 85.7 %
  ([Hy4-flash](../models/hy4-flash.md#pack)).

## Upgrading

- New config keys in a model entry, `hf` and `download`, and a new `basalt serve` flag, `--no-download`. Nothing is
  removed or renamed. A config that names a model file that is not there now downloads it; add `"download": false` or
  `--no-download` to keep the old error.
