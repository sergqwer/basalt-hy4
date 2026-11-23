#!/bin/bash
# serve.sh PACK LOG [extra args] - my own Basalt server on 127.0.0.1:8320 under the GPU lock (env passes through)
PACK=$1; LOG=$2; shift 2
source /d/Projects/hy4-data/bench-tools/gpu_free.sh
exec bash /d/Projects/hy4-data/bench-tools/gpu_lock.sh gptq bash -c "gpu_free && /d/Projects/basalt-hy4-gptq/bin/basalt.exe serve '$PACK' --server rust --host 127.0.0.1 --port 8320 --alias hy4-flash-nvfp4 --max-context 524288 --fit-max-tokens $* > '$LOG' 2>&1"
