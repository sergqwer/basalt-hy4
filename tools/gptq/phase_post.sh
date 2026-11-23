#!/bin/bash
# phase_post.sh - after GPTQ: (1) pack write (CPU + disk, GPU free: gpu.priority removed), (2) expert error on the eval
# dump (GPU), (3) the gate (phase_eval.sh). LF; never edited while running.
G=/d/Projects/hy4-data/gptq
PRI=/d/Projects/hy4-data/gpu.priority
RTN=D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt
GPTQ=D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4-GPTQ.basalt
cd $G
until grep -q "gptq done" logs/phase_gptq.log; do sleep 20; done
n=$(ls blobs/gptq/l*.json | wc -l)
[ "$n" -eq 41 ] || { echo "only $n layer blobs"; exit 1; }
rm -f $PRI
echo "- $(date +%H:%M) GPTQ 41 layers done; pack write (CPU/disk) now, GPU gap ~20 min, gpu.priority removed." >> LOG.md
echo "write start $(date +%T)"
./.venv/Scripts/python.exe tools/gptq_hy4.py write --blobs blobs/gptq --base $RTN --pack $GPTQ || { echo "write failed"; exit 1; }
echo "write done $(date +%T)"
echo "gptq eval $(date +%H:%M)" > $PRI
echo "- $(date +%H:%M) Pack written; gpu.priority back for the gate." >> LOG.md
source /d/Projects/hy4-data/bench-tools/gpu_free.sh
bash /d/Projects/hy4-data/bench-tools/gpu_lock.sh gptq bash -c "gpu_free && ./.venv/Scripts/python.exe tools/gptq_hy4.py analyze --dump dump/eval --layers 1-41 --step 2 --pack $RTN --pack $GPTQ --out eval/expert_error.jsonl" > logs/analyze.log 2>&1
echo "analyze done $(date +%T)"
bash tools/phase_eval.sh kld nll serve-rtn serve-gptq serve-noise speed
echo "post done $(date +%T)"
