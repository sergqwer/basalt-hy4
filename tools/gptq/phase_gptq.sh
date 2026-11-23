#!/bin/bash
# phase_gptq.sh - after the eval dump: GPTQ of layers 1-41 on the GPU (under gpu_lock), deleting each layer's train MoE
# dump once its blob is checked; then the RTN control blobs are not needed (the RTN pack is the control).
G=/d/Projects/hy4-data/gptq
until grep -q "eval dump done\|failed" $G/logs/phase_dump2.log; do sleep 10; done
grep -q failed $G/logs/phase_dump2.log && { echo "eval dump failed"; exit 1; }
sleep 5
echo "gptq math $(date +%H:%M)" > /d/Projects/hy4-data/gpu.priority
source /d/Projects/hy4-data/bench-tools/gpu_free.sh
cd $G
bash /d/Projects/hy4-data/bench-tools/gpu_lock.sh gptq bash -c "gpu_free && ./.venv/Scripts/python.exe tools/gptq_hy4.py quant --dump dump/train --layers 1-41 --out blobs/gptq --method gptq --batch 32 --delete-dumps"
echo "gptq done $(date +%T) rc $?"
