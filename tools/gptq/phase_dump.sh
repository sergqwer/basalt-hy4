#!/bin/bash
# the dump pass: train corpus (MoE layers 1-41 + dense inputs of 7 layers, every 16th row), then the eval corpus (MoE only)
G=/d/Projects/hy4-data/gptq
echo "gptq dumps $(date +%T)" > /d/Projects/hy4-data/gpu.priority
mkdir -p $G/dump/train/dense $G/dump/eval
export BASALT_GPTQ_DUMP=D:/Projects/hy4-data/gptq/dump/train BASALT_GPTQ_MOE_LAYERS=1-41 BASALT_GPTQ_DENSE_LAYERS=0,1,5,11,21,31,41 BASALT_GPTQ_DENSE_STRIDE=16
bash $G/tools/serve.sh D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt D:/Projects/hy4-data/gptq/logs/serve_dump_train.log &
until curl -s -m 3 http://127.0.0.1:8320/v1/models | grep -q loaded; do sleep 5; done
echo "train server up $(date +%T)"
cd $G && ./.venv/Scripts/python.exe tools/run_dump.py corpus/train.jsonl dump/train/log.jsonl > logs/dump_train.log 2>&1
echo "train dump done $(date +%T)"
bash $G/tools/stop_server.sh
export BASALT_GPTQ_DUMP=D:/Projects/hy4-data/gptq/dump/eval BASALT_GPTQ_DENSE_LAYERS=
unset BASALT_GPTQ_DENSE_LAYERS
bash $G/tools/serve.sh D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt D:/Projects/hy4-data/gptq/logs/serve_dump_eval.log &
until curl -s -m 3 http://127.0.0.1:8320/v1/models | grep -q loaded; do sleep 5; done
echo "eval server up $(date +%T)"
cd $G && ./.venv/Scripts/python.exe tools/run_dump.py corpus/eval.jsonl dump/eval/log.jsonl > logs/dump_eval.log 2>&1
echo "eval dump done $(date +%T)"
bash $G/tools/stop_server.sh
