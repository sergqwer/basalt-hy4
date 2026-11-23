#!/bin/bash
# phase_dump2.sh - (1) the 70 train B rows the first pass rejected (Python type names), types fixed, appended to
# dump/train (log dump/train/log_fixB.jsonl); (2) the eval dump with types fixed. Waits for >= 95 GB free RAM before
# each engine start (an engine that just stopped returns its RAM slowly).
G=/d/Projects/hy4-data/gptq
RTN=D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt
echo "gptq dump $(date +%H:%M)" > /d/Projects/hy4-data/gpu.priority
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -d '\r'; }
waitram() { until [ "$(freegb)" -ge 95 ]; do sleep 10; done; echo "free RAM $(freegb) GB $(date +%T)"; }
up() {
    bash $G/tools/serve.sh $RTN $1 &
    until curl -s -m 3 http://127.0.0.1:8320/v1/models | grep -q loaded; do
        sleep 5
        if grep -q "\[server\] stopped" $1 2>/dev/null; then echo "server failed: $(grep error $1 | head -1)"; exit 1; fi
    done
    echo "server up $(date +%T)"
}
cd $G
./.venv/Scripts/python.exe tools/fix_types.py corpus/train.jsonl dump/train/log.jsonl corpus/train_fixB.jsonl
./.venv/Scripts/python.exe tools/fix_types.py corpus/eval.jsonl --all corpus/eval_fixed.jsonl
waitram
export BASALT_GPTQ_DUMP=D:/Projects/hy4-data/gptq/dump/train BASALT_GPTQ_MOE_LAYERS=1-41 BASALT_GPTQ_DENSE_LAYERS=0,1,5,11,21,31,41 BASALT_GPTQ_DENSE_STRIDE=16
up D:/Projects/hy4-data/gptq/logs/serve_dump_fixb.log
./.venv/Scripts/python.exe tools/run_dump.py corpus/train_fixB.jsonl dump/train/log_fixB.jsonl > logs/dump_fixb.log 2>&1
echo "fixB done $(date +%T)"
bash $G/tools/stop_server.sh
unset BASALT_GPTQ_DENSE_LAYERS
export BASALT_GPTQ_DUMP=D:/Projects/hy4-data/gptq/dump/eval
waitram
up D:/Projects/hy4-data/gptq/logs/serve_dump_eval2.log
./.venv/Scripts/python.exe tools/run_dump.py corpus/eval_fixed.jsonl dump/eval/log.jsonl > logs/dump_eval.log 2>&1
echo "eval dump done $(date +%T)"
bash $G/tools/stop_server.sh
