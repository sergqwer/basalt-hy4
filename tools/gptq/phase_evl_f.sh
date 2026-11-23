#!/bin/bash
# RTN arm live Claude Code eval tasks (3) + bucket F generation (45 min), on my server without dumps
G=/d/Projects/hy4-data/gptq
echo "gptq evl+F $(date +%T)" > /d/Projects/hy4-data/gpu.priority
bash $G/tools/serve.sh D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt D:/Projects/hy4-data/gptq/logs/serve_evlf.log &
until curl -s -m 3 http://127.0.0.1:8320/v1/models | grep -q loaded; do sleep 5; done
echo "server up $(date +%T)"
cd $G/a2 && python run_tasks.py --eval --tag evl > run_evl.log 2>&1
echo "evl done $(date +%T)"
cd $G && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe tools/gen_f.py --minutes 40 > logs/gen_f.log 2>&1
echo "F done $(date +%T)"
bash $G/tools/stop_server.sh
rm -f /d/Projects/hy4-data/gpu.priority
echo "E1 first" > /d/Projects/hy4-data/gpu.next
echo "- $(date +%H:%M) evl+F done, server stopped, gpu.priority removed, gpu.next = E1 first: GPU gap ~15 min (I assemble the corpus on CPU)." >> $G/LOG.md
echo "gap open $(date +%T)"
