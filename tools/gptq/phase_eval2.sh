#!/bin/bash
# phase_eval.sh STEP... - the RTN vs GPTQ gate, same build (my worktree's bin), same day. Each engine run takes the GPU
# lock on its own (~60 s to a few minutes each); results under eval/.
# steps: kld nll speed serve-rtn serve-gptq
G=/d/Projects/hy4-data/gptq
BIN=/d/Projects/basalt-hy4-gptq/bin
RTN=D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4.basalt
GPTQ=D:/Projects/basalt-work/models/hy4/Hy4-flash-NVFP4-GPTQ.basalt
KLD=D:/Projects/hy4-data/gptq/kld/hy4flash-bf16-base.kld
NLL=D:/Projects/hy4-data/gptq/eval/own-nll-base.kld
source /d/Projects/hy4-data/bench-tools/gpu_free.sh
mkdir -p $G/eval
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -d '\r'; }
waitram() { until [ "$(freegb)" -ge 95 ]; do sleep 10; done; }   # an engine that just stopped returns its RAM slowly
lock() { waitram; bash /d/Projects/hy4-data/bench-tools/gpu_lock.sh gptq bash -c "gpu_free && $*"; }
gate() {   # gate NAME PACK BASE CHUNKS CACHE
    local d=$G/eval/gate-$1; mkdir -p $d
    [ -f $d/report.txt ] && { echo "skip $1"; return; }
    lock "$BIN/basalt-bench.exe --model '$2' --quality $4 --quality-t ${QT:-1} --kld-base '$3' --logs '$(cygpath -w $d)' -- --expert-cache $5 --diag-tier-frozen > '$d/report.txt' 2> '$d/err.txt'"
    echo "$(date +%H:%M) gate $1 done: $(grep -E 'KLD|top|PPL' $d/report.txt | head -3 | tr '\n' ' ')"
}
for step in "$@"; do
  case $step in
    kld)
      gate kld-rtn   $RTN  $KLD 100 4800
      gate kld-gptq  $GPTQ $KLD 100 4800
      gate kld-rtn-noise $RTN $KLD 100 4000 ;;
    nll)
      QT=3 gate nll-rtn   $RTN  $NLL 34 4800
      QT=3 gate nll-gptq  $GPTQ $NLL 34 4800
      QT=3 gate nll-rtn-noise $RTN $NLL 34 4000
      # coordinator: a short slot for FIXA (the tray's bug A) before the serve arms, <= 20 min
      rm -f /d/Projects/hy4-data/gpu.priority
      rest=$(tr -d '\r\n' < /d/Projects/hy4-data/gpu.next 2>/dev/null)   # names left from the shared gap wait for my end
      echo "FIXA,E3" > /d/Projects/hy4-data/gpu.next
      s0=$(date +%s)
      echo "- $(date +%H:%M) FIXA slot OPEN (after the NLL arms): gpu.priority removed, gpu.next = FIXA,E3, <= 25 min." >> $G/LOG.md
      while [ -n "$(tr -d ' ,\r\n' < /d/Projects/hy4-data/gpu.next 2>/dev/null)" ] && [ $(( $(date +%s) - s0 )) -lt 1500 ]; do sleep 15; done
      echo "gptq gate $(date +%H:%M)" > /d/Projects/hy4-data/gpu.priority
      echo "- $(date +%H:%M) FIXA slot CLOSED after $(( ($(date +%s) - s0) / 60 )) min (gpu.next left: '$(tr -d '\r\n' < /d/Projects/hy4-data/gpu.next 2>/dev/null)'); gpu.priority back." >> $G/LOG.md
      rest=$(echo "$rest" | tr "," "\n" | grep -vxE "FIXA|E3" | paste -sd, -)   # those two ran in the slot
      [ -n "$rest" ] && echo "$rest" > /d/Projects/hy4-data/gpu.next ;;
    speed)
      # the HF upload reads the GPTQ pack buffered (churns the file cache): wait for UPLOAD-OK, at most until 08:45
      until grep -q "UPLOAD-OK" $G/logs/hf_upload.log 2>/dev/null || [ "$(date +%H%M)" -ge 0845 ]; do sleep 30; done
      up=$(grep -q "UPLOAD-OK" $G/logs/hf_upload.log 2>/dev/null && echo "after UPLOAD-OK" || echo "UPLOAD NOT FINISHED (cache churn)")
      echo "- $(date +%H:%M) speed arms start, $up; one warm-up run discarded." >> $G/LOG.md
      d=$G/eval/speed-warmup; mkdir -p $d
      lock "$BIN/basalt-bench.exe --model '$RTN' -p 16384 -t conv --seed 7 --logs '$(cygpath -w $d)' > '$d/report.txt' 2> '$d/err.txt'"
      for arm in rtn gptq; do
        p=$RTN; [ $arm = gptq ] && p=$GPTQ
        d=$G/eval/speed-$arm; mkdir -p $d
        lock "$BIN/basalt-bench.exe --model '$p' -p 16384,65536 -t conv,struct,prose --seed 7 --logs '$(cygpath -w $d)' > '$d/report.txt' 2> '$d/err.txt'"
        echo "$(date +%H:%M) speed $arm done"
      done ;;
    serve-rtn|serve-gptq|serve-noise)
      # rtn: the RTN arm (live tasks tag evl, seed 1); noise: RTN again (live tag evn, CJK seed 2); gptq: tag evg
      arm=${step#serve-}; p=$RTN; [ $arm = gptq ] && p=$GPTQ
      waitram
      sl=D:/Projects/hy4-data/gptq/logs/serve_eval_$arm.log; rm -f $sl
      bash $G/tools/serve.sh $p $sl &
      until curl -s -m 3 http://127.0.0.1:8320/v1/models | grep -q loaded; do
        sleep 5
        grep -q "\[server\] stopped" $sl 2>/dev/null && { echo "server $arm failed: $(grep error $sl | head -1)"; continue 2; }
      done
      cd $G
      if [ $arm = noise ]; then
        ./.venv/Scripts/python.exe tools/evalsuite.py cjk rtn-noise --seed 2 > eval/cjk-rtn-noise.log 2>&1
        (cd a2 && python run_tasks.py --eval --tag evn > run_evn.log 2>&1)
      else
        ./.venv/Scripts/python.exe tools/evalsuite.py web $arm > eval/web-$arm.log 2>&1
        ./.venv/Scripts/python.exe tools/evalsuite.py cjk $arm --seed 1 > eval/cjk-$arm.log 2>&1
        ./.venv/Scripts/python.exe tools/evalsuite.py replay $arm > eval/replay-$arm.log 2>&1
        ./.venv/Scripts/python.exe tools/evalsuite.py loops $arm > eval/loops-$arm.log 2>&1
        tag=evl; [ $arm = gptq ] && tag=evg
        (cd a2 && python run_tasks.py --eval --tag $tag > run_$tag.log 2>&1)
      fi
      bash $G/tools/stop_server.sh
      echo "$(date +%H:%M) serve $arm done" ;;
  esac
done
