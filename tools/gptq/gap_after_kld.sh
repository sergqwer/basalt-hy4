#!/bin/bash
# gap_after_kld.sh - the shared GPU gap the coordinator asked for, after the KLD arms: once the last KLD gate
# (kld-rtn-noise) holds gpu_lock, end the parent scripts phase_post.sh / phase_eval.sh only (not their children, so
# that gate finishes normally and nothing starts after it); when it releases the lock: gpu.priority removed, gpu.next
# = "E3,E2,SRV,E1"; resume when gpu.next is empty or after 75 min with the rest of the gate. LF; not edited while running.
G=/d/Projects/hy4-data/gptq
D=/d/Projects/hy4-data
cd $G
until grep -q "gate kld-gptq done" logs/phase_post.log; do sleep 10; done
until grep -q "kld-rtn-noise" $D/gpu.lock/owner 2>/dev/null; do sleep 2; done
echo "noise gate holds the lock $(date +%T)"
for p in $(powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='bash.exe'\" | ? { \$_.CommandLine -match 'phase_post\.sh|phase_eval\.sh' } | % { \$_.ProcessId }" | tr -d '\r'); do
    taskkill //F //PID $p >/dev/null 2>&1 && echo "ended parent script $p"
done
while [ -d $D/gpu.lock ] && grep -q "kld-rtn-noise" $D/gpu.lock/owner 2>/dev/null; do sleep 10; done
echo "noise gate done $(date +%T): $(grep -E 'KLD|top' eval/gate-kld-rtn-noise/report.txt | head -2 | tr '\n' ' ')"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')" -lt 2500 ]; do sleep 5; done
rm -f $D/gpu.priority
echo "E3,E2,SRV,E1" > $D/gpu.next
t0=$(date +%s)
echo "- $(date +%H:%M) Shared GPU gap OPEN (coordinator): KLD arms done, my engine stopped, gpu.priority removed, gpu.next = E3,E2,SRV,E1; I resume when gpu.next is empty or at $(date -d @$((t0 + 4500)) +%H:%M) at the latest." >> LOG.md
echo "gap open $(date +%T)"
while [ -s $D/gpu.next ] && [ -n "$(tr -d ' ,\r\n' < $D/gpu.next)" ] && [ $(( $(date +%s) - t0 )) -lt 4500 ]; do sleep 20; done
left=$(cat $D/gpu.next 2>/dev/null | tr -d '\r\n')
echo "gptq gate $(date +%H:%M)" > $D/gpu.priority
echo "- $(date +%H:%M) Shared GPU gap CLOSED after $(( ($(date +%s) - t0) / 60 )) min; gpu.next left: '${left}'. gpu.priority back; gate continues (nll, serve arms, speed)." >> LOG.md
echo "gap closed $(date +%T) left '$left'"
bash tools/phase_eval.sh nll serve-rtn serve-gptq serve-noise speed
echo "post done $(date +%T)"
