#!/bin/bash
# stop_server.sh - stop MY Basalt server (command line has --port 8320) and its children; wait for VRAM to drop
pids=$(powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*--port 8320*' -and (\$_.Name -eq 'basalt.exe' -or \$_.Name -eq 'basalt-serve.exe') } | ForEach-Object { \$_.ProcessId }" | tr -d '\r')
echo "my server pids: $pids"
for p in $pids; do taskkill //T //F //PID $p >/dev/null 2>&1; done
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2500 ] && { echo "VRAM back: $used MiB"; break; }; sleep 2
done
powershell.exe -NoProfile -Command "'free RAM GB: ' + [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)"
