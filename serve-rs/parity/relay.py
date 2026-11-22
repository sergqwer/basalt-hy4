"""An engine binary in the middle: runs the real engine ($RELAY_ENGINE) with the same arguments and, for `--serve`, copies
every stdin byte to the engine and to $RELAY_RECORD. stdout and stderr are the engine's own (inherited)."""
import os
import subprocess
import sys
import threading

real = os.environ["RELAY_ENGINE"]
args = sys.argv[1:]
if "--serve" not in args or not os.environ.get("RELAY_RECORD"):
    sys.exit(subprocess.call([real, *args]))
rec = open(os.environ["RELAY_RECORD"], "ab", buffering=0)
p = subprocess.Popen([real, *args], stdin=subprocess.PIPE)


def pump():
    src = sys.stdin.buffer.raw
    while True:
        b = src.read(65536)
        if not b:
            break
        rec.write(b)
        try:
            p.stdin.write(b)
            p.stdin.flush()
        except OSError:
            break
    try:
        p.stdin.close()
    except OSError:
        pass


threading.Thread(target=pump, daemon=True).start()
sys.exit(p.wait())
