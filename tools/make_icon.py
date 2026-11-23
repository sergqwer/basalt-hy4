"""Regenerate assets/basalt.ico from assets/basalt-mark.svg.

Usage: python tools/make_icon.py   (needs `resvg` on PATH: cargo install resvg)

Each size is rendered from the SVG (no downscaling of one bitmap) and stored as
a PNG entry, which Windows reads since Vista; rc.exe and windres copy it as is.
"""
import os
import struct
import subprocess
import sys
import tempfile

SIZES = [16, 24, 32, 48, 64, 128, 256]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG = os.path.join(ROOT, "assets", "basalt-mark.svg")
ICO = os.path.join(ROOT, "assets", "basalt.ico")


def render(size, tmp):
    out = os.path.join(tmp, f"{size}.png")
    subprocess.run(["resvg", "-w", str(size), "-h", str(size), SVG, out], check=True)
    with open(out, "rb") as f:
        return f.read()


def pack(pngs):
    head = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    dirs, blobs = b"", b""
    for size, data in pngs:
        dim = 0 if size >= 256 else size  # 0 means 256 in an ICONDIRENTRY
        dirs += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
        blobs += data
    return head + dirs + blobs


def main():
    with tempfile.TemporaryDirectory() as tmp:
        pngs = [(s, render(s, tmp)) for s in SIZES]
    with open(ICO, "wb") as f:
        f.write(pack(pngs))
    print(f"wrote {ICO} ({len(SIZES)} sizes)")


if __name__ == "__main__":
    sys.exit(main())
