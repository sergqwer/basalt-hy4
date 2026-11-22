"""Image goldens: Vision.normalize (serve/engine/vision.py) on inputs Pillow builds -> tests/golden/images.json."""
import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from PIL import Image  # noqa: E402
from serve.engine.vision import Vision  # noqa: E402


def img(mode, size=(37, 23)):
    im = Image.new(mode, size)
    px = im.load()
    for y in range(size[1]):
        for x in range(size[0]):
            v = (x * 7 + y * 13) % 256
            px[x, y] = {"RGB": (v, 255 - v, (x * y) % 256), "RGBA": (v, 50, 200, (x * 9) % 256), "L": v,
                        "LA": (v, (y * 11) % 256), "P": v % 16}[mode]
    return im


def enc(im, fmt, **kw):
    b = io.BytesIO()
    im.save(b, format=fmt, **kw)
    return b.getvalue()


cases = {"png": enc(img("RGB"), "PNG"), "gif": enc(img("P"), "GIF"), "bmp": enc(img("RGB"), "BMP"),
         "jpeg": enc(img("RGB"), "JPEG", quality=90),
         "webp_lossless_rgba": enc(img("RGBA"), "WEBP", lossless=True),
         "webp_lossless_rgb": enc(img("RGB"), "WEBP", lossless=True),
         "tiff_rgb": enc(img("RGB"), "TIFF"), "tiff_l": enc(img("L"), "TIFF"), "tiff_rgba": enc(img("RGBA"), "TIFF"),
         "tga": enc(img("RGB"), "TGA"), "ppm": enc(img("RGB"), "PPM"), "pgm_la_png": enc(img("LA"), "PNG"),
         "junk": b"not an image at all"}
for o in (3, 6, 8):
    ex = Image.Exif()
    ex[0x0112] = o
    cases[f"jpeg_exif{o}"] = enc(img("RGB"), "JPEG", quality=95, exif=ex.tobytes())
out = []
for name, data in cases.items():
    try:
        r = {"out": base64.b64encode(Vision.normalize(data)).decode()}
    except ValueError as e:
        r = {"error": str(e)}
    out.append({"name": name, "in": base64.b64encode(data).decode(), **r})
Path(__file__).with_name("images.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print(len(out))
