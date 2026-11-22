"""Hand-built PDFs for the document goldens: text layouts pypdf reads, compressed streams, a ToUnicode font, a scan."""
import io
import zlib


def build(pages, compress=False, fonts=None):
    """pages: list of (content bytes, xobjects dict name->stream obj index or None). Returns PDF bytes."""
    objs = []

    def add(b):
        objs.append(b)
        return len(objs)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    extra = {}
    for name, f in (fonts or {}).items():
        extra[name] = f(add)
    kids = []
    pages_idx = len(objs) + 1 + 2 * len(pages) + 1
    page_objs = []
    for content, xo in pages:
        data = zlib.compress(content) if compress else content
        filt = b" /Filter /FlateDecode" if compress else b""
        c = add(b"<< /Length %d%s >>\nstream\n" % (len(data), filt) + data + b"\nendstream")
        fl = b"/F1 %d 0 R /F2 %d 0 R" % (font, font2) + b"".join(b" /%s %d 0 R" % (k.encode(), v) for k, v in extra.items())
        xs = b""
        if xo:
            xs = b" /XObject << " + b" ".join(b"/%s %d 0 R" % (k.encode(), v(add)) for k, v in xo.items()) + b" >>"
        page_objs.append((c, fl, xs))
    pages_n = len(objs) + len(page_objs) + 1
    for c, fl, xs in page_objs:
        kids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Contents %d 0 R /Resources << /Font << %s >>%s >> >>" % (pages_n, c, fl, xs)))
    assert add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))) == pages_n
    root = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_n)
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(out.tell())
        out.write(b"%d 0 obj\n" % i + o + b"\nendobj\n")
    x = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for o in offs:
        out.write(b"%010d 00000 n \n" % o)
    out.write(b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, root, x))
    return out.getvalue()


def tounicode_font(add):
    cmap = (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n1 begincodespacerange <0000> <FFFF> "
            b"endcodespacerange\n3 beginbfchar\n<0001> <0041>\n<0002> <0431>\n<0003> <0020>\nendbfchar\n"
            b"1 beginbfrange\n<0010> <0012> <0061>\nendbfrange\nendcmap end end")
    cm = add(b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream")
    desc = add(b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 500 >>")
    return add(b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>" % (desc, cm))


def image(w, h):
    def make(add):
        px = bytes((x * 7 + y * 3) % 256 for y in range(h) for x in range(w) for _ in range(3))
        d = zlib.compress(px)
        return add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 "
                   b"/Filter /FlateDecode /Length %d >>\nstream\n" % (w, h, len(d)) + d + b"\nendstream")
    return make


def corpus():
    one = b"BT /F1 12 Tf 72 720 Td (Hello PDF world) Tj ET"
    lines = (b"BT /F1 12 Tf 72 720 Td (First line) Tj 0 -14 Td (Second line) Tj T* (third) Tj ET\n"
             b"BT /F2 10 Tf 72 600 Td [(Kern)-200(ed) 30(text)] TJ ET")
    long_text = b"BT /F1 11 Tf 14 TL 72 740 Td " + b" ".join(b"(Line number %d of a longer page with words.) '" % i for i in range(30)) + b" ET"
    cid = b"BT /F3 12 Tf 72 700 Td <000100020003001000110012> Tj ET"
    scan = b"q 400 0 0 400 100 200 cm /Im1 Do Q"
    little = b"BT /F1 12 Tf 72 720 Td (Caption) Tj ET q 600 0 0 600 0 0 cm /Im1 Do Q"
    return {
        "one": build([(one, None)]),
        "lines": build([(lines, None)]),
        "compressed": build([(lines, None), (one, None)], compress=True),
        "long": build([(long_text, None)]),
        "cid": build([(cid, None)], fonts={"F3": tounicode_font}),
        "scan": build([(scan, {"Im1": image(80, 80)})]),
        "picture": build([(little, {"Im1": image(520, 520)})], compress=True),
        "empty_page": build([(b"", None), (one, None)]),
        "many": build([(b"BT /F1 12 Tf 72 720 Td (Page %d) Tj ET" % i, None) for i in range(1, 8)]),
    }
