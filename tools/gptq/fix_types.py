"""fix_types.py CORPUS DUMPLOG OUT - the corpus rows a dump pass rejected for Python type names in tool schemas
(xlam/glaive write "int", "float", "str", "list", "dict", "bool"), with the types mapped to JSON Schema names."""
import json, sys

MAP = {"int": "integer", "float": "number", "str": "string", "string": "string", "bool": "boolean", "boolean": "boolean",
       "list": "array", "array": "array", "dict": "object", "object": "object", "integer": "integer", "number": "number",
       "null": "null", "tuple": "array", "any": None, "none": "null"}


def fix(s):
    if isinstance(s, dict):
        out = {}
        for k, v in s.items():
            if k == "type" and isinstance(v, str):
                t = v.strip().lower()
                base = t.split("[")[0].split(",")[0].strip()          # "list[int]", "List[str]", "str, optional"
                m = MAP.get(base, "string")
                if m is not None:
                    out[k] = m
            elif k == "type" and isinstance(v, list):
                out[k] = [MAP.get(str(x).lower(), "string") or "string" for x in v]
            else:
                out[k] = fix(v)
        return out
    if isinstance(s, list):
        return [fix(x) for x in s]
    return s


corpus, log, out = sys.argv[1:4]
if log == "--all":        # every row, before a dump pass
    bad = {json.loads(l)["id"] for l in open(corpus, encoding="utf-8")}
else:
    bad = {json.loads(l)["id"] for l in open(log, encoding="utf-8") if json.loads(l).get("err")}
n = 0
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for l in open(corpus, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in bad:
            for t in r["body"].get("tools", []):
                fn = t.get("function", t)
                if "parameters" in fn:
                    fn["parameters"] = fix(fn["parameters"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
print("rejected rows", len(bad), "rewritten", n)
