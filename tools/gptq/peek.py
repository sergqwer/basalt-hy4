import pyarrow.parquet as pq, json, sys, glob
def show(p, n=1):
    t = pq.read_table(p)
    print("==", p, t.num_rows, t.schema.names)
    r = t.slice(0, n).to_pylist()[0]
    s = json.dumps(r, ensure_ascii=False, default=str)
    print(s[:2500])
for p in glob.glob("**/*.parquet", recursive=True):
    if "fineweb-edu" in p or "wikipedia" in p: 
        t = pq.read_metadata(p); print("==", p, t.num_rows); continue
    show(p)
for p in glob.glob("**/*.json", recursive=True):
    if "Berkeley" in p: continue
    d = json.load(open(p, encoding="utf-8"))
    print("==", p, type(d), len(d)); print(json.dumps(d[0], ensure_ascii=False)[:2000])
