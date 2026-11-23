"""gen_f.py - bucket F: OpenR1-Math problems (+ a few science/logic ones) answered by Hy4 itself (thinking on), through
my server. Stops at --minutes. Writes corpus/pools/F.jsonl rows like build_pools (body = the conversation with the
model's own answer, reasoning_content kept, so the server renders <think> for it)."""
import argparse, hashlib, json, random, time, urllib.request
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=45)
ap.add_argument("--port", type=int, default=8320)
ap.add_argument("--max-tokens", type=int, default=6000)
a = ap.parse_args()
G = r"D:\Projects\hy4-data\gptq"
rows = pq.read_table(G + r"\src\open-r1__OpenR1-Math-220k\all\default-00000-of-00010.parquet",
                     columns=["problem", "uuid", "problem_type"]).to_pylist()
rng = random.Random(1234)
rng.shuffle(rows)
UK = ["Розв'яжи задачу й поясни хід міркувань українською.", "Розв'яжи це. Відповідь дай українською, остаточну відповідь у \\boxed{}."]
EN = ["Solve the problem. Put the final answer in \\boxed{}.", "Solve this step by step and give the final answer in \\boxed{}."]
LOGIC = [
    "Three boxes are labeled 'apples', 'oranges' and 'mixed', and every label is wrong. You may draw one fruit from one box. How do you relabel all boxes correctly? Explain.",
    "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost? Show the reasoning.",
    "Чому небо вдень блакитне, а на заході сонця червоне? Поясни фізику розсіювання Релея кількісно.",
    "Estimate how many piano tuners work in a city of 3 million people. State every assumption.",
    "Explain why the time complexity of building a binary heap bottom-up is O(n), with the summation.",
    "Скільки існує способів розставити 8 ферзів на шахівниці так, щоб вони не били один одного? Опиши алгоритм пошуку з поверненням і оціни його складність.",
    "A reaction has activation energy 50 kJ/mol. By what factor does its rate change from 25 C to 35 C? Use the Arrhenius equation.",
    "In a room of 23 people, what is the probability that two share a birthday? Derive it.",
]
items = [("logic-%d" % i, p) for i, p in enumerate(LOGIC)] + [("openr1-" + r["uuid"], r["problem"]) for r in rows[:400]]
out = open(G + r"\corpus\pools\F.jsonl", "a", encoding="utf-8")
done = set()
try:
    for l in open(G + r"\corpus\pools\F.jsonl", encoding="utf-8"):
        done.add(json.loads(l)["source"])
except FileNotFoundError:
    pass
t_end = time.time() + a.minutes * 60
n = 0
for sid, prob in items:
    if time.time() > t_end:
        break
    if sid in done:
        continue
    uk = rng.random() < 0.25
    msg = prob + "\n\n" + rng.choice(UK if uk else EN)
    body = {"model": "hy4-flash-nvfp4", "messages": [{"role": "user", "content": msg}], "max_tokens": a.max_tokens}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % a.port, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    ch = r["choices"][0]
    m = ch["message"]
    if ch.get("finish_reason") != "stop":
        print("skip (finish %s) %s" % (ch.get("finish_reason"), sid), flush=True)
        continue
    asst = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        asst["reasoning_content"] = m["reasoning_content"]
    split = "eval" if int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 10 == 0 else "train"
    u = r.get("usage", {})
    row = {"id": "F-%s" % sid, "bucket": "F", "lang": "uk" if uk else "en", "source": sid, "split": split,
           "est_tokens": u.get("prompt_tokens", 0) + u.get("completion_tokens", 0), "api": "openai",
           "body": {"messages": [{"role": "user", "content": msg}, asst]}}
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    n += 1
    print("%s %s %s tok %.0fs" % (time.strftime("%H:%M"), sid, row["est_tokens"], time.time() - t0), flush=True)
print("F done:", n)
