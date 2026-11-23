import os
from huggingface_hub import hf_hub_download
OUT = r"D:\Projects\hy4-data\gptq\src"
want = {"Salesforce/xlam-function-calling-60k": ["xlam_function_calling_60k.json"],
        "bigcode/the-stack-smol": ["data/%s/data.json" % l for l in ["javascript","typescript","html","css","rust","c++","cuda","python","powershell","shell","c"]]}
for repo, files in want.items():
    for f in files:
        try:
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(OUT, repo.replace("/", "__")))
            print("OK", repo, f, os.path.getsize(p) >> 20, flush=True)
        except Exception as e:
            print("FAIL", repo, f, str(e).splitlines()[0][:120], flush=True)
