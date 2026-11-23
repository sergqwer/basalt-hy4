import os
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
fs = api.list_repo_files("jesdga/basalt-kld-bases", repo_type="dataset")
print(fs)
for f in fs:
    if "hy4" in f.lower() or f.endswith(".md") or "sha" in f.lower():
        p = hf_hub_download("jesdga/basalt-kld-bases", f, repo_type="dataset", local_dir=r"D:\Projects\hy4-data\gptq\kld")
        print("OK", f, os.path.getsize(p))
