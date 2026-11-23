import os, sys
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
from huggingface_hub import hf_hub_download, HfApi
OUT = r"D:\Projects\hy4-data\gptq\src"
api = HfApi()
want = {
 "nebius/SWE-agent-trajectories": ["data/train-00000-of-00012.parquet"],
 "SWE-Gym/OpenHands-SFT-Trajectories": ["data/train.success.oss-00000-of-00001.parquet"],
 "NousResearch/hermes-function-calling-v1": ["func-calling.json", "func-calling-singleturn.json"],
 "Team-ACE/ToolACE": ["data.json"],
 "glaiveai/glaive-function-calling-v2": ["glaive-function-calling-v2.json"],
 "Salesforce/xlam-function-calling-60k": ["xlam_function_calling_60k.json"],
 "bigcode/the-stack-smol": ["data/%s/data.json" % l for l in ["javascript","typescript","html","css","rust","c++","cuda","python","powershell","shell","c"]],
 "HuggingFaceFW/fineweb-2": ["data/ukr_Cyrl/test/000_00000.parquet","data/cmn_Hani/test/000_00000.parquet","data/rus_Cyrl/test/000_00000.parquet"],
 "wikimedia/wikipedia": ["20231101.uk/train-00000-of-00006.parquet"],
 "open-r1/OpenR1-Math-220k": ["all/default-00000-of-00010.parquet"],
 "CohereLabs/Global-MMLU": ["uk/test-00000-of-00001.parquet", "uk/dev-00000-of-00001.parquet"],
}
# fineweb-edu: one small shard from sample/10BT
fe = [s for s in api.list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset") if s.startswith("sample/10BT/")]
want["HuggingFaceFW/fineweb-edu"] = sorted(fe)[:1]
wk = [s for s in api.list_repo_files("wikimedia/wikipedia", repo_type="dataset") if s.startswith("20231101.uk/")]
want["wikimedia/wikipedia"] = sorted(wk)[:1]
bf = [s for s in api.list_repo_files("gorilla-llm/Berkeley-Function-Calling-Leaderboard", repo_type="dataset") if s.endswith(".json")]
want["gorilla-llm/Berkeley-Function-Calling-Leaderboard"] = bf
for repo, files in want.items():
    for f in files:
        try:
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(OUT, repo.replace("/", "__")))
            print("OK", repo, f, os.path.getsize(p) >> 20, "MiB", flush=True)
        except Exception as e:
            print("FAIL", repo, f, str(e).splitlines()[0][:160], flush=True)
