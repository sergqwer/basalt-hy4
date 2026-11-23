from huggingface_hub import HfApi
api = HfApi()
print(api.whoami()["name"])
for d in ["nebius/SWE-agent-trajectories","SWE-Gym/OpenHands-SFT-Trajectories","NousResearch/hermes-function-calling-v1","Team-ACE/ToolACE","glaiveai/glaive-function-calling-v2","Salesforce/xlam-function-calling-60k","bigcode/the-stack-smol","HuggingFaceFW/fineweb-2","wikimedia/wikipedia","HuggingFaceFW/fineweb-edu","open-r1/OpenR1-Math-220k","CohereLabs/Global-MMLU","gorilla-llm/Berkeley-Function-Calling-Leaderboard"]:
    try:
        info = api.dataset_info(d, files_metadata=False)
        files = [s.rfilename for s in info.siblings]
        print(d, "gated=", info.gated, "files", len(files), files[:6])
    except Exception as e:
        print(d, "ERR", str(e)[:150])
