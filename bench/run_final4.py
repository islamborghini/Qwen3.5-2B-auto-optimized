"""Final session #4: frozen config read from results/frozen_config.json (written by the coordinator from the opt6 ledger).
Same-session dev + held-out eval vs vLLM+MTP k=3 (plus the k3 variant for reporting), then IFEval on the frozen config."""
import json, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
fz = json.load(open(os.path.join(here, "frozen_config.json")))
kw = dict(fz["kwargs"]); k = kw.pop("spec_k"); os.environ["QWEN35_FROZEN"] = json.dumps(kw)
eng = f"custom_k{k}"
def run(args):
    print(">>>", " ".join(args), flush=True); return subprocess.run([sys.executable] + args).returncode
# vLLM first, frozen custom engine last (results are checkpointed per engine, so a teardown fault cannot lose the comparator)
run([os.path.join(here, "evaluate.py"), "--engines", f"vllm_mtp3,{eng}", "--stage", "full", "--tag", "final_full"])
run([os.path.join(here, "evaluate.py"), "--engines", f"vllm_mtp3,{eng}", "--stage", "heldout", "--split", "heldout", "--tag", "final_heldout"])
run([os.path.join(here, "ifeval.py"), "--engine", eng, "--limit", "100"])
