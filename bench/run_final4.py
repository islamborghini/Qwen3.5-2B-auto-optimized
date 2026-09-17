"""Final session #4: frozen config read from results/frozen_config.json (written by the coordinator from the opt6 ledger).
Same-session dev + held-out eval vs vLLM+MTP k=3 (plus the k3 variant for reporting), then IFEval on the frozen config."""
import json, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
fz = json.load(open(os.path.join(here, "frozen_config.json")))
kw = dict(fz["kwargs"]); k = kw.pop("spec_k"); os.environ["QWEN35_FROZEN"] = json.dumps(kw)
eng = f"custom_k{k}"; other = f"custom_k{3 if k != 3 else 2}"
def run(args):
    print(">>>", " ".join(args), flush=True); return subprocess.run([sys.executable] + args).returncode
run([os.path.join(here, "evaluate.py"), "--engines", f"{eng},{other},vllm_mtp3", "--stage", "full", "--tag", "final_full"])
run([os.path.join(here, "evaluate.py"), "--engines", f"{eng},{other},vllm_mtp3", "--stage", "heldout", "--split", "heldout", "--tag", "final_heldout"])
run([os.path.join(here, "ifeval.py"), "--engine", eng, "--limit", "100"])
