"""Final session: bounded optimization loop (dev screen) -> freeze incumbent -> same-session full dev eval vs strongest
original engines -> held-out eval -> IFEval (original + optimized). Each stage in a subprocess; results checkpointed."""
import json, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("OUT_DIR", "results")
def run(args):
    print(">>>", " ".join(args), flush=True)
    return subprocess.run([sys.executable] + args).returncode
run([os.path.join(here, "optimize.py"), "--max_candidates", "3", "--max_minutes", "15"])
led = json.load(open(os.path.join(OUT, "opt_ledger.json")))
inc = led["incumbent"]; kw = led["candidates"][inc]["kwargs"]
json.dump({"incumbent": inc, "kwargs": kw}, open(os.path.join(OUT, "frozen_config.json"), "w"))
print("FROZEN:", inc, kw, flush=True)
os.environ["QWEN35_FROZEN"] = json.dumps(kw)
eng = f"custom_k{kw.get('spec_k', 0)}"
run([os.path.join(here, "evaluate.py"), "--engines", f"{eng},vllm_mtp3", "--stage", "full", "--tag", "final_full"])
run([os.path.join(here, "evaluate.py"), "--engines", f"{eng},vllm_mtp3", "--stage", "heldout", "--split", "heldout", "--tag", "final_heldout"])
run([os.path.join(here, "ifeval.py"), "--engine", eng, "--limit", "100"])
run([os.path.join(here, "ifeval.py"), "--engine", "hf", "--limit", "100"])
