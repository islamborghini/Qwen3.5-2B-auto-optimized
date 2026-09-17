"""Lean final session (budget-constrained): same-session dev + held-out evaluation of the officially frozen config
(custom_k0 = spec_k 0 + compile_blocks, accepted by the pre-registered gate) AND the speculative config custom_k3
(rejected by that gate by 1.3% on one prompt; reported separately), against vLLM+MTP k=3; then IFEval for both and HF."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
os.environ["QWEN35_FROZEN"] = '{"compile_blocks": true}'
def run(args):
    print(">>>", " ".join(args), flush=True); return subprocess.run([sys.executable] + args).returncode
run([os.path.join(here, "evaluate.py"), "--engines", "custom_k0,custom_k3,vllm_mtp3", "--stage", "full", "--tag", "final_full"])
run([os.path.join(here, "evaluate.py"), "--engines", "custom_k0,custom_k3,vllm_mtp3", "--stage", "heldout", "--split", "heldout", "--tag", "final_heldout"])
run([os.path.join(here, "ifeval.py"), "--engine", "custom_k0", "--limit", "100"])
run([os.path.join(here, "ifeval.py"), "--engine", "custom_k3", "--limit", "100"])
run([os.path.join(here, "ifeval.py"), "--engine", "hf", "--limit", "100"])
