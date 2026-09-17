"""Final session #3: frozen config = k2_compile (lean, accepted in opt5). Same-session dev + held-out vs vLLM+MTP k=3,
plus custom_k3 for reporting; IFEval for custom_k2 (HF reference already in results/ifeval_hf.json)."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
os.environ["QWEN35_FROZEN"] = '{"compile_blocks": true, "lean": true}'
def run(args):
    print(">>>", " ".join(args), flush=True); return subprocess.run([sys.executable] + args).returncode
run([os.path.join(here, "evaluate.py"), "--engines", "custom_k2,custom_k3,vllm_mtp3", "--stage", "full", "--tag", "final_full"])
run([os.path.join(here, "evaluate.py"), "--engines", "custom_k2,custom_k3,vllm_mtp3", "--stage", "heldout", "--split", "heldout", "--tag", "final_heldout"])
run([os.path.join(here, "ifeval.py"), "--engine", "custom_k2", "--limit", "100"])
