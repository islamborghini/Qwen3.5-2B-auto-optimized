"""Baselines, each group in its own subprocess so a native crash cannot take the others down."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
for engines, tag in [("hf_eager,vllm_plain,vllm_mtp1,vllm_mtp2,vllm_mtp3", "baselines_full"), ("hf_compile,hf_eager", "hfcompile_full")]:
    rc = subprocess.run([sys.executable, os.path.join(here, "evaluate.py"), "--engines", engines, "--stage", "full" if tag == "baselines_full" else "screen", "--tag", tag]).returncode
    print("group", tag, "exit", rc, flush=True)
