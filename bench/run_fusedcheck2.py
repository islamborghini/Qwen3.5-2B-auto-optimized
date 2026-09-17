"""Reproduce the opt6 fault: the optimize loop itself (paired incumbent alive across dynamo.reset) under CUDA_LAUNCH_BLOCKING=1."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1", QWEN35_CANDIDATES="k2_compile,k2_compile_fused,k3_compile_fused")
sys.exit(subprocess.run([sys.executable, os.path.join(here, "optimize.py"), "--max_candidates", "3", "--max_minutes", "12"], env=env).returncode)
