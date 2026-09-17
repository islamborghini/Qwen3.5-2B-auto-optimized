"""Live demo on a Modal H100.

    modal run bench/demo.py --prompt "write me an html page for a hotel reservation"
    modal run bench/demo.py --prompt "..." --n-out 512 --skip-base

Streams the optimized engine's answer token by token, then runs the base model (HF transformers) on the same prompt
with the exact benchmark timing routine, and prints a summary.
"""
import os, sys
import modal

app = modal.App("qwen35-demo")
vol = modal.Volume.from_name("qwen35-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0", "flash-linear-attention==0.5.2", "transformers==5.17.0", "accelerate==1.15.0", "safetensors")
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error", "HF_HUB_VERBOSITY": "error",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("qwen35_fast", "/work/qwen35_fast").add_local_dir("bench", "/work/bench")
)


@app.function(gpu="H100", cpu=4.0, memory=32768, image=image, volumes={"/hf": vol}, timeout=1200)
def demo(prompt: str, n_out: int, skip_base: bool):
    # Run in a subprocess: the Modal function process's I/O threads contend for the GIL with HF's Python-bound
    # decode loop (14 tok/s in-process vs 50 tok/s in a subprocess); the benchmark used a subprocess too.
    import subprocess
    env = dict(os.environ, PYTHONPATH="/work", OUT_DIR="/tmp/out")
    subprocess.run([sys.executable, "-u", "/work/bench/demo_worker.py", prompt, str(n_out), "1" if skip_base else "0"], env=env, check=False)


@app.local_entrypoint()
def main(prompt: str = "Explain how a CPU cache hierarchy works and why it matters for performance.", n_out: int = 256, skip_base: bool = False):
    demo.remote(prompt, n_out, skip_base)

