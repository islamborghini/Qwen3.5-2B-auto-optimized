"""Side-by-side live demo on Modal H100s. Open two terminals and run, within a few seconds of each other:

    left : modal run bench/demo.py --mode base   --start-in 120 --prompt "write me an html page for a hotel reservation"
    right: modal run bench/demo.py --mode engine --start-in 120 --prompt "write me an html page for a hotel reservation"

Each loads its model (base ~20 s, engine ~70 s incl. compile), then both start generating at the same instant
(120 s after launch) and stream tokens live. Single run: omit --start-in. Long runs: --n-out 10000 (when the model ends its answer, the same follow-up
prompt is sent as a new turn, in both panes identically, until the target is reached; --follow-ups N caps the turns).
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
def demo(prompt: str, n_out: int, mode: str, start_at: float, follow_ups: int, follow_up: str):
    # Runs in a subprocess: the Modal function process's I/O threads contend for the GIL with HF's Python-bound loop.
    import subprocess
    env = dict(os.environ, PYTHONPATH="/work", OUT_DIR="/tmp/out")
    subprocess.run([sys.executable, "-u", "/work/bench/demo_worker.py", prompt, str(n_out), mode, str(start_at), str(follow_ups), follow_up], env=env, check=False)


@app.local_entrypoint()
def main(prompt: str = "Explain how a CPU cache hierarchy works and why it matters for performance.", n_out: int = 256,
         mode: str = "engine", start_in: float = 0, follow_ups: int = 3,
         follow_up: str = "Continue with the next chapters, in full and in the same format. Do not repeat earlier chapters."):
    """mode: 'engine' (optimized) or 'base' (HF transformers). start_in: seconds from now at which generation starts,
    so two terminals launched together begin at the same instant (models take different times to load)."""
    import time
    demo.remote(prompt, n_out, mode, time.time() + start_in if start_in else 0, follow_ups, follow_up)
