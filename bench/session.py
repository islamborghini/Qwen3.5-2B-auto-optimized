"""Generic Modal H100 session runner: `modal run bench/session.py --script bench/run_custom.py [--args '...']`.
Mounts qwen35_fast/ and bench/ into the container, runs the script as __main__, returns results/ files it wrote to /out.
"""
import json, os, subprocess, sys
import modal

app = modal.App("qwen35-session")
vol = modal.Volume.from_name("qwen35-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.29.0", "flash-linear-attention==0.5.2", "hf_transfer==0.1.9", "accelerate==1.15.0",
                    "lm-eval==0.4.13", "datasets", "langdetect", "immutabledict", "nltk")
    .env({"HF_HOME": "/hf", "VLLM_CACHE_ROOT": "/hf/vllm_cache", "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("qwen35_fast", "/work/qwen35_fast")
    .add_local_dir("bench", "/work/bench")
)


@app.function(gpu="H100", cpu=4.0, memory=32768, image=image, volumes={"/hf": vol}, timeout=5400)
def run(script: str, args: str):
    os.makedirs("/out", exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/work", OUT_DIR="/out")
    p = subprocess.run([sys.executable, f"/work/{script}"] + args.split(), cwd="/work", env=env)
    vol.commit()
    files = {}
    for root, _, names in os.walk("/out"):
        for n in names:
            fp = os.path.join(root, n)
            files[os.path.relpath(fp, "/out")] = open(fp, "rb").read()
    return p.returncode, files


@app.local_entrypoint()
def main(script: str, args: str = ""):
    rc, files = run.remote(script, args)
    os.makedirs("results", exist_ok=True)
    for name, data in files.items():
        path = os.path.join("results", name); os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(data); print("saved", path)
    print("exit code", rc)
