"""Live demo on a Modal H100: streams one prompt through the base model (HF transformers) and the optimized engine.
    modal run bench/demo.py --prompt "Explain how a CPU cache works" --n-out 256
"""
import os, sys, time
import modal

app = modal.App("qwen35-demo")
vol = modal.Volume.from_name("qwen35-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.29.0", "flash-linear-attention==0.5.2", "accelerate==1.15.0")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("qwen35_fast", "/work/qwen35_fast").add_local_dir("bench", "/work/bench")
)


@app.function(gpu="H100", cpu=4.0, image=image, volumes={"/hf": vol}, timeout=1200)
def demo(prompt: str, n_out: int, skip_base: bool):
    sys.path.insert(0, "/work"); os.environ["OUT_DIR"] = "/tmp/out"
    import torch
    from bench.common import model_path, load_hf
    from transformers import AutoTokenizer
    from qwen35_fast.engine import Engine

    path = model_path(); tok = AutoTokenizer.from_pretrained(path)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tok(ids, add_special_tokens=False)["input_ids"]
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    print(f"\nPrompt ({len(ids)} tokens): {prompt}\n", flush=True)

    def stream_print(t):
        sys.stdout.write(t); sys.stdout.flush()

    if not skip_base:
        print("=" * 30, "BASE MODEL: HF transformers eager (H100)", "=" * 30, flush=True)
        m = load_hf()
        from transformers.generation.streamers import BaseStreamer

        class S(BaseStreamer):
            def __init__(s): s.t = []; s.first = True; s.buf = []
            def put(s, v):
                if s.first: s.first = False; return           # prompt
                s.t.append(time.perf_counter()); s.buf.append(int(v[0]))
                if len(s.buf) % 16 == 0: stream_print(tok.decode(s.buf[-16:]))   # chunked: per-token flushes over the log stream slow the CPU-bound HF loop
            def end(s): pass
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():   # one-time warmup (Triton JIT of the fla kernels), excluded like in the benchmark
            m.generate(input_ids=x[:, :8], attention_mask=torch.ones_like(x[:, :8]), max_new_tokens=4, do_sample=False)
        s = S(); torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n_out, do_sample=False, streamer=s, eos_token_id=list(eos))
        stream_print(tok.decode(s.buf[len(s.buf) // 16 * 16:])); n = len(s.t); tps = (n - 1) / (s.t[-1] - s.t[0]) if n > 1 else 0
        print(f"\n\n[base] {n} tokens | time to first token {1000*(s.t[0]-t0):.0f} ms | decode {tps:.0f} tokens/s | total {s.t[-1]-t0:.2f} s\n", flush=True)
        del m; torch.cuda.empty_cache()

    print("=" * 30, "OPTIMIZED ENGINE: qwen35_fast (same weights, same H100)", "=" * 30, flush=True)
    t = time.perf_counter(); eng = Engine(path, spec_k=2, compile_blocks=True, fused_gdn=True); eng.ensure_graph()
    eng.generate(ids[:8], 4, ignore_eos=True)   # warm the compiled kernels
    print(f"[engine ready in {time.perf_counter()-t:.0f} s: weights + torch.compile + CUDA-graph capture, one-time]\n", flush=True)
    torch.cuda.synchronize(); t0 = time.perf_counter(); ft = {}
    def on_first(g): ft["tok"] = g.item(); ft["t"] = time.perf_counter()
    eng.prefill(ids, on_first_token=on_first)
    out = [ft["tok"]]; stream_print(tok.decode(out)); T = eng.spec_k + 1
    host = torch.empty(T + 1, dtype=torch.long, pin_memory=True); ev = torch.cuda.Event(); done = ft["tok"] in eos
    while len(out) < n_out and not done:
        eng.step(); host[:T].copy_(eng.step_tokens, non_blocking=True); host[T].copy_(eng.step_acc, non_blocking=True)
        ev.record(); ev.synchronize()
        new = host[: int(host[T]) + 1].tolist()
        for tkn in new:
            if tkn in eos or len(out) >= n_out: done = True; break
            out.append(tkn)
            if len(out) % 16 == 0: stream_print(tok.decode(out[-16:]))
    t_end = time.perf_counter(); n = len(out); stream_print(tok.decode(out[n // 16 * 16:]))
    print(f"\n\n[engine] {n} tokens | time to first token {1000*(ft['t']-t0):.0f} ms | decode {(n-1)/(t_end-ft['t']):.0f} tokens/s | total {t_end-t0:.2f} s", flush=True)


@app.local_entrypoint()
def main(prompt: str = "Explain in detail how a CPU cache hierarchy works and why it matters for performance.", n_out: int = 256, skip_base: bool = False):
    demo.remote(prompt, n_out, skip_base)
