"""Demo worker (runs as a subprocess inside the Modal container; no modal import here)."""
import os, sys, time


def _run(prompt: str, n_out: int, skip_base: bool):
    sys.path.insert(0, "/work")
    import warnings; warnings.filterwarnings("ignore")
    import torch
    from bench.common import model_path, load_hf, hf_greedy
    from transformers import AutoTokenizer
    from qwen35_fast.engine import Engine

    path = model_path(); tok = AutoTokenizer.from_pretrained(path)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    P = lambda *a: print(*a, flush=True)
    P(f"\nGPU: {torch.cuda.get_device_name(0)}   |   model: Qwen/Qwen3.5-2B (bf16, unchanged)   |   prompt: {len(ids)} tokens\n")

    # ---------------- optimized engine, streamed ----------------
    P("─" * 24, "OPTIMIZED ENGINE  (qwen35_fast: CUDA graph + fused kernels + exact MTP speculation)", "─" * 24)
    t = time.perf_counter(); eng = Engine(path, spec_k=2, compile_blocks=True, fused_gdn=True); eng.ensure_graph()
    eng.generate(ids[:8], 4, ignore_eos=True)
    P(f"[ready in {time.perf_counter()-t:.0f} s: load weights + compile + capture graph; one-time per process]\n")
    torch.cuda.synchronize(); t0 = time.perf_counter(); ft = {}
    def on_first(g): ft["tok"] = g.item(); ft["t"] = time.perf_counter()
    eng.prefill(ids, on_first_token=on_first)
    out = [ft["tok"]]; T = eng.spec_k + 1
    host = torch.empty(T + 1, dtype=torch.long, pin_memory=True); ev = torch.cuda.Event(); done = ft["tok"] in eos
    printed = 0
    def flush_text(final=False):
        nonlocal printed
        s = tok.decode(out[printed:])
        if final or "\n" in s or len(out) - printed >= 24:   # Modal shows stdout per line; flush on newlines / small chunks
            sys.stdout.write(s); sys.stdout.flush(); printed = len(out)
    while len(out) < n_out and not done:
        eng.step(); host[:T].copy_(eng.step_tokens, non_blocking=True); host[T].copy_(eng.step_acc, non_blocking=True)
        ev.record(); ev.synchronize()
        for tkn in host[: int(host[T]) + 1].tolist():
            if tkn in eos or len(out) >= n_out: done = True; break
            out.append(tkn)
        flush_text()
    t_end = time.perf_counter(); flush_text(final=True); n_eng = len(out)
    tps_eng = (n_eng - 1) / (t_end - ft["t"]); tot_eng = t_end - t0; ttft_eng = ft["t"] - t0
    P(f"\n\n[engine] {n_eng} tokens in {tot_eng:.2f} s  ->  {tps_eng:.0f} tokens/s decode, first token after {1000*ttft_eng:.0f} ms\n")
    eng.close(); del eng; torch.cuda.empty_cache()

    if skip_base:
        return
    # ---------------- base model, benchmark routine ----------------
    P("─" * 24, "BASE MODEL  (HF transformers eager, same weights, same GPU)", "─" * 24)
    m = load_hf()
    hf_greedy(m, ids[:8], 4)   # warmup (Triton JIT), excluded from timing as in the benchmark
    P(f"[generating {n_eng} tokens; HF eager is CPU/launch-bound at roughly 50 tokens/s, so this takes a few seconds...]\n")
    gen, ttft, tps, tot = hf_greedy(m, ids, n_eng)
    P(tok.decode(gen, skip_special_tokens=True))
    P(f"\n[base] {len(gen)} tokens in {tot:.2f} s  ->  {tps:.0f} tokens/s decode, first token after {1000*ttft:.0f} ms\n")

    P("─" * 30, "SUMMARY", "─" * 30)
    P(f"{'':22}{'base model':>14}{'optimized':>14}{'speedup':>10}")
    P(f"{'decode tokens/s':22}{tps:>14.0f}{tps_eng:>14.0f}{tps_eng/tps:>9.1f}x")
    P(f"{'time to first token':22}{1000*ttft:>11.0f} ms{1000*ttft_eng:>11.0f} ms{ttft/ttft_eng:>9.1f}x")
    P(f"{'total for the answer':22}{tot:>12.2f} s{tot_eng:>12.2f} s{tot/tot_eng:>9.1f}x")
    P("(same bf16 weights, greedy decoding; outputs differ only where bf16 rounding flips a near-tie)\n")




if __name__ == "__main__":
    _run(sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1")
