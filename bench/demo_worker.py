"""Demo worker (runs as a subprocess inside the Modal container; no modal import here).
argv: prompt n_out mode(base|engine) start_at(unix epoch seconds, 0 = now) ignore_eos(0|1)"""
import sys, time


def _run(prompt: str, n_out: int, mode: str, start_at: float, ignore_eos: bool):
    sys.path.insert(0, "/work")
    import warnings; warnings.filterwarnings("ignore")
    import torch
    from bench.common import model_path, load_hf
    from transformers import AutoTokenizer
    P = lambda *a: print(*a, flush=True)
    W = lambda s: (sys.stdout.write(s), sys.stdout.flush())

    path = model_path(); tok = AutoTokenizer.from_pretrained(path)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    title = {"base": "BASE MODEL  -  Qwen3.5-2B in HF transformers (eager)", "engine": "OPTIMIZED  -  Qwen3.5-2B in qwen35_fast (CUDA graph + fused kernels + exact MTP speculation)"}
    P("\n" + "=" * 100 + f"\n{title[mode]}\nGPU: {torch.cuda.get_device_name(0)} | bf16 weights, unchanged | greedy | prompt: {len(ids)} tokens\n" + "=" * 100)

    def wait_start():
        if start_at and start_at < time.time():
            P(f"[warn] missed the synchronized start by {time.time()-start_at:.0f} s (model loading took longer); starting now")
        if start_at > time.time():
            P(f"[ready] waiting for the synchronized start ({start_at - time.time():.0f} s)...")
            while start_at - time.time() > 0.5:
                time.sleep(0.2)
            while time.time() < start_at:
                pass
        P(f"\n>>> {prompt}\n")

    # ---- decoded-text streaming that is safe with byte-pair tokens: print the newly completed text only ----
    class Stream:
        def __init__(s): s.toks = []; s.shown = ""
        def push(s, t):
            s.toks.append(t); full = tok.decode(s.toks)
            if full.endswith("�"): return
            W(full[len(s.shown):]); s.shown = full

    if mode == "base":
        m = load_hf()
        from transformers.generation.streamers import BaseStreamer

        class S(BaseStreamer):
            def __init__(s): s.t = []; s.first = True; s.st = Stream()
            def put(s, v):
                if s.first: s.first = False; return
                s.t.append(time.perf_counter()); s.st.push(int(v[0]))
            def end(s): pass
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():   # warm on the real prompt (Triton JIT/autotune), excluded like in the benchmark
            m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=4, do_sample=False)
        P(f"[ready] model loaded and warmed up")
        wait_start(); s = S(); torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n_out, do_sample=False, streamer=s,
                       **({"min_new_tokens": n_out} if ignore_eos else {"eos_token_id": list(eos)}))
        s.st.finish(); n = len(s.t); tps = (n - 1) / (s.t[-1] - s.t[0]) if n > 1 else 0
        P(f"\n\n[base]  {n} tokens in {s.t[-1]-t0:.2f} s   |   {tps:.0f} tokens/s decode   |   first token after {1000*(s.t[0]-t0):.0f} ms")
        P("[base]  (HF eager is CPU-bound; it measured 50 tokens/s under the controlled benchmark, host CPUs vary)\n")
        return

    from qwen35_fast.engine import Engine
    max_len = (len(ids) + n_out + 512 + 255) // 256 * 256
    t = time.perf_counter(); eng = Engine(path, spec_k=2, compile_blocks=True, fused_gdn=True, max_len=max_len); eng.ensure_graph()
    eng.generate(ids, 4, ignore_eos=True)
    P(f"[ready] weights loaded, kernels compiled, CUDA graph captured ({time.perf_counter()-t:.0f} s, one-time)")
    wait_start(); st = Stream()
    torch.cuda.synchronize(); t0 = time.perf_counter(); ft = {}
    def on_first(g): ft["tok"] = g.item(); ft["t"] = time.perf_counter()
    eng.prefill(ids, on_first_token=on_first)
    out = [ft["tok"]]; st.push(ft["tok"]); T = eng.spec_k + 1
    host = torch.empty(T + 1, dtype=torch.long, pin_memory=True); ev = torch.cuda.Event(); done = (ft["tok"] in eos) and not ignore_eos
    while len(out) < n_out and not done:
        eng.step(); host[:T].copy_(eng.step_tokens, non_blocking=True); host[T].copy_(eng.step_acc, non_blocking=True)
        ev.record(); ev.synchronize()
        for tkn in host[: int(host[T]) + 1].tolist():
            if (tkn in eos and not ignore_eos) or len(out) >= n_out: done = True; break
            out.append(tkn); st.push(tkn)
    t_end = time.perf_counter(); n = len(out); st.finish()
    P(f"\n\n[engine]  {n} tokens in {t_end-t0:.2f} s   |   {(n-1)/(t_end-ft['t']):.0f} tokens/s decode   |   first token after {1000*(ft['t']-t0):.0f} ms\n")
    eng.close()


if __name__ == "__main__":
    _run(sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5] == "1")
