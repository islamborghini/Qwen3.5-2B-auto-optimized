"""Demo worker (subprocess inside the Modal container; no modal import).
argv: prompt n_out mode(base|engine) start_at(epoch s, 0=now) follow_ups(int) follow_up_text turn_max(int)
Multi-turn: when the model ends its answer, the same follow-up is sent as a new user turn (both panes identically)
until n_out generated tokens are reached or follow_ups is exhausted. Coherent long output, no forced continuation."""
import re, sys, time


def _run(prompt, n_out, mode, start_at, follow_ups, follow_up, turn_max):
    sys.path.insert(0, "/work")
    import warnings; warnings.filterwarnings("ignore")
    import torch
    from bench.common import model_path, load_hf
    from transformers import AutoTokenizer
    P = lambda *a: print(*a, flush=True)
    W = lambda s: (sys.stdout.write(s), sys.stdout.flush())
    path = model_path(); tok = AutoTokenizer.from_pretrained(path)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    clean = lambda s: re.sub(r"</?think>\n*", "", s)
    def encode(messages):
        text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return tok(text, add_special_tokens=False)["input_ids"]
    first_ids = encode([{"role": "user", "content": prompt}])
    title = {"base": "BASE MODEL  -  Qwen3.5-2B in HF transformers (eager)",
             "engine": "OPTIMIZED  -  Qwen3.5-2B in qwen35_fast (CUDA graph + fused kernels + exact MTP speculation)"}[mode]
    P("\n" + "=" * 100 + f"\n{title}\nGPU: {torch.cuda.get_device_name(0)} | bf16 weights, unchanged | greedy | target {n_out} tokens, up to {follow_ups} follow-up turns\n" + "=" * 100)

    class Stream:
        """Incremental detokenization, flushed at most every ~30 ms (tiny per-token writes would throttle the engine)."""
        def __init__(s): s.toks = []; s.done = 0; s.last = 0.0
        def push(s, t):
            s.toks.append(t); now = time.perf_counter()
            if now - s.last < 0.03: return
            text = tok.decode(s.toks[s.done:], skip_special_tokens=True)
            if text.endswith("\N{REPLACEMENT CHARACTER}") or text.endswith("<") or "<think" in text[-7:] or "</think" in text[-8:]: return
            W(clean(text)); s.done = len(s.toks); s.last = now
        def finish(s):
            if s.done < len(s.toks): W(clean(tok.decode(s.toks[s.done:], skip_special_tokens=True))); s.done = len(s.toks)

    if mode == "base":
        m = load_hf()
        from transformers.generation.streamers import BaseStreamer
        class S(BaseStreamer):
            def __init__(s, st): s.t = []; s.first = True; s.st = st; s.toks = []
            def put(s, v):
                if s.first: s.first = False; return
                s.t.append(time.perf_counter()); s.toks.append(int(v[0])); s.st.push(s.toks[-1])
            def end(s): pass
        def gen(ids, max_new, st):
            x = torch.tensor([ids], device="cuda"); s = S(st)
            with torch.no_grad():
                m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False, streamer=s, eos_token_id=list(eos))
            n = len(s.toks) - (1 if s.toks and s.toks[-1] in eos else 0)
            return s.toks[:n], s.t[:n]
        x = torch.tensor([first_ids], device="cuda")
        with torch.no_grad():
            m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=4, do_sample=False)   # warmup
        P("[ready] model loaded and warmed up")
    else:
        from qwen35_fast.engine import Engine
        max_len = (len(first_ids) + n_out + 128 * (follow_ups + 1) + 1023) // 256 * 256
        t = time.perf_counter(); eng = Engine(path, spec_k=2, compile_blocks=True, fused_gdn=True, max_len=max_len); eng.ensure_graph()
        eng.generate(first_ids, 4, ignore_eos=True)
        P(f"[ready] weights loaded, kernels compiled, CUDA graph captured ({time.perf_counter()-t:.0f} s, one-time)")
        def gen(ids, max_new, st):
            torch.cuda.synchronize(); ft = {}
            def on_first(g): ft["tok"] = g.item(); ft["t"] = time.perf_counter()
            eng.prefill(ids, on_first_token=on_first)
            if ft["tok"] in eos: return [], []
            out = [ft["tok"]]; ts = [ft["t"]]; st.push(ft["tok"]); T = eng.spec_k + 1
            host = torch.empty(T + 1, dtype=torch.long, pin_memory=True); ev = torch.cuda.Event(); done = False
            while len(out) < max_new and not done:
                eng.step(); host[:T].copy_(eng.step_tokens, non_blocking=True); host[T].copy_(eng.step_acc, non_blocking=True)
                ev.record(); ev.synchronize(); now = time.perf_counter()
                for tkn in host[: int(host[T]) + 1].tolist():
                    if tkn in eos or len(out) >= max_new: done = True; break
                    out.append(tkn); ts.append(now); st.push(tkn)
            return out, ts

    if start_at and start_at < time.time():
        P(f"[warn] missed the synchronized start by {time.time()-start_at:.0f} s (loading took longer); starting now")
    if start_at > time.time():
        P(f"[ready] waiting for the synchronized start ({start_at - time.time():.0f} s)...")
        while start_at - time.time() > 0.5: time.sleep(0.2)
        while time.time() < start_at: pass

    def loops(seq, block=48):   # first position where a 48-token block repeats an earlier block (greedy degeneration)
        seen = {}
        for i in range(0, len(seq) - block):
            key = tuple(seq[i:i + block])
            if key in seen and i - seen[key] >= block: return i
            seen.setdefault(key, i)
        return None
    CONT = "Continue exactly where you left off, without repeating anything."
    messages = [{"role": "user", "content": prompt}]; total = 0; t_first = None; t_last = None; turn = 0; dec_s = 0.0; all_toks = []
    t0 = time.perf_counter()
    while total < n_out and turn <= follow_ups:
        ids = encode(messages); st = Stream()
        P(f"\n>>> {prompt}\n" if turn == 0 else f"\n\n>>> [turn {turn + 1}] {messages[-1]['content']}\n")
        cap = min(turn_max, n_out - total)
        toks, ts = gen(ids, cap, st); st.finish()
        if not toks: break
        total += len(toks); t_first = t_first or ts[0]; t_last = ts[-1]; dec_s += ts[-1] - ts[0]; all_toks += toks
        # turns are capped at turn_max tokens: greedy decoding of a 2B model degenerates into repetition on very long
        # single answers (HF itself loops after ~2.9k tokens on this prompt); a fresh turn breaks the loop attractor.
        nxt = CONT if len(toks) >= cap else follow_up
        messages += [{"role": "assistant", "content": tok.decode(toks, skip_special_tokens=True)}, {"role": "user", "content": nxt}]
        turn += 1
    tag = "[base]  " if mode == "base" else "[engine]"
    P(f"\n\n{tag} {total} tokens over {turn} turn(s)   |   decode {(total - turn) / max(dec_s, 1e-9):.0f} tokens/s   |   wall time {t_last - t0:.1f} s (incl. {t_last - t0 - dec_s:.1f} s of prompt processing)   |   first token after {1000*(t_first - t0):.0f} ms")
    lp = loops(all_toks)
    P(f"{tag} repetition check: " + ("none detected" if lp is None else f"a 48-token block repeats from token {lp}"))
    if mode == "base": P("[base]   (HF eager is CPU-bound; it measured 50 tokens/s under the controlled benchmark, host CPUs vary)")
    P("")


if __name__ == "__main__":
    _run(sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7]))
