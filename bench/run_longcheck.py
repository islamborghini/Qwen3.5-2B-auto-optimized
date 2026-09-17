"""Long-generation divergence check: HF greedy vs engine variants on the textbook prompt, 3000 tokens."""
import os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from transformers import AutoTokenizer
from qwen35_fast.engine import Engine
N = int(os.environ.get("N_TOK", "3000"))
PROMPT = "Write a complete beginner's textbook on Python programming with 15 chapters. Write every chapter in full with explanations, code examples and exercises. Do not summarize or skip chapters."
path = model_path(); tok = AutoTokenizer.from_pretrained(path)
text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
ids = tok(text, add_special_tokens=False)["input_ids"]

def loops(seq, block=64):
    """index of first position where a 64-token block exactly repeats a block seen >= 64 tokens earlier, else None."""
    seen = {}
    for i in range(0, len(seq) - block):
        key = tuple(seq[i:i + block])
        if key in seen and i - seen[key] >= block: return i
        seen.setdefault(key, i)
    return None

res = {}
hf = load_hf(); t = time.perf_counter()
x = torch.tensor([ids], device="cuda")
with torch.no_grad():
    o = hf.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=N, min_new_tokens=N, do_sample=False)
hgen = o[0, x.shape[1]:].tolist(); print(f"HF: {len(hgen)} tokens in {time.perf_counter()-t:.0f}s, loop at {loops(hgen)}", flush=True)
lg = hf_logits(hf, ids + hgen)[len(ids) - 1: len(ids) - 1 + len(hgen)].float().cpu()
top2 = lg.topk(2, -1).values; margin = (top2[:, 0] - top2[:, 1])
res["hf"] = {"loop_at": loops(hgen), "tail": tok.decode(hgen[-200:])}
del hf; torch.cuda.empty_cache()
for name, kw in {"k0_plain": dict(spec_k=0, compile_blocks=True), "k2_plain": dict(spec_k=2, compile_blocks=True),
                 "k2_fused": dict(spec_k=2, compile_blocks=True, fused_gdn=True), "k0_fused": dict(spec_k=0, compile_blocks=True, fused_gdn=True)}.items():
    eng = Engine(path, max_len=(len(ids) + N + 1023) // 256 * 256, **kw)
    gen = eng.generate(ids, N, ignore_eos=True)["tokens"]
    mm = next((i for i in range(min(len(gen), len(hgen))) if gen[i] != hgen[i]), None)
    r = {"first_divergence": mm, "hf_margin_there": margin[mm].item() if mm is not None else None,
         "hf_top2_there": tok.decode([int(lg[mm].topk(2).indices[0])]) + " | " + tok.decode([int(lg[mm].topk(2).indices[1])]) if mm is not None else None,
         "engine_token_there": tok.decode([gen[mm]]) if mm is not None else None, "loop_at": loops(gen),
         "match_frac": sum(a == b for a, b in zip(gen, hgen)) / len(gen), "tail": tok.decode(gen[-200:])}
    res[name] = r; print(name, {k: v for k, v in r.items() if k != "tail"}, flush=True)
    eng.close(); del eng; torch.cuda.empty_cache()
save("longcheck.json", res)
