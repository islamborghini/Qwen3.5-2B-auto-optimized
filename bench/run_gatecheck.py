"""Gate diagnostics: per-prompt teacher-forced decode-logit diff vs HF for several engine configs, plus HF's own floor."""
import os, sys, json
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast.engine import Engine
wl = workloads("dev", [512, 2048]); path = model_path()
hf = load_hf(); ref = {}
for w in wl:
    x = torch.tensor([w["ids"]], device="cuda")
    with torch.no_grad():
        o_ = hf.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=256, min_new_tokens=256, do_sample=False, output_logits=True, return_dict_in_generate=True)
    gen = o_.sequences[0, x.shape[1]:].tolist(); dec = torch.stack([l_[0] for l_ in o_.logits]).float().cpu()
    full = hf_logits(hf, w["ids"] + gen)[len(w["ids"]) - 1: len(w["ids"]) - 1 + len(gen)].cpu()
    ref[w["id"]] = {"gen": gen, "dec": dec, "floor": (dec - full).abs().max().item(), "floor_p99": (dec - full).abs().max(-1).values.quantile(0.99).item()}
del hf; torch.cuda.empty_cache()
res = {"floor": {k: (round(v["floor"], 3), round(v["floor_p99"], 3)) for k, v in ref.items()}}
print("HF floors (max, p99 over steps):", res["floor"], flush=True)
for name, kw in {"k0_nonlean": dict(spec_k=0, compile_blocks=True, lean=False), "k0_lean": dict(spec_k=0, compile_blocks=True, lean=True),
                 "k0_eager_nonlean": dict(spec_k=0, lean=False), "k3_lean": dict(spec_k=3, compile_blocks=True, lean=True)}.items():
    torch._dynamo.reset(); eng = Engine(path, **kw); row = {}
    for w in wl:
        el = eng.forced_decode_logits(w["ids"], ref[w["id"]]["gen"]).cpu(); d = (el - ref[w["id"]]["dec"]).abs().max(-1).values
        row[w["id"]] = (round(d.max().item(), 3), round(d.quantile(0.99).item(), 3), int(d.argmax()))
    res[name] = row; print(name, row, flush=True)
    del eng; torch.cuda.empty_cache()
save("gatecheck.json", res)
