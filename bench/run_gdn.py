"""Fused GDN step kernel: unit test, step time / kernel count, TPS screen, teacher-forced logit diff vs HF."""
import json, os, sys, time, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast.engine import Engine
from qwen35_fast import gdn_step

res = {}
try:
    res["test"] = gdn_step.test()
except Exception:
    res["test_error"] = traceback.format_exc(); print(res["test_error"], flush=True)
path = model_path()
wl = workloads("dev", [512, 2048], ["prose", "structured"])
wl = [w for w in wl if w["id"] in ("dev-prose-512", "dev-structured-2048")]

# HF reference: decode-path logits (as bench/optimize.py)
hf = load_hf(); ref = {}
for w in wl:
    x = torch.tensor([w["ids"]], device="cuda")
    with torch.no_grad():
        o_ = hf.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=256, min_new_tokens=256,
                         do_sample=False, output_logits=True, return_dict_in_generate=True)
    gen = o_.sequences[0, x.shape[1]:].tolist(); dec = torch.stack([l_[0] for l_ in o_.logits]).float().cpu()
    full = hf_logits(hf, w["ids"] + gen)[len(w["ids"]) - 1: len(w["ids"]) - 1 + len(gen)].cpu()
    ref[w["id"]] = {"gen": gen, "dec": dec, "floor": (dec - full).abs().max().item()}
    print("HF floor", w["id"], ref[w["id"]]["floor"], flush=True)
del hf; torch.cuda.empty_cache()

for fused in [False, True]:
    tag = f"fused_gdn={fused}"
    try:
        eng = Engine(path, spec_k=0, compile_blocks=True, fused_gdn=fused)
        eng.generate(wl[0]["ids"], 8, ignore_eos=True)
        eng.prefill(wl[0]["ids"]); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(200): eng.step()
        torch.cuda.synchronize(); step_ms = (time.perf_counter() - t) / 200 * 1000
        from torch.profiler import profile, ProfilerActivity
        eng.prefill(wl[0]["ids"]); [eng.step(use_graph=False) for _ in range(3)]; torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(5): eng.step(use_graph=False)
            torch.cuda.synchronize()
        ka = prof.key_averages()
        r = {"step_ms": step_ms,
             "kernels_per_step": sum(e.count for e in ka if e.self_device_time_total > 0 and e.device_type == torch.autograd.DeviceType.CUDA) / 5,
             "tps": {}, "logit_diff": {}}
        for w in wl:
            eng.generate(w["ids"], 8, ignore_eos=True)
            o = eng.generate(w["ids"], 256, ignore_eos=True); r["tps"][w["id"]] = 255 / o["decode_s"]
            el = eng.forced_decode_logits(w["ids"], ref[w["id"]]["gen"]).cpu(); hl = ref[w["id"]]["dec"]
            d = (el - hl).abs().max(-1).values
            r["logit_diff"][w["id"]] = {"max": d.max().item(), "floor": ref[w["id"]]["floor"],
                                        "top1_disagree": int((el.argmax(-1) != hl.argmax(-1)).sum())}
        res[tag] = r; print(tag, json.dumps(r), flush=True)
        del eng; torch.cuda.empty_cache()
    except Exception:
        res[tag] = {"error": traceback.format_exc()}; print(res[tag]["error"], flush=True)
save("run_gdn.json", res)
