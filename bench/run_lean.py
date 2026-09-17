"""T1 measurement: lean (fused residual+norm, fused q/k norm+rope) vs non-lean decode path, compile_blocks=True.
Reports kernels/step, step_ms, TPS (dev 512/2048 prose+structured) and teacher-forced logit max|diff| vs HF decode
logits (+ HF's own prefill-vs-decode floor) on dev-prose-512 and dev-structured-2048."""
import json, os, sys, time, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast.engine import Engine
from torch.profiler import profile, ProfilerActivity

path = model_path(); res = {}
wl = workloads("dev", [512, 2048], ["prose", "structured"])
gate_wl = [w for w in wl if w["id"] in ("dev-prose-512", "dev-structured-2048")]

# HF reference (same construction as bench/optimize.py)
hf = load_hf(); ref = {}
for w in gate_wl:
    x = torch.tensor([w["ids"]], device="cuda")
    with torch.no_grad():
        o_ = hf.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=256, min_new_tokens=256,
                         do_sample=False, output_logits=True, return_dict_in_generate=True)
    gen = o_.sequences[0, x.shape[1]:].tolist(); dec = torch.stack([l_[0] for l_ in o_.logits]).float().cpu()
    full = hf_logits(hf, w["ids"] + gen)[len(w["ids"]) - 1: len(w["ids"]) - 1 + len(gen)].cpu()
    ref[w["id"]] = {"gen": gen, "dec": dec, "floor": (dec - full).abs().max().item()}
    print("HF floor", w["id"], ref[w["id"]]["floor"], flush=True)
del hf; torch.cuda.empty_cache()

for lean, emu in ((False, False), (True, False)):
    for k in (0, 2, 3):
        tag = f"lean{int(lean)}_emu{int(emu)}_k{k}"
        try:
            eng = Engine(path, spec_k=k, compile_blocks=True, lean=lean, emulate_casts=emu)
            eng.generate(wl[0]["ids"], 8, ignore_eos=True)
            eng.prefill(wl[0]["ids"]); torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(100): eng.step()
            torch.cuda.synchronize(); r = {"step_ms": (time.perf_counter() - t) / 100 * 1000, "tps": {}, "logit_diff": {}}
            eng.prefill(wl[0]["ids"]); [eng.step(use_graph=False) for _ in range(3)]; torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                for _ in range(5): eng.step(use_graph=False)
                torch.cuda.synchronize()
            ka = prof.key_averages()
            r["kernels_per_step"] = sum(e.count for e in ka if e.self_device_time_total > 0 and e.device_type == torch.autograd.DeviceType.CUDA) / 5
            for w in wl:
                eng.generate(w["ids"], 8, ignore_eos=True)
                o = eng.generate(w["ids"], 256, ignore_eos=True)
                r["tps"][w["id"]] = 255 / o["decode_s"]
            for w in gate_wl:
                el = eng.forced_decode_logits(w["ids"], ref[w["id"]]["gen"]).cpu()
                dd = (el - ref[w["id"]]["dec"]).abs().max(-1).values
                r["logit_diff"][w["id"]] = {"max_abs": dd.max().item(), "p99": dd.quantile(0.99).item(), "hf_floor": ref[w["id"]]["floor"],
                                            "top1_disagree": int((el.argmax(-1) != ref[w["id"]]["dec"].argmax(-1)).sum())}
                ref[w["id"]].setdefault("engine_logits", {})[tag] = el
            res[tag] = r
            print(tag, json.dumps(r), flush=True)
            del eng; torch.cuda.empty_cache()
        except Exception:
            res[tag] = {"error": traceback.format_exc()}; print(res[tag]["error"], flush=True)
for w in gate_wl:   # lean vs non-lean must be numerically identical per k
    E = ref[w["id"]]["engine_logits"]
    for k in (0, 2, 3):
        a, b = E.get(f"lean0_emu0_k{k}"), E.get(f"lean1_emu0_k{k}")
        if a is not None and b is not None:
            res[f"lean_vs_nonlean_k{k}_{w['id']}"] = (a - b).abs().max().item(); print("lean vs nonlean max|diff|", k, w["id"], res[f"lean_vs_nonlean_k{k}_{w['id']}"], flush=True)
save("run_lean3.json", res)
