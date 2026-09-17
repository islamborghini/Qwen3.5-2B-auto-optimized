"""Optimization session: GEMV kernel test/bench, compiled-step profile, engine screen with gemv on/off."""
import json, os, sys, time, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast.engine import Engine

res = {}
path = model_path()
# 1) GEMV kernel
try:
    from qwen35_fast import gemv as G
    G.test(); res["gemv_test"] = "pass"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): G.bench()
    res["gemv_bench"] = buf.getvalue(); print(buf.getvalue(), flush=True)
except Exception:
    res["gemv_error"] = traceback.format_exc(); print(res["gemv_error"], flush=True)
gemv_ok = "gemv_test" in res

# 2) engine screen: compile on; gemv off/on; k in 0,2,3
wl = workloads("dev", [512, 2048])
for use_gemv in ([False, True] if gemv_ok else [False]):
    for k in [0, 2, 3]:
        tag = f"k{k}_gemv{int(use_gemv)}"
        try:
            eng = Engine(path, spec_k=k, compile_blocks=True, use_gemv=use_gemv)
            eng.generate(wl[0]["ids"], 8, ignore_eos=True)
            eng.prefill(wl[0]["ids"]); torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(100): eng.step()
            torch.cuda.synchronize(); step_ms = (time.perf_counter() - t) / 100 * 1000
            r = {"step_ms": step_ms, "runs": {}, "mm_choice": {str(k_): v for k_, v in eng._mm_choice.items()}}
            for w in wl:
                eng.generate(w["ids"], 8, ignore_eos=True)
                o = eng.generate(w["ids"], 256, ignore_eos=True)
                r["runs"][w["id"]] = {"tps": 255 / o["decode_s"], "acc": o["spec"], "tokens_head": o["tokens"][:32]}
            if k == 0 and not use_gemv:
                from torch.profiler import profile, ProfilerActivity
                eng.prefill(wl[0]["ids"]); [eng.step(use_graph=False) for _ in range(3)]; torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                    for _ in range(5): eng.step(use_graph=False)
                    torch.cuda.synchronize()
                ka = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
                r["kernels_per_step"] = sum(e.count for e in ka if e.self_device_time_total > 0 and e.device_type == torch.autograd.DeviceType.CUDA) / 5
                r["top"] = [{"name": e.key[:60], "us": e.self_device_time_total / 5, "n": e.count / 5} for e in ka[:20] if e.self_device_time_total > 0]
            res[tag] = r
            print(tag, json.dumps({kk: (v if kk != "runs" else {i: round(x["tps"]) for i, x in v.items()}) for kk, v in r.items() if kk != "top"}), flush=True)
            for t_ in r.get("top", [])[:14]: print("   ", t_, flush=True)
            del eng; torch.cuda.empty_cache()
        except Exception:
            res[tag] = {"error": traceback.format_exc()}; print(res[tag]["error"], flush=True)
save("run_opt.json", res)
