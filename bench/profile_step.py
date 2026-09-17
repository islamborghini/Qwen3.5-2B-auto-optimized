"""Profile the custom engine decode step: kernel count + top kernels (eager), graph step time, MTP acceptance per k."""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast.engine import Engine

path = model_path(); res = {}
wl = workloads("dev", [512])
from safetensors import safe_open
wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(path, wm["mtp.norm.weight"]), "pt", device="cpu") as f:
    for n in ["mtp.norm.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_embedding.weight",
              "mtp.layers.0.input_layernorm.weight", "model.language_model.norm.weight", "model.language_model.layers.0.linear_attn.norm.weight"]:
        t = f.get_tensor(n).float(); res[f"w:{n}"] = {"mean": t.mean().item(), "std": t.std().item()}
print(json.dumps(res, indent=1), flush=True)

for k in [0, 1, 2, 3]:
    eng = Engine(path, spec_k=k)
    eng.generate(wl[0]["ids"], 8, ignore_eos=True)       # capture graph
    # graph step time
    eng.prefill(wl[0]["ids"]); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100): eng.step()
    torch.cuda.synchronize(); step_ms = (time.perf_counter() - t) / 100 * 1000
    # eager profile
    eng.prefill(wl[0]["ids"]); torch.cuda.synchronize()
    for _ in range(3): eng.step(use_graph=False)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(5): eng.step(use_graph=False)
        torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA or e.self_device_time_total > 0]
    ev = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
    n_kernels = sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0 and e.device_type == torch.autograd.DeviceType.CUDA) / 5
    top = [{"name": e.key[:70], "self_us_per_step": e.self_device_time_total / 5, "count_per_step": e.count / 5} for e in ev[:22] if e.self_device_time_total > 0]
    # acceptance over full generations
    acc = []
    for w in workloads("dev", [512, 2048]):
        o = eng.generate(w["ids"], 256, ignore_eos=True); o["decode_tps"] = 255 / o["decode_s"]
        acc.append({"id": w["id"], "tps": o["decode_tps"], **o["spec"]})
    r = {"graph_step_ms": step_ms, "kernels_per_step": n_kernels, "top": top, "gen": acc}
    res[f"k{k}"] = r
    print(f"k={k} step_ms={step_ms:.3f} kernels/step={n_kernels:.0f}", json.dumps(acc), flush=True)
    for t_ in top[:12]: print("   ", t_, flush=True)
    if k == 0:
        prof.export_chrome_trace(os.path.join(OUT, "trace_k0.json"))
    del eng; torch.cuda.empty_cache()
save("profile_step.json", res)
