"""Localize the fused-GDN illegal memory access: CUDA_LAUNCH_BLOCKING=1, phase markers, engines built/destroyed in sequence."""
import os, sys, gc
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast import gdn_step as G
from qwen35_fast.engine import Engine


def P(*a):
    torch.cuda.synchronize(); print("PHASE", *a, flush=True)


path = model_path(); ids = workloads("dev", [512])[0]["ids"]; ids2 = workloads("dev", [2048], ["structured"])[0]["ids"]
P("gdn_step.test"); G.test()
for k in [3, 2, 0, 1]:
    P(f"build k={k}"); eng = Engine(path, spec_k=k, compile_blocks=True, fused_gdn=True)
    P(f"k={k} prefill"); eng.prefill(ids)
    P(f"k={k} eager steps"); [eng.step(use_graph=False) for _ in range(20)]
    P(f"k={k} ensure_graph"); eng.ensure_graph()
    P(f"k={k} graph steps"); eng.prefill(ids); [eng.step() for _ in range(50)]
    P(f"k={k} generate"); r = eng.generate(ids, 256, ignore_eos=True); print("  tps", 255 / r["decode_s"], r["spec"], flush=True)
    P(f"k={k} forced_decode_logits"); eng.forced_decode_logits(ids, r["tokens"][:64])
    if k >= 2:
        P(f"k={k} tps 2 reps"); tps = {}
        for w, wid in ((ids, "prose-512"), (ids2, "structured-2048")):
            eng.generate(w, 8, ignore_eos=True); tps[wid] = [255 / eng.generate(w, 256, ignore_eos=True)["decode_s"] for _ in range(2)]
        print("  fused TPS", tps, flush=True)
    P(f"k={k} teardown"); del eng; gc.collect(); torch.cuda.synchronize(); torch._dynamo.reset(); torch.cuda.empty_cache()
P("done")
