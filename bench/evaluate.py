"""Frozen evaluator. Candidate workers must not edit this file, workloads.json, or the acceptance criteria.

Measurement boundaries (all engines):
  * prefill_s / ttft_s : request start -> first generated token available on host (prefill + first sampling).
  * decode_tps         : (n_gen - 1) / (t_last_token - t_first_token), token completion times observed on host
                         (HF: streamer.put after the per-step .cpu(); vLLM: async stream chunk arrival; custom: per-step
                         event sync). Host-side and launch overhead included; no extra syncs inside steps.
  * total_s            : request start -> all tokens on host.
  * peak_mem_gb        : torch.cuda.max_memory_allocated (HF/custom) or vLLM-reported in-process value (see notes).
Model load, graph capture, compilation, and warmup are excluded from steady-state numbers and reported as startup_s.

Usage (inside the Modal session):  python bench/evaluate.py --engines hf_eager,vllm_mtp3,custom --stage screen|full
"""
import argparse, asyncio, json, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *

N_OUT = WORKLOADS["n_out"]
STAGES = {"screen": dict(lengths=[512, 2048], reps=2), "full": dict(lengths=[128, 512, 2048, 8192], reps=5),
          "heldout": dict(lengths=[128, 512, 2048, 8192], reps=5)}


# ---------------------------------------------------------------- engines
class HFEngine:
    name = "hf_eager"
    def __init__(self, compile=False):
        t = time.perf_counter(); self.m = load_hf(); self.startup_s = time.perf_counter() - t; self.compile = compile
        if compile:
            self.m.generation_config.cache_implementation = "static"
            self.m.forward = torch.compile(self.m.forward, mode="reduce-overhead", fullgraph=False)
    def run(self, ids):
        torch.cuda.reset_peak_memory_stats()
        gen, ttft, tps, tot = hf_greedy(self.m, ids, N_OUT)
        return dict(tokens=gen, ttft_s=ttft, decode_tps=tps, total_s=tot, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    def close(self):
        del self.m; torch.cuda.empty_cache()


class VLLMEngine:
    def __init__(self, spec_k=0):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        self.name = f"vllm_mtp{spec_k}" if spec_k else "vllm_plain"
        kw = {"speculative_config": {"method": "mtp", "num_speculative_tokens": spec_k}} if spec_k else {}
        t = time.perf_counter()
        self.llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=MODEL, revision=REV, dtype="bfloat16", max_model_len=9216, gpu_memory_utilization=0.16, seed=0,
            limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=False, **kw))
        self.startup_s = time.perf_counter() - t
        self.loop = asyncio.new_event_loop(); self.i = 0

    async def _gen(self, ids):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=N_OUT, ignore_eos=True)
        self.i += 1
        t0 = time.perf_counter(); ts = []; last = None
        async for out in self.llm.generate({"prompt_token_ids": ids}, sp, request_id=f"r{self.i}"):
            n = len(out.outputs[0].token_ids)
            if n > len(ts):
                ts.extend([time.perf_counter()] * (n - len(ts)))   # several tokens may arrive in one chunk (spec decode)
            last = out
        gen = list(last.outputs[0].token_ids)
        return dict(tokens=gen, ttft_s=ts[0] - t0, decode_tps=(len(gen) - 1) / (ts[-1] - ts[0]), total_s=ts[-1] - t0,
                    peak_mem_gb=None)

    def run(self, ids):
        return self.loop.run_until_complete(self._gen(ids))
    def close(self):
        self.llm.shutdown()


class CustomEngine:
    name = "custom"
    def __init__(self, spec_k=3, **kw):
        from qwen35_fast.engine import Engine
        t = time.perf_counter(); self.e = Engine(model_path(), spec_k=spec_k, **kw)
        self.e.generate(workloads("dev", [128])[0]["ids"], 8, ignore_eos=True)   # graph capture
        self.startup_s = time.perf_counter() - t
        self.name = f"custom_k{spec_k}"
    def run(self, ids):
        torch.cuda.reset_peak_memory_stats()
        o = self.e.generate(ids, N_OUT, ignore_eos=True)
        return dict(tokens=o["tokens"], ttft_s=o["ttft_s"], decode_tps=(len(o["tokens"]) - 1) / o["decode_s"],
                    total_s=o["total_s"], peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    def close(self):
        del self.e; torch.cuda.empty_cache()


def make(name):
    if name == "hf_eager": return HFEngine()
    if name == "hf_compile": return HFEngine(compile=True)
    if name == "vllm_plain": return VLLMEngine(0)
    if name.startswith("vllm_mtp"): return VLLMEngine(int(name[8:]))
    if name.startswith("custom_k"): return CustomEngine(int(name[8:]))
    if name == "custom": return CustomEngine(3)
    raise ValueError(name)


# ---------------------------------------------------------------- protocol
def evaluate(engine_names, stage, split="dev", tag=None):
    st = STAGES[stage]; wl = workloads(split, st["lengths"])
    engines = {}
    res = {"stage": stage, "split": split, "gpu": torch.cuda.get_device_name(0), "engines": {}}
    for n in list(engine_names):
        try:
            e = make(n); engines[n] = e
            res["engines"][n] = {"startup_s": e.startup_s, "runs": {w["id"]: [] for w in wl}}
            for w in wl:  # warmup each configuration
                e.run(w["ids"])
        except Exception:
            import traceback
            res.setdefault("failed", {})[n] = traceback.format_exc()[-3000:]
            print("ENGINE FAILED", n, res["failed"][n][-800:], flush=True)
            engine_names = [x for x in engine_names if x != n]
            res["engines"].pop(n, None); engines.pop(n, None); torch.cuda.empty_cache()
    # alternate engine order per repetition to reduce drift effects; engines are all resident (serial execution)
    for rep in range(st["reps"]):
        order = engine_names if rep % 2 == 0 else engine_names[::-1]
        for w in wl:
            for n in order:
                r = engines[n].run(w["ids"])
                res["engines"][n]["runs"][w["id"]].append({k: v for k, v in r.items() if k != "tokens"} | {"tokens": r["tokens"]})
                print(n, w["id"], f"tps={r['decode_tps']:.1f} ttft={r['ttft_s']*1000:.1f}ms", flush=True)
    for e in engines.values():
        e.close()
    res["summary"] = summarize(res)
    save(f"eval_{tag or stage}_{split}.json", res)
    return res


def summarize(res):
    out = {}
    for n, d in res["engines"].items():
        out[n] = {}
        for wid, runs in d["runs"].items():
            tps = [r["decode_tps"] for r in runs]
            out[n][wid] = {"decode_tps_median": statistics.median(tps), "decode_tps_min": min(tps), "decode_tps_max": max(tps),
                           "ttft_ms_median": 1000 * statistics.median(r["ttft_s"] for r in runs),
                           "total_s_median": statistics.median(r["total_s"] for r in runs),
                           "peak_mem_gb": max((r["peak_mem_gb"] or 0) for r in runs) or None}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True); ap.add_argument("--stage", default="screen")
    ap.add_argument("--split", default="dev"); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    evaluate(a.engines.split(","), a.stage, a.split, a.tag)
