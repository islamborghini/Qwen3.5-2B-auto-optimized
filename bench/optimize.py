"""Bounded, resumable optimization loop over engine configurations (run inside the Modal session).

  python bench/optimize.py --max_candidates 6 --max_minutes 20

Each candidate = a set of Engine kwargs. For each candidate not yet in results/opt_ledger.json:
  correctness gate (prefill logits top-1 == HF on identical prefix, decode greedy agrees with HF until a near-tie)
  -> quick screen (dev 512 + 2048, 2 reps) -> accept if geomean TPS > incumbent by more than the run-to-run spread
  and no workload regresses > 5% (TTFT and peak memory checked too). The ledger persists across runs (resumable),
  and the loop stops at --max_candidates / --max_minutes. Candidate workers may add entries to CANDIDATES only.
"""
import argparse, json, math, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast.engine import Engine

CANDIDATES = {  # name -> Engine kwargs (order = priority). Earlier rounds (results/opt_ledger_*.json): layer-level
    # torch.compile fails the numerical gate; gemv kernels (DeepSeek v1/v2) are slower than cuBLAS except on lm_head.
    "k0_compile": dict(spec_k=0, compile_blocks=True),
    "k2_compile": dict(spec_k=2, compile_blocks=True),
    "k2_compile_fused": dict(spec_k=2, compile_blocks=True, fused_gdn=True),   # T3b two-kernel fused GDN step
    # "k3_compile_fused"/"k0_compile_fused": pending the T3c illegal-memory-access fix (results/run_opt6.log)
}
LEDGER = os.path.join(OUT, "opt_ledger.json")


def screen(eng, wl, reps, static_mem=0):
    """mem = this engine's static footprint (allocated delta at construction) + transient peak during a generate."""
    out = {}
    for w in wl:
        eng.generate(w["ids"], 8, ignore_eos=True)
        runs = [eng.generate(w["ids"], 256, ignore_eos=True) for _ in range(reps)]
        torch.cuda.reset_peak_memory_stats(); eng.generate(w["ids"], 256, ignore_eos=True)
        out[w["id"]] = {"tps": [255 / r["decode_s"] for r in runs], "ttft": [r["ttft_s"] for r in runs],
                        "mem_gb": (static_mem + torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()) / 1e9, "tokens": runs[-1]["tokens"]}
    return out


def correctness(eng, wl, ref):
    """Gate (defined before any candidate was judged under it):
    1. prefill: top-1 of logits on identical prefixes equals HF at every position;
    2. decode: teacher-forced logits through the engine's real decode/spec path vs HF's own decode-path logits over
       256 steps: max|diff| <= 2 x HF's own prefill-vs-decode noise floor (same tokens), and every top-1 disagreement
       must be at a position where HF's top-2 margin is below that max|diff| (i.e. a genuine near-tie)."""
    for w in wl:
        R = ref[w["id"]]; hgen = R["gen"]; ids = w["ids"] + hgen[:64]
        eng.reset(); tok = torch.tensor(ids, device="cuda"); pos = torch.arange(len(ids), device="cuda")
        with torch.no_grad():
            hn, _ = eng._body(tok, pos, prefill=True); lg = (hn[-65:] @ eng.embed.T).float().cpu()
        if (lg.argmax(-1) != R["logits_tail"].argmax(-1)).any():
            return False, f"{w['id']}: prefill top-1 mismatch"
        el = eng.forced_decode_logits(w["ids"], hgen).cpu(); hl = R["dec_logits"]
        d = (el - hl).abs().max(-1).values; tol = max(2 * R["floor"], 0.5)
        if d.max().item() > tol:
            return False, f"{w['id']}: decode logits max|diff| {d.max().item():.3f} > tol {tol:.3f} (HF floor {R['floor']:.3f})"
        top2 = hl.topk(2, -1).values; margin = top2[:, 0] - top2[:, 1]
        bad = ((el.argmax(-1) != hl.argmax(-1)) & (margin > d.max())).nonzero().flatten().tolist()
        if bad:
            return False, f"{w['id']}: top-1 mismatch at {bad[:5]} with HF margin {margin[bad[0]].item():.3f} > max|diff| {d.max().item():.3f}"
    return True, f"ok (max|diff| {d.max().item():.3f}, tol {tol:.3f})"


def geo(d):
    return math.exp(sum(math.log(statistics.median(v["tps"])) for v in d.values()) / len(d))


def spread(d):  # geomean of per-rep tps, min..max
    n = min(len(v["tps"]) for v in d.values())
    g = [math.exp(sum(math.log(v["tps"][i]) for v in d.values()) / len(d)) for i in range(n)]
    return min(g), max(g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--max_candidates", type=int, default=4)
    ap.add_argument("--max_minutes", type=float, default=20); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(); t0 = time.time()
    seed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_ledger_seed.json")   # committed copy -> resumable across sessions
    led = json.load(open(LEDGER)) if os.path.exists(LEDGER) else (json.load(open(seed)) if os.path.exists(seed) else {"candidates": {}, "incumbent": None})
    path = model_path(); wl = workloads("dev", [512, 2048])
    hf = load_hf(); ref = {}
    for w in wl:
        x = torch.tensor([w["ids"]], device="cuda")
        with torch.no_grad():
            o_ = hf.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=256, min_new_tokens=256,
                             do_sample=False, output_logits=True, return_dict_in_generate=True)
        gen = o_.sequences[0, x.shape[1]:].tolist(); dec = torch.stack([l_[0] for l_ in o_.logits]).float().cpu()
        lg = hf_logits(hf, w["ids"] + gen)
        full = lg[len(w["ids"]) - 1: len(w["ids"]) - 1 + len(gen)].cpu()
        floor = (dec - full).abs().max().item()   # HF's own decode path vs prefill path on identical tokens
        assert dec.shape == full.shape, (dec.shape, full.shape)
        ref[w["id"]] = {"gen": gen, "logits_tail": lg[len(w["ids"]) - 1: len(w["ids"]) + 64].cpu(), "dec_logits": dec, "floor": floor}
        led.setdefault("hf_noise_floor", {})[w["id"]] = {"max_abs_diff": floor, "top1_disagree": int((dec.argmax(-1) != full.argmax(-1)).sum())}
        print("HF noise floor", w["id"], led["hf_noise_floor"][w["id"]], flush=True)
    del hf; torch.cuda.empty_cache()
    if any(kw.get("use_gemv") for kw in CANDIDATES.values()):
        from qwen35_fast import gemv as G
        try:
            G.test(); G.bench(); led["gemv_test"] = "pass"
        except Exception:
            import traceback; led["gemv_test"] = traceback.format_exc()[-1500:]; print(led["gemv_test"], flush=True)
            for k_ in list(CANDIDATES):
                if CANDIDATES[k_].get("use_gemv"): CANDIDATES.pop(k_)
    n = 0; inc_eng = None; inc_name = None
    for name, kw in CANDIDATES.items():
        if name in led["candidates"] or n >= a.max_candidates or (time.time() - t0) / 60 > a.max_minutes:
            continue
        n += 1; rec = {"kwargs": kw, "t": time.time()}
        try:
            import gc; torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache(); base_mem = torch.cuda.memory_allocated()
            eng = Engine(path, **kw); torch.cuda.synchronize()
            static_mem = torch.cuda.memory_allocated() - base_mem
            ok, why = correctness(eng, wl, ref); rec["correct"] = ok; rec["why"] = why
            if not ok:   # for the record only (never promoted): 1-rep speed screen of the rejected candidate
                rec["screen_rejected"] = {k_: statistics.median(v["tps"]) for k_, v in screen(eng, wl, 1, static_mem).items()}
            if ok:
                rec["screen"] = screen(eng, wl, a.reps, static_mem); rec["geomean"] = geo(rec["screen"]); rec["spread"] = spread(rec["screen"])
                inc = led["incumbent"]; checks = {}
                if inc and inc_eng is not None and inc_name == inc:   # paired re-measurement of the incumbent right now
                    I = screen(inc_eng, wl, a.reps, led["candidates"][inc].get("static_mem", 0)); rec["incumbent_paired"] = {w_: statistics.median(v["tps"]) for w_, v in I.items()}
                elif inc:
                    I = led["candidates"][inc]["screen"]
                if inc:
                    inc_hi = max(spread(I)[1], led["candidates"][inc]["spread"][1]) if "incumbent_paired" in rec else led["candidates"][inc]["spread"][1]
                    checks["exceeds_noise"] = rec["spread"][0] > inc_hi
                    for wid in I:
                        C = rec["screen"][wid]
                        checks[f"{wid}:tps"] = statistics.median(C["tps"]) >= 0.95 * statistics.median(I[wid]["tps"])
                        checks[f"{wid}:ttft"] = statistics.median(C["ttft"]) <= 1.05 * statistics.median(I[wid]["ttft"])
                        checks[f"{wid}:mem"] = C["mem_gb"] <= 1.05 * I[wid]["mem_gb"]
                rec["checks"] = checks; rec["accepted"] = all(checks.values()) if inc else True; rec["static_mem"] = static_mem
                if rec["accepted"]:
                    led["incumbent"] = name
                    if inc_eng is not None: del inc_eng
                    inc_eng, inc_name = eng, name; eng = None
            if eng is not None: del eng
            torch.cuda.empty_cache()
        except Exception:
            import traceback; rec["error"] = traceback.format_exc()[-2000:]; rec["accepted"] = False
        led["candidates"][name] = rec
        json.dump(led, open(LEDGER, "w"), indent=1, default=str)
        print(name, {k: v for k, v in rec.items() if k in ("correct", "why", "geomean", "spread", "accepted", "error")}, flush=True)
    print("incumbent:", led["incumbent"])
