"""Custom engine: correctness vs HF (logits on identical prefixes + multi-step greedy decode) and speed screen.
Args: --spec 0,1,2,3  --hidden post_norm,pre_norm  --lengths 512,2048
"""
import argparse, json, os, sys, time, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from qwen35_fast.engine import Engine

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="0,1,2,3"); ap.add_argument("--hidden", default="post_norm,pre_norm")
ap.add_argument("--lengths", default="512,2048"); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--n_out", type=int, default=256); ap.add_argument("--skip_hf", action="store_true")
ap.add_argument("--compile", default="0,1")
a = ap.parse_args()
lengths = [int(x) for x in a.lengths.split(",")]
res = {"gpu": torch.cuda.get_device_name(0)}
path = model_path()
wl = workloads("dev", lengths)

# ---- HF reference (logits + greedy) ----
ref = {}
if not a.skip_hf:
    hf = load_hf()
    for w in wl:
        gen, ttft, tps, tot = hf_greedy(hf, w["ids"], a.n_out)
        lg = hf_logits(hf, w["ids"] + gen[:64])
        ref[w["id"]] = {"gen": gen, "ttft": ttft, "tps": tps, "logits_tail": lg[-65:].cpu()}
        print("HF", w["id"], f"tps={tps:.1f} ttft={ttft*1000:.1f}ms", flush=True)
    res["hf"] = {k: {"tps": v["tps"], "ttft": v["ttft"]} for k, v in ref.items()}
    del hf; torch.cuda.empty_cache()

# ---- custom engine ----
for hidden in a.hidden.split(","):
  for comp in [int(x) for x in a.compile.split(",")]:
    for k in [int(x) for x in a.spec.split(",")]:
        if k == 0 and hidden != a.hidden.split(",")[0]:
            continue
        tag = f"custom_k{k}_{hidden}_c{comp}"
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter(); eng = Engine(path, spec_k=k, mtp_hidden=hidden, compile_blocks=bool(comp)); load_s = time.perf_counter() - t
        r = {"load_s": load_s, "runs": {}}
        try:
            for w in wl:
                # warmup (also captures graph)
                eng.generate(w["ids"], 8, ignore_eos=True)
                outs = []
                for _ in range(a.reps):
                    o = eng.generate(w["ids"], a.n_out, ignore_eos=True)
                    o["decode_tps"] = (len(o["tokens"]) - 1) / o["decode_s"]
                    outs.append(o)
                gen = outs[-1]["tokens"]
                item = {"decode_tps": [o["decode_tps"] for o in outs], "ttft_s": [o["ttft_s"] for o in outs],
                        "total_s": [o["total_s"] for o in outs], "n_gen": len(gen), "text": None,
                        "acc_rate": (outs[-1]["spec"]["accepted"] / max(1, outs[-1]["spec"]["steps"] * max(k, 1)))}
                if w["id"] in ref:
                    hgen = ref[w["id"]]["gen"]
                    mm = next((i for i in range(min(len(hgen), len(gen))) if hgen[i] != gen[i]), None)
                    item["first_mismatch_vs_hf"] = mm
                    item["match_frac_vs_hf"] = sum(x == y for x, y in zip(hgen, gen)) / len(gen)
                    # logits on identical prefix: prompt + first 64 HF tokens, compare last 65 positions
                    ids = w["ids"] + hgen[:64]
                    eng.reset(); tokens = torch.tensor(ids, device="cuda"); pos = torch.arange(len(ids), device="cuda")
                    with torch.no_grad():
                        hn, _ = eng._body(tokens, pos, prefill=True)
                        lg = (hn[-65:] @ eng.embed.T).float().cpu()
                    rl = ref[w["id"]]["logits_tail"]
                    item["logit_max_abs_diff"] = (lg - rl).abs().max().item()
                    item["logit_ref_max_abs"] = rl.abs().max().item()
                    item["top1_agree_frac"] = (lg.argmax(-1) == rl.argmax(-1)).float().mean().item()
                r["runs"][w["id"]] = item
                print(tag, w["id"], json.dumps({kk: (round(statistics.median(v), 2) if isinstance(v, list) else v) for kk, v in item.items() if kk != "text"}), flush=True)
            r["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        except Exception:
            import traceback; r["error"] = traceback.format_exc(); print(r["error"], flush=True)
        res[tag] = r
        del eng; torch.cuda.empty_cache()
save("custom_screen.json", res)
