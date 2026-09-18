"""Render README result tables from results/*.json. Usage: python bench/fill_results.py > results/RESULTS.md"""
import json, math, os, statistics as st
R = "results"
def load(n):
    p = os.path.join(R, n); return json.load(open(p)) if os.path.exists(p) else None
def med(x): return st.median(x)
def fmt(runs, key="decode_tps"):
    v = [r[key] for r in runs]; return f"{med(v):.0f} ({min(v):.0f}-{max(v):.0f})"

base = load("eval_baselines_full_dev.json"); hfc = load("eval_hfcompile_full_dev.json")
fin = load("eval_final_full_dev.json"); held = load("eval_final_heldout_heldout.json")
led = load("opt_ledger.json"); frozen = load("frozen_config.json") or json.load(open("bench/frozen_config.json"))
out = []
if base and fin:
    custs = [n for n in fin["engines"] if n.startswith("custom")]; cust = custs[-1]
    wids = list(base["engines"]["hf_eager"]["runs"])
    orig = [n for n in base["engines"]]
    out.append("### Dev suite (12 workloads, 256 output tokens, batch 1, H100). Decode TPS: median (min-max) over 5 reps\n")
    cols = ["hf_eager", "hf_compile", "vllm_plain", "vllm_mtp1", "vllm_mtp2", "vllm_mtp3"] + custs
    out.append("| Workload | " + " | ".join(cols) + " | strongest original | speedup |"); out.append("|---" * (len(cols) + 3) + "|")
    sp = []
    for w in wids:
        cells = []; ref = 0; refn = ""
        for c in cols:
            src = fin if c in custs else (hfc if c == "hf_compile" else base)
            runs = src["engines"].get(c, {}).get("runs", {}).get(w) if src else None
            cells.append(fmt(runs) if runs else "n/a")
            if c not in custs and runs and med([r["decode_tps"] for r in runs]) > ref:
                ref = med([r["decode_tps"] for r in runs]); refn = c
        cm = med([r["decode_tps"] for r in fin["engines"][cust]["runs"][w]]); s = cm / ref; sp.append(s)
        out.append(f"| {w} | " + " | ".join(cells) + f" | {refn} | {s:.2f}x |")
    out.append(f"\n**Geometric-mean speedup of `{cust}` vs the strongest original baseline per workload: {math.exp(sum(map(math.log, sp))/len(sp)):.2f}x** "
               f"(vs hf_eager: {math.exp(sum(math.log(med([r['decode_tps'] for r in fin['engines'][cust]['runs'][w]]) / med([r['decode_tps'] for r in base['engines']['hf_eager']['runs'][w]])) for w in wids)/len(wids)):.1f}x).\n")
    # same-session comparator
    if "vllm_mtp3" in fin["engines"]:
        out.append("### Same-session comparison (custom vs vLLM+MTP k=3 measured back-to-back in the final session)\n")
        out.append("| Workload | vllm_mtp3 TPS | " + " | ".join(f"{c} TPS | ratio | {c} TTFT ms | {c} peak GB" for c in custs) + " | vllm_mtp3 TTFT ms |")
        out.append("|---" * (3 + 4 * len(custs)) + "|")
        for w in wids:
            a = fin["engines"]["vllm_mtp3"]["runs"][w]; cells = []
            for c in custs:
                b = fin["engines"][c]["runs"][w]
                cells.append(f"{fmt(b)} | {med([r['decode_tps'] for r in b])/med([r['decode_tps'] for r in a]):.2f}x | {1000*med([r['ttft_s'] for r in b]):.1f} | {max(r['peak_mem_gb'] for r in b):.2f}")
            out.append(f"| {w} | {fmt(a)} | " + " | ".join(cells) + f" | {1000*med([r['ttft_s'] for r in a]):.1f} |")
        for c in custs:
            g = math.exp(sum(math.log(med([r['decode_tps'] for r in fin['engines'][c]['runs'][w]]) / med([r['decode_tps'] for r in fin['engines']['vllm_mtp3']['runs'][w]])) for w in wids) / len(wids))
            out.append(f"\nSame-session geomean ratio {c} / vllm_mtp3: **{g:.2f}x**")
        out.append("")
    out.append("### Startup costs (excluded from TPS)\n")
    for n, src in [(c, fin if c in custs else (hfc if c == "hf_compile" else base)) for c in cols]:
        if src and n in src["engines"]: out.append(f"* {n}: {src['engines'][n]['startup_s']:.0f} s (model load + compile/graph capture + first warmup)")
    if fin.get("drift"): out.append(f"\nDrift check (final session): {fin['drift']}")
    if base.get("drift"): out.append(f"Drift check (baselines session): {base['drift']}")
if held:
    custs = [n for n in held["engines"] if n.startswith("custom")]
    out.append("\n### Held-out prompts (never used during optimization; same session as vllm_mtp3)\n")
    out.append("| Workload | vllm_mtp3 TPS | " + " | ".join(f"{c} TPS | ratio" for c in custs) + " |"); out.append("|---" * (1 + 1 + 2 * len(custs)) + "|")
    rs = {c: [] for c in custs}
    for w in held["engines"][custs[0]]["runs"]:
        a = held["engines"]["vllm_mtp3"]["runs"][w]; cells = []
        for c in custs:
            b = held["engines"][c]["runs"][w]; r = med([x["decode_tps"] for x in b]) / med([x["decode_tps"] for x in a]); rs[c].append(r)
            cells.append(f"{fmt(b)} | {r:.2f}x")
        out.append(f"| {w} | {fmt(a)} | " + " | ".join(cells) + " |")
    for c in custs:
        out.append(f"\nHeld-out geomean ratio {c} / vllm_mtp3: **{math.exp(sum(map(math.log, rs[c]))/len(rs[c])):.2f}x**")
    out.append("")
if led:
    out.append("\n### Optimization ledger (bench/optimize.py, dev screen 512+2048)\n")
    out.append("| candidate | kwargs | correct | geomean TPS (screen) | accepted |"); out.append("|---|---|---|---|---|")
    for n, c in led["candidates"].items():
        why = c.get('why','')[:60].replace('|', '\\|')
        out.append(f"| {n} | `{c['kwargs']}` | {c.get('correct')} ({why}) | {c.get('geomean', float('nan')):.0f} | {c.get('accepted')} |")
    out.append(f"\nFrozen incumbent: **{led['incumbent']}** `{frozen['kwargs'] if frozen else ''}`\n")
ia = load("ifeval_hf.json"); ibs = [load(n) for n in sorted(os.listdir(R)) if n.startswith("ifeval_custom")]
if ia and ibs:
    out.append("\n### IFEval 100-prompt subset (greedy, non-thinking, normal stopping, max 1280 new tokens)\n")
    out.append("| engine | prompt-level strict | inst-level strict | prompt-level loose | truncated | elapsed s | changed strict outcomes vs hf (F->T / T->F) | identical text vs hf |"); out.append("|---|---|---|---|---|---|---|---|")
    A = {s["doc_id"]: s for s in ia["samples"]}
    for e in [ia] + ibs:
        r = e["results"]; B = {s["doc_id"]: s for s in e["samples"]}
        ft = sum(1 for d in A if d in B and not A[d]["prompt_strict"] and B[d]["prompt_strict"]); tf = sum(1 for d in A if d in B and A[d]["prompt_strict"] and not B[d]["prompt_strict"])
        same = sum(A[d]["resp"] == B[d]["resp"] for d in A if d in B)
        out.append(f"| {e['engine']} | {r['prompt_level_strict_acc,none']:.3f} | {r['inst_level_strict_acc,none']:.3f} | {r['prompt_level_loose_acc,none']:.3f} | {e['truncated']}/{e['limit']} | {e['elapsed_s']:.0f} | {ft} / {tf} | {same}/{len(A)} |")
    out.append("")
print("\n".join(out))
