"""Apply the frozen acceptance criteria and render the results table.
Usage: python bench/report.py results/eval_full_dev.json [--candidate custom_k3] [--incumbent custom_k2]
Reference per workload = strongest ORIGINAL baseline (hf_eager, hf_compile, vllm_plain, vllm_mtp*) by median decode TPS.
Score = geometric mean over workloads of candidate_median_tps / reference_median_tps.
Accept candidate vs incumbent only if: no workload decode TPS regresses > 5%, no workload TTFT (prefill) or peak memory
rises > 5%, and the geomean improvement exceeds run-to-run noise (candidate min > incumbent max on the geomean of
per-run TPS is used as the "exceeds noise" test).
"""
import argparse, json, math, statistics, sys

ORIGINAL = ("hf_eager", "hf_compile", "vllm_plain", "vllm_mtp")


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def per_run_geomean(eng, wids, ref):
    """Geomean speedup for each repetition index (to estimate variability of the score)."""
    n = min(len(eng["runs"][w]) for w in wids)
    return [geomean([eng["runs"][w][i]["decode_tps"] / ref[w] for w in wids]) for i in range(n)]


def report(res, candidate=None, incumbent=None):
    S = res["summary"]; wids = list(next(iter(S.values())).keys())
    orig = [n for n in S if n.startswith(ORIGINAL)]
    ref = {w: max(S[n][w]["decode_tps_median"] for n in orig) for w in wids}
    ref_name = {w: max(orig, key=lambda n: S[n][w]["decode_tps_median"]) for w in wids}
    lines = ["| Workload | " + " | ".join(S) + " | strongest original | speedup (" + (candidate or "-") + ") |", "|---" * (len(S) + 3) + "|"]
    for w in wids:
        cells = [f"{S[n][w]['decode_tps_median']:.0f} (±{(S[n][w]['decode_tps_max']-S[n][w]['decode_tps_min'])/2:.0f})" for n in S]
        sp = f"{S[candidate][w]['decode_tps_median']/ref[w]:.2f}x" if candidate else "-"
        lines.append(f"| {w} | " + " | ".join(cells) + f" | {ref_name[w]} | {sp} |")
    out = {"table": "\n".join(lines), "reference": ref_name}
    if candidate:
        g = per_run_geomean(res["engines"][candidate], wids, ref)
        out["candidate_geomean"] = geomean([S[candidate][w]["decode_tps_median"] / ref[w] for w in wids])
        out["candidate_geomean_runs"] = g
        if incumbent:
            gi = per_run_geomean(res["engines"][incumbent], wids, ref)
            checks = {"geomean_exceeds_noise": min(g) > max(gi)}
            for w in wids:
                c, i = S[candidate][w], S[incumbent][w]
                checks[f"{w}:tps_regress<=5%"] = c["decode_tps_median"] >= 0.95 * i["decode_tps_median"]
                checks[f"{w}:ttft_rise<=5%"] = c["ttft_ms_median"] <= 1.05 * i["ttft_ms_median"]
                if c["peak_mem_gb"] and i["peak_mem_gb"]:
                    checks[f"{w}:mem_rise<=5%"] = c["peak_mem_gb"] <= 1.05 * i["peak_mem_gb"]
            out["incumbent_geomean_runs"] = gi; out["checks"] = checks; out["accept"] = all(checks.values())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--candidate"); ap.add_argument("--incumbent")
    a = ap.parse_args()
    r = report(json.load(open(a.path)), a.candidate, a.incumbent)
    print(r["table"]); print(json.dumps({k: v for k, v in r.items() if k != "table"}, indent=1, default=str))
