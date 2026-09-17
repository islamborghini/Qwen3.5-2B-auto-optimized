"""Feasibility benchmark on one H100: HF eager vs vLLM vs vLLM+MTP, batch 1, 512-token prompt, 256 output tokens.
Run: modal run bench/feasibility.py
"""
import json, os, time
import modal

MODEL = "Qwen/Qwen3.5-2B"
REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
N_OUT = 256

app = modal.App("qwen35-feas")
vol = modal.Volume.from_name("qwen35-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.29.0", "flash-linear-attention==0.5.2", "hf_transfer==0.1.9", "accelerate==1.15.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf", "VLLM_CACHE_ROOT": "/hf/vllm_cache"})
)


def build_prompt(tok, n_tokens: int, filler: str):
    """Chat-templated (non-thinking) prompt whose token count is exactly n_tokens."""
    fill_ids = tok(filler * 200, add_special_tokens=False)["input_ids"]
    n = n_tokens
    for _ in range(30):
        text = tok.decode(fill_ids[:n])
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=False, tokenize=False)
        ids = tok(ids, add_special_tokens=False)["input_ids"]
        if len(ids) == n_tokens:
            return ids
        n += n_tokens - len(ids)
    raise RuntimeError(f"could not hit {n_tokens} tokens, got {len(ids)}")


@app.function(gpu="H100", image=image, volumes={"/hf": vol}, timeout=2400)
def feas():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    t = time.perf_counter()
    snapshot_download(MODEL, revision=REV, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    vol.commit()
    res["download_s"] = time.perf_counter() - t
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    prompt = build_prompt(tok, 512, "The history of computing is a story of abstraction. ")
    res["prompt_len"] = len(prompt)
    print(json.dumps(res), flush=True)

    # ---------------- HF transformers eager ----------------
    try:
        import transformers
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.generation.streamers import BaseStreamer
        try:
            import fla; res["fla"] = fla.__version__
        except Exception as e:
            res["fla"] = f"missing: {e}"
        res["transformers"] = transformers.__version__

        class TS(BaseStreamer):
            def __init__(self): self.t = []
            def put(self, v): self.t.append(time.perf_counter())
            def end(self): pass

        t = time.perf_counter()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        res["hf_load_s"] = time.perf_counter() - t
        ids = torch.tensor([prompt], device="cuda")
        hf_runs = []
        for i in range(4):
            torch.cuda.synchronize(); ts = TS(); t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=N_OUT,
                                     min_new_tokens=N_OUT, do_sample=False, streamer=ts)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            gen = out[0, ids.shape[1]:].tolist()
            hf_runs.append({"ttft_s": ts.t[1] - t0, "decode_tps": (len(gen) - 1) / (ts.t[-1] - ts.t[1]),
                            "total_s": t1 - t0, "n_gen": len(gen)})
        res["hf_eager"] = hf_runs[1:]
        res["hf_greedy_ids"] = gen
        res["hf_text"] = tok.decode(gen[:60])
        res["hf_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(json.dumps({k: res[k] for k in ("hf_eager", "hf_text", "fla", "transformers", "hf_peak_mem_gb")}), flush=True)
        del model; import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        import traceback; res["hf_error"] = traceback.format_exc(); print(res["hf_error"], flush=True)

    # ---------------- vLLM ----------------
    from vllm import LLM, SamplingParams
    import vllm
    res["vllm"] = vllm.__version__

    def bench_vllm(tag, **kw):
        t = time.perf_counter()
        llm = LLM(model=MODEL, revision=REV, dtype="bfloat16", max_model_len=9216, gpu_memory_utilization=0.6,
                  limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, **kw)
        init_s = time.perf_counter() - t
        sp1 = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        spN = SamplingParams(temperature=0, max_tokens=N_OUT, ignore_eos=True)
        req = [{"prompt_token_ids": prompt}]
        llm.generate(req, spN)  # warmup
        runs = []
        for i in range(5):
            t = time.perf_counter(); llm.generate(req, sp1); t1 = time.perf_counter() - t
            t = time.perf_counter(); o = llm.generate(req, spN); tN = time.perf_counter() - t
            gen = list(o[0].outputs[0].token_ids)
            runs.append({"ttft_s": t1, "decode_tps": (len(gen) - 1) / (tN - t1), "total_s": tN, "n_gen": len(gen)})
        r = {"init_s": init_s, "runs": runs, "ids": gen, "text": tok.decode(gen[:60])}
        if "hf_greedy_ids" in res:
            h = res["hf_greedy_ids"]; m = next((i for i in range(min(len(h), len(gen))) if h[i] != gen[i]), None)
            r["first_mismatch_vs_hf"] = m
        res[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k != "ids"}), flush=True)
        del llm; import gc; gc.collect(); torch.cuda.empty_cache(); time.sleep(3)

    for tag, kw in [
        ("vllm_plain", {}),
        ("vllm_mtp1", {"speculative_config": {"method": "mtp", "num_speculative_tokens": 1}}),
        ("vllm_mtp2", {"speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}),
        ("vllm_mtp3", {"speculative_config": {"method": "mtp", "num_speculative_tokens": 3}}),
    ]:
        try:
            bench_vllm(tag, **kw)
        except Exception:
            import traceback; res[tag + "_error"] = traceback.format_exc(); print(res[tag + "_error"], flush=True)

    os.makedirs("/hf/results", exist_ok=True)
    json.dump(res, open("/hf/results/feasibility.json", "w"), indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def main():
    res = feas.remote()
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/feasibility.json", "w"), indent=1)
    print("saved results/feasibility.json")
