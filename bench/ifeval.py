"""IFEval 100-prompt subset regression check via lm-evaluation-harness (task `ifeval`), greedy, non-thinking template,
normal stopping. Runs the ORIGINAL model (HF eager) or the OPTIMIZED custom engine through a custom LM wrapper so the
harness actually executes the optimized implementation. Saves prompt-level strict outcomes for comparison.
Usage: python bench/ifeval.py --engine hf|custom_k3 [--limit 100]
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import *
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import lm_eval

MAX_GEN = 1280   # lm-eval ifeval default max_gen_toks
DATASET_REV = "main"


@register_model("qwen35_custom")
class CustomLM(LM):
    def __init__(self, engine, tok):
        super().__init__(); self.e = engine; self.tok = tok
        self.eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
        self.truncated = 0
    def generate_until(self, requests, disable_tqdm=False):
        outs = []
        for i, req in enumerate(requests):
            ctx, kw = req.args
            ids = self.tok(ctx, add_special_tokens=False)["input_ids"]
            o = self.e.generate(ids, kw.get("max_gen_toks", MAX_GEN), eos_ids=self.eos, ignore_eos=False)
            toks = [t for t in o["tokens"] if t not in self.eos]
            if len(o["tokens"]) >= kw.get("max_gen_toks", MAX_GEN): self.truncated += 1
            text = self.tok.decode(toks, skip_special_tokens=True)
            for s in kw.get("until", []):
                text = text.split(s)[0]
            outs.append(text); print(f"[{i+1}/{len(requests)}] {len(toks)} toks", flush=True)
        return outs
    def loglikelihood(self, requests): raise NotImplementedError
    def loglikelihood_rolling(self, requests): raise NotImplementedError
    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tok.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    @property
    def tokenizer_name(self): return "qwen35"
    @property
    def chat_template(self): return self.tok.chat_template


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True); ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path())
    t = time.perf_counter()
    if a.engine == "hf":
        from qwen35_fast.hf_wrapper import HFGen
        eng = HFGen(load_hf())
    else:
        from qwen35_fast.engine import Engine
        eng = Engine(model_path(), spec_k=int(a.engine.split("_k")[1]), max_len=4096)
    lm = CustomLM(eng, tok)
    r = lm_eval.simple_evaluate(model=lm, tasks=["ifeval"], limit=a.limit, apply_chat_template=True, log_samples=True,
                                gen_kwargs="max_gen_toks=1280,do_sample=False", random_seed=0)
    samples = [{"doc_id": s["doc_id"], "prompt_strict": s["prompt_level_strict_acc"], "inst_strict": s["inst_level_strict_acc"],
                "prompt_loose": s["prompt_level_loose_acc"], "resp": s["resps"][0][0]} for s in r["samples"]["ifeval"]]
    out = {"engine": a.engine, "limit": a.limit, "results": r["results"]["ifeval"], "truncated": lm.truncated,
           "elapsed_s": time.perf_counter() - t, "lm_eval": lm_eval.__version__, "samples": samples,
           "config": {"max_gen_toks": 1280, "enable_thinking": False, "greedy": True, "dataset": "google/IFEval", "dataset_rev": DATASET_REV}}
    save(f"ifeval_{a.engine}.json", out)
    print(json.dumps(r["results"]["ifeval"], indent=1))
