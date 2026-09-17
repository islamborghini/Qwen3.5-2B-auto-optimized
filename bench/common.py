import json, os, time
import torch

MODEL = "Qwen/Qwen3.5-2B"
REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
OUT = os.environ.get("OUT_DIR", "results")
WORKLOADS = json.load(open(os.path.join(os.path.dirname(__file__), "workloads.json")))


def model_path():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, revision=REV, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


def workloads(split="dev", lengths=None, kinds=None):
    return [w for w in WORKLOADS["workloads"] if w["split"] == split
            and (lengths is None or w["input_len"] in lengths) and (kinds is None or w["kind"] in kinds)]


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1)
    print("saved", name, flush=True)


def load_hf(device="cuda"):
    from transformers import Qwen3_5ForConditionalGeneration
    m = Qwen3_5ForConditionalGeneration.from_pretrained(model_path(), dtype=torch.bfloat16, device_map=device)
    return m.eval()


@torch.no_grad()
def hf_greedy(model, ids, n_out):
    """Greedy generation with HF; returns (gen_ids, ttft_s, decode_tps, total_s). Timestamps via streamer (sync per token)."""
    from transformers.generation.streamers import BaseStreamer

    class TS(BaseStreamer):
        def __init__(self): self.t = []
        def put(self, v): self.t.append(time.perf_counter())
        def end(self): pass
    x = torch.tensor([ids], device=model.device)
    torch.cuda.synchronize(); ts = TS(); t0 = time.perf_counter()
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n_out, min_new_tokens=n_out,
                         do_sample=False, streamer=ts)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    gen = out[0, x.shape[1]:].tolist()
    return gen, ts.t[1] - t0, (len(gen) - 1) / (ts.t[-1] - ts.t[1]), t1 - t0


@torch.no_grad()
def hf_logits(model, ids):
    x = torch.tensor([ids], device=model.device)
    return model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0].float()
