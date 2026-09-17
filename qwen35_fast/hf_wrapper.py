"""Adapter so the ORIGINAL HF model exposes the same generate(ids, n_out, eos_ids, ignore_eos) API as Engine."""
import time, torch


class HFGen:
    def __init__(self, model): self.m = model
    @torch.no_grad()
    def generate(self, ids, n_out, eos_ids=(), ignore_eos=False, **_):
        x = torch.tensor([ids], device=self.m.device); t0 = time.perf_counter()
        kw = dict(max_new_tokens=n_out, do_sample=False)
        if ignore_eos: kw["min_new_tokens"] = n_out
        else: kw["eos_token_id"] = list(eos_ids)
        out = self.m.generate(input_ids=x, attention_mask=torch.ones_like(x), **kw)
        return {"tokens": out[0, x.shape[1]:].tolist(), "total_s": time.perf_counter() - t0}
