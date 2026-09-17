"""Freeze the benchmark prompts as exact token-id sequences (non-thinking chat template).
Run once: python bench/workloads.py  -> bench/workloads.json (dev + heldout splits).
"""
import json
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B"
REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
LENGTHS = [128, 512, 2048, 8192]

BODIES = {
    "dev": {
        "prose": ("The history of computing is a story of abstraction. Each generation of engineers built layers "
                  "that hid the complexity below: transistors became gates, gates became instructions, instructions "
                  "became languages, and languages became frameworks. ",
                  "Summarize the passage above in three paragraphs and explain its main argument."),
        "code": ("def merge_intervals(intervals):\n    intervals.sort(key=lambda x: x[0])\n    out = []\n"
                 "    for lo, hi in intervals:\n        if out and lo <= out[-1][1]:\n"
                 "            out[-1][1] = max(out[-1][1], hi)\n        else:\n            out.append([lo, hi])\n    return out\n\n",
                 "Review the code above. List any bugs, then write unit tests in Python."),
        "structured": ("Item: widget-{i}; quantity: {q}; price: {p}.\n",
                       "Using the inventory above, produce a JSON report with total quantity, total value, "
                       "and the three most expensive items. Use exactly these keys: total_quantity, total_value, top3."),
    },
    "heldout": {
        "prose": ("Rivers shape the civilizations along their banks. Floods deposit silt, trade follows the current, "
                  "and cities rise where the water slows enough to cross. ",
                  "Write an essay of about four paragraphs responding to the passage above."),
        "code": ("class LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.d = {}\n"
                 "    def get(self, k):\n        if k not in self.d: return -1\n        v = self.d.pop(k)\n"
                 "        self.d[k] = v\n        return v\n    def put(self, k, v):\n        self.d.pop(k, None)\n"
                 "        self.d[k] = v\n        if len(self.d) > self.cap: self.d.pop(next(iter(self.d)))\n\n",
                 "Explain the code above, point out edge cases, and rewrite it with type hints and docstrings."),
        "structured": ("Task {i}: owner=team-{q}; priority={p}; status=open.\n",
                       "From the task list above, produce a markdown table grouped by owner with counts per priority, "
                       "then a numbered list of the five highest-priority tasks."),
    },
}


def render_body(kind, unit, n_units):
    if kind == "structured":
        return "".join(unit.format(i=i, q=(i * 7) % 50 + 1, p=round(1.5 + (i * 13) % 97, 2)) for i in range(n_units))
    return unit * n_units


def build(tok, kind, unit, task, n_tokens):
    """Exact-length prompt: repeat unit, tokenize, trim filler tokens, re-template until exact."""
    body = render_body(kind, unit, 4000)
    fill = tok(body, add_special_tokens=False)["input_ids"]
    for suffix in ["", " Thanks.", " Please be concise.", " Be thorough.", "\n", " Thank you very much."]:
        n = n_tokens
        for _ in range(12):
            content = tok.decode(fill[:max(n, 1)]) + "\n\n" + task + suffix
            ids = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                          enable_thinking=False, tokenize=False)
            ids = tok(ids, add_special_tokens=False)["input_ids"]
            if len(ids) == n_tokens:
                return content, ids
            n += n_tokens - len(ids)
    raise RuntimeError(f"{kind}/{n_tokens}: got {len(ids)}")


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, token=False)
    out = {"model": MODEL, "revision": REV, "n_out": 256, "enable_thinking": False, "workloads": []}
    for split, kinds in BODIES.items():
        for kind, (unit, task) in kinds.items():
            for L in LENGTHS:
                content, ids = build(tok, kind, unit, task, L)
                out["workloads"].append({"id": f"{split}-{kind}-{L}", "split": split, "kind": kind,
                                         "input_len": L, "content": content, "ids": ids})
    json.dump(out, open("bench/workloads.json", "w"))
    print(len(out["workloads"]), "workloads;", [(w["id"], len(w["ids"])) for w in out["workloads"]][:4])
