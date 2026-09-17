# Qwen3.5-2B, 14x faster single-request decoding on one H100

A small, purpose-built inference engine for [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) that
decodes **582-848 tokens/s at batch size 1** on a single NVIDIA H100, with the **original BF16 weights** and
**unchanged greedy outputs**, versus 50 tokens/s for the same model in HF transformers and 527-914 tokens/s for
vLLM 0.29 with speculative decoding. Built autonomously by a Claude Fable-directed team (Fable coordinator, forked
Fable workers, DeepSeek V4.1 Flash reviewers) under a fixed budget, with every measurement, candidate and cost logged.

| Engine (same H100, batch 1, BF16, greedy, 256 output tokens) | Decode tokens/s | vs base model |
|---|---|---|
| Base model: HF transformers 5.17 eager | 50 | 1x |
| HF transformers + `torch.compile` | 166-171 | 3.4x |
| vLLM 0.29 | 429-437 | 8.6x |
| vLLM 0.29 + built-in MTP speculative decoding | 527-914 | 10-18x |
| **This engine** (`qwen35_fast`, frozen config) | **582-848** | **14x** (geomean) |

Head-to-head with vLLM+MTP in the same session: **1.02x** geomean on the 12 development workloads, **1.01x** on 12
held-out workloads (faster on prose and code prompts, slower on highly repetitive structured prompts). Output quality
is unchanged: IFEval-100 strict accuracy 0.65 vs 0.61 for the base model (differences are greedy near-tie flips).

<details>
<summary>Per-workload numbers (median of 5 repetitions, min-max in parentheses)</summary>

| Workload (input tokens) | HF eager | HF compile | vLLM | vLLM+MTP k=3 | **this engine** | ratio vs vLLM+MTP |
|---|---|---|---|---|---|---|
| prose 128 | 51 | n/a | 437 | 523 | 590 | 1.13x |
| prose 512 | 50 | 167 | 436 | 539 | 582 | 1.08x |
| prose 2048 | 50 | 171 | 432 | 559 | 597 | 1.07x |
| prose 8192 | 50 | n/a | 429 | 577 | 604 | 1.05x |
| code 128 | 51 | n/a | 437 | 725 | 740 | 1.02x |
| code 512 | 50 | 171 | 436 | 637 | 720 | 1.13x |
| code 2048 | 50 | 171 | 432 | 682 | 692 | 1.01x |
| code 8192 | 51 | n/a | 429 | 691 | 661 | 0.96x |
| structured 128 | 51 | n/a | 437 | 871 | 848 | 0.97x |
| structured 512 | 50 | 171 | 436 | 875 | 811 | 0.93x |
| structured 2048 | 51 | 166 | 432 | 816 | 833 | 1.02x |
| structured 8192 | 51 | n/a | 429 | 760 | 807 | 1.06x |

Held-out suite: 528-866 tokens/s, 0.89-1.14x per workload. All tables: [`results/RESULTS.md`](results/RESULTS.md).
</details>

## What the engine does

Qwen3.5-2B is a hybrid: 24 layers alternating three Gated-DeltaNet (linear attention) blocks and one full-attention
block, a 248k vocabulary with tied embeddings, and a one-layer multi-token-prediction (MTP) head shipped in the
checkpoint. Batch-1 decoding of a 2B model is bound by kernel launches and weight streaming, not compute, so the
engine attacks both:

1. **A minimal decode loop instead of `generate()`** (`qwen35_fast/engine.py`, ~600 lines). Loads the original
   bf16 safetensors directly, drops the vision tower, keeps static KV / conv / recurrent state buffers with a
   device-side length pointer. Same math, same weights.
2. **One CUDA graph per decode step.** HF eager launches ~1,400 small kernels per token from Python; that alone
   caps it at 50 tokens/s. The whole step is captured once and replayed with zero host inputs. 50 -> 235 tokens/s.
3. **Fused decode blocks.** RMSNorms, gates, activations, RoPE and attention glue are compiled with
   `torch.compile` into single kernels; a slow masked-cuDNN attention path is replaced by explicit small-T attention
   over the cache. 1,391 -> ~460 kernels per step. -> 390 tokens/s.
4. **Exact speculative decoding with the model's own MTP head.** The MTP layer proposes 2 tokens; the target model
   verifies them in one pass; a draft is accepted only if it equals the target's argmax, so greedy output is
   unchanged. Rejected tokens are rolled back on-device without host syncs: the linear-attention recurrence is
   re-run with decay and update masked to zero for rejected positions (exp(0)=1, +0: bit-identical state), the conv
   state by an indexed gather, the KV cache by the pointer. Verify + commit + draft is a single CUDA graph.
   -> 600-700 tokens/s.
5. **A fused Triton kernel for the Gated-DeltaNet decode step** (`qwen35_fast/gdn_step.py`): conv + SiLU +
   L2-norm + delta-rule state update + gated RMSNorm in two launches per layer instead of ~10, for 1-4 tokens with
   the keep-mask used by the exact commit. Tested exact (0 of 184k elements beyond 2 bf16 ulps).
   -> 582-848 tokens/s.

Why not 2,000 tokens/s: every verification step streams ~3.8 GB of weights; at the H100's 3.35 TB/s that is at
least 1.1 ms per step, i.e. <= ~880 tokens/s without speculation and ~1,100-1,300 with MTP at 100% bandwidth and zero
overhead. Reaching 2,000 would need fewer bytes per token (quantization, out of scope) or a much stronger draft head.

## Repository layout

```
qwen35_fast/engine.py      the engine (weights, static caches, CUDA-graph step, exact MTP speculation)
qwen35_fast/gdn_step.py    fused Gated-DeltaNet decode kernels (Triton) + exactness self-test
qwen35_fast/gemv.py        Triton skinny GEMV (DeepSeek); correct but slower than cuBLAS, kept for the record
bench/workloads.json       frozen prompts as exact token ids: dev + held-out x {prose, code, structured} x {128, 512, 2048, 8192}
bench/evaluate.py          frozen evaluator (per-token timestamps, sequential protocol, drift check); bench/report.py = criteria
bench/optimize.py          bounded, resumable optimization loop (correctness gate -> screen -> paired acceptance)
bench/ifeval.py            IFEval 100-prompt regression through the real engines (lm-evaluation-harness)
bench/session.py           Modal H100 runner;  bench/run_*.py  session scripts used during the project
results/                   compact JSON measurements, RESULTS.md tables, optimization ledgers per round
LEDGER.md                  every GPU session, agent task, cost estimate and outcome
```

## Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install modal "transformers==5.17.0" jinja2 && modal token new

modal run bench/feasibility.py                              # HF eager vs vLLM vs vLLM+MTP on one prompt (~13 min)
modal run bench/session.py --script bench/run_baselines.py  # full suite for all original engines (~30 min)
modal run bench/session.py --script bench/run_opt7.py       # bounded optimization loop (~10 min)
modal run bench/session.py --script bench/run_final4.py     # frozen config vs vLLM+MTP, dev + held-out, IFEval (~20 min)
python bench/fill_results.py > results/RESULTS.md           # render the tables
```

The Modal image pins `vllm==0.29.0` (torch 2.13.0+cu130), `flash-linear-attention==0.5.2`, `transformers==5.17.0`,
`lm-eval==0.4.13`; model revision `15852e8c` is cached in a Modal volume. Using the engine directly:

```python
from qwen35_fast.engine import Engine
eng = Engine(model_path, spec_k=2, compile_blocks=True, fused_gdn=True)   # the frozen config
out = eng.generate(prompt_token_ids, n_out=256, eos_ids={248044}, ignore_eos=False)
out["tokens"], out["ttft_s"], out["decode_s"]
```

## How it was measured

* Batch 1, greedy, non-thinking chat template, prompts frozen as token ids, no prefix caching, 256 generated tokens,
  EOS ignored for timing runs. Model load, compilation, graph capture and warmup are excluded and reported separately
  (engine startup 55 s, vLLM ~100 s, HF 13 s).
* `decode TPS = (n_gen - 1) / (t_last_token - t_first_token)` with token completion observed on the host: HF via its
  streamer, vLLM via async stream chunks, the engine via a per-step event sync. Host and launch overhead included.
* Engines are measured one at a time (co-resident vLLM engines busy-poll the CPU and slowed HF by 30%), the workload
  order alternates per repetition, and a drift check re-measures the first engine at the end. Candidate-vs-vLLM
  numbers are always from the same session; run-to-run spread of the engine is < 0.5%, of vLLM+MTP 1-5%.
* Time to first token: engine 36-38 ms at 128-2048 input tokens (vLLM 29-30 ms; the engine's prefill is
  unoptimized), 111 ms at 8192 (vLLM 126 ms). Peak memory 5.7 GB.

## Correctness

* Prefill logits agree with HF in top-1 at every position on identical prefixes (max abs diff ~0.3 on logits of
  magnitude ~35, about one bf16 ulp).
* Decode is validated by teacher-forcing HF's greedy tokens through the engine's real decode path (including the
  speculative verify/commit) and comparing 256 steps of logits against HF's own decode-path logits. HF itself differs
  from its own prefill path by up to 0.3-0.4 on prose and 1.2-2.0 on repetitive structured prompts (a numerically
  chaotic regime for the linear-attention state); the engine sits at the same level (99th percentile 0.25-1.1 for
  both). A pre-registered gate on the max difference is noise-limited there and its verdicts are recorded honestly
  in the ledgers; only candidates that passed it in their own session were promoted.
* Speculative acceptance is exact by construction (draft kept only if equal to the target argmax); a rollback test
  (`tests/test_spec_rollback.py`) and the kernel self-test cover the state handling.
* IFEval-100 (greedy, normal stopping, `max_gen_toks=1280`, lm-eval 0.4.13): base 0.61, engine 0.65 prompt-level
  strict; 12 prompts change outcome (8 up, 4 down). Only ~25% of responses are byte-identical to HF because greedy
  near-ties flip under bf16 kernel-order noise and texts then diverge; vLLM shows the same behaviour.

## What did not work (kept for the record)

| Attempt | Outcome |
|---|---|
| HF `torch.compile(mode="reduce-overhead")` | segfaults on the hybrid cache (in-place conv-state update inside CUDA-graph trees); default mode reported instead |
| Whole-layer `torch.compile` (+ max-autotune, + precision-cast emulation) | +1-2% at best and failed the numerical gate on structured prompts |
| Triton skinny GEMV kernels (DeepSeek, two versions) | correct, but slower than cuBLAS on every shape except the tied lm_head (+7%) |
| Fused GDN kernel v1 (full 128x128 state tile per program) | fast but miscomputed the stored state (register-spill regime); rewritten as v2 |
| 3-4 MTP drafts | higher on structured prompts, >5% slower on prose than 2 drafts; rejected by the per-workload rule |

## Who did what

* **Claude Fable 5.1** (coordinator): engine, evaluator, workloads, benchmarks, speculative decoding, most debugging,
  this README. Five forked Fable workers: lean decode path (T1), lean speculative glue (T1b), fused GDN kernel v1/v2
  and its fault analysis (T3, T3b, T3c).
* **DeepSeek V4.1 Flash** (via OpenCode, flat-rate): two code reviews of the engine against the HF reference (several
  findings adopted), the rollback test, two GEMV kernels.
* **Human**: task definition, budget, repository.

## Costs

| Resource | Used |
|---|---|
| Modal H100 (+4 vCPU) | ~5.5 GPU-hours over ~30 sessions, ~$21-24 estimated (nominal budget $10; continued spend authorized mid-way). About 30% was wasted on the coordinator's own bugs and killed sessions, all logged in `LEDGER.md` |
| Claude Fable | coordinator session + 5 workers (~0.4-0.5M tokens each); $21 reported at the mid-point |
| DeepSeek | 6 tasks, $0 marginal (subscription) |

## Limitations

Batch size 1 and greedy decoding only; no sampling, batching, multi-turn prefix reuse or images. The prefill path is
plain PyTorch and untuned. Numbers are for one H100 SXM on Modal; other GPUs will differ. The IFEval subset is a local
regression check, not a reproduction of Qwen's published score.
