# Qwen3.5-2B batch-1 decode optimization on one H100 (Fable-directed, fixed budget)

Experiment: *what can a Fable-directed coding team accomplish under a fixed budget* ($100 Fable credits, $10 Modal,
DeepSeek V4.1 Flash via an existing OpenCode subscription) on a concrete target: maximize **batch-size-1 greedy decode
tokens/s** for `Qwen/Qwen3.5-2B` (revision `15852e8c`) on one Modal **H100**, keeping the original **BF16 weights**,
text-only, non-thinking mode, and byte-identical semantics (greedy outputs preserved up to bf16 kernel-order noise).

**Result summary.** A ~600-line custom engine (`qwen35_fast/engine.py` + two Triton kernels in
`qwen35_fast/gdn_step.py`): one CUDA graph per decode step, fused decode blocks via `torch.compile`, a fused
Gated-DeltaNet decode kernel, and *exact* speculative decoding with the checkpoint's own MTP head (2 drafts).
Frozen config `k2_compile_fused` reaches **582-848 decode TPS** on the 12-workload dev suite and **528-866** on the
held-out suite (H100, batch 1, BF16, greedy): **14x HF transformers eager** (50 TPS), 3.4-5x HF + torch.compile,
1.35-1.95x vLLM 0.29 without speculation, and **1.02x (dev) / 1.01x (held-out) geomean of vLLM 0.29 with its
built-in MTP speculative decoding** measured back-to-back in the same session (wins on prose and code prompts by
8-14%, loses on structured prompts by 7-11% where vLLM accepts more draft tokens per step). Output quality is
unchanged within noise (IFEval-100 strict 0.65 vs 0.61 for HF; differences are greedy near-tie flips). The 2,000 TPS
stretch hypothesis was **not** reached and is not reachable at BF16 on this GPU: streaming ~3.8 GB of weights per
verification step bounds non-speculative decode near ~880 TPS and MTP speculation near ~1,100-1,300 TPS even at 100%
of HBM bandwidth (see "Roofline"). Gains attributed separately (H100, dev suite medians):

| Source of gain | Decode TPS | Notes |
|---|---|---|
| Hardware (L4 -> H100) | not measured on L4 | plan changed to H100 before any L4 run; L4 (300 GB/s) roofline is ~75 TPS, i.e. ~11x lower than H100's |
| Original model, HF transformers eager (+fla kernels) | 50 | Python-overhead bound (24 layers x ~60 kernels, no graphs) |
| Enabling existing feature: HF `torch.compile` (default mode) | 166-171 | `reduce-overhead` mode segfaults on the hybrid cache |
| Enabling existing engine: vLLM 0.29 (CUDA graphs, compiled) | 429-437 | |
| Enabling existing feature: vLLM + MTP speculative decoding k=1/2/3 | 527-914 | strongest original baseline per workload (k=2 or k=3) |
| Generated code: custom engine, no speculation (`custom_k0`) | 388-390 | 0.9x vLLM plain; 7.7x HF eager |
| Generated code: + exact MTP speculation, 2 drafts, lean decode/spec glue (`k2_compile`) | 540-737 | 1.25-1.7x vLLM plain; 0.88x vLLM+MTP (same session) |
| Generated code: + fused GDN decode kernel (`k2_compile_fused`, **frozen config**) | 582-848 | **1.02x vLLM+MTP (dev), 1.01x (held-out)**; 14x HF eager |
| Generated code: 3 drafts (`custom_k3`, non-fused) | 469-788 | 0.91x vLLM+MTP; rejected by the per-workload 5% rule (slower than k2 on prose) |

## Attribution
* **Claude Fable 5.1** (coordinator): all design, the engine, evaluator, workloads, benchmarks, most bug fixes, this README.
* **DeepSeek V4.1 Flash** (via OpenCode, flat-rate subscription): code review of the engine against the HF reference
  (`candidates/review1/REVIEW.md`, two actionable findings adopted), and the Triton skinny-GEMV kernel
  `qwen35_fast/gemv.py` (Fable fixed two bugs: SMEM-oversized tiles and a store-mask shape; kept only where it beats cuBLAS).
* **Human** (islamborghini): task definition, budget, repository.

## Layout
```
qwen35_fast/engine.py   custom engine: weights from safetensors, static caches, CUDA graph step, exact MTP spec decode
qwen35_fast/gemv.py     Triton bf16 skinny GEMV (DeepSeek) with self-test; selected per weight shape only if faster
qwen35_fast/hf_wrapper.py  adapter giving the ORIGINAL HF model the same generate() API (used by the IFEval harness)
bench/workloads.py/.json   frozen prompts as exact token ids: dev + heldout x {prose, code, structured} x {128,512,2048,8192}
bench/evaluate.py       FROZEN evaluator (per-token timestamps, sequential protocol, drift check); bench/report.py = criteria
bench/optimize.py       bounded, resumable optimization loop over engine configs (correctness gate -> screen -> accept)
bench/ifeval.py         100-prompt IFEval regression via lm-evaluation-harness through the actual engines
bench/session.py        Modal H100 runner:  modal run bench/session.py --script bench/<x>.py --args "..."
bench/feasibility.py    first H100 feasibility benchmark (HF eager / vLLM / vLLM+MTP)
LEDGER.md               spending + candidate ledger;  results/  compact JSON measurements and logs
```

## Setup / reproduction
```
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install modal transformers jinja2 && modal token new
python bench/workloads.py                                   # regenerate frozen prompts (already committed)
modal run bench/feasibility.py                              # ~13 min H100: HF eager vs vLLM vs vLLM+MTP (512 in / 256 out)
modal run bench/session.py --script bench/run_baselines.py  # full 12-workload suite for hf_eager, hf_compile, vLLM, vLLM+MTP k=1..3
modal run bench/session.py --script bench/run_final.py      # bounded optimization loop -> frozen config -> dev/heldout eval -> IFEval
modal run bench/session.py --script bench/optimize.py --args "--max_candidates 4 --max_minutes 20"   # resumable loop only
```
The image (`bench/session.py`) pins `vllm==0.29.0` (torch 2.13.0+cu130), `flash-linear-attention==0.5.2`, `transformers==5.17.0`,
`lm-eval==0.4.13`; model files are cached in the Modal volume `qwen35-hf-cache`.

## Model and architecture of the engine
Qwen3.5-2B is a hybrid: 24 layers in blocks of 3x Gated DeltaNet (linear attention, 16 heads of 128, conv kernel 4) +
1x full attention (8 q heads / 2 kv heads, head_dim 256, gated output, partial RoPE 25%), hidden 2048, MLP 6144,
vocabulary 248,320 with tied embeddings, plus a one-layer MTP head (`mtp.*`) and an unused vision tower.

`qwen35_fast/engine.py` (text only):
* loads the original bf16 tensors straight from the safetensors (no re-quantization, no re-casting); optional weight
  concatenation (q|k|v, gate|up, qkv|z|b|a) so each projection group is one GEMM (same bf16 math, one launch);
* prefill: fla `chunk_gated_delta_rule`, causal conv via `F.conv1d`, SDPA causal attention, exact final states;
* decode: static KV cache with a device-side length pointer, explicit small-T attention over the cache (the cuDNN
  masked-SDPA path was 400 us per call), fla `fused_recurrent_gated_delta_rule` for the GDN state, tiny in-place conv
  state, tied-embedding argmax; **the whole step is one captured CUDA graph** replayed with zero host inputs;
* `compile_blocks=True`: the pure-torch pieces between GEMMs/fla kernels (RMSNorms, gates, SiLU, RoPE, attention) are
  `torch.compile`d into single fused kernels (1391 -> 505 kernels per step);
* **exact MTP speculative decoding** (`spec_k` drafts): the MTP layer proposes `k` tokens; the target model verifies
  `k+1` tokens in one pass; a draft is accepted only if it equals the target argmax, so greedy output is unchanged.
  State rollback is device-side and exact: rejected tokens are undone by re-running the GDN recurrence with
  `g=0, beta=0` for rejected positions (exp(0)=1, +0 -> bit-identical state), conv state by an indexed gather, and the
  KV cache by the length pointer. Verification + commit + drafting is a single CUDA graph; the host reads the
  accepted tokens with a one-step lag.

## Measurement boundaries (identical for every engine)
* `ttft_s` / prefill: request start -> first generated token on host (prefill + first argmax).
* `decode_tps = (n_gen - 1) / (t_last - t_first)` with token completion times observed on the host: HF via the
  generation streamer (its per-step `.cpu()` sync), vLLM via async stream chunk arrival, custom engine via a
  per-step event sync after the graph replay. Host-side and launch overhead included; no extra syncs inside steps.
  With speculative decoding several tokens complete at once; the tokens produced by the very first decode step
  are counted in the numerator but not the denominator (bias <= k/255 in the candidate's favour, ~1%).
* 256 generated tokens, EOS ignored, greedy, non-thinking chat template, no prefix caching, prompts as frozen token ids.
* Model load, torch.compile, CUDA-graph capture, and warmup are excluded and reported as startup.
* Peak memory: `torch.cuda.max_memory_allocated` (HF, custom); vLLM pre-allocates a KV pool (`gpu_memory_utilization`)
  so its "peak" is not comparable and is reported as N/A.

## Acceptance criteria (frozen in bench/report.py and bench/optimize.py)
Score = geometric mean over the 12 workloads of `candidate TPS / strongest-original TPS` (per workload). A candidate
replaces the incumbent only if its per-repetition geomean range does not overlap the incumbent's, no workload decode TPS
regresses > 5%, and no workload TTFT or peak memory rises > 5%. Correctness gate: prefill logits on identical prefixes
must agree in top-1 at every position (observed max abs logit diff ~0.3 on logits of magnitude ~35, i.e. ~1 bf16 ulp),
and multi-step greedy decode must agree with HF eager until the first position where HF's own top-2 margin is < 0.5
(bf16 kernel-order noise; vLLM itself diverges from HF at token 29 on the feasibility prompt for the same reason).

## Roofline (why 2,000 TPS is out of reach at BF16 on H100)
Per verification step the engine must stream ~2.74 GB of layer weights + 1.02 GB of tied lm_head (~3.76 GB). At the
H100's 3.35 TB/s that is >= 1.12 ms/step -> <= ~890 TPS without speculation. With `k` MTP drafts the step also pays
`k` extra lm_head reads for drafting (0.3 ms each) while yielding ~2.1-3.6 tokens/step at the measured acceptance
rates -> <= ~1,100-1,300 TPS even at 100% bandwidth and zero launch overhead. Reaching 2,000 TPS would require reading
fewer bytes per token (quantization, out of scope) or a larger accepted-tokens-per-step (a stronger draft head).

## Results
Rendered from `results/*.json` by `bench/fill_results.py` into `results/RESULTS.md` (all tables). Key figures:

* Dev suite, strongest original per workload = vLLM+MTP (k=2 or 3): 527-914 TPS. Frozen `k2_compile_fused`
  582-848 TPS, same-session geomean **1.02x** of vLLM+MTP k=3 (per workload 0.93-1.14x), 14x HF eager. Earlier
  accepted configs: `k2_compile` (no fused kernel) 0.88x; `custom_k0` 388-390 TPS (0.56x, 7.7x HF eager);
  `custom_k3` 0.91x. Full tables: `results/RESULTS.md`; earlier-round tables in `results/final4_k0k3/`.
* Held-out prompts (never used for tuning), same session: frozen config 528-866 TPS, **1.01x** of vLLM+MTP k=3
  (0.89-1.14x per workload).
* Run-to-run variability: custom engine medians move < 0.5% between repetitions and sessions; vLLM+MTP 1-5%.
* TTFT: custom 36-38 ms at 128-2048 input (vLLM 29-30 ms; the custom prefill is unoptimized), 111-112 ms at 8192
  (vLLM 126-129 ms). Peak memory 5.7 GB (vLLM pre-allocates its KV pool, not comparable).
* Startup: custom 55 s (load + torch.compile + graph capture), vLLM ~100 s, HF 13 s.
* IFEval-100 (prompt-level strict): HF 0.61, frozen custom_k2 0.65 (custom_k0 0.64, custom_k3 0.62); 9-12 prompts
  change outcome in both directions; only ~24/100 responses are byte-identical to HF because greedy near-ties flip under bf16 kernel-order
  noise and the texts then diverge (vLLM shows the same behaviour vs HF). No systematic regression.

### Correctness gate outcome (pre-registered, `bench/optimize.py`)
Gate: teacher-forced decode logits through the candidate's real decode path vs HF's own decode-path logits over 256
steps on six dev prompts; pass if max|diff| <= 2x HF's own prefill-vs-decode noise floor on that prompt (min 0.5) and
every top-1 disagreement sits at a genuine near-tie. Findings across seven sessions (`results/opt_ledger_round*.json`,
`results/gatecheck.json`):

* HF's own floor is 0.27-0.44 on prose/code prompts and 1.2-2.0 on the structured prompts (repetitive JSON-like
  continuations put the model in a numerically chaotic regime where the linear-attention state amplifies bf16 noise).
* The max statistic is a single-step outlier (always steps ~185-235 of the structured prompts); the 99th percentile
  over steps is 0.25-0.31 for every engine on prose/code and 0.8-1.2 on structured prompts, i.e. **indistinguishable
  from HF's own p99 (0.25-1.07)** for all configurations, including the ones the max-gate rejected.
* Consequently the gate verdict flips between sessions for identical code: the speculative configs were rejected by
  1.3% in one session and passed in three others; the non-speculative config passed four times and failed once
  (4.33 vs 4.0). Per the integrity rules the tolerance was never loosened; the ledger records every outcome, the
  accepted incumbent is the last candidate that passed in its own session (`k2_compile`, lean decode path), and
  candidates rejected by the max-gate are reported with their numbers but never promoted.
* Whole-layer `torch.compile` (with or without inductor's `emulate_precision_casts`) fails the max-gate on structured
  prompts (2.5-3.0 vs 2.4) and gains only 1-2%, so it was dropped. DeepSeek's two Triton GEMV kernels are correct but
  slower than cuBLAS except on the tied lm_head (+7%), so they were dropped. The first fused GDN decode-step Triton kernel
  (Fable worker T3) removed 144 launches/step but its in-place state write was corrupted (register-spill regime with a
  full 128x128 fp32 tile per program); the rewrite (T3b: two kernels, state tile per (head, 16 columns) held in
  registers across the T tokens, keep-mask for the exact commit) passes an exact test and is the `fused_gdn` option.

### Optimization ledger summary (dev screen: 512 + 2048 tokens, prose/code/structured, geomean decode TPS)
| candidate | what | gate | geomean TPS | verdict |
|---|---|---|---|---|
| k0_eager | CUDA graph, eager blocks | pass | 233-392 (sessions) | first incumbent |
| k0_compile | + fused elementwise blocks (torch.compile) | pass (4/5 sessions) | 390-410 | incumbent, then superseded |
| k2_compile | + exact MTP speculation, 2 drafts | pass (4/5) | 565-636 | accepted (lean + lean spec glue), then superseded |
| k3_compile | 3 drafts | pass (3/4) | 582-614 | rejected: >5% slower than k2 on prose (chain drafts cost more than they yield there) |
| k4_compile | 4 drafts | pass | 563-611 | rejected: slower on prose |
| k*_layer(_at)(_ep) | whole-layer compile (+max-autotune, +cast emulation) | fail / marginal | 577-584 | rejected |
| k*_gemv / gemv2 | Triton GEMV (DeepSeek v1/v2) per-shape selected | pass | = baseline | no gain |
| lean decode path (T1) | -50 launches/step, residual+norm fusion | pass | +2-3% | merged (default) |
| lean speculative glue (T1b) | static token buffer, fused acceptance, batched keep-mask/state writes, MTP-input norms fused | pass (p99 unchanged) | k2: 803->663 launches, -6% step | merged (default) |
| fused GDN step v1 (T3) | full 128x128 tile per program | test fails (register-spill regime) | (-6.6% step) | not enabled |
| k2_compile_fused (T3b kernel) | k2 + two-kernel fused GDN step (conv/silu/l2norm pre-kernel, 48 programs; delta-rule kernel per (head, 16-col) tile, T<=4, keep mask) | test exact (0/184k beyond 2 ulp); gate pass (2/2) | 665-670 (+12% vs paired k2) | **accepted frozen config** (`results/frozen_config.json`) |

## Costs and effort (estimates; see LEDGER.md for every session)
| Resource | Used | Notes |
|---|---|---|
| Modal H100 (+4 CPU) | ~4.5 GPU-hours, ~$17-19 estimated at $3.95/h + CPU | over the nominal $10 (the user authorized continued spend); ~35% went to sessions killed or wasted by my own bugs (missing dependency, wrong filename, a shadowed import, an orphaned app, contaminated baselines) |
| Claude Fable 5.1 | coordinator session + 4 forked workers (~0.4M tokens each) | $21 reported by the user at the mid-point; final figure in the user's dashboard |
| DeepSeek V4.1 Flash (OpenCode) | 5 tasks (2 reviews, 2 GEMV kernels, 1 test file) | flat-rate subscription, $0 marginal |

Startup overhead (excluded from decode TPS): custom engine 58 s (safetensors load ~10 s, torch.compile of the decode
blocks ~40 s, CUDA-graph capture ~2 s); vLLM ~100 s (compile + graph capture; cached artifacts on the volume);
HF eager 13 s. torch.compile/graph capture happens once per process.

## Limitations
* Batch size 1 only; no continuous batching, no sampling (greedy only), no multi-turn/prefix reuse, text only.
* The custom engine's prefill uses plain PyTorch/fla kernels and is not optimized (TTFT is reported, not tuned).
* `hf_compile` with `mode="reduce-overhead"` segfaults on this hybrid model (CUDA-graph trees vs in-place conv-state
  update); the default compile mode is what is reported.
* IFEval subset is a local regression check, not a reproduction of Qwen's published score.
