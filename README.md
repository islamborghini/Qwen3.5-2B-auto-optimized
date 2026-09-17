# Qwen3.5-2B batch-1 decode optimization on one H100 (Fable-directed, fixed budget)

Experiment: *what can a Fable-directed coding team accomplish under a fixed budget* ($100 Fable credits, $10 Modal,
DeepSeek V4.1 Flash via an existing OpenCode subscription) on a concrete target: maximize **batch-size-1 greedy decode
tokens/s** for `Qwen/Qwen3.5-2B` (revision `15852e8c`) on one Modal **H100**, keeping the original **BF16 weights**,
text-only, non-thinking mode, and byte-identical semantics (greedy outputs preserved up to bf16 kernel-order noise).

**Result summary** (numbers filled from `results/`; see "Results"): a ~370-line custom engine (`qwen35_fast/engine.py`)
with one CUDA graph per decode step, fused decode blocks via `torch.compile`, and *exact* speculative decoding with the
checkpoint's own MTP head, reaches __TBD__ TPS (geomean over the 12-workload suite) versus __TBD__ for the strongest
original engine (vLLM 0.29 + MTP) and 55 TPS for HF transformers eager. The 2,000 TPS stretch hypothesis was **not**
reached and is not reachable at BF16 on this GPU: streaming the ~3.8 GB of weights per verification step bounds
non-speculative decode near ~880 TPS and MTP speculation near ~1,100 TPS at 100% of HBM bandwidth (see "Roofline").

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
__TBD: results table (dev suite), same-session comparison, held-out suite, IFEval, startup costs, spending__

## Limitations
* Batch size 1 only; no continuous batching, no sampling (greedy only), no multi-turn/prefix reuse, text only.
* The custom engine's prefill uses plain PyTorch/fla kernels and is not optimized (TTFT is reported, not tuned).
* `hf_compile` with `mode="reduce-overhead"` segfaults on this hybrid model (CUDA-graph trees vs in-place conv-state
  update); the default compile mode is what is reported.
* IFEval subset is a local regression check, not a reproduction of Qwen's published score.
