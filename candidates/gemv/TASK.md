Write `qwen35_fast/gemv.py`: a Triton kernel for skinny bf16 matmul `y = x @ W.T` used for batch-1 LLM decode.
- x: [M, K] bf16 with M in {1,2,3,4}; W: [N, K] bf16 row-major (nn.Linear layout); y: [M, N] bf16.
- fp32 accumulation; output cast to bf16 once at the end. No bias.
- Must be memory-bandwidth-bound-optimal on H100: read W exactly once, coalesced 16-byte loads along K,
  enough parallelism to saturate HBM3 (grid over N blocks; add SPLIT_K with an fp32 partial buffer + a tiny reduce
  kernel or atomic-free two-pass reduce when N*M is too small to fill the GPU, e.g. N=32).
- Shapes that matter (K, N): (2048, 8224) (2048, 12288) (6144, 2048) (2048, 5120) (2048, 2048) (4096, 2048)
  (2048, 248320). Also (2048, 32).
- API: `gemv(x, W) -> y` picking a config by (M, K, N) from a small static table (no Triton autotune at runtime;
  autotune once offline is fine if you expose `bench()` that prints the best config per shape).
- Must be safe to call inside CUDA graph capture (no host syncs, no data-dependent Python, no allocations that vary).
  Preallocate any split-K workspace per shape in a module-level cache keyed by (M,K,N,device).
- Also include `test()` that, on a GPU, compares against `torch.matmul(x, W.T)` for all shapes and M=1..4 with
  random inputs (atol/rtol chosen for bf16 output: assert max abs diff <= 2 bf16 ulps of the reference magnitude),
  and `bench()` that reports achieved GB/s vs torch.matmul for each shape (use CUDA events, 50 iters after warmup).
- No GPU is available here; write carefully, keep it under ~200 lines, plain Triton (triton 3.x), no other deps.
Do not modify any other file. Put a short usage note at the top of gemv.py.
