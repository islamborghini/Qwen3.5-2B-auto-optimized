Rewrite `qwen35_fast/gemv.py` (keep the same public API: gemv(x, W), test(), bench(); keep the file under ~200 lines).
Measured on H100 with the current version (M=1, GB/s achieved, ours vs torch cuBLAS):
  (K=2048,N=8224): 912 vs 2569   (2048,12288): 1391 vs 2405   (6144,2048): 698 vs 1622   (2048,5120): 580 vs 1618
  (2048,2048): 233 vs 663        (4096,2048): 470 vs 1088     (2048,248320): 3060 vs 2866   (2048,32): 3.8 vs 10.6
The tensor-core tl.dot design with BLOCK_M=16 padding is badly latency/occupancy bound for M=1.
Goal: >= 2.9 TB/s on the big shapes and >= 1.5x cuBLAS on the small (N=2048) shapes, for M in {1,2,3,4}.
Design hints: classic GEMV: each program owns BLOCK_N rows of W (e.g. 8-32 rows) and loops over K in BLOCK_K chunks
(e.g. 512-1024, 16-byte vector loads: make K-chunk contiguous per row), multiplies by x (broadcast, x kept in registers/
L1 as bf16->fp32), reduces with tl.sum over K (fp32), no tl.dot; grid = N/BLOCK_N (thousands of programs for N>=2048 so
all 132 SMs stay busy); for N=2048 use small BLOCK_N (8) -> 256 programs, or SPLIT_K=2 with an fp32 workspace + tiny
reduce. Handle M<=4 by looping over M rows inside the program (x rows loaded once). num_warps 4, num_stages 2-3.
Keep: fp32 accumulation, bf16 output, no host syncs, no data-dependent allocations (workspace cached per shape),
CUDA-graph safe, static config table by (K,N) + heuristic fallback, test() with the 2-ulp check, bench() printing GB/s.
No GPU here: write carefully; I will run test()/bench() on the H100. Do not modify other files.
