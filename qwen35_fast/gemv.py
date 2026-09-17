"""gemv.py - batch-1 bf16 decode GEMV, y = x @ W.T, tuned for H100 HBM3 bandwidth.

    y = gemv(x, W)        # x:[M,K] bf16, W:[N,K] bf16 (nn.Linear layout), y:[M,N] bf16; M in 1..4
    test()                # GPU: correctness vs torch.matmul, all shapes, M=1..4
    bench()               # GPU: achieved GB/s vs torch.matmul per shape
    bench(autotune=True)  # offline: sweep configs and print the best per shape

Classic (no tl.dot) skinny GEMV: each program owns BLOCK_N rows of W and streams K in BLOCK_K
chunks with coalesced 16-byte loads; x is broadcast over the N rows and the K dot is a fp32
tl.sum. The M<=4 x-rows share one W tile (W read once); grid = N/BLOCK_N keeps all SMs busy
without tensor cores. When N*M cannot fill the GPU (N=32) SPLIT_K writes fp32 partials to a
cached workspace that a tiny second kernel reduces. Static (M,K,N) config table + heuristic
fallback: no runtime autotune, no host syncs, no variable-size allocations, CUDA-graph safe.
"""
import torch
import triton
import triton.language as tl

_MAX_TILE = 8192  # cap on BLOCK_M*BLOCK_N*BLOCK_K fp32 elements held live (register safety)


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, y_ptr, part_ptr, M, N, K, stride_xm, stride_wn, stride_ym,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_per = K // SPLIT_K
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mmask = offs_m < M
    nmask = offs_n < N
    x_base = x_ptr + offs_m[:, None] * stride_xm
    w_base = w_ptr + offs_n[:, None] * stride_wn
    k_base = pid_k * k_per
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k_per, BLOCK_K):
        kk = k_base + k0 + offs_k
        kmask = (k0 + offs_k) < k_per
        w = tl.load(w_base + kk[None, :], mask=nmask[:, None] & kmask[None, :], other=0.0)
        x = tl.load(x_base + kk[None, :], mask=mmask[:, None] & kmask[None, :], other=0.0)
        prod = x[:, None, :].to(tl.float32) * w[None, :, :].to(tl.float32)
        acc += tl.sum(prod, axis=2)
    out_mask = mmask[:, None] & nmask[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
                 acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(part_ptr + offs_m[:, None] * (SPLIT_K * N) + pid_k * N + offs_n[None, :],
                 acc, mask=out_mask)


@triton.jit
def _reduce_kernel(part_ptr, y_ptr, M, N, stride_ym, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_s = tl.arange(0, SPLIT_K)
    nmask = offs_n < N
    mmask = offs_m < M
    ptrs = (part_ptr + offs_m[:, None, None] * (SPLIT_K * N)
            + offs_s[None, :, None] * N + offs_n[None, None, :])
    part = tl.load(ptrs, mask=mmask[:, None, None] & nmask[None, None, :], other=0.0)
    acc = tl.sum(part, axis=1)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
             acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


def _npow2(x):
    return 1 << (x - 1).bit_length()


# (K, N) -> (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)
_BASE = {(2048, 248320): (16, 512, 1, 4, 3), (2048, 12288): (8, 512, 1, 4, 3),
         (2048, 8224): (8, 512, 1, 4, 3), (2048, 5120): (8, 512, 1, 4, 3),
         (2048, 2048): (8, 512, 2, 4, 3), (4096, 2048): (8, 512, 2, 4, 3),
         (6144, 2048): (8, 512, 2, 4, 3), (2048, 32): (8, 128, 32, 4, 2)}
# Static table keyed by (M, K, N); M only pads BLOCK_M, so reuse the (K, N) entries.
_TABLE = {(m, k, n): c for m in (1, 2, 3, 4) for (k, n), c in _BASE.items()}


def _heuristic(K, N):
    bn, bk, split = 8, 512, 1
    while (N // bn) * split < 264 and split < 64 and K % (2 * split) == 0 and K // (2 * split) >= 64:
        split *= 2
    return (bn, bk, split, 4, 3)


def _config(M, K, N):
    return _TABLE.get((M, K, N)) or _BASE.get((K, N)) or _heuristic(K, N)


_WS = {}


def _workspace(M, K, N, split, device):
    key = (M, K, N, split, str(device))
    if key not in _WS:
        _WS[key] = torch.empty((M, split, N), dtype=torch.float32, device=device)
    return _WS[key]


def _run(x, W, cfg):
    M, K = x.shape
    N = W.shape[0]
    bn, bk, split, warps, stages = cfg
    bm = _npow2(M)
    while bm * bn * bk > _MAX_TILE and bk > 64:  # keep the reduce tile in registers
        bk //= 2
    assert x.is_contiguous() and W.is_contiguous(), "gemv expects contiguous inputs"
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    part = y if split == 1 else _workspace(M, K, N, split, x.device)
    _gemv_kernel[(triton.cdiv(N, bn), split)](
        x, W, y, part, M, N, K, x.stride(0), W.stride(0), y.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, num_warps=warps, num_stages=stages,
    )
    if split > 1:
        rbn = min(256, _npow2(N))
        _reduce_kernel[(triton.cdiv(N, rbn),)](part, y, M, N, y.stride(0),
                                               BLOCK_M=bm, BLOCK_N=rbn, SPLIT_K=split, num_warps=4)
    return y


def gemv(x, W):
    """y = x @ W.T for x:[M,K] bf16, W:[N,K] bf16, M in 1..4. CUDA-graph-capture safe."""
    assert x.dtype == torch.bfloat16 and W.dtype == torch.bfloat16
    M, K = x.shape
    assert W.shape[1] == K
    return _run(x, W, _config(M, K, W.shape[0]))


_SHAPES = [(2048, 8224), (2048, 12288), (6144, 2048), (2048, 5120),
           (2048, 2048), (4096, 2048), (2048, 248320), (2048, 32)]


def _ulp_bf16(v):
    return torch.pow(2.0, torch.floor(torch.log2(v.abs().clamp_min(1e-30))) - 7)


def test():
    """On a GPU, check gemv against torch.matmul for every shape and M=1..4."""
    if not torch.cuda.is_available():
        print("test(): no CUDA GPU available")
        return
    torch.manual_seed(0)
    bad = 0
    for K, N in _SHAPES:
        for M in (1, 2, 3, 4):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            ref = torch.matmul(x, W.t()).float()
            diff = (gemv(x, W).float() - ref).abs()
            tol = 2.0 * _ulp_bf16(ref) + 1e-3  # 2 bf16 ulps of reference magnitude, small floor
            bad += (diff > tol).sum().item()
            print(f"K={K:6d} N={N:7d} M={M} max|d|={diff.max().item():.4g} "
                  f"worst={float((diff / tol).max()):.2f}x tol")
    assert bad == 0, f"{bad} element(s) exceeded the bf16 tolerance"
    print("test(): OK")


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


_CANDS = [(bn, bk, sp) for bn in (4, 8, 16, 32) for bk in (128, 256, 512) for sp in (1, 2, 4, 8)]


def _tune(x, W, nbytes, iters=30):
    M, K = x.shape
    N = W.shape[0]
    ref, best = torch.matmul(x, W.t()), None
    for bn, bk, sp in _CANDS:
        if bn > _npow2(N) or K % sp or K // sp < 16:
            continue
        cfg = (bn, bk, sp, 4, 3)
        try:
            ok = torch.allclose(_run(x, W, cfg), ref, atol=1e-2, rtol=1e-2)
        except Exception:
            continue
        if ok:
            gbps = nbytes / _time(lambda: _run(x, W, cfg), iters=iters, warmup=5) / 1e6
            best = (cfg, gbps) if best is None or gbps > best[1] else best
    return best


def bench(autotune=False):
    """Report achieved GB/s vs torch.matmul per shape (M=1). autotune=True sweeps configs."""
    if not torch.cuda.is_available():
        print("bench(): no CUDA GPU available")
        return
    torch.manual_seed(0)
    for K, N in _SHAPES:
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        nbytes = (N * K + K + N) * 2
        if autotune:
            best = _tune(x, W, nbytes)
            print(f"K={K:6d} N={N:7d}  best cfg={best[0]}  {best[1]:.1f} GB/s")
            continue
        ours = nbytes / _time(lambda: gemv(x, W)) / 1e6
        torchgbs = nbytes / _time(lambda: torch.matmul(x, W.t())) / 1e6
        print(f"K={K:6d} N={N:7d} M=1 cfg={_config(1, K, N)} "
              f"ours={ours:.1f} GB/s torch={torchgbs:.1f} GB/s")
