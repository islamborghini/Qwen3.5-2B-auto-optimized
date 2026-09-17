"""Fused Gated-DeltaNet decode step (batch 1, T<=4 tokens) in two Triton kernels (+ the engine's compiled gated norm).

A `_pre_kernel`  grid (3*H,): 4-tap causal conv over [conv_state(3) | x_0..x_{T-1}] + SiLU (bf16 rounding as eager),
                  fp32 l2norm of q/k (eps 1e-6), q*scale -> fp32 scratch [T, 3*H*D]. With WRITE_STATE it also shifts the
                  conv state by n_acc tokens (all loads before stores).
B `_rule_kernel` grid (H, D/BV): gated delta rule on the [K=D x BV] fp32 state tile kept in registers across the T
                  tokens; `keep[t]` in {0,1} scales g (log space) and beta so rejected speculative tokens leave the
                  state bit-identical (mirrors engine._commit); writes o (bf16, pre-norm) and, with WRITE_STATE, the state.
Verify pass (T>1): WRITE_STATE=0 (outputs only). Commit / T=1: WRITE_STATE=1, in place. No syncs, no allocations.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _win(x_ptr, sx, st_ptr, c, i: tl.constexpr):
    """Conv window value at position i (0..2 = conv state, 3.. = token i-3) for channels c."""
    if i < 3:
        return tl.load(st_ptr + c * 3 + i)
    else:
        return tl.load(x_ptr + (i - 3) * sx + c)


@triton.jit
def _pre_kernel(qkv_ptr, sx, nacc_ptr, st_ptr, w_ptr, scr_ptr, scale, T: tl.constexpr, D: tl.constexpr,
                H: tl.constexpr, WRITE_STATE: tl.constexpr):
    p = tl.program_id(0)
    grp = p // H
    c = p * D + tl.arange(0, D)          # channels of (group, head): q | k | v segments are each H*D wide
    w0 = tl.load(w_ptr + c * 4).to(tl.float32); w1 = tl.load(w_ptr + c * 4 + 1).to(tl.float32)
    w2 = tl.load(w_ptr + c * 4 + 2).to(tl.float32); w3 = tl.load(w_ptr + c * 4 + 3).to(tl.float32)
    for t in tl.static_range(T):
        acc = _win(qkv_ptr, sx, st_ptr, c, t).to(tl.float32) * w0 + _win(qkv_ptr, sx, st_ptr, c, t + 1).to(tl.float32) * w1 \
            + _win(qkv_ptr, sx, st_ptr, c, t + 2).to(tl.float32) * w2 + _win(qkv_ptr, sx, st_ptr, c, t + 3).to(tl.float32) * w3
        yb = acc.to(tl.bfloat16).to(tl.float32)                       # conv output rounded to bf16 (cuDNN/eager)
        y = (yb * tl.sigmoid(yb)).to(tl.bfloat16).to(tl.float32)      # silu on bf16 -> bf16
        nrm = tl.sqrt(tl.sum(y * y) + 1e-6)
        y = tl.where(grp < 2, y / nrm, y)                              # l2norm for q and k (fla, fp32)
        y = tl.where(grp == 0, y * scale, y)                           # q * D**-0.5
        tl.store(scr_ptr + t * (3 * H * D) + c, y)
    if WRITE_STATE:   # conv state <- window[n_acc : n_acc+3]
        nacc = tl.load(nacc_ptr)
        v0 = _win(qkv_ptr, sx, st_ptr, c, 0); v1 = _win(qkv_ptr, sx, st_ptr, c, 1); v2 = _win(qkv_ptr, sx, st_ptr, c, 2)
        for q in tl.static_range(1, 3 + T):
            wq = _win(qkv_ptr, sx, st_ptr, c, q)
            v0 = tl.where(nacc == q, wq, v0); v1 = tl.where(nacc + 1 == q, wq, v1); v2 = tl.where(nacc + 2 == q, wq, v2)
        tl.store(st_ptr + c * 3, v0); tl.store(st_ptr + c * 3 + 1, v1); tl.store(st_ptr + c * 3 + 2, v2)


@triton.jit
def _rule_kernel(scr_ptr, b_ptr, sb, a_ptr, sa, keep_ptr, nega_ptr, dtb_ptr, S_ptr, o_ptr, so,
                 T: tl.constexpr, D: tl.constexpr, H: tl.constexpr, BV: tl.constexpr, WRITE_STATE: tl.constexpr):
    h = tl.program_id(0)
    vb = tl.program_id(1)
    offk = tl.arange(0, D)
    offv = vb * BV + tl.arange(0, BV)
    nega = tl.load(nega_ptr + h)
    dtb = tl.load(dtb_ptr + h).to(tl.float32)
    Sp = S_ptr + h * D * D + offk[:, None] * D + offv[None, :]        # S[k, v] tile [D, BV]
    S = tl.load(Sp)
    for t in tl.static_range(T):
        keep = tl.load(keep_ptr + t)
        beta = tl.sigmoid(tl.load(b_ptr + t * sb + h).to(tl.float32)).to(tl.bfloat16).to(tl.float32) * keep
        ga = tl.load(a_ptr + t * sa + h).to(tl.float32) + dtb
        sp = tl.where(ga > 20.0, ga, libdevice.log1p(tl.exp(ga)))     # torch softplus
        eg = tl.exp(nega * sp * keep)
        base = scr_ptr + t * (3 * H * D)
        q = tl.load(base + h * D + offk)
        k = tl.load(base + H * D + h * D + offk)
        v = tl.load(base + 2 * H * D + h * D + offv)
        S = S * eg
        kv = tl.sum(S * k[:, None], 0)
        vn = beta * (v - kv)
        S += k[:, None] * vn[None, :]
        o = tl.sum(S * q[:, None], 0)
        tl.store(o_ptr + t * so + h * D + offv, o.to(tl.bfloat16))
    if WRITE_STATE:
        tl.store(Sp, S)


def gdn_step(qkv, b, a, keep, nacc, conv_state, conv_w, neg_expA, dt_bias, rec_state, scr, out, write_state=True):
    """qkv [T,3*H*D] bf16 (pre-conv rows, any row stride), b/a [T,H] bf16, keep fp32 [T], nacc int64 0-dim device
    tensor, conv_state [3*H*D,3] bf16, conv_w [3*H*D,1,4] bf16, neg_expA [H] fp32, dt_bias [H], rec_state [H,D,D] fp32,
    scr fp32 [>=T, 3*H*D] scratch, out [T,H*D] bf16 (pre-norm o). Returns out."""
    T = qkv.shape[0]
    H, D = rec_state.shape[0], rec_state.shape[1]
    _pre_kernel[(3 * H,)](qkv, qkv.stride(0), nacc, conv_state, conv_w, scr, D ** -0.5, T=T, D=D, H=H,
                          WRITE_STATE=int(write_state), num_warps=4)
    BV = 16
    _rule_kernel[(H, D // BV)](scr, b, b.stride(0), a, a.stride(0), keep, neg_expA, dt_bias, rec_state, out, out.stride(0),
                               T=T, D=D, H=H, BV=BV, WRITE_STATE=int(write_state), num_warps=4)
    return out


def test(n=3):
    """Compare against the engine's eager path (torch conv/silu + fla fused_recurrent) for T in 1..4 and keep masks."""
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    torch.manual_seed(0); dev = "cuda"; H, D = 16, 128; C = 3 * H * D
    worst_o = worst_s = 0.0; bad = 0; tot = 0
    scr = torch.empty(4, C, device=dev, dtype=torch.float32)
    for T in (1, 2, 3, 4):
        for nacc in range(1, T + 1):
            for i in range(n):
                qkv = torch.randn(T, C + 40, device=dev).bfloat16()[:, :C] * 2      # non-contiguous rows like the engine
                b = torch.randn(T, H, device=dev).bfloat16(); a = torch.randn(T, H, device=dev).bfloat16()
                cs = torch.randn(C, 3, device=dev).bfloat16(); cw = (torch.randn(C, 1, 4, device=dev) * 0.5).bfloat16()
                neg_expA = -torch.rand(H, device=dev).mul(4).exp(); dt_bias = torch.randn(H, device=dev).bfloat16()
                S = torch.randn(H, D, D, device=dev) * 0.3
                xc_all = torch.cat([cs[None], qkv.T[None]], -1)
                y = F.silu(F.conv1d(xc_all, cw, groups=C))[0].T
                q, k, v = [t.reshape(1, T, H, D) for t in y.split([H * D] * 3, -1)]
                beta = b.sigmoid()[None]; g = (neg_expA * F.softplus(a.float() + dt_bias))[None]
                o_ref, _ = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=S[None], output_final_state=False, use_qk_l2norm_in_kernel=True)
                keep = (torch.arange(T, device=dev) < nacc).float()
                _, st = fused_recurrent_gated_delta_rule(q, k, v, g=g * keep[None, :, None], beta=beta * keep[None, :, None].to(beta.dtype),
                                                         initial_state=S[None], output_final_state=True, use_qk_l2norm_in_kernel=True)
                S_ref = st[0]; cs_ref = xc_all[0, :, nacc: nacc + 3].contiguous(); o_ref = o_ref.reshape(T, -1)
                S_k, cs_k = S.clone(), cs.clone(); out = torch.empty(T, H * D, device=dev, dtype=torch.bfloat16)
                nacc_t = torch.tensor(nacc, device=dev)
                if T == 1:
                    gdn_step(qkv, b, a, keep, nacc_t, cs_k, cw, neg_expA, dt_bias, S_k, scr, out)
                else:   # verify pass (no state writes) then commit pass, as the engine does
                    gdn_step(qkv, b, a, torch.ones(T, device=dev), nacc_t, cs_k, cw, neg_expA, dt_bias, S_k, scr, out, write_state=False)
                    assert torch.equal(S_k, S) and torch.equal(cs_k, cs), "verify pass must not touch the state"
                    gdn_step(qkv, b, a, keep, nacc_t, cs_k, cw, neg_expA, dt_bias, S_k, scr, torch.empty_like(out))
                d = (out.float() - o_ref.float()).abs(); ulp = torch.pow(2.0, torch.floor(torch.log2(o_ref.float().abs().clamp_min(1e-30))) - 7)
                bad += (d > 2 * ulp + 1e-3).sum().item(); tot += d.numel()
                worst_o = max(worst_o, d.max().item()); worst_s = max(worst_s, ((S_k - S_ref).abs().max() / S_ref.abs().max()).item())
                assert torch.equal(cs_k, cs_ref), f"conv state mismatch T={T} nacc={nacc}"
    print(f"gdn_step.test: max|o diff|={worst_o:.4g}, elements beyond 2ulp: {bad}/{tot} ({100*bad/tot:.3f}%), state max rel diff={worst_s:.2e}", flush=True)
    assert bad / tot <= 1e-3 and worst_s <= 1e-5, "gdn_step tolerance exceeded"
    return worst_o, bad / tot, worst_s
