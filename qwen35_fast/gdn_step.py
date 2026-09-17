"""Fused Gated-DeltaNet decode step (batch 1, T<=4 tokens) as ONE Triton kernel, one program per head.

Per token: 4-tap causal conv over [conv_state(3) | x_0..x_{T-1}] + SiLU (bf16 rounding as eager), fp32 l2norm of q/k,
q*scale, gated delta rule on the [K x V] fp32 state streamed in BK-row chunks (low register pressure, fla-style),
gated RMSNorm (Qwen3_5RMSNormGated op order). `keep[t]` in {0,1} multiplies g (log space) and beta, so rejected
speculative tokens leave the state bit-identical (mirrors engine._commit). WRITE_STATE=0 writes the running state to a
scratch buffer (verify pass); WRITE_STATE=1 writes in place and shifts the conv state by n_acc tokens (commit / T=1).
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
def _conv_silu(x_ptr, sx, st_ptr, w_ptr, c, t: tl.constexpr):
    w0 = tl.load(w_ptr + c * 4).to(tl.float32); w1 = tl.load(w_ptr + c * 4 + 1).to(tl.float32)
    w2 = tl.load(w_ptr + c * 4 + 2).to(tl.float32); w3 = tl.load(w_ptr + c * 4 + 3).to(tl.float32)
    acc = _win(x_ptr, sx, st_ptr, c, t).to(tl.float32) * w0 + _win(x_ptr, sx, st_ptr, c, t + 1).to(tl.float32) * w1 \
        + _win(x_ptr, sx, st_ptr, c, t + 2).to(tl.float32) * w2 + _win(x_ptr, sx, st_ptr, c, t + 3).to(tl.float32) * w3
    yb = acc.to(tl.bfloat16).to(tl.float32)                      # conv output rounded to bf16 (cuDNN/eager)
    return (yb * tl.sigmoid(yb)).to(tl.bfloat16).to(tl.float32)  # silu on bf16 -> bf16


@triton.jit
def _gdn_kernel(qkv_ptr, sx, z_ptr, sz, b_ptr, sb, a_ptr, sa, keep_ptr, nacc_ptr, st_ptr, w_ptr, nega_ptr, dtb_ptr,
                gn_ptr, S_in, S_out, o_ptr, so, scale, eps,
                T: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, KOFF: tl.constexpr, VOFF: tl.constexpr,
                WRITE_STATE: tl.constexpr):
    h = tl.program_id(0)
    offs = tl.arange(0, D)
    ck = tl.arange(0, BK)
    nega = tl.load(nega_ptr + h)
    dtb = tl.load(dtb_ptr + h).to(tl.float32)
    gw = tl.load(gn_ptr + offs).to(tl.float32)
    cq = h * D + offs
    ckk = KOFF + h * D + offs
    cv = VOFF + h * D + offs
    for t in tl.static_range(T):
        keep = tl.load(keep_ptr + t)
        beta = tl.sigmoid(tl.load(b_ptr + t * sb + h).to(tl.float32)).to(tl.bfloat16).to(tl.float32) * keep
        ga = tl.load(a_ptr + t * sa + h).to(tl.float32) + dtb
        sp = tl.where(ga > 20.0, ga, libdevice.log1p(tl.exp(ga)))            # torch softplus
        eg = tl.exp(nega * sp * keep)
        qf = _conv_silu(qkv_ptr, sx, st_ptr, w_ptr, cq, t)
        kf = _conv_silu(qkv_ptr, sx, st_ptr, w_ptr, ckk, t)
        v = _conv_silu(qkv_ptr, sx, st_ptr, w_ptr, cv, t)
        qn = tl.sqrt(tl.sum(qf * qf) + 1e-6)
        kn = tl.sqrt(tl.sum(kf * kf) + 1e-6)
        # pass 1: kv[v] = sum_k S[k,v] * exp(g) * k[k]
        kv = tl.zeros([D], dtype=tl.float32)
        for c0 in tl.static_range(0, D, BK):
            kc = _conv_silu(qkv_ptr, sx, st_ptr, w_ptr, KOFF + h * D + c0 + ck, t) / kn
            if t == 0:
                Sc = tl.load(S_in + h * D * D + (c0 + ck)[:, None] * D + offs[None, :]) * eg
            else:
                Sc = tl.load(S_out + h * D * D + (c0 + ck)[:, None] * D + offs[None, :]) * eg
            kv += tl.sum(Sc * kc[:, None], 0)
        vn = beta * (v - kv)
        # pass 2: S_c = S_c*exp(g) + k_c (x) vn ; o += S_c^T q_c ; store chunk
        o = tl.zeros([D], dtype=tl.float32)
        for c0 in tl.static_range(0, D, BK):
            kc = _conv_silu(qkv_ptr, sx, st_ptr, w_ptr, KOFF + h * D + c0 + ck, t) / kn
            qc = _conv_silu(qkv_ptr, sx, st_ptr, w_ptr, h * D + c0 + ck, t) / qn * scale
            if t == 0:
                Sc = tl.load(S_in + h * D * D + (c0 + ck)[:, None] * D + offs[None, :]) * eg
            else:
                Sc = tl.load(S_out + h * D * D + (c0 + ck)[:, None] * D + offs[None, :]) * eg
            Sc += kc[:, None] * vn[None, :]
            o += tl.sum(Sc * qc[:, None], 0)
            tl.store(S_out + h * D * D + (c0 + ck)[:, None] * D + offs[None, :], Sc)
        tl.debug_barrier()   # next token reads S_out written by other threads of this program
        # gated RMSNorm: bf16 o, fp32 norm, bf16 * weight, * silu(z fp32), -> bf16
        ob = o.to(tl.bfloat16).to(tl.float32)
        hb = (ob * tl.rsqrt(tl.sum(ob * ob) / D + eps)).to(tl.bfloat16).to(tl.float32)
        hw = (gw * hb).to(tl.bfloat16).to(tl.float32)
        z = tl.load(z_ptr + t * sz + h * D + offs).to(tl.float32)
        tl.store(o_ptr + t * so + h * D + offs, (hw * (z * tl.sigmoid(z))).to(tl.bfloat16))
    if WRITE_STATE:   # conv state <- window[n_acc : n_acc+3] (all loads before any store)
        nacc = tl.load(nacc_ptr)
        for base in tl.static_range(3):
            c = base * KOFF + h * D + offs
            v0 = _win(qkv_ptr, sx, st_ptr, c, 0); v1 = _win(qkv_ptr, sx, st_ptr, c, 1); v2 = _win(qkv_ptr, sx, st_ptr, c, 2)
            for p in tl.static_range(1, 3 + T):
                wp = _win(qkv_ptr, sx, st_ptr, c, p)
                v0 = tl.where(nacc == p, wp, v0); v1 = tl.where(nacc + 1 == p, wp, v1); v2 = tl.where(nacc + 2 == p, wp, v2)
            tl.store(st_ptr + c * 3, v0); tl.store(st_ptr + c * 3 + 1, v1); tl.store(st_ptr + c * 3 + 2, v2)


def gdn_step(qkv, z, b, a, keep, nacc, conv_state, conv_w, neg_expA, dt_bias, gnorm_w, rec_state, rec_state_out, out,
             eps=1e-6, write_state=True):
    """qkv [T,3*H*D] bf16 (pre-conv rows, any row stride), z [T,H*D] bf16, b/a [T,H] bf16, keep fp32 [T] (1/0),
    nacc int64 0-dim device tensor (tokens to commit into the conv state), conv_state [3*H*D,3] bf16, conv_w [3*H*D,1,4]
    bf16, neg_expA [H] fp32, dt_bias [H], gnorm_w [D] bf16, rec_state [H,D,D] fp32 (read), rec_state_out [H,D,D] fp32
    (written; may alias rec_state), out [T,H*D] bf16. CUDA-graph safe (no syncs, no allocations)."""
    T = qkv.shape[0]
    H, D = rec_state.shape[0], rec_state.shape[1]
    _gdn_kernel[(H,)](qkv, qkv.stride(0), z, z.stride(0), b, b.stride(0), a, a.stride(0), keep, nacc, conv_state, conv_w,
                      neg_expA, dt_bias, gnorm_w, rec_state, rec_state_out, out, out.stride(0), D ** -0.5, eps,
                      T=T, D=D, BK=32, KOFF=H * D, VOFF=2 * H * D, WRITE_STATE=int(write_state), num_warps=4)
    return out


def test(n=3):
    """Compare against the engine's eager path (fla fused_recurrent + torch conv/norm) for T in 1..4 and keep masks."""
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    from qwen35_fast.engine import rms_norm_gated
    torch.manual_seed(0); dev = "cuda"; H, D = 16, 128; C = 3 * H * D
    worst_o = worst_s = 0.0; bad = 0; tot = 0
    for T in (1, 2, 3, 4):
        for nacc in range(1, T + 1):
            for i in range(n):
                qkv = torch.randn(T, C + 40, device=dev).bfloat16()[:, :C] * 2      # non-contiguous rows like the engine
                z = torch.randn(T, H * D, device=dev).bfloat16(); b = torch.randn(T, H, device=dev).bfloat16(); a = torch.randn(T, H, device=dev).bfloat16()
                cs = torch.randn(C, 3, device=dev).bfloat16(); cw = (torch.randn(C, 1, 4, device=dev) * 0.5).bfloat16()
                neg_expA = -torch.rand(H, device=dev).mul(4).exp(); dt_bias = torch.randn(H, device=dev).bfloat16()
                gn = (1 + 0.1 * torch.randn(D, device=dev)).bfloat16(); S = torch.randn(H, D, D, device=dev) * 0.3
                # reference: eager path (outputs with all tokens; state committed with keep mask, conv shifted by nacc)
                xc_all = torch.cat([cs[None], qkv.T[None]], -1)
                y = F.silu(F.conv1d(xc_all, cw, groups=C))[0].T
                q, k, v = [t.reshape(1, T, H, D) for t in y.split([H * D] * 3, -1)]
                beta = b.sigmoid()[None]; g = (neg_expA * F.softplus(a.float() + dt_bias))[None]
                o_ref, _ = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=S[None], output_final_state=False, use_qk_l2norm_in_kernel=True)
                keep = (torch.arange(T, device=dev) < nacc).float()
                _, st = fused_recurrent_gated_delta_rule(q, k, v, g=g * keep[None, :, None], beta=beta * keep[None, :, None].to(beta.dtype),
                                                         initial_state=S[None], output_final_state=True, use_qk_l2norm_in_kernel=True)
                S_ref = st[0]; cs_ref = xc_all[0, :, nacc: nacc + 3].contiguous()
                o_ref = rms_norm_gated(o_ref.reshape(T * H, D), z.reshape(T * H, D), gn, 1e-6).reshape(T, -1)
                # kernel: verify pass (scratch state) then commit pass (in place), as the engine does for T>1
                S_k, cs_k, scr = S.clone(), cs.clone(), torch.empty_like(S)
                out = torch.empty(T, H * D, device=dev, dtype=torch.bfloat16)
                nacc_t = torch.tensor(nacc, device=dev)
                if T == 1:
                    gdn_step(qkv, z, b, a, keep, nacc_t, cs_k, cw, neg_expA, dt_bias, gn, S_k, S_k, out)
                else:
                    gdn_step(qkv, z, b, a, torch.ones(T, device=dev), nacc_t, cs_k, cw, neg_expA, dt_bias, gn, S_k, scr, out, write_state=False)
                    assert torch.equal(S_k, S) and torch.equal(cs_k, cs), "verify pass must not touch the state"
                    o2 = torch.empty_like(out)
                    gdn_step(qkv, z, b, a, keep, nacc_t, cs_k, cw, neg_expA, dt_bias, gn, S_k, S_k, o2)
                d = (out.float() - o_ref.float()).abs(); ulp = torch.pow(2.0, torch.floor(torch.log2(o_ref.float().abs().clamp_min(1e-30))) - 7)
                bad += (d > 2 * ulp + 1e-3).sum().item(); tot += d.numel()
                worst_o = max(worst_o, d.max().item()); worst_s = max(worst_s, ((S_k - S_ref).abs().max() / S_ref.abs().max()).item())
                assert torch.equal(cs_k, cs_ref), f"conv state mismatch T={T} nacc={nacc}"
    print(f"gdn_step.test: max|o diff|={worst_o:.4g}, elements beyond 2ulp: {bad}/{tot} ({100*bad/tot:.3f}%), state max rel diff={worst_s:.2e}", flush=True)
    assert bad / tot <= 1e-3 and worst_s <= 1e-5, "gdn_step tolerance exceeded"
    return worst_o, bad / tot, worst_s
