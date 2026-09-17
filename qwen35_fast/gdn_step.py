"""One Triton kernel for the whole Gated-DeltaNet single-token decode update (batch 1, T=1), one program per head:
conv1d(4 taps)+SiLU on the head's q/k/v channels, conv-state shift, fp32 l2norm(q,k), q*scale, gated delta rule state
update (in place), o = S^T q, gated RMSNorm(o, z). Op order/dtypes mirror the eager path (cuDNN conv -> bf16 -> silu ->
bf16; fla fused_recurrent fp32 math; Qwen3_5RMSNormGated rounding). Usage: gdn_step(qkv, z, b, a, conv_state, conv_w,
neg_expA, dt_bias, gnorm_w, rec_state, out). test() checks against the eager ops on a GPU."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _conv_silu(x_ptr, st_ptr, w_ptr, c):
    x = tl.load(x_ptr + c)
    s0 = tl.load(st_ptr + c * 3); s1 = tl.load(st_ptr + c * 3 + 1); s2 = tl.load(st_ptr + c * 3 + 2)
    w0 = tl.load(w_ptr + c * 4); w1 = tl.load(w_ptr + c * 4 + 1); w2 = tl.load(w_ptr + c * 4 + 2); w3 = tl.load(w_ptr + c * 4 + 3)
    acc = s0.to(tl.float32) * w0.to(tl.float32) + s1.to(tl.float32) * w1.to(tl.float32) \
        + s2.to(tl.float32) * w2.to(tl.float32) + x.to(tl.float32) * w3.to(tl.float32)
    yb = acc.to(tl.bfloat16).to(tl.float32)                     # conv output rounded to bf16
    y = (yb * tl.sigmoid(yb)).to(tl.bfloat16).to(tl.float32)     # silu on bf16 -> bf16
    # shift conv state in place (each channel owned by exactly one program)
    tl.store(st_ptr + c * 3, s1); tl.store(st_ptr + c * 3 + 1, s2); tl.store(st_ptr + c * 3 + 2, x)
    return y


@triton.jit
def _gdn_step_kernel(qkv_ptr, z_ptr, b_ptr, a_ptr, st_ptr, w_ptr, nega_ptr, dtb_ptr, gn_ptr, S_ptr, o_ptr,
                     scale, eps, D: tl.constexpr, KOFF: tl.constexpr, VOFF: tl.constexpr, DBG: tl.constexpr):
    h = tl.program_id(0)
    offs = tl.arange(0, D)
    q = _conv_silu(qkv_ptr, st_ptr, w_ptr, h * D + offs)
    k = _conv_silu(qkv_ptr, st_ptr, w_ptr, KOFF + h * D + offs)
    v = _conv_silu(qkv_ptr, st_ptr, w_ptr, VOFF + h * D + offs)
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    q = q * scale
    if DBG == 1:
        tl.store(o_ptr + 3 * h * D + offs, q); tl.store(o_ptr + (3 * h + 1) * D + offs, k); tl.store(o_ptr + (3 * h + 2) * D + offs, v)
        return
    beta = tl.sigmoid(tl.load(b_ptr + h).to(tl.float32)).to(tl.bfloat16).to(tl.float32)   # HF: b.sigmoid() in bf16
    ga = tl.load(a_ptr + h).to(tl.float32) + tl.load(dtb_ptr + h).to(tl.float32)
    sp = tl.where(ga > 20.0, ga, libdevice.log1p(tl.exp(ga)))                            # torch softplus
    g = tl.load(nega_ptr + h) * sp
    Sp = S_ptr + h * D * D + offs[:, None] * D + offs[None, :]                            # S[k, v]
    S = tl.load(Sp) * tl.exp(g)
    kv = tl.sum(S * k[:, None], 0)
    if DBG == 2:   # dump decay/beta/kv (head-major, D floats each) for diagnosis
        tl.store(o_ptr + 3 * h * D + offs, tl.zeros([D], tl.float32) + tl.exp(g))
        tl.store(o_ptr + (3 * h + 1) * D + offs, tl.zeros([D], tl.float32) + beta)
        tl.store(o_ptr + (3 * h + 2) * D + offs, kv)
        return
    vn = beta * (v - kv)
    S += k[:, None] * vn[None, :]
    o = tl.sum(S * q[:, None], 0)
    tl.store(Sp, S)
    # gated RMSNorm (Qwen3_5RMSNormGated): input is the bf16 o, fp32 norm, bf16 * weight, * silu(z fp32), -> bf16
    ob = o.to(tl.bfloat16).to(tl.float32)
    hn = ob * tl.rsqrt(tl.sum(ob * ob) / D + eps)
    hb = hn.to(tl.bfloat16).to(tl.float32)
    hw = (tl.load(gn_ptr + offs).to(tl.float32) * hb).to(tl.bfloat16).to(tl.float32)
    z = tl.load(z_ptr + h * D + offs).to(tl.float32)
    out = hw * (z * tl.sigmoid(z))
    tl.store(o_ptr + h * D + offs, out.to(tl.bfloat16))


def gdn_step(qkv, z, b, a, conv_state, conv_w, neg_expA, dt_bias, gnorm_w, rec_state, out, eps=1e-6, dbg=False):
    """qkv [6144] bf16 (pre-conv row), z [2048] bf16, b/a [16] bf16, conv_state [6144,3] bf16 (updated in place),
    conv_w [6144,1,4] bf16, neg_expA [16] fp32, dt_bias [16], gnorm_w [128] bf16, rec_state [16,128,128] fp32 (in place),
    out [2048] bf16. dbg=True: out is fp32 [3*H*D] and receives the post-l2norm q,k,v per head (test only)."""
    H, D = rec_state.shape[0], rec_state.shape[1]
    _gdn_step_kernel[(H,)](qkv, z, b, a, conv_state, conv_w, neg_expA, dt_bias, gnorm_w, rec_state, out,
                           D ** -0.5, eps, D=D, KOFF=H * D, VOFF=2 * H * D, DBG=int(dbg), num_warps=8)
    return out


def _torch_ref(q, k, v, g, beta, S):
    """Plain-torch gated delta rule for T=1 (q,k,v [H,D] fp32 post conv/silu, S [H,D,D]). Returns o [H,D], S_new."""
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    S = S * g.exp()[:, None, None]
    kv = (S * k[:, :, None]).sum(1)
    vn = beta[:, None] * (v - kv)
    S = S + k[:, :, None] * vn[:, None, :]
    return (S * q[:, :, None]).sum(1), S


def test(n=5):
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    from qwen35_fast.engine import rms_norm_gated
    torch.manual_seed(0); dev = "cuda"; H, D = 16, 128; C = 3 * H * D
    worst_o = worst_s = 0.0; bad = 0; tot = 0
    for i in range(n):
        qkv = torch.randn(C, device=dev).bfloat16() * 2
        z = torch.randn(H * D, device=dev).bfloat16(); b = torch.randn(H, device=dev).bfloat16(); a = torch.randn(H, device=dev).bfloat16()
        cs = torch.randn(C, 3, device=dev).bfloat16(); cw = (torch.randn(C, 1, 4, device=dev) * 0.5).bfloat16()
        neg_expA = -torch.rand(H, device=dev).mul(4).exp(); dt_bias = torch.randn(H, device=dev).bfloat16()
        gn = (1 + 0.1 * torch.randn(D, device=dev)).bfloat16(); S = torch.randn(H, D, D, device=dev) * 0.3
        # reference = the engine's eager T=1 path
        cs_ref, S_ref = cs.clone(), S.clone()
        xc_all = torch.cat([cs_ref[None], qkv[None, :, None]], -1)
        y = F.silu(F.conv1d(xc_all, cw, groups=C))[0].T
        q, k, v = [t.reshape(1, 1, H, D) for t in y.split([H * D] * 3, -1)]
        beta = b.sigmoid()[None, None]; g = (neg_expA * F.softplus(a.float() + dt_bias))[None, None]
        o, st = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=S_ref[None], output_final_state=True, use_qk_l2norm_in_kernel=True)
        S_ref = st[0]; cs_ref = xc_all[0, :, 1:]
        o_ref = rms_norm_gated(o.reshape(H, D), z.reshape(H, D), gn, 1e-6).reshape(-1)
        if i == 0:  # diagnostics: where does the kernel diverge? (conv/silu+l2norm inputs, fla-vs-torch recurrence)
            qf, kf, vf = [t.reshape(H, D).float() for t in y.split([H * D] * 3, -1)]
            o_t, S_t = _torch_ref(qf, kf, vf, g[0, 0], beta[0, 0].float(), S.clone())
            print(f"  fla vs torch-ref: state rel diff {((S_ref - S_t).abs().max() / S_t.abs().max()).item():.2e}, o max diff {(o.reshape(H, D).float() - o_t).abs().max().item():.3g}", flush=True)
            dbg = torch.empty(3 * H * D, device=dev, dtype=torch.float32); cs_d, S_d = cs.clone(), S.clone()
            gdn_step(qkv, z, b, a, cs_d, cw, neg_expA, dt_bias, gn, S_d, dbg, dbg=True)
            dbg = dbg.reshape(H, 3, D)
            qn = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * D ** -0.5; kn = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            print(f"  kernel q/k/v vs torch: {(dbg[:, 0] - qn).abs().max().item():.3g} {(dbg[:, 1] - kn).abs().max().item():.3g} {(dbg[:, 2] - vf).abs().max().item():.3g}", flush=True)
        out = torch.empty(H * D, device=dev, dtype=torch.bfloat16)
        gdn_step(qkv, z, b, a, cs, cw, neg_expA, dt_bias, gn, S, out)
        if i == 0:
            print(f"  kernel vs torch-ref: state rel diff {((S - S_t).abs().max() / S_t.abs().max()).item():.2e}", flush=True)
        d = (out.float() - o_ref.float()).abs(); ulp = torch.pow(2.0, torch.floor(torch.log2(o_ref.float().abs().clamp_min(1e-30))) - 7)
        bad += (d > 2 * ulp + 1e-3).sum().item(); tot += d.numel()
        worst_o = max(worst_o, d.max().item()); worst_s = max(worst_s, ((S - S_ref).abs().max() / S_ref.abs().max()).item())
        assert torch.equal(cs, cs_ref), "conv state mismatch"
    print(f"gdn_step.test: max|o diff|={worst_o:.4g}, elements beyond 2ulp: {bad}/{tot} ({100*bad/tot:.3f}%), state max rel diff={worst_s:.2e}", flush=True)
    assert bad / tot <= 1e-3 and worst_s <= 1e-4, "gdn_step tolerance exceeded"
    return worst_o, bad / tot, worst_s
