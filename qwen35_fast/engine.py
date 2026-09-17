"""Minimal text-only Qwen3.5-2B inference engine for batch-1 greedy decoding on one GPU.

Exact BF16 weights (loaded straight from the HF safetensors). Static caches, one CUDA graph per decode step,
optional exact speculative decoding with the checkpoint's own MTP head (greedy outputs are preserved:
a draft token is accepted only if it equals the target model's argmax at that position).
"""
import json, os, time
import torch
import torch._dynamo
import torch.nn.functional as F
from safetensors import safe_open
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

MODEL = "Qwen/Qwen3.5-2B"
REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def rms_norm_zc(x, w, eps):  # Qwen3_5RMSNorm: zero-centered weight, fp32 math
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


def rms_norm_gated(x, gate, w, eps):  # Qwen3_5RMSNormGated (same op order as HF)
    xf = x.float()
    h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    h = w * h.to(x.dtype)
    return (h * F.silu(gate.float())).to(x.dtype)


def gdn_pre(xc_all, conv_w, b, a, neg_expA, dt_bias, conv_dim):
    """conv(+silu) over [1,conv_dim,3+T] -> y [T,conv_dim]; beta=sigmoid(b); g=-exp(A)*softplus(a+dt_bias) (fp32)."""
    y = F.silu(F.conv1d(xc_all, conv_w, groups=conv_dim))[0].T
    return y, b.sigmoid()[None], (neg_expA * F.softplus(a.float() + dt_bias))[None]


def mlp_act(g, u):
    return F.silu(g) * u


def attn_post(o, gate):
    return o * torch.sigmoid(gate)


def dec_attn(q, kc, vc, pos, arangeL, rep, scale):
    """q [1,nq,T,D]; kc/vc [1,nkv,L,D]; keys at positions <= pos[t] attend. Explicit matmul path for tiny T."""
    _, nq, T, D = q.shape; nkv = kc.shape[1]
    qg = q.view(1, nkv, rep * T, D)                                            # group query heads per kv head
    s = torch.matmul(qg, kc.transpose(-1, -2)).float() * scale                # [1,nkv,rep*T,L]
    mask = (arangeL[None, :] <= pos[:, None]).repeat(rep, 1)[None, None]      # [1,1,rep*T,L] rows ordered (head, t)
    s = s.masked_fill(~mask, float("-inf")).softmax(-1).to(q.dtype)
    return torch.matmul(s, vc).view(1, nq, T, D)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):  # q,k: [B,H,T,D]; cos/sin: [T, rot]
    r = cos.shape[-1]
    cos, sin = cos[None, None], sin[None, None]
    qr, qp = q[..., :r], q[..., r:]
    kr, kp = k[..., :r], k[..., r:]
    return torch.cat([qr * cos + rotate_half(qr) * sin, qp], -1), torch.cat([kr * cos + rotate_half(kr) * sin, kp], -1)


class Engine:
    def __init__(self, model_path, device="cuda", max_len=8704, spec_k=0, mtp_hidden="post_norm", fuse_proj=True,
                 compile_blocks=False, use_gemv=False):
        self.dev = torch.device(device)
        self.compile_blocks = compile_blocks
        self.use_gemv = use_gemv
        self.mm = lambda x, W: x @ W.T
        self._mm_choice = {}
        fns = dict(norm=rms_norm_zc, gnorm=rms_norm_gated, gdn_pre=gdn_pre, mlp_act=mlp_act, attn_post=attn_post, rope=apply_rope, dec_attn=dec_attn)
        if compile_blocks:
            torch._dynamo.config.cache_size_limit = 64
            fns = {k: torch.compile(v, dynamic=False) for k, v in fns.items()}
        self.f = fns
        self.cfg = c = json.load(open(os.path.join(model_path, "config.json")))["text_config"]
        self.eps = c["rms_norm_eps"]; self.H = c["hidden_size"]; self.nl = c["num_hidden_layers"]
        self.layer_types = c["layer_types"]; self.hd = c["head_dim"]; self.nq = c["num_attention_heads"]; self.nkv = c["num_key_value_heads"]
        self.gh = c["linear_num_value_heads"]; self.gk = c["linear_key_head_dim"]; self.gv = c["linear_value_head_dim"]
        self.conv_k = c["linear_conv_kernel_dim"]; self.conv_dim = 2 * self.gh * self.gk + self.gh * self.gv
        self.rot = int(self.hd * c["rope_parameters"]["partial_rotary_factor"])
        self.max_len = max_len; self.spec_k = spec_k; self.mtp_hidden = mtp_hidden; self.fuse_proj = fuse_proj
        self._load(model_path)
        self._alloc()
        inv = 1.0 / (c["rope_parameters"]["rope_theta"] ** (torch.arange(0, self.rot, 2, dtype=torch.float32) / self.rot))
        fr = torch.arange(max_len, dtype=torch.float32)[:, None] * inv[None]
        fr = torch.cat([fr, fr], -1)
        self.cos = fr.cos().to(torch.bfloat16).to(self.dev); self.sin = fr.sin().to(torch.bfloat16).to(self.dev)
        self.graphs = {}
        if use_gemv:
            self._select_gemv()

    def _select_gemv(self):
        """Per weight shape, keep whichever is faster for M=1..spec_k+1: Triton gemv or cuBLAS (both exact-bf16 GEMV)."""
        from qwen35_fast.gemv import gemv
        Ws = {}
        for d in self.layers + ([self.mtp] if self.mtp else []):
            for v in d.values():
                if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[1] in (self.H, 2 * self.H, 6144):
                    Ws.setdefault(tuple(v.shape), v)
        Ws.setdefault(tuple(self.embed.shape), self.embed)
        def timeit(fn, n=30):
            for _ in range(5): fn()
            s, t = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
            for _ in range(n): fn()
            t.record(); torch.cuda.synchronize(); return s.elapsed_time(t) / n
        for shape, W in Ws.items():
            for M in range(1, self.spec_k + 2):
                x = torch.randn(M, shape[1], device=self.dev, dtype=torch.bfloat16)
                tt, tg = timeit(lambda: x @ W.T), timeit(lambda: gemv(x, W))
                self._mm_choice[(M, shape)] = "gemv" if tg < tt else "torch"
        def mm(x, W):
            return gemv(x, W) if self._mm_choice.get((x.shape[0], tuple(W.shape))) == "gemv" else x @ W.T
        self.mm = mm

    # ---------------------------------------------------------------- weights
    def _load(self, path):
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        files = {}
        def get(name):
            f = idx[name]
            if f not in files:
                files[f] = safe_open(os.path.join(path, f), "pt", device=str(self.dev))
            return files[f].get_tensor(name)
        P = "model.language_model."
        self.embed = get(P + "embed_tokens.weight")  # tied lm_head
        self.final_norm = get(P + "norm.weight")
        self.layers = []
        for i in range(self.nl):
            L = f"{P}layers.{i}."
            d = {"ln1": get(L + "input_layernorm.weight"), "ln2": get(L + "post_attention_layernorm.weight"),
                 "gate": get(L + "mlp.gate_proj.weight"), "up": get(L + "mlp.up_proj.weight"), "down": get(L + "mlp.down_proj.weight")}
            if self.fuse_proj:
                d["gate_up"] = torch.cat([d.pop("gate"), d.pop("up")], 0)
            if self.layer_types[i] == "linear_attention":
                A = L + "linear_attn."
                d.update(qkv=get(A + "in_proj_qkv.weight"), z=get(A + "in_proj_z.weight"), b=get(A + "in_proj_b.weight"),
                         a=get(A + "in_proj_a.weight"), conv_w=get(A + "conv1d.weight"), A_log=get(A + "A_log"),
                         dt_bias=get(A + "dt_bias"), gnorm=get(A + "norm.weight"), out=get(A + "out_proj.weight"))
                d["neg_expA"] = -d["A_log"].float().exp()
                if self.fuse_proj:  # one GEMM for qkv|z|b|a
                    d["in_all"] = torch.cat([d["qkv"], d["z"], d["b"], d["a"]], 0)
            else:
                A = L + "self_attn."
                d.update(q=get(A + "q_proj.weight"), k=get(A + "k_proj.weight"), v=get(A + "v_proj.weight"), o=get(A + "o_proj.weight"),
                         qn=get(A + "q_norm.weight"), kn=get(A + "k_norm.weight"))
                if self.fuse_proj:
                    d["qkv_w"] = torch.cat([d["q"], d["k"], d["v"]], 0)
            self.layers.append(d)
        self.mtp = None
        if self.spec_k > 0:
            M = "mtp."
            m = {"fc": get(M + "fc.weight"), "norm": get(M + "norm.weight"), "pre_e": get(M + "pre_fc_norm_embedding.weight"),
                 "pre_h": get(M + "pre_fc_norm_hidden.weight"), "ln1": get(M + "layers.0.input_layernorm.weight"),
                 "ln2": get(M + "layers.0.post_attention_layernorm.weight"),
                 "gate": get(M + "layers.0.mlp.gate_proj.weight"), "up": get(M + "layers.0.mlp.up_proj.weight"),
                 "down": get(M + "layers.0.mlp.down_proj.weight"), "q": get(M + "layers.0.self_attn.q_proj.weight"),
                 "k": get(M + "layers.0.self_attn.k_proj.weight"), "v": get(M + "layers.0.self_attn.v_proj.weight"),
                 "o": get(M + "layers.0.self_attn.o_proj.weight"), "qn": get(M + "layers.0.self_attn.q_norm.weight"),
                 "kn": get(M + "layers.0.self_attn.k_norm.weight")}
            if self.fuse_proj:
                m["gate_up"] = torch.cat([m.pop("gate"), m.pop("up")], 0); m["qkv_w"] = torch.cat([m["q"], m["k"], m["v"]], 0)
            self.mtp = m
        torch.cuda.synchronize()

    # ---------------------------------------------------------------- state
    def _alloc(self):
        dev, L = self.dev, self.max_len
        n_attn = sum(t == "full_attention" for t in self.layer_types) + (1 if self.spec_k > 0 else 0)
        self.kc = torch.zeros(n_attn, 1, self.nkv, L, self.hd, dtype=torch.bfloat16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        n_lin = sum(t == "linear_attention" for t in self.layer_types)
        self.conv_state = torch.zeros(n_lin, 1, self.conv_dim, self.conv_k - 1, dtype=torch.bfloat16, device=dev)
        self.rec_state = torch.zeros(n_lin, 1, self.gh, self.gk, self.gv, dtype=torch.float32, device=dev)
        self.n = torch.zeros((), dtype=torch.long, device=dev)          # committed length (main model)
        self.pending = torch.zeros((), dtype=torch.long, device=dev)    # next token, not yet processed by main model
        T = self.spec_k + 1
        self.drafts = torch.zeros(self.spec_k, dtype=torch.long, device=dev)
        self.step_tokens = torch.zeros(T, dtype=torch.long, device=dev)  # greedy tokens produced by last step
        self.step_acc = torch.zeros((), dtype=torch.long, device=dev)    # number of accepted drafts in last step
        self.arangeT = torch.arange(T, device=dev)
        self.stat_steps = torch.zeros((), dtype=torch.long, device=dev)
        self.stat_acc = torch.zeros((), dtype=torch.long, device=dev)
        self.arangeL = torch.arange(L, device=dev)
        self.arangeC = torch.arange(self.conv_k - 1, device=dev)

    def reset(self):
        self.conv_state.zero_(); self.rec_state.zero_(); self.n.zero_(); self.pending.zero_(); self.drafts.zero_()
        self.kc.zero_(); self.vc.zero_()

    # ---------------------------------------------------------------- blocks
    def _attn(self, w, x, cos, sin, kv_idx, pos, kv_len, prefill):
        """x: [T,H]. pos: [T] positions (device). kv_len: device scalar (valid cache length before this call)."""
        T = x.shape[0]
        mm = (lambda a, W: a @ W.T) if prefill else self.mm
        if self.fuse_proj:
            qkv = mm(x, w["qkv_w"])
            q, k, v = qkv.split([self.nq * self.hd * 2, self.nkv * self.hd, self.nkv * self.hd], -1)
        else:
            q, k, v = mm(x, w["q"]), mm(x, w["k"]), mm(x, w["v"])
        q, gate = q.view(T, self.nq, 2 * self.hd).chunk(2, -1)
        gate = gate.reshape(T, -1)
        norm, rope = (rms_norm_zc, apply_rope) if prefill else (self.f["norm"], self.f["rope"])
        q = norm(q, w["qn"], self.eps).transpose(0, 1)[None]          # [1,nq,T,D]
        k = norm(k.view(T, self.nkv, self.hd), w["kn"], self.eps).transpose(0, 1)[None]
        v = v.view(T, self.nkv, self.hd).transpose(0, 1)[None]
        q, k = rope(q, k, cos, sin)
        kc, vc = self.kc[kv_idx], self.vc[kv_idx]
        if prefill:
            kc[:, :, :T] = k; vc[:, :, :T] = v
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            kc.index_copy_(2, pos, k); vc.index_copy_(2, pos, v)
            o = self.f["dec_attn"](q, kc, vc, pos, self.arangeL, self.nq // self.nkv, self.hd ** -0.5)
        o = o.transpose(1, 2).reshape(T, -1)
        o = attn_post(o, gate) if prefill else self.f["attn_post"](o, gate)
        return mm(o, w["o"])

    def _gdn(self, w, x, li, prefill, save):
        T = x.shape[0]
        mm = (lambda a_, W: a_ @ W.T) if prefill else self.mm
        if self.fuse_proj:
            allp = mm(x, w["in_all"])
            qkv, z, b, a = allp.split([self.conv_dim, self.gh * self.gv, self.gh, self.gh], -1)
        else:
            qkv, z, b, a = mm(x, w["qkv"]), mm(x, w["z"]), mm(x, w["b"]), mm(x, w["a"])
        xc = qkv.T[None]                                                       # [1,conv_dim,T]
        cw = w["conv_w"]                                                       # [conv_dim,1,k]
        if prefill:
            xc_all = torch.cat([torch.zeros_like(self.conv_state[li]), xc], -1)
        else:
            xc_all = torch.cat([self.conv_state[li], xc], -1)
        pre = gdn_pre if prefill else self.f["gdn_pre"]
        y, beta, g = pre(xc_all, cw, b, a, w["neg_expA"], w["dt_bias"], self.conv_dim)   # y [T,conv_dim]; beta/g [1,T,gh]
        q, k, v = y.split([self.gh * self.gk, self.gh * self.gk, self.gh * self.gv], -1)
        q = q.reshape(1, T, self.gh, self.gk); k = k.reshape(1, T, self.gh, self.gk); v = v.reshape(1, T, self.gh, self.gv)
        if prefill:
            o, st = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
            self.rec_state[li].copy_(st); self.conv_state[li].copy_(xc_all[:, :, -(self.conv_k - 1):])
        else:
            o, st = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=self.rec_state[li],
                                                     output_final_state=True, use_qk_l2norm_in_kernel=True)
            if T == 1:
                self.rec_state[li].copy_(st); self.conv_state[li].copy_(xc_all[:, :, 1:])
            else:
                save.append((li, q, k, v, g, beta, xc_all))
        gn = rms_norm_gated if prefill else self.f["gnorm"]
        o = gn(o.reshape(T * self.gh, self.gv), z.reshape(T * self.gh, self.gv), w["gnorm"], self.eps)
        return mm(o.reshape(T, -1), w["out"])

    def _mlp(self, w, x, prefill):
        mm = (lambda a, W: a @ W.T) if prefill else self.mm
        if self.fuse_proj:
            g, u = mm(x, w["gate_up"]).chunk(2, -1)
        else:
            g, u = mm(x, w["gate"]), mm(x, w["up"])
        act = mlp_act if prefill else self.f["mlp_act"]
        return mm(act(g, u), w["down"])

    def _body(self, tokens, pos, prefill, save=None):
        """Run the main model over tokens [T] at positions pos [T]. Returns post-final-norm hidden [T,H] and pre-norm hidden."""
        x = self.embed[tokens]
        cos, sin = self.cos[pos], self.sin[pos]
        li = ai = 0
        norm = rms_norm_zc if prefill else self.f["norm"]
        for i, w in enumerate(self.layers):
            h = norm(x, w["ln1"], self.eps)
            if self.layer_types[i] == "linear_attention":
                h = self._gdn(w, h, li, prefill, save); li += 1
            else:
                h = self._attn(w, h, cos, sin, ai, pos, self.n, prefill); ai += 1
            x = x + h
            x = x + self._mlp(w, norm(x, w["ln2"], self.eps), prefill)
        return norm(x, self.final_norm, self.eps), x

    def _commit(self, save, n_acc):
        """Commit GDN/conv states for the first n_acc tokens of the last multi-token step (device-side, exact)."""
        keep = (self.arangeT < n_acc).to(torch.float32)[None, :, None]        # [1,T,1]
        for li, q, k, v, g, beta, xc_all in save:
            _, st = fused_recurrent_gated_delta_rule(q, k, v, g=g * keep, beta=beta * keep.to(beta.dtype),
                                                     initial_state=self.rec_state[li], output_final_state=True, use_qk_l2norm_in_kernel=True)
            self.rec_state[li].copy_(st)
            self.conv_state[li].copy_(xc_all.index_select(2, n_acc + self.arangeC))

    def _mtp_layer(self, x, cos, sin, pos, prefill=False):
        m = self.mtp
        norm = rms_norm_zc if prefill else self.f["norm"]
        h = norm(x, m["ln1"], self.eps)
        h = self._attn(m, h, cos, sin, self.kc.shape[0] - 1, pos, self.n, prefill)
        x = x + h
        x = x + self._mlp(m, norm(x, m["ln2"], self.eps), prefill)
        return norm(x, m["norm"], self.eps)

    def _mtp_in(self, hidden, tokens, prefill=False):
        m = self.mtp
        norm = rms_norm_zc if prefill else self.f["norm"]
        e = norm(self.embed[tokens], m["pre_e"], self.eps)
        h = norm(hidden, m["pre_h"], self.eps)
        return (torch.cat([e, h], -1) @ m["fc"].T) if prefill else self.mm(torch.cat([e, h], -1), m["fc"])

    def _argmax(self, h):
        return self.mm(h, self.embed).argmax(-1)

    # ---------------------------------------------------------------- steps
    @torch.no_grad()
    def prefill(self, ids):
        """Process prompt ids (list[int]); returns first generated token (device scalar in self.pending)."""
        self.reset()
        T = len(ids)
        tokens = torch.tensor(ids, device=self.dev)
        pos = torch.arange(T, device=self.dev)
        hn, hp = self._body(tokens, pos, prefill=True)
        g = self._argmax(hn[-1:])
        self.n.fill_(T); self.pending.copy_(g[0])
        if self.spec_k > 0:
            hid = hn if self.mtp_hidden == "post_norm" else hp
            # MTP over the whole prompt: inputs (h_i, t_{i+1}) for i < T-1, and (h_{T-1}, g) for the last row
            nxt = torch.cat([tokens[1:], g])
            x = self._mtp_in(hid, nxt, prefill=True)
            self.n.fill_(0)  # not used in prefill attention path (is_causal)
            mo = self._mtp_layer(x, self.cos[pos], self.sin[pos], pos, prefill=True)
            self.n.fill_(T)
            self._draft_chain(mo[-1:], g, T)
        return g

    def _draft_chain(self, m_last, tok, pos0):
        """Given MTP output hidden for the last committed position and the pending token, produce spec_k drafts."""
        d = self._argmax(m_last)
        self.drafts[0].copy_(d[0])
        for j in range(1, self.spec_k):
            p = (pos0 + (j - 1)).reshape(1) if isinstance(pos0, torch.Tensor) else torch.tensor([pos0 + j - 1], device=self.dev)
            x = self._mtp_in(m_last, d)
            m_last = self._mtp_layer(x, self.cos[p], self.sin[p], p)
            d = self._argmax(m_last)
            self.drafts[j].copy_(d[0])

    def _step_impl(self):
        """One decode step (T = spec_k + 1 tokens). Updates state; writes self.step_tokens / self.step_acc."""
        tokens = torch.cat([self.pending.reshape(1), self.drafts]) if self.spec_k else self.pending.reshape(1)
        pos = self.n + self.arangeT
        save = []
        hn, hp = self._body(tokens, pos, prefill=False, save=save)
        g = self._argmax(hn)                                                   # [T]
        if self.spec_k == 0:
            self.step_tokens.copy_(g); self.step_acc.zero_()
            self.n.add_(1); self.pending.copy_(g[0])
            return
        # accepted drafts: longest prefix where draft_j == g_{j-1}
        ok = (self.drafts == g[:-1]).to(torch.long)
        acc = torch.cumprod(ok, 0).sum()                                       # 0..k
        n_acc = acc + 1                                                        # tokens committed by main model
        self._commit(save, n_acc)
        self.step_tokens.copy_(g); self.step_acc.copy_(acc)
        self.stat_steps.add_(1); self.stat_acc.add_(acc)
        # MTP pass over the T rows (rows beyond acceptance are harmless; their cache slots get overwritten later)
        hid = hn if self.mtp_hidden == "post_norm" else hp
        x = self._mtp_in(hid, g)
        mo = self._mtp_layer(x, self.cos[pos], self.sin[pos], pos)
        m_last = mo.index_select(0, acc.reshape(1))
        new_pending = g.index_select(0, acc.reshape(1))
        self.n.add_(n_acc)
        self._draft_chain(m_last, new_pending, self.n)
        self.pending.copy_(new_pending[0])

    @torch.no_grad()
    def ensure_graph(self):
        """Capture the decode-step graph once, on disposable dummy state (warmup runs mutate state)."""
        if "step" in self.graphs:
            return
        self.prefill([1, 2, 3, 4])
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step_impl()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step_impl()
        self.graphs["step"] = g
        self.stat_steps.zero_(); self.stat_acc.zero_()
        torch.cuda.synchronize()

    @torch.no_grad()
    def step(self, use_graph=True):
        if not use_graph:
            return self._step_impl()
        self.ensure_graph()
        self.graphs["step"].replay()

    @torch.no_grad()
    def generate(self, ids, n_out, eos_ids=(), ignore_eos=False, use_graph=True):
        """Returns dict(tokens, ttft_s, decode_s, prefill_s, total_s). Timing per README measurement notes."""
        if use_graph:
            self.ensure_graph()   # one-time startup cost, excluded from timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first = self.prefill(ids)
        ev_first = torch.cuda.Event(enable_timing=True); ev_first.record()
        first_tok = first.item()  # sync: first token complete
        t_first = time.perf_counter()
        out = [first_tok]
        T = self.spec_k + 1
        host = torch.empty(T + 1, dtype=torch.long, pin_memory=True)
        ev = torch.cuda.Event(enable_timing=True)
        eos = set(eos_ids)
        if not ignore_eos and first_tok in eos:
            torch.cuda.synchronize(); t_end = time.perf_counter()
            return {"tokens": out, "ttft_s": t_first - t0, "decode_s": 0.0, "total_s": t_end - t0}
        while len(out) < n_out:
            self.step(use_graph)
            host[:T].copy_(self.step_tokens, non_blocking=True); host[T].copy_(self.step_acc, non_blocking=True)
            ev.record(); ev.synchronize()
            acc = int(host[T]); new = host[: acc + 1].tolist()
            for t in new:
                out.append(t)
                if len(out) >= n_out or (not ignore_eos and t in eos):
                    break
            if not ignore_eos and any(t in eos for t in new):
                break
        t_end = time.perf_counter()
        st = {"steps": int(self.stat_steps), "accepted": int(self.stat_acc)}; self.stat_steps.zero_(); self.stat_acc.zero_()
        return {"tokens": out[:n_out], "ttft_s": t_first - t0, "decode_s": t_end - t_first, "total_s": t_end - t0, "spec": st}
