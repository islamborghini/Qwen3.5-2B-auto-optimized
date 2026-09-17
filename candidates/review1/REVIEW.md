# Review of `qwen35_fast/engine.py` vs the HF reference

Reviewed revision: 376 lines, `md5 13ceca4001fb87d13ddca6c23ba323ec` (working tree, ~23:16).
Note: the file changed twice while I was reviewing it (commit `958daaa` and an uncommitted
refactor added `compile_blocks`, `gdn_pre`, spec counters). Two issues below (#3, #4) were
already fixed by those edits; I document them because the task asks about exactly those areas.
Findings #1, #2 are still present in the current revision.

---

## 1. `_argmax` upcasts logits to fp32 before argmax — differs from HF greedy (line 265)

`return (h @ self.embed.T).float().argmax(-1)`.

HF `Qwen3_5ForCausalLM.forward` returns **bf16** logits (no upcast; see reference lines
1727-1730: "do not upcast them to float if we are not computing the loss"), and
`GenerationMixin.greedy_search` does `torch.argmax` directly on those bf16 logits. Argmaxing
after a `.float()` cast changes the decision whenever two token logits round to the same bf16
value (or are separated by < 1 ulp of the bf16 value), so the engine's greedy token can
diverge from HF `generate(do_sample=False)` at near-ties. This is the only place in the
main-model path where the engine's token selection can differ from HF without a forward-pass
difference, so it is worth removing for byte-exact greedy matching.

Fix: `return (h @ self.embed.T).argmax(-1)` (keep bf16). If an fp32 comparison is wanted for
debugging, do it at the call site, not in the sampling path used by `prefill`/`_step_impl`.

## 2. CUDA-graph capture warmup advances committed state — first `generate()` is wrong (lines 334-344)

```python
if g is None:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            self._step_impl()          # these RUN, they are not captured
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        self._step_impl()              # captured, not executed
    self.graphs["step"] = g
g.replay()                             # 3rd execution of this decode step
```

`_step_impl` is stateful: it advances `self.n`, rewrites `self.pending` / `self.drafts`,
appends to the full-attention KV cache, and updates `self.conv_state` / `self.rec_state`.
The two warmup calls execute (only the capture context suppresses execution), so after
capture the model has already silently consumed two decode steps; `g.replay()` then performs
a third. The tokens returned by the first `generate()` after construction therefore skip two
steps (the bench hides this by throwing away a dedicated warmup `generate`,
`bench/evaluate.py:77` / `bench/run_custom.py`); a plain one-shot `generate()` on a fresh
`Engine` is incorrect.

Fix: warm up against disposable state and restore before recording/replaying. E.g. snapshot
the mutable state (`n`, `pending`, `drafts`, `conv_state`, `rec_state`, `kc`, `vc`, and the
spec counters) before the warmup loop and `copy_` it back after capture; or build the graph
in a fresh `Engine`/with a dummy prompt, or re-run `prefill(ids)` after capture. Simply
skipping the warmup is not safe (cuBLAS/Triton/FLA workspaces must be initialized first).

Related, smaller: `stat_steps` / `stat_acc` are incremented inside `_step_impl` (line 318),
so the warmup iterations and the capture warm-up are counted in the reported acceptance
stats. Increment them outside the captured region (or subtract the warmup count).

## 3. FLA `fused_recurrent_gated_delta_rule` positional args (was lines 181-182/216-217; now fixed)

FLA 0.5.2 (`flash-linear-attention==0.5.2`, `bench/feasibility.py:15`) declares
`fused_recurrent_gated_delta_rule(q, k, v, g, gk, gv, beta, scale, initial_state, ...)`
(`fla/ops/gated_delta_rule/fused_recurrent.py`). An earlier revision of `engine.py` called it as
`fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=..., ...)`, which binds the
head-wise `beta` tensor to the 5th positional parameter `gk`, not `beta`:
`beta` then defaults to `torch.ones_like(q[...,0])` and `USE_GK` is enabled, so the kernel
loads a `[B,T,HV]` tensor as a `[B,T,HV,K]` decay and applies an unintended per-key-dim
`exp(gk)` decay (plus out-of-bounds reads). This corrupts every decode step and every
`_commit` (prefill, which uses `chunk_gated_delta_rule`, whose 5th positional *is* `beta`,
was unharmed). The current revision correctly passes `g=g, beta=beta` (lines 202, 205, 243);
keep them keyword-only — the two FLA entry points do not share a parameter order.

## 4. `_draft_chain` MTP RoPE position off by one (was lines 265/267; now fixed)

Reference (vLLM `spec_decode/autoregressive/speculator.py`):
`_prefill` stores the last accepted token's position `P`; `prepare_decode_inputs` then uses
`position = P + 1` for the first MTP decode step and `update_draft_inputs` increments by one
per step. The MTP convention is "output hidden at position `P` + token at `P+1`, evaluated at
position `P`" (the comment in the reference: *"output hidden at position P and token at P+1
are used to draft the token at P+2"*). The engine's recursion, called with
`pos0 = pending-token position = P+1`, used `p = pos0 + j`, i.e. one RoPE step too large for
every draft after the first, so it read the MTP KV cache (including stale entries written by
the previous speculative pass) at the wrong relative position and produced poorer drafts.
The current revision uses `p = pos0 + (j - 1)` (line 294), which is correct.

(Impact note: because `_step_impl` accepts a draft only when `draft == target_argmax`, the
final greedy token sequence is unaffected by draft quality — this is an acceptance/latency bug,
not an output bug. Still, the stale-cache read is a real MTP-semantics error.)

## 5. `linear_num_key_heads` is never read (lines 70, 199-200) — latent

The engine sizes and reshapes the GDN query/key with `self.gh = linear_num_value_heads`
(`conv_dim = 2*gh*gk + gh*gv`, `q/k.reshape(1,T,gh,gk)`) and skips the reference's
`repeat_interleave(num_v_heads // num_k_heads)` GVA step entirely. This is correct for the
2B checkpoint (`linear_num_key_heads == linear_num_value_heads == 16`, verified against
`Qwen/Qwen3.5-2B/config.json`), but with a config where they differ it would either crash in
the `in_all` split or silently compute wrong grouped-value attention.

Fix: read `c["linear_num_key_heads"]`, use it for the q/k head count, and apply
`repeat_interleave` when `gh > nk_heads` before calling the gated-delta-rule ops.

## 6. Misc / minor

- `_draft_chain(self, m_last, tok, pos0)` never uses `tok` (line 289). Harmless (the pending
  token is already folded into `m_last` via `_mtp_in(hid, g)`), but misleading; either drop
  the parameter or assert `tok == argmax(m_last)`.
- `mtp_hidden="pre_norm"` (lines 279, 320) feeds the pre-final-normalization residual stream
  to the MTP. The reference feeds the post-norm hidden (`Qwen3_5Model` returns
  `self.norm(hidden_states)`, and the MTP then applies `pre_fc_norm_hidden`). The default
  (`post_norm`) is right; the alternative option is not, and it silently changes drafts.
- `reset()` (line 153) does not zero `self.kc` / `self.vc`. It happens to be safe (prefill
  overwrites `[0,T)` and decode masks positions above `pos`), but combined with finding #2 it
  is the kind of stale-cache state that makes capture ordering fragile; zeroing the caches in
  `reset()` costs little and removes the reasoning dependency.

---

## Areas checked and found consistent with the reference

- `rms_norm_zc`: zero-centered weight, fp32 variance/normalize, `(1+w)` applied in fp32,
  final cast (matches `Qwen3_5RMSNorm`).
- `rms_norm_gated`: norm → `weight * x.to(dtype)` → `* silu(gate.float())` → cast
  (matches `Qwen3_5RMSNormGated` op order and dtype).
- RoPE: `rotate_half` (half-split, not interleaved) applied to the first `rot = 64` dims,
  `cos/sin` = `[f_0..f_{d/2-1}, f_0..f_{d/2-1}]`, fp32 frequency computation cast to bf16;
  text-only mrope collapses to the same layout as `recomposition_frequencies`.
- q/k RMSNorm applied per head dim before RoPE (before `transpose`), head-split
  `view(...,nq,2*hd).chunk(2)` gives q first / gate second.
- Attention output gate: `o * sigmoid(gate)` applied before `o_proj`; GQA via
  `enable_gqa=True` equals `repeat_kv`; scale `head_dim**-0.5`.
- GDN: conv applied to raw `[q,k,v]` concat with `silu`, cached state shape
  `[conv_dim, k-1]`, `beta = sigmoid(b)`, `g = -exp(A_log)*softplus(a+dt_bias)` in fp32,
  `use_qk_l2norm_in_kernel=True` (FLA default scale `K**-0.5` equals the reference's
  `query * query.shape[-1]**-0.5`), gated RMSNorm on `(o, z)`, output-flatten ordering.
- Residual structure, MLP `down(silu(gate)*up)`, final norm + tied-embedding logits.
- KV-cache indexing: prefill writes `[0,T)`, decode `index_copy_` at `pos` with a boolean
  causal mask `arange(L) <= pos`; MTP has its own cache slot (`n_attn = #full + 1`).
- Spec acceptance/rollback: `ok = (drafts == g[:-1])`, prefix `acc`, `n_acc = acc+1`,
  `_commit` re-runs the recurrence with `g*keep` (log-space decay) and `beta*keep`
  (post-sigmoid) so rejected steps neither decay nor write state, and truncates the conv
  state with `xc_all.index_select(2, n_acc + arangeC)`; `new_pending = g[acc]` is the correct
  bonus/correction token. Draft positions are now correct (#4).
- No host syncs, data-dependent shapes, or Python ints derived from device tensors inside
  `_step_impl`/`_commit`/`_draft_chain`: `n_acc`, `acc`, `pos`, `p` and the index/keep
  tensors are all device tensors. The only capture defect is the warmup ordering (#2).
