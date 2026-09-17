# Review of `qwen35_fast/engine.py` (post T1+T3 merge)

Scope: working tree at `5c0455e` (merge `0779b65`), static review only. Focus: lean residual path, `_gdn`/fused
branch, `_attn`/MTP cache, `prefill`/`generate` timing, `_commit`, `forced_decode_logits`, graph capture.

## Findings

1. **[Critical] `fused_gdn` branch routes GDN decode into a kernel that is known-wrong — lines 88-91, 265-269.**
   When `fused_gdn=True` and `T == 1`, `_gdn` calls `qwen35_fast/gdn_step.py:gdn_step` *instead of* the eager
   `gdn_pre`/`fla_recurrent`/`gnorm` path (line 267-269). That kernel updates `conv_state` and `rec_state` in place,
   but its own `test()` fails: `state max rel diff=6.74e-01`, `elements beyond 2ulp: 1741/10240 (17.0%)`
   (`results/run_gdn_1.log` in the T3 worktree; commit `2a68eaa` message says "unit test FAILS (state update
   mismatch), do not enable"). End-to-end it is worse than the eager path: with `fused_gdn=True` the teacher-forced
   gate gives `max|diff|=8.16` vs HF floor `0.27` with `38` top-1 disagreements on `dev-prose-512`, vs `0.42`/`4`
   for `fused_gdn=False` (`run_gdn.json`). So enabling the flag corrupts the recurrent/conv state and every later
   token. The merge left this wired and selectable (only the README marks it "not enabled"; there is no runtime
   guard). Fix: do not ship this branch enabled. Either remove the branch, or assert `not fused_gdn` in
   `__init__` until the kernel is corrected — the fix is the K-chunked rewrite with a scratch verify state and a
   `keep` mask (`worktree-agent-aa3480d868a691f93`, commit `6b9e462`), which is the only version that passes the
   kernel test. Note the branch is also inert for `spec_k > 0` (guard is `T == 1`, and `T = spec_k + 1`), so it buys
   nothing for the shipped spec config; there is no reason to keep it as-is.

2. **[Medium] `mtp_hidden="pre_norm"` feeds the wrong hidden to the MTP — lines 113, 401, 442 (with 327-330).**
   `hp` is `_body`'s second return value, i.e. the pre-final-norm residual stream. In the lean decode path this is
   correctly the *true* residual sum (`_pre_norm` returns `(x + h, norm(x + h))`, so `_body` returns `(hn, x)` with
   `x` = residual): that part is fine. The problem is the option itself: the MTP module is `pre_fc_norm_hidden` applied
   to the base model's **post-norm** hidden (the HF reference returns `self.norm(hidden_states)` before the MTP head),
   so `mtp_hidden="pre_norm"` silently changes drafting semantics and only degrades drafts. Fix: drop the
   `pre_norm` option (always use `hn`), or document it as unsupported/experimental; the default `post_norm` is correct.

3. **[Low] `_body` decode path crashes if called without `save` — lines 303, 316-318.**
   `_body(..., prefill=False)` defaults `save=None`, but a GDN layer with `T > 1` executes
   `save.append(sv)` (line 318), raising `AttributeError`. Only `_step_impl` currently passes a list, so this is
   latent, but it is a footgun for the direct `_body` calls used by tooling/tests (e.g. `bench/optimize.py:49`).
   Fix: default to `save=[]` (`if save is None: save = []`) or `assert save is not None` when `not prefill`.

4. **[Low] `generate()` early-EOS return is inconsistent with the normal return — lines 517-519 vs 531-533.**
   The EOS-on-first-token branch returns before `stat_steps`/`stat_acc` are zeroed and omits the `"spec"` key that the
   normal return includes. Callers read `o["spec"]` (`bench/run_opt.py:37`, `bench/profile_step.py:42`), so an EOS
   first token (when `ignore_eos=False`) would `KeyError` and would also leak acceptance stats into the next call.
   Fix: reset the stats and include `"spec"` in the early return (or `goto` the common tail).

## Checked and found correct

- (1) Residual/`_pre_norm` refactor is algebraically identical to the pre-T1 eager path (verified per layer: the
  pending `h` is always the previous MLP output, `_pre_norm` restores `x + h` before the next `ln1`, the final norm
  is applied to the full residual sum, and the MTP final norm gets `x + h`). `hp` is the true residual stream.
- (2) `_gdn` signature `(w, x, conv_state, rec_state, prefill, save)` matches both call sites (`_body` prefill line
  313, `_layer_dec` line 348); the only defect is the fused branch above.
- (3) `_attn(w, x, cos, sin, kc, vc, pos, prefill)` matches all three call sites; the MTP uses `self.kc[-1]`/
  `self.vc[-1]`, which is the MTP slot allocated by `_alloc` (`n_attn = #full_attention + (1 if spec_k)`).
- (4) `prefill(ids, on_first_token)` boundary matches the README/frozen evaluator ("request start -> first token on
  host, prefill + first argmax"); the callback fires once, after the argmax and before MTP drafting, with a host sync.
- (5) `_commit` unpacks `(conv_state, rec_state, q, k, v, g, beta, xc_all)` exactly as `_gdn` appends it; the
  `g*keep`/`beta*keep` rollback is an exact identity on rejected rows and the conv gather
  `xc_all.index_select(2, n_acc + arangeC)` selects the correct window.
- (6) `forced_decode_logits` indexing is consistent (`i = n - L`, rows `0..acc` are the conditioned rows).
- (7) `_step_impl`/`_commit`/`_draft_chain` contain no host syncs or data-dependent shapes (`n_acc`, `acc`, `pos`,
  `p`, index/keep tensors are all device tensors); the one capture hazard is the fused Triton branch in finding 1.
