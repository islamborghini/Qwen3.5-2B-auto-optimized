Review `qwen35_fast/engine.py` (current working tree; it was just merged from two branches) for bugs introduced by the
merge or by the recent refactors, WITHOUT running anything. Focus on: (1) `_body` / `_layer_dec` / `_pre_norm` residual
handling (the "lean" path folds the pending residual into the next layer's norm — check the final norm and the MTP
layer receive the correct residual-summed hidden and that `hp` (pre-norm hidden) is still the true residual stream);
(2) `_gdn` signature `(w, x, conv_state, rec_state, prefill, save)` and every call site; the `fused_gdn` branch;
(3) `_attn` with explicit `kc, vc` args and the MTP call using `self.kc[-1]`; (4) `prefill(ids, on_first_token)` and
`generate()` timing boundaries; (5) `_commit` consuming the `save` tuples `(conv_state, rec_state, q, k, v, g, beta, xc_all)`;
(6) `forced_decode_logits`; (7) anything that would break CUDA-graph capture (host syncs, data-dependent shapes) in
`_step_impl`. Write findings to `candidates/review2/REVIEW.md` as a numbered list with line numbers, what is wrong,
and a concrete fix. Be concise; skip style. Do not edit engine.py.
