Write `tests/test_spec_rollback.py` (pytest, GPU required, ~80 lines): a correctness test for the exact speculative-decoding
state rollback in `qwen35_fast/engine.py`. Construct `Engine(model_path, spec_k=3)` (model path: use
`bench.common.model_path()`), prefill dev workload `dev-prose-128` from `bench/workloads.json`, then compare two ways of
advancing the state by the same accepted tokens: (A) the spec step `_step_impl()` with drafts deliberately set so that
exactly `a` drafts are accepted (a in 0..3: set `eng.drafts` = the true next tokens for the first `a` positions obtained
from a k=0 reference engine, then a wrong token), versus (B) a `spec_k=0` Engine that consumes the same `a+1` tokens one
by one via `_step_impl()` (set `pending` before each step). After both, assert: `rec_state` allclose (atol 1e-5 rel 1e-4),
`conv_state` exactly equal, `n` equal, KV cache rows [0:n] allclose for all attention layers, and the next-token logits
from both engines (via `eng.last_hn @ eng.embed.T` after one more T=1-equivalent step) argmax-equal. Also test that
`_commit` with n_acc = T leaves the state identical to a plain `fused_recurrent_gated_delta_rule` call over the T tokens.
Use `pytest.skip` if no CUDA. Do not modify other files. Read engine.py first (methods: prefill, _step_impl, _commit,
step_acc, pending, drafts, n, kc, vc, rec_state, conv_state, last_hn).
