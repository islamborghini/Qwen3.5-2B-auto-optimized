"""Correctness test for the exact speculative-decoding state rollback in qwen35_fast/engine.py.

For a in 0..3 we script a spec_k=3 engine's drafts so exactly `a` are accepted, then require its committed
state to match a spec_k=0 engine that consumed the same a+1 tokens one at a time. A second test checks that
`_commit(save, T)` reproduces a plain fused_recurrent_gated_delta_rule pass over all T tokens.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from bench.common import WORKLOADS, model_path
from qwen35_fast.engine import Engine
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

IDS = next(w["ids"] for w in WORKLOADS["workloads"] if w["id"] == "dev-prose-128")


@pytest.fixture(scope="module")
def engines():
    mp = model_path()
    return Engine(mp, spec_k=3), Engine(mp, spec_k=0)


@torch.no_grad()
def test_spec_rollback_matches_sequential(engines):
    e3, e0 = engines
    V, K = e3.embed.shape[0], e3.spec_k
    for a in range(K + 1):
        e3.prefill(IDS)
        e0.prefill(IDS)
        base_n = int(e3.n)
        true_next = []                       # g_0 .. g_a, the k=0 reference's greedy tokens
        for _ in range(a + 1):
            e0._step_impl()
            true_next.append(int(e0.step_tokens[0]))
        wrong = (true_next[a] + 1) % V       # index a is deliberately not the main model's argmax
        e3.drafts.copy_(torch.tensor(true_next[:a] + [wrong] * (K - a), dtype=torch.long, device=e3.dev))
        e3._step_impl()
        assert int(e3.step_acc) == a
        assert int(e3.n) == int(e0.n) == base_n + a + 1
        assert int(e3.pending) == int(e0.pending)
        assert torch.allclose(e3.rec_state, e0.rec_state, atol=1e-5, rel=1e-4)
        assert torch.equal(e3.conv_state, e0.conv_state)
        n = int(e3.n)
        for i in range(e0.kc.shape[0]):      # main-model full-attention layers (the MTP layer is e3-only)
            assert torch.allclose(e3.kc[i, :, :, :n], e0.kc[i, :, :, :n], atol=1e-5, rel=1e-4)
            assert torch.allclose(e3.vc[i, :, :, :n], e0.vc[i, :, :, :n], atol=1e-5, rel=1e-4)
        # One more T=1-equivalent step: forcing step_acc == 0 makes the spec engine commit only `pending`.
        e0._step_impl()
        e3.drafts.fill_((int(e0.step_tokens[0]) + 1) % V)
        e3._step_impl()
        assert int(e3.step_acc) == 0
        assert int((e3.last_hn[0] @ e3.embed.T).argmax()) == int((e0.last_hn[0] @ e0.embed.T).argmax())


@torch.no_grad()
def test_commit_full_recompute(engines):
    e3, _ = engines
    e3.prefill(IDS)
    T = e3.spec_k + 1
    tokens = torch.cat([e3.pending.reshape(1), e3.drafts])
    init_rec = e3.rec_state.clone()
    save = []
    e3._body(tokens, e3.n + e3.arangeT, prefill=False, save=save)
    assert len(save) > 0
    refs = []
    for i, (_, _, q, k, v, g, beta, xc_all) in enumerate(save):
        assert torch.equal(e3.rec_state[i], init_rec[i])          # the forward must not touch state
        _, st = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=init_rec[i],
                                                 output_final_state=True, use_qk_l2norm_in_kernel=True)
        refs.append((st, xc_all))
    e3._commit(save, T)
    for i, (st, xc_all) in enumerate(refs):
        assert torch.equal(e3.rec_state[i], st)
        assert torch.equal(e3.conv_state[i], xc_all.index_select(2, T + e3.arangeC))
