You are reviewing `qwen35_fast/engine.py`, a minimal batch-1 greedy inference engine for Qwen3.5-2B (text only),
against the Hugging Face reference implementation in `candidates/review1/hf_reference_modeling_qwen3_5.py`
(classes Qwen3_5RMSNorm, Qwen3_5RMSNormGated, Qwen3_5GatedDeltaNet, Qwen3_5Attention, Qwen3_5DecoderLayer,
Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb). Model config: hidden 2048, 24 layers (pattern 3x linear_attention
then 1x full_attention), head_dim 256, 8 q heads, 2 kv heads, partial_rotary_factor 0.25, rope_theta 1e7,
attn_output_gate=True, linear attention: 16 heads, key/value head dim 128, conv kernel 4, tied embeddings.
Text-only inputs, so mrope collapses to ordinary 1-D RoPE (all three position streams equal).

Do NOT run anything on GPU; there is no GPU here. Do NOT edit engine.py. Write your findings to
`candidates/review1/REVIEW.md` as a numbered list. For each finding: the line(s) in engine.py, what differs from the
HF reference (or what is a bug), and a concrete fix. Focus on numerical/semantic mismatches that would change model
outputs: normalization order and dtype, RoPE layout (rotate_half vs interleaved, which dims), q/k norm placement,
attention gate, GQA, GDN conv state layout and activation, gated delta rule args (g, beta, scale, l2norm),
gated RMSNorm, residual structure, argmax dtype, KV-cache indexing, speculative-decoding acceptance and state rollback
logic (_commit, _step_impl, _draft_chain). Also flag anything that would break CUDA-graph capture (host syncs,
data-dependent shapes, Python ints derived from device tensors inside _step_impl). Be concise; skip style comments.
