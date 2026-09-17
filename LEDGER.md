# Spending & candidate ledger

Coordinator: Claude Fable 5.1 (this session). Workers: DeepSeek V4.1 Flash via OpenCode (`opencode-go` subscription, flat-rate; no per-call charge observed — `opencode stats` reports cumulative cost of prior 131 days as $6.87, so calls are treated as covered by the existing subscription allowance).

Budgets: Fable $100 (reserve $20) · Modal $10 (reserve $2). Modal H100 assumed ≈ $3.95/h (estimate; check dashboard).

| ts (UTC) | who | action | est. cost | notes |
|---|---|---|---|---|
| 2026-09-16 | Fable | env inspection, model/config research | ~$3 Fable | no GPU |
| 2026-09-16 | DeepSeek | probe call ("OK") | $0 (subscription) | access confirmed |
| 2026-09-16 | Modal H100 | feasibility run #1 (crashed at prompt build; model downloaded) | ~$0.10 | ~1.5 min |
| 2026-09-16 | Modal H100 | feasibility run #2 (HF failed: no accelerate; vLLM failed: flashinfer needs nvcc) | ~$0.45 | ~7 min wasted |
| 2026-09-16 | Modal H100 | feasibility run #3 (complete): HF eager 55 TPS, vLLM 423, vLLM+MTP k1/2/3 = 579/641/665 | ~$0.85 | ~13 min |
| 2026-09-16 | DeepSeek | review #1 of qwen35_fast/engine.py vs HF reference | $0 (subscription) | candidates/review1 |
| 2026-09-16 | Modal H100 | custom engine screen #1 (correctness + k=0..3, both MTP hidden variants) | est ~$0.6 | launched |
| 2026-09-16 | Modal H100 | custom screen #1 result: prefill exact vs HF; decode wrong (fla positional-arg bug: beta passed as gk); crash at k=2 | ~$0.35 | bug found by reading fla-core source |
| 2026-09-16 | Modal H100 | custom screen #2 (after fix) | est ~$0.6 | launched |
| 2026-09-16 | Modal H100 | custom screen #2 result: k0=237 TPS, decode matches HF up to near-tie flips; spec k1-3 slower (acceptance unknown) | ~/bin/zsh.6 | |
| 2026-09-16 | Modal H100 | profile session (kernel counts, MTP acceptance) | est ~/bin/zsh.4 | launched |
| 2026-09-16 | Modal H100 | profile #1-#2 crashed on my bugs (filename, torch shadowing) | ~$0.15 | pyflakes now run before every launch |
| 2026-09-16 | Modal H100 | profile #3: k0 4.16ms/step 1391 kernels; k1 8.2ms (masked cuDNN SDPA 2.9ms); MTP acceptance ~68% prose / 83% code | ~$0.35 | results/profile_step.json |
| 2026-09-16 | Modal H100 | custom screen #3: candidates C1 compile_blocks, C2 explicit decode attention; k=0..3 | est ~$0.7 | launched |
