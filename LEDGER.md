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
