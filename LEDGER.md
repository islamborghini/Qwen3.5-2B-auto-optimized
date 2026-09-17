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
| 2026-09-16 | Modal H100 | screen #3 result: compile+explicit-attn: k0 401 TPS; k2 486-692; k3 454-780 | ~/bin/zsh.7 | accepted as new incumbent |
| 2026-09-17 | Modal H100 | baselines full stage (hf_eager, hf_compile, vllm_plain, vllm_mtp1-3), 12 workloads x 5 reps | est ~/bin/zsh.9 | launched |
| 2026-09-17 | Modal H100 | baselines #1 segfault in hf_compile(reduce-overhead) before any measurement | ~/bin/zsh.3 | documented; relaunched with subprocess isolation, compile mode=default |
| 2026-09-17 | DeepSeek | gemv.py Triton skinny-GEMV kernel (reviewed by Fable: tile configs reduced to fit SMEM) | /bin/zsh (subscription) | candidates/gemv |
| 2026-09-17 | Modal H100 | run_opt: gemv test/bench + compiled-step profile + k0/2/3 gemv on/off | est ~/bin/zsh.5 | launched (parallel to baselines, separate GPU) |
| 2026-09-17 | Modal H100 | baselines #2 killed: co-resident vLLM engines starved HF/others of CPU (HF 36 vs 55 TPS); protocol -> sequential, cpu=4 | ~/bin/zsh.9 | contaminated numbers discarded |
| 2026-09-17 | Modal H100 | run_opt: compiled k0 2.53ms/step, 505 kernels (mm 1.62ms); gemv test failed (mask bug, fixed) | ~/bin/zsh.4 | results/run_opt.json |
| 2026-09-17 | Modal H100 | baselines #3 (sequential, cpu=4) | est ~.0 | launched |
| 2026-09-17 | Modal H100 | orphaned app from killed baselines #2 ran ~13 extra min before manual stop | ~/bin/zsh.9 | lesson: pkill does not stop ephemeral app; use modal app stop |
| 2026-09-17 | Modal H100 | final session: optimize loop (7 candidates) -> frozen config -> dev full + heldout vs vllm_mtp3 (same session) -> IFEval custom + HF | est ~.0 (uses reserve) | launched |
| 2026-09-17 | Modal H100 | final #1 killed: greedy-divergence gate rejected spec candidates (structured-2048 @159, HF margin 11) and froze k0; gate replaced by teacher-forced decode-logit check | ~/bin/zsh.6 | gemv: lm_head 3060 vs cuBLAS 2866 GB/s; N=32 slower |
| 2026-09-17 | Modal H100 | final #2: new decode-logit gate; candidates k0/k2/k3 compile; dev full + heldout vs vllm_mtp3 same-session; IFEval custom + HF | est ~.0 | launched |
| 2026-09-17 | Modal H100 | final #2 crashed on logit-slice shape bug (HF load only) | ~/bin/zsh.3 | fixed; final #3 launched |
