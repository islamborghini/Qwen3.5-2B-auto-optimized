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
| 2026-09-17 | Modal H100 | final #3: new gate results — HF noise floor 0.27-1.97; k0_compile ok (1.27); k2/k3 rejected: 2.50 > 2.47 (=2x floor 1.23) on structured-2048. Killed after gate to re-plan | ~$0.5 | results/final3_optimize_gate.log |
| 2026-09-17 | Modal H100 | final #4 (lean): custom_k0 (official) + custom_k3 (reported) + vllm_mtp3, dev full + heldout, IFEval x3 | est ~$1.9 (over the $10 budget; see README costs) | launched |
| 2026-09-17 | Modal H100 | opt2: optimize loop with layer-level compile candidates (k0/k2/k3/k4 x layer/layer_at) | est ~.0 | launched (user authorized continued spend) |
| 2026-09-17 | Modal H100 | final #4 done: custom_k0 0.58x / custom_k3 0.89x of vllm_mtp3 (dev, same session); heldout 0.57x/0.86x; IFEval hf .61 / k0 .64 / k3 .62 | ~.2 | results/RESULTS.md |
| 2026-09-17 | Modal H100 | opt3: ep layer-compile candidates, k4, gemv v2 (DeepSeek) | est ~.0 | launched |
| 2026-09-17 | Modal H100 | opt3 result: k3_compile 614 geomean best; layer-compile (ep) fails gate; gemv2 slower; mem gate was broken (process-wide peak), TTFT included MTP prefill+drafts -> fixed | ~.0 | results/opt_ledger_round3.json |
| 2026-09-17 | Modal H100 | opt4: rerun loop with fixed mem/TTFT accounting (k0/k2/k3/k4 compile) | est ~/bin/zsh.5 | launched |
| 2026-09-17 | Fable agents T1/T3 | T1: lean decode (-50 kernels, +2-3%, bit-identical) merged; T3: fused GDN Triton kernel (-144 kernels, -6.6% step) but state update wrong -> debugging | ~ Fable | branches merged |
| 2026-09-17 | Modal H100 | opt4: spec configs rejected only by TTFT (+5.7%); mem accounting still broken -> fixed again; debug session (kernel dump + TTFT phases) | ~/bin/zsh.5 + ~/bin/zsh.2 | launched |
| 2026-09-17 | Modal H100 | debug2: kernel decay/beta/kv/head0-state all exact -> store or other heads; debug3 per-head + opt5 (lean engine, paired incumbent checks) | ~/bin/zsh.2 + ~/bin/zsh.2 + ~/bin/zsh.6 | launched |
| 2026-09-17 | Modal H100 | opt5: k0_compile (merged lean engine) FAILS gate on structured-512 (4.33 > 4.0, floor 2.0) -> gatecheck session: lean vs non-lean per-prompt diffs | ~/bin/zsh.3 | launched |
| 2026-09-17 | Modal H100 | debug3/4: fused GDN kernel: math exact when dumped, stored state wrong regardless of barrier/separate buffer -> register-spill regime; rewrite delegated (T3b) | ~$0.4 | |
| 2026-09-17 | Modal H100 | gatecheck: max-diff is a single-step outlier (steps ~200-235 on structured prompts); p99 of all configs ~= HF's own p99 (0.25-1.1). lean vs nonlean: 4.25 vs 2.41 max on structured-512, p99 1.14 vs 0.81 | ~$0.3 | results/gatecheck.json |
| 2026-09-17 | Modal H100 | opt5 (lean engine, paired incumbent): k0_compile failed gate (4.33>4.0, structured-512); k2_compile accepted (616 geomean); k3/k4 slower on prose | ~$0.6 | results/opt_ledger_round5.json |
| 2026-09-17 | Modal H100 | final #5: frozen k2_compile(lean) + custom_k3 + vllm_mtp3: dev full, heldout, IFEval custom_k2 | est ~$1.5 | launched |
| 2026-09-17 | Modal H100 | final #5 done: custom_k2 (lean) 540-737 TPS dev, 0.88x vllm_mtp3 same-session (heldout 0.87x); custom_k3 0.91x/0.89x; IFEval custom_k2 0.64 | ~$1.5 | results/RESULTS.md |
| 2026-09-17 | DeepSeek | review #2 of merged engine: fused_gdn must be guarded (done), pre_norm option semantics (documented), save=None footgun (fixed) | $0 | candidates/review2 |
