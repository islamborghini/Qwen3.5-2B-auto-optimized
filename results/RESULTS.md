### Dev suite (12 workloads, 256 output tokens, batch 1, H100). Decode TPS: median (min-max) over 5 reps

| Workload | hf_eager | hf_compile | vllm_plain | vllm_mtp1 | vllm_mtp2 | vllm_mtp3 | custom_k0 | custom_k3 | strongest original | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| dev-prose-128 | 51 (50-52) | n/a | 437 (437-437) | 533 (532-534) | 560 (559-560) | 527 (527-528) | 390 (389-390) | 479 (479-479) | vllm_mtp2 | 0.86x |
| dev-prose-512 | 50 (50-51) | 167 (165-169) | 436 (436-437) | 558 (557-560) | 560 (559-560) | 540 (540-541) | 390 (389-390) | 488 (488-488) | vllm_mtp2 | 0.87x |
| dev-prose-2048 | 50 (50-51) | 171 (170-172) | 432 (432-433) | 569 (567-570) | 592 (592-593) | 565 (564-565) | 390 (390-390) | 479 (479-479) | vllm_mtp2 | 0.81x |
| dev-prose-8192 | 50 (50-51) | n/a | 429 (428-430) | 566 (565-568) | 571 (569-572) | 574 (573-575) | 389 (388-389) | 469 (468-469) | vllm_mtp3 | 0.82x |
| dev-code-128 | 51 (49-51) | n/a | 437 (437-437) | 620 (619-622) | 719 (718-720) | 733 (733-734) | 390 (390-390) | 610 (609-610) | vllm_mtp3 | 0.83x |
| dev-code-512 | 50 (50-52) | 171 (169-172) | 436 (436-436) | 588 (587-589) | 643 (643-644) | 638 (637-638) | 390 (390-390) | 583 (583-583) | vllm_mtp2 | 0.91x |
| dev-code-2048 | 50 (50-53) | 171 (170-172) | 432 (432-433) | 617 (597-618) | 701 (700-703) | 688 (687-689) | 390 (390-390) | 603 (603-603) | vllm_mtp2 | 0.86x |
| dev-code-8192 | 51 (50-51) | n/a | 429 (428-429) | 610 (509-611) | 697 (695-699) | 702 (702-703) | 388 (388-388) | 568 (567-569) | vllm_mtp3 | 0.81x |
| dev-structured-128 | 51 (50-51) | n/a | 437 (436-437) | 648 (543-649) | 798 (796-799) | 914 (913-915) | 390 (389-390) | 788 (788-789) | vllm_mtp3 | 0.86x |
| dev-structured-512 | 50 (50-51) | 171 (170-171) | 436 (436-436) | 650 (648-651) | 798 (797-799) | 875 (874-878) | 390 (390-390) | 735 (734-735) | vllm_mtp3 | 0.84x |
| dev-structured-2048 | 51 (50-52) | 166 (164-169) | 432 (432-433) | 641 (639-644) | 784 (781-784) | 856 (856-863) | 390 (390-390) | 766 (766-767) | vllm_mtp3 | 0.90x |
| dev-structured-8192 | 51 (50-51) | n/a | 429 (428-429) | 639 (637-641) | 765 (760-766) | 823 (820-824) | 388 (388-389) | 731 (730-731) | vllm_mtp3 | 0.89x |

**Geometric-mean speedup of `custom_k3` vs the strongest original baseline per workload: 0.85x** (vs hf_eager: 11.8x).

### Same-session comparison (custom vs vLLM+MTP k=3 measured back-to-back in the final session)

| Workload | vllm_mtp3 TPS | custom_k0 TPS | ratio | custom_k0 TTFT ms | custom_k0 peak GB | custom_k3 TPS | ratio | custom_k3 TTFT ms | custom_k3 peak GB | vllm_mtp3 TTFT ms |
|---|---|---|---|---|---|---|---|---|---|
| dev-prose-128 | 523 (514-526) | 390 (389-390) | 0.74x | 39.8 | 4.78 | 479 (479-479) | 0.92x | 44.8 | 4.97 | 30.4 |
| dev-prose-512 | 534 (525-539) | 390 (389-390) | 0.73x | 40.0 | 4.79 | 488 (488-488) | 0.91x | 45.6 | 4.98 | 29.6 |
| dev-prose-2048 | 559 (556-570) | 390 (390-390) | 0.70x | 41.1 | 4.96 | 479 (479-479) | 0.86x | 46.3 | 5.16 | 39.2 |
| dev-prose-8192 | 577 (576-579) | 389 (388-389) | 0.67x | 105.5 | 5.53 | 469 (468-469) | 0.81x | 115.5 | 5.73 | 135.0 |
| dev-code-128 | 725 (712-736) | 390 (390-390) | 0.54x | 39.9 | 4.78 | 610 (609-610) | 0.84x | 44.7 | 4.97 | 30.3 |
| dev-code-512 | 631 (616-637) | 390 (390-390) | 0.62x | 40.2 | 4.79 | 583 (583-583) | 0.92x | 45.7 | 4.98 | 29.9 |
| dev-code-2048 | 682 (681-695) | 390 (390-390) | 0.57x | 40.8 | 4.96 | 603 (603-603) | 0.88x | 46.6 | 5.16 | 39.9 |
| dev-code-8192 | 691 (642-706) | 388 (388-388) | 0.56x | 106.1 | 5.53 | 568 (567-569) | 0.82x | 114.9 | 5.73 | 138.5 |
| dev-structured-128 | 871 (791-888) | 390 (389-390) | 0.45x | 39.4 | 4.78 | 788 (788-789) | 0.91x | 45.0 | 4.97 | 31.2 |
| dev-structured-512 | 842 (631-850) | 390 (390-390) | 0.46x | 39.9 | 4.79 | 735 (734-735) | 0.87x | 45.6 | 4.98 | 30.1 |
| dev-structured-2048 | 816 (625-840) | 390 (390-390) | 0.48x | 40.7 | 4.96 | 766 (766-767) | 0.94x | 46.9 | 5.16 | 40.0 |
| dev-structured-8192 | 760 (730-778) | 388 (388-389) | 0.51x | 105.4 | 5.56 | 731 (730-731) | 0.96x | 115.2 | 5.75 | 145.1 |

Same-session geomean ratio custom_k0 / vllm_mtp3: **0.58x**

Same-session geomean ratio custom_k3 / vllm_mtp3: **0.89x**

### Startup costs (excluded from TPS)

* hf_eager: 13 s (model load + compile/graph capture + first warmup)
* hf_compile: 8 s (model load + compile/graph capture + first warmup)
* vllm_plain: 102 s (model load + compile/graph capture + first warmup)
* vllm_mtp1: 97 s (model load + compile/graph capture + first warmup)
* vllm_mtp2: 100 s (model load + compile/graph capture + first warmup)
* vllm_mtp3: 103 s (model load + compile/graph capture + first warmup)
* custom_k0: 58 s (model load + compile/graph capture + first warmup)
* custom_k3: 10 s (model load + compile/graph capture + first warmup)

Drift check (final session): {'engine': 'custom_k0', 'workload': 'dev-prose-128', 'tps_end': [389.3460417917089, 389.67235912528037, 389.8534526653392], 'tps_start': [389.6916764957094, 389.5715431670089, 389.5927623980503, 389.3695105798855, 389.6531794234045]}
Drift check (baselines session): {'engine': 'hf_eager', 'workload': 'dev-prose-128', 'tps_end': [50.54479313809542, 50.63432136157368, 50.42124682768365], 'tps_start': [51.36572196796822, 50.423214666271505, 49.99672608693329, 51.994163109965726, 50.767039286163545]}

### Held-out prompts (never used during optimization; same session as vllm_mtp3)

| Workload | vllm_mtp3 TPS | custom_k0 TPS | ratio | custom_k3 TPS | ratio |
|---|---|---|---|---|---|
| heldout-prose-128 | 514 (513-520) | 392 (392-392) | 0.76x | 436 (436-436) | 0.85x |
| heldout-prose-512 | 483 (479-490) | 392 (390-392) | 0.81x | 409 (409-409) | 0.85x |
| heldout-prose-2048 | 551 (542-554) | 392 (391-392) | 0.71x | 408 (408-409) | 0.74x |
| heldout-prose-8192 | 483 (469-484) | 391 (390-391) | 0.81x | 393 (393-393) | 0.81x |
| heldout-code-128 | 731 (711-737) | 392 (392-392) | 0.54x | 558 (558-558) | 0.76x |
| heldout-code-512 | 700 (678-701) | 392 (392-392) | 0.56x | 589 (589-589) | 0.84x |
| heldout-code-2048 | 707 (631-709) | 392 (392-392) | 0.55x | 589 (587-589) | 0.83x |
| heldout-code-8192 | 680 (674-683) | 391 (390-391) | 0.57x | 587 (586-587) | 0.86x |
| heldout-structured-128 | 871 (862-881) | 392 (386-392) | 0.45x | 724 (724-724) | 0.83x |
| heldout-structured-512 | 946 (940-954) | 392 (391-392) | 0.41x | 800 (800-800) | 0.85x |
| heldout-structured-2048 | 923 (919-926) | 392 (391-392) | 0.42x | 788 (784-788) | 0.85x |
| heldout-structured-8192 | 937 (932-939) | 390 (389-391) | 0.42x | 784 (783-785) | 0.84x |

Held-out geomean ratio custom_k0 / vllm_mtp3: **0.57x**

Held-out geomean ratio custom_k3 / vllm_mtp3: **0.83x**


### IFEval 100-prompt subset (greedy, non-thinking, normal stopping, max 1280 new tokens)

| engine | prompt-level strict | inst-level strict | prompt-level loose | truncated | elapsed s | changed strict outcomes vs hf (F->T / T->F) | identical text vs hf |
|---|---|---|---|---|---|---|---|
| hf | 0.610 | 0.730 | 0.650 | 11/100 | 1271 | 0 / 0 | 100/100 |
| custom_k0 | 0.640 | 0.755 | 0.690 | 10/100 | 134 | 7 / 4 | 23/100 |
| custom_k3 | 0.620 | 0.736 | 0.690 | 10/100 | 99 | 5 / 4 | 25/100 |

