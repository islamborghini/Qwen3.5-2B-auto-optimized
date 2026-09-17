### Dev suite (12 workloads, 256 output tokens, batch 1, H100). Decode TPS: median (min-max) over 5 reps

| Workload | hf_eager | hf_compile | vllm_plain | vllm_mtp1 | vllm_mtp2 | vllm_mtp3 | custom_k2 | custom_k3 | strongest original | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| dev-prose-128 | 51 (50-52) | n/a | 437 (437-437) | 533 (532-534) | 560 (559-560) | 527 (527-528) | 536 (533-538) | 523 (521-524) | vllm_mtp2 | 0.93x |
| dev-prose-512 | 50 (50-51) | 167 (165-169) | 436 (436-437) | 558 (557-560) | 560 (559-560) | 540 (540-541) | 555 (554-556) | 523 (523-525) | vllm_mtp2 | 0.93x |
| dev-prose-2048 | 50 (50-51) | 171 (170-172) | 432 (432-433) | 569 (567-570) | 592 (592-593) | 565 (564-565) | 545 (544-547) | 527 (523-528) | vllm_mtp2 | 0.89x |
| dev-prose-8192 | 50 (50-51) | n/a | 429 (428-430) | 566 (565-568) | 571 (569-572) | 574 (573-575) | 526 (525-528) | 512 (510-513) | vllm_mtp3 | 0.89x |
| dev-code-128 | 51 (49-51) | n/a | 437 (437-437) | 620 (619-622) | 719 (718-720) | 733 (733-734) | 663 (663-665) | 686 (683-687) | vllm_mtp3 | 0.94x |
| dev-code-512 | 50 (50-52) | 171 (169-172) | 436 (436-436) | 588 (587-589) | 643 (643-644) | 638 (637-638) | 640 (639-640) | 629 (627-629) | vllm_mtp2 | 0.98x |
| dev-code-2048 | 50 (50-53) | 171 (170-172) | 432 (432-433) | 617 (597-618) | 701 (700-703) | 688 (687-689) | 644 (643-646) | 669 (667-671) | vllm_mtp2 | 0.95x |
| dev-code-8192 | 51 (50-51) | n/a | 429 (428-429) | 610 (509-611) | 697 (695-699) | 702 (702-703) | 557 (556-559) | 567 (566-568) | vllm_mtp3 | 0.81x |
| dev-structured-128 | 51 (50-51) | n/a | 437 (436-437) | 648 (543-649) | 798 (796-799) | 914 (913-915) | 745 (737-746) | 862 (860-865) | vllm_mtp3 | 0.94x |
| dev-structured-512 | 50 (50-51) | 171 (170-171) | 436 (436-436) | 650 (648-651) | 798 (797-799) | 875 (874-878) | 727 (727-730) | 806 (805-806) | vllm_mtp3 | 0.92x |
| dev-structured-2048 | 51 (50-52) | 166 (164-169) | 432 (432-433) | 641 (639-644) | 784 (781-784) | 856 (856-863) | 735 (733-736) | 839 (837-841) | vllm_mtp3 | 0.98x |
| dev-structured-8192 | 51 (50-51) | n/a | 429 (428-429) | 639 (637-641) | 765 (760-766) | 823 (820-824) | 723 (720-726) | 793 (790-793) | vllm_mtp3 | 0.96x |

**Geometric-mean speedup of `custom_k3` vs the strongest original baseline per workload: 0.93x** (vs hf_eager: 12.9x).

### Same-session comparison (custom vs vLLM+MTP k=3 measured back-to-back in the final session)

| Workload | vllm_mtp3 TPS | custom_k2 TPS | ratio | custom_k2 TTFT ms | custom_k2 peak GB | custom_k3 TPS | ratio | custom_k3 TTFT ms | custom_k3 peak GB | vllm_mtp3 TTFT ms |
|---|---|---|---|---|---|---|---|---|---|
| dev-prose-128 | 546 (540-546) | 536 (533-538) | 0.98x | 32.2 | 4.94 | 523 (521-524) | 0.96x | 33.8 | 4.97 | 28.7 |
| dev-prose-512 | 559 (558-561) | 555 (554-556) | 0.99x | 32.5 | 4.95 | 523 (523-525) | 0.94x | 32.9 | 4.98 | 26.1 |
| dev-prose-2048 | 585 (585-585) | 545 (544-547) | 0.93x | 32.3 | 5.12 | 527 (523-528) | 0.90x | 37.7 | 5.16 | 33.0 |
| dev-prose-8192 | 592 (590-593) | 526 (525-528) | 0.89x | 106.4 | 5.69 | 512 (510-513) | 0.87x | 106.4 | 5.73 | 111.1 |
| dev-code-128 | 760 (759-762) | 663 (663-665) | 0.87x | 31.7 | 4.94 | 686 (683-687) | 0.90x | 31.9 | 4.97 | 25.0 |
| dev-code-512 | 661 (660-661) | 640 (639-640) | 0.97x | 32.5 | 4.95 | 629 (627-629) | 0.95x | 34.7 | 4.98 | 23.7 |
| dev-code-2048 | 713 (711-714) | 644 (643-646) | 0.90x | 33.0 | 5.12 | 669 (667-671) | 0.94x | 36.5 | 5.16 | 33.6 |
| dev-code-8192 | 726 (724-728) | 557 (556-559) | 0.77x | 107.0 | 5.69 | 567 (566-568) | 0.78x | 106.7 | 5.73 | 107.6 |
| dev-structured-128 | 947 (942-948) | 745 (737-746) | 0.79x | 39.7 | 4.94 | 862 (860-865) | 0.91x | 34.6 | 4.97 | 26.7 |
| dev-structured-512 | 906 (904-907) | 727 (727-730) | 0.80x | 34.3 | 4.95 | 806 (805-806) | 0.89x | 32.6 | 4.98 | 26.1 |
| dev-structured-2048 | 887 (883-888) | 735 (733-736) | 0.83x | 34.1 | 5.12 | 839 (837-841) | 0.95x | 35.2 | 5.16 | 33.4 |
| dev-structured-8192 | 848 (847-850) | 723 (720-726) | 0.85x | 106.3 | 5.72 | 793 (790-793) | 0.93x | 106.3 | 5.75 | 113.1 |

Same-session geomean ratio custom_k2 / vllm_mtp3: **0.88x**

Same-session geomean ratio custom_k3 / vllm_mtp3: **0.91x**

### Startup costs (excluded from TPS)

* hf_eager: 13 s (model load + compile/graph capture + first warmup)
* hf_compile: 8 s (model load + compile/graph capture + first warmup)
* vllm_plain: 102 s (model load + compile/graph capture + first warmup)
* vllm_mtp1: 97 s (model load + compile/graph capture + first warmup)
* vllm_mtp2: 100 s (model load + compile/graph capture + first warmup)
* vllm_mtp3: 103 s (model load + compile/graph capture + first warmup)
* custom_k2: 54 s (model load + compile/graph capture + first warmup)
* custom_k3: 7 s (model load + compile/graph capture + first warmup)

Drift check (final session): {'engine': 'custom_k2', 'workload': 'dev-prose-128', 'tps_end': [545.9218767683043, 545.8141698602691, 545.8771033738841], 'tps_start': [536.896095534525, 537.7522333704266, 536.2524262898789, 534.1333575780991, 532.7928426439871]}
Drift check (baselines session): {'engine': 'hf_eager', 'workload': 'dev-prose-128', 'tps_end': [50.54479313809542, 50.63432136157368, 50.42124682768365], 'tps_start': [51.36572196796822, 50.423214666271505, 49.99672608693329, 51.994163109965726, 50.767039286163545]}

### Held-out prompts (never used during optimization; same session as vllm_mtp3)

| Workload | vllm_mtp3 TPS | custom_k2 TPS | ratio | custom_k3 TPS | ratio |
|---|---|---|---|---|---|
| heldout-prose-128 | 536 (536-536) | 508 (506-509) | 0.95x | 489 (488-491) | 0.91x |
| heldout-prose-512 | 507 (485-508) | 479 (478-480) | 0.95x | 463 (462-465) | 0.91x |
| heldout-prose-2048 | 570 (453-570) | 493 (492-494) | 0.87x | 478 (476-479) | 0.84x |
| heldout-prose-8192 | 497 (392-497) | 518 (516-519) | 1.04x | 488 (488-490) | 0.98x |
| heldout-code-128 | 741 (591-760) | 637 (636-639) | 0.86x | 616 (615-617) | 0.83x |
| heldout-code-512 | 724 (717-726) | 657 (654-659) | 0.91x | 672 (670-673) | 0.93x |
| heldout-code-2048 | 727 (726-729) | 638 (637-640) | 0.88x | 633 (630-633) | 0.87x |
| heldout-code-8192 | 701 (699-702) | 612 (612-614) | 0.87x | 615 (614-616) | 0.88x |
| heldout-structured-128 | 907 (905-908) | 743 (737-745) | 0.82x | 786 (784-788) | 0.87x |
| heldout-structured-512 | 989 (983-990) | 768 (765-769) | 0.78x | 880 (876-882) | 0.89x |
| heldout-structured-2048 | 951 (947-953) | 752 (749-753) | 0.79x | 866 (859-868) | 0.91x |
| heldout-structured-8192 | 962 (961-963) | 739 (737-740) | 0.77x | 850 (848-851) | 0.88x |

Held-out geomean ratio custom_k2 / vllm_mtp3: **0.87x**

Held-out geomean ratio custom_k3 / vllm_mtp3: **0.89x**


### Optimization ledger (bench/optimize.py, dev screen 512+2048)

| candidate | kwargs | correct | geomean TPS (screen) | accepted |
|---|---|---|---|---|
| k0_compile | `{'spec_k': 0, 'compile_blocks': True}` | False (dev-structured-512: decode logits max|diff| 4.332 > tol 4.00) | nan | None |
| k2_compile | `{'spec_k': 2, 'compile_blocks': True}` | True (ok (max|diff| 2.281, tol 2.526)) | 616 | True |
| k3_compile | `{'spec_k': 3, 'compile_blocks': True}` | True (ok (max|diff| 2.281, tol 2.526)) | 588 | False |
| k4_compile | `{'spec_k': 4, 'compile_blocks': True}` | True (ok (max|diff| 2.281, tol 2.526)) | 577 | False |

Frozen incumbent: **k2_compile** ``


### IFEval 100-prompt subset (greedy, non-thinking, normal stopping, max 1280 new tokens)

| engine | prompt-level strict | inst-level strict | prompt-level loose | truncated | elapsed s | changed strict outcomes vs hf (F->T / T->F) | identical text vs hf |
|---|---|---|---|---|---|---|---|
| hf | 0.610 | 0.730 | 0.650 | 11/100 | 1271 | 0 / 0 | 100/100 |
| custom_k0 | 0.640 | 0.755 | 0.690 | 10/100 | 134 | 7 / 4 | 23/100 |
| custom_k2 | 0.640 | 0.748 | 0.680 | 16/100 | 103 | 7 / 4 | 23/100 |
| custom_k3 | 0.620 | 0.736 | 0.690 | 10/100 | 99 | 5 / 4 | 25/100 |

