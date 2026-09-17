### Dev suite (12 workloads, 256 output tokens, batch 1, H100). Decode TPS: median (min-max) over 5 reps

| Workload | hf_eager | hf_compile | vllm_plain | vllm_mtp1 | vllm_mtp2 | vllm_mtp3 | custom_k2 | strongest original | speedup |
|---|---|---|---|---|---|---|---|---|---|
| dev-prose-128 | 51 (50-52) | n/a | 437 (437-437) | 533 (532-534) | 560 (559-560) | 527 (527-528) | 600 (600-601) | vllm_mtp2 | 1.07x |
| dev-prose-512 | 50 (50-51) | 167 (165-169) | 436 (436-437) | 558 (557-560) | 560 (559-560) | 540 (540-541) | 582 (581-582) | vllm_mtp2 | 1.04x |
| dev-prose-2048 | 50 (50-51) | 171 (170-172) | 432 (432-433) | 569 (567-570) | 592 (592-593) | 565 (564-565) | 618 (617-618) | vllm_mtp2 | 1.04x |
| dev-prose-8192 | 50 (50-51) | n/a | 429 (428-430) | 566 (565-568) | 571 (569-572) | 574 (573-575) | 604 (603-604) | vllm_mtp3 | 1.05x |
| dev-code-128 | 51 (49-51) | n/a | 437 (437-437) | 620 (619-622) | 719 (718-720) | 733 (733-734) | 700 (700-700) | vllm_mtp3 | 0.95x |
| dev-code-512 | 50 (50-52) | 171 (169-172) | 436 (436-436) | 588 (587-589) | 643 (643-644) | 638 (637-638) | 719 (718-720) | vllm_mtp2 | 1.12x |
| dev-code-2048 | 50 (50-53) | 171 (170-172) | 432 (432-433) | 617 (597-618) | 701 (700-703) | 688 (687-689) | 738 (736-738) | vllm_mtp2 | 1.05x |
| dev-code-8192 | 51 (50-51) | n/a | 429 (428-429) | 610 (509-611) | 697 (695-699) | 702 (702-703) | 661 (660-662) | vllm_mtp3 | 0.94x |
| dev-structured-128 | 51 (50-51) | n/a | 437 (436-437) | 648 (543-649) | 798 (796-799) | 914 (913-915) | 848 (846-849) | vllm_mtp3 | 0.93x |
| dev-structured-512 | 50 (50-51) | 171 (170-171) | 436 (436-436) | 650 (648-651) | 798 (797-799) | 875 (874-878) | 811 (811-812) | vllm_mtp3 | 0.93x |
| dev-structured-2048 | 51 (50-52) | 166 (164-169) | 432 (432-433) | 641 (639-644) | 784 (781-784) | 856 (856-863) | 834 (833-838) | vllm_mtp3 | 0.97x |
| dev-structured-8192 | 51 (50-51) | n/a | 429 (428-429) | 639 (637-641) | 765 (760-766) | 823 (820-824) | 807 (807-808) | vllm_mtp3 | 0.98x |

**Geometric-mean speedup of `custom_k2` vs the strongest original baseline per workload: 1.01x** (vs hf_eager: 14.0x).

### Same-session comparison (custom vs vLLM+MTP k=3 measured back-to-back in the final session)

| Workload | vllm_mtp3 TPS | custom_k2 TPS | ratio | custom_k2 TTFT ms | custom_k2 peak GB | vllm_mtp3 TTFT ms |
|---|---|---|---|---|---|
| dev-prose-128 | 528 (527-529) | 600 (600-601) | 1.14x | 35.7 | 4.94 | 30.1 |
| dev-prose-512 | 538 (537-539) | 582 (581-582) | 1.08x | 36.4 | 4.95 | 30.1 |
| dev-prose-2048 | 567 (566-568) | 618 (617-618) | 1.09x | 35.8 | 5.12 | 39.0 |
| dev-prose-8192 | 576 (575-576) | 604 (603-604) | 1.05x | 112.3 | 5.69 | 125.6 |
| dev-code-128 | 736 (735-737) | 700 (700-700) | 0.95x | 36.0 | 4.94 | 27.8 |
| dev-code-512 | 637 (617-638) | 719 (718-720) | 1.13x | 36.4 | 4.95 | 28.2 |
| dev-code-2048 | 691 (664-695) | 738 (736-738) | 1.07x | 36.7 | 5.12 | 40.0 |
| dev-code-8192 | 703 (684-705) | 661 (660-662) | 0.94x | 111.3 | 5.69 | 125.5 |
| dev-structured-128 | 915 (885-918) | 848 (846-849) | 0.93x | 35.6 | 4.94 | 29.5 |
| dev-structured-512 | 871 (844-875) | 811 (811-812) | 0.93x | 37.0 | 4.95 | 30.3 |
| dev-structured-2048 | 860 (857-862) | 834 (833-838) | 0.97x | 37.8 | 5.12 | 38.9 |
| dev-structured-8192 | 824 (822-825) | 807 (807-808) | 0.98x | 111.9 | 5.72 | 125.9 |

Same-session geomean ratio custom_k2 / vllm_mtp3: **1.02x**

### Startup costs (excluded from TPS)

* hf_eager: 13 s (model load + compile/graph capture + first warmup)
* hf_compile: 8 s (model load + compile/graph capture + first warmup)
* vllm_plain: 102 s (model load + compile/graph capture + first warmup)
* vllm_mtp1: 97 s (model load + compile/graph capture + first warmup)
* vllm_mtp2: 100 s (model load + compile/graph capture + first warmup)
* vllm_mtp3: 103 s (model load + compile/graph capture + first warmup)
* custom_k2: 55 s (model load + compile/graph capture + first warmup)

Drift check (final session): {'engine': 'vllm_mtp3', 'workload': 'dev-prose-128', 'tps_end': [529.8059212456122, 530.508533642719, 530.3447754954936], 'tps_start': [528.5107529906481, 528.1129037292599, 527.9588409747331, 526.5628240587627, 527.8866974796244]}
Drift check (baselines session): {'engine': 'hf_eager', 'workload': 'dev-prose-128', 'tps_end': [50.54479313809542, 50.63432136157368, 50.42124682768365], 'tps_start': [51.36572196796822, 50.423214666271505, 49.99672608693329, 51.994163109965726, 50.767039286163545]}

### Held-out prompts (never used during optimization; same session as vllm_mtp3)

| Workload | vllm_mtp3 TPS | custom_k2 TPS | ratio |
|---|---|---|---|
| heldout-prose-128 | 520 (519-520) | 552 (552-552) | 1.06x |
| heldout-prose-512 | 489 (489-490) | 560 (560-560) | 1.14x |
| heldout-prose-2048 | 553 (552-555) | 594 (593-595) | 1.07x |
| heldout-prose-8192 | 482 (482-483) | 528 (527-528) | 1.09x |
| heldout-code-128 | 734 (725-737) | 726 (724-726) | 0.99x |
| heldout-code-512 | 700 (677-702) | 718 (717-719) | 1.03x |
| heldout-code-2048 | 707 (702-711) | 730 (730-731) | 1.03x |
| heldout-code-8192 | 681 (680-683) | 704 (703-704) | 1.03x |
| heldout-structured-128 | 879 (877-881) | 837 (837-838) | 0.95x |
| heldout-structured-512 | 953 (951-955) | 866 (866-866) | 0.91x |
| heldout-structured-2048 | 923 (920-924) | 844 (843-846) | 0.91x |
| heldout-structured-8192 | 935 (935-937) | 835 (834-836) | 0.89x |

Held-out geomean ratio custom_k2 / vllm_mtp3: **1.01x**


### Optimization ledger (bench/optimize.py, dev screen 512+2048)

| candidate | kwargs | correct | geomean TPS (screen) | accepted |
|---|---|---|---|---|
| k0_compile | `{'spec_k': 0, 'compile_blocks': True}` | False (dev-structured-512: decode logits max|diff| 4.363 > tol 3.93) | nan | None |
| k2_compile | `{'spec_k': 2, 'compile_blocks': True}` | True (ok (max|diff| 2.562, tol 2.875)) | 594 | True |
| k2_compile_fused | `{'spec_k': 2, 'compile_blocks': True, 'fused_gdn': True}` | True (ok (max|diff| 2.406, tol 2.875)) | 665 | True |

Frozen incumbent: **k2_compile_fused** ``


### IFEval 100-prompt subset (greedy, non-thinking, normal stopping, max 1280 new tokens)

| engine | prompt-level strict | inst-level strict | prompt-level loose | truncated | elapsed s | changed strict outcomes vs hf (F->T / T->F) | identical text vs hf |
|---|---|---|---|---|---|---|---|
| hf | 0.610 | 0.730 | 0.650 | 11/100 | 1271 | 0 / 0 | 100/100 |
| custom_k0 | 0.640 | 0.755 | 0.690 | 10/100 | 134 | 7 / 4 | 23/100 |
| custom_k2 | 0.650 | 0.761 | 0.700 | 10/100 | 97 | 8 / 4 | 25/100 |
| custom_k3 | 0.620 | 0.736 | 0.690 | 10/100 | 99 | 5 / 4 | 25/100 |

