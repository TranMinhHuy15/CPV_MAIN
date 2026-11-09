# CPV301 — Temporal history for collision prediction on Nexar

Code, training logs, per-seed metrics and analysis evidence for the CPV301 project at FPT University.
Authors: Tran Minh Huy, Hoang Duy Luu, Duong Duy Loi and Tran Thanh Dat. Mentor: Hai.

**Review 3 evidence release, 2026-11-09:** [`v1.0-review3`](https://github.com/TranMinhHuy15/CPV_MAIN/tree/v1.0-review3).
Assembled from starting commit `2a929b15ac5f1f254d6ef119ce051b1d8daad660`; see the release tag for the immutable publication commit.
Review 3: 2026-11-11; manuscript submission target: 2026-11-28 (team schedule).

## Research questions

| RQ | Question and scope | Evidence |
|---|---|---|
| RQ1 | RiskProp V6, TOP V2 and AdaLEA V4 reference implementations under one scalar evaluator; each retains its method-specific training recipe | [Reference results](rq1_reference/README.md) |
| RQ2 | End-to-end bounded causal LSTM with H7, H13 and H26 snippet histories | [Context-length study](rq2_context_length/README.md) |
| RQ3 | Six matched-initialization adapters on frozen H7 and frozen RiskProp V6, V3.2 | [Adapter study](rq3_adapters/v32_matched_init/README.md) |

RQ2/RQ3 are exploratory follow-ups: the test set had already been inspected. This repository does not claim that additional history is intrinsically useless or that a null contrast establishes equivalence.

## Main results

Test: 1,344 clipped videos, seeds 42/43/44. Values below are **mean ± sample SD across the three seed-level metrics**, not averages of training epochs.

### RQ1 — implemented reference systems

| Model | mAP | Full mAUC | AP@1.5s |
|---|---:|---:|---:|
| RiskProp V6 | 0.8389 ± 0.0021 | 0.8561 ± 0.0056 | 0.8013 ± 0.0049 |
| TOP V2 | 0.7871 ± 0.0067 | 0.8162 ± 0.0051 | 0.7441 ± 0.0141 |
| AdaLEA V4 | 0.7662 ± 0.0038 | 0.7847 ± 0.0015 | 0.7527 ± 0.0056 |

Source: [RQ1_table.csv](rq1_reference/summary/RQ1_table.csv), independently recomputed from the stored per-video scalar predictions and pinned test metadata.
RiskProp seed 44 is the rerun selected at epoch 35; the original failed-resume run is excluded. These are project implementations under the project protocol, not claims of exact published-paper reproduction.

### RQ2 — history length

| Model | mAP | Full mAUC | AP@1.5s |
|---|---:|---:|---:|
| H7 | 0.8066 ± 0.0190 | 0.8291 ± 0.0115 | 0.7489 ± 0.0191 |
| H13 | 0.8051 ± 0.0181 | 0.8298 ± 0.0120 | 0.7511 ± 0.0222 |
| H26 | 0.7983 ± 0.0234 | 0.8170 ± 0.0157 | 0.7585 ± 0.0190 |

Source: [RQ2_table.csv](rq2_context_length/summary/RQ2_table.csv). H26 belongs to an earlier execution campaign. H7/H13 test capture used the documented strict-FP32 recovery path; precision and provenance differences must be disclosed.
H7 has the highest mAP point estimate; H26 has the highest AP@1.5s point estimate. These observations do not establish a significant ranking or an optimal history length.

### RQ3 — what the controls show

On frozen H7, all six adapter-vs-base mAP contrasts have positive point estimates and nominal 95% CIs above zero. AP@1.5s CIs include zero.
The H7 A7–C7 mAP contrast is −0.028413 pp, CI [−0.399163, +0.386411]: adding real history did not show a reliable benefit over repeating the current snippet in this contrast.
On frozen V6, adapter-vs-base CIs include zero. All 10 planned Holm-adjusted tests **within each base** have adjusted p=1.00.

See [raw contrasts](rq3_adapters/v32_matched_init/summary/RQ3_contrasts.csv) and [interpretation limits](rq3_adapters/v32_matched_init/README.md).

## Verify a number

1. Open [EVIDENCE_MAP.md](EVIDENCE_MAP.md) and find the table/quantity.
2. Follow the repository file and exact CSV field/JSON key.
3. Use [ARTIFACTS.md](ARTIFACTS.md) for the originating archive, size, checksum and available Drive link.
4. See [checks/REPORT_TO_HUY.md](checks/REPORT_TO_HUY.md) for disagreements and unresolved items.

`python tools/recompute_tables.py` recomputes the scalar RQ1 tables and aggregates the recorded RQ2/RQ3 point estimates using only the Python standard library. It does not train or execute archived training scripts, and does not rerun the 5,000-draw bootstrap.

## Metrics and evaluator

[evaluator/rq2-28-v1](evaluator/rq2-28-v1/README.md): three lead-time cohorts, AP/mAP, full ROC AUC, raw partial AUC divided by 0.1, separately named standardized partial AUC, and descriptive index-based mTTA.
Index-based mTTA is not guaranteed physical warning time. Its test-negative-frame rank thresholds are not validation-calibrated deployment thresholds.

## Data, code and large artifacts

[Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction), revision `7535d0656dac31d7da2846913bb69c6331d2a70a`.
The split is 1,200 training + 300 validation + 1,344 test = 2,844 records. Video, decoded frames, feature caches and model weights are not distributed in this repository. Small metadata/split files and predictions are preserved to support numerical replay.
Large original archives remain at the locations listed in [ARTIFACTS.md](ARTIFACTS.md). A consolidated public Drive release has not been created or access-verified in this local build.

[REPRODUCE.md](REPRODUCE.md) explains numerical replay, dependency restoration and the limits of retraining. [DEVIATIONS.md](DEVIATIONS.md) records runtime repairs and excluded runs. [THIRD_PARTY.md](THIRD_PARTY.md) records upstream provenance and license notices.
