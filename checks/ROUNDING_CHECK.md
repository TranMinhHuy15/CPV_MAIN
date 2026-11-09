# Rounding audit against the handover snapshot

| Quantity | Snapshot | Exact source value | Correct display |
|---|---:|---:|---:|
| A7/H7 vs B0, Δ AP@1.5s pp | +0.40 | +0.39473139883196745 | **+0.39** (two decimals) |
| A26/V6 vs B0, Δ mAP CI lower pp | −0.14 | −0.13479705472715808 | **−0.13** (two decimals) |
| Frozen V6 base AP@1.5s | 0.8010 | 0.8010546136830617 | **0.8011** (four decimals) |

Source: each base’s exact `summary/paired_bootstrap.csv`, and V6 `summary/all_models_all28_metrics.csv` (three base rows). These are not alterations of the source results. The manuscript has not been edited.

The Holm family contains ten tests per base, all adjusted p=1.00. RQ3 66 recorded point estimates (33 per base, including supplementary AUC contrasts) are independently recomputed from the per-seed table; see `numeric_verification.json`. All66 within-base CI/p-value rows and18 cross-base CI/p-value rows were independently reconstructed from the archived draws and matched exactly; all20 Holm-adjusted p-values also matched. No fresh resampling was performed. See `bootstrap_draw_reconciliation.json`.
