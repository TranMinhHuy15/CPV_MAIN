# Shared evaluator provenance: rq2-28-v1

The canonical archived evaluator is extracted from `inputs/rq3_selected_source_v1.zip` inside the V3.2 bundle, under `base_h7/rq1_evaluation/`. Its `author_report.py`, `author_reference_metrics.py` and `engine/metrics.py` have exactly the same hashes as the H7/H13/H26 packages; see [comparison](../../checks/evaluator_hash_comparison.json) and [provenance](../../checks/evaluator_provenance.json).

This is a collection of recorded modules, not an assertion that every function uses the same TTA convention. `author_report.py` provides the source-index convention used by RQ2; `engine/metrics.py` also contains differently named physical-time functions. The capture/reference route must be followed when identifying a reported metric.

- AP: non-interpolated average precision; group clips at lead 0.5/1.0/1.5 s.
- mAP: arithmetic mean of the three cohort AP values.
- Full mAUC: arithmetic mean of the full ROC AUC values for those groups.
- Raw partial AUC /0.1: integrate ROC TPR over FPR [0,0.1], divide by 0.1. Its random-ranking expectation is 0.05.
- Standardized partial AUC: the separately named McClish/scikit-learn transformation; random-ranking reference 0.5. Do not mix scales.
- Index-based mTTA: source snippet-index difference divided by 10, inferred anchors, misses zero, no future score extrapolation. It is not necessarily elapsed physical seconds and is not automatically equivalent to a published benchmark’s mTTA.
- FAR in the temporal routine is pooled negative-frame FAR; it is not per-video false alarm probability. Thresholds selected using test-negative ranks are descriptive ROC analysis, not deployment calibration.

No archived evaluator/training script was executed by the release build. The independently written `tools/recompute_tables.py` verifies scalar AP/AUC/pAUC from saved predictions. Stored bootstrap draws were separately reconciled (66 within-base and18 interaction intervals/p-values); generating fresh resamples and replaying dense-curve mTTA remain distinct steps.
