# RQ1 — reference implementations

Each system retains its own training recipe. The shared scalar evaluator controls evaluation definitions, not training budgets, architectures or optimization. Compare the complete implemented systems; do not attribute all differences solely to one architecture or loss.

- [Summary](summary/RQ1_table.csv), [per-seed values and score-file provenance](summary/RQ1_per_seed.csv).
- [RiskProp V6 seeds 42/43](riskprop_v6/README.md).
- [RiskProp V6 seed 44 rerun](riskprop_v6_seed44_rerun/README.md).
- [TOP V2](top_v2/README.md).
- [AdaLEA V4](adalea_v4/README.md).

Scalar scores are replayed against the pinned `metadata_snapshot/solution.csv`. Public/private rows are kept separate; Combined metrics are recomputed on the union of clips in each lead-time group, not averaged from Public and Private metrics.

RiskProp and TOP scalar reports explicitly contain no continuous-curve mTTA. AdaLEA has a separate full-curve report. Earlier V6 replay mTTA exists in the earlier RQ3 evidence, but must not be silently substituted into the original RQ1 scalar report. See the reproducibility limitations in the checks report.
