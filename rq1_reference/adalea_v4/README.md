# AdaLEA V4, seeds 42/43/44

Source archive: `AdaLEA_V4_RESULTS_20261025_193759_UTC_2188.tar`; see [ARTIFACTS.md](../../ARTIFACTS.md) for its measured checksum.
`code/` (or the explicitly qualified code candidate), `evidence/` and `metrics/` retain separate source/configuration, run evidence and spreadsheets.
Training and evaluation code was inspected as text, not executed by the local release audit. Large checkpoints are excluded.

AdaLEA is the team’s PyTorch reimplementation of Suzuki et al. (2018), not an official AdaLEA author repository. `full_test_metrics.json` distinguishes source-index, physical strict-FAR and physical TP-only TTA conventions.

Canonical numerical results for the current paper are in [RQ1_table.csv](../summary/RQ1_table.csv), with exact source score-file paths in [RQ1_per_seed.csv](../summary/RQ1_per_seed.csv).
