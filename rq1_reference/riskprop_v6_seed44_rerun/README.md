# RiskProp V6 seed 44 rerun

Source archive: `seed44_rerun2_PACK_20261103_114108.tar`; see [ARTIFACTS.md](../../ARTIFACTS.md) for its measured checksum.
`code/` (or the explicitly qualified code candidate), `evidence/` and `metrics/` retain separate source/configuration, run evidence and spreadsheets.
Training and evaluation code was inspected as text, not executed by the local release audit. Large checkpoints are excluded.

The nine core launcher/training/evaluation file hashes match the recorded run manifest for all four available rerun ZIP variants. This verifies the core source but cannot uniquely identify which wrapper ZIP was uploaded. `code_candidate_v3/` is retained with that qualification. The selected checkpoint is epoch 35; the original seed44 run is excluded.

Canonical numerical results for the current paper are in [RQ1_table.csv](../summary/RQ1_table.csv), with exact source score-file paths in [RQ1_per_seed.csv](../summary/RQ1_per_seed.csv).
