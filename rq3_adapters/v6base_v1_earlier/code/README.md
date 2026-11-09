# RQ3-V6BASE — history modules on a frozen RiskProp V6 (code-only release)

9 runs: 3 modules (M1, M1+M2, M1+DX) × 1 predeclared configuration (c2_lr_double) × 3 frozen V6 bases,
50 epochs each. Bases: V6 seed 42 and 43 (original three-seed run) and seed 44 from the 2026-11-03
rerun (the original seed-44 run is excluded because of its resume defect). Built on the v1.1 H7 package;
the frozen base, the third module and the comparison set are what changed.

Read **AI_HANDOFF.md** (structure, formulas, inputs, phases, outputs) and **PREREGISTERED_ANALYSIS.md**
(decision rule fixed before training) first.

## Inputs prepared by the operator
1. Nexar_HF, original MP4s: train / test-public / test-private (2,844 files).
2. `rq3_selected_source_v1.zip` (sha256 464a7635…), the same source archive as v1.1.
3. Three V6 run folders, each holding exactly: `run_manifest.json`, `training_status.json`,
   `validation_history.jsonl`, `validation_predictions.json`, `test_predictions.json` and the one selected
   `best_mAUC@_epoch_N.pth` (built on Windows by `prep_v6_bases.ps1` into `V6_BASES.tar`).

## Run
```bash
python3 run.py --config inputs.json preflight
python3 run.py --config inputs.json all
python3 bootstrap_paired.py --output-root OUT --solution OUT/checked_source/base_h7/metadata_snapshot/solution.csv
```
Or let `rq3_v6base.sh auto` do everything inside tmux (download, prep, run, bootstrap, pack).

## Safety gates
- Preflight: seed, source commit, data revision, preprocessing and the data adapter hash of each V6 run must
  match the source used for extraction; each folder must hold exactly the earliest-max-validation checkpoint.
- **V6 validation parity (before any training):** our frozen-V6 logits on the cached validation features must
  reproduce each V6 run's saved validation snippet probabilities (300 × 30) within 2e-3; otherwise stop.
- Adapter outputs are zero-initialised, so every module starts exactly at V6; frozen weights are checked
  bit-for-bit after training; 8-video full-pipeline parity per path at test time.
- Test report: re-scored V6 vs the V6 run's own saved test predictions (diagnostic, `v6_rescore_parity.csv`).

## Outputs (comparison/)
`RESULTS.md`, `per_seed_summary.csv` (9 rows), `per_seed_all28_metrics_vs_V6.csv`, `all_models_all28_metrics.csv`,
`v6_rescore_parity.csv`, `VALIDATION_SELECTION.md`, `paired_bootstrap.csv`, `paired_bootstrap_decisions.json`.

CPU tests: `python3 -m unittest discover -s tests` (22 tests). GPU end-to-end has not been run in the
authoring session; the gates above are what protect the real run.
