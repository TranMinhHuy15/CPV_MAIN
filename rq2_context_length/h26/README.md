# H26

Seeds: 42, 43, 44. History: 26 snippet vectors. Training: 50 epochs per seed.
This is the historical Long arm. The Short/H1 files in the enclosing original backup are not included as current RQ2 evidence.

The original relative run paths are preserved under `evidence/`. Search `rq2_test_report.json` for `summary_metrics`, `selected_checkpoint`, `checkpoint_sha256`, `capture_fingerprint`, and metric caveats. `validation_history.jsonl` contains epoch-level validation, not test results.
Lossless `.gz` files decompress to the original text unless an explicit security-redaction entry says otherwise. Sources and transformations are indexed in [checks](../../checks/README.md).

Three identical upstream source-ZIP dependencies have been deduplicated into the shared evaluator directory. Before using an original training launcher, run `python tools/restore_shared_dependencies.py` from the repository root; its copies are ignored by Git. This local audit has not launched training.
