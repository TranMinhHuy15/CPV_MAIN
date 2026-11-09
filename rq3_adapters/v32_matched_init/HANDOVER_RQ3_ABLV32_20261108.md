# HANDOVER — RQ3 matched ablation V3.2 on frozen H7 and frozen RiskProp V6 (run 2026-11-07, closed 2026-11-08)

Prepared by Tran Minh Huy (via Claude). Status: **finished, all 8 result files downloaded and hash-verified, both Vast machines destroyed after verification.** Exploratory study (the Nexar test set was inspected in earlier rounds).

## 1. Where this sits in the paper (final RQ definitions, decided 2026-11-07)
One thread: *does temporal history help collision prediction on Nexar?*
- **RQ1 — reference implementations.** RiskProp V6, TOP, AdaLEA, each with its own code/recipe, one shared evaluator, 3 seeds. RiskProp V6 83.89 ± 0.21 mAP / 80.13 AP@1.5s (seed 44 = rerun) > TOP 78.71 > AdaLEA 76.62 mAP. Paper RQ1 table must use these numbers (old table has 82.54 ± 2.53).
- **RQ2 — bounded causal LSTM context length (H7 / H13 / H26), end-to-end, same loss and recipe.** Longer context does not help: mAP / AP@1.5s H7 80.66 / 74.89, H13 80.51 / 75.11, H26 79.83 / 75.85. H1 is not an RQ2 arm (one LSTM step is not a real recurrent model — mentor's critique, 2026-11-04); H3 dropped.
- **RQ3 — lightweight history adapters on a frozen predictor.** Frozen H7: adapters recover mAP (+0.4 to +1.0 in v1.1) but not AP@1.5s, best 81.70 < V6. Frozen V6: no gain (|Δ| ≤ 0.25 pp). This ablation (V3.2) is the supporting analysis for RQ3: it isolates what produces the gain. It is **not** a fourth RQ.
- Thesis: per-snippet appearance carries the signal; extra history adds noise, not information.

## 2. What was run (V3.2)
- Six adapter arms × 3 seeds (42, 43, 44) × 50 epochs = 18 trained runs per base, one predeclared config `c2_lr_double` (AdamW 2e-4, wd 1e-4, dropout 0.1, width 512, bound 1). Arms: C7 (current snippet repeated 7×), R7 (last LSTM state), A7 (= M1), A26 (M1 over 26 snippets), U3 (3 windows, uniform fusion), G3 (= M1+M2, learned fusion).
- **Matched initialization:** all arms at the same seed start from the same canonical-G3 shared weights (policy `canonical-g3-shared-state-cpu-fork-v3`, verified identical across arms). This fixes the V2 limitation where A7/G3 were reused from older runs with non-identical init.
- Checkpoint rule: earliest epoch with max validation mAUC@ on 300 validation videos. Test: 1,344 videos × 3 seeds = 4,032 paired. Paired stratified bootstrap, 5,000 resamples, RNG 20261106. Main family: 10 contrasts with Holm correction. Cross-base interaction I = (X − B0)_H7 − (X − B0)_V6 (exploratory, uncorrected).
- Frozen bases: H7 (RQ2/RQ3-v1.1 checkpoints), V6 (seeds 42, 43 original; seed 44 = 2026-11-03 rerun, epoch 35).
- Machines (Vast.ai RTX 4080-class): H7 C.59522618 (Norway), V6 C.59522620 (North Carolina). Launched 2026-11-07 12:05 (local). V6 finished 23:57; H7 test finished 01:15 (8/11) then the report step failed (see §6) and was rerun at 05:10.
- Code package `RQ3_ABLV3_2_RECHECK_FIXED_20261107` (bundle sha256 631f7f3588da4e655353325bc5ebb463c2794cfd2a09b43c561b9dea8a2955fe), 58/58 CPU tests; this was the first CUDA end-to-end run (pilot C7 seed 42, epoch 2 → resume → 3 passed).

## 3. Results (test, percentage points)
Baselines (frozen model re-scored in this pipeline): H7 mAP 82.56 / 78.76 / 80.66 (mean 80.66), AP@1.5s 76.64 / 75.17 / 72.86 (mean 74.89) — **matches RQ2's H7 row exactly**. V6 mAP 83.92 / 84.06 / 83.66 (mean 83.88), AP@1.5s 80.22 / 79.56 / 80.53.

**Δ vs frozen base [CI95]:**

| Arm | H7 ΔmAP | H7 ΔAP@1.5s | V6 ΔmAP | V6 ΔAP@1.5s |
|---|---|---|---|---|
| C7 | +0.57 [0.13, 1.06] | +0.25 [−0.67, 1.10] | +0.23 [−0.02, 0.49] | +0.01 [−0.51, 0.52] |
| R7 | +0.54 [0.10, 0.97] | +0.33 [−0.53, 1.19] | +0.33 [−0.02, 0.72] | +0.02 [−0.61, 0.69] |
| A7 (M1) | +0.54 [0.11, 1.03] | +0.40 [−0.43, 1.22] | +0.27 [−0.12, 0.61] | −0.01 [−0.60, 0.65] |
| A26 | +1.31 [0.56, 2.01] | +0.64 [−0.82, 2.06] | +0.22 [−0.14, 0.54] | −0.07 [−0.75, 0.63] |
| U3 | +1.16 [0.52, 1.74] | +0.65 [−0.70, 1.87] | +0.28 [−0.04, 0.63] | +0.03 [−0.61, 0.65] |
| G3 (M1+M2) | +1.16 [0.49, 1.78] | +0.39 [−0.85, 1.58] | +0.28 [−0.10, 0.67] | −0.03 [−0.69, 0.68] |

- H7: all six arms raise mAP, 3/3 seeds positive each; none raises AP@1.5s.
- V6: no arm has a CI excluding 0 on either metric.
- **Main (Holm) contrasts, both bases: all null, p_holm = 1.0.** Key ones: A7 vs C7 ΔmAP −0.03 [−0.40, 0.39] on H7 (history adds nothing over the current snippet alone); A7 vs R7 +0.00; G3 vs U3 +0.00; U3 / G3 vs A26 ≈ −0.15 [CIs include 0].
- Reading: the H7 mAP gain comes from adapter capacity, not from history. Wide-window arms (A26, U3, G3) sit about 0.6–0.7 pp above C7 on H7, but those pairs are outside the Holm family — descriptive only.
- **Cross-base interaction (exploratory, uncorrected), ΔmAP:** U3 +0.87 [0.19, 1.51] and G3 +0.88 [0.13, 1.56] "effect present"; A26 +1.09 [0.39, 1.76] "trend, seeds disagree" (seed 42 = −0.63); C7 +0.34, R7 +0.21, A7 +0.28 no detectable effect. AP@1.5s: no arm. Outputs: `cross_base_interaction.csv/.json`.

## 4. Selected epochs (from `selection_frozen_before_test.json`, frozen before test)

| Base | Arm | s42 | s43 | s44 |
|---|---|---:|---:|---:|
| H7 | C7 | 11 | 40 | 18 |
| H7 | R7 | 14 | 9 | 20 |
| H7 | A7 | 14 | 15 | 9 |
| H7 | A26 | 11 | 9 | 20 |
| H7 | U3 | 12 | 9 | 18 |
| H7 | G3 | 14 | 9 | 9 |
| V6 | C7 | 13 | 2 | 16 |
| V6 | R7 | 20 | 2 | 10 |
| V6 | A7 | 15 | 2 | 16 |
| V6 | A26 | 18 | 2 | **1** |
| V6 | U3 | 11 | 2 | 16 |
| V6 | G3 | 20 | 2 | 10 |

Notes for questions: on V6 seed 43 every arm selects epoch 2, and A26 seed 44 selects epoch 1. At epoch 1 the adapter output is a near-constant offset near its bound, so the run reproduces the frozen V6 exactly (ΔmAP = ΔAP = 0.0000). This is why A26-V6 is pulled toward 0.

## 5. Integrity (all verified)
Files (Downloads on user's PC; also kept for long-term storage): sha256
- `rq3_ablv32_h7_RESULTS_20261107_2214.tar` f76a87a9aa468edf3abd1c921ffe5af53e24ba6dcc5b2857018f075df55f86e9 (verified by user's certutil)
- `RQ3ABLV32_H7_LOGS_METRICS_ALL.zip` eb823bc55a23edf7383336bc53c15d9d7d125529537ee6b638f19bbda833fd79
- `RQ3ABLV32_H7_RESULTS_SUMMARY.zip` 5d5ac5af1d05d67599cbcca4f828ee86ad08009b6a4d05d8d254637f7c1c8a86
- `rq3_ablv32_v6_RESULTS_20261107_1657.tar` 2e497969194fa4c9b0841c7d98db16f3c985a04bff12ddd86d4f3c8d106434a1 (verified by user's certutil)
- `RQ3ABLV32_V6_LOGS_METRICS_ALL.zip` 1ca52e987077f11d48f7d5297201273905c5165a8615810059b46e5f4b6cfea9
- `RQ3ABLV32_V6_RESULTS_SUMMARY.zip` 4df87d983b54b8f1f1e0e6936a81227ece0e581024904e09b840c34f334fcfbf
- Run scripts: `rq3_ablv32_h7.sh` sha256 0785cf4eea1656147f9225284946f1d3ce0060303abf3f19550cfcc28124adab, `rq3_ablv32_v6.sh` 3fa55f87431d0adb2add679538da86324675ffe62fcf4a4db6aa0c21031a1824; inputs `rq3_selected_source_v1.zip` 464a7635fe5e5c3d837078daab8cd42f4d657896f5602f80822c74f19a2a4220.
- Other checks: both logs zips contain 18 trials, 3 test-complete markers, 21 `test_predictions.json` (7 models × 3 seeds), the pre-test selection lock, full `comparison/`; token scan over all logs found no HF token. V6 re-score parity vs original V6 test predictions: max |Δscore| 1.5e-3 / 1.2e-3 / 3.7e-3, mean ≤ 2.2e-4. The H7 b0 mean matches RQ2's H7 row (80.66 / 74.89).
- Compacted run: `*.pth` and snapshots are deleted after export of `selected_adapter.npz` (in the tar, with hashes); feature caches excluded. Test cannot be re-scored from checkpoints without retraining.

## 6. Deviations and disclosures (state these in the paper/appendix)
1. **H7 report-step crash and in-memory patch.** After all three H7 test seeds finished (exit 0, 2026-11-07 18:15 UTC), `reporting.py` line 84 raised `TypeError: must be real number, not NoneType`: the print formats the v1.1 parity values, which are `None` because this run used `prior_study_root=null`. The prediction files were already written. Source files were **not** edited (their hashes are part of `study_contract.json`). Instead the report step was rerun from a short Python wrapper that replaces `reporting.prior_parity` in memory so `None` becomes `nan`, then bootstrap (`launch.py ... analyze`) and `rq3_ablv32_h7.sh pack` were run normally. Effect: `h7_rescore_vs_v11_parity.csv` shows `nan` (comparison with the v1.1 scoring was not performed); no result value changes. The auto log of the H7 machine therefore ends with `EXIT=1` and the TypeError, and this rerun is documented only here. Wrapper:
   ```python
   import sys, runpy
   H='/workspace/RQ3_ABLV3_2_RECHECK_FIXED_20261107/RQ3_ABLV3_H7'
   sys.path.insert(0,H)
   import reporting
   orig=reporting.prior_parity
   def patched(out,seed):
       r=orig(out,seed)
       for k in ('max_abs_score_diff','mean_abs_score_diff','share_within_1e_3'):
           if r[k] is None: r[k]=float('nan')
       return r
   reporting.prior_parity=patched
   sys.argv=[H+'/run.py','--config',H+'/inputs.json','report']
   runpy.run_path(H+'/run.py',run_name='__main__')
   ```
2. First CUDA end-to-end run of this package; pilot passed.
3. Exploratory: test set seen before; interaction analysis uncorrected, three fixed seeds, no causal claim.
4. V3.2 supersedes V2 for A7/G3 (V2 reused older runs with non-identical init, so V2 cross-run contrasts were exploratory). V2 numbers remain valid as a separate earlier round.
5. V2 history (not part of V3.2): ETA-formatting crash `if len(epoch)==3` after 9 runs on the V6 machine, resumed; fixed as `set(epoch)==set(ARMS)` in V3/V3.2. The old zips on Drive still contain the buggy `reporting.py`.
6. Stray `/workspace/hf.txt` risk: machines destroyed, HF tokens to be revoked by the user.

## 7. Log guide (answering "why did this come out?")
Paths inside `outputs/rq3_ablv3_2_{h7_42_43_44|v6_42_43_44r}/` (in the tar; text logs also in `*_LOGS_METRICS_ALL.zip`).
- Which epoch was chosen and why: `selection_frozen_before_test.json` (frozen before test, `selected_epoch`, `validation_metrics`) + `trials/c2_lr_double/<arm>/seed_<s>/run/named_epochs.json` (per-epoch validation) + `comparison/validation_config_ranking.csv`, `VALIDATION_SELECTION.md`.
- Why "supported / not supported": `comparison/paired_bootstrap.csv`, `paired_bootstrap_decisions.json`, rules in `analysis_plan.md` (archived before training; hash in `rq3abl32*_preregistration_sha256.txt`).
- Per-seed numbers: `comparison/per_seed_summary.csv`, `per_seed_all28_metrics_vs_{H7|V6}.csv`, `all_models_all28_metrics.csv`; per-video scores in `test/runs/<model>/seed_<s>/run/test_predictions.json`.
- Base sanity: V6 `comparison/v6_rescore_parity.csv`, `teachers/*_v6_validation_parity.json`; H7 `h7_rescore_vs_v11_parity.csv` (nan, see §6).
- What was run and with which code: `study_contract.json`, `inputs.resolved.json`, `checked_source/`, `prepared.json`, `runtime_identity.json`, evidence folder (`nvidia_smi*.txt`, `pip_freeze_*.txt`, `SHA256SUMS_outputs.txt`).
- Timing and crash story: `logs/`, `launcher_logs/`, `test/timing_seed*.json`, auto log `rq3abl32*_auto_1107_0505.log` (H7 ends with the TypeError, see §6).

## 8. Open items
- User: revoke HF tokens used; unshare Drive folder `1Or5EZcRN5PN4VYCuvrBF7LAh_hGqQqYw` (anyone-with-link Editor); copy the 8 files to `D:\FA26\CPV\new\` / group Drive.
- Paper: update RQ1 table; write RQ3 with H7 (v1.1), V6 (v1) and this matched ablation as the supporting analysis; Review 3 on 11/11, paper due 28/11.
- Optional: per-arm latency/VRAM table; selected-epoch figure.

## 9. Message to partner (accurate)
"V3.2 matched-init finished on both bases (18 runs each, 3 seeds). V6: no arm differs from frozen V6 on mAP or AP@1.5s; all Holm contrasts null. H7: all six arms raise mAP by +0.5 to +1.3 pp (CIs exclude 0, 3/3 seeds), none raises AP@1.5s; A7 vs C7 is −0.03 [−0.40, 0.39], so the gain does not need history. One report-step crash on the H7 machine (a print that formatted `None` because the optional v1.1 prior was not configured); source untouched, rerun with an in-memory patch, parity vs v1.1 shown as nan. Hashes of the 2 tars and 4 zips match."
