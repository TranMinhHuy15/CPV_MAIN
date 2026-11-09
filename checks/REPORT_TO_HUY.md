# Local release report to Huy — 2026-11-09

## State

The local repository is assembled at `D:\CPV_RELEASE\CPV_MAIN`, cloned from public main commit `2a929b15ac5f1f254d6ef119ce051b1d8daad660`. This report records the pre-publication audit. The approved GitHub publication is identified by tag `v1.0-review3`; see README and Git history for publication state. The manuscript and bibliography were not edited. No training script from an archive was executed, and no cloud machine or paid job was started.

## Verified results

- RQ1: independently replayed all9 scalar score files. RiskProp V6 mAP0.8389042539757049±0.002110657025110011; TOP0.787080263351088±0.006714544739211153; AdaLEA V4 0.7661781611169356±0.0038231799978909547. Sample SD is across seeds42/43/44; V6 seed44 uses the epoch35 rerun.
- TOP/AdaLEA: 22 spreadsheet mean/SD cells reconciled with scalar replay; zero mismatches at source precision.
- RQ2: all9 test reports provide28 metrics each, preserved per seed. Means/sample SD were independently aggregated. H7/H13/H26 mAP=0.8066178196608471 /0.8051288527421444 /0.7983115089493391. Checkpoint paths are in `RQ2_per_seed.csv`.
- RQ3: all42 model–seed scalar prediction files replayed; maximum metric difference 1.67e-15. All66 contrast point estimates match per-seed tables exactly. All66 within-base and18 cross-base CI/p-value rows match the archived5,000-draw arrays exactly. All20 adjusted p-values match. These are checks of stored outputs; no new resampling or new scientific contrasts were introduced.
- Holm: ten tests separately per base, all adjusted p=1.00. Interactions and secondary intervals remain uncorrected; no equivalence claim is made.

## Resolved source questions

- H7 original-v1 ZIP is proven by the exact checksum retained in the executed log export. The later FIXED package is not substituted;23 differing member paths are recorded.
- Three main evaluator modules match the V3.2 canonical source for H7,H13,H26 (nine comparisons).
- 55 runtime-source hash comparisons match. The earlier executed worker matches its runtime contract, while **both distributed earlier V6 ZIPs contain a different worker**; see `earlier_worker_package_comparison.json`.
- All nine core source hashes for the V6 seed44 rerun match all four wrapper ZIP variants. This verifies the training/evaluation core but cannot identify which wrapper was uploaded. The v3 candidate directory remains explicitly qualified.
- The five V3.2 bundle/log/summary file pins match the handover. 35 local artifacts have measured hashes; full V3.2 result tarball hashes are separately labeled as handover-reported only.

## Paper display corrections for the partner

1. A7/H7 ΔAP1.5: +0.39473139883196745pp → **+0.39pp**, not+0.40 at two decimals.
2. A26/V6 ΔmAP lower CI: −0.13479705472715808pp → **−0.13pp**, not−0.14.
3. Frozen V6 AP1.5 mean:0.8010546136830617 → **0.8011**, not0.8010.

These values are preserved at full precision in the source CSVs. The paper was not changed. Comparisons are against the supplied handover snapshot, not a new live Overleaf inspection.

## Explicit remaining limits

- Exact seed44 wrapper ZIP provenance remains unresolved even though its core code matches.
- RQ1 original RiskProp/TOP scalar reports do not define continuous-curve mTTA. Any later replay mTTA in the manuscript needs an explicit source/clock label, not attribution to the original scalar report.
- The exact final figure generator and current TeX were not supplied; no figure/code or reference edits are fabricated.
- Consolidated Drive release, upload of local-only artifacts and anonymous Drive access checks are not performed. Original Drive IDs remain linked, and missing uploads are labeled in ARTIFACTS.
- Dense-curve mTTA has not been recomputed in this release. GPU training/capture was not rerun. Test-set reuse remains exploratory; bootstrap uncertainty conditions on three fixed seeds.

## Local packaging

Original source archives are unchanged. Large checkpoints, Nexar media, caches and dense curve dumps remain external. Long text logs use lossless gzip. Duplicate source ZIP dependencies have a restoration helper. Two small bootstrap NPZs are statistical arrays, not weights.
Secret-pattern scan: 0 redacted files; 0 remaining findings in the inspected text and embedded source/XLSX text. This scan is not a general security guarantee.

Use `README.md`, `EVIDENCE_MAP.md`, `DEVIATIONS.md`, `REPRODUCE.md`, `ARTIFACTS.md` and the machine-readable checks to review the evidence published with the user's authorization.
