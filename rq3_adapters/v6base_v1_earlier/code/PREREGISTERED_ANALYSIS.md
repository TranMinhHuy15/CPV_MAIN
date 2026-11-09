# Pre-registered test analysis — frozen-V6 study (fixed BEFORE the `test` phase)

Written 2026-11-05, before any module in this study is trained or scored. Do not edit after `tune`
starts; the sha256 of this file and `bootstrap_paired.py` is recorded in the run log.

## Question
Do causal history modules trained on top of a **frozen RiskProp V6** (author code, no RNN) improve
V6 on the held-out 1,344-video Nexar test set?

Bases: V6 seed 42 and 43 (original three-seed run) and seed 44 from the 2026-11-03 **rerun**
(the original seed-44 run had a resume defect and is excluded from this study).

## Modules (one predeclared configuration, c2_lr_double: AdamW lr 2e-4, wd 1e-4, dropout 0.1, width 512)
- **M1**: projection 2048→512, LSTM512 over the last 7 snippets, causal age-penalised attention, bounded
  residual on the V6 logit (zero-initialised output, so training starts exactly at V6).
- **M1+M2**: M1 with windows 7/13/26 fused by softmax.
- **M1+DX**: M1 with causal lag differences u_t − u_(t−k), k = 1, 3, 6, between snippet tokens.
- Epoch per run: earliest maximum validation mAUC@ (300 validation videos). No test-based selection.

## Zero-parameter control
- **smooth7**: causal mean of the last 7 points of the b0 (V6) risk curve. Fixed in advance, not tuned.

## Gate before training (stop if it fails)
Frozen-V6 logits computed from our cached features must reproduce each V6 run's own saved validation
snippet probabilities (300 videos × 30 snippets) to max |Δp| ≤ 2e-3.

## Primary comparison and metric
- Each module vs **b0 = the same frozen V6, re-scored in the same test run** (paired, same seed).
- Primary metric: Combined **AP@1.5s**. Secondary: Combined mAP, mAUC.
- Guardrail: mean ΔmAP ≥ −1.0 percentage point.

## Statistics
- Paired bootstrap over test videos, resampling within each (lead-time group × label) stratum, identical
  indices for both sides. 5,000 resamples, RNG seed 20261105 (`bootstrap_paired.py`).
- Per seed: Δ, 95 % percentile CI, P(Δ ≤ 0). Across seeds: mean of the three per-seed Δ, CI from the
  mean of the per-seed bootstrap draws.

## Decision rule (per module, and for smooth7 as a control)
"Supported improvement" only if ALL hold:
1. 95 % CI lower bound of the 3-seed mean ΔAP@1.5s > 0;
2. ΔAP@1.5s > 0 in all three seeds;
3. mean ΔmAP ≥ −1.0 pp.
Otherwise "not supported (trend or no effect)". Three modules are tested without multiplicity
correction; any single "supported" result is described as exploratory.

## Reported but NOT used for decisions
- Each module vs smooth7 (does the module beat trivial temporal smoothing?).
- mAP / mAUC CIs; all 28 metrics per seed; re-scored V6 vs the V6 run's own saved test predictions.
- Comparison with the v1.1 H7 study (different base) is descriptive only.

## Context
The test set was inspected in earlier studies (RQ1/RQ2/RQ3 v1.1), so this study is exploratory.
