# RQ3 — V3.2 matched-initialization adapters

Six arms × three seeds × 50 epochs on each frozen base: 18 runs for H7 and 18 for V6. The predeclared configuration is `c2_lr_double`; it is not a new sweep over learning rates.

| Arm | Controlled input/readout |
|---|---|
| C7 | Current snippet repeated seven times; no additional historical snippet |
| R7 | Real H7 sequence, final LSTM state |
| A7 | Real H7 sequence, age-biased attention |
| A26 | H26 adapter window; distinct from the RQ2 end-to-end LSTM2048 model |
| U3 | H7/H13/H26 windows with uniform fusion |
| G3 | Same windows with learned fusion |

Projection 2048→512 plus LayerNorm, reset LSTM512, bounded residual logit correction. Same-seed shared parameters start from the canonical-G3 state (`canonical-g3-shared-state-cpu-fork-v3`). The frozen base is not trained. Selecting the earliest epoch with maximum validation mAUC is recorded in each `selection_frozen_before_test.json`.

## Results and inference limits

[All 28 metrics per seed](h7/summary/all_models_all28_metrics.csv), [V6 counterpart](v6/summary/all_models_all28_metrics.csv), [aggregated mean/SD](summary/RQ3_all28_mean_sd.csv), [contrasts](summary/RQ3_contrasts.csv).

The adapter-vs-base CIs shown in the paper are secondary, nominal, unadjusted intervals. The main Holm family contains five adapter-pair contrasts × two metrics = ten tests **per base**. All adjusted p-values are 1.00. Failure to reject does not prove equality.

The 5,000 stratified bootstrap draws resample 1,344 clip records, with common draws across models, seeds and bases (RNG 20261106). 4,032 denotes clip–seed pairs, not independent resampling units. The bootstrap conditions on the three trained seeds and does not estimate a population distribution of training seeds.

C7 can improve H7 mAP without new history. This limits attributing the A7 gain to temporal information; it does not prove that all improvements are solely parameter-count effects or that history contains no information. Cross-base interactions are exploratory and uncorrected; see [cross_base](cross_base/cross_base_interaction.csv).

## Provenance

The [original handover](HANDOVER_RQ3_ABLV32_20261108.md) is preserved as historical evidence, not silently rewritten. Its statements “none raises AP” and “history adds noise” should be read alongside the more limited statistical interpretation above: AP point estimates often increase, while their CIs include zero; the mechanism of noise is not established.

H7 B0 reproduces the RQ2 H7 scalar row. V6 re-scoring has small documented differences from the original RQ1 scores; its own frozen B0 is the comparison baseline. H7’s final report step needed an in-memory `None`→`nan` formatting patch after predictions had been written. No training or prediction score was changed by the local release work.

`evidence/` preserves executed run paths; large step logs use lossless gzip. Dense curves, paired-commit caches, weights and feature caches remain outside Git in the original archives. The selected adapter NPZ weights are external. Retraining from the original package is distinct from numerical replay of stored results.
