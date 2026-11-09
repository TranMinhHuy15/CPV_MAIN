# RQ2 — bounded causal LSTM context length

Current study arms: H7, H13 and H26. Historical H1/H3/H5 variants are not current paper arms.
Each snippet contains native frames n−12, n−9, n−6, n−3, n. Adjacent endpoint indices differ by 3 native frames. At nominal 30 FPS, total spans are 1.0 s (H7), 1.6 s (H13), and 2.9 s (H26). These are nominal spans, not timestamp-verified uniform clocks.

The encoder outputs a 2048-dimensional vector per snippet. A single-layer unidirectional 2048-hidden LSTM is reset for each bounded prediction window. H is the number of vectors/LSTM steps, not the number of images inside one snippet. The final hidden state feeds the classifier.

Recorded training setup: 25 warm-up + 30 target snippets; loss on the 30 targets, BCE + 1.5 FFR + 1.1 AMC; microbatch 2 with accumulation 8 (effective 16), SGD lr 0.002, momentum 0.9, weight decay 1e−4, milestones 20/40, gamma 0.1, 50 epochs, three independently initialized seeds. Checkpoints are selected by validation full mAUC. H7/H13 choose epochs 10/5/8; check H26’s selected checkpoint in each run report.

H26 came from an earlier execution campaign. “Earlier campaign” does not mean an established batch-size difference: the inspected contracts record effective batch 16. Differences in capture precision and runtime recovery are documented rather than assumed absent.

- [All 28 metrics, mean and sample SD](summary/RQ2_table.csv).
- [Per-seed report and selected checkpoint](summary/RQ2_per_seed.csv).
- [H7 provenance](h7/README.md), [H13 provenance](h13/README.md), [H26 provenance](h26/README.md).
