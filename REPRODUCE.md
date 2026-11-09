# Reproduction routes

## 1. Numerical replay without a GPU

From the repository root, run:

```bash
python tools/recompute_tables.py
python tools/verify_release.py
```

The first script uses the standard library. It joins saved RQ1 scalar predictions to the pinned test labels/groups, handles tied scores, computes average precision and ROC areas, and regenerates seed means/sample SDs. It also checks all recorded RQ3 contrast point estimates against the per-seed metric table. It does not recompute dense-curve TTA. `python tools/verify_bootstrap_draws.py` (NumPy required) reconstructs all66 within-base and18 interaction confidence intervals/p-values from the original stored resample arrays and verifies Holm; it does not generate new resamples.
The second script checks the release file checksums and reports repository size and unexpected model/data artifacts. It does not validate scientific hypotheses.

## 2. Restore deduplicated upstream source ZIPs

```bash
python tools/restore_shared_dependencies.py
```

This copies the two checksum-matched canonical upstream source ZIPs into the original RQ2 package paths. They are ignored by Git. The archived launcher code itself is not rewritten. Before doing GPU work, use a new output directory and read the original package README/contract and all deviations.

## 3. GPU retraining / full capture

Training was performed on Vast.ai NVIDIA RTX4080-class instances; exact hardware and runtime records are preserved per run in `environment.json`, `runtime_identity.json`, `nvidia_smi*.txt` and the archived logs. CPU code decodes/indexes data and orchestrates runs; CUDA runs the neural network. This local release audit did not rerun GPU training.

Original recorded environment: managed Python3.8.5, PyTorch2.1.0+cu118, torchvision0.16.0, CUDA11.8, mmengine0.10.7, mmcv2.2.0 and author-source mmaction2. Read the actual run contract rather than assuming all systems use one optimization recipe.

- RQ1: each system retains its package’s executed method-specific training recipe; seeds42/43/44, with V6 seed44 replaced only by the verified rerun.
- RQ2: end-to-end encoder+bounded LSTM2048, 50 epochs, effective batch16, SGD0.002, milestones20/40, validation full-mAUC selection. H7/H13 require the documented strict-FP32 evaluation recovery rather than silently bypassing parity checks.
- RQ3 V3.2: frozen H7/V6 bases; 6 adapter arms×3seeds per base, 50epochs, AdamW2e−4, weight decay1e−4, width512, dropout0.1, bound1. The original analysis plan records the initialization policy, checkpoint selection and bootstrap.

The original shell launchers use historical `/workspace` paths; inspect their arguments and construct an isolated workspace first. We did not run scripts from result archives. Download Nexar separately using an appropriately scoped personal token entered locally; never commit it or append it to a command saved in this repository.

Large original packages/checkpoints are linked in [ARTIFACTS.md](ARTIFACTS.md). Restoring code dependencies does not restore model weights, CUDA libraries or private local paths automatically. The replay helpers do not start paid jobs, upload artifacts or change sharing. Git publication is performed separately with the user's authorization.
