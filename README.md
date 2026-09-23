# GLO-NCA: Global Context-Aware Neural Cellular Automata for Brain Tumor Segmentation

> **MS Thesis — Muhammad Waqas, Air University, Islamabad.**
> *For academic and educational use only — not a medical device (see [LICENSE](LICENSE)).*

GLO-NCA is a **lightweight, global-context-aware Neural Cellular Automata (NCA)**
model for **multi-modal 3D brain-tumor segmentation** on BraTS-METS. A plain NCA
cell only sees its local neighbourhood; GLO-NCA adds **Squeeze-and-Excitation
(SE) channel context** and a **spatial global-context** block, applied at every
NCA step, so each cell's update is informed by the whole volume. The production
model has **30,209 inference parameters**.

---

## Status
```
Production config:     configs/glo_nca_production.yaml   (the only config)
Parameters:            30,209 inference / 75 auxiliary / 30,284 training
Split:                 898 train / 200 val / 198 test, subject-disjoint, frozen
GCP pipeline:          validated on an NVIDIA L4 (10-epoch engineering run)
Final 300-epoch run:   NOT STARTED
```
No test-set segmentation result is claimed. The 10-epoch GCP run was an
engineering validation of the pipeline, not a thesis result.

## Production architecture

Every value below is verified from the constructed model, not only from the
config text.

| Component | Value |
|---|---|
| Working volume | 128³ |
| Level 1 (global) | 48³ · 24 channels · 15 NCA steps · kernel 5 |
| Level 2 (fine) | 64³ · 24 channels · 15 NCA steps · kernel 5 |
| Level 3 | absent |
| SE channel attention | ON, at every NCA step |
| Spatial global context | ON, kernel 7, at every NCA step |
| Fusion | learned `Conv3d(48 → 24)` |
| Deep supervision | ON, weight 0.4 (training only) |
| Loss | Focal Tversky (α 0.40, β 0.60, γ 1.33) + CE 0.5, empty-region BCE 0.1 |
| Precision | BF16 forward, FP32 loss |
| Optimiser | AdamW, LR 0.0016 → 1e-5 cosine, weight decay 1e-4 |
| Warmup | 3 epochs (2,694 of 269,400 steps) |
| Sampling | small-lesion-aware (boost 2.0, ET ≤ 100 resampled voxels), training only |
| EMA / gradient clip | 0.999 / 1.0 |
| Batch / seed | 1 / 42 |
| Budget | 300 epochs max, early stopping patience 15, min_delta 0.01 |
| Post-processing | min component voxels WT 50 · TC 5 · ET 0 |

Model code: `src/models/Model_GLO_NCA_GlobalContext.py` (production model),
`src/models/Model_GLO_NCA_V3.py` (multi-level base, fusion, deep supervision),
`src/models/Model_GLO_NCA_Cell.py` (NCA cell, SE block, spatial GC block).

## Dataset
**BraTS-MET 2025 (MICCAI-LH BraTS-MET Challenge, Training set).**
- **1,296 cases** across **810 subjects**: 650 top-level + 646 in the nested
  `UCSD - Training/` cohort. Case discovery is recursive.
- **Subject-disjoint** split: **898 / 200 / 198 cases** (567 / 121 / 122 subjects).
- Split fingerprint (SHA256):
  `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d`.
- The test split is never used for training, thresholds, checkpoint selection
  or early stopping.

## Quick start

```bash
# install (Python 3.10+, CUDA 12.1)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-docker.txt

# verify the production config before a long run (fails closed on a wrong config)
python scripts/verify_glo_nca_production_config.py configs/glo_nca_production.yaml

# verify the frozen split against the dataset
python scripts/check_split.py --split split/master_split.json --data-root /path/to/BraTS-MET

# train
DATA_ROOT=/path/to/BraTS-MET python train.py --config configs/glo_nca_production.yaml

# resume
python train.py --resume experiments/<experiment-id>
```

On GCP, use `cloud/scripts/` — `setup_gcp.sh`, `pretrain_gate.sh`,
`run_training.sh`, `resume_training.sh`. Training runs in Docker (see
`Dockerfile`); the entrypoint sets `--shm-size=8g`, which DataLoader workers
require.

## Repository layout
```
train.py                  production entry point
configs/                  glo_nca_production.yaml (the production config)
split/                    frozen subject-disjoint split + data-quality policy
src/                      model, agents, datasets, losses, experiment runner
cloud/                    GCP setup, training, sync, resume, systemd unit
scripts/                  production gate scripts used by cloud/scripts/
```
This repository contains production code only. Tests, audits, reports, docs,
historical configs and optional ops tools are kept in a local `extra/` folder
that is git-ignored and excluded from the Docker image. They remain available
in the git history of earlier commits.

## Limitations
- No final segmentation accuracy yet: the 300-epoch run has not been executed.
- No ablation run exists, so the individual contribution of the spatial
  global-context block, warmup, or small-lesion sampling is not measured.
- HD95 and lesion sizes are in voxels on the resampled 128³ grid, not mm.
- GLO-NCA targets parameter and compute efficiency; it is not claimed to beat
  large U-Net or Transformer models on absolute accuracy.

## Acknowledgements & references
Built on the open-source **Med-NCA / M3D-NCA** framework by John Kalkhof et al.
(MIT-licensed). The global-context design, the multi-level architecture, the
BraTS multi-modal/multi-label pipeline and the evaluation are the thesis
contributions.
- Kalkhof et al., *Med-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata*, IPMI 2023.
- Kalkhof & Mukhopadhyay, *M3D-NCA: Robust 3D Segmentation with Built-In Quality Control*, MICCAI 2023.

## License
MIT, for **academic/educational use only** — see [LICENSE](LICENSE). Not for clinical use.
