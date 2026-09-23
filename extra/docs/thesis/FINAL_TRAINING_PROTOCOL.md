# GLO-NCA V3 — Final Training Protocol

The exact experiment to be run for the thesis. **Config of record:**
`configs/v3_multilevel_ckpt.yaml`. Every value below is verified against that
config and the code (`src/experiment/runner.py`, `src/models/Model_GLO_NCA_V3.py`).

## Model
| Item | Value |
|---|---|
| Model | GLO-NCA V3 (`GLO_NCA_V3_MultiLevel`) |
| Parameters | **40,656** (measured) |
| Levels (resolution) | Level 1 = 32³, Level 2 = 96³, Level 3 = 128³ |
| Per-level channels | 24 / 24 / 16 |
| Per-level NCA steps | 20 / 20 / 10 |
| SE channel context | enabled (`use_attention: true`) |
| Spatial global context | enabled (`use_spatial: true`) |
| Fusion | learnable concatenation (`feature_fusion.type: concat`) |
| Fire rate | 0.6 · Hidden 128 · Dropout 0.1 |
| Gradient checkpointing | **ON** (memory-only; identical math/outputs) |

## Data
| Item | Value |
|---|---|
| Dataset | BraTS-MET 2025 (Training), 1,296 cases / 810 subjects |
| Modalities | t1n, t1c, t2w, t2f (T1, T1ce, T2, FLAIR) |
| Output | WT / TC / ET (nested, multi-label sigmoid) |
| Split file | `split/master_split.json` (subject-disjoint, frozen) |
| Split SHA256 | `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d` |
| Split counts | train 898 / val 200 / test 198 (subjects 567 / 121 / 122) |

## Training
| Item | Value |
|---|---|
| Epochs | **300** |
| Batch size | 1 |
| Seed | 42 |
| Augmentation | light |
| Loss | Focal-Tversky + BCE (`FocalTverskyCELoss`, β=0.75, γ=1.33, ce_weight=0.5) |
| Optimizer | AdamW (lr 1.6e-3, weight_decay 1e-4) |
| Scheduler | Cosine LR (η_min 1e-5, over the full run) |
| EMA | enabled (decay 0.999) |
| Gradient clipping | enabled (max_norm 1.0) |
| Checkpoint frequency | every 10 epochs (+ best + last) |

## Evaluation
- Metrics: **Dice, mIoU, HD95**. **HD95 in voxels** on the resampled grid (no mm).
- **Thresholds tuned on validation only** (3-epoch smoothed best-epoch selection).
- **Test set frozen**, evaluated **once** at the tuned thresholds. No test leakage.
- Single clean inference — **no ensemble, no TTA**.

## Reproducibility artifacts recorded per run
git commit, config identity, dataset identity (case_count 1296, subject_count
810, patient_id_hash), split SHA256, seed, environment (GPU/CUDA/PyTorch),
parameter count, full checkpoint state (model/optimizer/scheduler/EMA/epoch/best/
RNG), metrics CSVs, graphs, and the experiment manifest.

## Hardware & how to launch
Validated on **NVIDIA L4 24 GB** (128³ TRUE FIT with checkpointing, ~9 GB peak).
```bash
# on the GPU node, dataset present and DATA_ROOT set:
python scripts/gpu_memory_gate_v3.py --config configs/v3_multilevel_ckpt.yaml   # expect 96³/128³ TRUE FIT
./cloud/scripts/pretrain_gate.sh configs/v3_multilevel_ckpt.yaml                 # expect PASS/READY
./cloud/scripts/run_training.sh configs/v3_multilevel_ckpt.yaml                  # 300-epoch final run
```

## Preconditions before the final run (must all hold)
1. Real BraTS-MET dataset present on the training node; `check_split.py` confirms
   the split covers it and the SHA256 matches.
2. `gpu_memory_gate_v3.py` reports 96³ **and** 128³ TRUE FIT on the chosen GPU.
3. `pretrain_gate.sh` reports `FINAL PRE-TRAINING GATE: PASS / GLO-NCA TRAINING: READY`.
4. The 3-epoch operational smoke test (`configs/v3_smoke_3epoch.yaml`) has passed.

Only when all four hold is the 300-epoch run authorised. Any deviation from the
values above is a methodology change and must be explicitly approved.
