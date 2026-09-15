# GLO-NCA — Project Overview (MS Thesis)

**Title:** GLO-NCA: Global Context-Aware Neural Cellular Automata for Brain Tumor Segmentation
**Author:** Muhammad Waqas · **Institution:** Air University, Islamabad · **Degree:** MS Thesis

## Research problem
Volumetric brain-tumor segmentation on multi-modal MRI (BraTS) is dominated by
large CNN/Transformer models (U-Net, nnU-Net, Swin-UNETR, UNETR) with millions of
parameters and substantial compute/memory needs. This limits deployment on
low-resource hardware and increases training cost. Neural Cellular Automata (NCA)
offer an extremely small, iterative alternative, but a plain NCA only propagates
information locally and struggles with the global reasoning tumor segmentation
requires.

## Motivation
If a tiny NCA can be given cheap **global context** and a **hierarchical
multi-scale** refinement, it may deliver useful multi-region tumor segmentation
at a tiny fraction of the parameters of mainstream models — attractive for
resource-constrained and reproducible research settings.

## Research objective
Design and evaluate **GLO-NCA**, a global-context-aware NCA for multi-modal 3D
BraTS segmentation, and quantify the contribution of the global-context
mechanism and the multi-level design against a controlled baseline.

## Architecture (summary)
A single unified **GLO-NCA V3** model with three nested NCA levels —
global (32³) → regional (96³) → fine (128³) — connected by learnable projections
and a learnable concatenation-fusion head, producing nested WT/TC/ET regions.
Each level is the same lightweight NCA cell augmented with SE channel context and
a spatial global-context block. **40,656 parameters total.** Full detail:
[`../architecture/GLO_NCA_V3_ARCHITECTURE.md`](../architecture/GLO_NCA_V3_ARCHITECTURE.md).

## Novelty
1. A **global-context-aware NCA cell** (SE + spatial global context) that adds
   whole-volume reasoning at negligible parameter cost.
2. **GLO-NCA V3**: a unified multi-level NCA with learnable cross-level
   projections and learnable fusion (not an ensemble).
3. A reproducible, thesis-grade protocol built around a **subject-disjoint**
   split and strict frozen-test discipline.

## Dataset
BraTS-MET 2025 (MICCAI-LH Challenge, Training): **1,296 cases / 810 subjects**,
four modalities (t1n/t1c/t2w/t2f) + segmentation. Split subject-disjointly into
898/200/198 cases. See [`../../split/README.md`](../../split/README.md).

## Preprocessing
Foreground crop, resample to the level resolution, per-modality intensity
normalization, on-the-fly light augmentation (flips / 90° rotations / mild
intensity jitter) at training time. Labels are mapped to nested WT/TC/ET regions.
(Methodology is inherited unchanged from the validated V2 pipeline.)

## Training
Final: **300 epochs**, batch 1, seed 42, AdamW + cosine LR, EMA, gradient
clipping, Focal-Tversky + BCE (β=0.75, γ=1.33). Opt-in gradient checkpointing
makes the 128³ step fit a 24 GB GPU without changing the model.
See [`FINAL_TRAINING_PROTOCOL.md`](FINAL_TRAINING_PROTOCOL.md).

## Evaluation
Per-region Dice, mIoU, HD95 (voxels on the resampled grid). Thresholds tuned on
validation only; test set frozen and evaluated once; single clean inference.

## Reproducibility
Fixed master split with a committed SHA256 fingerprint; seeded runs; full
checkpoint/resume (model/optimizer/scheduler/EMA/epoch/RNG); config + dataset +
split identity recorded in every experiment manifest.

## Computational efficiency
40,656 parameters; 128³ training step fits ~9 GB VRAM with checkpointing
(measured on NVIDIA L4). Orders of magnitude smaller than mainstream 3D
segmentation networks.

## Limitations
No final accuracy is available until the 300-epoch run completes; HD95 is in
voxels, not mm; efficiency is the goal rather than beating large models on
absolute Dice; real-data training needs a ≥~10 GB GPU.

## Current experimental status
Software + memory + real-data software-path validation complete; 128³ TRUE FIT on
L4 with checkpointing confirmed. **3-epoch operational smoke test pending dataset
transfer to the training node; final 300-epoch training not started.**
