# GLO-NCA V3 — Architecture

This document describes the **actual implemented** architecture in
`src/models/Model_GLO_NCA_V3.py` (class `GLO_NCA_V3_MultiLevel`), which reuses the
V2 cell `src/models/Model_BasicNCA3D.py` (class `BasicNCA3D`). Measured parameter
count: **40,656**.

## Overview
GLO-NCA V3 is a **single unified model** (one forward path, one segmentation head
— *not* an ensemble or averaging) with **three nested NCA levels** that refine a
shared region from global context to fine boundaries, connected by **learnable
projections** and a **learnable fusion** head.

```
                 Multi-modal MRI (B, X, Y, Z, 4)   [T1, T1ce, T2, FLAIR]
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Level 1 — Global         │  resolution 32³, channels 24
                 │ BasicNCA3D + SE + spatial│  NCA steps 20, kernel 7
                 └────────────┬─────────────┘
                              │  learnable 1×1×1 projection  +  upsample
                              ▼
                 ┌─────────────────────────┐
                 │ Level 2 — Regional       │  resolution 96³, channels 24
                 │ BasicNCA3D + SE + spatial│  NCA steps 20, kernel 3
                 └────────────┬─────────────┘
                              │  learnable 1×1×1 projection  +  upsample
                              ▼
                 ┌─────────────────────────┐
                 │ Level 3 — Fine           │  resolution 128³, channels 16
                 │ BasicNCA3D + SE + spatial│  NCA steps 10, kernel 3
                 └────────────┬─────────────┘
                              │
                              ▼
                 Learnable multi-level fusion
                 (each level's state → 1×1×1 project to fine width → resize to
                  128³ → concatenate → 1×1×1 fuse conv)
                              │
                              ▼
                 Segmentation head (1×1×1 conv → 3 channels)
                              │
                              ▼
                        WT / TC / ET     (multi-label sigmoid)
```

## The NCA cell (`BasicNCA3D`, shared with V2)
Channels-last tensors `(B, X, Y, Z, C)`. One update step per cell:
1. **Perceive** — a depthwise 3×3×3 (or 7×7×7) conv over the state, concatenated
   with the cell identity.
2. **Global context** — SE channel attention (`use_attention`) reweights channels
   using whole-volume statistics; a spatial global-context block (`use_spatial`)
   modulates by whole-volume spatial context. Both are the thesis novelty and add
   negligible parameters.
3. **Update MLP** — `fc0` (→ hidden 128) → BatchNorm → ReLU → optional dropout →
   `fc1` (→ channels).
4. **Stochastic fire-rate** — a random per-cell mask (`fire_rate` 0.6) gates the
   residual update `x ← x + dx`.
The step is repeated `nca_steps` times; the first `input_channels` (=4) channels
hold the modalities and are preserved across steps (seed convention).

## Cross-level information flow
- Each level builds a **seed**: the modalities (resized to that level's
  resolution) in the first 4 channels, zeros elsewhere.
- The previous level's **full state** is passed through a **learnable 1×1×1
  projection** (`FeatureProjection`) to the next level's state-channel width,
  upsampled to the next resolution, and **added into the next seed's state
  channels**. This is the nested coarse-to-fine cascade.

## Learnable fusion (the V3 contribution)
After all levels run, each level's final state is projected to the finest width
(`level_to_fine`), resized to 128³, **concatenated**, and fused by a 1×1×1 conv
(`fuse`). A final 1×1×1 `seg_head` produces 3 logits → sigmoid → WT/TC/ET. The
fusion is **learned**, not an average or vote.

## Parameter breakdown (measured)
| Component | Parameters |
|---|---|
| Level 1 NCA | 18,861 |
| Level 2 NCA | 11,277 |
| Level 3 NCA | 7,811 |
| Cross-level projections | 800 |
| Fusion + segmentation head | 1,907 |
| **Total** | **40,656** |

## Configuration knobs (in `configs/v3_multilevel_ckpt.yaml`)
`model.level{1,2,3}.{enabled, resolution, channels, nca_steps, kernel_size}`,
`model.use_attention`, `model.use_spatial`, `model.feature_fusion.type`
(`concat` | `add`), `model.fire_rate`, `model.hidden`, `model.dropout`, and the
memory-only `memory.gradient_checkpointing` (default OFF). Levels can be toggled
via `enabled` (used by the ablations `v3_ablation_A/B/C.yaml`); the runner adapts
the level count, projections and fusion automatically.

## What is NOT part of the architecture
- **Gradient checkpointing** — memory-only; identical math/outputs (see the
  memory-optimization section of the README and
  `reports/validation/V3_GRADIENT_CHECKPOINTING_REPORT.md`).
- **No ensemble, no test-time augmentation.**

## Relationship to V2 (baseline)
V2 (`configs/gcp_full.yaml`, 30,138 params) is the two-level coarse-to-fine
baseline and is **frozen**. V3 reuses the same `BasicNCA3D` cell unchanged and
adds the third level, the learnable cross-level projections, and the learnable
fusion head. Both share the dataset, master split, seed, loss and evaluation
protocol so V2↔V3 is a controlled comparison.
