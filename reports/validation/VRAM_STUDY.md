# GLO-NCA V3 — VRAM Requirement Study (measurement-only)

**Date:** 2026-09-14 · GPU used for measurement: **NVIDIA L4 (23.66 GB usable)**
**Model:** EXISTING production `src/models/Model_GLO_NCA_V3.py` + `configs/v3_multilevel.yaml`,
UNCHANGED (channels/NCA steps/resolutions/loss/optimizer as-is). No gradient
checkpointing. Batch 1. Fresh process per (resolution, mode) so an OOM never
contaminates the next. Synthetic input tensors of the correct shape (dataset not
needed to measure memory). Per-level ratio = production: level1=res/4, level2=3·res/4, level3=res.

## Measured (real, on the L4)
| L3 res | voxels | model state | forward-only peak | forward+backward peak | fit? |
|---|---|---|---|---|---|
| 32³  | 32,768    | ~0 GB | 0.071 GB | **1.575 GB** | ✅ TRUE FIT |
| 64³  | 262,144   | ~0 GB | 0.494 GB | **10.93 GB** | ✅ TRUE FIT |
| 96³  | 884,736   | ~0 GB | 1.647 GB | **OOM** (≥22.47 GB before OOM) | ❌ |
| 128³ | 2,097,152 | ~0 GB | 3.888 GB | **OOM** (≥22.58 GB before OOM) | ❌ |

Notes:
- **Model state ≈ 0 GB** — 40,656 params is negligible; memory is 100% activations.
- **Forward-only is cheap** even at 128³ (3.9 GB) → inference fits easily; the binding
  constraint is the **backward (autograd) graph**, driven by NCA step-unrolling
  (20 steps L1/L2, 10 steps L3) at high resolution with a 128-wide hidden volume per step.
- Backward peak scales ~**linearly with voxel count**: 32³→64³ is ×8 voxels and
  1.575→10.93 GB (×6.9). Per-voxel backward ≈ 42 KB (64³ slope) to 48 KB (32³ slope).

## Estimates for the OOM resolutions (labelled ESTIMATE)
Extrapolating fwd+bwd ∝ voxel count from the two TRUE-FIT points:
- **96³:** 10.93 × (96³/64³) = 10.93 × 3.375 ≈ **~37 GB** (consistent with the >22.5 GB pre-OOM partial). **ESTIMATE.**
- **128³:** 10.93 × (128³/64³) = 10.93 × 8 ≈ **~87 GB** (64³ slope); ~101 GB (32³ slope). **ESTIMATE: ~87–101 GB.**

These are for the raw autograd graph of one forward+backward+optimizer step at batch 1,
matching the production training step. Real training adds optimizer state (AdamW ≈ 2×
params — negligible here), EMA (one param-set copy — negligible), and framework/fragmentation
overhead, so use conservative headroom on top.

## Requirement with headroom
- **96³ alone:** ~37 GB measured-consistent → needs a **40 GB** GPU minimum; comfortable on 48–80 GB.
- **128³ (the production fine level):** ~87–101 GB estimated → **no single 80 GB GPU is safe**;
  needs **≥ ~96–128 GB**, i.e. an **H100/H200 94–141 GB**, or multi-GPU activation sharding.

## Conclusions
```
MEASURED 96³ REQUIREMENT:        > 22.5 GB (OOM on 24 GB L4); est. full ~37 GB (fwd+bwd, batch 1)
ESTIMATED 128³ REQUIREMENT:      ~87–101 GB (extrapolated from 32³ & 64³; ESTIMATE, not measured)
MINIMUM PRACTICAL VRAM (128³):   ~110–128 GB with safety headroom
RECOMMENDED GPU (128³):          NVIDIA H100 80GB is NOT sufficient on its own;
                                 use H200 141GB, OR 2× H100/A100-80GB with activation
                                 sharding, OR reduce the autograd footprint
                                 (e.g. gradient checkpointing across NCA steps —
                                 a memory-only change, requires your approval as it
                                 edits the model file).
RECOMMENDED GPU (96³ only):      A100 40GB (fits ~37 GB) or A100/H100 80GB (comfortable)
EXPECTED VRAM HEADROOM:          128³ on 141GB H200 ≈ 30–50 GB free (safe);
                                 96³ on 80GB A100 ≈ 40+ GB free (very safe)
CONFIDENCE:                      HIGH for 32³/64³/96³/128³ forward (measured) and 96³ direction;
                                 MEDIUM for the 128³ backward absolute number (linear
                                 extrapolation over an 8× voxel jump; real peak could be
                                 ±20%). A100 80GB is explicitly NOT recommended for 128³.
```

## Important caveat (honest)
**A100 80 GB is NOT sufficient for the unchanged V3 at 128³** on this evidence
(~87–101 GB estimated). Do not provision an 80 GB GPU expecting 128³ to fit. The
realistic paths for the unchanged architecture at 128³ are a **larger-memory GPU
(H200 141 GB)**, **multi-GPU**, or an **explicitly-approved memory-only optimization**
(gradient checkpointing) that does not change the model's math/outputs.
