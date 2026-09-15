# GLO-NCA: Old Kaggle vs V3 — Exact Workload Comparison

**Date:** 2026-09-15 · No GCP compute · Methodology frozen · No code changed.

Every number below is labelled **FACT** (read from the actual old Kaggle source),
**CALCULATED** (arithmetic from those facts), **MEASURED** (observed on real
hardware), or **HYPOTHESIS** (unconfirmed). The earlier audit's "64³×64 steps"
guess is **superseded** by the real configs and must not be used.

## Source of the old numbers (FACT)
The actual old Kaggle experiments are archived at
`archive/kaggle_experiment_history/kaggle_v4.py … kaggle_v7.py`. v5/v6/v7 share
the proven recipe. Extracted directly from the code:

| Setting | Value | Source (FACT) |
|---|---|---|
| Levels | **2** | `INPUT_SIZE = [[32,32,24],[64,64,48]]` (2 entries) |
| Level resolutions | **32×32×24** and **64×64×48** (not cubes) | same line |
| NCA steps | **[20, 20]** = 40 total | `STEPS = [20, 20]` |
| Training input | **1 random 64×64×48 patch per case** | `patchify_multimodal` — "Random 3D patch" |
| Channels / hidden | 24 / 128 | `CHANNEL_N=24`, `HIDDEN=128` |
| Batch / workers | 1 / 4 | `BATCH_SIZE=1`, `NUM_WORKERS=4` |
| Epochs | **150** | `EPOCHS = 150` |
| Cases | **882** (v6/v7 era) | v6/v7 headers "882 cases"; `N_PATIENTS=None` = all |
| Gradient checkpointing | **OFF** (did not exist) | absent from all kaggle_v* |
| Foreground crop + nonzero z-norm | yes | `_foreground_bbox`, nonzero norm |
| Validation | every epoch, full-image | `evaluate(...)` per epoch |
| Loss / opt / sched | Focal-Tversky+BCE / AdamW / cosine | matches V3 |

**Key fact:** the old model trained on a **single 64³-ish patch per case**, never
the full volume. This is the dominant reason it was fast.

## Current V3 (FACT from configs + model)
| Setting | Value | Source |
|---|---|---|
| Levels | **3** | `v3_multilevel*.yaml` level1/2/3 |
| Level resolutions | **32³, 96³, 128³** (cubes) | config |
| NCA steps | **[20, 20, 10]** = 50 total | config |
| Training input | **full 128³ volume** (no patchify) | `Agent_GLO_NCA_V3` processes whole volume |
| Cases | **896** (culling smoke) / 898 (canonical) | split |
| Gradient checkpointing | **ON** on L4 (2× backward) | `memory.gradient_checkpointing=true` |

## Workload math (CALCULATED from the facts above)
Voxel-steps = spatial voxels × NCA steps (the honest measure of NCA compute).

### Old Kaggle (per case)
| Level | Voxels | Steps | Voxel-steps |
|---|---|---|---|
| L1 32×32×24 | 24,576 | 20 | 491,520 |
| L2 64×64×48 (patch) | 196,608 | 20 | 3,932,160 |
| **Total/case** | | | **4,423,680** |

Per epoch (882 cases): **3.90 billion** voxel-steps. Checkpointing OFF (×1).

### Current V3 (per case)
| Level | Voxels | Steps | Voxel-steps |
|---|---|---|---|
| L1 32³ | 32,768 | 20 | 655,360 |
| L2 96³ | 884,736 | 20 | 17,694,720 |
| L3 128³ (full) | 2,097,152 | 10 | 20,971,520 |
| **Total/case** | | | **39,321,600** |

Per epoch (896 cases): **35.2 billion** (no ckpt) / **70.5 billion** (ckpt ×2).

### Ratios (CALCULATED)
| Comparison | Ratio |
|---|---|
| Per-case, V3 full-volume vs old **patch** | **8.9×** |
| Per-case, plus checkpointing 2× | **17.8×** |
| **Per-epoch, V3 (ckpt) vs old** | **~18×** |

## Why the old Kaggle run was fast (deliverable #12 — proven, not asserted)
Ranked by contribution:
1. **Patch vs full volume (biggest):** old trained on ONE 64³ patch/case; V3 trains
   the whole 128³ volume. ~5.3× more voxels at the fine level alone, and V3 adds a
   full 96³ regional level the old run never had.
2. **3 levels vs 2:** V3 adds the 96³ regional level (17.7 M voxel-steps/case — by
   itself larger than the *entire* old per-case workload of 4.4 M).
3. **Gradient checkpointing (2×):** old had none; V3 needs it to fit the L4's
   24 GB, doubling backward compute. (Removable on H100 — runtime only.)
4. **50 vs 40 NCA steps:** 1.25×.
5. **896 vs 882 cases + 300 vs 150 epochs:** more total work, but per-epoch time is
   the like-for-like number above.

**Conclusion (FACT-based):** the old run was fast because it was a *fundamentally
lighter* pipeline (2-level, single 64³ patch, no checkpointing) — **not** because
V3 has a bug. V3 legitimately does ~18× more work per epoch. On top of that real
cost sit two *fixable* factors: checkpointing (2×, removable on H100) and the
cache-across-epochs defeat (repeats preprocessing each epoch).

## The old run had NO patchify-equivalent bug in V3 — is that a methodology gap?
The old run's speed came partly from **patch-based training** (1 patch/case). V3
deliberately does **full-volume** training (its multi-level design is meant to see
the whole volume). Switching V3 to patch training would be a **METHODOLOGY CHANGE**
— **forbidden without approval**. It is listed here only to explain the speed gap,
**not** proposed.

## What remains to MEASURE on Kaggle H100 (not assumed)
- checkpoint ON vs OFF wall-time and peak VRAM at 96³ & 128³ (does V3 fit on H100
  without checkpointing → removes the 2×?);
- real per-stage timing (load/preproc/H2D/fwd/bwd/opt/EMA) via the profiler on
  REAL data;
- cache-across-epochs behaviour with workers 0/2/4 (confirm the repeat-preprocess
  finding);
- validation-vs-train split of epoch time.

## Correction to the earlier audit
`GLO_NCA_V3_PERFORMANCE_AUDIT.md` estimated the old run as "64³×64 steps,
single-level, ~21× ratio." The **real** old run is **2-level, 32×32×24 + 64×64×48
patch, 40 steps, 882 cases, 150 epochs**, giving a **~18×** per-epoch ratio. Use
**these** numbers.
