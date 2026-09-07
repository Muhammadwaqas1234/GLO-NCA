# GLO-NCA — Complete Code Guide

**Global Context-Aware Neural Cellular Automata for Brain Tumor Segmentation**
MS Thesis — Muhammad Waqas, Air University, Islamabad.

This document explains the entire codebase from top to bottom: what every file
does, how data flows through the model, what each hyperparameter means, and the
research decisions behind the design. Read it start-to-finish to understand the
whole project.

---

## Table of Contents

1. [What the project is](#1-what-the-project-is)
2. [The big idea (NCA + global context)](#2-the-big-idea)
3. [How data flows end-to-end](#3-how-data-flows-end-to-end)
4. [Repository map](#4-repository-map)
5. [File-by-file walkthrough](#5-file-by-file-walkthrough)
6. [Every hyperparameter explained](#6-every-hyperparameter-explained)
7. [The loss functions](#7-the-loss-functions)
8. [Training loop & the v2 fixes](#8-training-loop--the-v2-fixes)
9. [Evaluation, thresholds & the diagnostic report](#9-evaluation)
10. [The metrics: Dice, mIoU, HD95](#10-the-metrics)
11. [Hardware & VRAM guide](#11-hardware--vram-guide)
12. [How to run (Kaggle, laptop, GCP/Docker)](#12-how-to-run)
13. [Troubleshooting / FAQ](#13-troubleshooting--faq)
14. [Glossary](#14-glossary)
15. [Version history (v4 → v7 → v2)](#15-version-history)
16. [Results & honest positioning](#16-results--honest-positioning)

---

## 1. What the project is

GLO-NCA segments brain tumors from **multi-modal MRI (BraTS)**. Given a patient's
four MRI scans it predicts three tumor regions:

| Input (4 channels) | Output (3 regions) |
|---|---|
| T1, T1ce (T1c), T2, FLAIR | **WT** Whole Tumor, **TC** Tumor Core, **ET** Enhancing Tumor |

The regions are **nested**: ET ⊆ TC ⊆ WT. They are treated as three independent
binary channels (multi-label), each with its own sigmoid — not a single softmax —
because they overlap.

The headline property: the model has **~30,000 parameters**, roughly **1000×
smaller** than U-Net (~30M) or transformers like Swin UNETR (~62M), yet reaches
comparable Dice. It fits a 6 GB laptop GPU. That efficiency is the thesis
contribution.

---

## 2. The big idea

### Neural Cellular Automata (NCA)
An NCA is a tiny neural network (an "update rule") applied to **every voxel**,
repeatedly, for several steps. Each step, a voxel looks at its local
neighbourhood (a 3D conv), computes a small update, and changes its own state.
After many steps the local rule produces a globally coherent segmentation —
like cells in tissue self-organising. Because the *same* small rule is reused at
every voxel and every step, the parameter count stays tiny.

**The limitation NCA has:** a plain NCA only ever sees a *local* neighbourhood.
It struggles to know *where in the whole brain* the tumor is.

### The GLO-NCA novelty: global context
GLO-NCA adds two cheap "global context" blocks so each voxel's update is also
informed by the whole volume:

- **SE block (channel attention)** — squeezes the whole volume into one number
  per channel, then re-weights channels. Answers *"which modality/feature
  matters globally?"*
- **Spatial GC block (spatial attention)** — pools across channels to a per-voxel
  attention map. Answers *"which regions of the volume matter?"*

Both add only a few hundred parameters — that is the "global context, low params"
trade-off the thesis claims.

### Coarse-to-fine cascade
To get global context on a small VRAM budget, two NCA levels are stacked:

1. **Low-res level** — runs on the whole (downscaled) volume → captures global
   tumor location.
2. **High-res level** — takes the upscaled low-res features + a high-res *patch*
   → refines fine boundaries.

Training uses a random patch at the high-res level (cheap); inference runs the
full volume.

---

## 3. How data flows end-to-end

```
patient folder (4 .nii modalities + seg)
      │
      ▼  load + stack → (X, Y, Z, 4)
[foreground crop]      remove black background (Swin-UNETR style)
      │
      ▼  resize to training size (LINEAR for image, NEAREST for label)
[cache]                stored once so later epochs are fast
      │
      ├─ (train only) random 3D PATCH, biased toward ET
      ├─ (train only) AUGMENT (light or heavy)
      ▼  per-channel NONZERO z-normalisation (brain voxels only)
[seed]                 place 4 modalities into the first 4 NCA state channels
      │
      ▼
[NCA level 0 (low-res, kernel 7)]  → global context
      │  upscale features, concat with high-res image
      ▼
[NCA level 1 (high-res, kernel 3)] → fine detail
      │
      ▼  read 3 output channels → sigmoid
   WT / TC / ET probabilities
      │
      ▼  threshold → Dice / mIoU / HD95
```

---

## 4. Repository map

```
.
├── train.py                     ← MAIN entry point (v2 recipe, GCP/Docker ready)
├── Dockerfile                   ← GPU training image for GCP
├── .dockerignore
├── requirements-docker.txt      ← pinned runtime deps
├── kaggle_v4..v7.py             ← experiment history (v4 = first strong, v7 = final)
├── README.md
├── CODE_GUIDE.md                ← this file
├── LICENSE
└── src/
    ├── models/Model_BasicNCA3D.py   ← the NCA + SE + spatial GC blocks
    ├── datasets/
    │   ├── Nii_Gz_Dataset_3D.py     ← BraTS loader (crop, norm, patch, augment)
    │   ├── Dataset_3D.py            ← base 3D dataset scaffolding
    │   ├── Dataset_Base.py          ← base Dataset (paths, length, cache)
    │   └── Data_Instance.py         ← in-memory volume cache
    ├── agents/
    │   ├── Agent.py                 ← base agent + metrics (IoU, HD95) + NQM test()
    │   ├── Agent_NCA.py             ← NCA data prep + seeding
    │   ├── Agent_Multi_NCA.py       ← multi-level batch step
    │   └── Agent_GLO_NCA.py         ← the coarse-to-fine cascade logic
    ├── losses/LossFunctions.py      ← Dice, DiceCE, Tversky, FocalTversky
    └── utils/
        ├── Experiment.py            ← config merge, data split, logging
        └── helper.py                ← file IO + NQM visualisation helpers
```

---

## 5. File-by-file walkthrough

### `train.py` — the main script
The professional entry point. It:
1. Reads config from **environment variables** (so the same code runs on Kaggle,
   laptop and GCP without edits) — `DATA_ROOT`, `OUT_DIR`, `EPOCHS`, `PATCH`,
   `AUG_LEVEL`, `N_PATIENTS`, etc.
2. Finds the BraTS dataset (`find_data_root`).
3. Builds the split (`make_split`) — 70/15/15 train/val/test, seeded.
4. Builds the two-level NCA (`BasicNCA3D` × 2), the agent, and the Experiment.
5. Trains with a custom `_clipped_batch_step` (gradient-norm clipping + the v2
   empty-region loss fix), EMA, cosine LR.
6. Selects the best model on a **smoothed** validation score.
7. Runs the final test, **tunes per-region thresholds on val**, prints results
   and a **diagnostic report**.

Key helper functions inside:
- `find_data_root()` — resolves `$DATA_ROOT` or auto-detects the folder holding
  patient sub-folders with `.nii` files.
- `worker_init_fn()` — seeds each DataLoader worker (reproducible patches).
- `_collect_probs()` — one clean forward pass over a split, returns (prob, gt).
- `_score()` — Dice/mIoU/HD95 from probs given per-region thresholds.
- `tune_thresholds()` — grid-searches the best decision threshold per region on
  **validation** (legitimate post-processing; test labels never seen).
- `_diagnose()` — turns the run into a GOOD/OK/WATCH report.
- `_clipped_batch_step()` — one training step with proper `clip_grad_norm_` and
  the empty-region loss handling.

### `src/models/Model_BasicNCA3D.py` — the model
Contains three classes:

- **`SEBlock3D`** — channel global-context. Global-average-pools the volume to
  one value per channel, runs a tiny 2-layer MLP, produces a per-channel gate in
  [0,1], multiplies the features by it. Cost: ~2·C²/r params.
- **`GCSpatialBlock3D`** — spatial global-context. Takes avg + max over the
  channel axis → two (1,D,H,W) maps → a small 3D conv → per-voxel attention gate.
  Cost: a few dozen params.
- **`BasicNCA3D`** — the NCA itself:
  - `perceive(x)` — local conv (`p0`), then (optionally) SE then spatial GC, then
    concatenate with the cell identity.
  - `update(x)` — perceive → `fc0` → BatchNorm → ReLU → (dropout) → `fc1`, then a
    **stochastic fire mask** (each voxel updates with probability `fire_rate`),
    added residually to the state.
  - `forward(x, steps, fire_rate)` — applies `update` `steps` times, keeping the
    input channels fixed.

The blocks are only built when enabled (`use_attention`, `use_spatial`), so the
plain baseline keeps the exact original parameter count — this makes the ablation
"baseline vs global-context" a single flag.

### `src/datasets/Nii_Gz_Dataset_3D.py` — the BraTS loader
Class `Dataset_NiiGz_3D_BraTS`. Per patient it:
- finds the modality/seg files tolerant to naming (`t1n/t1c/t2w/t2f` for BraTS
  2024, or older `t1/t1ce/t2/flair`; `.nii` or `.nii.gz`; `_` or `-`).
- `_foreground_bbox` + crop — removes the black background so the resized volume
  is brain-only (recovers small ET/TC detail). **This was the biggest accuracy
  win (v3 → v4).**
- `_labels_to_regions` — converts raw labels {1,2,3/4} into the 3 nested regions
  WT/TC/ET (ET = label 4, or 3 on some Kaggle copies).
- `rescale3d` — resizes with **LINEAR** for images (v2 fix — cubic overshoots and
  corrupts tiny regions) and NEAREST for labels.
- caches the processed volume so only epoch 1 is slow.
- `patchify_multimodal` — random 3D patch, biased toward a chosen region
  (`prioritize_region=2` = ET). Falls back to a WT-valid patch if ET is absent
  (v2 fix — previously fell through to a possibly-empty patch).
- `_augment` — light or heavy augmentation (see §8).
- nonzero z-normalisation — normalises using brain voxels only, keeping zeros as
  zeros so the background stays background.

The other dataset files are framework scaffolding:
- **`Dataset_3D.py` / `Dataset_Base.py`** — base classes: paths, `__len__`,
  `set_size`, state (train/val/test).
- **`Data_Instance.py`** — a dict-based cache (`Data_Container`) that stores each
  processed volume once.

### `src/agents/` — the training/inference logic
- **`Agent.py`** — `BaseAgent`: optimizer/scheduler setup, the generic
  `batch_step`, and module-level metric functions `iou_score`, `hd95_score`
  (+ `_surface_distances`). It also contains `test()` and `labelVariance()` which
  implement **NQM** ("quality control": variance over stochastic inferences) —
  kept because the thesis lists NQM as a feature, though `train.py` uses its own
  evaluation.
- **`Agent_NCA.py`** — NCA-specific `prepare_data` (moves to device, `make_seed`
  puts the 4 modalities into the first state channels) and `getInferenceSteps`.
- **`Agent_Multi_NCA.py`** — the multi-level `batch_step` (per-region loss, one
  optimizer per level).
- **`Agent_GLO_NCA.py`** — the coarse-to-fine `get_outputs`. Two paths:
  - **training (patch):** run low-res level, upscale features, extract a random
    high-res patch, run high-res level.
  - **inference (`full_img=True`):** same cascade but on the whole volume.

### `src/losses/LossFunctions.py`
Four losses kept (see §7): `DiceLoss` (used by NQM), `DiceCELoss`,
`TverskyCELoss`, `FocalTverskyCELoss` (the one `train.py` uses).

### `src/utils/Experiment.py`
Merges the config, builds the `DataSplit`, sets the dataset size, and provides
`get_from_config`. Also holds config defaults (so old configs still work) —
including the v2 additions `augment`, `augment_level`, `prioritize_region`.

### `src/utils/helper.py`
File IO (`dump/load_json/pickle`) used by `Experiment`, plus the NQM
visualisation helpers (`convert_image`, `orderArray`, `encode`) used by
`Agent.test()`.

---

## 6. Every hyperparameter explained

These are the values in `train.py` (defaults = the v2 recipe).

| Name | Value | What it does |
|---|---|---|
| `CHANNEL_N` | 24 | NCA state channels per voxel (first 4 hold the modalities) |
| `HIDDEN` | 128 | hidden width of the NCA update MLP |
| `STEPS` | [20, 20] | NCA update steps at [low-res, high-res] level — more = sharper, slower |
| `FIRE_RATE` | 0.6 | probability a voxel updates each step (stochastic regularisation) |
| `PATCH` (env) | 96 | high-res patch size: 64 (laptop/Kaggle), 96 (balanced), 128 (GCP only) |
| `LR_START` | 1.6e-3 | initial learning rate |
| `LR_MIN` | 1e-5 | final LR (cosine decay floor) |
| `TVERSKY_BETA` | 0.75 | false-negative weight (β>α penalises *missed* tumor → higher recall on small ET) |
| `FOCAL_GAMMA` | 1.33 | focal exponent — focuses learning on hard/under-segmented voxels |
| `DROPOUT` | 0.1 | dropout in the NCA MLP (regularisation) |
| `USE_EMA` / `EMA_DECAY` | True / 0.999 | exponential moving average of weights — smoother, more robust model |
| `GRAD_CLIP` | 1.0 | gradient-norm clip (training stability) |
| `BEST_WINDOW` | 3 | epochs averaged for **smoothed** best-model selection (see §9) |
| `AUG_LEVEL` (env) | heavy | `light` or `heavy` augmentation |
| `PRIORITIZE_REGION` | 2 | bias patches toward ET (rarest region) |
| `PRIORITIZE_MASKS` | 0.7 | probability a training patch must contain the region |
| `EPOCHS` (env) | 150 | training epochs (200 is generous; more rarely helps) |

---

## 7. The loss functions

BraTS regions are tiny and imbalanced (ET can be a few hundred voxels in a whole
brain), so the loss choice matters a lot.

- **`DiceLoss`** — classic overlap loss. Used by the NQM `test()` path.
- **`DiceCELoss`** — Dice + per-channel BCE. Standard, used by some kaggle_v*.
- **`TverskyCELoss`** — generalises Dice with separate false-positive (α) and
  false-negative (β) penalties. Setting **β > α (0.75/0.25)** punishes *missed*
  tumor harder → raises recall on small regions.
- **`FocalTverskyCELoss`** (the one `train.py` uses) — raises Tversky to a power
  **γ = 1.33**, concentrating gradient on the hard, under-segmented voxels. The
  SOTA choice for tiny ET on BraTS. Combined with ET-aware patch sampling.

**Important v2 subtlety (the empty-region fix):** Focal-Tversky on an *empty*
target produces a near-maximal "predict nothing" loss. Applying it to absent
regions on every patch collapsed TC/ET to 0. The v2 fix: full Focal-Tversky when
the region is present; a small (0.1×) BCE-only term when absent — the model still
learns "not tumor" without the collapse.

---

## 8. Training loop & the v2 fixes

`train.py` runs a standard loop: for each epoch, iterate the DataLoader, call
`_clipped_batch_step`, update EMA, then evaluate on val and save the best.

**The v2 improvements over v7 (all in this codebase):**

1. **Linear interpolation** (was cubic) — cubic overshoots at tumor edges,
   creating ringing that hurt ET/TC. *(dataset `rescale3d`)*
2. **Empty-region loss fix** — described in §7. This was the bug that pinned
   TC/ET at 0 on the first v2 run.
3. **Smoothed best-epoch selection** — save on the 3-epoch rolling mean, not a
   single spiky epoch (§9).
4. **Per-region threshold tuning** — tune on val, apply to test (§9).
5. **Proper gradient-norm clipping** — `clip_grad_norm_` instead of a
   per-element hook that lingered on parameters.
6. **Real augmentation, light/heavy** — v7's `USE_AUG=True` was a *no-op* (the
   dataset had no augmentation code). v2 adds actual transforms:
   - **light:** flips, 90° rotations, per-modality intensity scale/shift.
   - **heavy:** light + label-safe **elastic deformation**, gamma, Gaussian
     noise, Gaussian blur (the nnU-Net-style set).
   Geometric transforms use the same field for image and label so masks stay
   aligned; intensity transforms touch the image only and keep zeros as zeros.
7. **`PATCH` and `AUG_LEVEL` env toggles** — A/B experiments without editing
   code. (3-level cascade deliberately **not** added — v4 testing already found
   it did not help.)

---

## 9. Evaluation, thresholds & the diagnostic report

### Single clean inference
Evaluation is **one forward pass** — no ensemble, no TTA. The 200-case study
showed averaging stochastic/flipped passes *washed out* TC/ET (plain 0.766 beat
ensemble+TTA 0.738), so it was removed.

### Smoothing (best-model selection)
Because the val set is small (30 cases at 140-scale), a single epoch can spike by
luck and then generalise worse. `train.py` therefore saves the model when the
**rolling mean of the last `BEST_WINDOW` (=3) epochs** peaks — a consistently
good point, not a fluke. Larger `BEST_WINDOW` = steadier but laggier.

### Per-region threshold tuning
Tiny regions often peak below the default 0.5 threshold. After training, the
best per-region threshold is grid-searched on **validation**, then applied to
test. Results are reported at both 0.5 and tuned, so the gain is explicit.

### The diagnostic report
Every run ends with a report — printed and saved to `results.json` — giving one
verdict per signal so a cheap run tells you what to change before a paid GCP run:

| Signal | Catches |
|---|---|
| `[overfit]` | val→test gap (too big = overfitting / small-val noise) |
| `[thresh]` | did threshold tuning help, and where |
| `[converge]` | still rising (train more) vs plateaued vs falling (overfit late) |
| `[weakest]` | which region drags the mean (where to focus next) |
| `[bestep]` | best epoch too early = likely a lucky spike |
| `[vram]` | headroom to try a bigger patch |

Each line ends in **GOOD / OK / WATCH** plus a one-line overall verdict.

---

## 10. The metrics

The model is scored with three standard segmentation metrics, per region.

- **Dice** (0–1, higher better) — overlap between prediction and ground truth:
  `2·|P∩G| / (|P|+|G|)`. The primary BraTS metric. 1.0 = perfect, 0 = no overlap.
- **mIoU / Jaccard** (0–1, higher better) — `|P∩G| / |P∪G|`. Stricter than Dice
  (always ≤ Dice); measures the same idea a different way.
- **HD95** (lower better) — the 95th-percentile Hausdorff distance: how far apart
  the prediction and ground-truth *surfaces* are, ignoring the worst 5% of
  outliers. Measures boundary quality, not just overlap.

> **⚠️ HD95 units — read this before the thesis defense.** The code computes HD95
> in **voxels on the resampled training grid** (the volume is cropped and resized
> before inference). The output is labelled `HD95(vox)`. If your slides/report
> state HD95 in **millimetres (mm)**, that is not what the code currently
> produces — after resizing, a voxel is no longer 1 mm. To report true mm you
> must store each patient's original voxel spacing and scale the distance back,
> or inverse-resample the prediction to native resolution before measuring.
> Until then, report the number as **voxels**, not mm, to stay accurate.

---

## 11. Hardware & VRAM guide

Peak VRAM depends mostly on `PATCH`. Measured/estimated for the 2-level cascade
at `STEPS=[20,20]`:

| `PATCH` | Patch (high-res) | ~Peak VRAM | 6 GB laptop | 16 GB (T4/GCP) | 40 GB (A100) |
|---|---|---|---|---|---|
| 64 | 64³ | ~2.4 GB | ✅ easily | ✅ | ✅ |
| 96 | 96³ | ~5–6 GB | ⚠️ risky | ✅ | ✅ |
| 128 | 128³ | ~10–19 GB | ❌ OOM | ✅ | ✅ |

Notes:
- The diagnostic report's `[vram]` line prints the actual peak so you know if a
  bigger patch is feasible next time.
- More **epochs** do not cost more VRAM (only more time). The plateau in the val
  curve means ~200 epochs is enough; 500 mostly burns compute.
- The model itself is ~30K params — trivial memory; the cost is the 3D activation
  volumes during the NCA steps.

---

## 12. How to run

### Cheap Kaggle test (internet ON, dataset attached)
```python
import os, sys, subprocess
os.chdir("/kaggle/working")
REPO = "/kaggle/working/GLO-NCA"
subprocess.run(["rm", "-rf", REPO])
subprocess.run(["git","clone","-q","-b","v2",
    "https://github.com/Muhammadwaqas1234/GLO-NCA.git", REPO], check=True)
subprocess.run([sys.executable,"-m","pip","install","-q",
    "torchio","scipy","nibabel","opencv-python-headless"], check=False)
os.environ.update(N_PATIENTS="40", EPOCHS="20", PATCH="64", AUG_LEVEL="heavy")
os.chdir(REPO)
print("exit:", os.system(f"{sys.executable} train.py"))
```
(Requires: internet ON in Session options, and a BraTS dataset added via
**+ Add Input → Datasets**.)

### Laptop (6 GB GPU)
```bash
DATA_ROOT=/path/to/BraTS OUT_DIR=./out PATCH=64 python train.py
```
`PATCH=64` fits ~2.4 GB. `PATCH=96` is risky on 6 GB; `PATCH=128` needs a GCP-class GPU.

### GCP (Docker, full 882-case run)
```bash
docker build -t glo-nca .
docker run --gpus all \
    -e DATA_ROOT=/data -e OUT_DIR=/out -e EPOCHS=200 -e PATCH=96 -e AUG_LEVEL=light \
    -v /mnt/brats:/data:ro -v /mnt/checkpoints:/out \
    glo-nca
```
Outputs `best.pth`, `results.json`, `curves.png` to `OUT_DIR`.

---

## 13. Troubleshooting / FAQ

Real issues hit while running this project, and their fixes.

**`Could not resolve host: github.com` / `Temporary failure in name resolution`**
→ The Kaggle notebook has **internet OFF**. It is a *per-session* setting that
resets to off on every restart. Fix: Settings → **Turn on internet** (the menu
label says "Turn on" only when it's currently off), wait for the session to
restart fully, then re-run. If the toggle won't stick, your account needs
one-time **phone verification** (kaggle.com → Settings → Phone Verification).

**`BraTS dataset not found` / `/kaggle/input` is empty**
→ No **dataset** is attached. Right sidebar → **+ Add Input → Datasets tab**
(not "Notebooks") → search `brats2024-small-dataset` → Add. `train.py`
auto-detects it under `/kaggle/input`.

**`No space left on device` when downloading the dataset**
→ `/kaggle/working` is only ~20 GB; downloading + unzipping a 6.5 GB dataset
overflows it. Use **Add Input** instead (mounts read-only, zero working disk) —
never download when you can attach.

**`Unable to read current working directory` after `rm -rf`**
→ Your shell was *inside* the folder you just deleted. Add `os.chdir("/kaggle/working")`
before the `rm`/clone, or Restart the session.

**TC/ET stuck at 0.000 for many epochs**
→ This was the v2 empty-region loss bug (now fixed on the `v2` branch). If you
ever see it again after a change, suspect the loss being applied to absent
regions, or heavy aug emptying the tiny ET mask.

**Val is high but test is much lower (big gap)**
→ Overfitting or small-val noise. Mitigations already in v2: smoothed
best-epoch, EMA, augmentation. On more data (882) the gap shrinks. Judge on the
**test** number and the diagnostic `[overfit]` line, not peak val.

**Should I train 500 epochs for a better result?**
→ No. The val curve plateaus by ~epoch 90–120; extra epochs mostly waste GCP
money and can overfit. 200 is generous. The real levers are **more data (882)**
and a **bigger patch**, not more epochs.

**Can I mix BraTS 2024 (train) and 2026 (val/test)?**
→ Not as separate splits — that measures domain shift, not model quality. Pool
both, shuffle, then split train/val/test from the combined set. Keep any
cross-dataset test as a separate "external validation" number.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **NCA** | Neural Cellular Automata — a tiny update rule applied to every voxel, iterated for several steps |
| **WT / TC / ET** | Whole Tumor / Tumor Core / Enhancing Tumor — the three nested BraTS regions (ET ⊆ TC ⊆ WT) |
| **Modalities** | The four MRI scans per patient: T1, T1ce (T1c), T2, FLAIR |
| **Fire rate** | Probability a voxel updates on a given NCA step (stochastic regularisation) |
| **SE block** | Squeeze-and-Excitation — channel (global) attention |
| **Spatial GC block** | Spatial global-context — per-voxel (global) attention |
| **Foreground crop** | Cropping to the brain's bounding box, removing black background |
| **Nonzero z-norm** | Normalising intensities using only brain (non-zero) voxels |
| **Patchify** | Taking a random 3D sub-cube for training (cheaper than the full volume) |
| **ET-aware sampling** | Biasing training patches to contain the rare ET region |
| **Tversky** | A Dice-like loss with separate false-positive/negative weights |
| **Focal-Tversky** | Tversky raised to a power γ — focuses on hard voxels |
| **EMA** | Exponential Moving Average of weights — a smoothed, more robust copy |
| **Grad clip** | Capping the gradient norm for training stability |
| **Smoothing (best-epoch)** | Saving the model on a rolling-mean val score, not a single spike |
| **Threshold tuning** | Picking the best probability cutoff per region on validation |
| **TTA / ensemble** | Test-time augmentation / averaging passes — removed here (hurt TC/ET) |
| **Deep supervision** | nnU-Net trick (loss at multiple decoder scales) — not used here |
| **HD95** | 95th-percentile Hausdorff distance — boundary quality (lower better) |
| **Dice / mIoU** | Overlap metrics (higher better) |
| **Cascade** | The coarse-to-fine 2-level (low-res → high-res) structure |

---

## 15. Version history

| Version | Key change | 200-case result (WT/TC/ET) |
|---|---|---|
| v4 | foreground crop + nonzero-norm + Tversky; no aug | 0.885 / 0.734 / 0.729 |
| v5 | + spatial GC block + hidden 128 | val ~0.83 |
| v6 | + ET-aware sampling + ensemble/TTA | ensemble hurt TC/ET |
| v7 | + Focal-Tversky + EMA + grad-clip + dropout (aug was a no-op) | plain 0.86/0.72/0.72 |
| **v2** | clean `train.py`, ensemble removed, **real** aug (light/heavy), linear interp, empty-region loss fix, threshold tuning, smoothed best, PATCH/AUG toggles, Docker, diagnostics | in progress on 882 |

`main` + tag `v1.0` = the stable pre-v2 baseline; `v2` branch = all improvements.

---

## 16. Results & honest positioning

**GLO-NCA does not beat nnU-Net / Swin UNETR on raw Dice** — no 30K-parameter
model can. On BraTS the big models reach WT ~0.89–0.91; GLO-NCA reaches WT
~0.86–0.90 with **~1000× fewer parameters**.

Where GLO-NCA is **at SOTA level**: preprocessing (foreground crop + nonzero
z-norm) and the imbalance-aware loss (Focal-Tversky + ET-aware sampling).

Where it differs by design (defensible trade-offs, not mistakes): a small patch,
single-pass inference, and no deep supervision — all chosen to keep the model
tiny.

**The contribution is efficiency-per-Dice:** comparable accuracy at a fraction of
the parameters, running on low-resource hardware. On the full 882-case dataset
the regularisation-heavy v2 recipe is expected to generalise better (smaller
val→test gap) than the earlier small-data versions — which is exactly what the
diagnostic report is built to measure.

---

*For setup and a short overview see [README.md](README.md). This guide is the
full technical reference.*
