# fast_grid4864 — Evidence Record

**Status: EVALUATED-ONLY. NOT PROMOTED.**

A Category-C candidate configuration, recorded here so a future promotion
decision rests on evidence rather than recollection. Nothing in this document
authorises its use as the production architecture.

---

## 1. What it is

| | production | `fast_grid4864` |
|---|---|---|
| geometry | 48³ / 64³ | 48³ / 64³ (same) |
| NCA steps | 20 + 20 = 40 | **15 + 15 = 30** |
| spatial GC kernel | **k = 7** | **k = 5** |
| patchify | OFF | **ON** |
| parameters | **33,089** | **32,217** |
| L1 / L2 perception kernels | 7 / 3 | 7 / 3 (same) |
| SE, fusion, global-context source | unchanged | unchanged |

Three changes, two of which touch the thesis contribution: the spatial
global-context kernel shrinks from 7 to 5, and patchify means the
high-resolution level trains on a crop rather than the whole volume.

---

## 2. What was measured

Two Kaggle T4 runs, **200 cases, local 129/31/40 split, seed 42**. Both are
engineering runs on a small dataset; neither used the 898/200/198 master
split.

### 2.1 30 epochs, 32³/48³ grid (an earlier, even faster variant)

| | |
|---|---|
| WT / TC / ET Dice | 0.737 / 0.642 / 0.626 |
| mean Dice | 0.668 |
| IoU | 0.595 / 0.493 / 0.474 |
| HD95 (voxels) | 4.63 / 5.08 / 5.12 |
| mean epoch | 72.6 s |
| mean iteration | 1.664 s |
| peak VRAM | 684 MB |
| parameters | 32,217 |

### 2.2 30 epochs, 48³/64³ grid (`fast_grid4864` proper)

| | |
|---|---|
| WT / TC / ET Dice | **0.774 / 0.677 / 0.661** |
| mean Dice | **0.704** |
| IoU | 0.640 / 0.534 / 0.515 |
| HD95 (voxels) | 5.69 / 5.48 / 5.53 |
| mean epoch | 226.7 s |
| mean iteration | 1.664 s |
| peak VRAM | 936 MB |
| parameters | 32,217 |
| best epoch | 20–21 (WT 0.789, TC 0.727, ET 0.713) |
| convergence | plateaued by ~epoch 26 |

Both runs completed their resume test (weights, EMA, epoch/step, scheduler,
continue-training) and their architecture identity gate.

### 2.3 Speed, production preset, same T4

Measured by `time_presets()` on the Kaggle T4, 129 train / 31 val:

| preset | iteration | projected epoch | 50 epochs |
|---|---|---|---|
| `production` (40 steps, k=7) | 2.979 s | 407.7 s | 5.66 h |
| `fast_grid4864` | 1.677 s | 232.1 s | 3.22 h |
| `fast` (32³/48³) | 0.448 s | 63.3 s | 0.88 h |

`fast_grid4864` is **1.76× faster** than production on this hardware.

---

## 3. What was NOT measured

**This is the reason the candidate cannot be promoted.**

| Missing evidence | Why it matters |
|---|---|
| **Production preset, same data, same epochs** | The 0.704 mean Dice has no production-preset counterpart on the 200-case split. Without it, the Dice difference between the two configurations is unknown — only the speed difference is established. |
| Any run on the 898/200/198 master split | All Dice figures come from a 129-case local split. |
| Overfitting behaviour over a full budget | 30 epochs on 129 cases says little about 300 epochs on 898. |
| Variance across seeds | Single run per configuration; no confidence interval. |
| Test-set performance | Correctly withheld — the test split must stay frozen until a configuration is chosen. |

**NOT MEASURED — REQUIRES EXTERNAL RUN.** A fair comparison needs the
production preset trained on the same data for the same number of epochs.
That was not executed here, and no substitute figure has been invented.

---

## 4. Honest reading of the evidence

What the data supports: `fast_grid4864` is **1.76× faster** than production
on a T4, converges by roughly epoch 26 on a small dataset, and reaches
0.774 / 0.677 / 0.661 there.

What the data does **not** support: any claim that it matches, beats, or
loses to the production architecture. The comparison run does not exist.

The 32³/48³ variant (§2.1) does give a controlled grid comparison, since only
the grid differed: the larger grid gained **+0.036 mean Dice for 3.2× the
time**. That is informative about grid size, and says nothing about the step
count or the k=7 → k=5 change.

---

## 5. Promotion requirements

Before `fast_grid4864` could become production, all of the following:

1. A production-preset run on identical data, split, epochs and seed.
2. Both configurations trained on the 898/200/198 master split.
3. Evidence that k=5 does not degrade the spatial global-context
   contribution the thesis claims — this is the contribution itself, not an
   incidental hyperparameter.
4. Evidence that patchify does not harm whole-volume behaviour, given the
   high-resolution level would no longer see the full volume during training.
5. Explicit supervisory approval recorded as a scientific decision.

Speed alone is not a promotion argument.

---

## 6. Current status

`fast_grid4864` is reachable only through `use_preset("fast_grid4864")` in
the standalone. The production default is the 33,089-parameter architecture.
Every artifact from a Category-C run is stamped
`result_may_be_reported_as_production: false`.

**EVALUATED-ONLY. NOT PROMOTED.**
