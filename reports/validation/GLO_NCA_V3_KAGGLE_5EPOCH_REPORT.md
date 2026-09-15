# GLO-NCA V3 — Kaggle H100 5-Epoch Validation (plan + results)

**Date:** 2026-09-15 · **No GCP compute.** · **Methodology frozen.** ·
**Status: NOT YET RUN** — this documents the plan; results are filled in after you
run `notebooks/kaggle_v3_h100_profile.py` on Kaggle. No 200/300-epoch run.

## Purpose
Validate the corrected/measured V3 pipeline on an H100 (80 GB) with the REAL
BraTS-MET data, at exactly 5 epochs, and answer the audit's open questions with
measured numbers (per-stage timing, checkpoint on/off, H100 fit, throughput).

## Config
`configs/v3_kaggle_5epoch.yaml` — derived from the frozen production config.

### Frozen (identical to thesis methodology) — unchanged
V3 · 32³→96³→128³ · 40,656 params · SE + spatial GC · concat fusion · 4 MRI
modalities · WT/TC/ET sigmoid multilabel · Focal-Tversky+BCE (β=0.75, γ=1.33) ·
AdamW · cosine LR · EMA 0.999 · grad-clip · batch=1 · seed=42 · light aug ·
val-only threshold tuning · single-pass test · HD95 in voxels · master split.

### Runtime differences (documented, scientifically harmless)
| Setting | Production/GCP | Kaggle | Why (classification) |
|---|---|---|---|
| `memory.gradient_checkpointing` | `true` (L4 must, to fit 24 GB) | **`false`** | H100 80 GB fits unchanged V3 128³ without recompute → ~2× faster. **RUNTIME** memory choice; identical architecture/outputs. **Confirm H100 peak-VRAM headroom with the profiler first.** |
| GPU | NVIDIA L4 24 GB | NVIDIA H100 80 GB | hardware only |
| `training.workers` | 4 | tune per Kaggle CPU (profiler sweep) | **RUNTIME** DataLoader tuning; no data change |

**No architecture, loss, split, steps, resolution, channels, batch, or evaluation
change.** Checkpointing OFF is a memory decision, not an architectural one.

## Invalid-case decision (deliverable #22) — EXPLICIT, not silent
The canonical `split/master_split.json` (898/200/198, SHA
`d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d`) **includes**
two cases with invalid segmentation labels found by the frozen validator:
- `BraTS-MET-01094-002` (stray label 6, 129 voxels)
- `BraTS-MET-01184-002` (stray label 8, 28 voxels)

Both are in the **train** split only (val/test unaffected). The runner's frozen
dataset validation will **reject the whole run** on these until a decision is made.

**This is a data-integrity decision that affects the scientific split, so it is
NOT made automatically. STATUS: OPEN — the user will select an option later.**
The Kaggle run is BLOCKED on this choice (the runner's frozen validation rejects
the canonical split until it is resolved). The three documented options:

1. **Exclude the 2 from train for the Kaggle 5-epoch smoke** (as done for the L4
   run): run-local split `master_split_culling.json` (896/200/198, SHA
   `52f885f2…`), canonical file untouched, val/test identical. Fine for an
   **operational** smoke; must NOT silently become the final scientific split.
   → Clearly labelled **OPERATIONAL SMOKE ONLY**.
2. **Fix the 2 seg files** (remap stray labels 6/8 → background) — a defensible
   data cleanup that keeps 898 train; needs explicit approval and documentation,
   and would change the canonical split's content fingerprint.
3. **Investigate whether the established methodology remaps labels** in the loader
   (so these are not truly training-invalid, only validator-strict).

**No option is applied yet.** When the user picks one, it is documented here and
applied. Until then the canonical split and dataset remain byte-for-byte untouched
(SHA `d30d71956ee9…09559d`), and the Kaggle final-5-epoch step does not run.
The two invalid case IDs for reference: `BraTS-MET-01094-002`, `BraTS-MET-01184-002`.

## Micro-benchmark protocol (run in order; STOP on any failure)
- **TEST 1** — 1 case, forward only (H100, ckpt off) → runs, sane output shape.
- **TEST 2** — 1 case, forward + backward → gradients flow, no OOM.
- **TEST 3** — 2–4 cases, full optimizer step → loss decreases across a few steps.
- **TEST 4** — 10 cases, throughput benchmark → cases/s, per-stage timing.
- **TEST 5** — short smoke (few cases, 1 epoch via train.py) → full path OK.
- **FINAL** — exactly **5 epochs** on the real split.

## Measured results — TO FILL AFTER RUNNING
### Model micro-bench (Section 2)
| R | ckpt | fwd (ms) | step (ms) | peak VRAM (GB) |
|---|---|---|---|---|
| 96 | off | | | |
| 96 | on | | | |
| 128 | off | | | |
| 128 | on | | | |

**H100 fit without checkpointing at 128³:** peak = ___ GB of 80 → PASS/FAIL.
**Checkpoint speedup (off vs on) at 128³:** ___×.

### Per-stage breakdown (profiler, Section 3)
load ___ / preprocess ___ / H2D ___ / forward ___ / backward ___ / opt ___ /
EMA ___ ms per case. Call counts: forward ___ (expect cases×3), update ___
(expect cases×50). **No duplicate passes: PASS/FAIL.**

### DataLoader sweep (Section 4)
workers 0/2/4 → ___ / ___ / ___ cases/s. Cache persists across epochs: YES/NO.

### 5-epoch run
| epoch | train_loss | WT Dice | TC Dice | ET Dice | mean Dice | mIoU | HD95 | lr | epoch_s | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | |
| … | | | | | | | | | | |

**Per-case time:** ___ · **Per-epoch:** ___ · **Projected 300-epoch:** ___ h ·
**Projected Kaggle cost:** ___ · **Projected GCP (L4, ckpt on):** ___ h.

## Final status — TO FILL
`GLO-NCA V3 KAGGLE 5-EPOCH: PASS / BLOCKED — <reason>`
