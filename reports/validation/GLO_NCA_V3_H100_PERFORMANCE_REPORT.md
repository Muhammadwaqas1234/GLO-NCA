# GLO-NCA V3 — H100 Performance Report

## STATUS
**PARTIAL MEASUREMENT DONE on Kaggle Tesla T4 (NOT H100).** Kaggle's free tier
assigned a **Tesla T4 (15.6 GB)**, not an H100 — so the "does V3 fit without
checkpointing on 80 GB" question is NOT answered; a T4 has far less VRAM than an
H100. What the T4 run DID measure (real, on-GPU): checkpoint ON works and its peak
VRAM + per-case time, and that V3 128³/96³ **OOM without checkpointing on 16 GB**.

### MEASURED — Kaggle Tesla T4 (15.6 GB), torch 2.10.0+cu128, batch 1, synthetic input
| Resolution | Checkpoint | Result |
|---|---|---|
| 96³  | OFF | **OOM** (needs > 14.56 GB free) |
| 96³  | ON  | fwd 6.16 s · bwd 17.28 s · **total 23.4 s/case** · peak **4.97 GB** |
| 128³ | OFF | **OOM** (needs > 14.56 GB free) |
| 128³ | ON  | fwd 8.86 s · bwd 24.45 s · **total 33.3 s/case** · peak **9.07 GB** |

Params **40,656** (match). **FACTS from this run:**
- V3 128³ peak with checkpointing = **9.07 GB** — identical to the earlier L4 gate
  (cross-hardware confirmation the memory number is real).
- **Without checkpointing, V3 OOMs even at 96³ on a 16 GB T4** → checkpointing is
  REQUIRED on ≤16 GB GPUs (T4/L4-tier is tight). This does NOT tell us about H100.
- Per-case time on T4 is very slow (~33 s/case at 128³) because the T4 is a weak,
  older GPU — an H100/A100 would be far faster; do not use T4 timing as the target.

**Checkpoint OFF vs ON slowdown and VRAM-reduction could NOT be computed** (OFF
OOM'd, so no baseline). That still needs a big-VRAM GPU (A100/H100).

### DECISION (user, 2026-09-15): accept T4 findings — checkpointing ON everywhere
Conclusion adopted: **checkpointing stays ON on all ≤24 GB GPUs (T4/L4)** — this is
now MEASURED-necessary (V3 OOMs without it at 96³/128³ on 16 GB). The H100
no-checkpoint path is **out of scope** (Kaggle gave a T4; not pursued). Applied:
`configs/v3_kaggle_5epoch.yaml` `gradient_checkpointing` → **true** (was a
hypothetical-H100 `false`). Production configs unchanged (`v3_multilevel_ckpt.yaml`
already true; `v3_multilevel.yaml` unchanged). Next real work: get BraTS data onto a
GPU box to run the real-data TESTs (3,4,7,8,10) — the data-pipeline/patchify
measurements that matter more than the checkpoint-off speedup.

---

## (original plan) STATUS
**BLOCKED — awaiting Kaggle H100 measurement.** The code-side audit is complete
and no unmeasured optimization has been applied. This box has **no GPU/torch**, so
every timing/VRAM number below is a placeholder to be filled by running
`notebooks/kaggle_v3_h100_profile.py` on a Kaggle H100. Labels: **VERIFIED FACT**
(read from source), **MEASURED** (on hardware), **CALCULATED** (arithmetic),
**HYPOTHESIS** (unconfirmed).

---

## A. Executive summary
- The old Kaggle run was fast because it was a **lighter pipeline** (2 levels,
  single 64³ patch/case, 40 steps, no checkpointing) — **VERIFIED FACT** from
  `archive/kaggle_experiment_history/kaggle_v5–v7.py`.
- Current V3 does **~18× more work per epoch** (3 levels, full 128³ volume, 50
  steps, +2× checkpointing) — **CALCULATED**. This is legitimate workload, not a
  bug.
- Two avoidable-overhead candidates remain, to be **MEASURED** before fixing:
  **(E)** gradient checkpointing 2× (removable on H100 if it fits), **(A1)** the
  DataLoader recreated per epoch defeating the in-memory cache, and **(B3)** the
  patchify ET-scan loop that is a no-op when patch==volume.
- **No P0 correctness bug found. No duplicate training pass.** — **VERIFIED FACT**
  (call-count analysis).

## B. Actual old Kaggle workload — **VERIFIED FACT** (source)
2 levels · 32×32×24 + 64×64×48 · 20+20=40 steps · 1 random 64³ patch/case ·
ch24 · hidden128 · fire0.6 · batch1 · workers4 · 150 epochs · ~882 cases ·
checkpointing OFF. (`kaggle_v5.py`/`v6.py`/`v7.py`.)

## C. Actual V3 workload — **VERIFIED FACT** (config + model)
3 levels · 32³/96³/128³ · 20+20+10=50 steps · full 128³ volume (patchify is a
no-op crop, see F) · 40,656 params · batch1 · seed42 · checkpointing ON (L4).

## D. Mathematical comparison — **CALCULATED**
See `GLO_NCA_V3_OLD_VS_V3_WORKLOAD.md`. Per-case 4.42 M (old) vs 39.3 M (V3)
voxel-steps = 8.9×; ×2 checkpointing = 17.8×; per-epoch ~18×.

## E. Confirmed bottlenecks — **none yet MEASURED**
All bottleneck claims require Kaggle timing. Ranked *hypotheses*:
1. GPU compute of the NCA unroll (esp. L2 96³ + L3 128³) — **HYPOTHESIS** (GPU was
   observed 100% on L4 → compute-bound, but stage split unmeasured).
2. Checkpointing 2× recompute — **HYPOTHESIS**, measure ON/OFF.
3. Per-epoch repeated preprocessing (cache defeat, A1) — **HYPOTHESIS**, measure
   epoch1 vs epoch2 load time.

## F. Patchify no-op — **VERIFIED FACT** (source), fix **NOT yet applied**
`Experiment.set_size()` uses `input_size[-1] = [128,128,128]`; dataset resizes each
volume to 128³; `patchify_multimodal` then extracts a 128³ patch from a 128³
volume → `random.randint(0, 0) = 0` on all axes → the "patch" is the whole volume.
With `priotize_masks=0.7, prioritize_region=2` (**VERIFIED FACT**, `_build_v3`), ~70%
of cases enter the retry loop and scan `label[...,2].max()` up to **50×** for zero
benefit (only one possible position).

**Why the obvious fix is NOT yet applied (correctness caution):** patchify uses the
Python `random` module (`random.uniform`, `random.randint`), and **augmentation runs
AFTER patchify using the same `random` stream**. `random.randint(0,0)` still
*consumes* RNG state, and the loop's iteration count varies per case (1 if ET found,
up to 50 if not). So short-circuiting the loop **changes the Python RNG stream that
reaches augmentation** → the run would **not be bit-identical** to current runs.
This makes the fix a **P1 that must be validated** with a reproducibility/
equivalence test, NOT a blind edit.

**MEASURED (locally, NumPy-only, `scripts/test_patchify_equivalence.py`):** for the
patch==volume case across seeds {0,1,42,123}:
- current patchify output == full volume (img & label **bit-identical**): **TRUE**
  for all seeds → the crop is provably a no-op.
- a *naive* `if patch==volume: return` short-circuit consumes **0** `random.*`
  draws vs the current **4** → **RNG SHIFT confirmed**. Because the dataset shares
  the Python `random` stream with augmentation (which runs next in `__getitem__`),
  the naive fix would change augmented outputs → **NOT bit-identical training**.

**Conclusion (VERIFIED):** the wasted work is real and the output is provably the
full volume, but a bit-identical fix must **preserve the exact `random.*` draw
count** (which itself varies with ET presence, 3–150 randint draws).

**RNG-PRESERVING CANDIDATE DESIGNED + VERIFIED (locally, synthetic).** A candidate
fast path exists that, for patch==volume, computes the region/WT label max **once**
(the patch is the whole volume, so every loop iteration's slice is identical) and
reproduces the **exact** `random.*` sequence the current loop makes:
- `not contains_mask` OR region present → 1 iteration worth of draws (3 randint);
- region absent (+ prioritize) → full 50-iteration draws (150 randint).

Verified across seeds × {ET present/absent} × {prioritize on/off}: **output
bit-identical AND `random.getstate()` identical after patchify** in every case
(`scripts/test_patchify_equivalence.py` and the notebook's TEST 10). Because the
Python RNG state is preserved exactly, **augmentation (which runs next on the same
stream) is unaffected** → the fix would be bit-identical training.

**Verdict: SAFE-TO-APPLY pending REAL-DATA confirmation.** The design is proven in
isolation; TEST 10 re-runs the same equivalence assertion on **real BraTS cases**
(real label distributions) on Kaggle. Only if TEST 10 prints **"SAFE TO APPLY"** on
real data will the fix be applied to `Nii_Gz_Dataset_3D.py` (P1), with the same
equivalence test kept as a regression guard. It saves ~49 redundant 128³ label
`.max()` scans on the ~70% of cases where ET is absent — pure CPU-overhead removal.

## G. Hypotheses not yet proven
Checkpointing overhead %, DataLoader-wait %, cache-across-epochs effect, per-level
NCA time split, H100 fit-without-checkpointing, patchify overhead in real epoch
terms. All **HYPOTHESIS** until measured.

## H. Runtime-only fixes implemented
**NONE yet.** Per the measure-before-fix mandate and the absence of a local GPU, no
model/runner/dataset code has been changed in this stage. Candidates (all P1) are
staged for measurement first: patchify short-circuit (F), persistent DataLoader +
`persistent_workers` (A1), removing per-batch `if 1 in targets`/`.item()` syncs
(C1/C2). Each will be applied **only** after Kaggle measurement shows it helps and a
regression proves equivalence.

## I–N. Measured results — TO FILL FROM KAGGLE
### K. H100 memory (checkpoint OFF) at 128³, batch 1
peak allocated ___ GB / reserved ___ GB of 80 → fits with headroom? ___ (**MEASURED**)
### L. Checkpoint ON vs OFF (128³)
| mode | fwd ms | bwd ms | total ms | peak GB | samples/s |
|---|---|---|---|---|---|
| OFF | | | | | |
| ON | | | | | |
→ checkpoint slowdown ___% ; VRAM reduction ___% (**MEASURED**)
### M. DataLoader workers (real data)
| workers | first-batch s | steady s | cases/s | GPU-util% |
|---|---|---|---|---|
| 0 | | | | |
| 2 | | | | |
| 4 | | | | |
Cache epoch1 vs epoch2 load time: ___ vs ___ (**MEASURED** → confirms/denies A1)
### Per-level NCA time (128³, ckpt OFF)
L1 ___ / L2 ___ / L3 ___ ms (**MEASURED**)
### N. 1-epoch then 5-epoch (real data)
train_s / val_s / total_s / peak VRAM / GPU-util / losses / Dice / IoU / HD95 — TO FILL.

## O. Remaining risks
- Bit-identical claim for the patchify fix (RNG-stream) — must be tested.
- H100 without checkpointing may still not leave room for larger batch (not needed;
  batch stays 1).
- DataLoader cache fix must not change sampling/augmentation semantics.

## P. Invalid-label decision status
**OPEN** — user will select later (`GLO_NCA_V3_KAGGLE_5EPOCH_REPORT.md`). Canonical
split untouched (SHA `d30d719…`).

## Q. Recommendation
**BLOCKED** until the Kaggle H100 measurements above are collected. Next action:
run `notebooks/kaggle_v3_h100_profile.py` (TESTs 1–10), record numbers here, then
apply only the fixes measurement justifies (with regression), then 1-epoch →
5-epoch → resume on H100. **GCP remains blocked.**
