# GLO-NCA V3 — Performance Audit

**Date:** 2026-09-15 · **No GCP compute used** (VM `culling` already terminated). ·
**No methodology changed.** · **No code changed yet** (measure-before-fix).

## Status
**NEEDS FIX (pending Kaggle H100 measurement).** The ~90 min/epoch is explained
primarily by a **genuinely large workload** (not a single smoking-gun bug), plus
**two real implementation inefficiencies** worth fixing. All conclusions below are
**code-read hypotheses to be confirmed by the profiler on Kaggle** — this box has
no GPU/torch, so nothing here was executed locally. Numbers marked *measured* come
only from the GCP run we observed; everything else is *theoretical* or *to be
measured*.

---

> **UPDATE 2026-09-15 (supersedes the old-workload estimate below):** the ACTUAL
> old Kaggle pipeline has now been read from `archive/kaggle_experiment_history/`
> (kaggle_v5/v6/v7). It is **2-level (32×32×24 + 64×64×48 patch), 40 NCA steps,
> 882 cases, 150 epochs, NO checkpointing** — and it trained on a **single 64³
> patch per case**, not the full volume. The "64³×64 single-level, ~21×" figure
> in section 4 below is a **superseded hypothesis**; the fact-based ratio is
> **~18×** per epoch. See `GLO_NCA_V3_OLD_VS_V3_WORKLOAD.md` for the authoritative
> comparison. The **dominant** reason V3 is slower is now confirmed: **old =
> single 64³ patch/case, 2 levels; V3 = full 128³ volume, 3 levels** (+2×
> checkpointing). Additional confirmed finding **B3** below.

## 1. Root cause (summary)
The observed ~90 min was **epoch 1 with a COLD cache** (first-ever load of 896
cases) **plus** the intrinsically heavy V3 compute. The dominant cost is GPU
compute of the NCA unroll, amplified ~2× by gradient checkpointing. The data
pipeline is a secondary cost that is **worse than it should be** because the
in-memory preprocessing cache is defeated across epochs (finding B2).

Critically: **GPU utilisation was observed at 100%** during the GCP epoch 1. That
means the GPU was **not** starved by data loading — the bottleneck is real GPU
compute, i.e. the NCA unroll + checkpointing recompute, *not* idle-waiting on I/O.

---

## 4. Theoretical workload (computed, not guessed)
Config `v3_smoke_5epoch.yaml` (= production ckpt config + epochs): levels
L1 32³/20 steps, L2 96³/20 steps, L3 128³/10 steps; batch 1; 896 train cases.

| Quantity | Per epoch | Formula |
|---|---|---|
| NCA `update` calls | **44,800** | 896 × (20+20+10) |
| NCA `forward` calls | **2,688** | 896 × 3 levels |
| backward calls | **896** | 1 per case (batch 1) |
| optimizer steps | **896** | 1 per case |
| **With checkpointing ON** | effective **89,600** update executions | each update recomputed once in backward |

Voxel-step workload (voxels × steps), the honest measure of NCA compute:

| Level | Voxels | Steps | Voxel-steps/case |
|---|---|---|---|
| L1 32³ | 32,768 | 20 | 655,360 |
| L2 96³ | 884,736 | 20 | 17,694,720 |
| L3 128³ | 2,097,152 | 10 | 20,971,520 |
| **Total** | | | **39,321,600 / case** |

- **Per epoch (896 cases): 35.2 billion voxel-steps.**
- **With checkpointing (fwd + recompute): 70.5 billion effective voxel-steps/epoch.**

### Apples-to-apples vs the old Kaggle run (200 cases × 200 epochs in ~4–5 h)
If the old run was a typical single-level ~64³ NCA (~64 steps, **no**
checkpointing) — to be confirmed by you:

| Run | Voxel-steps/case | Cases | Voxel-steps/epoch |
|---|---|---|---|
| Old Kaggle (hypothetical 64³×64, no ckpt) | 16.8 M | 200 | 3.35 B |
| **Current V3 (128³ 3-level ×2 ckpt)** | 78.6 M (effective) | 896 | **70.5 B** |
| **Ratio** | ~4.7× | 4.5× | **~21× more work per epoch** |

**~21× more work per epoch** ≈ fully consistent with "old epochs were minutes, new
epoch was ~90 min." So the slowness is **mostly the workload**: 4.5× more cases ×
~2.3× more voxel-steps/case × 2× checkpointing recompute. This is *expected cost*,
not a bug — but the 2× (checkpointing) and the cache defeat (B2) are addressable.

---

## Findings (classified)

### A. Bug-level / correctness-adjacent
- **A1 — DataLoader cache is defeated across epochs.** The dataset caches
  load+crop+resize+label in an **in-memory dict on the dataset object**
  (`Data_Instance.py`, `self.data = {}`). But the runner **recreates the
  DataLoader inside the epoch loop** (`runner.py:434`), and with `num_workers>0`
  each worker gets a **forked copy** of the dataset and fills its **own** cache,
  which is **discarded when the workers are torn down at end of epoch**. Net
  effect: **every epoch re-reads and re-preprocesses all 896 cases from disk**,
  instead of once. This is the single biggest *fixable* pipeline cost.
  *Fix (runtime, no methodology change): create the DataLoader once outside the
  loop with `persistent_workers=True`, or pre-warm/serve the cache from the main
  process. Numerically identical data; only avoids repeated disk+preprocess.*

### B. Implementation inefficiencies (hot path: `BasicNCA3D.update`, run 44,800×/epoch)
- **B1 — Per-step tensor layout churn.** `update()` performs **~6
  `.transpose(1,4)`** per call (`Model_BasicNCA3D.py:158–178`) feeding
  `Linear`/`BatchNorm` that force contiguous copies; `forward()` adds a `.clone()`
  and a full `torch.concat` rebuild **every step**
  (`Model_BasicNCA3D.py:201–204`). At 44,800 updates/epoch this is a large number
  of allocations/copies. *Potential fix (implementation, must prove bit-identical):
  reduce redundant transposes / avoid the per-step concat rebuild. HIGH RISK of
  changing numerics — only if the profiler shows it matters and equivalence tests
  pass.*
- **B2 — V3 `_to_cf`/`_to_cl` force `.contiguous()`** on every resize/projection/
  fusion (`Model_GLO_NCA_V3.py:53,58`). Several copies per forward. Secondary to
  the unroll cost.
- **B3 (new, confirmed by code) — patchify is a no-op crop that still runs the
  50-iteration ET-aware label scan.** V3 resizes each volume to 128³
  (`input_size[-1] = [128,128,128]`) and then `patchify_multimodal` extracts a
  **128³ patch from a 128³ volume** → `random.randint(0, 0) = 0` (the patch IS the
  whole volume). But the ET-aware retry loop
  (`Nii_Gz_Dataset_3D.py:335`, up to 50 iterations of label slicing + `.max()`)
  still runs, scanning labels 50× per case per epoch for **zero benefit** (there is
  only one possible patch position). *Fix (implementation, numerically identical:
  when volume size == patch size, skip the retry loop / return the volume). Pure
  wasted CPU removal; does not change which voxels are trained on.*
  **NOTE:** this also means V3 effectively trains on the FULL 128³ volume — which is
  the intended multi-level design, NOT a bug. Switching to true sub-patch training
  (like the old 64³ run) would be a METHODOLOGY CHANGE and is out of scope.

### C. GPU-sync stalls (per batch, 896×/epoch — minor vs unroll)
- **C1 — `if 1 in targets[..., m]:`** (`runner.py` batch step) does a Python
  membership test on a **GPU tensor** → forces GPU→CPU sync + full scan, ×3
  regions/batch.
- **C2 — `loss_ret[m] = loss_loc.item()`** ×3/batch → 3 more syncs.
  *Fix (runtime): compute region-presence with a single fused `(targets==1).any()`
  on-device and defer `.item()`. Correctness-preserving.*

### D. Confirmed NOT bottlenecks (ruled out by code read)
- **HD95 is eval-only** (`metrics_eval.py` `score`/`score_per_case`) — **not** in
  the training batch step. ✔
- **No GCS/network call inside the training loop.** The GCS sync runs in a
  separate background watcher at an interval (`_train_entrypoint.sh`), never per
  batch. ✔ (And this run used host Python, no Docker/systemd, so no sync at all.)
- **Checkpoints are saved per epoch, not per batch** (`runner.py` best/last after
  the epoch). ✔
- **EMA** iterates a handful of small tensors (40,656 params) per batch — cheap. ✔
- **No duplicate training passes / nested loops.** Call-count math is consistent:
  1 forward + 1 backward + 1 opt.step per case; the profiler asserts this.

### The big one — E. Gradient checkpointing ~2× compute
Checkpointing was added so unchanged V3 128³ fits the L4 (9.07 GB TRUE FIT). It
**recomputes each NCA step's forward during backward** → roughly **2× the NCA
compute**. On the L4 (a low-power 24 GB card) this is the difference between
fitting and OOM, but it **doubles the dominant cost**. On an **H100 (80 GB)** the
unchanged architecture should fit **without** checkpointing with large headroom —
making `gradient_checkpointing=false` a **pure runtime speedup on H100 that does
NOT change the architecture or outputs**. This is the highest-leverage, safest win
and is exactly what the Kaggle test will quantify.

---

## What must be MEASURED on Kaggle H100 (before any code change)
Using `scripts/profile_v3_training.py` and the Kaggle notebook:
1. **Checkpoint ON vs OFF** at 96³ and 128³ → forward/backward/total per case, and
   **peak VRAM** (does unchanged V3 fit on H100 without checkpointing, with
   headroom?).
2. **Per-stage breakdown** (load / preprocess / H2D / forward / backward / opt /
   EMA) → confirm the unroll dominates and quantify B1/B2.
3. **DataLoader workers 0/2/4/8** throughput and whether the cache persists across
   epochs (confirm A1).
4. **Call-count assertion** → prove no duplicate passes (forward=cases×levels,
   update=cases×Σsteps).
5. **Per-case + per-epoch time**, then project 300-epoch runtime and Kaggle cost.

---

## Q&A (deliverable #28) — answers so far (⚠ = needs Kaggle to finalize)
1. **Why ~90 min/epoch?** Epoch-1 cold cache + ~21× more work than the old
   Kaggle run (4.5× cases × ~2.3× voxel-steps/case × 2× checkpointing). ⚠ split by stage on Kaggle.
2. **Stage breakdown?** ⚠ measured on Kaggle (profiler).
3. **Actual code bug?** One real inefficiency (A1, cache defeated across epochs) +
   minor sync stalls (C1/C2). No incorrect duplicate training pass.
4. **Duplicate computation?** No duplicate *training* pass; but A1 causes duplicate
   *preprocessing* every epoch. Checkpointing intentionally recomputes forward.
5. **DataLoader a bottleneck?** Partially (A1). ⚠ quantify on Kaggle.
6. **Repeated NIfTI preprocessing?** Yes, due to A1. ⚠ quantify.
7. **Checkpointing the major bottleneck?** Likely the largest *fixable* GPU factor
   (~2×). ⚠ measure ON vs OFF.
8. **How much faster is checkpoint OFF on H100?** ⚠ measure.
9. **Does V3 fit without checkpointing on H100?** Expected yes (80 GB). ⚠ measure.
10–14. per-case/epoch/300-epoch/cost/GCP projections — ⚠ from measurement.
15. **Safe optimizations?** A1 (persistent cache), C1/C2 (sync removal),
   checkpoint=OFF on H100 (runtime), pin_memory/non_blocking. All non-methodology.
16. **Needs approval?** Only if B1 (touching NCA numerics) proves necessary — that
   would be classified and shown as a diff with equivalence tests first.
17–18. **Ready for final Kaggle / GCP?** Not yet — measurement first.

## Methodology changes
**NONE.** No architecture/loss/split/steps/resolution/channels/eval changes made
or proposed. Checkpointing OFF (if used on H100) is a **runtime** memory choice
with **identical** architecture and outputs.
