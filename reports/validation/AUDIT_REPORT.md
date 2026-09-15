# GLO-NCA V3 — Real-Data Profiler Recovery Audit

**Date:** 2026-09-15 · Diagnostic only · No GPU run during this audit · No
production source or canonical split modified.

## A. Current local sample path
`C:\Users\raiwa\kaggle_brats100\MICCAI-LH-BraTS2025-MET-Challenge-Training`
(staging dir; contains ONLY the data folder + `dataset-metadata.json`).

## B. Current Kaggle dataset reference
`muhammadwaqasabid/glo-nca-brats-100` (private). Exists (view API 200). Kaggle
`totalBytes` reported ~33.5 GB — **not trusted** as a case count (metadata quirk);
the kernel's DATASET_PREFLIGHT verifies actual content from inside Kaggle.

## C. Expected sample count
100 (deterministic seed 42).

## D. Actual LOCAL count — VERIFIED PASS
`scripts/validate_kaggle_sample.py --check-et` → 100 cases, 100 seg, 50 flat /
50 UCSD, 94 ET+ / 6 ET−, all 4 modalities + seg per case, IDs == expected, 0 dup.

## E. Expected case IDs
Recorded in `sample_100.json` and the authoritative
`sample_100_manifest.json` (`scripts/build_sample_manifest.py`):
- `case_ids_sha256 = ba58aa9bf42e32d05e631067b14a1507da4d086c11ff7a7cf9cb3575f5ee89e3`
- `manifest_sha256 = 6392c60d70f68c49f0e58cc2aa8b316264cfe376c7ceeb6192e88e73a9dd6914`

## F. How the Kaggle dataset is currently discovered
Kernel globs `/kaggle/input/**/*-seg.nii.gz`, excluding the code copy
(`/GLO-NCA/`, `/working/`), then takes the parent-of-parent as the data root.

## G. How the profiler discovers cases
DATASET_PREFLIGHT builds `disc{cid->{dir,cohort,seg}}` from the discovered segs,
sorted → deterministic `ORDER`. Stages slice `ORDER[:10]`, `[:25]`, `[:100]`.

## H. Silent truncation — FIXED
Previous `min(n, len(ids))` **removed**. `grep -c 'min(n_cases,len' = 0`. Stages
now **fail-closed**: `if n_cases > len(ids): fail_closed(...)`. Preflight aborts
(`sys.exit(2)`) before any model build if discovered != expected (count, seg,
cohort 50/50, ET 94/6, IDs SHA, no dup, no missing files). `fail_closed` appears
11× guarding every invariant.

## I. Production DataLoader — YES
Kernel builds objects via `runner._build_dispatch` (same as `train.py`) and
iterates `torch.utils.data.DataLoader(ds, shuffle=True, batch_size=1,
num_workers=4, pin_memory=cuda, worker_init_fn=_worker_init)` recreated per pass —
**verbatim** `runner.py:434-436`. No persistent_workers (matches production).

## J. Production FocalTverskyCELoss — YES
`from src.losses.LossFunctions import FocalTverskyCELoss`, constructed
`alpha=1-β, β=0.75, γ=1.33, ce_weight=0.5` (matches `runner.py:360`), with the same
empty-region BCE fallback. No BCE substitution.

## K. Production V3 model — YES
Built by `build_v3_from_config` via `_build_dispatch`; the exact production module.

## L. Checkpointing — TRUE
`configs/v3_kaggle_5epoch.yaml` `memory.gradient_checkpointing: true`; kernel reads
`model.gradient_checkpointing` and records it. (Measured-necessary on ≤16 GB GPUs.)

## M. Production source modified — NO
`git status src/` shows 6 pre-existing modified files (earlier-session V3 /
checkpointing work), none touched in this recovery pass. Only profiler/scripts/
reports/packaging created. Canonical split SHA `d30d71956ee9…09559d` intact.

## New deliverables (this pass)
- `scripts/validate_kaggle_sample.py` — local fail-closed sample validator (ran: PASS)
- `scripts/build_sample_manifest.py` — manifest + SHA (ran: manifest_sha `6392c60d…`)
- `scripts/test_preflight_logic.py` — 6 synthetic fail-closed tests (ran: 6/6 PASS)
- hardened kernel `glo_nca_v3_real_profile.py` — DATASET_PREFLIGHT (fail-closed),
  no truncation, per-stage save, PROFILE_PASS labeling (not "epochs"), code identity.

## Config identity (verified)
V3 params 40,656 (runtime-verified in kernel) · L1/L2/L3 = 32/96/128 · 50 NCA steps ·
seed 42 · ckpt TRUE · batch 1 · workers 4.

## NOT changed
Architecture, levels, NCA steps, loss, optimizer, EMA, augmentation, evaluation,
threshold, split, canonical split, V2. Zero epochs of training performed.
