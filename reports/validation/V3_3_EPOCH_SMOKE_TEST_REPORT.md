# GLO-NCA V3 — 3-Epoch Real-Data Smoke Test: BLOCKED at Step 1 (GPU memory gate)

**Date:** 2026-09-14 · **Model:** production V3 (UNCHANGED) · configs/v3_multilevel.yaml
**Outcome:** **NOT RUN.** The mandatory Step-1 GPU memory gate cannot pass on any
GPU available to this project, so per the task's own hard-STOP rule the smoke
test was not started. **No VM started for this task; $0 GPU spend.**

## Why it is blocked (not a workaround-able issue)
The 3-epoch smoke test executes the **same production training step** (forward +
backward at 128³, batch 1) as the 300-epoch run — fewer epochs do **not** reduce
per-step memory. That step was measured to OOM on the L4 and to need far more VRAM
than any available GPU:

Measured on NVIDIA L4 (23.66 GB), unchanged production V3:
| L3 res | forward+backward | fit? |
|---|---|---|
| 32³ | 1.575 GB | ✅ |
| 64³ | 10.93 GB | ✅ |
| 96³ | OOM (≥22.5 GB) | ❌ |
| 128³ | OOM (≥22.6 GB) | ❌ |

Estimated unchanged-128³ training requirement: **~87–101 GB** (from VRAM_STUDY.md).

## Available GPUs on project `even-continuity-501915-f9` (measured this task)
- Global cap `GPUS_ALL_REGIONS = 1`.
- Quota exists ONLY for: K80, P100, P4, T4, V100, **L4** — all **≤24 GB**.
- **A100 / H100 / H200 quota = 0** (none can be created).

**Conclusion:** no currently-provisionable GPU can TRUE-FIT the unchanged V3 128³
training step. The L4 (largest available, 24 GB) OOMs at 96³ and 128³.

## What was NOT done (correctly, per rules)
- Did NOT start a GPU VM for a run guaranteed to OOM on batch 1.
- Did NOT modify the V3 architecture / resolutions / channels / NCA steps.
- Did NOT enable gradient checkpointing.
- Did NOT start the 300-epoch training.
- Did NOT regenerate the master split or touch the dataset.

## Required before this smoke test can run (your decision — account/methodology level)
Exactly one of:
1. **Provision a ≥~110–128 GB GPU** — request **H200 141 GB** quota (or multi-GPU
   with activation sharding). A100/H100 **80 GB is NOT sufficient** for unchanged 128³.
2. **Approve a memory-only optimization** (gradient checkpointing across NCA steps)
   — identical model math/outputs/loss; edits `Model_GLO_NCA_V3.py`. This would
   very likely fit an A100/H100 80 GB. Requires your explicit approval (kept out
   of all measurements so far).
3. Re-scope the smoke test to a resolution that fits an available GPU — but that
   is a methodology change to the production config, which is frozen; **not done**.

Separately, before any real-data run: the earlier Google-Drive download produced
**SEG_COUNT=0** (gdown could not fetch the 1,296-folder public dataset). The
dataset must be completed on the VM (rclone, or a fixed gdown/auth path) — a
blocker to resolve once a suitable GPU exists.

```
========================================
GLO-NCA V3 3-EPOCH SMOKE TEST
========================================
GPU:                         none suitable available (L4 24GB largest; A100+ quota=0)
VRAM:                        L4 = 23.66 GB (insufficient)
Production 128³ TRUE FIT:    FAIL (needs ~87–101 GB; measured OOM on L4)
Pre-training gate:           NOT RUN (blocked at Step 1)
Epochs completed:            0 (smoke test not started)
Average epoch time:          N/A
Peak VRAM:                   N/A (no run)
Checkpoint:                  N/A
Resume:                      N/A
GCS durability:              N/A
300-epoch training:          NOT STARTED
GPU VM:                      TERMINATED (glo-nca-v3-l4; $0 GPU billing)
Estimated 300-epoch GPU time: cannot estimate (no epoch measured; blocked)
Estimated 300-epoch GPU cost: cannot estimate (blocked)
FINAL SMOKE TEST:            FAIL — BLOCKED (no GPU can fit unchanged V3 128³)
========================================
```
