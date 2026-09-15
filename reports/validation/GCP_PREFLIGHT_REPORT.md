# GLO-NCA V3 — Overnight GCP Pre-Flight Report

**Run:** autonomous overnight pre-flight · **Date:** 2026-09-14 (UTC)
**Project:** `even-continuity-501915-f9` · **Region/zone:** us-central1 / us-central1-a
**Branch:** `v3-multilevel` · **Frozen V2 base:** `f9e5501`

> Outcome: **BLOCKED on GPU quota.** No GPU VM could be created, so all
> GPU-dependent gates (memory 96³/128³, real-data smoke, checkpoint/resume on GPU,
> pretrain gate) could **not** run. Nothing was faked. **$0 compute spent** (no VM
> ever ran). Scientific methodology untouched; master split verified intact.

---

## THE BLOCKER (quota — requires your action)
Creating the L4 VM fails with:
```
Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.
```
Google enforces two GPU quotas; this project has:
| Quota | Value |
|---|---|
| `GPUS_ALL_REGIONS` (global cap) | **0** ❌ (must be ≥ 1) |
| `NVIDIA_L4_GPUS` (us-central1) | 1 ✅ |
| `NVIDIA_A100_GPUS` | 0 |

With the **global** cap at 0, **no GPU of any type/region can be created.** Per the
task rules (Step 3, §19) this is a QUOTA failure → **STOP, do not retry, do not
waste money.** I attempted creation once (to surface the exact error), it failed
pre-billing, and I did not retry.

### What you must do (account-level, I cannot)
1. Console → **IAM & Admin → Quotas** (project `even-continuity-501915-f9`).
2. Filter **"GPUs (all regions)"** (`GPUS_ALL_REGIONS`) → request limit **1**.
3. Confirm **"NVIDIA L4 GPUs"** in **us-central1** ≥ 1 (already 1).
4. Submit (justification: "single L4 for a thesis segmentation run").
5. If auto-rejected: **Billing → Upgrade account** to a full paid account (free-trial
   accounts often cannot get GPU quota), then re-request.

Lead time: minutes–hours typically; up to ~2 business days. You'll get an email.

---

## What was completed overnight (cost-free, verified)

### 1–2. Project / region / config
- Auth OK (`mo.waqas@techclomate.com`), billing enabled, project set.
- `cloud/config/gcp.env` created (git-ignored): L4 / `g2-standard-8` / us-central1-a,
  bucket `glo-nca-v3-even-continuity-501915`. Image family corrected to the current
  **`common-cu129-ubuntu-2204-nvidia-580`** (the old `common-cu121-debian-11` was
  retired by Google; host driver 580 is backward-compatible with our CUDA-12.1 image).

### 3. GPU quota — **BLOCKED** (see above).

### 4/6. Existing-resource + cost audit — **$0 compute**
- **No GLO-NCA VM exists** → zero GPU billing from this work.
- Unrelated resources in the project (`copilot-*`, `voice-*` VMs [TERMINATED],
  `voice-prod-ip`) belong to your other work — **left untouched** per policy.
- Bucket `gs://glo-nca-v3-even-continuity-501915` exists with layout
  (`datasets/`, `experiments/`, `logs/`, `backups/`).

### 5. VM — **NOT CREATED** (quota).

### 7. Dataset in GCS — **INCOMPLETE, do NOT trust**
The earlier laptop `rsync` authenticated intermittently ("Anonymous caller" 401s)
and left a **partial** copy:
```
files in GCS:            5544  (expected 6480)
COMPLETE cases (5 files): 831
incomplete cases:         450
cases entirely missing:    15
=> 465 complete cases short of 1296
```
This partial data is **kept** (a resumed transfer will fill only the missing files,
efficiently) but must be **completed + exhaustively validated before any training**.
**Exhaustive 1296-case validation: NOT RUN** (needs the complete dataset; deferred
to the VM as planned). Best next path: **download from your Google Drive folder
directly on the VM** (datacenter bandwidth, VM service-account auth — avoids the
laptop's slow/flaky upload entirely).

### 8. Master split — **PASS (verified)**
```
SHA256:   d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d  ✓ matches
cases:    train 898 | val 200 | test 198  (total 1296)
subjects: train 567 | val 121 | test 122  (total 810)
subject leakage: 0   |  all partitions pairwise-disjoint
```
Not regenerated, not modified.

### 9–14. GPU/CUDA/PyTorch, memory gate, smoke, checkpoint/resume-on-GPU, pretrain gate
**NOT RUN — all require the GPU VM, which is blocked on quota.** Not faked.

### 15. Resource/cost record
`gcp_preflight/resource_usage.{json,csv}` — compute cost **$0** (no VM ran),
storage ~cents. GPU/VRAM/CPU/RAM utilization = NOT AVAILABLE (no GPU).

---

## Ready-to-go (the moment quota is granted)
- Repo pre-flight complete (V3 40,656 params; 32/96/128; SE+spatial; concat fusion;
  300 epochs; loss β0.75/γ1.33; master split referenced) — verified in prior audits.
- `cloud/config/gcp.env` valid; bucket created; image family fixed.
- Plan: create L4 → download dataset from Drive on the VM → exhaustive validate all
  1296 → memory gate (96³+128³ TRUE FIT) → real-data smoke → checkpoint/resume →
  `pretrain_gate.sh` → report → STOP (no training).

---

```
============================================================
GLO-NCA V3 — OVERNIGHT GCP PRE-FLIGHT FINAL REPORT
============================================================
Project:                even-continuity-501915-f9
Region:                 us-central1
Zone:                   us-central1-a
VM:                     NOT CREATED (blocked: GPU quota)
GPU:                    nvidia-l4 (intended, 24 GB)
VRAM:                   NOT MEASURED (no GPU)

VM created:             —
VM running:             —
VM stopped/deleted:     — (never created)

Dataset:                INCOMPLETE in GCS (831/1296 complete cases; needs
                        completion via VM Drive download, then validation)
Cases:                  1296 expected (831 complete in GCS)
Subjects:               810 (per master split)

Master split:           VERIFIED
Master split SHA:       d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d
Subject leakage:        0

CUDA:                   NOT MEASURED (no GPU VM)
PyTorch:                NOT MEASURED (no GPU VM)
Docker:                 NOT MEASURED (no GPU VM)

96³ TRUE FIT:           NOT RUN (blocked on GPU)
128³ TRUE FIT:          NOT RUN (blocked on GPU)

Real-data smoke:        NOT RUN (blocked on GPU)
Checkpoint:             NOT RUN (blocked on GPU)
Resume:                 NOT RUN (blocked on GPU)
Evaluation:             NOT RUN (blocked on GPU)
GCS durability:         bucket OK; dataset partial; report synced (see below)

Pre-flight runtime:     ~cost-free steps only
GPU runtime:            0 s
Estimated compute cost: $0.00
Storage cost:           ~cents (bucket + partial dataset)
Other cost:             NOT AVAILABLE
Total known cost:       $0 compute; ~cents storage

Resource usage report:  gcp_preflight/resource_usage.{json,csv}
Cost report:            included above (compute $0)
Audit report:           this file

============================================================
FINAL PRE-TRAINING GATE:  FAIL (blocked — not run)
GLO-NCA V3 TRAINING:      NOT READY
============================================================
BLOCKER: GPUS_ALL_REGIONS quota = 0 → request GPU quota ≥ 1 (and upgrade
billing if needed). Then re-run this pre-flight; it will create the L4,
complete + validate the dataset, and run all GPU gates.
```
