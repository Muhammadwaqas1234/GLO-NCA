# GLO-NCA V3 Real BraTS Dataset Validation & GCP-Readiness Report

**Date:** 2026-09-13 · **Branch:** `v3-multilevel` · **V2 frozen baseline:** `f9e5501`
**Dataset:** MICCAI-LH-BraTS2025-MET-Challenge-Training (BraTS-MET 2025)

> Repository-side (local) work only. No full training, no GCP usage, no dataset
> upload. Real-data results come from the actual production loader / V3 model /
> loss / checkpoint code on real cases; they are **not** scientific metrics.
> GPU-memory fit at 96³/128³ is explicitly **NOT** tested locally — it must run
> on the target GCP GPU.

---

## 1. Environment
Python 3.12.9 · torch 2.5.1+cu121 · CUDA 12.1 · NVIDIA RTX 3050 6 GB (local only,
not the production target) · Windows 11.

## 2. Repository State
Branch `v3-multilevel`, base `f9e5501`. Nothing committed. V2 **methodology core**
(model, agents, losses, evaluation scoring, all V2 configs) is byte-unchanged;
all edits are additive infrastructure (case discovery, split, runner wiring,
cloud scripts) + docs.

## 3. Dataset Scope (decided & implemented)
Each BraTS-MET case/timepoint directory is an **independent segmentation sample**
(timepoints are **not** collapsed — V3 is a single-volume model). Ingestion scope
= **all valid BraTS-MET training case directories**, discovered recursively and
files-validated.

- Top-level cohort: **650** cases
- Nested `UCSD - Training/` cohort: **646** cases (that folder is a container, not a case)
- **Total valid cases: 1296** (discovered from the filesystem, never hardcoded)

## 4. Dataset Structure & Discovery
`src/experiment/datasource.py:discover_cases` walks the root recursively and
accepts a directory as a case only if it directly contains all four modalities +
seg (`{case}-t1n|t1c|t2w|t2f|seg.nii.gz`). Container dirs are skipped; case IDs
are deterministically sorted; duplicate IDs are rejected. Each case resolves to a
path **relative** to the root, so nested cases load correctly. For a flat dataset
this returns exactly the old top-level mapping (backward-compatible — verified by
the V2 regression).

## 5. Dataset Case Count
```
VALID CASES = 1296   (top-level 650 + nested UCSD 646; 'UCSD - Training/' = container)
```

## 6. Modality Validation — PASS
Cases contain t1n/t1c/t2w/t2f + seg (modern hyphen naming). The loader/validator
already support this convention. No methodology change.

## 7. Label Validation — PASS
Real seg labels observed `{0,1,2,3}` ⊂ allowed `{0,1,2,3,4}`. ET=3 (BraTS-MET);
`_labels_to_regions` derives nested WT/TC/ET correctly (verified on real cases).
No new mapping invented.

## 8. NIfTI Integrity — PASS (sampled + validator)
Sampled flat + nested cases: valid NIfTI, per-case MRI/seg shapes consistent, no
NaN/Inf, sane intensities. Cross-cohort in-plane size differs (240²×155 vs
512²×184) — handled by the pipeline's resize at load time. `validate_dataset` is
now recursion-aware and reports top-level vs nested counts.

## 9. Master Split — created, subject-disjoint, verified
`scripts/create_master_split.py` (seed 42, 70/15/15) now builds a
**SUBJECT-disjoint** split via `datasource.make_split`/`build_master_split`, which
group cases by base subject and assign whole subjects to a partition.

```
cases:    1296  -> train 898 | val 200 | test 198
subjects:  810  -> train 567 | val 121 | test 122
split_sha256:    d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d
patient_id_hash: 7e9ff2cf8042ff0071b804c441cddd31dd8979de2bb1886b4a9107d6c8e4cb47
grouping:        subject-disjoint (base id = case id minus -<timepoint>)
```
`scripts/check_split.py` verifies pairwise-disjointness, fingerprint, AND
subject-disjointness (0 straddling subjects).

## 10. Multi-Timepoint & Subject Semantics
Case ID = `BraTS-MET-<subject>-<timepoint>`. Stripping `-<timepoint>` gives the
base subject (`datasource.subject_of`). 1296 cases → **810 subjects**; 287
subjects have >1 timepoint (up to 5). IDs do **not** have the timepoint stripped
for the sample identity (each timepoint remains a distinct case). The two cohorts
have **no base-subject overlap and no case-id collisions**.

## 11. Data-Leakage Gate — PASS
The build fails loudly if any subject straddles partitions, and the verifier
re-checks it. Result: **0 straddling subjects → no temporal leakage.** Rule
documented in `split/README.md` and here.

## 12. Dataset Identity — recorded
The runner records, per experiment: `dataset_case_count` (1296, case/timepoint
dirs — not collapsed), `subject_count` (810), `patient_id_hash` (over case IDs),
`id_semantics`, and `split.split_sha256`. Terminology uses **case_id** (timepoint)
vs **subject** explicitly; no false claim of longitudinal collapsing.

## 13. Real-Data V3 Software Path — PASS (from prior real-data validation)
Preprocessing → forward → loss+backward (grad-clip, optimizer) → checkpoint →
resume (epoch 2→3) → evaluation (Dice/mIoU/HD95 in voxels) → validation-only
threshold tuning, all executed on real BraTS-MET cases via production code.
Peak VRAM at 32³ true-fit = 0.243 GB (local). 96³/128³ NOT run locally.

## 14. V3 Production Config — PASS
`configs/v3_multilevel.yaml`: v3; level1/2/3 enabled @ 32/96/128³ (global →
regional → fine); SE + spatial GC on; learnable concat fusion; batch 1; seed 42;
light aug; 200 epochs; patch_size 128 (= finest level; the runner derives lower
levels by downsampling — no contradiction with per-level resolutions; V3 configs
bypass the V2 patch-table via the additive `config.py` v3 branch); loss β0.75/γ1.33;
references `split/master_split.json`. Measured params = **40,656**.

## 15. V3 Ablation Matrix — expressible & constructs (documented, NOT run)
| Config | Levels | Params | Fusion |
|---|---|---|---|
| `v3_ablation_A_global96.yaml` (Global+96) | 2 | 33,089 | concat |
| `v3_ablation_B_global128.yaml` (Global+128) | 2 | 28,223 | concat |
| `v3_ablation_C_add_fusion.yaml` (3-level, additive fusion) | 3 | 39,872 | add |
| `v3_multilevel.yaml` (V3-D, full) | 3 | 40,656 | concat |
All share the master split, seed 42, loss, evaluation, threshold protocol.

## 16. Loader / Infrastructure Changes (documented)
- `datasource.py`: `discover_cases`/`discover_case_ids`/`case_path_map`/`subject_of`;
  `list_patients` → recursive; `make_split` → subject-disjoint; `build_master_split`
  → subject metadata + hard leakage gate.
- `dataset_validation.py`: recursive discovery, cohort counts.
- `Nii_Gz_Dataset_3D.py`: `getFilesInPath` → recursive, relative paths (discovery
  only; preprocessing/labels/`__getitem__` **unchanged**).
- `runner.py`: case→relative-path map so nested cases load; subject count in identity.
- `check_split.py`: subject-disjointness check.
- `cloud/scripts/pretrain_gate.sh`: V2+V3 aware (V3 param report + 96³/128³ GPU gate).
- `cloud/scripts/upload_dataset.sh`: recursive verification by discovered cases; preserves `UCSD - Training/`.
- `scripts/gpu_memory_gate_v3.py`: NEW production 96³/128³ memory probe.

## 17. V2 Preservation — PASS
`git diff f9e5501` on V2 model/agents/losses/metrics/configs = empty. V2 param
count 30,138. V2 smoke runs end-to-end through the shared runner (COMPLETED).

## 18. GCP Readiness (what is done vs still required on GCP)
**Done locally (repository-side):** ingestion policy, recursive discovery,
subject-disjoint canonical split + verification, dataset identity, runner wiring,
V3 configs + ablations, V2+V3 pre-train gate, recursive-safe transfer scripts,
GPU-memory gate script, compile/parse/regression tests.

**Still required — GCP only (must NOT be faked):**
- GPU memory gate at 96³ **and** 128³ on the real GCP GPU (`scripts/gpu_memory_gate_v3.py`).
- `./cloud/scripts/pretrain_gate.sh configs/v3_multilevel.yaml` on the VM (real-data smoke, checkpoint/resume, GCS round-trip).
- Dataset upload + integrity verification to GCS.

## Local vs GCP separation
```
LOCAL (done): dataset discovery, master split, identity, V3 runner integration,
              config/ablations, cloud scripts, pre-train gate wiring, tests.
GCP  (todo) : 96³/128³ GPU memory fit, real-data GCP smoke, GCS transfer, then
              ablations + final 200-epoch V3 training.
```

---

```
=============================================
GLO-NCA V3 — GCP READINESS IMPLEMENTATION
=============================================
Dataset scope:               ALL VALID BraTS-MET TRAINING CASES
Top-level:                   650
UCSD:                        646
Total:                       1296

Case discovery:              PASS  (recursive, files-validated, dedup)
Modality validation:         PASS
Label validation:            PASS  ({0,1,2,3}; WT/TC/ET derived)
Longitudinal handling:       PASS  (each timepoint = independent case)
Subject leakage protection:  PASS  (subject-disjoint; 0 straddling subjects)
Master split:                PASS  (898/200/198; subjects 567/121/122; seed 42)
Split SHA256:                d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d
Dataset identity:            PASS  (case_count 1296, subject_count 810)
V3 production config:         PASS  (40,656 params; 32/96/128; concat fusion)
V3 runner:                   PASS  (V2+V3 dispatch; nested paths load)
V2 regression:               PASS  (30,138 params; byte-unchanged; COMPLETED)
Cloud scripts:               PASS  (recursion-safe; V3-aware gate; 96/128 probe)
GCP pretrain gate:           READY (wired; runs on VM)
Local 128³:                  NOT REQUIRED
GCP GPU memory:              NOT TESTED — MUST RUN ON GCP (96³ + 128³)
Full training:               NOT RUN
=============================================
FINAL STATUS: READY FOR GCP PRE-FLIGHT
=============================================
NEXT STEP:
Commit the V3 branch + canonical master split (user decision), then on the GCP
GPU VM run `scripts/gpu_memory_gate_v3.py --config configs/v3_multilevel.yaml`
and `./cloud/scripts/pretrain_gate.sh configs/v3_multilevel.yaml`. Only if the
96³ + 128³ memory gate PASSES on the real GPU proceed to ablations + final V3
training. Do NOT start GCP training automatically.
```
