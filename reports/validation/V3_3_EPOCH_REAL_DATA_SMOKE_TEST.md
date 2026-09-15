# GLO-NCA V3 — 3-Epoch Real-Data Smoke Test

## Purpose
Operational validation (NOT the final scientific training) of the full V3
production pipeline on **real** BraTS-MET data for exactly 3 epochs, using
`configs/v3_smoke_3epoch.yaml` (derived from the final config; `epochs: 3`,
gradient checkpointing ON, everything else identical). Its metrics would be
software/operational diagnostics only — never evidence of convergence or final
accuracy.

## Status: NOT RUN — BLOCKED (dataset not available on a training node)

The 3-epoch run is **real-data** by definition (it must exercise real dataset
discovery, all four modalities, labels, and master-split coverage). At the time
of this repository preparation, **no validated real BraTS-MET dataset is present
on a GPU training node**:

- The GCS dataset bucket created earlier is no longer present (returned 404).
- The direct Google-Drive download onto the VM (`gdown --folder`) returned
  **SEG_COUNT = 0** — Google's public-folder limits prevented fetching the
  1,296-case nested dataset; no cases landed.

Per the project's strict rules (do not fabricate metrics; do not present
synthetic data as real; do not hide failures), the smoke test is **not** run on
synthetic data and **not** reported with invented numbers.

Importantly, this is a **data-transfer blocker, not a model/GPU/repository
blocker**. The GPU capability is already proven:
- **128³ forward+backward TRUE FIT at ~9.07 GB on NVIDIA L4 24 GB** with the
  opt-in, memory-only gradient checkpointing (self-verified; see
  `V3_GRADIENT_CHECKPOINTING_GPU_GATE.md`).
- V3 real-data software path (preprocessing → forward → loss → backward →
  checkpoint → resume → evaluation) passed earlier on real cases at reduced
  resolution (`REAL_DATA_V3_VALIDATION_REPORT.md`).

## What must happen before this test can run
1. Land the complete real BraTS-MET dataset on the training node via a reliable
   method (the public `gdown --folder` path is insufficient):
   - `rclone` with a configured Google Drive remote (handles large folders), or
   - a fixed/authenticated `gdown`/service-account path, or
   - re-upload to GCS from a machine with working auth, then `cache_dataset.sh`.
2. `python scripts/check_split.py --split split/master_split.json --data-root <DATA>`
   → split covers 1,296 cases; SHA256 `d30d71956ee9…09559d` matches.
3. `python scripts/gpu_memory_gate_v3.py --config configs/v3_multilevel_ckpt.yaml`
   → 96³ and 128³ TRUE FIT (already demonstrated on L4).
4. Launch exactly 3 epochs: `run_training.sh configs/v3_smoke_3epoch.yaml`,
   record the per-epoch metrics table (loss, Dice/IoU/HD95 per region, LR, epoch
   time, GPU peak), checkpoint/resume verification, threshold discipline, and GCS
   sync — then STOP the VM.

## Environment / config / split (to be recorded when run)
- GPU / VRAM / CUDA / PyTorch / git commit / config identity — recorded at run time.
- Dataset identity: case_count 1296, subject_count 810.
- Master split SHA256: `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d`.

## Decision
```
3-EPOCH SMOKE TEST:        NOT RUN — BLOCKED (real dataset not present on training node)
FINAL 300-EPOCH TRAINING:  BLOCKED (gated behind the 3-epoch operational smoke test)
```
Repository, model, config, split, and GPU-fit are all READY; the only missing
prerequisite is a reliable transfer of the real dataset to the training node.
