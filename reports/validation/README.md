# Validation & pre-flight reports (development record)

**These are engineering/operational validation reports produced while preparing
the GLO-NCA V3 experiment. They are NOT the final scientific thesis results.**
The final scientific results (Dice/mIoU/HD95 on the frozen test set) will be
produced only by the 300-epoch training run defined in
[`docs/thesis/FINAL_TRAINING_PROTOCOL.md`](../../docs/thesis/FINAL_TRAINING_PROTOCOL.md).

Read them newest-first; later reports supersede earlier ones where they overlap.

| Report | What it establishes | Status |
|---|---|---|
| `V3_GRADIENT_CHECKPOINTING_GPU_GATE.md` | **Decisive:** production V3 128³ forward+backward **TRUE FITS the NVIDIA L4 24 GB** with gradient checkpointing (9.07 GB peak). Self-verified (checkpointing proven active). | current |
| `V3_GRADIENT_CHECKPOINTING_REPORT.md` | Implementation + bit-identical equivalence (OFF vs ON) + ~22.5× activation-memory reduction of the opt-in, memory-only gradient checkpointing. | current |
| `VRAM_STUDY.md` | Measured VRAM of the **unchanged** V3 (no checkpointing): 32³/64³ fit; 96³/128³ OOM on L4; 128³ ≈ 87–101 GB without checkpointing. Motivates the checkpointing work. | historical (measurement) |
| `REAL_DATA_V3_VALIDATION_REPORT.md` | Real BraTS-MET data: recursive discovery (1296 cases / 810 subjects), subject-disjoint master split, V3 real-data forward/backward/checkpoint/resume/eval on the software path. | historical (development) |
| `FINAL_GCP_READINESS_REPORT.md` | Repository-side GCP-readiness audit + low-cost deployment plan. | historical (development) |
| `GCP_PREFLIGHT_REPORT.md` | Earliest overnight pre-flight; documented the (since-resolved) GPU-quota blocker. | historical (superseded) |
| `V3_3_EPOCH_SMOKE_TEST_REPORT.md` | Records that the 3-epoch smoke test was **blocked** before checkpointing existed (unchanged V3 could not fit any available GPU). Superseded by the GPU-gate PASS above. | historical (superseded) |
| `resource_usage.{json,csv}`, `gcp_resource_usage.json` | Recorded GCP resource/cost bookkeeping during pre-flight. | historical (evidence) |

The authoritative, current facts are summarised in the top-level
[`README.md`](../../README.md) status block and in
[`docs/architecture/GLO_NCA_V3_ARCHITECTURE.md`](../../docs/architecture/GLO_NCA_V3_ARCHITECTURE.md).
