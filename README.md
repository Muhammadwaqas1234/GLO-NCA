# GLO-NCA: Global Context-Aware Neural Cellular Automata for Brain Tumor Segmentation

> **MS Thesis — Muhammad Waqas, Air University, Islamabad.**
> *For academic and educational use only — not a medical device (see [LICENSE](LICENSE)).*

GLO-NCA is a **lightweight, global-context-aware Neural Cellular Automata (NCA)**
architecture for **multi-modal 3D brain-tumor segmentation** on BraTS. A plain
NCA only communicates between neighbouring cells; GLO-NCA augments it with
**Squeeze-and-Excitation (SE) channel context** and a **spatial global-context**
block so the model reasons about whole-volume tumor location while remaining
extremely small (**40,656 parameters** for the V3 model).

---

## Status
```
Repository:            READY FOR SUPERVISOR REVIEW
Final architecture:    GLO-NCA V3 (multi-level, 32³ → 96³ → 128³)
Final config:          configs/v3_multilevel_ckpt.yaml   (300 epochs)
V3 parameters:         40,656   (measured)
V2 baseline:           30,138   (frozen, preserved)
Master split:          subject-disjoint, frozen (SHA256 d30d7195…09559d)
GPU validated:         NVIDIA L4 24 GB — 128³ TRUE FIT with gradient checkpointing
3-epoch smoke test:    PENDING (blocked on dataset transfer to the training node)
Final 300-epoch run:   NOT STARTED
```
No scientific segmentation results (Dice/mIoU/HD95 on the test set) are claimed
until the final 300-epoch run defined in
[`docs/thesis/FINAL_TRAINING_PROTOCOL.md`](docs/thesis/FINAL_TRAINING_PROTOCOL.md)
is executed. Engineering/operational validation to date is recorded under
[`reports/validation/`](reports/validation/).

---

## Research objective
Deep 3D segmentation networks (U-Net, nnU-Net, Swin-UNETR, UNETR) achieve strong
BraTS accuracy but carry millions of parameters and heavy compute, limiting use
on low-resource hardware. **GLO-NCA investigates whether a tiny NCA, given an
inexpensive global-context mechanism and a hierarchical multi-scale refinement,
can produce competitive multi-region tumor segmentation at a fraction of the
parameter budget.**

## Scientific contribution
- A **global-context-aware NCA cell** — SE channel attention (`use_attention`)
  plus a spatial global-context block (`use_spatial`) injecting whole-volume
  context at negligible parameter cost.
- **GLO-NCA V3**, a *single unified* model with **three nested NCA levels**
  (global → regional → fine) connected by **learnable feature projections** and a
  **learnable concatenation fusion** head — not an ensemble.
- A reproducible, thesis-grade experiment harness (fixed subject-disjoint split,
  validation-only threshold tuning, single-pass frozen-test evaluation).

## Dataset
**BraTS-MET 2025 (MICCAI-LH BraTS-MET Challenge, Training set).** Enumerated by
recursive, files-validated case discovery (`src/experiment/datasource.py`):
- **1,296 valid cases** across **810 subjects** (287 subjects have >1 timepoint).
- Two cohorts: 650 top-level cases + 646 in a nested `UCSD - Training/` sub-cohort.
- **Subject-disjoint** master split (no timepoint of a subject leaks across
  partitions): **train 898 / val 200 / test 198 cases** (567 / 121 / 122 subjects).
- Split fingerprint (SHA256): `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d`.

Input modalities (BraTS-MET naming): **t1n (T1), t1c (T1ce), t2w (T2), t2f (FLAIR)**.
Output regions (nested, multi-label sigmoid): **WT** (Whole Tumor),
**TC** (Tumor Core), **ET** (Enhancing Tumor).

## GLO-NCA V3 architecture
```
Multi-modal MRI (T1, T1ce, T2, FLAIR)
        │
        ▼
Level 1 — Global      (low-resolution full-volume context; GLO-NCA + SE + spatial GC)
        │  learnable projection + upsample
        ▼
Level 2 — Regional    (96³; GLO-NCA context refinement)
        │  learnable projection + upsample
        ▼
Level 3 — Fine        (128³; GLO-NCA boundary refinement)
        │
        ▼
Learnable feature fusion (per-level projection → concat → 1×1×1 fuse conv)
        │
        ▼
WT / TC / ET
```
Full detail: [`docs/architecture/GLO_NCA_V3_ARCHITECTURE.md`](docs/architecture/GLO_NCA_V3_ARCHITECTURE.md).
**Measured parameter count: 40,656.**

## Memory optimization (implementation-level, not architectural)
The 128³ level's backward pass is activation-heavy. GLO-NCA V3 supports **opt-in
gradient checkpointing** (`memory.gradient_checkpointing`, default **OFF**) that
recomputes each NCA step's activations during backprop instead of storing them.

> Gradient checkpointing is **not** an architectural contribution. The model,
> layers, channels, NCA steps, resolutions, inputs, outputs, loss and optimizer
> are **unchanged**; only backward-pass activation memory is reduced. Verified
> **bit-identical** outputs/gradients OFF vs ON, ~22.5× activation-memory
> reduction, and a measured **128³ TRUE FIT at 9.07 GB on an NVIDIA L4 24 GB**
> (see [`reports/validation/V3_GRADIENT_CHECKPOINTING_GPU_GATE.md`](reports/validation/V3_GRADIENT_CHECKPOINTING_GPU_GATE.md)).

## Training configuration
- **Final training target: 300 epochs** — `configs/v3_multilevel_ckpt.yaml`
  (batch 1, seed 42, light augmentation, checkpointing ON).
- **3-epoch smoke test: operational validation only** —
  `configs/v3_smoke_3epoch.yaml` (derived from the final config, `epochs: 3`).
  Its metrics are **not** scientific results and do not indicate convergence.

Loss **Focal-Tversky + BCE** (β=0.75, γ=1.33); optimizer **AdamW**; **cosine LR**;
**EMA**; **gradient-norm clipping**. Values in
[`docs/thesis/FINAL_TRAINING_PROTOCOL.md`](docs/thesis/FINAL_TRAINING_PROTOCOL.md).

## Evaluation
Per-region **Dice**, **mIoU**, **HD95**. **HD95 is reported in voxels on the
resampled grid** (no physical-space/mm conversion is implemented). Multi-label
sigmoid (WT/TC/ET), single clean inference — **no ensemble, no TTA**.

## Experimental discipline
Fixed subject-disjoint master split (never regenerated); **validation-only**
threshold tuning; **frozen test set** evaluated **once** after thresholds are
frozen; no test leakage; smoothed best-epoch model selection; EMA; full
checkpoint/resume (model, optimizer, scheduler, EMA, epoch, best score, RNG).

## Hardware (validated)
- Local **RTX 3050 6 GB**: sufficient for CPU-level software validation only;
  cannot fit the production 96³/128³ backward pass (expected).
- **NVIDIA L4 24 GB (GCP)**: directly validated. Unchanged V3 at 128³ needs
  ~87–101 GB (OOM on L4); **with gradient checkpointing the 128³ training step
  fits at ~9.07 GB peak** — no A100/H100/H200 required. This is a measured
  result, not an estimate.

## Documentation map
```
README.md                                   ← you are here
CODE_GUIDE.md                               ← code walkthrough
docs/thesis/PROJECT_OVERVIEW.md             ← academic overview
docs/thesis/FINAL_TRAINING_PROTOCOL.md      ← exact final experiment
docs/architecture/GLO_NCA_V3_ARCHITECTURE.md← architecture detail
docs/reproducibility/                       ← runbook & reproducibility notes
reports/validation/                         ← development/pre-flight records (historical)
cloud/README.md                             ← GCP operations
split/README.md                             ← master-split policy
archive/                                    ← historical experiment scripts (not used)
```

## Reproducibility (quick start)
```bash
# Python 3.12
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-docker.txt

# software validation (synthetic; no dataset needed)
python scripts/validate_v3_local.py --device cpu

# with the real dataset available, verify the frozen split:
python scripts/check_split.py --split split/master_split.json --data-root /path/to/BraTS-MET
```
Full protocol and GCP workflow: `docs/thesis/FINAL_TRAINING_PROTOCOL.md` and
`cloud/README.md`.

## Limitations (honest)
- No final segmentation accuracy is available yet — the 300-epoch run has not been
  executed; do not read development/pre-flight reports as scientific results.
- HD95 is in voxels on the resampled grid, not millimeters.
- Real-data training requires a ≥~10 GB GPU with checkpointing (validated on L4);
  the local 6 GB GPU is for software checks only.
- GLO-NCA targets parameter/compute efficiency; it is not claimed to beat large
  U-Net/Transformer models on absolute accuracy.

## Acknowledgements & references
Built on the open-source **Med-NCA / M3D-NCA** framework by John Kalkhof et al.
(MIT-licensed). The global-context design, the V3 multi-level architecture, the
BraTS multi-modal/multi-label pipeline, and the evaluation are the thesis
contributions.
- Kalkhof et al., *Med-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata*, IPMI 2023.
- Kalkhof & Mukhopadhyay, *M3D-NCA: Robust 3D Segmentation with Built-In Quality Control*, MICCAI 2023.

## License
MIT, for **academic/educational use only** — see [LICENSE](LICENSE). Not for clinical use.
