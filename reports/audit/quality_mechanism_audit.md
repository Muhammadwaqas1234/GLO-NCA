# GLO-NCA — Quality Mechanism Audit

> **SUPERSEDED IN ONE RESPECT — read this first.**
> This report describes the sampling experiment as *ET-aware / ET-positive*
> sampling. A later measurement (`reports/analysis/et_voxel_census*`) showed
> that **every measured BraTS-METS training case contains enhancing tumour**,
> so ET-positive oversampling is not meaningful on this cohort. The mechanism
> is **small-lesion-aware sampling**: ET volume is the weighting *signal*, and
> small-lesion emphasis is the *purpose*. The experiment config was renamed to
> `configs/experiments/glo_nca_small_lesion_sampling.yaml`.
> See `reports/audit/phase1_final_local_validation.md` for the corrected
> account. Everything else in this report stands.


Audit of the nine candidate quality mechanisms against the ACTIVE production
training dependency graph. Every classification below is backed by runtime
evidence, not by filenames, config keys, comments or historical versions.

Baseline commit at audit start: `441b142`, working tree clean.

---

## A. Production baseline

**PRODUCTION BASELINE: METHODOLOGICALLY UNCHANGED.**

Verified from the constructed model, not from config text:

| Property | Value |
|---|---|
| Parameters | 29,337 inference / 75 auxiliary / 29,412 training |
| Levels | L1 48³/24ch/15 steps/k=5 · L2 64³/24ch/15 steps/k=5 · L3 disabled |
| Working volume | 128³, patchify OFF, ROI 1.0 |
| Fusion | learned `Conv3d(48 → 24)` |
| Loss | Tversky α=0.40 β=0.60, γ=1.33, ce 0.5, empty-BCE 0.1 |
| Deep supervision | enabled, weight 0.4 |
| Optimisation | LR 0.0016 → 1e-5 cosine, wd 1e-4, EMA 0.999, clip 1.0, BF16 |
| Early stopping | patience 15, min_delta 0.01, max 300 epochs |
| Split | 898 / 200 / 198, seed 42 |

Three changes touched the production side. None alters training methodology:

1. **Dead `sampling.*` keys annotated as INERT.** No value changed; the keys
   are retained so enabling patchify keeps historical behaviour.
2. **Lesion-size stratified evaluation added** — evaluation/reporting only.
3. **`.gitignore` root-anchored** — engineering fix, see section C.

The production sampler remains `_EpochSampler`: **uniform case-level sampling**.

---

## B. Mechanism table

| Mechanism | Status | Production | Experiment | Runtime verified | Speed impact |
|---|---|---|---|---|---|
| MRI intensity normalization | IMPLEMENTED | YES | NO | YES | already in baseline |
| Lesion-size stratified evaluation | IMPLEMENTED (new) | YES | NO | YES | 8.4 ms once per run |
| Lesion-aware case sampling | MISSING → EXPERIMENT | NO | YES | YES | 1.0 ms/epoch measured |
| LR warmup | MISSING → EXPERIMENT | NO | YES | YES | 0 (schedule arithmetic) |
| Boundary-aware loss | **DEFERRED — cost measured, rejected** | NO | NO | n/a | **+198%/epoch measured** |
| Dynamic loss weighting | **DEFERRED — no justification** | NO | NO | n/a | not measured |
| Hard-case mining | **DEFERRED — risk not justified** | NO | NO | n/a | not measured |
| Foreground-aware sampling | merged into lesion-aware sampling | NO | YES | YES | see above |
| Small-lesion oversampling | merged into lesion-aware sampling | NO | YES | YES | see above |
| Class-balanced sampling | merged into lesion-aware sampling | NO | YES | YES | see above |

Per section 10 of the brief, the last three are **one** mechanism expressed as
case weights on a single sampler. No competing sampler was created.

---

## C. Files changed

**Production (methodology-neutral)**

- `configs/glo_nca_production.yaml` — annotated the inert `sampling.*` block;
  documented the stratified-evaluation output. No value changed.
- `src/experiment/runner.py` — imports `lesion_strata`; writes
  `reports/{validation,test}_by_lesion_size.csv` beside the existing per-case
  CSVs. Consumes rows that already exist; adds no model pass.
- `.gitignore` — anchored `experiments/` to `/experiments/`. The bare pattern
  also matched `configs/experiments/` and `docs/experiments/`, which would have
  silently untracked the experimental configs. Run outputs remain ignored.

**New source**

- `src/experiment/lesion_strata.py` — stratified evaluation (production).
- `src/experiment/lesion_sampler.py` — `LesionAwareSampler` (experiment only).
- `src/experiment/warmup.py` — `WarmupCosineLR` (experiment only).

**New configs** — each differs from the baseline by exactly one mechanism,
asserted by test:

- `configs/experiments/glo_nca_small_lesion_sampling.yaml`
- `configs/experiments/glo_nca_lr_warmup.yaml`

No config exists for any deferred mechanism.

**Tests** — `scripts/test_quality_mechanisms.py`.

---

## D. Tests

`scripts/test_quality_mechanisms.py`: **65 / 65 passed, 0 failures.**

Covers stratum boundaries, empty-GT separation, HD95 validity accounting,
CSV schema, sampler weighting/determinism/epoch-length/degenerate inputs,
warmup LR schedule/peak/floor/resume, and 19 production-baseline invariants
including explicit assertions that no experimental key leaked into production.

One assertion was corrected during development: it asserted that
`load_state_dict` writes the LR into the optimizer immediately. PyTorch defers
that to the next `step()`. The scheduler was correct; the test was wrong, and it
now asserts the real contract — that a resumed schedule reproduces the
uninterrupted LR sequence. No test was weakened to obtain a pass.

---

## E. Runtime validation

```
Lesion-aware sampler (experiment arm B):
  sampler class constructed:               YES (LesionAwareSampler)
  weighted sampling active:                YES
  ET-positive cases receive modified prob: YES
  epoch length equals baseline:            YES
  validation sampler unchanged:            YES (uniform, not applied)
  test accessed:                           NO

Production baseline (arm A) still uniform:
  runner constructs _EpochSampler:         YES
  runner imports LesionAwareSampler:       NO
  production config has sampler key:       NO

Lesion-size stratified evaluation (production, eval-only):
  runner imports lesion_strata:            YES
  wired on validation rows:                YES
  wired on test rows:                      YES
  additional model passes:                 NONE
  affects thresholds/checkpoint/stopping:  NO

LR warmup (experiment):
  runner imports warmup:                   NO
  production config has warmup_epochs:     NO
  experiment config warmup_epochs:         3
```

---

## F. Performance (measured only)

Baseline reference: L4, 898 cases, batch 1, 2 workers, BF16, ~0.75 s/iter,
**825 s warm epoch** (measured in the Phase 2 10-epoch run).

| Item | Measured | Method |
|---|---|---|
| Sampler construction (898 cases) | 3.14 ms, once per run | local timing |
| Sampler index draw | 1.00 ms/epoch = 0.00012% of an epoch | median of 20 epochs |
| Stratified evaluation (200 cases) | 8.39 ms, once per run | local timing |
| Warmup | no added work — same step count, different LR value | by construction |
| **Boundary loss EDT (128³, 1 region)** | **608 ms** | median of 5, SciPy |
| **Boundary loss, 3 regions × 898 cases** | **1638 s/epoch → +198%** | derived |
| **Boundary loss over 300 epochs** | **+136.5 GPU-hours** | derived |

Not yet measured and required before arm B runs: the one-off pass computing ET
voxel counts over the 898 training cases.

Boundary-loss timings are single-threaded CPU on the local machine, not the L4
VM; 2 dataloader workers could hide part of the cost. Even a 2x improvement
leaves it far above any other candidate, which is sufficient to defer it.

---

## G. Scientific changes

**Engineering only (no scientific effect)**
- `.gitignore` anchoring.
- Config comments and annotations.

**Evaluation / reporting only (no training effect)**
- Lesion-size stratified evaluation. Adds reporting granularity; cannot alter
  training, threshold tuning, checkpoint selection or early stopping. Existing
  overall validation/test metrics are unchanged.

**Methodology changes — QUARANTINED IN EXPERIMENT CONFIGS, NOT IN THE BASELINE**
- Lesion-aware case sampling (`glo_nca_small_lesion_sampling.yaml`).
- LR warmup (`glo_nca_lr_warmup.yaml`).

No methodology change was applied to `configs/glo_nca_production.yaml`.

---

## H. Deferred mechanisms

**Boundary-aware loss — DEFERRED, cost measured and rejected.**
A surface-distance term needs a Euclidean distance transform per region per
case per epoch: measured at 608 ms per region, 1638 s per epoch, **+198%**, or
+136.5 GPU-hours over 300 epochs. Its expected benefit is to HD95 rather than
Dice. Cost is disproportionate; deferred rather than implemented.

*Correction on the record:* earlier in this session an estimate of +11–18% per
epoch was given for this mechanism. That figure was an unmeasured guess and was
wrong by roughly an order of magnitude. The brief required measurement before
implementation, and the measurement is what changed the decision.

**Dynamic loss weighting — DEFERRED, no defensible justification.**
The baseline already addresses class imbalance three ways: Tversky β=0.60
penalising false negatives, focal γ=1.33, and empty-region BCE 0.1. The brief
permits declining when justification is weak. Adding a scheduled weighting
introduces tuning axes that cannot be defended without experiments there is no
budget to run. **NOT IMPLEMENTED — DEFERRED.**

**Hard-case mining — DEFERRED, risk not justified.**
Requires a periodic training-set-wide scoring pass whose cost was not measured,
and carries a real failure mode: "hard" cases in BraTS-METS correlate with
annotation ambiguity, so mining can amplify label noise. It also creates a
feedback loop between model state and data selection that complicates
reproducibility. **DEFERRED — SCIENTIFIC/ENGINEERING JUSTIFICATION REQUIRED.**

**MRI intensity normalization — NOT RE-IMPLEMENTED (already present).**
Verified at `src/datasets/Nii_Gz_Dataset_3D.py:308-325`: brain-mask-restricted
per-channel z-scoring (`nonzero_norm`, active in production), with a torchio
z-norm + 0.5/99.5 percentile rescale fallback. It runs inside the deterministic
preprocessing head that the on-disk cache stores, before augmentation, which is
applied per epoch outside the cache. Duplicating it would have double-normalised
the inputs.

---

## I. Git status

- Commit at audit start: `441b142`
- New commits: **NONE**
- Working tree: **dirty** (changes present for review, not committed)
- Pushed: **NO**

Per the brief, nothing is committed or pushed until explicitly instructed.

---

## J. A/B design for lesion-aware sampling

**Arm A** — `configs/glo_nca_production.yaml`, unchanged.
**Arm B** — `configs/experiments/glo_nca_small_lesion_sampling.yaml`, identical except
the sampler.

Shared and asserted by test: split, seed 42, optimizer, scheduler, architecture,
loss, augmentation, EMA, checkpoint rules, early stopping, post-processing.
Epoch length is deliberately identical (`len(dataset)` draws), so optimiser-step
count and the cosine schedule are not confounded.

Compare on validation only: WT/TC/ET Dice, mean Dice, HD95, lesion-size
stratified metrics, training time, peak VRAM, convergence behaviour.
**The test split is not touched for intermediate comparison.**

**MEASURED RESULT: NOT YET MEASURED.**

---

## Units caveat for the stratified evaluation

Each case is foreground-cropped to its own brain bounding box and then resampled
to 128³, so the scale factor **differs per case** and no voxel spacing is carried
through the pipeline. Lesion sizes are therefore counted in **resampled voxels**,
which are not a physical volume and are only approximately comparable between
cases.

The repository defines no BraTS-METS size convention, and none was invented. The
default bins (small < 100, medium < 1000, large ≥ 1000 resampled voxels) are
declared as **analysis bins, not clinical categories**, and are configurable.
Cases with no ground truth for a region are placed in a separate `absent_gt`
stratum so they can neither depress nor inflate a size stratum's Dice.

---

## Pre-flight before any GCP spend

Completed locally: unit tests (65/65), config validation, runner import,
sampler activation proof, validation/test isolation proof, cost measurement.

**Still outstanding and blocking a long Spot run:**

1. Small real-data smoke test through `scripts/test_epoch_completes.py` with
   the stratified-evaluation path active.
2. ET voxel-count pass measured on the real 898-case training split.
3. GCS checkpoint write / read-back / resume verification — the brief requires
   an actual upload, object existence check and resume, not a bucket-exists
   check.
4. Spot preemption recovery verification.

Items 3 and 4 were not re-verified in this audit.
