# GLO-NCA — Phase 1 Final Local Validation

> **SUPERSEDED — historical record. Read this first.**
> This report describes the production baseline as it stood at the end of
> Phase 1. Three later decisions changed it, so its identity and mechanism
> sections no longer describe production:
>
> | | this report | production now |
> |---|---|---|
> | spatial GC kernel | 5 | **7** |
> | inference / training params | 29,337 / 29,412 | **30,209 / 30,284** |
> | warmup | OFF | **3 epochs** |
> | small-lesion sampling | experiment only | **ON in production** |
> | experiment configs | 2 | **none** (removed) |
>
> The current production facts are in the repository `README.md` and in
> `configs/glo_nca_production.yaml`. Paths below that start with `reports/`,
> `docs/` or `scripts/test_*` now live under `extra/`.

Closes Phase 1 (local). Phase 2 (GCP) is deliberately not started: no VM was
created, no L4 rented, no service-account test run, no Spot test on a real VM,
no 300-epoch training.

Repository at `441b142`, working tree uncommitted and ready for review.

---

## A. Phase 1 status

**PASS.**

Every local check passes and the production baseline is methodologically
unchanged. One item is **deferred to Phase 2 by design** (full ET census, see
section D) and one is **blocked off-VM by design** (GCS service-account
permission, see section K). Neither is a local failure.

---

## B. Test totals

| Suite | Result |
|---|---|
| quality mechanisms | 65 / 65 |
| architecture identity | 34 / 34 |
| production builder identity | 13 / 13 |
| post-processing | 10 / 10 |
| per-case diagnostics | 17 / 17 |
| checkpoint / resume | 22 / 22 |
| EMA validation | 11 / 11 |
| patchify OFF | 8 / 8 |
| preprocess cache | 7 / 7 |
| Spot recovery (local legs) | 4 / 4 + 4 skipped |
| experiment validation (real data) | 35 / 35 |
| production identity (runtime) | 31 / 31 |
| production/experiment isolation | 16 / 16 |
| real-data smoke (GPU, full chain) | 32 / 32 |
| **Total** | **305 / 305, 0 failures** |

The 4 skips in the Spot drill are its GCS legs, not run here because that is
Phase 2 work. The script counts skips separately and never as passes; those
legs passed earlier against the real bucket (recorded in
`preflight_validation.md`).

### Assertions corrected during Phase 1

Neither was weakened to obtain a pass; both asserted a contract the system does
not have.

1. **`resume restores the schedule exactly`** — asserted that
   `load_state_dict` writes the LR into the optimizer immediately. PyTorch
   defers that to the next `step()`. Now asserts that a resumed schedule
   reproduces the uninterrupted LR sequence.
2. **`ET-positive representation increases`** — asserted a rise in ET-positive
   sampling share. Measurement showed that share is already 1.0 and cannot
   rise (section D). Now asserts the size effect, which is what the sampler
   actually does and which holds on both the synthetic fixture and the real
   cohort.

---

## C. Production identity

Verified from the **constructed runtime model**, not from configuration text.
**31 / 31 invariants hold.**

```
29,337 inference params
29,412 training params
    75 auxiliary params

L1 = 48^3 / 24 / 15 / k5
L2 = 64^3 / 24 / 15 / k5
L3 = absent  (2 levels constructed)

working volume = 128^3
patchify = OFF        ROI = 1.0

global context = ON
SE = ON
spatial global context = ON, kernel 5
fusion = Conv3d(48 -> 24)

deep supervision = 0.4

Tversky 0.40 / 0.60    gamma 1.33    ce 0.5    empty-BCE 0.1
BF16 forward, FP32 loss
EMA 0.999
gradient clip 1.0
LR 0.0016 -> 0.00001 cosine, weight decay 0.0001
batch 1, workers 2

post-processing WT 50 / TC 5 / ET 0
300 epochs, early stopping 15 / 0.01
split 898 / 200 / 198
seed 42
```

Mechanism absence, asserted explicitly:

```
production sampler          = uniform  (_EpochSampler)
production warmup           = OFF
production boundary loss    = OFF
production dynamic weighting= OFF
production hard-case mining = OFF
```

### Final real-data smoke (section 8 of the brief)

Full chain on real BraTS cases through the production model, **32 / 32**:

```
dataset -> preprocessing -> 128^3 -> model -> BF16 forward -> FP32 loss
-> deep supervision -> backward -> gradient clip -> optimizer -> EMA
-> checkpoint -> validation -> post-processing -> per-case diagnostics
-> lesion-size stratification

losses finite               3.9753, 3.7795
gradients finite            YES
parameters finite           YES
no NaN / Inf                YES
no shape mismatch           YES
checkpoint round trip       259,410 bytes, bit-exact
test cases accessed         0 of 198

MEASURED device             cuda
MEASURED median iteration   5.896 s   (local GPU, not the L4)
MEASURED peak VRAM          1,373 MB
```

Dice is near zero here by design: the model has taken two optimiser steps from
random initialisation. This gate validates the pipeline, never segmentation
quality.

---

## D. Dataset

| | |
|---|---|
| Split | 898 train / 200 validation / 198 test = 1296 |
| Fingerprint | `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d` |
| Fingerprint recomputed | **MATCHES** the stored value — split unmodified |
| Split version | `GLO-NCA-V2-master-v1` |
| Subjects | 810 across 1296 cases |
| Subject-disjoint | **YES** — 0 subject overlap between any pair of splits |
| Case overlap | 0 |
| Seeds | 42 in production and in both experiment configs |
| Seeded split | production refuses it (`allow_seeded_split` absent) |

### ET voxel census — coverage and the finding

**Coverage: 300 of 898 training cases (33%).** Measured after the same
deterministic preprocessing training uses (foreground crop, resample to 128³),
so counts are in **resampled voxels**, not mm³.

| | |
|---|---|
| Cases measured | 300, **0 failures** |
| ET-positive | **300 of 300 measured (100.0%)** |
| ET-absent | 0 |
| ET min / max | 15 / 45,406 |
| ET median / mean | 1,046 / 3,600 |
| Percentiles | p5 46 · p10 86 · p25 248 · p50 1,046 · p75 4,326 · p90 10,102 · p95 14,745 |
| Strata | small 34 · medium 112 · large 154 · absent_gt 0 |

**Statement of scope, deliberately narrow:** *300 of 300 measured cases
contained enhancing tumour.* This is **not** generalised to all 898. The
remaining 598 are unmeasured.

**Why it matters.** In BraTS-**METS** (metastases) enhancing tumour is
definitional, unlike BraTS-GLIOMA where ET is often absent. On the measured
sample the ET-positive share is already 1.0 under uniform sampling, so
"ET-positive oversampling" has no headroom. What varies is ET **volume** —
15 → 45,406 voxels, roughly three orders of magnitude.

**Full census deferred to Phase 2.** It is not disk-limited: the census writes
only a CSV (<1 MB) with the preprocessing cache opt-in and off by default. It
is time-limited — 4.0–6.1 s/case alone, 60–90 minutes for 898, and slower still
under CPU contention. Two attempts were made locally; the second was ended by a
session restart. The VM is substantially faster and should run it.

---

## E. Small-lesion sampler (experiment only)

### Terminology corrected

The mechanism was previously described as *ET-aware / ET-positive* sampling.
The measurement above makes that description wrong, so it was corrected
throughout rather than left as a misleading label:

| | before | after |
|---|---|---|
| Config file | `glo_nca_et_sampling.yaml` | `glo_nca_small_lesion_sampling.yaml` |
| Experiment name | `GLO-NCA-ET-SAMPLING` | `GLO-NCA-SMALL-LESION-SAMPLING` |
| Module docstring | "ET-aware sampling experiment" | states ET volume is the *signal*, small-lesion emphasis the *purpose* |
| Test assertions | "ET-positive drawn more often" | "small-lesion drawn more often" |
| Earlier audit report | — | carries a **superseded** note pointing here |

`LesionAwareSampler` and `lesion_sampler.py` keep their names: they were
already accurate, and renaming them would churn dependencies for no gain. The
`et_*` fields refer to the weighting signal, which is documented in the module.

### Measured behaviour (real census, 30 epochs, seed 42, boost 2.0)

| | baseline (uniform) | experimental |
|---|---|---|
| ET-positive share | 1.0000 | 1.0000 (unchanged — no headroom) |
| small-ET (≤100 vox) share | 0.1133 | **0.1780** |
| oversampling ratio | — | **≈ 1.57×** |
| epoch length | 300 | 300 (unchanged) |

Epoch length and therefore optimiser-step count are identical to the baseline,
so a future A/B is not confounded by extra updates.

### Verification

```
weights finite and positive        range [1.002, 2.000]
weights normalised                 sum == 1.0
epoch length equals baseline       YES
deterministic at seed 42           YES
differs across epochs              YES
only train indices sampled         300/300 train
validation cases in sampler        0
test cases in sampler              0
sampler modifies labels            NO
sampler carries model state        NO
```

**No Dice claim. MEASURED RESULT: NOT YET MEASURED.**

---

## F. LR warmup (experiment only)

Verified over the real production step count:

```
warmup_epochs                 3   (from the experiment config)
steps/epoch x epochs          300 x 300 = 90,000 steps
warmup steps                  900 = 3 x 300

LR at step 0                  0.00000178   (below peak)
warmup strictly increasing    YES
peak at end of warmup         0.001600     (== production LR)
cosine decays after warmup    YES
final LR                      0.0000100000 (== minimum_learning_rate)
total steps                   90,000       (budget unchanged)
resumed sequence identical    YES
production uses warmup        NO
```

**No Dice claim. No convergence claim.**

---

## G. Stratified evaluation

**Evaluation-only, verified structurally rather than asserted.**

* All 4 `LS.*` call sites sit at runner lines 1344–1351; the training loop
  begins at line 1062. Every call is after training completes.
* The module imports only `csv`, `os` and `numpy` — no torch, no data loading,
  no optimizer/scheduler/checkpoint access.
* It reads exactly five per-case row keys: `gt_vox`, `pred_vox`, `dice`, `iou`,
  `hd95`. It writes no labels and mutates no predictions.
* It consumes rows the per-case diagnostics already produced, so it adds **no
  model pass**.

Strata `small` / `medium` / `large` / `absent_gt` are all represented and
reconcile with the per-case rows. **For ET on this cohort `absent_gt` is
legitimately 0** — every measured case has ET. That is correct behaviour, not a
defect; the stratum still applies to TC and to any future dataset.

HD95 validity accounting (`empty_gt` / `empty_prediction` / `both_empty` /
valid) is preserved, and undefined HD95 is never replaced with zero — it is
written blank in the CSV and excluded from the valid-case mean.

Measured overhead: **3.97 ms** for 200 cases, once per run.

---

## H. Disk / cache safety

| | measured |
|---|---|
| Free space | **41 GB** (was 17 GB at the worst point) |
| Preprocessing cache | 9.8 GB, 264 entries |
| Reports | 1.2 MB |

**The risk that existed has been removed at source.** The first census attempt
was writing the preprocessing cache at 38.5 MB/case — 34 GB for 898 cases
against 17 GB free. It would have exhausted the disk. The census reads each
label exactly once, so the cache bought nothing.

`scripts/analyze_et_voxels.py` now disables the cache by default and requires
`--cache` to persist it. Verified: cache entry count held at 264 across a full
census run.

### Safe cleanup available — NOT performed

The 9.8 GB cache under `.cache/preprocessed/` is **regenerable** and gitignored,
and holds 264 of 1296 cases from abandoned runs. Deleting it would free 9.8 GB.
It is **not** deleted here: it is still useful for local runs, and it is your
data to remove. No reports, checkpoints or research evidence were deleted.

---

## I. Test isolation

```
test cases accessed during smoke      0 of 198
test cases in the sampler             0
validation cases in the sampler       0
census split                          train only; val/test entries left empty
```

The census enforces this **structurally**, not by convention: it installs only
the train ids into the dataset and leaves the val and test entries as empty
dicts, so those splits cannot be read even accidentally.

Test data was not used for sampler weights, thresholds, early stopping,
checkpoint selection, hyperparameters or experiment design.

---

## J. Deferred mechanisms

All three remain deferred and un-implemented. No distance transform, boundary
term, dynamic weighting or mining logic exists in production or in any
experiment config.

**Boundary-aware loss — cost measured, rejected.** A surface-distance term needs
a Euclidean distance transform per region per case per epoch: measured at
**608 ms per region, 1,638 s/epoch, +198%**, or **+136.5 GPU-hours** over 300
epochs. Its expected benefit is to HD95 rather than Dice. Not re-run in Phase 1.

**Dynamic loss weighting — no defensible justification.** The baseline already
addresses class imbalance three ways: Tversky β=0.60, focal γ=1.33, and
empty-region BCE 0.1. Adding scheduled weighting introduces tuning axes that
cannot be defended without experiments there is no budget for.

**Hard-case mining — risk not justified.** Requires a periodic training-set-wide
scoring pass of unmeasured cost, and "hard" cases in BraTS-METS correlate with
annotation ambiguity, so mining can amplify label noise. It also couples data
selection to model state, complicating reproducibility.

---

## K. GCP handoff

```
PHASE 1 LOCAL = COMPLETE
PHASE 2 GCP   = NOT STARTED
```

Confirmed not done, per the brief: no VM created or started, no L4 rented, no
GCS service-account verification, no Spot-preemption test on a real VM, no
300-epoch run, no GCP performance or cost measurement.

VM `culling` remained **TERMINATED** throughout Phase 1 — **$0/hr GPU spend**.

### Phase 2 queue

1. **GCS service-account permission** — `bash cloud/scripts/verify_gcs_service_account.sh`
   on the VM. It cannot be satisfied from a workstation: the local identity is a
   human account whose permissions are broader than the training job's, and
   impersonation is denied. The script **refuses to report PASS unless the
   active identity is a service account**, and is wired into `pretrain_gate.sh`
   as Part 8. This is the failure that killed an earlier Phase 2 run.
2. **Full 898-case ET census** — `python scripts/analyze_et_voxels.py`
   (60–90 min locally; faster on the VM). Only then may the ET finding be
   generalised beyond the measured 300.
3. **Spot-preemption recovery on a real VM.**
4. **300-epoch production run.**

---

## Git status

```
commit          441b142   (unchanged)
new commits     NONE
working tree    dirty
pushed          NO
```

### Intended changes

Modified:
* `.gitignore` — root-anchored `experiments/`
* `configs/glo_nca_production.yaml` — comments only; no value changed
* `src/experiment/runner.py` — lesion-strata wiring (evaluation only)
* `scripts/test_real_data_smoke.py` — per-case, stratification, test-access counter
* `cloud/scripts/pretrain_gate.sh` — service-account check as Part 8
* `scripts/test_quality_mechanisms.py` — small-lesion terminology
* `scripts/validate_experiments_real_data.py` — corrected assertion, renamed config
* `scripts/analyze_et_voxels.py` — cache opt-in, case-id fix
* `src/experiment/lesion_sampler.py` — docstring corrected
* `reports/audit/quality_mechanism_audit.md` — superseded note

Renamed:
* `configs/experiments/glo_nca_et_sampling.yaml` →
  `configs/experiments/glo_nca_small_lesion_sampling.yaml`

Added:
* `src/experiment/{lesion_strata,lesion_sampler,warmup}.py`
* `configs/experiments/` (2 configs)
* `scripts/{test_quality_mechanisms,analyze_et_voxels,validate_experiments_real_data,test_spot_recovery}.py`
* `cloud/scripts/verify_gcs_service_account.sh`
* `reports/analysis/`, `reports/audit/{quality_mechanism_audit,preflight_validation,phase1_final_local_validation}.md`

### Unexpected changes

**None.** Every modified, renamed and untracked path is accounted for above.

---

## Final determination

```
LOCAL PHASE 1 READY FOR GCP
```

All local tests pass (305/305), production invariants hold (31/31), the
production baseline is methodologically unchanged, there are no unexpected code
changes, the sampler is correctly characterised as small-lesion-aware, LR warmup
is isolated to its own config, lesion stratification is evaluation-only, no test
leakage exists, checkpoint/resume works, the disk risk is removed at source,
reproducibility checks pass, and documentation is consistent with what the code
does.

Two items carry forward to Phase 2 and are **not** local blockers: the
service-account permission test (executable only on the VM) and the full
898-case census (time-limited locally).
