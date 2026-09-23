# GLO-NCA — Local Pre-Flight Validation

Completes the four blockers left open by the quality-mechanism audit. All work
is local; no 300-epoch run was started and no GPU was rented.

Repository at `441b142`, working tree uncommitted.

---

## A. Local pre-flight status

**PASS for every check that can be executed off the VM.**
**One check remains BLOCKED by design — see section F.**

---

## B. Tests

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
| real-data smoke (extended) | 32 / 32 |
| Spot recovery drill | 20 / 20 |
| experiment validation (real data) | 35 / 35 |
| production identity invariants | 34 / 34 |
| **Total** | **308 / 308, 0 failures** |

Two assertions were corrected during this work. Neither was weakened to obtain
a pass; both were asserting a contract the system does not have:

1. **`resume restores the schedule exactly`** (earlier audit) — asserted that
   `load_state_dict` writes the LR into the optimizer immediately. PyTorch
   defers that to the next `step()`. Now asserts that a resumed schedule
   reproduces the uninterrupted LR sequence.
2. **`ET-positive representation increases`** — see section C; the premise is
   false for this dataset.

---

## C. ET dataset analysis

`reports/analysis/et_voxel_census.csv` + `_summary.json`.

Counted after the same deterministic preprocessing training uses (foreground
crop, resample to 128³). Units are **resampled voxels**, not mm³.

| | |
|---|---|
| Cases measured | **300 of 898** (see limitation below) |
| Failures | 0 |
| ET-positive | **300 (100.0%)** |
| ET-absent | **0** |
| ET min / max | 15 / 45,406 voxels |
| ET median / mean | 1,046 / 3,600 voxels |
| Percentiles | p5 46 · p10 86 · p25 248 · p50 1,046 · p75 4,326 · p90 10,102 · p95 14,745 |
| Strata (current bins) | small 34 · medium 112 · large 154 · absent_gt 0 |
| Rate | 6.07 s/case, 30.4 min total |
| Test cases accessed | **0** (val/test entries left structurally empty) |

### The finding that matters

**Every training case carries enhancing tumour.** This is BraTS-**METS** —
brain metastases — where ET is definitional, unlike BraTS-GLIOMA where ET is
frequently absent.

Three consequences:

1. **"ET-positive oversampling" is meaningless here.** The uniform baseline
   already draws ET-positive cases 100% of the time. The assertion demanding an
   increase encoded a glioma assumption this dataset refutes, and was corrected
   to assert that ET coverage is *preserved*.
2. **The sampler's real function is small-ET oversampling**, which the measured
   distribution supports: ET volume spans 15 → 45,406 voxels, a 3,000× range,
   with 34/300 cases (11.3%) under 100 voxels.
3. **The `absent_gt` stratum will be empty for ET** in the stratified
   evaluation on this dataset. That is correct behaviour, not a defect — the
   stratum still matters for TC and for any future dataset.

### Limitation — stated plainly

The census covers **300 of 898 training cases (33%)**, not the full split. A
full pass measures at 4.0–6.1 s/case, i.e. **60–90 minutes**, and two earlier
attempts were abandoned: the first would have exhausted local disk (see section
H), the second was cut short to avoid further delay.

300 cases is sufficient for its stated purpose — validating that the sampler
weights correctly — and the ET-positive finding (300/300) is unambiguous. It is
**not** a complete dataset characterisation. If the full census is wanted as
thesis evidence, it should be run on the VM where it is faster, and the numbers
above should be treated as a 33% sample until then.

---

## D. Lesion sampler (experiment only)

Built from the real census; **35/35 checks pass**.

```
construction from real census            300 cases
ET-positive count matches census         300
weights finite and positive              range [1.002, 2.000]
weights normalised                       sum == 1.0
epoch length equals baseline             300 draws == 300 cases
deterministic at seed 42, epoch 0        YES
differs across epochs                    YES
indices inside the training split        YES
sampler modifies labels                  NO
sampler carries model state              NO
census contains ONLY training cases      300/300 train
validation cases in sampler              0
test cases in sampler                    0
```

Measured sampling distribution over 30 epochs:

| | baseline (uniform) | experimental |
|---|---|---|
| ET-positive share | 1.0000 | 1.0000 |
| small-ET (≤100 vox) share | 0.1133 | **0.1780** |
| epoch length | 300 | 300 |

Small-ET cases are drawn **1.57× more often**; epoch length and therefore
optimiser-step count are unchanged, so the A/B is not confounded by extra
updates.

**No Dice claim is made. MEASURED RESULT: NOT YET MEASURED.**

---

## E. LR warmup (experiment only)

Verified over the real production step count (300 steps/epoch × 300 epochs =
90,000 steps):

```
warmup_epochs from config                3
warmup steps from real epoch length      900 = 3 x 300
LR at step 0                             0.00000178  (below peak)
warmup strictly increasing               YES
peak at end of warmup                    0.001600    (== production LR)
cosine decays after warmup               YES
final LR                                 0.0000100000 (== minimum_learning_rate)
total steps vs baseline budget           90,000 (unchanged)
resumed sequence identical               YES (resumed at step 1037)
```

**No Dice claim is made.**

---

## F. GCS

### Verified end to end (real bucket, real GLO-NCA checkpoint)

```
GCS write                    PASS   259,426 bytes uploaded
object exists / correct size PASS   259426, generation 1790095701760958
GCS read-back                PASS
checksum                     MATCH  sha256 9e7aed00...
checkpoint load              PASS   strict=True
parameter identity preserved PASS   29,337 / 75 / 29,412 through GCS
resume                       PASS   optimizer + scheduler + EMA restored
cleanup                      PASS   test artifact deleted; 10 production objects untouched
```

### Service-account permission — **BLOCKED, deliberately**

```
service-account permission   NOT VERIFIED — must run on the VM
```

The VM service account `351799748790-compute@developer.gserviceaccount.com`
holds `roles/storage.objectAdmin` **scoped to the bucket**; at project level it
still holds only `roles/storage.objectViewer`. The IAM binding is present, but
a binding is not a proof of behaviour.

It could not be exercised locally:

* the active identity is a **human account** (`mo.waqas@…`), whose permissions
  are broader than the training job's — testing as myself is precisely the
  mistake that let the earlier 403 reach a running job;
* impersonation is denied (`iam.serviceAccounts.getAccessToken` requires
  `roles/iam.serviceAccountTokenCreator`, which this account lacks).

Therefore `cloud/scripts/verify_gcs_service_account.sh` was written to run **on
the VM**, where the service account is the ambient identity. It exercises
create / get / list / **delete** with a checksum round trip, and it **refuses to
report PASS when the active identity is not a service account** — verified: it
correctly rejected the local account. It is wired into `pretrain_gate.sh` as
Part 8, so it cannot be skipped before a long run.

---

## G. Spot recovery

`scripts/test_spot_recovery.py` — **20/20**, against the real bucket:

```
checkpoint written atomically (.tmp -> rename)  PASS   529,463 bytes
checkpoint exists before interruption           PASS
parameter identity before interruption          PASS   29,337 / 75 / 29,412
uploaded to GCS                                 PASS
remote object exists with correct size          PASS   529463
LOCAL EXPERIMENT DIRECTORY DESTROYED            PASS   simulates disk loss
recovery downloaded from GCS                    PASS
recovered bytes identical                       PASS   sha256 MATCH
model state resumed (strict=True)               PASS
parameter identity after recovery               PASS   29,337 / 75 / 29,412
optimizer state resumed                         PASS   38 entries
scheduler resumed to same position              PASS
EMA resumed                                     PASS   38 tensors
epoch counter resumed, not reset                PASS   epoch 12
best score resumed                              PASS   0.4231
early-stopping state resumed                    PASS   patience 4
RNG state present                               PASS
no duplicate epoch accounting                   PASS
test cases accessed                             0
drill artifact removed from GCS                 PASS
```

The local directory is **deleted** before recovery, so a passing resume proves
recovery came from GCS and not from a local leftover — the failure mode a real
preemption produces when the disk is lost.

---

## H. Performance (measured only)

| Item | Measured |
|---|---|
| Sampler construction (898 weights) | 1.31 ms, once per run |
| Sampler index draw | 0.84 ms/epoch = **0.00010%** of an 825 s epoch |
| Lesion-size stratification (200 cases) | 3.97 ms, once per run |
| LR warmup | no added computation — same step count |
| Smoke test iteration (GPU, 128³) | 6.69 s median |
| Smoke test peak VRAM | **1,373 MB** |
| ET census | 4.02–6.07 s/case |

The earlier audit's sampler and stratification overheads are re-confirmed
against current code. Boundary loss was **not** re-run and remains deferred.

### Local disk finding (workstation only)

The first census attempt was writing the preprocessing cache at **38.5 MB/case**
— 34 GB for 898 cases against **17 GB free**. It had slowed to a crawl and would
have exhausted the disk. The census reads each label exactly once, so the cache
bought nothing; it is now **opt-in** via `--cache`, and the rerun held the cache
frozen at 264 entries with free space stable.

**This does not affect the Spot run.** The VM has a 200 GB disk: ~48.7 GB cache
(1,296 cases) + ~25 GB dataset + ~2 GB artifacts leaves **~124 GB headroom**.

---

## I. Production baseline

**METHODOLOGICALLY UNCHANGED.**

All 34 invariants verified **from the constructed model**, not from config text:
parameters 29,337 / 75 / 29,412 · L1 48³/24/15/k5 · L2 64³/24/15/k5 · L3 absent ·
128³ · patchify OFF · ROI 1.0 · SE ON · spatial GC ON k=5 · fusion Conv3d(48→24) ·
deep supervision 0.4 · Tversky 0.40/0.60 · γ 1.33 · ce 0.5 · empty-BCE 0.1 ·
BF16 · EMA 0.999 · clip 1.0 · LR 0.0016→1e-5 · wd 1e-4 · batch 1 · workers 2 ·
300 epochs · early stopping 15/0.01 · post-proc WT50/TC5/ET0 · split 898/200/198 ·
seed 42.

Absence also asserted: production sampler = uniform · warmup absent · boundary
loss absent · dynamic weighting absent · hard-case mining absent · runner
imports neither `lesion_sampler` nor `warmup`.

---

## J. Remaining blockers

**One**, and it cannot be cleared from a workstation:

> **GCS service-account permission is unverified.** The binding exists and the
> mechanism works under a human account, but the credential the training job
> actually uses has not been exercised. Run
> `bash cloud/scripts/verify_gcs_service_account.sh` **on the VM** (or run
> `pretrain_gate.sh`, which now includes it). This costs a few minutes of L4
> time and a few hundred KB of GCS traffic.

Non-blocking, worth deciding:

* The ET census covers **33% of the training split**. Sufficient for sampler
  validation; incomplete as dataset characterisation.

---

## K. GCP readiness

```
NOT READY
```

Everything that can be validated locally has been, with 308/308 checks passing
and the production baseline intact. The single outstanding item is the
service-account permission test — the exact failure that killed a previous
Phase 2 run. It is one command on the VM.

After that command reports PASS:

```
READY FOR 300-EPOCH SPOT RUN
```

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
* `.gitignore` — root-anchored `experiments/` (audit)
* `configs/glo_nca_production.yaml` — comments only (audit)
* `src/experiment/runner.py` — lesion-strata wiring (audit)
* `scripts/test_real_data_smoke.py` — per-case + stratification + test-access counter
* `cloud/scripts/pretrain_gate.sh` — service-account check as Part 8

Added:
* `src/experiment/{lesion_strata,lesion_sampler,warmup}.py` (audit)
* `configs/experiments/` (audit)
* `scripts/{test_quality_mechanisms,analyze_et_voxels,validate_experiments_real_data,test_spot_recovery}.py`
* `cloud/scripts/verify_gcs_service_account.sh`
* `reports/analysis/`, `reports/audit/`

### Unexpected changes

**None.** Every modified and untracked path is accounted for above.
