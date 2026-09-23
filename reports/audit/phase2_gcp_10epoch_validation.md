# GLO-NCA — Phase 2 GCP 10-Epoch Validation

Engineering validation of the GCP/GCS/Spot/checkpoint/recovery pipeline on the
real NVIDIA L4. **This is not a thesis performance experiment.** Its Dice
numbers are not converged, were produced at a non-production learning rate, and
must not be extrapolated.

The 300-epoch production run was **not** started.

---

## A. Environment

| | Measured |
|---|---|
| Project | `even-continuity-501915-f9` |
| VM | `culling` |
| Zone | `us-east1-c` |
| Machine | `g2-standard-8` (8 vCPU, 31 GB) |
| GPU | **NVIDIA L4, 23034 MiB** |
| Driver | **580.178.04** |
| Provisioning | **SPOT**, termination action **STOP** |
| Python | 3.10.12 |
| PyTorch | **2.5.1+cu121**, CUDA 12.1 |
| BF16 supported | **True** |
| Git commit (VM) | **`08cf30cd170cc4cf1fec3f33af8b858bc22644e2`** |

---

## B. Repository gate

| | SHA |
|---|---|
| Phase 1 commit | `27784a437de80943c9c88c89bcd3984224ba1e54` |
| Phase 2 config commit (HEAD) | `08cf30cd170cc4cf1fec3f33af8b858bc22644e2` |
| `origin/v3-multilevel` | `08cf30cd170cc4cf1fec3f33af8b858bc22644e2` |
| VM `git rev-parse HEAD` | `08cf30cd170cc4cf1fec3f33af8b858bc22644e2` |
| **All match** | **YES** |

All 13 required Phase 2 files verified present on the remote branch before the
VM was started, plus the runner's lesion-stratification wiring.

### Two blockers found by this gate

**1. The VM was running the wrong code.** On start, `/opt/glo-nca` was at
`466eda5` — *two commits before* the Phase 1 handoff baseline. Every Phase 1
deliverable was absent. Without the SHA gate the run would have silently
trained stale code.

**2. The Docker image was stale.** The pipeline runs inside `glo-nca:latest`,
not a host venv, so `git pull` alone does not update the running code. The
existing 15.2 GB image predated Phase 1 and lacked `lesion_strata.py` and the
validation config. It was rebuilt (`sha256:8a6bcc71…`, ~2 min) and both
containers were verified bound to that image ID before any measurement.

---

## C. GCS gate

Run **on the VM**, as the ambient training identity — not from a workstation.

```
active identity  351799748790-compute@developer.gserviceaccount.com
bucket           gs://glo-nca-v3-even-continuity-501915
object           experiments/_sa_permission_check/probe.bin

storage.objects.create (write)          PASS
storage.objects.get (exists, 262144 B)  PASS
storage.objects.get (read back)         PASS
checksum after round trip               MATCH
storage.objects.list                    PASS
storage.objects.delete (cleanup)        PASS

service-account permission              PASS
```

This closes the Phase 1 blocker. The earlier Phase 2 run died on a GCS 403
because the credential was never tested as the service account; this gate
refuses to report PASS unless the active identity *is* a service account.

### Checkpoint round trip (production mechanism)

```
upload (gcloud storage rsync)   PASS   7.0 MiB/s
remote object exists            PASS   544118 bytes, generation 1790169754767255
read back                       PASS   383.9 MiB/s
checksum  local    f598eb8f03f85d2516e89c379c9733d03cd75bd1c2fd521d27affdcd1bfae759
          readback f598eb8f03f85d2516e89c379c9733d03cd75bd1c2fd521d27affdcd1bfae759
          MATCH
```

---

## D. Dataset

```
dataset validation       PASS 1296/1296 (pass 1296, fail 0)
cohorts                  top-level 650, nested 646
split fingerprint        d30d71956ee9
train / val / test       898 / 200 / 198
TEST CASES ACCESSED      0 / 198
```

See section I for how the test firewall was enforced.

---

## E. Full ET census — NOT COMPLETED

**Deferred to a later session.** Two attempts failed:

1. Killed by CPU contention with the training job (8 vCPUs, load 7.79); the
   rate decayed monotonically 1.43 → 3.00 s/case before the process died at
   600/898, producing no CSV.
2. Not retried, to stop the L4 idling while a CPU-only job ran.

The Phase 1 result stands: **300 of 898 cases measured, 300/300 ET-positive**,
documented as a 33% sample. This is a *measurement*, not a gate for the
300-epoch decision. It should be run on a cheap CPU-only VM, where the measured
L4 rate of 1.43 s/case implies ~21 minutes for all 898.

---

## F. Production identity

Verified from the **constructed model inside the training image**, on the L4.
**20 / 20 PASS.**

```
29,337 inference params      75 auxiliary params      29,412 training params

L1 = 48^3 / 24 / 15 / k5     L2 = 64^3 / 24 / 15 / k5     L3 = absent
working volume 128^3         patchify OFF        ROI 1.0
SE ON       spatial GC ON, k5        fusion Conv3d(48 -> 24)
deep supervision 0.4
Tversky 0.40/0.60   gamma 1.33   ce 0.5   empty-BCE 0.1
BF16 forward, FP32 loss      EMA 0.999      gradient clip 1.0
batch 1      workers 2      seed 42      split 898/200/198

production sampler          = uniform (_EpochSampler)
production warmup           = OFF
production boundary loss    = OFF
production dynamic weighting= OFF
production hard-case mining = OFF
```

---

## G. Ten epochs — measured

Config: `configs/gcp_phase2_10epoch_lr_validation.yaml`, experiment id
`PHASE2-10EP-VALIDATION`.

| Epoch | Train Loss | Val Dice (mean) | WT | TC | ET | Epoch s | LR | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.1451 | 0.1182 | 0.2921 | 0.0244 | 0.0380 | **3138.4** | 3.90e-4 | 1.078 |
| 2 | 1.6906 | 0.2807 | 0.3554 | 0.2500 | 0.2368 | 823.8 | 3.63e-4 | 1.078 |
| 3 | 1.5893 | 0.3163 | 0.3862 | 0.2881 | 0.2747 | 823.9 | 3.20e-4 | 1.078 |
| 4 | 1.4969 | 0.3369 | 0.4028 | 0.3108 | 0.2971 | 826.0 | 2.65e-4 | 1.078 |
| 5 | 1.4671 | 0.3457 | 0.4126 | 0.3196 | 0.3048 | 825.8 | 2.05e-4 | 1.078 |
| 6 | 1.3825 | 0.3522 | 0.4201 | 0.3262 | 0.3104 | 825.9 | 1.45e-4 | 1.078 |
| 7 | 1.3477 | 0.3630 | 0.4285 | 0.3395 | 0.3209 | 825.3 | 9.04e-5 | 1.078 |
| 8 | 1.3117 | 0.3720 | 0.4371 | 0.3483 | 0.3307 | 825.5 | 4.72e-5 | 1.078 |
| 9 | 1.2774 | **0.3812** | 0.4481 | 0.3562 | 0.3393 | 825.6 | 1.95e-5 | 1.078 |
| 10 | 1.2582 | 0.3773 | 0.4444 | 0.3519 | 0.3356 | 854.5 | 1.00e-5 | 1.078 |

**HD95 was not computed**: the config sets `hd95_every_epochs: 10`, and the
final evaluation that would have written it was deliberately stopped to
preserve the test firewall. The `hd95_*` columns are `nan` for every epoch.

### Timing and utilisation (measured)

```
epoch 1 (cold, builds 40 GB cache)   3138.4 s
warm epoch median                     825.6 s   (min 823.8, max 854.5)
iteration                             0.919 s/case over 898 cases
total training wall time              2.94 h
peak VRAM                             1.078 GB

active GPU utilization                99.9 %   (791 samples, measured DURING
                                               forward/backward only; DATA_WAIT,
                                               CHECKPOINT and IDLE excluded)
GPU temperature                       73 C mean (max 79 C observed)
GPU power                             34.8 W mean
SM clock                              72.5 % of maximum
```

**Throttle finding:** the runner reports that SM clock averaged 72.5% of
maximum during active compute, so absolute timings are inflated by roughly
1.4×. Ratios remain valid. An unthrottled epoch would be nearer ~590 s.

### Comparison with the historical reference

| | Historical (earlier Phase 2) | This run | Note |
|---|---|---|---|
| Warm epoch | 825 s | **825.6 s** | effectively identical |
| Peak VRAM | 1.08 GB | **1.078 GB** | effectively identical |
| Iteration | 0.748 s | 0.919 s | see caveat |

Epoch time and VRAM match the historical reference almost exactly, so the
Phase 1 changes introduced **no measurable regression**. The iteration figure
is not directly comparable: the two runs used different learning rates and this
one ran under a measured 72.5% SM clock, so the difference is not evidence of
a regression.

---

## H. Learning rate — configuration choice, stated explicitly

```
Production LR            0.0016   (configs/glo_nca_production.yaml, UNCHANGED)
Phase 2 validation LR    0.0004   (this run only)
```

The reduced rate was used solely for this 10-epoch engineering validation, at
the user's explicit instruction. It lives in a separate config file; the
production config was not modified. **Because the LR differs, this run's loss
and Dice curve are not comparable to a production run**, and not comparable to
the earlier 10-epoch reference either.

The validation config differs from production in exactly three keys, asserted
programmatically: `experiment.name`, `optimizer.learning_rate`,
`training.epochs`. The `model`, `loss`, `data`, `ema`, `gradient`,
`evaluation`, `performance`, `memory` and `sampling` sections are byte-identical
to production.

---

## I. Checkpoints

```
best.pth   129,102 bytes   sha256 1d2a191ac1c482ca743b3f63223275ca1049d8ba...
           schema {ep, m, val_mean, val_smooth}
           STRICT LOAD OK   29,337 / 75 / 29,412
           ep 10   val_mean 0.377268   val_smooth 0.376829

last.pth   544,118 bytes   sha256 f598eb8f03f85d2516e89c379c9733d03cd75bd1...
           schema {model, optimizer, scheduler, ema, epoch, best_score,
                   best_epoch, early_stopping, rng_state, history, config, format}
           STRICT LOAD OK   29,337 / 75 / 29,412
           epoch 10   best_score 0.376829   format glo-nca-v2-ckpt-1
```

Also produced: `checkpoints/top_k/`, `checkpoints/periodic/`,
`checkpoints/model_internal/`.

### Resume validation (from the GCS read-back copy, not the local original)

```
model strict load                  PASS   29,337/75/29,412
optimizer state restored           PASS
scheduler restored                 PASS   last_epoch 8980
LR state correct                   PASS   1.000e-05
EMA restored                       PASS   38 tensors
epoch restored (not reset)         PASS   10
best score restored                PASS   0.376829
early-stopping state restored      PASS
RNG state present                  PASS
config embedded (no silent drift)  PASS
forward pass on restored model     PASS   out (1,3,64,64,64), finite
eval() returns bare tensor         PASS   (aux heads training-only)
```

---

## J. Spot recovery

**SIMULATED disk-loss drill against the real bucket. No actual GCP Spot
preemption occurred during this run**, and none is claimed.

```
[1] checkpoint before interruption   544,118 bytes, sha f598eb8f...
[2] GCS object verified              544118 bytes, generation 1790169754767255
[3] LOCAL CHECKPOINT DIRECTORY DESTROYED   (rm -rf, simulates Spot disk loss)
[4] recovered from GCS               395.4 MiB/s
    recovered sha256                 f598eb8f...  IDENTICAL
[5] post-recovery strict load        PASS 29,337/75/29,412
    epoch / best_score restored      10 / 0.376829
    optimizer + scheduler + EMA      present
    early_stopping + rng_state       present
```

The local directory was genuinely deleted before recovery, so a passing resume
proves recovery came from GCS and not from a local leftover.

---

## K. Test isolation

```
TEST CASES ACCESSED = 0 / 198
```

Enforced, not merely observed. The runner's post-training evaluation calls
`collect_probs(agent, ds, "test")` unconditionally. When the run entered
`state: evaluating` after epoch 10, the container was **stopped before the test
split was opened**.

Verified afterwards:

* no test metrics in any report
* no test predictions (`predictions/` empty)
* no `*test*` artifacts except `split/test.txt`, which is the split manifest
  (a list of case ids), not evaluation output

**Consequence, stated plainly:** stopping there also prevented the final
validation post-processing, per-case diagnostics and lesion-size stratification
from being written, since they run in the same evaluation phase. Those paths
were verified on the L4 by the smoke test instead (section L).

---

## L. Real-data GPU smoke test

Run on the L4 before the 10 epochs, through the full production chain.
**32 / 32 PASS.**

```
dataset -> preprocessing -> 128^3 -> L1 -> L2 -> spatial GC -> fusion
-> BF16 forward -> FP32 loss -> deep supervision -> backward -> clip
-> optimizer -> EMA -> checkpoint -> validation -> post-processing
-> per-case diagnostics -> lesion-size stratification

losses finite                    4.3931, 4.0953
gradients / parameters finite    YES
no NaN / Inf / shape mismatch    YES
checkpoint round trip            259,410 bytes
lesion stratification            12 region x stratum rows, all strata incl. absent_gt
test cases accessed              0

median iteration                 2.410 s   (measured under CPU contention)
peak VRAM                        1372 MB
```

---

## M. Cost

**Billing not verified against an invoice.**

```
VM runtime (GCP metadata)   2026-09-23 00:18:17 PT -> 06:33:45 PT = 6.26 h

list-rate estimate, us-east1 SPOT:
  g2-standard-8      $0.2268/h x 6.26 = $1.42
  NVIDIA L4          $0.2040/h x 6.26 = $1.28
  200 GB disk        (while running)   = $0.07
  ------------------------------------------------
  approx                                 $2.76
```

These are published list rates, not an invoice. Spot pricing varies and any
credits or discounts are not reflected. **Actual cost: not yet verified.**

### Where the 6.26 h went — honest accounting

Only **2.94 h** was training. The remainder was overhead, much of it avoidable:

| | |
|---|---|
| 10 epochs (incl. 52 min cold cache build) | 2.94 h |
| Dataset validation scan, run twice (~25 min each) | ~0.8 h |
| ET census that died at 600/898 | ~0.5 h |
| shm crash, diagnosis and relaunch | ~0.3 h |
| Setup, image rebuild, gates, verification, teardown | ~1.7 h |

The census and the first training attempt were run concurrently on 8 vCPUs.
They thrashed (load 7.79), the census rate decayed 1.43 → 3.00 s/case, and the
L4 sat at 0% utilisation while both fought for CPU. That was an avoidable
operator error.

---

## N. Bugs found that would have broken the 300-epoch run

**1. `--shm-size` is not set by `cloud/scripts/_train_entrypoint.sh`.**

```
RuntimeError: DataLoader worker (pid 93) exited unexpectedly
ERROR: Unexpected bus error ... insufficient shared memory (shm)
```

Docker defaults `/dev/shm` to **64 MB**; the host had 16 GB. PyTorch DataLoader
workers pass 128³ volumes through shared memory and need far more. This crashed
the first 10-epoch attempt **after** the full dataset-validation scan had
already been paid for. Production uses `workers: 2`, so the 300-epoch run would
have failed identically.

*Fix required before the production run:* add `--shm-size=8g` to both
`docker run` invocations in `_train_entrypoint.sh`. Not applied here, as the
brief forbids unapproved Phase 2 changes.

**2. The preprocessing cache is ephemeral on GCP.**

The Phase 1 fingerprint cache writes to `/app/.cache/` *inside* the container.
With `--rm`, it is destroyed on exit, so the dataset-validation scan (~25 min)
and the 40 GB first-pass volume cache (~52 min of epoch 1) are rebuilt on every
launch. It works locally, never on GCP.

*Fix suggested:* mount a persistent cache, e.g. `-v /out/cache:/app/.cache`.
For a single 300-epoch run this costs ~77 min once; for any restart or resume it
is paid again.

---

## O. Results are not a thesis result

```
10-epoch run  = engineering / pipeline / training validation
300-epoch run = future thesis production experiment
```

The model reached validation mean Dice **0.3812 at epoch 9** (WT 0.4481 /
TC 0.3562 / ET 0.3393), with loss falling and Dice rising monotonically. That
demonstrates the pipeline trains correctly and improves.

It does **not** demonstrate segmentation quality. The run used a non-production
learning rate (0.0004 vs 0.0016), ran for 10 of a budgeted 300 epochs, and is
nowhere near converged. **No final Dice is forecast from these numbers, no
convergence is claimed, no clinical competitiveness is claimed, and no A/B
improvement is claimed.**

---

## P. Final gate

```
PHASE 2 — 10-EPOCH VALIDATION PASS
```

Every safety-critical gate passed:

| Gate | Result |
|---|---|
| Repository SHA (local = remote = VM) | PASS |
| GCS service-account create/read/list/delete + checksum | PASS |
| GPU environment (L4, driver, CUDA, BF16) | PASS |
| Production model identity (20/20, on the L4) | PASS |
| Real-data GPU smoke (32/32) | PASS |
| 10 real production epochs, no NaN/Inf/shape error | PASS |
| Checkpoint creation (best + last + top_k + periodic) | PASS |
| GCS upload, object exists, read-back, checksum MATCH | PASS |
| Resume from the GCS copy (12/12) | PASS |
| Spot recovery drill (simulated disk loss, real bucket) | PASS |
| Test firewall (0/198 accessed) | PASS |
| Artifacts recovered locally | PASS |
| VM stopped | PASS — TERMINATED |

### 300-epoch readiness

```
NOT READY — two entrypoint fixes required first
```

The infrastructure is proven, but the pipeline as committed would crash:

1. **`--shm-size=8g` must be added to `_train_entrypoint.sh`** (section N.1).
   Without it the 300-epoch run dies at first data load, after paying for the
   validation scan. This is a hard blocker.
2. **Persistent cache mount** (section N.2) — not a blocker, but saves ~77 min
   of L4 time per launch and far more across any resume.

Once fix 1 is applied and pushed, the repository is ready for the 300-epoch
Spot production run. The full ET census (section E) remains outstanding but
does not gate it.

---

## Q. Artifacts

Downloaded to `reports/phase2_gcp_10epoch/` and verified readable:

```
train.csv                        per-epoch loss, LR, epoch_seconds, VRAM
validation.csv                   per-epoch Dice/IoU/HD95 per region
gpu_diagnostics_summary.json     791 samples, active-only utilisation
dataset_validation_report.json   1296/1296 PASS
training.log                     full run log
config.yaml                      the exact config used
status.json                      final run state
```

Also synchronised to
`gs://glo-nca-v3-even-continuity-501915/experiments/PHASE2-10EP-VALIDATION/`,
including both checkpoints, verified by read-back and checksum.

---

## R. Git status

```
commit          08cf30cd170cc4cf1fec3f33af8b858bc22644e2
                (27784a4 phase1 + 08cf30c phase2 config)
pushed          YES — origin/v3-multilevel, SHAs verified equal
this report     untracked, not committed
artifacts       reports/phase2_gcp_10epoch/ untracked, not committed
```

No Phase 2 code changes were committed. The two entrypoint fixes in section N
are **recommended, not applied**, and await approval.

---

## S. Final safety check

```
VM stopped                        YES — TERMINATED
test touched                      NO  — 0/198
300-epoch run started             NO
production architecture changed   NO
production LR changed             NO  — still 0.0016
GCS service account verified      YES — on the VM, as the service account
10 epochs completed               YES
artifacts recovered               YES
```
