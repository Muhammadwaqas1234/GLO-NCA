# PHASE 2 — TRACE ONE COMPLETE TRAINING ITERATION

**Local implementation only. GCP VMs created: 0. GCP GPU used: none. Cloud cost: $0.**
**Production training runs: 0.**

Branch `v3-multilevel` · Phase 1 closure commit `6dbc464`

---

## 1. What this phase delivers

A reusable, observational profiling framework that traces **one complete
training iteration** end to end — disk → NIfTI → preprocessing → patchify →
DataLoader → H→D → V3 levels 1/2/3 → fusion → loss → backward → grad-clip →
optimizer → EMA → checkpoint — and traces **validation separately**.

**It measures. It does not optimize.** No architecture, resolution, NCA-step,
loss, optimizer, EMA, split, threshold or checkpoint-policy change was made.

## 2. Framework

| File | Role |
|---|---|
| `src/profiling/__init__.py` | public API (`get_profiler`, `configure`, `write_reports`) |
| `src/profiling/timer.py` | CUDA-aware section timing; true no-op when disabled; warmup handling; per-iteration + out-of-band statistics |
| `src/profiling/memory.py` | host RSS / system RAM / GPU allocated / reserved / peak kept strictly separate; verified release; FIT-vs-SPILL classification |
| `src/profiling/cuda.py` | bounded `torch.profiler` window + device identity |
| `src/profiling/report.py` | component table, contributor ranking, memory gauges, validation table |
| `configs/profile_local_v3.yaml` | **non-production** local harness config |

### Disabled by default — verified

Production configs contain no `profiling:` section, so `configure()` returns a
`NullProfiler`. Measured: **200,000 disabled sections cost 99.7 ms total
(~0.5 µs each)** and insert **zero CUDA synchronisation**, so normal training is
unaffected.

```
default profiler: NullProfiler  | enabled: False
configs/v3_multilevel_ckpt.yaml -> NullProfiler | enabled: False
```

### CUDA correctness

CUDA is asynchronous: `perf_counter()` around a GPU call times the *launch*, not
the work. GPU sections are marked `cuda=True` and synchronised on both edges, so
the reported milliseconds are real. Pure-CPU sections (NIfTI, crop, resample,
patchify) are **not** synchronised — we do not manufacture sync points that
production does not have.

## 3. Bugs found and fixed (all pre-existing, surfaced by this work)

| # | Defect | Impact | Fix |
|---|---|---|---|
| 1 | **`Experiment` holds a `SummaryWriter`** (thread lock) and the dataset holds `self.exp` | `TypeError: cannot pickle '_thread.lock'` — **any `workers > 0` run fails on Windows spawn**; production uses `workers: 4` | `__getstate__` drops the writer (parent-only object; workers never use it) |
| 2 | **`_worker_init` was a nested closure; `_EpochSampler` a local class** | `AttributeError: Can't get local object 'run.<locals>._worker_init'` — again fatal for `workers > 0` on spawn | hoisted both to module level as picklable `_WorkerInit` / `_EpochSampler`; seeding formula and `(epoch, index)` contract unchanged |
| 3 | Profiler/MemorySampler held torch + psutil handles | would have reintroduced the same pickling failure | `__getstate__`/`__setstate__` drop and lazily restore them |
| 4 | Profiling data lost when a run failed | measurements are the deliverable | `_dump_profiling_on_exit()` persists traces on the failure path too |

Defects 1 and 2 are **latent production bugs on Windows**, independent of
profiling. On Linux (fork) they would not fire, which is why they survived to
now. Both fixes are behaviour-preserving.

## 4. Local harness configuration (NOT production)

`configs/profile_local_v3.yaml` — exists to validate the *measurement system* on
hardware that cannot hold 128³.

| | Harness | Production |
|---|---|---|
| Levels | 16 / 24 / 32 | **32 / 96 / 128** |
| NCA steps | 4 + 4 + 3 = 11 | **20 + 20 + 10 = 50** |
| hidden | 64 | **128** |
| Cases | 20 (seeded split) | 898 / 200 / 198 canonical |
| workers | **0** (see below) | 4 |
| Loss / optimizer / EMA / clip / seed / ckpt | **identical to production** | identical |

`workers: 0` is required for input-pipeline attribution: with `workers > 0` the
dataset runs in spawned processes whose profiler copy is discarded, leaving only
`data/dataloader_wait` visible. The harness trades production's worker count for
visibility, and this report says so rather than hiding it.

**Timings below are NOT production performance.** The production 128³ numbers
remain **GCP REQUIRED — NOT EXECUTED LOCALLY**.

## 5. Environment

```
GPU      NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute 8.6, 6.0 GB)
PyTorch  2.5.1+cu121   CUDA build 12.1
Python   3.12.9
Host RAM 15.6 GB
Data     real BraTS-MET (local cohort), 4 modalities + seg
```

## 6. Measured training iteration (harness scale)

Bounded to 2 warmup + 4 profiled iterations; warmup excluded from statistics.

| Component | Mean (ms) | % iteration |
|---|---:|---:|
| `data/dataloader_wait` | 2400.88 | 231.0% † |
| `data/nifti_materialize` | 1517.05 | input pipeline* |
| `train/backward` | 485.70 | 46.7% |
| `train/forward` | 475.80 | 45.8% |
| `data/foreground_crop` | 347.15 | input pipeline* |
| `model/level2` | 225.26 | 21.7% |
| `model/level1` | 166.44 | 16.0% |
| `data/resample` | 154.45 | input pipeline* |
| `model/level3` | 69.68 | 6.7% |
| `train/optimizer_step` | 28.76 | 2.8% |
| `train/ema` | 19.37 | 1.9% |
| `train/loss` | 17.15 | 1.7% |
| `data/nifti_header` | 14.90 | input pipeline* |
| `model/fusion` | 13.11 | 1.3% |
| `train/grad_clip` | 8.93 | 0.9% |
| `data/augment` | 4.26 | input pipeline* |
| `train/h2d_transfer` | 1.03 | 0.1% |
| `data/labels_to_regions` | 0.66 | input pipeline* |
| `train/zero_grad` | 0.63 | 0.1% |
| `data/patchify` | 0.34 | input pipeline* |
| **`total/iteration`** | **1043.99** | **100%** |

\* `data/*` stages run inside the input pipeline, not inside the timed
iteration, so no "% of iteration" is computed for them.
† `data/dataloader_wait` **is** inside the timed loop — it is the time the
training loop sat blocked waiting for a batch. It exceeds 100% because with
`workers: 0` the whole input pipeline is serialised into that wait.

### Evidence-supported observations (harness scale only)

1. **NIfTI materialisation dominates the input pipeline**: 1517 ms vs 14.9 ms for
   header load — a ~100× split. Confirms the cost is gzip decompression + voxel
   materialisation in `get_fdata()`, not file opening.
2. **`backward` ≈ `forward`** (486 vs 476 ms). With gradient checkpointing ON,
   backward carries NCA recompute, so this ratio is expected and is now measured
   rather than assumed.
3. **Level 2 is the most expensive level** (225 ms), consistent with it carrying
   the largest voxels × steps product at harness scale.
4. **`patchify` is 0.34 ms** — the Phase 1 hoist left it negligible; it is *not*
   a cost centre.
5. **H→D transfer is 1.03 ms** — not a bottleneck at this scale.

**No component is declared "the bottleneck" for production**: these are
harness-scale numbers on a different resolution and worker configuration.

## 7. Validation — measured separately

This is the key instrument for the ~90-minute-epoch question, because a slow
epoch may be validation rather than training.

| Section | Count | Mean (ms) | Total (ms) |
|---|---:|---:|---:|
| `validation/total` | 1 | 9007.62 | **9007.62** |
| `validation/val_inference` | 6 | 270.79 | 1624.73 |
| `validation/hd95` | 63 | 18.76 | **1181.88** |
| `validation/test_inference` | 3 | 323.22 | 969.66 |
| `train/checkpoint_write` | 1 | 71.14 | 71.14 |
| `validation/threshold_dice` | 63 | 0.87 | 54.70 |
| `validation/iou` | 63 | 0.84 | 52.72 |
| `validation/val_probabilities` | 6 | 0.70 | 4.20 |

**Observation:** at harness scale a single validation pass (9.0 s) costs ~8.6×
one training iteration (1.04 s), and **HD95 alone (1182 ms) costs more than
Dice + IoU combined (107 ms)** — roughly 11× per call. That is a measured,
reproducible signal worth carrying into the production trace.

## 8. Memory

Host RSS, system RAM, GPU allocated, GPU reserved and GPU peak are sampled per
iteration and reported as **distinct** quantities, never mixed. Verified release
(`gc.collect()` → `empty_cache()` → re-sample) reports actual before/after
numbers rather than asserting that release happened.

`MemorySampler.classify_peak()` marks a peak above ~95% of physical VRAM as
**SPILL**, never FIT — Windows CUDA sysmem fallback pages to host RAM instead of
raising OOM, and a spill must never be reported as a fit.

## 9. PyTorch profiler

Bounded window (`wait=1, warmup=1, active=3`) exporting a Chrome trace to
`<run>/profiler/*.pt.trace.json`. Verified generated. Deliberately **not** run
over a whole epoch: that produces multi-GB traces and distorts the timings.
Profiling artifacts are git-ignored.

## 10. Tests executed

| # | Test | Result |
|---|---|---|
| 1 | import / compileall (`src`, `scripts`, `train.py`) | **PASS** |
| 2 | Phase 1 closure audit (39 checks) | **PASS 39/39** |
| 2 | Phase 2 regression (16 checks) | **PASS 16/16** |
| 2 | Local GPU verification (24 checks) | **PASS 24/24** |
| 3 | Profiling **disabled** → NullProfiler, ~0.5 µs/section, no CUDA sync | **PASS** |
| 4 | Local profiling run, real BraTS-MET data, exit 0 | **PASS** |
| 5 | Bounded `torch.profiler` Chrome trace | **PASS** |
| 6 | Validation profiling (inference, probabilities, Dice, IoU, HD95) | **PASS** |
| — | 128³ production trace | **NOT RUN — GCP REQUIRED** |

## 11. Test-set discipline

Unchanged. Profiling adds timing only: thresholds are still tuned on validation,
the test set is still collected once, and no checkpoint or parameter is selected
using test metrics. `validation/test_*` sections time the **existing** single
test pass; they do not introduce an extra evaluation.

## 12. Potential optimizations — NOT IMPLEMENTED

Recorded for a later, separately approved phase. **Nothing here was changed.**

1. `data/nifti_materialize` (1517 ms) dominates the input pipeline — a caching or
   decompression strategy *might* help, but production uses persistent workers
   whose cache behaviour differs; must be re-measured at production scale first.
2. `validation/hd95` (1182 ms over 63 calls) is ~11× Dice/IoU per call.
3. `data/foreground_crop` (347 ms) is a full-volume boolean reduction.
4. `backward ≈ forward` is consistent with checkpoint recompute — quantifying the
   exact checkpointing overhead needs an explicitly labelled A/B experiment, not
   a change to the production trace.

## 13. Limitations

1. **Harness scale ≠ production.** 16/24/32 with 11 NCA steps, not 32/96/128 with
   50. Do not quote these timings as production performance.
2. **`workers: 0`** for attribution; production runs `workers: 4`, which moves
   the input pipeline into parallel processes.
3. **Worker-side attribution** is unavailable with `workers > 0` (spawned
   profiler copies are discarded). Only `data/dataloader_wait` is visible then.
4. **128³ cannot run locally** — 6 GB card, spills to host RAM.
5. `p95` is suppressed below 5 samples rather than computed from noise.

## 14. Status

```
GCP VM CREATED:              NO
GCP GPU USED:                NO
CLOUD COST:                  $0
Production training runs:    0

128³ production trace:       NOT RUN LOCALLY — GCP REQUIRED
```

The measurement system is complete and validated locally. The next step is a
separate, explicitly approved decision about what must be measured at 128³ on
GCP — bounded, then VM terminated.
