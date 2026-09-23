# GLO-NCA — Category-C Final Readiness Report

Engineering and training-readiness assessment of the active repository.
Every figure below was measured or verified locally; nothing is estimated,
and nothing is carried over from documentation without re-checking it.

---

## A. Git

| | |
|---|---|
| HEAD | `6ab094024257a75ac8198a9328ed01596d0fd162` |
| Branch | `v3-multilevel` |
| Remote | `origin` (Muhammadwaqas1234/GLO-NCA) |
| Push | verified — local HEAD == `origin/v3-multilevel` |
| Working tree | clean |

No history rewrite, no force push, no amended published commit. `main` and
`v2` untouched.

---

## B. Final Architecture — **29,337 parameters**

Built from `configs/glo_nca_production.yaml` alone, with no hard-coded
patching. The identity artifact is derived from the **constructed model**,
not from the config or from documentation.

| | |
|---|---|
| parameters | **29,337** |
| working volume | 96³ |
| Level 1 | 48³ / 24 ch / 15 NCA steps / perception k=5 |
| Level 2 | 64³ / 24 ch / 15 NCA steps / perception k=5 |
| Level 3 | absent |
| total NCA steps | 30 |
| spatial global context | k=5 |
| SE | ON |
| fusion | learned Conv3d(48 → 24) |
| global-context source | full 96³ working volume |
| patchify | **OFF** (explicitly resolved, not left ambiguous) |
| ROI | 1.0 |
| batch | 1 |
| seed | 42 |
| BatchNorm | `ChannelsLastBatchNorm` (A1) |

The identity gate is **fail-closed**: changing `spatial_kernel_size` from 5
to 7 produces `FAIL … 30209 expected 29337`, verified by injection.

---

## C. Dataset and Frozen Split

| | |
|---|---|
| train / validation / test | **898 / 200 / 198** |
| split SHA-256 | `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d` |
| modification | none — file unchanged all session |

---

## D. Precision Benchmark — MEASURED

RTX 3050 (sm_86), batch 1, 3 warm-up + 8 timed iterations:

| precision | iteration | std | peak allocated |
|---|---|---|---|
| fp32 | 4380.5 ms | 1.6 | 1316 MB |
| **bf16** | **3997.2 ms** | **2.0** | 979 MB |
| fp16 | 4004.0 ms | **55.3** | 979 MB |

bf16 and fp16 are within noise on speed, but fp16's spread is **27× wider**,
so bf16 wins on stability as much as on time. The production config already
specifies bf16; no setting was changed.

**The GPU was power-capped at 930 / 2100 MHz (44%) throughout.** Absolute
times are inflated; the ratio between precisions is the usable result.

**NVIDIA L4: NOT MEASURED. Tesla T4: NOT MEASURED.** No such measurement
exists in this repository, and none has been inferred.

---

## E. GPU Diagnostics

Samples carry an explicit phase (`IDLE`, `DATA_WAIT`, `ACTIVE_GPU`,
`VALIDATION`, `CHECKPOINT`), and utilization is summarised **only from
`ACTIVE_GPU`** samples. A reading taken while a NIfTI is loading describes
the pause, not the computation — that distinction is the point of the gate.

Sampling is rate-limited because `nvidia-smi` costs tens of milliseconds and
polling per step would distort the timings being collected. Anything the
hardware does not expose is recorded `NOT AVAILABLE`.

The summary warns when `ACTIVE_GPU` utilization is below 50% or the SM clock
is below 75% of maximum, naming what to investigate rather than adjusting a
scientific setting to mask an infrastructure problem.

---

## F. Timing

`PhaseTimer` synchronises CUDA **at section boundaries only** and reports the
overhead that synchronisation adds — an instrument that hides its own cost
cannot be trusted. Sections nest, and every emitted summary states that
nested percentages **do not sum to 100%**.

---

## G. Artifact System — 16/16

`run_metadata.json` · `architecture_identity.json` · `hardware_metadata.json`
· `precision_benchmark.json` · `timing_summary.json` · `epoch_metrics.csv` ·
`validation_history.csv` · `checkpoint_manifest.json` ·
`best_checkpoint_metadata.json` · `top3_checkpoint_metadata.json` ·
`periodic_checkpoint_metadata.json` · `final_checkpoint_metadata.json` ·
`resume_verification.json` · `overfitting_report.json` ·
`gpu_diagnostics.csv` · `final_experiment_report.md`

A value that cannot be obtained is written `NOT MEASURED`, `NOT AVAILABLE` or
`BLOCKED`. A fabricated zero would look like evidence, which is worse than an
absent field. Concretely: an absent precision benchmark records
`NOT MEASURED` rather than synthesising numbers; fewer than three validation
points yields `NOT MEASURED` rather than an invented trend; an unexecuted
resume check says so rather than claiming verification.

`audit_consistency()` re-reads what was written and fails if identity fields
diverge. Verified to catch an injected 33,089-vs-29,337 mismatch and to pass
again once corrected. A mismatch is reported, never silently repaired.

---

## H. Checkpoint System

`best.pth` (validation metric only) · `top_k/best_1..3.pth` (ranked, **never
averaged, no SWA**) · periodic snapshots **every 5 epochs** · `last.pth`
final, **distinct from best**. Evaluation always loads best, never final.

---

## I. Resume Verification

Verified on **real BraTS data**: model weights, optimizer, scheduler, EMA,
epoch, global step, validation history, early-stopping patience and
state-machine state all restore correctly.

Early-stopping state is persisted into every checkpoint. Without it the
patience counter resets on each restart, and a plateaued run would train
another full patience window after every preemption — which matters directly
for Spot.

---

## J. Early Stopping

Monitor = validation mean foreground Dice (WT/TC/ET), **3-epoch rolling
mean** · patience 15 · min_delta 0.001 · mode max.

Training loss is deliberately **not** monitored: a falling training loss is
exactly what overfitting looks like.

---

## K. State Machine

Eleven states, enumerated transitions, persisted atomically to
`reports/state.json`. Invalid transitions raise rather than being recorded —
a run that silently jumped `CREATED → COMPLETED_300` would produce a
directory that lies about what ran.

Two distinctions the format protects: a **resumed** run must not look like a
new experiment, and an **extended** run must not look like the original
budget. Both are recorded as history, not inferred afterwards.

---

## L. Test Firewall

AST-based, so comments and docstrings are structurally invisible — grepping
for "test" would flag the codebase's own documentation. Verified by injecting
deliberate leakage (`len(te)` beside the stop decision): the audit failed
with `refs=['te']`, and passed again on revert. **An audit that cannot fail
proves nothing.**

The test split is never read for early stopping, best-checkpoint selection,
top-k ranking, or extension decisions.

---

## M. Repository Cleanup

**Fixed — a live default that trained the wrong architecture.**
`scripts/local_gpu_smoke_test.py` defaulted `--config` to
`configs/gcp_full.yaml` (the V2 baseline). Now defaults to the production
config.

**Fixed — documentation naming the wrong config as production.** `train.py`
advertised `v3_multilevel_ckpt.yaml` as "production V3" (it is the frozen
reference); the cloud scripts advertised V2-era configs. All now name
`glo_nca_production.yaml`.

**Fixed — the Kaggle docstring** claimed 33,089 while the file asserts 29,337.

**Not deleted, deliberately.** The five `configs/ablation_*.yaml` have no code
references and appeared obsolete. They were removed, then **restored**:
`scripts/verify_phase3_ready.py` reads them by name, and
`REAL_DATA_V3_VALIDATION_REPORT.md` cites their parameter counts as thesis
evidence. Deleting them would have broken a working audit and orphaned frozen
evidence to make a directory listing shorter.

**Defect introduced and fixed during this work:** a Windows text-mode write
converted the two cloud shell scripts to CRLF. `bash -n` passed, but
`phase1_closure_audit` caught it — `\r` breaks shell parsing. Both restored
to LF; audit passes.

---

## N. Regression Results

| suite | result |
|---|---|
| new-gate regression (8 suites) | **363 pass / 0 fail** |
| production identity | 30/30 |
| production config gate | 24/24 |
| phase1 closure audit | PASS |
| phase2 final verification | PASS |
| preprocess cache | 7/7 |
| preflight logic | 6/6 |
| Kaggle self-check | PASS |
| `compileall` (src, scripts, train.py, kaggle) | PASS |

**Known pre-existing failure, untouched:** `scripts/verify_phase3_ready.py`
is a frozen v2-era audit asserting branch `v2` and an unmodified
`Agent_GLO_NCA_V3.py`. Both are false by design on `v3-multilevel`, and it
was already failing before this work. It was neither weakened nor deleted.

---

## O. GCP

**VM created: NO · Spend: $0 · Training executed: NO · Spot-ready: YES**

Spot readiness rests on verified resume, 5-epoch periodic snapshots, and
persisted early-stopping and lifecycle state. Provisioning is documented in
`docs/operations/SPOT_L4_RUNBOOK.md` and was **not performed**.

---

## P. Scientific Quality Boundary

**NO QUALITY CLAIM WITHOUT REAL TRAINING/EVALUATION.**

The 29,337-parameter architecture has **no Dice, IoU or HD95 evidence**. It
was promoted on speed grounds by explicit decision. The k7 → k5 change
shrinks the receptive field of the spatial global-context block, which *is*
the thesis contribution, and that cost has not been measured.

Parameter count and timing say nothing about segmentation quality. This
report certifies infrastructure, not science.

---

## Q. Gate Matrix

| Gate | Status |
|---|---|
| §17 Precision benchmark | **COMPLETE** |
| §18 GPU diagnostics | **COMPLETE** |
| §19 Timing instrumentation | **COMPLETE** |
| §21 Artifact expansion | **COMPLETE** |
| §26 Training state machine | **COMPLETE** |
| §3 / §20 Obsolete-code cleanup | **COMPLETE** |
| A1 equivalence | COMPLETE |
| Early stopping / top-k / periodic / final | COMPLETE |
| Post-300 extension (freeze policy) | COMPLETE |
| Test firewall | COMPLETE |
| Checkpoint fingerprint | COMPLETE |
| Kaggle standalone identity | COMPLETE |
| Real-data smoke | COMPLETE |
| `fast_grid4864` comparison | **NOT REQUIRED** — not the selected model |

---

## R. Certification

# CERTIFIED — ENGINEERING/TRAINING READINESS

The active repository is ready to execute the GLO-NCA training experiment:
one production config, one entry point, one architecture, verified
checkpoint/resume, enforced test firewall, and a complete artifact trail.

This certification covers **infrastructure only**. Segmentation quality
remains **NOT VALIDATED WITHOUT REAL THESIS TRAINING/EVALUATION**.
