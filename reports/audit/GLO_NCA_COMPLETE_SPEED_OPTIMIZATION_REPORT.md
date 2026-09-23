# GLO-NCA — COMPLETE SPEED OPTIMIZATION REPORT

> **NAMING NOTE (added after this report was written).**
> `src/models/Model_BasicNCA3D.py` was renamed to
> `src/models/Model_GLO_NCA_Cell.py`, and the class `BasicNCA3D` to
> `GLO_NCA_Cell`. The rename was cosmetic: no behaviour, tensor shape or
> parameter changed, and production identity stayed 29,337 / 75 / 29,412,
> verified from the constructed model before and after. Any file path, class
> name or line number cited below refers to that file under its former name;
> the evidence recorded here still stands.

**Status:** measurement and candidate evaluation only.
**Production source, production config and the thesis split were NOT modified.**
**No change has been promoted to production. No final thesis architecture selected.**

---

## 1. Executive Summary

One safe engineering optimization was found, proven mathematically equivalent, and
measured at **1.40×** on the full model with the parameter count unchanged at 33,089.
It is **not integrated** — it requires editing a protected file and shipping a
checkpoint-migration shim, both of which need explicit approval.

Six scientific candidates were measured in isolation and in combination. All are
**CATEGORY C** and remain isolated.

Three findings beyond the speed numbers:

1. **The fused-BatchNorm change breaks checkpoint loading** (`ncas.0.bn.bn.weight`
   → `ncas.0.bn.weight`). Any integration must carry a state-dict remap or every
   existing checkpoint and frozen thesis artefact fails to load.
2. **The documented cache size is wrong by 3.4×.** `configs/glo_nca_production.yaml:72`
   states 4.75 MB/case (~4.3 GB); the measured 96³ figure is **16.03 MB/case = 14.06 GB**.
   The 4.75 MB figure is exactly 64³ geometry — it was measured at the wrong resolution.
3. **Gradient-checkpoint granularity is already optimal.** Per-step checkpointing beats
   every coarser chunking tested. Nothing to gain; §10 closed.

---

## 2. Hardware / Software

| | |
|---|---|
| GPU | NVIDIA RTX 3050 6GB Laptop, sm_86, 20 SMs, 6144 MB |
| Python / PyTorch / CUDA | 3.12.9 / 2.5.1+cu121 / 12.1 |
| Precision | **bf16** (Ampere native) |
| Batch size | 1 |
| git SHA | `7476e18ae1182ce204da919f70a15887ad74820a` |
| Split SHA | `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d` |
| Split counts | 898 / 200 / 198 |

### CRITICAL MEASUREMENT CAVEAT (§18)

`nvidia-smi` reported **SW Power Cap: Active** and **SW Thermal Slowdown: Active**
for the entire session, with the SM clock pinned at **930 / 2100 MHz (44%)**.

A sustained-load probe confirmed this is a **hard driver/firmware lock, not heat**:
at 100% utilization the clock stayed at 930 MHz, 55 °C, drawing 34 W of a 70 W limit.
It never recovered.

**Consequence:** every absolute millisecond in this report is roughly **2.3× inflated**.
Ratios, speedups, equivalence results, VRAM figures and parameter counts are unaffected.
**Absolute epoch estimates must not be quoted as expected wall-clock time.**

---

## 3. Baseline Identity — VERIFIED (§0)

All 15 assertions passed before any experiment ran.

| Property | Value |
|---|---|
| Parameters | **33,089** |
| Working volume | 96³ |
| Level 1 | 48³, 24 ch, 20 steps, perception k=7 |
| Level 2 | 64³, 24 ch, 20 steps, perception k=3 |
| Level 3 | ABSENT |
| Total NCA steps | 40 |
| Spatial GC kernel | (7, 7, 7) |
| SE + spatial GC | both ON |
| Fusion | Conv3d(48 → 24) |
| Gradient checkpointing | ON |
| Patchify | OFF (structurally — `forward(self, mod_cl)` takes no patch argument) |
| Global context | reads full 96³ (`max|delta| = 0.460938` on outer-rim-only perturbation) |

**Baseline timing (frozen):** forward 1421.8 ms, backward 6818.4 ms (82.7%),
**total 8227.1 ms**, VRAM 1460 MB, bwd/fwd 4.80×.

> The harness reimplements production semantics from `src/models/Model_BasicNCA3D.py`
> (the double-transpose BatchNorm form), **not** from the session-modified Kaggle file.
> An earlier harness build silently inherited `SPATIAL_GC_KERNEL = 5` and produced
> 32,217 parameters; this was caught and a hard identity assertion added.

---

## 4. Measurement Methodology

CUDA events, forward and backward timed separately. Warm-up excluded and reported
separately. Median of ≥6 steady-state samples; p95 and spread recorded.

**Where a result looked anomalous it was re-measured rather than reported.** COMBO-A
measured 5856 ms three times and 7129 ms once — a 22% swing in an identical
configuration. A 4-round interleaved A/B test resolved it:

| round | BASE | COMBO-A | ratio | clock |
|---|---|---|---|---|
| 0 | 8224.9 | 5859.8 | 1.40× | 930 |
| 1 | 8278.0 | 5856.1 | 1.41× | 930 |
| 2 | 8229.1 | 5859.7 | 1.40× | 930 |
| 3 | 8225.1 | 5855.9 | 1.40× | 930 |

Spread: BASE 0.6%, COMBO-A 0.1%. **Verified COMBO-A = 1.40×.** The 7129 ms sample
is discarded as drift.

---

## 5–8. Safe Engineering Optimizations

### A1 — Fused BatchNorm

Production wraps `BatchNorm3d` in two transposes of the 128-channel tensor
(`Model_BasicNCA3D.py:162-164`) purely because `BatchNorm3d` requires channels-first.
Normalising over (B,X,Y,Z) per channel **is** a 2D batch-norm over the flattened
(N, C) view, so `F.batch_norm` on a reshape gives identical mathematics without the copies.

**Semantics preserved:** `track_running_stats=False` means `running_mean`/`running_var`
are `None` — no buffers exist, momentum is irrelevant, and batch statistics are used in
both train and eval. `F.batch_norm(..., None, None, ..., training=True)` reproduces this exactly.

**Equivalence proven (float64):**

| quantity | max abs difference |
|---|---|
| output | 2.665e-15 |
| d/dx | 1.844e-14 |
| d/dweight | 1.819e-12 |
| d/dbias | 9.095e-13 |
| relative output | 2.693e-16 |

Train and eval modes both match to 9.5e-07 in bf16.

**Isolated block speed:** 48³ 1.74×, 64³ 1.81×.

### A2 — Stochastic mask dtype

`stochastic.float()` forces an fp32 tensor under bf16 autocast, promoting the product.
The mask is bool either way; 0.0 and 1.0 are exactly representable in bf16/fp16/fp32,
so no value changes. `torch.rand()` is called identically, so **the RNG stream is
unchanged and reproducibility is preserved**.

**Measured: 1.00× — no benefit on its own.** Not proposed on speed grounds.

### Full-model results

| ID | Configuration | fwd | bwd | total | speedup | VRAM | params |
|---|---|---|---|---|---|---|---|
| BASE | production | 1418.0 | 6800.5 | 8218.7 | 1.00× | 1686 | 33,089 |
| **A1** | fused BN only | 1364.1 | **4501.6** | **5865.7** | **1.40×** | 1616 | 33,089 |
| A2 | mask dtype only | 1411.5 | 6797.0 | 8208.2 | 1.00× | 1706 | 33,089 |
| **COMBO-A** | A1 + A2 | 1357.8 | 4498.6 | **5857.9** | **1.40×** | 1632 | 33,089 |

**All of the gain is A1, and all of it is in backward** (6800 → 4502 ms).

### §3 Layout census

| site | operation | class |
|---|---|---|
| L158 | `x_in.transpose(1,4)` | REQUIRED (Conv3d needs NCHW) |
| L160 | `dx.transpose(1,4)` | REQUIRED (Linear acts on last dim) |
| **L162** | `dx.transpose(1,4)` | **REDUNDANT** (only for BatchNorm3d) |
| **L164** | `dx.transpose(1,4)` | **REDUNDANT** (pairs with L162) |
| L176 | `dx.transpose(1,4)` | REQUIRED |
| L177 | `x.transpose(1,4)` | REQUIRED |

4 of 6 are required. The 2 redundant ones are exactly what A1 removes —
**A1 *is* the layout fix.** They move 1.05 GB (L1) + 2.50 GB (L2) per iteration.

### §5 Autograd nodes

A1 removes exactly **80 `TransposeBackward0` nodes** (239 → 159) = 2 per step × 40 steps.
Total node count is unchanged (2,070) because `F.batch_norm` is one node where the
transposed chain was several — the *work per node* dropped, not the count.

### §4 / §6 — no further safe gains

`.clone()` and `.contiguous()` calls are all REQUIRED or AUTOGRAD-SAFETY. Removing the
NCA `.clone()` ran without error but **correctness under checkpoint recompute is not
proven by a successful run** — classified **E, inconclusive**, not adopted.

Buffer preallocation was **rejected on correctness grounds without measurement**: NCA
states are autograd-live across 40 steps, so in-place reuse would corrupt the graph.

---

## 9. Gradient Checkpointing (§10) — keep ON, granularity already optimal

| chunk size | total | vs chunk=1 | VRAM |
|---|---|---|---|
| **1 (production)** | **5856.8** | **1.00×** | 1398 |
| 2 | 5878.9 | 1.00× | 1268 |
| 4 | 5890.2 | 0.99× | 1704 |
| 10 | 5896.3 | 0.99× | 3478 |
| 20 | 9015.4 | 0.65× | 6602 |

Coarser chunking is never faster. **§10 closed, no change proposed.**

---

## 10. Scientific Candidates — ALL CATEGORY C (§11–§15)

Measured on top of COMBO-A. **None promoted.**

| ID | Change | total | vs BASE | VRAM | params |
|---|---|---|---|---|---|
| S15 | 15+15 steps | 4391.7 | 1.88× | 1136 | 33,089 |
| GC5 | spatial GC k=5 | 5127.8 | 1.61× | 1404 | **32,217** |
| L1K5 | L1 perception k=5 | 4983.9 | 1.65× | 1404 | **27,857** |
| GCsep | separable GC | 4791.8 | 1.72× | 1406 | **31,809** |
| G40/56 | geometry 40³/56³ | 3733.6 | 2.21× | 964 | 33,089 |
| G32/48 | geometry 32³/48³ | 2138.5 | **3.85×** | 614 | 33,089 |

**Parameter impact:** steps and geometry leave 33,089 untouched (an NCA shares one
update rule across all voxels). **L1 perception k=5 costs 16% of the parameters** —
by far the largest identity change.

**Separable GC (§14):** 42 vs 686 MACs per voxel (16.3× fewer), but a factorised kernel
is a **strictly smaller function class** than a dense 7³ kernel. **NOT EQUIVALENT** —
classified SCIENTIFIC VARIANT, requires segmentation metrics before any consideration.

**Geometry voxel-step accounting:** 48/64 = 7,454,720 (1.00×); 40/56 = 4,792,320 (0.64×);
32/48 = 2,867,200 (0.38×).

---

## 11. Combination Experiments (§25)

| ID | Contents | total | vs BASE | VRAM | params |
|---|---|---|---|---|---|
| BASE | production | 8227.1 | 1.00× | 1414 | 33,089 |
| **COMBO-A** | fused BN + mask dtype | **5857.9** | **1.40×** | 1404 | **33,089** |
| COMBO-B | A + 15+15 steps | 4460.9 | 1.84× | 1136 | 33,089 |
| COMBO-C | B + spatial GC k=5 | 3845.3 | 2.14× | 1136 | 32,217 |
| COMBO-D | C + L1 perception k=5 | 3202.5 | 2.57× | 1136 | 26,985 |
| COMBO-E | A + geometry 40/56 | 3730.1 | 2.21× | 964 | 33,089 |
| **COMBO-E2** | D + geometry 40/56 | **2053.4** | **4.01×** | 826 | 26,985 |
| COMBO-F | D + separable GC | 2950.1 | 2.79× | 1136 | 26,577 |

**Speedups do not multiply**, as the brief warned. COMBO-A (1.40×) and 15+15 (1.88×)
would naively predict 2.63×; COMBO-B **measured 1.84×**. Only measured combinations
are reported.

---

## 12. GPU-Specific (§17)

| precision | fwd | bwd | total | vs bf16 |
|---|---|---|---|---|
| bf16 | 1358.3 | 4499.7 | 5858.1 | 1.00× |
| fp16 | 1343.4 | 4461.1 | 5805.2 | 1.01× |
| TF32 + cudnn.benchmark | 1358.4 | 4498.6 | 5857.3 | 1.00× |

All within noise on Ampere.

- **Tesla T4 / L4: NOT MEASURED.** T4 is Turing (sm_75) where bf16 is *emulated* and
  fp16 has native tensor cores — **this ranking may invert there.** No extrapolation made.
- **`torch.compile`: NOT MEASURED** — requires Triton, unavailable on Windows.

---

## 13. Cache Discrepancy (§20) — RESOLVED

Measured by writing real arrays through the documented format
(`torch.save`, float32 image + uint8 labels):

| geometry | img | lab | **file** | 898 cases |
|---|---|---|---|---|
| **96³ (production)** | 13.50 MB | 2.53 MB | **16.03 MB** | **14.06 GB** |
| 128³ | 32.00 MB | 6.00 MB | 38.00 MB | 33.33 GB |
| **64³** | 4.00 MB | 0.75 MB | **4.75 MB** | 4.17 GB |

**The documented 4.75 MB/case is exactly 64³ geometry.** It was measured at 64³ but is
documented against a 96³ production config.

**The correct production figure is 16.03 MB/case = 14.06 GB, not 4.3 GB — a 3.4×
understatement of the real disk requirement.**

Per instruction, **no value was changed**; which number is correct was determined factually.

---

## 14. Validation Speed (§22)

| | inference/case | 200 cases |
|---|---|---|
| production | 1388.6 ms | 277.7 s/epoch |
| COMBO-A | 1400.0 ms | 280.0 s/epoch |

Validation is full-volume inference and is **unaffected by A1** (backward-only gain).
**No protocol or frequency change proposed.**

---

## 15. Correctness Gates (§26/§27) — ALL PASS

15/15 on COMBO-A: construction, parameter count 33,089, forward, no NaN/Inf, finite loss,
all 36 params received gradient, finite grad-clip, optimizer step, checkpoint
save/load/resume, optimizer-state restore.

**Global-context safety test:** with the central region held identical and only the outer
slabs perturbed, both production and COMBO-A produce `max|delta| = 0.460938` — **identical
to six decimal places.** Independent confirmation that A1 changes nothing observable.

> ET ⊆ TC ⊆ WT holds on the untrained model but is reported as INFO only: the loss trains
> the three regions as independent sigmoids, so hierarchy is a property of a trained model,
> not an architectural gate.

---

## 16. NOT MEASURED — honest gaps

| Item | Why |
|---|---|
| §19 real-data pipeline | No NIfTI files in this working copy |
| §21 DataLoader worker sweep | Same — a synthetic sweep would measure only tensor copying |
| §17 T4 / L4 | Hardware unavailable; no extrapolation from sm_86 |
| `torch.compile` | Triton unavailable on Windows |
| Segmentation metrics (Dice/HD95) for every candidate | Requires real training on real data; **no candidate has been validated for accuracy** |

**Existing real-data evidence** (user's Kaggle run, workers=2): `train/dataloader_wait`
= 276.6 s of 5126.7 s = **5.4%** — the GPU was not primarily data-starved there.

---

## 17. Issue Classification (§29)

| Item | Class | Evidence |
|---|---|---|
| A1 fused BatchNorm | **B** — equivalence proven; needs checkpoint remap + training validation | fp64 output 2.7e-15, all gradients < 1.9e-12 |
| A2 mask dtype | **B** (D on speed) | 1.00× — no benefit alone |
| Keep checkpointing ON | **A** — already correct | OFF is 3.6× slower, 6.7× VRAM |
| Checkpoint granularity = 1 | **A** — already optimal | coarser never faster |
| NCA steps 15+15 | **C** | 1.88×, params unchanged |
| Spatial GC k=5 | **C** — thesis contribution | 1.61×, params 32,217 |
| L1 perception k=5 | **C** | 1.65×, **params 27,857 (−16%)** |
| Separable GC | **C** — NOT equivalent | 1.72×, smaller function class |
| Geometry 40/56, 32/48 | **C** | 2.21× / 3.85× |
| Fusion, loss, optimizer, EMA, H2D | **D** | all < 1% in prior audit |
| Cache size discrepancy | **resolved factually** | 4.75 MB = 64³, correct 96³ = 16.03 MB |
| Removing NCA `.clone()` | **E** | ran, but correctness not proven |
| Real-data pipeline, DataLoader, T4/L4 | **E** | not measurable here |

---

## 18. Integration Blocker for A1

State-dict keys differ at the real nesting level:

```
production : ncas.0.bn.bn.weight, ncas.0.bn.bn.bias
fused      : ncas.0.bn.weight,    ncas.0.bn.bias
-> RuntimeError: Missing key(s) / Unexpected key(s)
```

**A naive swap breaks every existing checkpoint and every frozen thesis artefact.**
Any integration must include a `_load_from_state_dict` remap of legacy `bn.bn.*` keys.
The full patch, including that remap, is prepared and awaiting review.

**Files that would be touched:** `src/models/Model_BasicNCA3D.py` — a **protected file**.

---

## 19. Final Comparison (§33)

Epoch estimates assume 898 training cases. **Absolute times come from a GPU locked at
44% clock — use the ratios, not the hours.**

| Configuration | steps | L1 | L2 | GC | params | iter (ms) | VRAM | epoch | status |
|---|---|---|---|---|---|---|---|---|---|
| **ORIGINAL PRODUCTION** | 20+20 | 48³ k7 | 64³ k3 | k7 | 33,089 | 8227.1 | 1414 | 2.05 h | **frozen baseline** |
| **SAFE-OPTIMIZED (COMBO-A)** | 20+20 | 48³ k7 | 64³ k3 | k7 | **33,089** | **5857.9** | 1404 | 1.46 h | **candidate, not applied** |
| 15+15 | 15+15 | 48³ k7 | 64³ k3 | k7 | 33,089 | 4460.9 | 1136 | 1.11 h | CATEGORY C |
| GC k=5 | 15+15 | 48³ k7 | 64³ k3 | k5 | 32,217 | 3845.3 | 1136 | 0.96 h | CATEGORY C |
| L1 k=5 | 15+15 | 48³ k5 | 64³ k3 | k5 | 26,985 | 3202.5 | 1136 | 0.80 h | CATEGORY C |
| geometry 40/56 | 20+20 | 40³ k7 | 56³ k3 | k7 | 33,089 | 3730.1 | 964 | 0.93 h | CATEGORY C |
| **most aggressive (E2)** | 15+15 | 40³ k5 | 56³ k3 | k5 | 26,985 | 2053.4 | 826 | 0.51 h | CATEGORY C |

**No final thesis architecture is selected. No candidate is promoted.**

---

## 20. Recommended Next Investigation

1. **Correct the cache figure in `configs/glo_nca_production.yaml:72-73`** — a factual
   error affecting RAM/disk planning for the 898-case run, independent of any optimization.
2. **Decide on A1** — proven equivalent, 1.40×, zero identity change, but requires editing
   a protected file plus a checkpoint-migration shim.
3. **Re-measure on unthrottled hardware (your T4)** for trustworthy absolutes and to settle
   bf16-vs-fp16, which may invert on Turing.
4. **Only then** consider Category C candidates, each of which needs Dice/HD95 validation
   on real training before it can be judged — speed alone cannot justify any of them.

---

## 21. Safety Verification (§34)

- [x] Production source unchanged — `Model_BasicNCA3D.py` 09-15, `Model_GLO_NCA_GlobalContext.py` 09-17, `preprocess_cache.py` 09-16, `runner.py` 09-19 00:01, `train.py` 09-16; all predate this work (session ran 21:36–22:30 on 09-19)
- [x] `configs/glo_nca_production.yaml` unchanged (09-19 14:01)
- [x] Thesis split unchanged — SHA `d30d719…`, 898/200/198
- [x] No GCP, no VM, no cloud execution, no cloud cost
- [x] Checkpointing never disabled in any proposed configuration
- [x] No architecture silently changed; all Category C candidates isolated
- [x] No git commit, no git push (HEAD still `7476e18`)
- [x] No thesis evidence deleted or modified
- [x] All 14 profiling scripts isolated in the session scratch directory, never imported by production code
- [x] No estimate presented as a measurement; all gaps labelled NOT MEASURED
