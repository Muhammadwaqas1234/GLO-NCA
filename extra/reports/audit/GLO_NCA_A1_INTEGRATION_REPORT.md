# GLO-NCA — A1 INTEGRATION REPORT

> **NAMING NOTE (added after this report was written).**
> `src/models/Model_BasicNCA3D.py` was renamed to
> `src/models/Model_GLO_NCA_Cell.py`, and the class `BasicNCA3D` to
> `GLO_NCA_Cell`. The rename was cosmetic: no behaviour, tensor shape or
> parameter changed, and production identity stayed 29,337 / 75 / 29,412,
> verified from the constructed model before and after. Any file path, class
> name or line number cited below refers to that file under its former name;
> the evidence recorded here still stands.

**Scope:** integration of A1 (fused channels-last BatchNorm) into the production
model. This report is separate from, and does not modify,
`GLO_NCA_COMPLETE_SPEED_OPTIMIZATION_REPORT.md`.

---

## 1. Decision

**APPLIED.**

All gates in §4, §6–§13 and §17 of the integration brief passed. A1 is an
**implementation-level performance optimization preserving the mathematical
BatchNorm operation and the GLO-NCA architecture.** It is not a new
architecture, not a scientific contribution, not a GLO-NCA variant, not an
ablation and not a new global-context formulation.

**Measured speedup: 1.402× (paired, interleaved, n=4 rounds).**

---

## 2. Why

`update()` carries its hidden activation channels-last `(B, X, Y, Z, C)`, but
`BatchNorm3d` requires channels-first. The production code therefore transposed
a 128-channel tensor into NCHW and back around every normalisation:

```python
dx = dx.transpose(1,4)      # 27 MB (48³) / 64 MB (64³) copy
dx = self.bn(dx)
dx = dx.transpose(1,4)      # and again
```

Across 40 NCA steps that moves ≈3.5 GB per iteration and adds 80 autograd nodes
per iteration, all purely to satisfy a layout requirement. Normalising over
`(B, X, Y, Z)` per channel on a channels-last tensor **is** a 2D batch-norm over
the flattened `(N, C)` view, so `F.batch_norm` on a reshape (a view, not a copy)
computes the identical function without either transpose.

---

## 3. Files Changed

| File | Change |
|---|---|
| `src/models/Model_BasicNCA3D.py` | **only file modified** — 104 insertions, 3 deletions |

The three deleted lines are exactly:

```
-        self.bn = torch.nn.BatchNorm3d(hidden_size, track_running_stats=False)
-        dx = dx.transpose(1,4)
-        dx = dx.transpose(1,4)
```

Three change sites: new `ChannelsLastBatchNorm` class (after imports),
construction at former line 112, transpose pair at former lines 162–164.
No refactoring, no renames, no unrelated edits.

**Not modified:** `Model_GLO_NCA_GlobalContext.py`, `Model_GLO_NCA_V3.py`,
`Agent_GLO_NCA_V3.py`, `runner.py`, `train.py`, `configs/glo_nca_production.yaml`,
`split/master_split.json`.

---

## 4. Production Identity — Before / After

| Property | Before | After |
|---|---|---|
| Parameters | 33,089 | **33,089** |
| Working volume | 96³ | 96³ |
| Level 1 | 48³, 24 ch, 20 steps, k=7 | 48³, 24 ch, 20 steps, k=7 |
| Level 2 | 64³, 24 ch, 20 steps, k=3 | 64³, 24 ch, 20 steps, k=3 |
| Level 3 | ABSENT | ABSENT |
| Total NCA steps | 40 | 40 |
| Spatial GC | dense (7,7,7) | dense (7,7,7) |
| SE | ON | ON |
| Fusion | Conv3d 48→24 | Conv3d 48→24 |
| Global-context source | full 96³ (roi_fraction 1.0) | full 96³ (roi_fraction 1.0) |
| Patchify | OFF | OFF |
| Gradient checkpointing | ON, granularity 1 | ON, granularity 1 |
| Batch | 1 | 1 |
| Split | 898 / 200 / 198 | 898 / 200 / 198 |
| Seed | 42 | 42 |

**Architecture: unchanged.**

---

## 5. Mathematical Equivalence

Measured in the **real production code path**, BASE and A1 built with identical
weights and fed identical inputs. BASE was reconstructed by substituting the
original transpose + `BatchNorm3d` form back into the same builder.

| Check | Max abs difference | Tolerance | Result |
|---|---|---|---|
| Output (fp32) | **0.000e+00** | 1e-4 | PASS — bit-identical |
| Loss | **0.000e+00** (0.694010 both) | 1e-4 | PASS |
| Input gradient d/dx | 3.553e-15 | 1e-4 | PASS |
| Parameter gradients (36 tensors) | 7.839e-05 | 1e-4 | PASS |
| Parameter count | equal (33,089) | exact | PASS |
| Parameter names after mapping | identical | exact | PASS |

Prior float64 isolated-block verification: output 2.665e-15, d/dx 1.844e-14,
d/dweight 1.819e-12, d/dbias 9.095e-13.

The residual parameter-gradient difference (7.8e-05) is fp32 accumulation-order
noise from a different reduction shape, not a semantic difference — the forward
output is exactly identical and the fp64 check shows ~1e-12.

**Semantics preserved:** `track_running_stats=False` means `running_mean` and
`running_var` are `None` — no buffers exist, `momentum` is inert, and batch
statistics are used in train **and** eval. `F.batch_norm(x, None, None, w, b,
training=True, momentum=0.0, eps)` reproduces exactly that, including
BatchNorm's biased variance.

---

## 6. Checkpoint Migration

### Correction to the prior speed-audit report

The speed audit stated that A1 would break checkpoint loading
(`ncas.0.bn.bn.weight` → `ncas.0.bn.weight`). **That finding was an artefact of
the benchmark harness, not the real codebase.** The harness wrapped BatchNorm in
an extra module, nesting keys one level deeper than production.

Verified against the real production model:

| | Keys |
|---|---|
| Production (`BatchNorm3d`) | `ncas.0.bn.weight`, `ncas.0.bn.bias` |
| A1 (`ChannelsLastBatchNorm`) | `ncas.0.bn.weight`, `ncas.0.bn.bias` |

`ChannelsLastBatchNorm` registers `weight`/`bias` at its own top level, exactly
where `BatchNorm3d` did. There are no running-stat buffers to carry.
**The key sets are identical and NO migration is required.**

### Mapping (defensive net, normally inert)

`_load_from_state_dict` is implemented anyway and does two things only:

1. remaps a hypothetical nested `<prefix>bn.weight` → `<prefix>weight`;
2. drops `running_mean` / `running_var` / `num_batches_tracked` if a legacy
   tracking BatchNorm is ever encountered, and **reports the drop** through
   `error_msgs` rather than silently discarding it.

No `strict=False`. No key is silently swallowed. No checkpoint file on disk is
modified.

### Results

| Test | Result |
|---|---|
| Old checkpoint → A1 model, `strict=True` | **PASS** |
| New A1 checkpoint → A1 model, `strict=True` | **PASS** |
| Full state-dict key sets match | **PASS** |
| Every parameter exactly equal after load (36 tensors) | **PASS** — max diff 0.000e+00 |
| Nested `bn.bn.*` remapped correctly | PASS (defensive path) |
| Unrelated missing key still raises `RuntimeError` | **PASS** — not swallowed |

---

## 7. Resume Validation

Verified on the real-data smoke run:

| Component | Result |
|---|---|
| Model weights | PASS — identical after reload |
| Optimizer (AdamW) state | PASS |
| Scheduler (CosineAnnealingLR) state | PASS — `last_epoch` restored |
| EMA (decay 0.999) | PASS — all tensors equal |
| Epoch | PASS — 4 restored |
| Global step | PASS — 17 restored |
| Resumed model continues training | PASS — loss 0.274577, finite |

---

## 8. Real-Data Smoke — PASS

Run on genuine BraTS volumes from
`MICCAI-LH-BraTS2025-MET-Challenge-Training`, using cases drawn from the frozen
thesis split (`BraTS-MET-00002-000`, `-00003-000`, `-00005-000`, `-00006-000`).
Not synthetic data. Four cases only — a smoke test, not training.

| Stage | Result |
|---|---|
| NIfTI load + resample + z-normalise | PASS — 16.0 s for 4 cases |
| Image shape / dtype | PASS — (96,96,96,4), finite |
| Label shape, strictly binary | PASS — (96,96,96,3) |
| Forward (bf16 autocast, checkpointing ON) | PASS |
| Loss | PASS — finite every step |
| Backward + grad clip | PASS — grad norms 0.86 → 1.57 |
| Optimizer step | PASS |
| EMA update | PASS |
| Checkpoint save / load / resume | PASS |

Loss decreased monotonically across the four real cases:
**0.673700 → 0.593507 → 0.498881 → 0.421108**.

---

## 9. Performance

Paired and interleaved, identical GPU, precision, batch, input, seed, warm-up,
iteration count, checkpointing, geometry, NCA steps and global-context settings.

| round | BASE (ms) | A1 (ms) | ratio |
|---|---|---|---|
| 0 | 8238.1 | 5865.1 | 1.405× |
| 1 | 8217.8 | 5861.8 | 1.402× |
| 2 | 8216.2 | 5862.6 | 1.401× |
| 3 | 8220.0 | 5864.9 | 1.402× |

| | fwd | bwd | total | p95 | spread | VRAM |
|---|---|---|---|---|---|---|
| BASE | 1417.2 | 6800.6 | 8218.9 | 8220.9 | 0.05% | 1412 MB |
| **A1** | 1364.0 | **4498.6** | **5863.8** | 5871.4 | 0.16% | 1394 MB |

- **Median speedup: 1.402×** (min 1.401×, max 1.405×)
- Backward: **1.512×** · Forward: 1.039× · VRAM: **−18 MB**

The gain is concentrated in backward, as expected: the removed transposes
contributed two autograd nodes per NCA step.

This is materially the same as the ~1.40× the speed audit predicted, so that
figure stands — now confirmed in the real production code path rather than a
harness. The discarded 7,129 ms outlier was not used.

---

## 10. GPU Conditions

| | |
|---|---|
| GPU | NVIDIA RTX 3050 6GB Laptop, sm_86 |
| SM clock | **930 / 2100 MHz (44%)** |
| Temperature | 46 °C |
| Throttle flags | **SW Power Cap: Active · SW Thermal Slowdown: Active · HW Thermal Slowdown: Active** |
| Precision | bf16 |
| Batch | 1 |

**The GPU was power- and thermally capped throughout.** Absolute milliseconds
are inflated (~2.3×) and **no epoch estimate is derived from them**. The paired
per-round ratio is the valid quantity, and its 0.05–0.16% spread across four
interleaved rounds shows the comparison is sound despite the cap.

---

## 11. Scientific Impact

**No thesis architecture change was introduced.**

Parameter count, geometry, NCA step count, perception kernels, spatial
global-context kernel and formulation, SE, fusion, patchify state, checkpointing
state and granularity, batch size, split, seed, loss, optimizer, EMA,
augmentation policy and validation protocol are all unchanged.

Forward output is **bit-identical** in fp32, and the global-context sensitivity
test returns the same value for both implementations
(**0.454716 vs 0.454716, difference 0.000e+00**) with the central 64³ region held
identical and only the outer slabs perturbed — A1 does not remove or alter
whole-volume global context.

---

## 12. Remaining Candidates — ALL NOT PROMOTED

Every Category-C candidate from the speed audit remains experimental and
unimplemented. None has segmentation (Dice/HD95) validation.

| Candidate | Audit speedup | Status |
|---|---|---|
| NCA steps 15+15 | 1.88× | **NOT PROMOTED** |
| Spatial GC k=5 | 1.61× | **NOT PROMOTED** |
| L1 perception k=5 | 1.65× | **NOT PROMOTED** |
| Separable global context | 1.72× | **NOT PROMOTED** — not mathematically equivalent |
| Geometry 40³/56³ | 2.21× | **NOT PROMOTED** |
| Geometry 32³/48³ | 3.85× | **NOT PROMOTED** |
| COMBO-B … COMBO-F | up to 4.01× | **NOT PROMOTED** |

---

## 13. Separate Finding — Cache Documentation (NOT changed)

Reported here for completeness and **deliberately not acted on**, per the
instruction to keep it separate from A1 integration.

`configs/glo_nca_production.yaml:72-73` documents the preprocessing cache as
4.75 MB/case (~4.3 GB for 898 cases). The measured 96³ figure is
**16.03 MB/case = 14.06 GB**. The 4.75 MB value is exactly 64³ geometry.

This is independently corroborated by the repository's own test suite:
`scripts/test_preprocess_cache.py` reports `"volume": "64^3", "disk_mb_per_case": 4.75`.

**No config value was changed.** This requires a separate decision.

---

## 14. Full Gate Summary (§17)

| Gate | Result |
|---|---|
| Import tests | PASS |
| `compileall` | PASS |
| Production identity gate | PASS |
| Parameter-count gate (33,089) | PASS |
| Architecture gate | PASS |
| Global-context gate | PASS — 0.454716 both |
| Gradient gate (36/36 params, no NaN/Inf, ×2 iters) | PASS |
| Checkpoint save / load | PASS |
| Old-checkpoint load (strict) | PASS |
| New-checkpoint load (strict) | PASS |
| Optimizer / scheduler / EMA restore | PASS |
| Full resume | PASS |
| Real-data smoke | PASS |
| Deterministic equivalence | PASS — output 0.000e+00 |
| Performance benchmark | PASS — 1.402× |
| Existing suite: `test_preprocess_cache.py` | PASS 7 / FAIL 0 |
| Existing suite: `test_preflight_logic.py` | 6/6 PASS |

---

## 15. Safety

- No `git commit`, no `git push` — HEAD still `7476e18`
- No GCP, no VM, no cloud execution, no cloud cost
- Gradient checkpointing never disabled to make A1 pass
- Thesis split untouched — SHA `d30d719…`, 898/200/198
- No reports deleted, no historical evidence modified
- The speed audit was **not** rewritten to imply A1 was already applied
- All verification scripts isolated in the session scratch directory
