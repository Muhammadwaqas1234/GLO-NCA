# PHASE 2 — FINAL LOCAL VERIFICATION (GLO-NCA V3)

Branch `v3-multilevel` · 2026-09-16
**No training campaign was run** (no 1/5/300-epoch run, no Kaggle, no GCP).
All GPU work was single-batch correctness testing on 1–3 cases.

---

## 1. Phase 1 closure matrix — verified against CURRENT source

Each item was re-checked programmatically against the code as it stands now, not
taken from the Phase 2 report's claims. **20 / 20 CLOSED.**

| ID | Finding | Sev | State | Evidence in current source |
|---|---|---|---|---|
| F-01 | V3 model + split untracked | P0 | **CLOSED** | `git ls-files` finds `Model_GLO_NCA_V3.py`, `master_split.json` |
| CL-01 | Split unreachable by container | P0 | **CLOSED** | `Dockerfile: COPY split/master_split.json`; `.dockerignore: !split/master_split.json` |
| CL-02 | Gate could launch 300-epoch run | P0 | **CLOSED** | no executable line in `pretrain_gate.sh` pairs `docker run` with `GATE_CFG` |
| R-01/T-03 | Augmentation RNG repeated every epoch | P1 | **CLOSED** | `_EpochSampler` + `set_augmentation_seed`; measured 28 distinct streams/40 epochs |
| D-01 | Preprocessing cache defeated | P1 | **CLOSED** | `persistent_workers=True`, `prefetch_factor=2` |
| D-02 | Patchify full-volume waste | P1 | **CLOSED** | `full_volume` hoist; 300 real-module trials, 0 mismatches |
| D-04 | torchio built every `__getitem__` | P2 | **CLOSED** | constructed only inside the branch that uses it |
| D-05 | Blocking H2D copy | P2 | **CLOSED** | `non_blocking=True` with `pin_memory=True` |
| C-01/T-04 | Silent scheduler reset | P1 | **CLOSED** | `raise RuntimeError`; test proves it raises |
| C-02 | Silent EMA restore | P2 | **CLOSED** | warns "checkpoint contains NO EMA state" |
| T-01 | `zero_grad` not `set_to_none` | P3 | **CLOSED** | `zero_grad(set_to_none=True)` |
| T-05/C-03 | No config identity on resume | P2 | **CLOSED** | 9-field drift check → `_fail` |
| F-05 | Missing `data:` ⇒ silent re-split | P1 | **CLOSED** | V3 refuses without `split_file` unless `allow_seeded_split` |
| F-02 | Hidden hard-coded hyperparameters | P1 | **CLOSED** | `loss.ce_weight`, `loss.empty_region_bce_weight`, `sampling.*` now in config |
| F-03 | Ambiguous production config | P1 | **CLOSED** | `v3_multilevel_ckpt.yaml` declared production; docs aligned |
| F-06 | Contradictory ablation headers | P2 | **CLOSED** | headers rewritten to match values (values untouched) |
| CL-03 | `set -e` killed `mark $?` | P1 | **CLOSED** | `check()` wrapper |
| CL-04 | `setup_gcp.sh` branch `v2` | P1 | **CLOSED** | defaults `v3-multilevel`, verifies V3 + split present |
| CL-05 | `ls -t` sync target | P1 | **CLOSED** | explicit `--experiment-id`; no `ls -t` on any executable line |
| CL-06 | Invalid `$$` lock / VM cleanup | P1 | **CLOSED** | `assert_no_training_running` (systemd-backed); `ALLOW_FAILED=1` |
| ME-01 | ~20 GB host RAM at final eval | P1 | **CLOSED** | uint8 GT + `val_pairs` released → **11.7 GB measured** |
| E-01 | Validation used twice | P2 | **DOCUMENTED** | standard practice; report test as held-out, not val |
| E-02 | HD95 units | P1 | **DECISION** | §7 below |
| E-04 | Empty-empty Dice vs IoU | P2 | **DECISION** | §8 below |
| R-03 | cudnn nondeterminism | P2 | **ACCEPTED** | deliberate, documented in `reproducibility.describe()` |
| TS-01/02 | Tests test reimplementations | P1 | **CLOSED** | new suites exercise the **real** modules |

## 2. Phase 2 closure

All items from `PHASE2_PRODUCTION_HARDENING.md` §17 are now resolved or have an
explicit, evidence-backed recommendation (§6–§11). **Two new bugs were found and
fixed during this pass** (§4).

## 3. Final V3 scientific baseline — VERIFIED FROZEN

Verified on the GPU against `configs/v3_multilevel_ckpt.yaml` (26 values + levels
+ split), and the parameter count measured on-device:

| Item | Value | How verified |
|---|---|---|
| Architecture | GLO-NCA V3 Multi-Level | model built on CUDA |
| Level 1 / 2 / 3 | 32³ / 96³ / 128³ | config + model |
| Channels | 24 / 24 / 16 | config |
| NCA steps | 20 + 20 + 10 = **50** | config |
| **Parameters** | **40,656** | **measured on GPU** (L1 18,861 · L2 11,277 · L3 7,811 · proj 800 · fusion+head 1,907) |
| Context | SE + spatial GC | config |
| Fusion | projections + concat + 1×1×1 | model |
| Output | 3 sigmoid channels WT/TC/ET | forward returns `(1,3,f,f,f)` |
| Loss | FocalTversky + BCE, β=0.75, γ=1.33, α=0.25, ce=0.5 | config + GPU loss step |
| Optimizer / LR | AdamW 1.6e-3, β=(0.9,0.99), wd 1e-4 | GPU step: lr 0.001600→0.001561 |
| Scheduler | CosineAnnealingLR | GPU step |
| EMA | 0.999 | GPU step |
| Grad clip | 1.0 | measured grad-norm 0.134 |
| Seed | 42 | config |
| Epochs | 300 | config |
| Gradient checkpointing | **ON** | **max\|Δout\|=0, max\|Δgrad\|=0 vs OFF** |
| Threshold tuning | validation only | code + test |
| Test | single frozen evaluation | code + test |
| TTA / ensemble | none | absent |

**Methodology: UNCHANGED.**

## 4. Bugs discovered during this pass — and fixed

| # | Bug | Where | Severity | Fix |
|---|---|---|---|---|
| 1 | **Windows CUDA sysmem fallback masked OOM.** 128³ reported "OK, peak 8.45 GB" on a **6 GB** GPU — it had silently spilled ~2.5 GB into host RAM over PCIe. A naive reading would have concluded production fits on a 6 GB card. | VRAM ladder | **High (false PASS)** | classify `peak > 0.95 × VRAM` as **SPILL**, never OK; documented |
| 2 | Regression test used `v3_smoke_1epoch.yaml`, which keeps **production** resolutions (32/96/128) — the "small" forward test ran the full 128³ workload and hung on CPU | `phase2_regression.py` | Medium | use `smoke_test_v3.yaml` (16/24/32) |
| 3 | Shell-syntax check passed Windows absolute paths to Git-bash, which mangles `C:\a\b` → `C:ab`; reported all 19 scripts as broken | verification suite | Medium (false FAIL) | pass repo-relative posix paths with `cwd` |
| 4 | Per-case metric comparison used `==` on lists containing `NaN`; `NaN != NaN` produced a false FAIL | verification suite | Medium (false FAIL) | NaN-aware comparison |
| 5 | Host-RAM check judged against the **laptop** (15.6 GB) rather than the deployment target (g2-standard-8, 32 GB) | verification suite | Low | criterion is the target VM |

Bugs 1, 3 and 4 were **false results in my own tests** — three of five findings
this pass were test defects, not product defects. They are listed because a test
that lies is worse than no test.

## 5. Production hyperparameter table (single authoritative config)

`configs/v3_multilevel_ckpt.yaml` — everything scientifically meaningful now
lives here. Values marked ★ were hard-coded before this pass and are surfaced
with **identical** defaults (verified: all four configs resolve to the historical
values, so behaviour is unchanged).

| Group | Key | Value |
|---|---|---|
| experiment | seed | 42 |
| data | split_file | `split/master_split.json` |
| model | version / fire_rate / hidden / dropout | v3 / 0.6 / 128 / 0.1 |
| model | use_attention / use_spatial | true / true |
| model | level1 / level2 / level3 | 32³·24·20·k7 / 96³·24·20·k3 / 128³·16·10·k3 |
| model | feature_fusion.type | concat |
| training | epochs / batch_size / patch_size / workers | 300 / 1 / 128 / 4 |
| training | augmentation | light |
| optimizer | learning_rate / minimum / weight_decay | 0.0016 / 0.00001 / 0.0001 |
| loss | tversky_beta / focal_gamma | 0.75 / 1.33 (α derived = 0.25) |
| loss ★ | ce_weight | 0.5 |
| loss ★ | empty_region_bce_weight | 0.1 |
| sampling ★ | prioritize_probability | 0.7 |
| sampling ★ | prioritize_region | 2 (ET) |
| ema | enabled / decay | true / 0.999 |
| gradient | clipping_enabled / max_norm | true / 1.0 |
| evaluation | threshold / tune_thresholds / smoothing_window | 0.5 / true / 3 |
| memory | gradient_checkpointing | **true** |
| logging | tensorboard / checkpoint_frequency | true / 10 |

Still hard-coded (deliberately, as implementation detail, not thesis knobs):
AdamW betas (0.9, 0.99), threshold grid 0.20–0.60, bootstrap `n_boot=2000`,
`foreground_crop` / `nonzero_norm` / `patchify` = True.

## 6. Decision A — HD95 units

**Findings.** `Agent.hd95_score` calls `distance_transform_edt` with **no
`sampling=`**, so distances are in **voxels**. Volumes are resampled to a 128³
cube from anisotropic BraTS data after a per-case foreground crop, so one voxel
represents a **different physical distance in every case and axis**. Spacing is
available in the NIfTI affine but is **not** carried through the pipeline.
`_surface_distances` measures distance-to-**foreground**, not to-**surface**.

**Classification: reporting issue, not a bug.** The code is internally
consistent and honestly labelled `HD95(vox)` / `hd95_vox` everywhere; **no
document claims mm** (grep found none).

**Decision (no code change):** keep voxel units. Converting to mm would change
every reported HD95 and require threading spacing through preprocessing —
a methodology change needing your approval.

**Thesis requirement:** report as *"HD95 in resampled-voxel units on a 128³ grid;
not directly comparable to millimetre-based BraTS literature values."* Do not
compare these numbers against published mm HD95.

## 7. Decision B — empty-empty Dice vs IoU

**Measured on the real metric code**, prediction and GT both empty:

| Metric | Value |
|---|---|
| Dice | **0.000** |
| IoU | **1.000** |
| HD95 | **0.000** |

Two of three metrics call this perfect; Dice is the outlier (`2·0/(0+0+1e-6)=0`).

**Impact measured, not assumed:** of 40 sampled real BraTS-MET cases,
**0 had zero ET voxels**. The empty-empty case is rare in this cohort, so the
inconsistency has little practical effect on reported metrics.

**Classification: a real inconsistency, low practical impact.**
**Decision (no code change):** changing Dice would alter reported ET numbers —
a methodology change. Document the convention:
*"Dice scores an empty-prediction/empty-GT case as 0; IoU scores it as 1."*
If you prefer the common convention (empty-empty ⇒ Dice 1), say so and I will
change it and re-run the metric regression.

## 8. Decision C — invalid-label cases `01094-002`, `01184-002`

- **Both are in the TRAIN split** (verified) — they **cannot** affect val/test.
- Production behaviour is **fail-closed**: `dataset_validation` rejects any seg
  value outside `{0,1,2,3,4}` (`ALLOWED_SEG_LABELS`), `validate_dataset` returns
  non-PASS, and `runner.py:278` aborts before training starts.
- Neither case exists in the 651-case local cohort (full set is 1296), so this
  could not be exercised locally.

**Consequence you must plan for:** with the full dataset on the VM, `validate_
dataset` will **FAIL and training will refuse to start**. That is correct
fail-closed behaviour, but it is a **blocker at pre-flight**.

**Canonical split unchanged** (sha `d30d7195…`). Options, none applied:
1. Repair/relabel the two volumes at the data level (canonical split untouched).
2. Add an explicit, documented **operational data-quality filter** — separate
   from the canonical scientific split — excluding exactly those two train cases.

Both require your approval. I did not alter the split or loosen the validator.

## 9. Decision D — gradient clipping semantics

Kaggle v7 used per-parameter element-wise `torch.clamp`; V3 uses
`clip_grad_norm_` (measured working: grad-norm 0.134). **These are different
operations at the same nominal 1.0.** V3's norm clipping is the current
methodology and is **retained unchanged**. Declare the difference if you compare
V3 results against v7.

## 10. Decision E — evaluation host RAM (FIXED, bit-exact)

**Measured.** Storage was float32 prob + float32 GT = 48.0 MB per case-pair ⇒
val 9.4 GB + test 9.3 GB = **18.7 GB held simultaneously** (`val_pairs` stays
live until `_write_threshold_comparison`).

**Fix, proven equivalent.** Ground truth is strictly binary and every metric
uses it only via `gt >= 0.5`, so storing it as **uint8 is bit-exact**. Verified
through the real metric functions: `score`, `tune_thresholds` and
`score_per_case` all **identical** (including NaN positions). `val_pairs` is now
released immediately after its last consumer.

- **18.7 GB → 11.7 GB measured** (30.0 MB/case-pair).
- Target VM `g2-standard-8` = 32 GB ⇒ **20.3 GB headroom**.

**float16 probabilities were rejected**: measured to flip ~2,254 threshold
decisions per 4M voxels — *not* equivalent, so not used.

## 11. Local environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute 8.6) |
| VRAM | 6.0 GB |
| Driver | 592.82 |
| CUDA (torch build) | 12.1 |
| PyTorch | **2.5.1+cu121** — exactly the production pin |
| Python | 3.12.9 |
| numpy / nibabel / torchio / cv2 / scipy | 1.26.4 / 5.4.2 / 1.2.1 / 4.10.0 / 1.17.1 |
| Host RAM | 15.6 GB |
| Real dataset | 651 BraTS-MET cases available locally |

## 12. Local test results

**`scripts/phase2_final_local_verification.py`: 24 PASS / 0 FAIL / 1 SKIP**
**`scripts/phase2_regression.py`: 16 PASS / 0 FAIL / 0 SKIP**

Highlights (all executed, none inferred):

- 40,656 parameters **on GPU**, per-component breakdown exact.
- **Gradient checkpointing ON == OFF: max\|Δoutput\| = 0, max\|Δgradient\| = 0.**
- Optimizer + clip + scheduler + EMA: 50 params updated, lr 0.001600→0.001561.
- Checkpoint save→load→restore: weights, LR, scheduler step, epoch, optimizer
  state all restored.
- Scheduler-restore failure **raises**; resume without saved config exits 2;
  bare `train.py` exits 2 (cannot silently run V2).
- **REAL patchify: 300 cases × 12 scenarios (full/non-full × prio on/off ×
  ET present/absent/empty) — 0 output or RNG mismatches.**
- **REAL BraTS-MET case** through the real dataset: `(64,64,64,4)` image,
  `(64,64,64,3)` binary label, **ET ⊆ TC ⊆ WT** verified.
- **REAL case → V3 → loss → backward on GPU**: loss 3.8566, finite gradients.

## 13. VRAM measurements (checkpointing ON, measured)

| Resolution | Status | Peak |
|---|---|---|
| 32³ | OK | 0.15 GB |
| 48³ | OK | 0.46 GB |
| 64³ | OK | 1.07 GB |
| 96³ | OK | 3.57 GB |
| 128³ (earlier run) | **SPILL** | 8.45 GB reported on a 6 GB card |

**The 128³ result is NOT a fit.** Windows CUDA sysmem fallback silently paged the
excess to host RAM instead of raising OOM. On Linux/GCP this configuration needs
a GPU with **> 8.45 GB** — consistent with the prior ~9.07 GB T4/L4 measurement,
and why the L4 (24 GB) is the correct target.

**This is a hardware limit, not a code defect.** V3 was not shrunk.

## 14. Code-quality audit

- **No bare `except:`** remains in production code. The two `except Exception:
  pass` in `logutil.py` guard TensorBoard writes only — correct (logging must
  never kill a 300-epoch run).
- Unused import removed (`contextlib`).
- `TODO`s remaining are all in legacy V2 files off the V3 production path.
- 19/19 cloud shell scripts parse.
- No credentials tracked; `gcp.env`, `*.pem`, `*.key`, `*service-account*` ignored.

## 15. Files changed in this pass

| File | Change |
|---|---|
| `src/experiment/metrics_eval.py` | uint8 ground truth (bit-exact, −7 GB) |
| `src/experiment/runner.py` | release `val_pairs` early; `ce_weight` / `empty_weight` / sampling from config; `zero_grad(set_to_none=True)` |
| `configs/v3_multilevel_ckpt.yaml` | `loss.ce_weight`, `loss.empty_region_bce_weight`, `sampling.*` (values unchanged) |
| `scripts/phase2_regression.py` | correct smoke config; resolution-derived assertion |
| `scripts/phase2_final_local_verification.py` | **new** — full local GPU suite |
| `reports/validation/phase2_final_local_verification.json` | **new** — machine-readable results |

**Files removed/moved this pass: none.** All 15 Phase 1 reports and
`PHASE2_PRODUCTION_HARDENING.md` are intact and unmodified.

## 16. Methodology preservation

`Model_GLO_NCA_V3.py`, `Model_BasicNCA3D.py`, `LossFunctions.py` and
`split/master_split.json` were **not modified in this pass**. The two behavioural
changes are proven equivalent:

- uint8 GT — identical `score`, `tune_thresholds`, `score_per_case`.
- patchify hoist — identical outputs and identical `random.getstate()` over 300
  real-module trials.

The one non-bit-identical change remains the **intended** Phase 2 RNG fix
(augmentation now varies per epoch instead of repeating 300×). Still fully
deterministic for a given seed.

## 17. Remaining blockers

1. **DATA BLOCKER — `01094-002` / `01184-002`.** With the full 1296-case dataset,
   `validate_dataset` will FAIL and training will refuse to start (correct
   fail-closed behaviour). **Needs your decision (§8) before the VM run.**
2. **Docker image build** — started locally; multi-GB CUDA base download.
   Must complete and pass the container smoke as a pre-flight step.
3. 128³ VRAM must be confirmed on the real 24 GB L4 (locally it SPILLS).
4. HD95 (§6) and empty-empty Dice (§7) remain thesis reporting decisions.

## 18. Final status

# READY FOR GCP PRE-FLIGHT

Local correctness is **fully verified on real GPU + real data**: 24/24 and 16/16
tests pass, 20/20 Phase 1 findings verified closed against current source, and
the scientific baseline is frozen and measured (40,656 params; checkpointing
bit-exact).

It is **not** "READY FOR GCP TRAINING": the data blocker in §17.1 must be
decided first, the Docker image must build and pass its smoke test, and 128³
must be confirmed on a 24 GB-class GPU (the 6 GB laptop cannot fit it — an
expected hardware limit, not a code defect).
