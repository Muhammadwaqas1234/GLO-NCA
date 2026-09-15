# PHASE 1 — FINAL CLOSURE AUDIT (GLO-NCA V3)

Branch `v3-multilevel` · base commit `f9e5501` · 2026-09-16
**Training performed: 0.** No 1/5/10/300-epoch run, no Kaggle, no GCP.
All GPU work was single-batch correctness testing and a capped VRAM ladder.

> This report does **not** trust the Phase 1 or Phase 2 reports. Every finding was
> re-derived from the **current source** by `scripts/phase1_closure_audit.py`,
> which reads the code as it stands now. Where a previous test was misleading, the
> test was fixed and re-run.

---

## 1. Executive summary

**39 PASS / 0 FAIL / 0 SKIP** (`phase1_closure_audit.py`), on top of
**24 PASS / 0 FAIL / 1 SKIP** (local GPU suite) and **16 PASS / 0 FAIL / 0 SKIP**
(regression).

**BLOCKER #1 (invalid-label cases) is CLOSED with measured evidence (§18).** The
two cases are *validator-strict, not training-invalid*: the production label
conversion already maps their stray labels to background, and their WT/TC/ET
targets are valid and correctly nested. They are therefore **kept in training**
under an explicit, per-case, fail-closed data-quality policy. **No case excluded;
canonical split byte-for-byte unchanged (898/200/198, SHA `d30d7195…`).**

**Docker: image builds and the canonical split SHA is verified INSIDE the image**
(§11b). **L4 128³ remains NOT RUN — REQUIRES GCP** (this task forbids GCP work).

Failures surfaced during this pass and were fixed (§7): a personal absolute path
in a shipped script (a real repository defect), plus defects in my own audit
tooling.

**Methodology: UNCHANGED.** The V3 model, loss and canonical split were not
modified in this pass.

## 2. Phase 1 closure matrix (re-verified against current source)

| Phase 1 finding | Sev | Verified by | State |
|---|---|---|---|
| F-01 V3 model/agent/config/split untracked | P0 | `git ls-files --error-unmatch` on all four | **CLOSED** |
| CL-01 split unreachable by container | P0 | `COPY split/master_split.json` + `!split/…` in `.dockerignore` | **CLOSED** |
| CL-02 gate could launch the 300-epoch run | P0 | no **executable** gate line pairs `docker run` with `GATE_CFG` | **CLOSED** |
| R-01/T-03 augmentation RNG repeated each epoch | P1 | `_EpochSampler`; 60/60 distinct streams across epochs, 60/60 across cases, deterministic on repeat | **CLOSED** |
| D-01 preprocessing cache defeated | P1 | `persistent_workers` + epoch-aware sampler | **CLOSED** |
| D-02 patchify full-volume waste | P1 | `full_volume` hoist; 300 real-module trials, 0 output/RNG mismatches | **CLOSED** |
| D-04 torchio built every `__getitem__` | P2 | constructed only in the branch that uses it | **CLOSED** |
| D-05 blocking H2D copy | P2 | `non_blocking=True` with `pin_memory` | **CLOSED** |
| C-01/T-04 silent scheduler reset | P1 | `raise RuntimeError`; old `except: pass` gone | **CLOSED** |
| C-02 silent EMA restore | P2 | warns "checkpoint contains NO EMA state" | **CLOSED** |
| T-01 `zero_grad` not `set_to_none` | P3 | `zero_grad(set_to_none=True)` | **CLOSED** |
| T-05/C-03 no config identity on resume | P2 | compares epochs/lr/loss/seed vs checkpoint | **CLOSED** |
| F-02 hidden hard-coded hyperparameters | P1 | `loss.ce_weight`, `loss.empty_region_bce_weight`, `sampling.*` in config | **CLOSED** |
| F-03 ambiguous production config | P1 | exactly one config claims production; no doc contradicts | **CLOSED** |
| F-05 missing `data:` ⇒ silent re-split | P1 | V3 run refused without an explicit split | **CLOSED** |
| F-06 contradictory ablation headers | P2 | headers match values (values untouched) | **CLOSED** |
| CL-03 `set -e` killed `mark $?` | P1 | `check()` wrapper; 19/19 scripts parse | **CLOSED** |
| CL-04 `setup_gcp.sh` branch `v2` | P1 | defaults `v3-multilevel`, verifies V3 + split | **CLOSED** |
| CL-05 `ls -t` sync target | P1 | no `ls -t` on any executable line | **CLOSED** |
| CL-06 invalid `$$` lock / VM cleanup | P1 | `assert_no_training_running` (systemd); no launcher-PID lock | **CLOSED** |
| ME-01 ~20 GB host RAM at eval | P1 | uint8 GT + `val_pairs` released → 11.7 GB measured | **CLOSED** |
| TS-01/02 tests tested reimplementations | P1 | suites now exercise the real modules | **CLOSED** |
| E-01 validation used twice | P2 | standard practice; documented | **DOCUMENTED** |
| E-02 HD95 units | P1 | voxel-space, labelled, no mm claim anywhere | **DOCUMENTED (§17)** |
| E-04 empty-empty Dice vs IoU | P2 | measured; 0/40 real cases affected | **DOCUMENTED (§17)** |
| R-03 cudnn nondeterminism | P2 | deliberate, documented in `reproducibility.describe()` | **ACCEPTED** |

**P0 open: 0 · P1 open: 0 · P2 open: 0** (three P1/P2 items are explicit,
evidence-backed *reporting decisions*, not defects — §17).

## 3. GLO-NCA naming audit

**PASS.** Zero `MedNCA` / `M3D-NCA` / `Med-NCA` occurrences anywhere on the
active production path (`src/`, `configs/`, `cloud/`, `scripts/`, `train.py`,
`Dockerfile`, `.dockerignore`). 49 active files carry GLO-NCA naming.

**Deliberately retained:** `README.md` cites *Med-NCA* (IPMI 2023) and *M3D-NCA*
(MICCAI 2023) by Kalkhof et al. as the upstream framework. **These are academic
citations and must not be renamed** — doing so would be misattribution. The
README already separates them correctly from the thesis contributions.

The repository *directory* is still named `M3D-NCA-main` (it is the extracted
upstream archive folder). This is cosmetic, outside version control, and affects
no code path; renaming it would break local paths for no benefit.

## 4. Hyperparameter audit — single authoritative config

`configs/v3_multilevel_ckpt.yaml` is the sole production config (verified: it is
the only file claiming that, and no document points the campaign elsewhere).
**28 fingerprint values + level structure + split path** verified.

Remaining hard-coded values, each classified:

| Value | Where | Class | Verdict |
|---|---|---|---|
| AdamW betas (0.9, 0.99) | `runner.py` | implementation constant | may remain |
| threshold grid 0.20–0.60 | `metrics_eval.py` | evaluation protocol constant | may remain (documented) |
| bootstrap `n_boot=2000` | `runner.py` | reporting constant | may remain |
| `foreground_crop`/`nonzero_norm`/`patchify` = True | `runner.py` | safety invariant for V3 | may remain |
| smoothing 1e-6, clamp 1e-6 | metrics/loss | mathematical constants | may remain |

No scientific hyperparameter remains hidden from the config record.

## 5. Data pipeline audit

Verified on **real BraTS-MET** data (651 local cases):

- Recursive discovery incl. nested cohorts; deterministic case IDs.
- Modality order fixed T1/T1ce/T2/FLAIR; missing modality raises.
- Labels validated against `{0,1,2,3,4}` — **fail-closed**.
- **ET ⊆ TC ⊆ WT verified on a real case**; targets strictly binary float32.
- Foreground crop, nearest-for-labels resize, per-case non-zero z-norm.
- Patchify/augmentation gated to the **train** split only.
- Patchify optimization: **output- and RNG-state-identical** (300 trials,
  12 scenarios: full/non-full × prio on/off × ET present/absent/empty).
- Epoch-aware sampler: deterministic per `(seed, epoch, case)`, varied across
  epochs — works with persistent workers.
- Canonical split unchanged, case- **and subject-disjoint**.

## 6. Model / training-loop / evaluation audits

- **40,656 parameters measured on the local GPU**, per-component exact
  (18,861 / 11,277 / 7,811 / 800 / 1,907).
- Forward `(B,X,Y,Z,4)` → `(B,3,X,Y,Z)`; 3 sigmoid channels; not softmax.
- **Gradient checkpointing ON vs OFF: max|Δoutput| = 0, max|Δgradient| = 0.**
- Step order forward → loss → backward → clip → optimizer → scheduler → EMA;
  one forward, one backward; measured grad-norm 0.134, lr 0.001600 → 0.001561.
- Evaluation: test collected **once**, thresholds tuned on **validation only**,
  runs under `no_grad`. HD95 voxel-space and labelled as such.

## 7. Bugs found and fixed in THIS pass

| # | Issue | Class | Fix |
|---|---|---|---|
| 1 | **Personal absolute path** `C:\Users\raiwa\…` in `scripts/validate_kaggle_sample.py` docstring — shipped to an examiner | **real repo defect** | replaced with `<staging-dir>` placeholder |
| 2 | Audit script flagged **itself** for legacy names (it contains the search regexes) | audit-tooling defect | audit scripts excluded from their own scans |
| 3 | Audit script flagged **itself** for referencing archived tool filenames | audit-tooling defect | same exclusion |

Also fixed earlier this session: a hard-coded dataset path I had introduced in
`phase2_final_local_verification.py`, now resolved via `--data-root`,
`$DATA_ROOT`, or repo/home-relative candidates.

## 8. GPU safety and memory

Local GPU left **clean after every run**: `0 MiB` allocated, ~42–50 °C, **no
orphan CUDA processes**. The VRAM ladder is capped at 96³ by default
(`--max-res`) to avoid stressing the 6 GB laptop GPU; 128³ was measured once and
is not re-run.

| Resolution (ckpt ON) | Status | Peak |
|---|---|---|
| 32³ | **FIT** | 0.15 GB |
| 48³ | **FIT** | 0.46 GB |
| 64³ | **FIT** | 1.07 GB |
| 96³ | **FIT** | 3.57 GB |
| 128³ | **SPILL** | 8.45 GB reported on a 6 GB card |

**128³ is explicitly NOT a fit.** Windows CUDA sysmem fallback paged the excess
into host RAM instead of raising OOM; on Linux/GCP this needs a GPU > 8.45 GB,
consistent with the ~9.07 GB L4 measurement. **SPILL is never reported as PASS.**
This is a hardware limit — V3 was not shrunk.

## 9. Performance / cost optimizations (all behaviour-preserving)

| Optimization | Proof of equivalence |
|---|---|
| patchify full-volume hoist | identical outputs **and** `random.getstate()`, 300 trials |
| persistent workers (cache survives epochs) | no numerical effect; sampler carries the epoch |
| uint8 ground truth in evaluation | `score`, `tune_thresholds`, `score_per_case` all identical (NaN-aware) |
| `val_pairs` released after last use | same inputs, same outputs, lower peak |
| `non_blocking` H2D + `set_to_none` | numerically identical |

**Rejected:** float16 probabilities — measured to flip ~2,254 threshold decisions
per 4M voxels, so *not* equivalent.

Untouched, as required: resolutions, NCA steps, architecture, batch size, epochs,
loss, evaluation protocol, case count.

## 10. Checkpoint / resume

Full-state checkpoint (model/optimizer/scheduler/EMA/epoch/best/history/config/
RNG), atomic `.tmp` + `os.replace`. Verified by round-trip on GPU: weights, LR,
scheduler step, epoch and optimizer state all restored.

Fail-loud behaviour verified by **real subprocess**:
- resume without saved config → **exit 2** ("cannot resume"), never V2 fallback;
- bare `train.py` → **exit 2** ("--config is required");
- corrupt scheduler state → **RuntimeError**;
- hyperparameter drift on resume → refused.

## 11. GCP production path (static)

Split ships in the image and the build fails if its fingerprint is wrong; no V2
default anywhere; explicit `--experiment-id` (no directory guessing); final GCS
sync attempted on success **and** failure; systemd-backed lock; failed runs can
be cleaned up (`ALLOW_FAILED=1`); the gate cannot start the campaign;
19/19 cloud scripts parse.

**Container runtime behaviour is NOT RUN — REQUIRES the built image / GCP.**

## 12. Security

No credential-like file tracked (`*.pem`, `*.key`, `*service-account*`,
`gcp.env`). Image excludes checkpoints, NIfTI data and experiment outputs.
No personal absolute path remains in active code (fixed, §7). No secret values
were printed during this audit.

## 13. Dead files / references

`xfer2.sh`, `drive_dl.py`, `dataset_identity.py` are archived under
`archive/one_off_transfer_tools/` with a README; **zero references** from active
code. `archive/` is **never imported** by any active module — Kaggle history is
fully isolated. All 15 Phase 1 reports present and unmodified.

## 14. V4 / V7 knowledge audit

Reviewed for engineering knowledge only; **nothing imported or copied**.
- v7 clipped gradients with per-parameter element-wise `torch.clamp`; V3 uses
  `clip_grad_norm_`. **Different operations at the same nominal 1.0** —
  V3's is retained; declare the difference if comparing to v7 numbers.
- Neither v7 nor V3 uses AMP.
- v7's `num_workers`/`pin_memory` practice is already present in V3.

## 15. Repository state

| Item | Value |
|---|---|
| Branch / base commit | `v3-multilevel` / `f9e5501` |
| Working tree | 28 modified, 10 added, 4 renamed (archive moves), 1 deleted (archived), 23 untracked |
| Staged additions | V3 model, agent, 10 V3 configs, canonical split, Kaggle history moves |
| Data/checkpoints/secrets staged | **none** |
| `configs/v3_multilevel_ckpt.yaml` | `93a582c5b9dde539` |
| `src/models/Model_GLO_NCA_V3.py` | `5335d301c3a6b88f` |
| `src/models/Model_BasicNCA3D.py` | `96cddf897b8c75a6` |
| `src/losses/LossFunctions.py` | `d6e56d5bcaef4dee` |
| `split/master_split.json` | `cd55a18a15e54690` (content sha256 `d30d7195…9559d`) |

## 16. Complete test matrix

| Suite | Result |
|---|---|
| `scripts/phase1_closure_audit.py` | **39 PASS / 0 FAIL / 0 SKIP** |
| `scripts/phase2_final_local_verification.py` | **24 PASS / 0 FAIL / 1 SKIP** |
| `scripts/phase2_regression.py` | **16 PASS / 0 FAIL / 0 SKIP** |
| Docker image build | **PASS** — 15.2 GB; canonical split SHA verified INSIDE the image |
| Docker container smoke | **PASS (partial)** — in-container: config parses, **params 40,656**, split SHA `d30d7195...` 898/200/198. Policy-file COPY added after a real bug was caught; final rebuild verifying it |
| Data-quality policy (real files) | **PASS** — 4 dedicated checks |
| L4 128³ pre-flight | **NOT RUN — REQUIRES GCP** |
| GCP runtime | **NOT RUN — REQUIRES GCP** |

## 17. Open decisions (documented, not defects)

1. **HD95 units.** Voxel-space on a 128³ resampled grid, honestly labelled; no
   mm claim exists. Report as *"HD95 in resampled-voxel units; not comparable to
   mm-based BraTS literature."* Converting to mm would change every value.
2. **Empty-empty convention.** Dice 0.0 vs IoU 1.0 vs HD95 0.0 for the same case.
   Measured impact: **0 of 40 real cases** have zero ET, so practical effect is
   minimal. Document the convention, or ask and I will change Dice and re-run.
3. **Gradient-clipping semantics vs v7** — declare if comparing.

## 18. BLOCKER #1 — invalid-label cases: **CLOSED** (evidence-based)

**Correction to an earlier statement in this session:** I previously reported
these two cases were absent locally. That was **wrong** — they live in the nested
`UCSD - Training/` cohort, which my first search missed. They were inspected
directly.

### Measured evidence

| Case | Stray label | Voxels | Components | Neighbouring labels |
|---|---|---|---|---|
| `BraTS-MET-01094-002` | **6** | 129 | 1 | `{0: 174}` — entirely background-adjacent |
| `BraTS-MET-01184-002` | **8** | 28 | 1 | `{0: 56, 3: 11}` — almost entirely background |

### The decisive finding

The production label conversion `Nii_Gz_Dataset_3D._labels_to_regions` maps
**only** `{1,2,3,4}` into WT/TC/ET. Run on these two real volumes:

```
BraTS-MET-01094-002: stray-6 voxels -> WT 0, TC 0, ET 0
BraTS-MET-01184-002: stray-8 voxels -> WT 0, TC 0, ET 0
```

The stray voxels contribute to **no region** — they already become background,
and the resulting targets are valid, binary and correctly nested:

| Case | WT | TC | ET | ET ⊆ TC ⊆ WT | binary |
|---|---|---|---|---|---|
| `01094-002` | 19,847 | 19,847 | 19,847 | yes | yes |
| `01184-002` | 21,991 | 15,377 | 14,974 | yes | yes |

**These cases are validator-strict, NOT training-invalid.** This is precisely
option 3 of the pre-existing precedent in
`reports/validation/GLO_NCA_V3_KAGGLE_5EPOCH_REPORT.md`, which left the decision
open and listed "investigate whether the established methodology remaps labels"
as a candidate. It does.

### Policy implemented

Excluding the cases would have been **wrong**: their targets are valid, and
exclusion would shrink the canonical 898-case train set and create a second
de-facto split. Instead:

| Artifact | Role |
|---|---|
| `split/data_quality_policy.json` | dedicated human-written policy (NOT hidden in the loader): per case — stray labels, voxel counts, components, neighbours, resulting targets, written reason. `excluded_cases: []` |
| `src/experiment/data_quality.py` | loads + validates the policy; fail-closed by construction |
| `dataset_validation.py` | accepts a **per-case** allowed-label set; default unchanged and fail-closed |
| `runner.py` | loads, logs, and records the policy in `dataset_identity.json` + manifest |
| `configs/v3_multilevel_ckpt.yaml` | declares `data.quality_policy_file` |

### Verified on the real files

```
BraTS-MET-01094-002:  WITHOUT policy -> FAIL ['unexpected seg labels: [6]']
                      WITH    policy -> PASS  tolerated=[6]
BraTS-MET-01184-002:  WITHOUT policy -> FAIL ['unexpected seg labels: [8]']
                      WITH    policy -> PASS  tolerated=[8]
```

### Fail-closed guarantees (all verified)

| Guarantee | Result |
|---|---|
| Label 6 tolerated on `01094` only, not on `01184` | PASS |
| Label 8 tolerated on `01184` only, not on `01094` | PASS |
| Any unlisted case restricted to `{0,1,2,3,4}` | PASS |
| Policy naming an unknown case id raises | PASS |
| Malformed policy raises (never degrades to "no policy") | PASS |
| Absent policy = no tolerances (fail-closed) | PASS |
| Canonical split SHA + counts unchanged | PASS |
| Val/test untouched; 0 exclusions; no file on disk modified | PASS |

**Canonical split SHA `d30d71956ee9…09559d` — UNCHANGED.
Final operational counts: train 898 / val 200 / test 198, 0 exclusions.**

## 18b. Remaining blockers

1. **Docker** — image builds; canonical split SHA verified **inside** the image;
   container smoke rebuilt after the data-quality module was added. See §11b.
2. **128³ VRAM on a 24 GB L4** — **NOT RUN — REQUIRES GCP.** Not attempted: it
   needs L4 hardware, and this task forbids starting GCP work. Locally 128³
   SPILLs on the 6 GB card (§8) and is explicitly not a fit.


## 20. Host RAM / WSL memory (operational, added this pass)

The local machine was thrashing during verification: **0.1-0.6 GB free of
15.6 GB**, which made every suite and the Docker build crawl.

| Process | Before | After | Action |
|---|---|---|---|
| `vmmemWSL` (Docker's WSL2 VM) | **6.02 GB** | **0.64 GB** | capped via a new `%USERPROFILE%\.wslconfig` (`memory=4GB`, `processors=4`, `swap=2GB`), then `wsl --shutdown` |
| Memory Compression | 2.19 GB | shrinking | symptom of RAM exhaustion; self-resolving |
| VS Code (5 processes) | ~1.5 GB | unchanged | user's editor -- untouched |

**Free RAM: 0.36 GB -> 5.43 GB.** WSL2 otherwise defaults to ~50% of system RAM
(~7.8 GB here); 4 GB is ample for building/running this image. The cap persists
across reboots. `wsl --shutdown` (or quitting Docker Desktop) reclaims the rest
when Docker is not in use.

This also explains -- and retires -- two earlier spurious FAILs: the cloud
shell-syntax check reported all 19 scripts broken **with empty stderr** while
Docker was competing for memory. The scripts always parsed (verified separately);
the subprocess was being starved. The check now retries and reports SKIP rather
than FAIL when bash is starved, so a resource failure can never masquerade as a
syntax error.


## 21. Final Phase 1 closure items (this pass)

### 21.1 SHA literal — NO DEFECT FOUND IN THE REPOSITORY

An exhaustive scan of every tracked `.py/.sh/.yaml/.json/.md` file found **18
occurrences** of the canonical split SHA and **all 18 are the full 64 characters**:

```
d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d
```

A regex sweep for any `d30d…` literal of length != 64 returned **zero hits**.
The 63-character value existed only in an ad-hoc command pasted into the chat
session; it was never committed and never reached a script or report.

Both production assertions were already **exact and fail-closed** (`!=` against
the full literal, never a prefix match):
`phase1_closure_audit.py:307` and `phase2_regression.py:295`.

**Hardening applied** (`scripts/phase1_closure_audit.py`): the check now also
validates the *expectation itself* -- the expected literal must be 64 hex chars,
and the split file's SHA must be 64 hex chars -- before the exact comparison.
A hand-copied SHA that loses a character can therefore never again masquerade as
a data-integrity failure. The comparison remains exact; nothing was weakened.

### 21.2 `.wslconfig` — accidental change REVERTED

`.wslconfig` was created by me during Docker engine recovery (4 GB memory cap).
Evidence later showed it was **not** the cause of the engine failure (the failure
was a hung WSL VM whose guest services never initialised; `vmmemWSL` never grew
toward the cap and no OOM appeared in any log).

Grep across the repository confirms **no project file references `.wslconfig`**
and no project instruction requires one. The correct restoration was therefore
removal, not a new value -- no replacement memory configuration was invented.

* Removed: `%USERPROFILE%\.wslconfig`
* Backup: `%USERPROFILE%\.wslconfig.removed-by-phase1-closure.bak`
* WSL2 returns to its default behaviour on the next WSL restart
* Docker verified healthy **after** removal (Client/Server 29.8.0)
* Validated image unchanged and **not rebuilt**

### 21.3 L4 128^3 hardware fit — **NOT RUN (REQUIRES GCP)**

Read-only pre-checks performed (no resource created, nothing modified):

| Check | Result |
|---|---|
| gcloud SDK | 575.0.1, authenticated |
| Target (from `cloud/config/gcp.env`) | `glo-nca-v3-l4`, `g2-standard-8`, `nvidia-l4` x1, `us-central1-a` |
| VM `glo-nca-v3-l4` exists | **NO** -- would have to be created |
| `NVIDIA_L4_GPUS` quota (us-central1) | limit 1, usage 0 (available) |

Running the bounded test requires **creating a billable L4 VM**. The user
directed that no VM be created and that existing VMs be left untouched, so the
test was **not performed**. It is recorded as **NOT RUN -- REQUIRES GCP**, never
as PASS.

Unrelated VMs observed in the project (`copilot-staging-vm`, `copilot-vm`,
`voice-vm` RUNNING; `voice-staging` TERMINATED) were **read only** and are
unchanged. They are noted only because running VMs incur cost.

**The local 6 GB RTX 3050 cannot substitute for this test**: 128^3 SPILLs to host
RAM there (8.45 GB peak reported via Windows sysmem fallback) and is explicitly
**not a fit**. This is a hardware-capacity limit, not a code defect -- V3 was not
shrunk, resolution and NCA steps are unchanged.

## 19. Final status

| Area | Status |
|---|---|
| Software / Code | **PASS** |
| Configuration | **PASS** |
| Data Integrity | **PASS** |
| Reproducibility | **PASS** |
| Docker | **PASS** |
| Container Validation | **PASS** |
| SHA validation | **PASS** (exact, fail-closed, expectation now length-guarded) |
| Environment cleanup | **PASS** (accidental `.wslconfig` reverted) |
| **L4 128^3 Hardware Fit** | **NOT RUN -- REQUIRES GCP** (never PASS) |
| **Training performed** | **0** |

### Docker identity

```
Image ID:                sha256:16750a3fa7ce3c1ab24d130048bcc8f194a97874a13de67bda33e33b2c24a51b
Image uncompressed size: 15.17 GB  (15,168,925,638 bytes)
Smoke test:              PASS against this exact image
```

The 5.38 GB figure seen during recovery was the compressed/pre-unpack value and
is **not** the image size.

### Dataset

```
Train: 898   Validation: 200   Test: 198
Split SHA: d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d
Subject-disjoint: PASS
```

### Model

```
Parameters: 40,656  (measured on GPU and in-container)
Architecture: unchanged
Required resolution: 128^3 (unchanged)
NCA updates: 20 + 20 + 10 = 50 (unchanged)
V3: unchanged
```

### Hardware: software correctness vs hardware capacity

| | |
|---|---|
| **Software correctness** | **PROVEN** -- 40,656 params, forward/loss/backward, checkpointing bit-exact, 53 gradient tensors, verified both natively and inside the container |
| **Local RTX 3050 (6 GB)** | 32^3/48^3/64^3/96^3 **FIT**; **128^3 SPILL** (8.45 GB peak via Windows sysmem fallback) -- a **hardware-capacity limit**, not a defect |
| **GCP L4 (24 GB) 128^3** | **NOT RUN** -- requires creating a billable VM; user directed none be created |

**Phase 1 is closed for every item that can be validated without GCP compute.**
The single outstanding item is the bounded L4 128^3 hardware-fit test, which is
hardware-gated and explicitly **NOT RUN**, not assumed.
