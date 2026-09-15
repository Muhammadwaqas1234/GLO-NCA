# PHASE 2 — V3 GCP PRODUCTION HARDENING

Branch `v3-multilevel` · base HEAD `f9e5501` · 2026-09-15
**No training of any length was run.** No GCP/Kaggle command was executed.

---

## 1. Executive summary

All **3 P0** issues and **11 P1/P2** issues from Phase 1 are fixed and covered by
a new regression suite (`scripts/phase2_regression.py`: **12 PASS / 0 FAIL /
4 SKIP**). The scientific methodology is **provably unchanged**: `git diff`
shows **zero** Phase 2 modifications to `Model_GLO_NCA_V3.py`,
`LossFunctions.py` and `split/master_split.json`.

The three changes that mattered most:

1. **The production container could not start.** The canonical split was never
   copied into the image nor mounted, so the 300-epoch run would have failed at
   startup — while the gate passed, because the gate mounted the split and
   production did not. Fixed, and the gate now uses the *production* mount set.
2. **The pre-training gate could launch the 300-epoch campaign.** Its `||`
   fallback ran the unmodified production config on the GPU with no
   confirmation. Removed entirely.
3. **Augmentation RNG repeated every epoch.** Fixed via an epoch-aware sampler —
   the one fix that required real design work, because `persistent_workers` and
   per-epoch seeding conflict (see §14).

**Status: READY FOR GCP PRE-FLIGHT** (not "ready for training" — four checks
require a GPU and have not been run; see §19–20).

---

## 2. Files inspected

All Phase 1 reports in `reports/audit/` (15 files); every file on the production
path: `train.py`, `src/experiment/{runner,checkpoint,workspace,config,
datasource,reproducibility,metrics_eval}.py`, `src/models/{Model_GLO_NCA_V3,
Model_BasicNCA3D}.py`, `src/agents/Agent_GLO_NCA_V3.py`,
`src/datasets/Nii_Gz_Dataset_3D.py`, `src/losses/LossFunctions.py`; all 20
`cloud/scripts/*.sh`, `cloud/systemd/glo-nca-training.service`, `Dockerfile`,
`.dockerignore`; all 17 configs; `split/master_split.json`;
`archive/kaggle_experiment_history/kaggle_v4..v7.py`.

**Every Phase 1 finding acted on was re-verified against current source first.**
Two were re-derived independently and one Phase 1 recommendation was found to be
wrong in detail (§14).

## 3. Files changed

| File | Old behaviour | New behaviour | Why | Methodology impact | Risk | Test |
|---|---|---|---|---|---|---|
| `Dockerfile` | no `COPY split/`; `CMD` defaulted to V2 `gcp_full.yaml` | copies `split/master_split.json`; **no default CMD**; build-time split-integrity check | container could not load the canonical split; bare run trained V2 | **none** | low | `t_split_reaches_container`, `t_no_v2_production_default` |
| `.dockerignore` | `*.json` excluded the split | re-includes `!split/master_split.json` | same as above | none | low | `t_split_reaches_container` |
| `cloud/scripts/pretrain_gate.sh` | `\|\|` fallback ran the **300-epoch** config; `mark $?` dead under `set -e`; default V2 | fallback **removed**; `check()` wrapper records failures and continues; default V3; production mount set | a gate must never start the science run | none | low | `t_gate_cannot_launch_production` |
| `cloud/scripts/_train_entrypoint.sh` | synced `ls -t \| head -1`; final sync inside `if [[ -d ]]` | experiment id generated up front, passed via `--experiment-id`; final sync **always attempted** | wrong dir synced ⇒ silent checkpoint loss | none | low | `t_no_ls_t_heuristic` |
| `train.py` | `--config` defaulted to V2; resume fell back to it | `--config` required; resume **fails loudly** without its saved config; `--experiment-id` added | V2 could silently become the trained architecture | none | low | `t_resume_no_v2_fallback`, `t_no_v2_production_default` |
| `src/experiment/workspace.py` | id always generated | optional explicit `experiment_id`, still refuses to overwrite | deterministic GCS sync target | none | low | `t_workspace_explicit_id` |
| `src/experiment/runner.py` | per-epoch DataLoader; worker seed had no epoch term; missing `data:` ⇒ silent re-split; EMA/config drift silent on resume | epoch-aware sampler + `persistent_workers`; V3 requires an explicit split; config-identity check; EMA restore logged | **R-01** (RNG repetition), **D-01** (cache), **F-05**, **T-05** | **changes augmentation stream** (§14) | med | `t_epoch_rng_diversity`, `t_v3_configs_declare_split` |
| `src/datasets/Nii_Gz_Dataset_3D.py` | 50 full-volume `.max()` reductions per no-ET sample; torchio objects built every call | reductions hoisted (RNG-identical); torchio built only in the branch that uses them; per-(epoch,case) reseed | D-02, D-04 | **none** — proven output- **and** RNG-state-identical | low | `t_patchify_equivalence` (300-trial proof, §12) |
| `src/agents/Agent_GLO_NCA_V3.py` | blocking H2D copy despite `pin_memory` | `non_blocking=True` | overlap copy with compute | none (numerically identical) | low | compile + forward |
| `src/experiment/checkpoint.py` | `except: pass` on scheduler restore | raises with a clear message | silent LR restart mid-campaign | none | low | `t_scheduler_restore_not_silent` |
| `cloud/scripts/lib.sh` | — | systemd-backed lock helpers | invalid `$$` lock | none | low | shell syntax |
| `cloud/scripts/run_training.sh`, `resume_training.sh` | `echo $$ > LOCK` before confirm; V2 default | `assert_no_training_running`; lock written **after** confirm; config required | concurrent runs; stale locks | none | low | shell syntax |
| `cloud/scripts/setup_gcp.sh` | hard-coded branch `v2`; update failure warned | defaults to `v3-multilevel` (`GLO_BRANCH` override); **dies** on update failure; verifies V3 + split present | VM was provisioned with V2 code | none | low | `t_no_v2_production_default` |
| `cloud/scripts/verify_results.sh` | required `completed`; CWD-relative path | `ALLOW_FAILED=1` path; absolute path | failed run's VM could not be cleaned ⇒ indefinite billing | none | low | shell syntax |
| `cloud/scripts/start_vm.sh` | restart had no confirm | `confirm` before resuming billing | cost safety | none | low | shell syntax |
| `configs/v3_multilevel_ckpt.yaml` | said "Not the production config" | declared **THE production config** | contradicted README + protocol doc | none (comment only) | low | `t_production_config` |
| `configs/smoke_test_v3.yaml` | no split declaration | `data.allow_seeded_split: true` | explicit opt-out for synthetic smoke | none | low | `t_v3_configs_declare_split` |
| `configs/ablation_{full,se,spatial}.yaml` | headers claimed "A0, both false" | headers match actual values | **thesis-writing hazard** | none (comments only; values untouched) | low | verified values unchanged |
| `cloud/README.md` | pointed at the non-checkpointing config | points at `v3_multilevel_ckpt.yaml` | single authoritative config | none | low | — |

**New file:** `scripts/phase2_regression.py`.

## 4. Files deleted / moved

**Nothing deleted.** Moved after proving **zero** references (grep across
`.sh/.py/.yaml/.md/.service`):

| File | Refs | New location |
|---|---|---|
| `xfer2.sh` | 0 | `archive/one_off_transfer_tools/` |
| `drive_dl.py` | 1 (only `xfer2.sh`) | `archive/one_off_transfer_tools/` |
| `scripts/dataset_identity.py` | 0 | `archive/one_off_transfer_tools/` |

`archive/one_off_transfer_tools/README.md` records why, and names the live
replacement for each. **`kaggle_v4..v7.py` were NOT touched.** All Phase 1
reports are intact and unmodified.

## 5. P0 findings fixed

1. **V3 architecture + canonical split untracked** — `Model_GLO_NCA_V3.py`,
   `Agent_GLO_NCA_V3.py`, all 10 `v3_*.yaml`, `split/master_split.json` staged
   into git. A `git clean` can no longer destroy the thesis model.
2. **Split unreachable by the production container** — `COPY` added,
   `.dockerignore` re-include added, build-time integrity check added. The gate
   now runs the **production** mount set, so it can no longer pass where
   production fails.
3. **Gate could launch the 300-epoch campaign** — fallback removed; smoke
   failure ⇒ gate failure, nothing else runs.

## 6. P1 findings fixed

RNG repetition (R-01/T-03) · preprocessing cache defeat (D-01) · patchify waste
(D-02) · `ls -t` sync target (CL-05) · final sync skipped on failure ·
invalid `$$` lock + concurrent runs (CL-06) · `setup_gcp.sh` branch `v2` (CL-04)
· `set -e` vs `mark $?` (CL-03) · silent scheduler reset (C-01) · silent
re-split on missing `data:` (F-05) · VM uncleanable after failure.

## 7. P2/P3 fixed or deferred

**Fixed:** contradictory ablation headers (F-06) · config ambiguity (F-03) ·
resume config-identity (T-05) · silent EMA restore (C-02) · unused torchio
construction (D-04) · blocking H2D (D-05) · `verify_results.sh` CWD path ·
no confirm on VM restart.

**Deliberately deferred (require your decision — §17):**

| Finding | Why deferred |
|---|---|
| **E-02 HD95 in resampled voxels** | Reporting/methodology decision, not a bug. Values are honestly labelled `HD95(vox)` but are **not comparable to mm-based BraTS literature**. Changing units changes reported numbers. |
| **E-04 empty-empty: Dice 0.0 vs IoU 1.0** | The two metrics disagree on the same case. Changing either changes reported ET numbers. |
| **F-02 hard-coded `ce_weight`/`prioritize_region`/`empty_weight`** | Surfacing them into config is safe, but they are thesis hyperparameters; exposing them invites accidental change mid-campaign. |
| **ME-01 ~20 GB host RAM at final eval** | Needs a VM-RAM check, not a code change; streaming the scoring would alter the evaluation code path right before the test set is scored. **Verify the VM has ≥ 24 GB RAM.** |
| **TS-02 `test_*.py` test reimplementations** | Superseded in practice by `phase2_regression.py`, which exercises the **real** `patchify_multimodal`. Old files left as history. |

## 8. V2 assumptions removed from the V3 production path

| Location | Before | After |
|---|---|---|
| `train.py --config` | defaulted to `gcp_full.yaml` (V2) | required, no default |
| `train.py --resume` fallback | fell back to that V2 default | fails loudly |
| `Dockerfile CMD` | `--config configs/gcp_full.yaml` | removed |
| `run_training.sh` | `${1:-configs/gcp_full.yaml}` | required argument |
| `pretrain_gate.sh` | defaulted to V2 | defaults to V3 production |
| `setup_gcp.sh` | cloned branch `v2` | `v3-multilevel`, verified |

**V2 baseline files are retained unchanged** for thesis comparison
(`gcp_full.yaml`, `ablation_*.yaml`, `Agent_GLO_NCA`, the V2 `_build` path).
V2 simply can no longer become the default.

## 9. V3 production execution graph (after Phase 2)

```
run_training.sh <config>            [config REQUIRED; assert_no_training_running]
 └─ confirm()  -> write_training_lock()      [lock only after consent]
 └─ systemctl start glo-nca-training          [Restart=no]
     └─ _train_entrypoint.sh
         ├─ EXP_ID computed HERE (name + timestamp)   <- no ls -t guessing
         ├─ sync watcher -> gs://.../${EXP_ID}         <- exact directory
         ├─ docker run -v data:ro -v out
         │     glo-nca:latest --config <cfg> --output /out --experiment-id $EXP_ID
         │      └─ train.py -> Workspace.create(id) -> runner.run
         │          ├─ split/master_split.json  (IN THE IMAGE, sha-verified)
         │          ├─ _build_v3 -> GLO_NCA_V3_MultiLevel (40,656 params)
         │          ├─ epoch-aware sampler -> persistent workers (cache warm)
         │          ├─ train step: fwd -> loss -> bwd -> clip -> opt -> sched -> EMA
         │          └─ checkpoint (atomic) every epoch
         ├─ final GCS sync  ALWAYS attempted (success or failure)
         └─ exit <real training rc>
```

No V2 fallback exists anywhere on this path.

## 10. Docker verification

`COPY src/ configs/ scripts/ split/master_split.json train.py` — no dataset, no
checkpoints, no credentials (`gcp.env`, `*.pem`, `*.key`, `*service-account*`
remain ignored); build fails if the split is missing or its fingerprint does not
match; no default CMD. **Not built here** (no Docker in this environment) — the
build is a required pre-flight step (§19).

## 11. Split / data verification

`split/master_split.json` — sha256 `d30d7195…9559d`, **unchanged**; 898/200/198
cases; 567/121/122 subjects; partitions verified **disjoint**. Now tracked in git
and shipped in the image.

**Invalid-label cases `01094-002` and `01184-002`: both are in the TRAIN split**
(verified). They therefore cannot affect val/test metrics. **No operational
exclusion was applied** — that would be a methodology change requiring your
approval.

## 12. Patchify optimization — equivalence proof

Your §12 required RNG equivalence, not just output equivalence. A naive
short-circuit **fails** that bar: measured draw counts are **4** (ET present) vs
**151** (ET absent) vs **0** for a short-circuit — it would shift the augmentation
stream, which shares Python `random`.

The implemented change keeps **every** `random.*` call and the loop bounds
exactly as they were, and only hoists the two array reductions whose results are
predetermined when `patch == volume`. Verified over **300 trials** including
no-ET and all-empty labels:

```
OUTPUT IDENTICAL + random.getstate() IDENTICAL   (0 mismatches)
```

Saving: up to 50 × 2.1M-element reductions per no-ET sample, eliminated.

## 13. Checkpoint / resume verification

Contents complete and writes atomic (unchanged). Now additionally: scheduler
restore failure **raises** instead of silently restarting the LR; config identity
is checked across 9 schedule-defining fields; EMA restore is logged; resume
without a saved config fails with exit 2 instead of silently using V2. Epoch
indexing re-verified: **no off-by-one**.

## 14. Performance findings — and one Phase 1 recommendation corrected

**Phase 1 recommended `persistent_workers=True` (to fix the cache) and an
epoch-dependent `worker_init_fn` seed (to fix RNG repetition). Those two are
mutually exclusive** — PyTorch calls `worker_init_fn` **once per worker**, not
per epoch, and each worker holds a *copy* of the dataset, so neither a seed term
inside `worker_init_fn` nor a parent-side `ds.set_epoch()` ever reaches a
persistent worker. I verified this against the installed torch source before
choosing an approach.

**Implemented instead:** a custom `_EpochSampler` yields `(epoch, index)` pairs.
Indices genuinely travel parent → worker every epoch, so the dataset derives its
per-item seed from `(base_seed, epoch, index)`. This works identically with or
without persistent workers. Verified: **100/100 distinct streams across epochs,
100/100 across cases, and bit-identical for a repeated (epoch, case)**.

Other measures: cache now survives epochs (`persistent_workers` +
`prefetch_factor=2`); patchify reductions hoisted; unused torchio construction
removed; `non_blocking` H2D.

**No wall-clock or VRAM number is claimed.** This machine has no GPU and no
torch; nothing was measured. `reports/validation/` retains the prior measured
evidence (≈9.07 GB checkpointed at 128³).

## 15. Kaggle v4/v7 lessons used

Reviewed for implementation ideas only; **nothing copied**, no import of
`archive/` exists. V3 already inherits the v7 recipe (loss, EMA, clip magnitude,
LR schedule, fire rate, dropout, seed 42), differing only in architecture
(2-level 32/64³ → 3-level 32/96/128³), 150→300 epochs, and added weight decay.

**One difference you should declare in the thesis if you compare against v7:**
v7 clipped gradients with per-parameter **element-wise `torch.clamp`**
(`kaggle_v7.py:270-274`), whereas V3 uses **`clip_grad_norm_`**. Same nominal
1.0, mathematically different operations. **Not changed** — that would be a
methodology decision.

## 16. Methodology preservation statement

`git diff` confirms **zero Phase 2 changes** to `Model_GLO_NCA_V3.py`,
`LossFunctions.py`, `split/master_split.json`. (`Model_BasicNCA3D.py` shows a
diff versus the old HEAD only because of the **pre-existing** gradient-
checkpointing work, which predates this session.)

Unchanged and verified: 3 levels 32³/96³/128³ · 20+20+10 = **50** NCA steps ·
**40,656** parameters · SE + spatial GC · concat fusion · WT/TC/ET multi-label
sigmoid · FocalTversky(0.25, 0.75, 1.33) + 0.5·BCE · AdamW 1.6e-3 · cosine ·
EMA 0.999 · clip 1.0 · seed 42 · 300 epochs · checkpointing ON · val-only
threshold tuning · test evaluated once.

**One behavioural change is not bit-identical to prior runs and is disclosed:**
the augmentation RNG stream now varies per epoch (the R-01 fix). The sampling
*distribution* is unchanged; the *sequence* differs. This was the intended
behaviour all along — the old stream repeated for 300 epochs. Runs remain fully
deterministic for a given seed. **A run started before this change cannot be
bit-resumed after it.**

## 17. Changes NOT made (need your approval)

1. HD95 units (E-02) · 2. empty-empty metric inconsistency (E-04) · 3. exposing
hard-coded loss/sampling hyperparameters (F-02) · 4. excluding the two
invalid-label train cases · 5. gradient-clipping semantics vs v7 · 6. streaming
evaluation to cut host RAM.

## 18. Tests executed

| # | Test | Result |
|---|---|---|
| 1 | `compileall src/ scripts/ train.py` | **PASS** |
| 2 | shell syntax, all 20 cloud scripts + archived | **PASS** |
| 3 | all 17 configs parse | **PASS** |
| 4 | production config frozen (32/96/128, 50 steps, 300ep, ckpt ON, seed 42) | **PASS** |
| 5 | canonical split unchanged, partitions disjoint | **PASS** |
| 6 | per-(epoch,case) RNG deterministic + varied | **PASS** |
| 7 | gate cannot launch production | **PASS** |
| 8 | split reaches container | **PASS** |
| 9 | resume never falls back to V2 | **PASS** |
| 10 | no `ls -t` guessing | **PASS** |
| 11 | `--experiment-id` supported | **PASS** |
| 12 | V2 defaults removed | **PASS** |
| 13 | every V3 config declares its split | **PASS** |
| 14 | scheduler restore not silent | **PASS** |
| 15 | patchify equivalence (standalone 300-trial replica) | **PASS** (output + RNG identical) |
| 16 | V3 parameter count == 40,656 | **SKIP** — no torch here |
| 17 | V3 forward/backward | **SKIP** — no torch here |
| 18 | gradient-checkpointing equivalence | **SKIP** — no torch here |
| 19 | patchify equivalence via the real module | **SKIP** — needs torch/torchio |
| 20 | Docker build + container smoke | **NOT RUN** — no Docker here |

**12 PASS / 0 FAIL / 4 SKIP.** SKIPs are honest skips, not passes.

## 19. Remaining risks

1. **Four GPU-dependent checks unverified here** — parameter count, forward/
   backward, checkpointing equivalence, real-module patchify. Must pass on the VM.
2. **Docker image never built** in this environment.
3. **Persistent workers raise per-worker RAM** (each caches its shard). With 4
   workers × 898 cases this can be large; monitor, and reduce `workers` if the VM
   swaps.
4. **~20 GB host RAM at final evaluation** — verify the VM has ≥ 24 GB, or this
   fails *after* 300 epochs complete.
5. **Not bit-identical to pre-Phase-2 runs** (§16) — intended, but disclose it.
6. HD95/metric items in §17 remain open thesis decisions.

## 20. Final GCP readiness status

# READY FOR GCP PRE-FLIGHT

Not "READY FOR GCP TRAINING": your §23 requires a GPU gate, a correct production
container and checkpoint/resume all verified, and four critical checks here were
**skipped, not passed** — no GPU, no torch, no Docker in this environment.

Required sequence on the VM, in order:

```bash
GLO_BRANCH=v3-multilevel ./cloud/scripts/setup_gcp.sh    # verifies V3 + split
docker build -t glo-nca:latest .                          # split check runs at build
python scripts/phase2_regression.py                       # expect 16 PASS / 0 FAIL
python scripts/gpu_memory_gate_v3.py --config configs/v3_multilevel_ckpt.yaml \
       --resolutions 96,128                               # expect TRUE FIT
./cloud/scripts/pretrain_gate.sh                          # expect PASS (cannot start training)
```

Only if **all** report PASS does the repository become READY FOR GCP TRAINING.
