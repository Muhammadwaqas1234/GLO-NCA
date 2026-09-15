# PHASE 1 — FILE-BY-FILE AUDIT RECORDS

Records for every **production runtime** file. Config/cloud/script/test records
are in CONFIG_AUDIT.md, CLOUD_AUDIT.md, TEST_AUDIT.md and
PHASE1_REPOSITORY_INVENTORY.md. Recommendations are findings only — **nothing
implemented**.

---
## FILE: train.py
**PURPOSE** CLI entry point. **ACTUAL ROLE** matches — thin wrapper, all logic in
`src/experiment/`. **USED BY** operator, Docker ENTRYPOINT.
**IMPORTS** `config.load_config`, `Workspace`, `runner`.
**INPUTS** 5 CLI args. **OUTPUTS** exit code; failure report on exception.
**HARDCODED** `--config` default `configs/gcp_full.yaml` (**V2**).
CORRECTNESS PASS · PERFORMANCE PASS · MEMORY PASS · REPRODUCIBILITY PASS ·
ERROR HANDLING **PASS (good)** — top-level guard writes `failure_report.txt`,
sets status, distinguishes SIGINT (130). LEAKAGE N/A · METHODOLOGY **CONCERN**
DUPLICATION None · DEAD CODE None
**FINDINGS** (1) default config is V2, so a bare `python train.py` silently runs
the wrong architecture. (2) `:63-64` on `--resume` with a missing workspace
config falls back to that same V2 default — fails loudly downstream, but
confusingly. **SEVERITY P3** **RECOMMENDATION** make `--config` required, or
default to the V3 production config once F-03 is resolved.

---
## FILE: src/experiment/runner.py
**PURPOSE** orchestrate a full fault-tolerant experiment. **ACTUAL ROLE** matches;
704 lines; also owns the training step, split resolution, EMA, eval and packaging.
**USED BY** `train.py`. **EXPORTS** `run`, `_clipped_batch_step`, `_build_v3`.
**CONFIGURATION** nearly every config key. **HARDCODED** see CONFIG_AUDIT F-02.
CORRECTNESS **CONCERN** · PERFORMANCE **CONCERN** · MEMORY **CONCERN** ·
REPRODUCIBILITY **FAIL** · ERROR HANDLING PASS · LEAKAGE **PASS** ·
METHODOLOGY **CONCERN** · DUPLICATION None · DEAD CODE None
**FINDINGS** T-02 loader recreated per epoch (P1); **T-03 worker seeds do not
vary by epoch, so augmentation RNG repeats (P1)**; T-01 `zero_grad` default (P2);
T-05 no config-identity check on resume (P2); F-05 missing `data:` silently
re-splits (P1); F-02 invisible hard-coded methodology (P1). Training step order,
EMA timing, grad-clip placement, empty-region handling and frozen-test discipline
are all **correct**. **SEVERITY P1**

---
## FILE: src/models/Model_GLO_NCA_V3.py  *(UNTRACKED)*
**PURPOSE** unified 3-level GLO-NCA. **ACTUAL ROLE** matches exactly.
**USED BY** `runner._build_v3`. **IMPORTS** `BasicNCA3D`.
**EXPORTS** `GLO_NCA_V3_MultiLevel`, `LevelSpec`, `FeatureProjection`,
`build_v3_from_config`.
**INPUTS** `(B,X,Y,Z,4)` **OUTPUTS** `(B,3,X,Y,Z)` — both verified.
**CONFIGURATION** all `model.*` plus `memory.gradient_checkpointing`.
CORRECTNESS **PASS** · PERFORMANCE PASS · MEMORY **CONCERN** ·
REPRODUCIBILITY PASS · ERROR HANDLING PASS · LEAKAGE N/A ·
METHODOLOGY **PASS** · DUPLICATION None · DEAD CODE None
**FINDINGS** Architecture verified correct: 3 levels, strict flow, spatial
alignment, channel bookkeeping, learnable fusion, single sigmoid, **40,656
params confirmed by independent recomputation**. M-03: `build_v3_from_config:243`
defaults checkpointing OFF and `v3_multilevel.yaml` has no `memory:` block, so
the file named production by `cloud/README.md` runs **without** the checkpointing
the brief requires (P1). F-01: **file is untracked** (P0).
**SEVERITY P0 (untracked)** **RECOMMENDATION** commit it; resolve F-03.

---
## FILE: src/models/Model_BasicNCA3D.py
**PURPOSE** per-cell NCA rule + SE + spatial GC. **ACTUAL ROLE** matches.
**USED BY** V3 (3 levels) **and V2** — shared.
CORRECTNESS **PASS** · PERFORMANCE **CONCERN** · MEMORY **CONCERN** ·
REPRODUCIBILITY **PASS** · ERROR HANDLING PASS · METHODOLOGY **PASS** ·
DUPLICATION None · DEAD CODE None
**FINDINGS** Gradient checkpointing **verified correct**: `use_reentrant=False`,
`preserve_rng_state=True`, one-step boundary, no double-wrapping, all levels,
OFF under `no_grad`. Outputs/gradients/RNG unchanged — the "checkpointing is
safe" claim holds. M-01 (~300 transposes per forward) and M-02 (per-step
`clone` + `concat`) are inherited V2 code (INFO). Memory peak is the
`hidden=128` activation at L2/L3. **SEVERITY INFO**
**RECOMMENDATION** **DO NOT MODIFY** — shared with V2; any change breaks the
V2-vs-V3 comparison.

---
## FILE: src/agents/Agent_GLO_NCA_V3.py  *(UNTRACKED)*
**PURPOSE** adapter presenting V3 through the V2 agent interface.
**ACTUAL ROLE** matches; correctly skips `make_seed`.
CORRECTNESS PASS · PERFORMANCE **CONCERN** · MEMORY PASS ·
REPRODUCIBILITY PASS · METHODOLOGY PASS · DEAD CODE None (inherited only)
**FINDINGS** D-05 `.to(device)` without `non_blocking=True` despite
`pin_memory=True` (P2). Inherits unused `Pool` / `make_seed` / `ExponentialLR` —
harmless and documented. F-01 untracked (P0). **SEVERITY P0 (untracked)**

---
## FILE: src/datasets/Nii_Gz_Dataset_3D.py
**PURPOSE** BraTS loading, preprocessing, patchify, augmentation.
**ACTUAL ROLE** matches. **USED BY** runner (train + eval).
CORRECTNESS **PASS** · PERFORMANCE **FAIL** · MEMORY PASS ·
REPRODUCIBILITY **CONCERN** · ERROR HANDLING PASS (raises on missing modality) ·
LEAKAGE **PASS** · METHODOLOGY PASS · DUPLICATION None · DEAD CODE **confirmed**
**FINDINGS** Label mapping, interpolation modes, modality order, per-case
normalisation and train-only gating of patchify/augment are all **correct**.
**D-01** cache never works across epochs (P1). **D-02** patchify is a guaranteed
no-op at 128 cubed yet can burn 50 x 2.1M-element reductions per sample (P1).
**D-03** `rescale3d` issues roughly 1,415 Python-level `cv2.resize` calls per
case (P2). **D-04** torchio transforms constructed every call, never used (P2).
**SEVERITY P1** **RECOMMENDATION** all candidate fixes are RNG-stream-affecting —
see DATA_PIPELINE_AUDIT section 4 before touching.

---
## FILE: src/experiment/metrics_eval.py
**PURPOSE** evaluation + threshold tuning. **ACTUAL ROLE** matches.
CORRECTNESS **CONCERN** · PERFORMANCE **CONCERN** · MEMORY **CONCERN** ·
LEAKAGE **PASS** · METHODOLOGY **CONCERN**
**FINDINGS** Frozen-test discipline **correct**: tuning is val-only, test is
collected once, thresholds are frozen. E-02 HD95 is voxel-based on a resampled
cube and measures distance-to-foreground (P1, thesis risk). E-04 empty-empty
scores Dice **0.0** but IoU **1.0** — the two metrics disagree on the same input
(P2, verified numerically). Eval loader uses `num_workers=0` (P2).
`collect_probs` holds roughly 10 GB (val) and 20 GB (val+test live) of host RAM
(P1 on small hosts). **SEVERITY P1**

---
## FILE: src/experiment/checkpoint.py
**PURPOSE** complete-state checkpointing. **ACTUAL ROLE** matches.
CORRECTNESS **CONCERN** · ERROR HANDLING **CONCERN** · REPRODUCIBILITY PASS
**FINDINGS** Contents complete (model/optimizer/scheduler/EMA/epoch/best/history/
config/RNG) and writes are **atomic** (`.tmp` + `os.replace`) — genuinely good.
**C-01** bare `except: pass` around scheduler restore can silently restart the
cosine LR mid-campaign (P1). C-04 `weights_only=False` is justified and
acceptable in this workflow. **SEVERITY P1**

---
## FILE: src/experiment/reproducibility.py
CORRECTNESS PASS · REPRODUCIBILITY **CONCERN**
**FINDINGS** Seeds python/numpy/torch/cuda; captures and restores full parent RNG
state. `describe()` is **honest** about not enforcing bit-exactness. R-02 worker
RNG is not capturable; R-03 cudnn left in default mode (deliberate, documented).
The actual defect (R-01) lives in `runner._worker_init`, not here.
**SEVERITY P2**

---
## FILE: src/losses/LossFunctions.py
CORRECTNESS **PASS** · METHODOLOGY PASS · DEAD CODE **confirmed**
**FINDINGS** `FocalTverskyCELoss` verified: single sigmoid, BCE on clamped probs
(`1e-6, 1-1e-6` — numerically stable), Tversky with alpha 0.25 / beta 0.75, focal
gamma 1.33. **No double sigmoid, no detach, no CPU transfer, no redundant
allocation.** Gradient flow intact. `DiceLoss` / `DiceCELoss` / `TverskyCELoss`
are unused (INFO — historical/ablation value; do not delete).
**SEVERITY INFO** **RECOMMENDATION** no change.

---
## FILE: src/utils/Experiment.py
CORRECTNESS **CONCERN** · REPRODUCIBILITY **CONCERN**
**FINDINGS** Legacy glue retained for V2 compatibility. `get_from_config` returns
`None` for any unknown key (`:186-193`), making config typos structurally
undetectable (P2). `set_size:124-126` is the mechanism behind D-02 — it sets
`self.size` to 128 cubed, used by **both** the resize and the patchify. Inert V3
placeholders (`channel_n=16`, `inference_steps=10`) are written into the
experiment `config.dt`, where a reader could mistake them for real V3 settings
(P3). **SEVERITY P2**

---
## FILE: src/agents/Agent_Multi_NCA.py
DEAD CODE **confirmed** · CORRECTNESS **CONCERN**
**FINDINGS** `batch_step:8-33` is **never called** (the runner uses its own
`_clipped_batch_step`) **and diverges from the live path** — it lacks gradient
clipping and the empty-region BCE fallback. A maintenance trap: "fixing" it would
suggest the live path was fixed. **SEVERITY P2**
**RECOMMENDATION** do not delete yet (V2 imports the class); document as dead.

---
## FILE: src/agents/Agent.py
**PURPOSE** base agent + `iou_score` / `hd95_score`. **USED BY** all agents;
metrics imported by `metrics_eval`.
CORRECTNESS **CONCERN** · METHODOLOGY **CONCERN**
**FINDINGS** E-02 (HD95 units and surface definition) and E-04 (empty-empty
IoU 1.0 vs Dice 0.0) both originate here (`:29-77`). `_make_optimizer:108-121`
correctly defaults to AdamW. **SEVERITY P1**

---
## FILE: src/agents/Agent_GLO_NCA.py (V2 only)
**FINDINGS** `:17-18` reads `stacked_models` and `scaling_factor` — neither is
defined in any YAML nor in the flat dict, so both are silently `None` and never
used. Note the near-miss with the real key `scale_factor` (`:32`). Demonstrates
the silent-None hazard. **SEVERITY P3**

---
## FILE: src/datasets/Data_Instance.py
**FINDINGS** Correct as written, but its stated purpose ("only needs to be done
once", `:4-5`) is **not achieved in production** — see D-01. The defect is in how
the runner drives it, not in this file. **SEVERITY INFO**

---
## FILE: src/datasets/Dataset_3D.py / Dataset_Base.py
**FINDINGS** `Dataset_3D.preprocessing` and `__getitem__` are fully overridden
and **dead** (the 2D-oriented `preprocessing` is unreachable for BraTS).
`Dataset_Base` is live. **SEVERITY P3**

---
## FILES: src/experiment/{config,datasource,dataset_validation,environment,
graphs,logutil,statistics,workspace,diagnostics}.py
All **live** (imported by `runner.py:32-41`); **no unused experiment module**.
**FINDINGS** `config.py` thin validation (F-07, P2). `datasource.py` is a
**strength**: sha256-verified split load that refuses a tampered file, plus a
hard subject-disjointness gate. `dataset_validation.py` is a genuine gate, not
decorative. Others PASS. **SEVERITY P2 (config.py), PASS elsewhere**
