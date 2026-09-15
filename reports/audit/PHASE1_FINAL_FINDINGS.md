# PHASE 1 — FINAL FINDINGS

Audit-only. No production code was modified. No training, Kaggle, or GCP command
was executed.

## Findings table

| ID | File | Line/Function | Finding | Sev | Evidence | Confidence | Safe Fix? | Methodology Impact |
|----|------|---------------|---------|-----|----------|------------|-----------|--------------------|
| F-01 | `src/models/Model_GLO_NCA_V3.py`, `src/agents/Agent_GLO_NCA_V3.py`, `configs/v3_*.yaml`, `split/master_split.json` | whole files | Entire V3 architecture + canonical split are **untracked** at HEAD `f9e5501` | **P0** | `git ls-files` returns none of them | CONFIRMED | Yes — `git add` | None (protects it) |
| CL-01 | `Dockerfile:48-51`, `.dockerignore:34`, `cloud/scripts/_train_entrypoint.sh:54-61` | container build/run | `split/` is never COPY'd nor mounted in the **production** path; `*.json` is also ignored. Training cannot load the master split | **P0** | verified all three files | CONFIRMED | Yes — add mount/COPY | None (restores intended split) |
| CL-02 | `cloud/scripts/pretrain_gate.sh:99-108` | smoke fallback | `\|\|` fallback runs the **unmodified 300-epoch config** on the GPU, with **no confirm**, contradicting the script header (`:6`) | **P0** | read verbatim | CONFIRMED | Yes — remove fallback | None |
| T-03 / R-01 | `src/experiment/runner.py:424-428` | `_worker_init` | Worker seed = `cfg.seed + worker_id`, **no epoch term**; loader recreated each epoch (`:434`) so the same 4 seeds recur → augmentation/patch RNG repeats every epoch | **P1** | source; `random`/`np.random` both used by `_augment`/`patchify` | CONFIRMED (defect); effect size UNMEASURED | Yes, one line — **but shifts the RNG stream** | Reduces augmentation diversity; fixing changes run-to-run continuity |
| D-01 | `src/experiment/runner.py:434-437`, `src/datasets/Data_Instance.py` | epoch loop | Preprocessing cache never persists: new DataLoader per epoch, `num_workers=4`, no `persistent_workers`; workers mutate their own copy and die | **P1** | grep confirms `persistent_workers` absent repo-wide | CONFIRMED | Yes | None |
| D-02 | `src/datasets/Nii_Gz_Dataset_3D.py:335-376` | `patchify_multimodal` | At production 128³, `self.size` == volume shape, so `randint(0,0)`=0 — patch extraction is a **guaranteed no-op**, yet can still run 50 iterations × 2.1M-element `.max()` | **P1** | `Experiment.set_size:124-126` → `self.size=(128,128,128)` used by both resize and patchify | CONFIRMED | Yes — **but removes RNG draws, shifting augmentation stream** | None on distribution; breaks bit-continuity |
| M-03 / F-03 | `configs/v3_multilevel.yaml` vs `_ckpt.yaml`; `Model_GLO_NCA_V3.py:243` | `memory.gradient_checkpointing` | Two near-identical configs; docs contradict each other on which is production; the non-ckpt one runs **without** the required checkpointing | **P1** | `README.md:19` vs `cloud/README.md:171`; ckpt file says "Not the production config" | CONFIRMED | Yes — pick one, delete/rename other | **Decision required**: ~8× VRAM vs ~2× wall-clock |
| C-01 / T-04 | `src/experiment/checkpoint.py:76-79` | `restore_into` | Bare `except: pass` on scheduler restore → silent fresh cosine at LR 1.6e-3 mid-campaign, unlogged | **P1** | source; optimizer restore is unguarded by contrast | CONFIRMED | Yes — log + re-raise | Would corrupt a run if triggered |
| E-02 | `src/agents/Agent.py:48-77` | `hd95_score` | HD95 in **voxels on a 128³ resampled cube** (no `sampling=`), and measures distance-to-foreground not to-surface | **P1** | source | CONFIRMED | Reporting change only | **Not comparable to mm-based BraTS literature** |
| F-05 | `src/experiment/runner.py:293, 312-316`; `config.py:65-68` | split resolution | Missing `data:` section → **silently generates a fresh seeded split** and trains. `data` is not a required section | **P1** | `configs/ablation.yaml` has no `data:` | CONFIRMED | Yes — require it | Would silently change the split |
| F-02 | `runner.py:63,120,123,137,363`; `metrics_eval.py:85` | hard-coded | `ce_weight=0.5`, `empty_weight=0.1`, `priotize_masks=0.7`, `prioritize_region=2`(ET), betas, optimizer, threshold grid are **absent from every config** | **P1** | grep across all 17 configs | CONFIRMED | Yes — surface into config | Saved config does not fully describe methodology |
| CL-04 | `cloud/scripts/setup_gcp.sh:31-37` | branch | Hard-codes branch **`v2`** — a VM provisioned this way gets V2 code | **P1** | source | CONFIRMED | Yes | Would run wrong architecture |
| CL-05 | `_train_entrypoint.sh:35-40` | sync watcher | Target chosen by `ls -t \| head -1`; wrong dir → live experiment **never synced to GCS**; errors silenced | **P1** | source | CONFIRMED | Yes | Checkpoint-loss risk |
| CL-06 | `verify_results.sh:62`, `delete_vm.sh:32-35`, `run_training.sh:57`, `start_vm.sh:17` | cleanup/lock | VM survives a failed run (verify requires `completed`); lock stores the launcher PID not the systemd process → concurrent runs possible; no confirm on VM restart | **P1** | source | CONFIRMED | Yes | Cost + GPU contention |
| CL-03 | `cloud/scripts/lib.sh:7` + gate scripts | `set -e` vs `mark $?` | `set -e` inherited by all cloud scripts makes the `mark $?` accumulation idiom dead; gates abort on first failure with no summary | **P1** | shell-semantics; `pretrain_gate.sh:159` unreachable | CONFIRMED | Yes | None |
| ME-01 | `src/experiment/metrics_eval.py:27-29` + `runner.py:520-523` | `collect_probs` | All case probabilities held on host: ~10 GB per val pass, **~20 GB with val+test live** at final eval | **P1** on small hosts | computed: 128³×3×4 B × 2 × 398 | CONFIRMED | Yes — stream/threshold incrementally | None |
| TS-01/02 | `scripts/test_*.py` | whole files | No test framework; both `test_*` files test **reimplementations**, not production code; one docstring falsely claims it imports the real method | **P1** | `test_patchify_equivalence.py:11` vs `:48` | CONFIRMED | Yes | Tests can pass while production is broken |
| E-04 | `Agent.py:42-44` vs `metrics_eval.py:41-43` | empty-empty | Same all-empty case scores **Dice 0.0** but **IoU 1.0** | P2 | verified numerically | CONFIRMED | Yes | Metrics disagree; affects reported ET |
| D-03 | `Nii_Gz_Dataset_3D.py:232-240` | `rescale3d` | Two Python slice loops → ~1,415 `cv2.resize` calls per case, repeated every epoch (via D-01) | P2 | source | CONFIRMED cost; POSSIBLE significance | Yes | None |
| D-04 | `Nii_Gz_Dataset_3D.py:142-143` | `__getitem__` | torchio transforms constructed every call; never used (`nonzero_norm=True`) | P2 | source | CONFIRMED | Yes | None |
| D-05 | `Agent_GLO_NCA_V3.py:60-61` | `prepare_data` | `.to(device)` without `non_blocking=True` despite `pin_memory=True` | P2 | source | CONFIRMED | Yes | None |
| T-05 / C-03 | `runner.py:392-406` | resume | `ck["config"]` saved but **never compared**; hyperparameter drift on resume passes silently | P2 | source | CONFIRMED | Yes | Could silently alter the LR schedule |
| C-02 | `runner.py:396-397` | EMA restore | Falsy `ck["ema"]` → silently keeps fresh EMA, unlogged | P2 | source | CONFIRMED | Yes | Bounded (EMA re-converges) |
| F-06 | `ablation_{full,se,spatial}.yaml:2-10` | headers | Copy-pasted A0-baseline headers **contradict** the files' own `use_attention`/`use_spatial` values | P2 | compared `:2-10` vs `:34-35` | CONFIRMED | Yes — docs only | **A methodology chapter written from these comments would be wrong** |
| F-04 | all 17 configs | `evaluation.threshold` | Present everywhere, **read nowhere**; 0.5 is hard-coded | P2 | grep | CONFIRMED | Yes | None |
| F-07 | `config.py:65-99` | validation | Only section names + 2 value checks; no type/range checks, unknown keys never rejected | P2 | source | CONFIRMED | Yes | Typos silently default |
| R-03 | `reproducibility.py` | cudnn | `deterministic`/`benchmark` never set | P2 | grep: absent repo-wide | CONFIRMED | — | Deliberate + documented; do not claim bit-exactness |
| E-01 | `runner.py:486, 521` | validation reuse | Validation drives both model **and** threshold selection | P2 | source | CONFIRMED | N/A | Val metrics are optimistically biased — report test only |
| P-01 | `Agent_Multi_NCA.py:8-33` | `batch_step` | Dead **and divergent** from live path (no grad clip, no empty-region BCE) | P2 | grep: never called | CONFIRMED | Do not delete yet | Maintenance trap |
| U-01 | `xfer2.sh`, `scripts/dataset_identity.py` | whole files | **Zero references** anywhere | P3 | grep | CONFIRMED UNUSED | Do not delete yet | None |
| T-01 | `runner.py:60-61` | `zero_grad` | No `set_to_none=True` | P3 | source | CONFIRMED | Yes | Negligible (40k params) |
| A-01 | `kaggle_v7.py:270-274` vs `runner.py:77` | grad clip | v7 used element-wise `clamp`; V3 uses `clip_grad_norm_` — different operations at the same nominal 1.0 | P2 | source | CONFIRMED | N/A | Declare if comparing V3 to v7 results |

---

# EXECUTIVE SUMMARY

**TOTAL FILES AUDITED:** 136 (83 tracked + 53 untracked)
**TOTAL SOURCE FILES:** 31 · **CONFIG:** 17 · **TEST:** 13 · **CLOUD:** 22 ·
**HISTORICAL:** 4

**P0:** 3 · **P1:** 14 · **P2:** 14 · **P3:** 4 · **INFO:** 5

**CONFIRMED PERFORMANCE ISSUES:** D-01 (cache never persists), D-02 (patchify
no-op with up to 50 wasted full-volume reductions), D-03 (~1,415 cv2 calls/case),
D-04 (unused transforms), D-05 (blocking transfer), eval loader `num_workers=0`.
**LIKELY:** GPU-bound overall (single prior 100%-utilisation observation, **not
re-measured in this audit**).
**POSSIBLE:** wall-clock significance of every CPU-side item above — all are
currently overlapped with GPU compute and **none were measured**.

| Area | Status |
|---|---|
| **CORRECTNESS** | **PASS** — architecture, loss, NCA update, training-step order, checkpointing math and 40,656 params all verified correct |
| **REPRODUCIBILITY** | **FAIL** — R-01 worker seeds do not vary by epoch |
| **MEMORY** | **CONCERN** — VRAM fine *if* checkpointing is on (F-03 unresolved); ~20 GB host RAM at final eval |
| **DATA PIPELINE** | **CONCERN** — correct outputs, substantial wasted work |
| **TRAINING LOOP** | **PASS on order/math**, CONCERN on RNG + resume robustness |
| **EVALUATION** | **PASS on leakage discipline**, CONCERN on HD95 units + metric inconsistency |
| **CHECKPOINT** | **PASS on completeness/atomicity**, CONCERN on silent scheduler reset |
| **CLOUD** | **FAIL** — production path cannot load the split (CL-01); gate can launch the campaign (CL-02) |

**CONFIRMED UNUSED FILES:** `xfer2.sh`, `scripts/dataset_identity.py`
**POSSIBLY UNUSED:** `Agent_Multi_NCA.batch_step`, `Dataset_3D.preprocessing`,
`Agent_NCA.Pool`/`Persistence`, `DiceLoss`/`DiceCELoss`/`TverskyCELoss`,
`local_gpu_smoke_test.py` (V2-only)
**DUPLICATED CODE AREAS:** `v3_multilevel.yaml` ≡ `_ckpt.yaml`;
`v3_kaggle_5epoch` ≈ `v3_smoke_5epoch`; both `test_*.py` duplicate production
logic; `_clipped_batch_step` duplicates `Agent_Multi_NCA.batch_step`

## MOST IMPORTANT 5 ISSUES

1. **F-01 — the entire V3 architecture and canonical split are untracked.** One
   `git clean` loses the thesis model. Fix before anything else.
2. **CL-01 — the production container cannot load the master split.** The
   300-epoch run as currently configured will fail at startup; the gate passes
   because it uses a different mount set.
3. **CL-02 — `pretrain_gate.sh` can silently launch the full 300-epoch campaign**
   on a GPU with no confirmation, contradicting its own header.
4. **R-01/T-03 — augmentation RNG repeats every epoch**, so 300 epochs see far
   less augmentation diversity than intended. Effect size unmeasured.
5. **F-03/M-03 — which config is production is genuinely ambiguous**, and the two
   candidates differ precisely in the gradient-checkpointing setting the brief
   requires to be TRUE.

## TOP 5 SAFE OPTIMIZATION OPPORTUNITIES (Phase 2 candidates — none applied)

1. **`persistent_workers=True` + hoist the DataLoader out of the epoch loop.**
   Makes the existing cache actually work. No math change. *Caveat: changes
   worker lifetime, so `_worker_init` fires once — interacts with R-01; design
   them together.*
2. **Patchify short-circuit when `img.shape[:3] == size`.** Removes up to
   50 × 2.1M-element reductions per sample. Output-identical. *Caveat: consumes
   no RNG draws, shifting the augmentation stream — must be a deliberate
   decision.*
3. **Enable `memory.gradient_checkpointing` in the resolved production config**
   — already verified output/gradient/RNG-identical. The only material,
   methodology-neutral memory lever.
4. **`non_blocking=True` on the H2D transfer** (`pin_memory` is already on) and
   `zero_grad(set_to_none=True)`. Trivially safe, no numerical effect.
5. **Hoist the two unused torchio transforms out of `__getitem__`**, and stream
   or incrementally threshold `collect_probs` to cut ~20 GB host RAM. No
   numerical effect.

Every one of these must still answer the 10 questions in section 25 of the brief
before implementation, and items 1 and 2 change RNG streams.

## TOP 5 ITEMS THAT MUST NOT BE CHANGED

1. **`Model_BasicNCA3D.py`** — shared with V2. Any edit (transposes, clones,
   SE/GC blocks) breaks the V2-vs-V3 comparison.
2. **NCA steps 20/20/10 = 50, level resolutions 32/96/128, channels 24/24/16,
   `hidden=128`, fire_rate 0.6** — the architecture identity; 40,656 params.
3. **The loss recipe** — FocalTversky α=0.25 / β=0.75 / γ=1.33 / ce_weight=0.5
   plus the 0.1-weighted empty-region BCE. Verified correct.
4. **The canonical split** — `split/master_split.json`, sha256
   `d30d7195…`, subject-disjoint. Commit it; never regenerate.
5. **The frozen-test discipline** — val-only threshold tuning, test collected
   once, thresholds frozen. Verified sound; do not "optimize" this path.

---

## Verification that nothing was modified

`git status --porcelain src/ configs/ train.py` shows **no new modifications**
from this audit. All output is confined to `reports/audit/`.

**STOPPING HERE per section 31. Awaiting your review before Phase 2.**
