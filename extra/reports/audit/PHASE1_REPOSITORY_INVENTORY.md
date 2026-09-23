# PHASE 1 — REPOSITORY INVENTORY

> **NAMING NOTE (added after this report was written).**
> `src/models/Model_BasicNCA3D.py` was renamed to
> `src/models/Model_GLO_NCA_Cell.py`, and the class `BasicNCA3D` to
> `GLO_NCA_Cell`. The rename was cosmetic: no behaviour, tensor shape or
> parameter changed, and production identity stayed 29,337 / 75 / 29,412,
> verified from the constructed model before and after. Any file path, class
> name or line number cited below refers to that file under its former name;
> the evidence recorded here still stands.

Classification: A production runtime · B production config · C test ·
D diagnostic/profiling · E cloud/deployment · F documentation · G historical ·
H generated · I apparently unused · J needs tracing.

**Nothing was deleted.** "Unused" is a classification, proven by reference grep.

## Source (`src/`) — 30 files

| File | Class | Used by | Note |
|---|---|---|---|
| `train.py` | A | entry point | CLI wrapper; `--config` defaults to **V2** |
| `src/experiment/runner.py` | A | `train.py` | 704 lines; the orchestrator |
| `src/experiment/config.py` | A | runner | thin validation (F-07) |
| `src/experiment/checkpoint.py` | A | runner | atomic writes; silent scheduler except (C-01) |
| `src/experiment/datasource.py` | A | runner, dataset | split load + sha256 verify |
| `src/experiment/dataset_validation.py` | A | runner | hard gate |
| `src/experiment/metrics_eval.py` | A | runner | Dice/IoU/HD95, threshold tuning |
| `src/experiment/reproducibility.py` | A | runner | seeds + RNG state |
| `src/experiment/statistics.py` | A | runner | bootstrap CIs |
| `src/experiment/workspace.py` | A | runner, train.py | experiment dirs |
| `src/experiment/environment.py` | A | runner | env/git capture |
| `src/experiment/logutil.py` | A | runner | CSV/TB/logger |
| `src/experiment/graphs.py` | A | runner | figures |
| `src/experiment/diagnostics.py` | A | runner | verdict report |
| `src/models/Model_GLO_NCA_V3.py` | A | runner | **UNTRACKED** (F-01) |
| `src/models/Model_BasicNCA3D.py` | A | V3 + V2 | shared — changing it changes V2 |
| `src/agents/Agent_GLO_NCA_V3.py` | A | runner | **UNTRACKED** (F-01) |
| `src/agents/Agent_NCA.py` | A | V3 base | `make_seed`/`Pool` dead on V3 path |
| `src/agents/Agent.py` | A | base + `iou_score`/`hd95_score` | metrics live here |
| `src/losses/LossFunctions.py` | A | runner | only `FocalTverskyCELoss` used |
| `src/datasets/Nii_Gz_Dataset_3D.py` | A | runner | the data pipeline |
| `src/datasets/Dataset_3D.py` | A (partly I) | base | `preprocessing`/`__getitem__` dead |
| `src/datasets/Dataset_Base.py` | A | base | |
| `src/datasets/Data_Instance.py` | A (ineffective) | dataset | cache defeated (D-01) |
| `src/utils/Experiment.py` | A | runner | legacy glue; `get_from_config` silent-None |
| `src/utils/helper.py` | A | Experiment, Agent_NCA | |
| `src/agents/Agent_GLO_NCA.py` | A (V2 only) | runner `_build` | reads 2 never-defined keys |
| `src/agents/Agent_Multi_NCA.py` | A (V2) / **I** | `Agent_GLO_NCA` | **`batch_step` dead and divergent** from the live path (no grad clip, no empty-region BCE) |
| `src/*/__init__.py` (5) | A | packaging | |

**Unused within otherwise-live files:** `DiceLoss`, `DiceCELoss`, `TverskyCELoss`
(`LossFunctions.py`) — only `FocalTverskyCELoss` is used. `Agent_NCA.Pool` /
`Persistence` machinery — unreachable (no config sets `Persistence`).

## Configs — 17 files: class **B**. See CONFIG_AUDIT.md.

## Scripts (`scripts/`) — 14 files

| File | Class | Refs | Note |
|---|---|---|---|
| `validate_dataset.py` | A | 4 | production gate |
| `create_master_split.py` | A | 11 | refuses overwrite without `--force` |
| `check_split.py` | A/C | 13 | strong subject-leakage check |
| `preflight_gcp.py` | A | 3 | **pulls V2-only checks into a V3 gate (P1)** |
| `make_ablation_table.py`, `make_thesis_tables.py`, `make_figures.py` | A | 2 each | read saved outputs only; no fabricated numbers |
| `gpu_memory_gate_v3.py` | C/D | 14 | most-referenced; excellent discipline |
| `validate_v3_local.py`, `local_gpu_smoke_test.py`, `profile_v3_training.py` | C/D | 1–4 | `local_gpu_smoke_test` is **V2**, not V3 |
| `test_patchify_equivalence.py`, `test_preflight_logic.py` | C | 1–2 | **test reimplementations (TS-02)** |
| `validate_kaggle_sample.py`, `build_sample_manifest.py` | C/D | 2 each | |
| `verify_phase3_ready.py` | C/D | 2 | **asserts branch `v2` → always FAIL here** |
| `dataset_identity.py` | **I** | **0** | zero references; also cannot handle nested cohorts |

## Cloud (`cloud/`) — 22 files: class **E**. See CLOUD_AUDIT.md.

Notable: `pretrain_gate.sh` = **DANGEROUS** (CL-02); `setup_gcp.sh` = OBSOLETE
(branch `v2`); `glo-nca-training.service` = exemplary.

## Root / misc

| File | Class | Note |
|---|---|---|
| `Dockerfile`, `.dockerignore`, `requirements-docker.txt` | B/E | **missing `split/` (CL-01)** |
| `xfer2.sh` | **I** | **zero references**; no `set -e`; writes to production GCS prefix unvalidated |
| `drive_dl.py` | D | 1 ref (`xfer2.sh:11`); hard-coded public Drive ID (not a secret) |
| `README.md`, `CODE_GUIDE.md`, `CITATION.cff`, `LICENSE`, `docs/**` | F | `cloud/README.md` contradicts `README.md` on production config |
| `archive/kaggle_experiment_history/kaggle_v4..v7.py` | **G** | see below |
| `reports/validation/**` | H/F | prior evidence; retained |
| `split/master_split.json` | B | **UNTRACKED** (F-01) |
| `notebooks/**` | D | |

## Archive — verified isolation (PASS)

`grep -rn "kaggle_v[4-7]"` over all `.py`/`.sh`/`.yaml`/`.md`/`.service` returns
**documentation references only**. `grep -rn "archive\."` over `.py` returns
**zero imports**. `Dockerfile` never copies `archive/`.

**Confirmed: no archive/Kaggle code is imported by production, and no old
settings leaked into V3.** V3 faithfully inherits the v7 hyperparameter recipe
(loss, EMA, clip magnitude, LR schedule, fire rate, dropout, seed 42) and changes
only the architecture (2-level 32/64³ → 3-level 32/96/128³), epochs 150→300, and
adds `weight_decay: 1e-4`. Scientific lineage intact.

**One item worth your attention (P2, POSSIBLE):** kaggle v7 clipped gradients via
per-parameter hooks doing **element-wise `torch.clamp`** (`kaggle_v7.py:270-274`),
whereas the V3 runner uses **`clip_grad_norm_`** (`runner.py:77`). These are
mathematically different operations at the same nominal value of 1.0. Not a
defect — but if the thesis compares V3 against v7 results, the difference in
clipping semantics should be declared.

## Totals

| Category | Count |
|---|---|
| Tracked files | 83 |
| Untracked project files | 53 |
| **Total audited** | **136** |
| Source (`.py` under `src/` + `train.py`) | 31 |
| Configs (`.yaml`) | 17 |
| Scripts (`scripts/*.py`) | 14 |
| Cloud files | 22 |
| Historical (archive) | 4 |
| Docs/reports | ~40 |
| **Confirmed unused** | **2** (`xfer2.sh`, `scripts/dataset_identity.py`) |
