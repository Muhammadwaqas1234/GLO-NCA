# CONFIGURATION AUDIT

17 configs. `configs/` tracked: `ablation.yaml`, `ablation_baseline.yaml`,
`ablation_full.yaml`, `ablation_se.yaml`, `ablation_spatial.yaml`,
`gcp_full.yaml`, `smoke_test.yaml`. Untracked: `smoke_test_v3.yaml`,
`v3_ablation_A/B/C.yaml`, `v3_kaggle_5epoch.yaml`, `v3_multilevel.yaml`,
`v3_multilevel_ckpt.yaml`, `v3_smoke_1/3/5epoch.yaml`.

## 1. FINDING F-03 (P1, CONFIRMED) — the production config is ambiguous

The two candidates are **byte-identical except for a trailing `memory:` block**,
and the documentation contradicts itself:

| Source | Says production is |
|---|---|
| `README.md:19` | `configs/v3_multilevel_ckpt.yaml` (300 epochs) |
| `README.md:101` | `configs/v3_multilevel_ckpt.yaml` |
| `docs/thesis/FINAL_TRAINING_PROTOCOL.md:4,61-63` | `configs/v3_multilevel_ckpt.yaml` |
| **`cloud/README.md:171,178`** | **`configs/v3_multilevel.yaml`** |
| `v3_multilevel_ckpt.yaml` (trailing comment) | "**Not the production config**" |

So `v3_multilevel_ckpt.yaml` is named as production by three documents while
**describing itself as not production**, and `cloud/README.md` names the other
one. The only functional difference is `memory.gradient_checkpointing`, which
is exactly the setting the audit brief says must be TRUE.

**This must be resolved before any run.** Launching the wrong one changes peak
VRAM by roughly 8× and wall-clock by roughly 2×. Given two near-identical
filenames, mis-selection is likely.

## 2. Config keys — read, dead, and silent

**Read by code:** `experiment.{name,seed}`, `dataset.{root,number_of_patients,
modalities,split_seed}`, `data.split_file`, `model.{version,level1/2/3.*,
feature_fusion.type,fire_rate,use_attention,use_spatial,hidden,dropout,channel_n,
steps}`, `training.{epochs,batch_size,patch_size,augmentation,workers}`,
`optimizer.{learning_rate,minimum_learning_rate,weight_decay}`,
`loss.{tversky_beta,focal_gamma}`, `ema.{enabled,decay}`,
`gradient.{clipping_enabled,max_norm}`, `evaluation.{tune_thresholds,
smoothing_window}`, `logging.{tensorboard,checkpoint_frequency}`,
`memory.gradient_checkpointing`.

**FINDING F-04 (P2, CONFIRMED) — defined in YAML, never read:**
- `evaluation.threshold: 0.5` — present in **all 17 configs**, read **nowhere**.
  The 0.5 baseline is hard-coded at `runner.py:520,522`. A pure no-op key that
  looks meaningful.
- `model.name` — only `experiment.name` drives the workspace (`train.py:69`).

**FINDING F-02 (P1, CONFIRMED) — read by code, absent from every config
(invisible methodology):**

| Value | Hard-coded | Thesis-relevant? |
|---|---|---|
| `ce_weight = 0.5` | `runner.py:363` | **YES** — a loss hyperparameter |
| `alpha = 1 - tversky_beta` | `runner.py:361` | derived, not settable |
| `empty_weight = 0.1` | `runner.py:63` | **YES** — empty-region loss weight |
| `priotize_masks = 0.7` | `runner.py:137` | **YES** — patch sampling bias |
| `prioritize_region = 2` (ET) | `runner.py:137` | **YES** — ET-aware sampling |
| `betas = (0.9, 0.99)` | `runner.py:123` | YES |
| optimizer `"adamw"` | `runner.py:120` | YES |
| threshold grid 0.20–0.60 | `metrics_eval.py:85` | YES |
| `foreground_crop/nonzero_norm/patchify = True` | `runner.py:135-137` | YES |

These are stable, deliberate choices — but they are **not in the saved per-run
config record**, so `config_resolved.json` does not fully describe the
methodology. For a thesis this is a reproducibility-documentation gap.

## 3. FINDING F-05 (P1, CONFIRMED) — a missing `data:` section silently re-splits

`runner.py:293`:
```python
split_file = cfg.get("data", "split_file") if cfg.section("data") else None
```
If the `data:` section or key is absent, control falls to `:312-316`, which
builds a **fresh seeded split** and trains on it. `config.py:65-68` does not
include `data` in `_REQUIRED_SECTIONS`, so validation does not catch it.

The comment at `ablation_baseline.yaml:25-26` claims "Missing file -> hard FAIL
(no silent regeneration)". That guarantee holds only for a *configured-but-absent*
file (`datasource.py:258-263` raises) — **not** for an omitted key.
`configs/ablation.yaml` has **no `data:` section** and would hit this path.

Mitigation that exists: `split_meta["source"] = "seeded"` is recorded in the
manifest, so it is **detectable post-hoc** — but not prevented.

## 4. FINDING F-06 (P2, CONFIRMED) — ablation headers contradict their own content

`ablation_full.yaml:2-10`, `ablation_se.yaml:2-10` and `ablation_spatial.yaml:2-10`
all carry the **copy-pasted header of `ablation_baseline`** ("ABLATION A0: plain
NCA baseline… `use_attention: false, use_spatial: false`"), while their actual
`model:` blocks set different flags:

| File | Header claims | Actually sets (`:34-35`) |
|---|---|---|
| `ablation_full.yaml` | A0 baseline, both false | `use_attention: true, use_spatial: true` |
| `ablation_se.yaml` | A0 baseline, both false | `true / false` |
| `ablation_spatial.yaml` | A0 baseline, both false | `false / true` |

**A methodology chapter written from these comments would be wrong.** The code
behaviour is correct; only the documentation lies. High risk for a thesis.

## 5. FINDING F-07 (P2, CONFIRMED) — config validation is very thin

`config.py:65-99` validates only that 10 section *names* exist, that
`patch_size` is in {64,96,128} (V2) or >0 (V3), and that `augmentation` is in
{none,light,heavy}. **No key is type- or range-checked, and no unknown key is
rejected.** A typo (`epoch:` for `epochs:`) passes validation and then either
crashes deep in the run or silently uses a default. Combined with
`Experiment.get_from_config` returning `None` for any unknown key
(`Experiment.py:186-193`), typos are structurally undetectable.

## 6. Other config findings

- **P2 — `configs/ablation.yaml`** claims "gcp_full defaults" (`:5`) but uses
  `epochs: 150` / `augmentation: heavy` vs `gcp_full`'s 200 / light. Stale, and
  it lacks `data:` (see F-05).
- **P3 — duplicate 5-epoch configs.** `v3_kaggle_5epoch.yaml` and
  `v3_smoke_5epoch.yaml` are near-identical full production configs differing
  only in `experiment.name`. `v3_kaggle_5epoch.yaml` also carries the full
  *production* header ("thesis training config") with `epochs: 5` — easy to
  mistake for the real thing.
- **P2 — `gcp.env` image-family drift.** Live `gcp.env:17` uses
  `common-cu129-ubuntu-2204-nvidia-580`; `gcp.env.example:39` documents
  `common-cu121-debian-11`; the image is CUDA 12.1 (`Dockerfile:21,41`). Works
  (driver is backward-compatible) but the example is misleading.
- **P2 — `Dockerfile:65`** `CMD ["--config", "configs/gcp_full.yaml"]` defaults
  to the **V2** config, and `Dockerfile:11-13` documents an `EPOCHS` env var that
  nothing reads (`train.py:40-48`).
- **P3 — `train.py:40`** `--config` defaults to `configs/gcp_full.yaml` (V2). A
  bare `python train.py` silently runs V2.

## 7. Which configs are actually used

| Purpose | Config |
|---|---|
| **V3 production** | `v3_multilevel_ckpt.yaml` **or** `v3_multilevel.yaml` — **UNRESOLVED (F-03)** |
| V3 smoke | `v3_smoke_1/3/5epoch.yaml`, `smoke_test_v3.yaml`, `v3_kaggle_5epoch.yaml` |
| V3 ablation | `v3_ablation_A/B/C.yaml` |
| V2 frozen baseline | `gcp_full.yaml`, `ablation_*.yaml` |
| V2 smoke | `smoke_test.yaml` |
| Stale | `ablation.yaml` (no `data:`, contradicts its own comment) |
