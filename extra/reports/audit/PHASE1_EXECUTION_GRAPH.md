# PHASE 1 — EXECUTION GRAPH (V3 production path)

> **NAMING NOTE (added after this report was written).**
> `src/models/Model_BasicNCA3D.py` was renamed to
> `src/models/Model_GLO_NCA_Cell.py`, and the class `BasicNCA3D` to
> `GLO_NCA_Cell`. The rename was cosmetic: no behaviour, tensor shape or
> parameter changed, and production identity stayed 29,337 / 75 / 29,412,
> verified from the constructed model before and after. Any file path, class
> name or line number cited below refers to that file under its former name;
> the evidence recorded here still stands.

Built from an exhaustive import grep, then traced by reading each call site.

## 1. Production execution path

```
train.py:main
 ├─ config.load_config(configs/v3_multilevel*.yaml)     -> Config
 ├─ Workspace.create(./experiments, cfg.name)           -> experiment dir
 └─ runner.run(cfg, ws, resume, device_str)
     ├─ logutil.get_logger / environment.write_environment / repro.describe
     ├─ repro.set_all_seeds(42)
     ├─ datasource.resolve_data_root
     ├─ dataset_validation.validate_dataset             [HARD GATE]
     ├─ _build_dispatch -> _is_v3(cfg)==True -> _build_v3
     │   ├─ Dataset_NiiGz_3D_BraTS()                    (Dataset_3D -> Dataset_Base -> Data_Container)
     │   ├─ Model_GLO_NCA_V3.build_v3_from_config
     │   │   └─ GLO_NCA_V3_MultiLevel
     │   │       ├─ 3 × BasicNCA3D (SEBlock3D + GCSpatialBlock3D)
     │   │       ├─ 2 × FeatureProjection (cross-level)
     │   │       ├─ 3 × FeatureProjection (level_to_fine) + fuse Conv3d
     │   │       └─ seg_head Conv3d(16->3)
     │   ├─ Agent_GLO_NCA_V3(ca)                        (-> Agent_NCA -> BaseAgent)
     │   └─ Experiment(config, ds, ca, agent)           -> agent.set_exp -> initialize
     │       └─ BaseAgent.initialize -> AdamW + ExponentialLR  [scheduler DISCARDED below]
     ├─ datasource.load_master_split                    [sha256-verified]
     ├─ agent.scheduler = [CosineAnnealingLR(...)]      [REPLACES the ExponentialLR]
     ├─ FocalTverskyCELoss(0.25, 0.75, 1.33, ce_weight=0.5)
     ├─ EMA init (decay 0.999)
     ├─ [resume] checkpoint.load_checkpoint -> restore_into -> repro.restore_rng_state
     └─ epoch loop (300)
         ├─ DataLoader(ds, shuffle, bs=1, workers=4, pin_memory)   [RECREATED EACH EPOCH]
         ├─ _clipped_batch_step -> prepare_data -> get_outputs -> backward -> clip -> step
         ├─ ema_update()
         ├─ metrics_eval.evaluate(agent, ds, "val")     [workers=0]
         ├─ CSVLogger / TensorBoard / status
         └─ checkpoint.save_checkpoint(last) [+ periodic every 10]
     ── after training ──
         ├─ load best.pth -> load_state_dict
         ├─ ME.collect_probs(val) -> ME.tune_thresholds     [VAL ONLY]
         ├─ ME.collect_probs(test) -> ME.score × 2          [TEST ONCE]
         ├─ diagnostics.diagnose
         ├─ statistics.summarize_per_case (bootstrap, seeded)
         ├─ graphs.generate
         └─ ws.write_manifest / write_status("completed")
```

## 2. CLI -> config -> runtime value flow

| CLI arg | Consumed at | Effect |
|---|---|---|
| `--config` | `train.py:68` | selects the YAML. **Default is `configs/gcp_full.yaml` (V2)** |
| `--resume` | `train.py:60-66` | reloads workspace config; `--config` then ignored unless the workspace copy is missing |
| `--output` | `train.py:70` | experiments base dir |
| `--experiment` | `train.py:69` | overrides `cfg.name` |
| `--device` | `runner.py:254-256` | forces device, else auto-cuda |

All five CLI arguments are consumed. **No ignored CLI arguments.**

## 3. Hard-coded values that override or bypass config

Documented in full in CONFIG_AUDIT.md. The methodology-relevant ones:

| Value | Hard-coded at | Config key? |
|---|---|---|
| `ce_weight = 0.5` | `runner.py:363` | **none** |
| `alpha = 1 - tversky_beta` | `runner.py:361` | derived, not settable |
| `empty_weight = 0.1` | `runner.py:63` | **none** |
| `priotize_masks = 0.7` | `runner.py:137` | **none** |
| `prioritize_region = 2` (ET) | `runner.py:137` | **none** |
| `betas = (0.9, 0.99)` | `runner.py:123` | **none** |
| optimizer `"adamw"` | `runner.py:120` | **none** |
| `foreground_crop/nonzero_norm/patchify = True` | `runner.py:135-137` | **none** |
| threshold grid 0.20–0.60 | `metrics_eval.py:85` | **none** |
| bootstrap `n_boot=2000` | `runner.py:563` | **none** |

These are all stable methodology choices, but several (`ce_weight`,
`prioritize_region`, `empty_weight`) are **thesis-relevant hyperparameters that do
not appear in the saved per-run config record**. See CONFIG_AUDIT F-02.

## 4. Dead branches / discarded objects on the V3 path

| Item | Status | Evidence |
|---|---|---|
| `ExponentialLR` built in `BaseAgent.initialize` | **built then discarded** | `Agent.py:103` -> replaced `runner.py:355-358`. Documented as intentional (`Agent_GLO_NCA_V3.py:44-49`) |
| `Agent_NCA.make_seed` | **not called by V3** | V3 seeds internally; documented `Agent_GLO_NCA_V3.py:25-28` |
| `Agent_NCA.Pool` / `Persistence` | **unreachable** | no config sets `Persistence`; `get_from_config` returns None |
| `Agent_Multi_NCA.batch_step` | **dead** | runner uses its own `_clipped_batch_step`. **Differs from the live path** (no grad clip, no empty-region BCE) — a maintenance trap |
| `Dataset_3D.preprocessing` / `__getitem__` | **dead** | fully overridden by `Nii_Gz_Dataset_3D` |
| `BasicNCA3D` import in runner | used by V2 `_build` only | `runner.py:25` |
| `channel_n=16`, `inference_steps=10` in V3 flat config | **inert placeholders** | `runner.py:128` — never read by the V3 agent, but ARE written into the Experiment's `config.dt` artifact, where they could mislead a reader |
| torchio transforms in `__getitem__` | **constructed, never used** (nonzero_norm=True) | `Nii_Gz_Dataset_3D.py:142-143` |

**No V2 code is accidentally *used* by V3**, and **no Kaggle/archive code is
imported by production** (verified: `archive/` appears in no import statement).
No profiling code is imported by production.

## 5. Silent fallbacks that could change methodology

| Fallback | Risk | Evidence |
|---|---|---|
| Missing `data:` section -> **seeded synthetic split**, training proceeds | **P1** | `runner.py:293` + `:312-316`. The "hard FAIL" guarantee only covers a *configured-but-absent* file |
| Scheduler restore failure -> silent fresh cosine | **P1** | `checkpoint.py:76-79` |
| `--resume` with missing workspace config -> falls back to `--config` default = **V2** `gcp_full.yaml` | P3 (fails loudly downstream) | `train.py:63-64` |
| `Experiment.get_from_config` returns `None` for any unknown key | P2 — makes typos undetectable | `Experiment.py:186-193` |
