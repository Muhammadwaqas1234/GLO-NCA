# PHASE 1 — FROZEN BASELINE

Audit-only. No production file was modified. Recorded 2026-09-15.

## Repository state

| Item | Value |
|---|---|
| Repository root | `C:\Users\raiwa\Downloads\M3D-NCA-main` |
| Branch | `v3-multilevel` |
| HEAD commit | `f9e5501af397a4e8fa69c0320dc16ac57e521cd8` (2026-09-08) |
| Main branch | `main` |
| Tracked files | 83 |
| Untracked (project-relevant) | 53 |

**Critical state fact:** the entire V3 architecture is **untracked**.
`src/models/Model_GLO_NCA_V3.py`, `src/agents/Agent_GLO_NCA_V3.py`, all
`configs/v3_*.yaml`, and `split/master_split.json` are *not committed* at
HEAD. `git stash`, a bad `git clean -fdx`, or a fresh clone loses the thesis
model. See finding F-01.

`.gitignore` ends with `!split/master_split.json`, so the split is intended to
be tracked but currently is not (it was created after the last commit).

## Canonical split

| Field | Value |
|---|---|
| File | `split/master_split.json` |
| `split_version` | `GLO-NCA-V2-master-v1` |
| `split_sha256` | `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d` |
| `patient_id_hash` | `7e9ff2cf8042ff0071b804c441cddd31dd8979de2bb1886b4a9107d6c8e4cb47` |
| Seed | 42 |
| Grouping | subject-disjoint (base id = case id minus `-<timepoint>`) |
| Cases | train 898 / val 200 / test 198 (total 1296) |
| Subjects | 810 (train 567 / val 121 / test 122) |

Subject-disjoint grouping is correct and prevents same-patient timepoint leakage
across splits.

## Production V3 configuration

Production config: **`configs/v3_multilevel.yaml`**.
`configs/v3_multilevel_ckpt.yaml` is byte-identical except for a trailing
`memory.gradient_checkpointing: true` block.

| Setting | Value | Source |
|---|---|---|
| Model version | `v3` | `v3_multilevel.yaml` |
| Level 1 | res 32³, 24 ch, **20 steps**, k=7 | config |
| Level 2 | res 96³, 24 ch, **20 steps**, k=3 | config |
| Level 3 | res 128³, 16 ch, **10 steps**, k=3 | config |
| Total NCA steps | **50** | 20+20+10 |
| Fusion | `concat` (learnable 1×1×1) | config |
| SE / spatial GC | both `true` | config |
| hidden | 128 | config |
| dropout | 0.1 | config |
| fire_rate | 0.6 | config |
| Epochs | 300 | config |
| batch_size | 1 | config |
| patch_size | 128 | config |
| workers | 4 | config |
| augmentation | `light` | config |
| Optimizer | AdamW, lr 1.6e-3, wd 1e-4, betas (0.9,0.99) | config + `runner.py:120-125` |
| Scheduler | `CosineAnnealingLR`, T_max = epochs×ceil(898/1), eta_min 1e-5 | `runner.py:355-358` |
| EMA | enabled, decay 0.999 | config |
| Grad clip | enabled, max_norm 1.0 | config |
| Loss | `FocalTverskyCELoss(alpha=0.25, beta=0.75, gamma=1.33, ce_weight=0.5)` | `runner.py:360-363` |
| Empty-region | BCE × 0.1 | `runner.py:63-71` |
| Thresholds | tuned on validation only, grid 0.20–0.60 | `metrics_eval.py:81-99` |
| Best-epoch | smoothed val mean Dice, window 3 | config + `runner.py:452-453` |
| **gradient_checkpointing** | **`false`** in the production config | absent from `v3_multilevel.yaml` |

### Loss parameter verification

The prompt states expected `alpha=0.25, beta=0.75, gamma=1.33, ce_weight=0.5`.
Confirmed, but note these are **derived, not configured**: `runner.py:361` sets
`alpha = 1 - tversky_beta = 1 - 0.75 = 0.25`. `ce_weight=0.5` is
**hard-coded** at `runner.py:363` and cannot be configured. Matches spec.

## V3 parameter count — VERIFIED

**40,656 total parameters.** Confirmed by independent analytic recomputation
from the layer definitions (not by trusting a report or a cached number):

| Component | Parameters |
|---|---|
| Level 1 (C=24, k=7, SE+GC, hidden 128) | 18,861 |
| Level 2 (C=24, k=3, SE+GC, hidden 128) | 11,277 |
| Level 3 (C=16, k=3, SE+GC, hidden 128) | 7,811 |
| Cross-level projections (24→20, 24→12) | 800 |
| Fusion + segmentation head | 1,907 |
| **Total** | **40,656** |

Matches the expected figure exactly. `runner._model_info` reports this
*measured*, not hard-coded (`runner.py:198-199`), which is correct practice.

## Tensor contract — VERIFIED

- Model input: `(B, X, Y, Z, 4)` channels-last — `Model_GLO_NCA_V3.py:170-177`
- Model output: `(B, 3, X, Y, Z)` channels-first — `:211-212`
- Agent re-permutes to `(B, X, Y, Z, 3)` for the shared runner — `Agent_GLO_NCA_V3.py:81`

Contract matches the specification.

## V2 status

Frozen baseline, untouched. `configs/gcp_full.yaml` + `ablation_*.yaml`, driven
by `Agent_GLO_NCA` over two `BasicNCA3D` models (`runner.py:_build`). V3 shares
the dataset, Experiment, runner loop, loss, optimizer, scheduler, EMA and
evaluation; only model + agent differ. That sharing is real and verified, so the
V2↔V3 comparison is methodologically sound.

## Evaluation / test protocol

- Thresholds tuned on **validation pairs only** (`runner.py:520-521`).
- Test collected once into `test_pairs`, then scored at 0.5 and at the frozen
  tuned thresholds (`runner.py:522-524`). **No test-set threshold tuning.**
- HD95 reported in **voxels**, correctly labelled `HD95(vox)` / `hd95_vox`
  throughout (`runner.py:597`, `:610`, `:629`). Not mm — see E-03.
