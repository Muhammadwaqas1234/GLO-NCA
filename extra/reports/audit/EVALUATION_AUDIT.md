# EVALUATION / METRICS AUDIT

## 1. Test discipline — PASS (no leakage)

| Requirement | Verdict | Evidence |
|---|---|---|
| Train/val/test separation | PASS | subject-disjoint master split, sha256-verified on load |
| Thresholds tuned on validation ONLY | PASS | `runner.py:520-521` — `tune_thresholds(val_pairs)`; test never enters `tune_thresholds` |
| Test evaluated once (one forward pass) | PASS | `collect_probs(agent, ds, "test")` called exactly once `:522`; subsequent scorings reuse the cached arrays |
| No threshold tuning on test | PASS | `score(test_pairs, thresholds)` applies **frozen** val-derived thresholds `:524` |
| Eval un-augmented, deterministic | PASS | `patchify`/`_augment` gated on `state=="train"` (`Nii_Gz_Dataset_3D.py:183,191`) |
| Eval under no_grad | PASS | `metrics_eval.py:23` |

`test_pairs` is scored three times (`:523`, `:524`, `:564`) but all three reuse
the same cached probabilities — **no repeated inference**. Correct and efficient.

**Verdict: the frozen-test discipline is genuinely implemented.** This is a
strength of the codebase.

## 2. FINDING E-01 (P2, CONFIRMED) — validation is used twice

Validation drives **both** model selection (smoothed best-epoch, `runner.py:486`)
and threshold selection (`:521`). This is standard practice and not leakage, but
it means **validation metrics are optimistically biased** and must not be
presented in the thesis as a held-out estimate. Only the test numbers are
held-out. Ensure the write-up says so.

## 3. FINDING E-02 (P1, CONFIRMED) — HD95 is not comparable to published BraTS HD95

Two distinct issues in `Agent.py:56-77`:

**(a) Units are voxels on a resampled cube, not millimetres.**
`hd95_score` calls `distance_transform_edt` **without a `sampling=` argument**, so
distances are in voxel units. Worse, the volume has already been resized to a
128×128×128 **cube** by `rescale3d` from anisotropic, variably-sized BraTS data
(after a foreground crop whose extent differs per case). So one "voxel" means a
**different physical distance in every case and every axis**. Averaging such
values across cases is not a physically meaningful quantity.

The code is *honest* about the unit — labelled `HD95(vox)` / `hd95_vox`
throughout (`runner.py:597,610,629`) — so this is not deception. But published
BraTS HD95 figures are in **mm**; these numbers **cannot be compared to them**.

**(b) It measures distance-to-foreground, not distance-to-surface.**
`_surface_distances` (`Agent.py:48-53`) computes the EDT of the complement and
samples it at the other mask's voxels, i.e. distance to the nearest foreground
voxel rather than to the nearest **boundary** voxel. For the 95th percentile the
two largely coincide (extreme points are near boundaries), but it is not the
textbook HD95 definition.

**Recommendation (audit finding only):** either report it explicitly as
"HD95 in resampled-voxel units, not comparable to mm-based literature", or
compute HD95 on the original grid with `sampling=` set from the NIfTI affine.
**Do not silently relabel.** This is a thesis-methodology risk, not a code bug.

## 4. Dice and mIoU — PASS

- Dice `metrics_eval.py:41-43`: `2·|P∩T| / (|P|+|T|+1e-6)`. Standard. Note both
  empty → `0/1e-6 = 0`, i.e. a perfect empty prediction scores **0**, not 1.
  Conservative (penalises the model) and consistent across all regions and both
  arms of the comparison, so it does not bias V2-vs-V3. Worth a footnote since
  some BraTS protocols score empty-empty as 1 — this choice makes ET numbers
  look *worse* than those protocols. Severity **INFO**, but methodologically
  relevant to report.
- mIoU `Agent.py:29-44`: standard Jaccard; **returns 1.0 when both empty**
  (`:42-44`). **Inconsistent with Dice's 0.0 for the same case.** Confirmed
  inconsistency, severity **P2** — the two metrics disagree on the same input.
- `score` averages per-case, skipping NaN for HD95 only (`:48-51`). Correct.

## 5. Statistics — PASS

`score_per_case` (`:55-72`) uses identical metric definitions to `score`, and
`STATS.summarize_per_case(..., n_boot=2000, seed=cfg.seed)` (`runner.py:563`)
produces seeded bootstrap CIs from per-case values at the **frozen** thresholds.
Correct, and per-case export enables proper significance testing.

## 6. Performance note

Evaluation runs a DataLoader with **`num_workers=0`** (`metrics_eval.py:21`,
`Agent.py:401`) — default, single-process. Combined with the cache defeat
(D-01), every validation epoch re-loads and re-preprocesses all 200 val cases
**serially in the main process**, 300 times. This is a **CONFIRMED** inefficiency
and is more likely to be on the critical path than the training-side loader,
because there is no worker parallelism to hide it at all.
