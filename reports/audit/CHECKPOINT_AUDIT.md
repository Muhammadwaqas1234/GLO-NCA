# CHECKPOINT / RESUME AUDIT

## 1. Checkpoint contents — COMPLETE (PASS)

`checkpoint.build_checkpoint:13-33` stores every required element:

| Element | Present | Line |
|---|---|---|
| epoch | YES | `:22` |
| model state_dict(s) | YES | `:23` |
| optimizer state | YES | `:24` |
| scheduler state | YES | `:25` |
| EMA weights | YES | `:26` |
| best_score / best_epoch | YES | `:27-28` |
| history | YES | `:29` |
| config identity | YES (saved) | `:30` |
| RNG state (python/numpy/torch/cuda) | YES | `:31` |
| format version tag | YES `"glo-nca-v2-ckpt-1"` | `:32` |

Thresholds are **not** stored — correct, because they are derived at final
evaluation from validation and never used during training.

**This is a genuinely complete checkpoint.** Better than typical research code.

## 2. Write safety — PASS

`save_checkpoint:36-42` and `save_best_weights:45-55` both write to `path + ".tmp"`
then `os.replace(tmp, path)`. `os.replace` is atomic on POSIX and Windows, so a
crash mid-write cannot corrupt the existing checkpoint. **Correct.**

`best` and `last` are strictly separate files (`ws.best_ckpt` vs `ws.last_ckpt`),
plus periodic snapshots every `checkpoint_frequency=10` epochs (`runner.py:500-501`).
Best is saved only on improvement (`:486-491`) and stores **EMA weights** when EMA
is enabled (`:488`) — matching the eval path, which loads `bw["m"]` into the models
(`:515-517`). Consistent.

## 3. Logical resume test: N epochs -> checkpoint -> resume -> N+1

Traced:

1. Epoch index `ep = 4` (5th epoch) completes. `build_checkpoint(epoch=ep+1=5)`
   (`runner.py:494`), saved to `last.pth` (`:499`).
2. `python train.py --resume <dir>` -> `Workspace.open_existing`, config read from
   the **workspace copy** (`train.py:61-65`) — so the resumed run uses the exact
   config the run started with. Good.
3. `restore_into` loads model/optimizer/scheduler (`:394-395`); EMA (`:396-397`);
   history/best/best_epoch (`:398-400`); `start_epoch = 5` (`:401`); RNG restored
   (`:402`).
4. Loop `range(5, 300)` (`:431`) -> first iteration logs `ep+1 = 6`.

**No off-by-one. No duplicated epoch. No skipped epoch. PASS.**

## 4. FINDING C-01 (P1, CONFIRMED) — silent scheduler reset

`checkpoint.py:76-79` — bare `except: pass` around `scheduler.load_state_dict`,
with no logging. On failure the run silently continues with a fresh
`CosineAnnealingLR` at step 0, restarting the LR at 1.6e-3 mid-campaign.
Optimizer restore (`:74-75`) is unguarded and would raise loudly — the
inconsistency is the defect. Detectable post-hoc only by inspecting the `lr`
column of `metrics/train.csv`.

Full detail in TRAINING_LOOP_AUDIT §6 (T-04).

## 5. FINDING C-02 (P2, CONFIRMED) — EMA restore failure is silent

`runner.py:396-397`: `if ema is not None and ck.get("ema"):`. If the checkpoint
has no `ema` key (or it is falsy), the freshly-initialised EMA — a copy of the
just-restored weights — is silently kept and no warning is logged. EMA then
re-converges over ~1/(1-0.999) ≈ 1000 steps, so the damage is bounded, but it is
undetectable from the logs.

## 6. FINDING C-03 (P2, CONFIRMED) — resume never verifies config identity

`ck["config"]` is saved but never compared on resume. Resuming with a changed
`epochs` recomputes `T_max` (`runner.py:353-354`) while restoring the old
scheduler step count, producing a cosine curve matching neither config. A model
shape change would raise; hyperparameter drift passes silently.
See TRAINING_LOOP_AUDIT §7 (T-05).

## 7. FINDING C-04 (P2, CONFIRMED) — `weights_only=False` on load

`checkpoint.py:58-62` loads with `weights_only=False`, which permits arbitrary
pickle execution. The justification given in the comment is **valid** (the
checkpoints embed numpy/torch RNG state objects, which `weights_only=True`
rejects) and the files are self-produced. Risk is acceptable **provided
checkpoints are only ever loaded from your own GCS bucket**. Becomes a real
issue only if a checkpoint from an untrusted source is ever loaded. Noted, not
a defect in the current workflow.

## 8. Resume-path gap

`train.py:60-65`: on `--resume`, if the workspace config is missing it silently
falls back to `args.config` — which defaults to `configs/gcp_full.yaml`, i.e. the
**V2** config. Resuming a V3 experiment whose `config/config.yaml` went missing
would attempt to rebuild a V2 model and fail on state_dict mismatch (loudly), so
this is not silent corruption — but the fallback default is wrong-version and
would produce a confusing error. Severity **P3**.
