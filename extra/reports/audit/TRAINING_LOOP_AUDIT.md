# TRAINING LOOP AUDIT

`src/experiment/runner.py` — `_clipped_batch_step:56-81` and the epoch loop `:431-504`.

## 1. One training iteration, traced

```
prepare_data(data)                    :57   -> .to(device), blocking
get_outputs(data)                     :58   -> V3 forward, (B,X,Y,Z,3) logits
for opt in agent.optimizer: zero_grad():59-61
per-region loss accumulation          :64-72
loss.backward()                       :74
clip_grad_norm_(net.parameters(),1.0) :76-77
opt.step(); sch.step()                :78-80
ema_update()                          :439  (in the epoch loop, AFTER the step)
```

**Order is CORRECT:** forward → loss → backward → clip → optimizer.step →
scheduler.step → EMA. Grad clipping happens after `backward()` and before
`step()`, which is the only correct placement. EMA updates after the optimizer
step, so it tracks post-update weights — correct.

## 2. Verified-correct (PASS)

| Check | Verdict | Evidence |
|---|---|---|
| Duplicate forward | NONE | exactly one `get_outputs` per step `:58` |
| Duplicate backward | NONE | one `loss.backward()` `:74` |
| Scheduler stepping | PASS (per-step by design) | `sch.step()` `:80` is per *batch*; `T_max = epochs × ceil(898/1)` `:353-354` matches per-batch stepping. Consistent — cosine completes exactly at the end of training |
| Grad clip value | PASS | `max_norm=1.0` from config `:379-380` |
| EMA correctness | PASS | float-only params updated `:376-377`; buffers copied at init `:368-369` |
| Loss accumulation | PASS | summed across 3 regions, single backward |
| Empty-region handling | PASS | `:66-71` — when a region is absent, a 0.1-weighted BCE replaces Tversky, preventing Tversky collapse on the all-zero target. Documented and deliberate |
| Memory leak between iterations | NONE FOUND | `loss_ret[m] = loss_loc.item()` `:72` detaches to a Python float; `losses` accumulates floats, not tensors `:441`. History stores floats. No graph retained across iterations |
| Best-epoch selection | PASS | on **smoothed** val mean, window 3 `:452-453,486` |
| Gradient accumulation | N/A | batch_size 1, no accumulation configured or implemented |
| autocast / GradScaler | ABSENT | no mixed precision anywhere. Training is pure fp32 |

## 3. FINDING T-01 (P2, CONFIRMED) — `zero_grad()` without `set_to_none=True`

`runner.py:60-61` calls `opt.zero_grad()` with default `set_to_none=False` on
PyTorch < 2.0 semantics. Modern PyTorch defaults `set_to_none=True` for
`Optimizer.zero_grad()`, so actual behaviour depends on the installed version.
With `set_to_none=False` the gradient buffers are filled with zeros each step
instead of freed — a small extra memory write over ~40k params (negligible here,
0.16 MB) and one extra kernel. Very low impact given the tiny parameter count.
**Not worth changing for memory; listed for completeness.**

## 4. FINDING T-02 (P1, CONFIRMED) — DataLoader recreated every epoch

`runner.py:434-437` builds a new `DataLoader` inside the epoch loop. Two
consequences:

1. **Worker process churn.** 4 workers are spawned and torn down 300 times. On
   Windows/spawn this is expensive; on Linux/fork it is moderate. Combined with
   no `persistent_workers`, this is the mechanism behind the cache defeat
   (DATA_PIPELINE_AUDIT finding D-01).
2. **Worker RNG restarts identically every epoch** — see T-03.

## 5. FINDING T-03 (P1, CONFIRMED) — augmentation RNG repeats every epoch

`runner.py:424-428`:

```python
def _worker_init(worker_id):
    s = (cfg.seed + worker_id) % (2 ** 32)
    np.random.seed(s)
    import random as _r
    _r.seed(s)
```

The seed depends **only** on `cfg.seed` (42, constant) and `worker_id` (0–3).
It does not depend on the epoch. Because a fresh loader spawns fresh workers each
epoch (T-02), `_worker_init` runs again every epoch with the **same four seeds**.

Both `random` and `np.random` are reseeded, and both are what the augmentation
and patch sampling actually use:
- `_augment` — `random.random/randint/uniform` (`Nii_Gz_Dataset_3D.py:266,270,
  278,289,290,294,298,305`) and `np.random.normal` (`:302`), `np.random.rand`
  (`:319-321`).
- `patchify_multimodal` — `random.uniform/randint` (`:344,354-356`).

**Consequence: every epoch applies the identical sequence of augmentations in the
identical order to the identically-shuffled-per-worker stream.** Over 300 epochs
the model sees far fewer distinct augmented views than intended — augmentation
diversity collapses toward a single fixed augmented copy of the dataset per
worker slot.

Caveats that limit but do not eliminate the effect:
- `shuffle=True` order is drawn in the **parent** process from the torch RNG,
  which does advance across epochs. So *which case* a given worker handles varies
  per epoch, and the pairing of (case, augmentation draw) changes.
- Therefore it is not "the same augmented dataset every epoch" exactly, but the
  *pool of augmentation parameter draws* is identical each epoch and is consumed
  in the same order.

Classification: **CONFIRMED defect in the seeding logic.** Effect size on final
Dice is **NOT measured** — do not claim a magnitude. This is the single most
important reproducibility/methodology finding in the audit.

The fix (Phase 2) is one line — mix the epoch into the seed — but it **changes
the RNG stream**, so it cannot be applied silently to a campaign in progress.

## 6. FINDING T-04 (P1, CONFIRMED) — resume can silently reset the LR schedule

`checkpoint.py:76-79`:

```python
for s, sd in zip(schedulers, ckpt.get("scheduler", [])):
    try:
        s.load_state_dict(sd)
    except Exception:
        pass  # scheduler shape can shift if epochs changed; keep going
```

A bare `except: pass` with no logging. If scheduler restore fails for any reason,
training continues with a **freshly constructed `CosineAnnealingLR` at step 0** —
i.e. the LR jumps back to 1.6e-3 in the middle of a 300-epoch campaign and the
cosine restarts. There is no warning, no log line and no non-zero exit. A silently
restarted LR schedule mid-campaign would corrupt the thesis run and be very hard
to detect after the fact (though `train.csv` does record `lr` per epoch, so it is
*detectable* post-hoc by inspecting that column).

Contrast: optimizer restore (`:74-75`) is **not** wrapped and would raise loudly.
The inconsistency is the defect.

Related: EMA restore (`runner.py:396-397`) requires `ck.get("ema")` to be truthy;
otherwise the freshly-initialised EMA (a copy of the just-restored weights) is
silently kept, also unlogged. Less severe — EMA re-converges — but same pattern.

## 7. FINDING T-05 (P2, CONFIRMED) — resume never validates config identity

`checkpoint.py:30` saves `config`, but `runner.py:392-406` restores
epoch/best/history/EMA/RNG and **never compares** `ck["config"]` to the current
config. Resuming with a different `epochs` value silently recomputes `T_max`
(`:353-354`) while restoring the old scheduler step count — producing a cosine
curve that matches neither config. A model-shape change would raise, but
hyperparameter drift passes silently.

## 8. CPU vs GPU bottleneck — evidence, not assumption

**Direct measurement is not possible in this environment** (no GPU, no torch).
The only evidence available is the prior GCP observation recorded in
`reports/validation/GLO_NCA_V3_PERFORMANCE_AUDIT.md`: **GPU utilisation was
observed at 100%** during an epoch. If that observation is sound, the GPU is
**not** starved and the CPU pipeline is not currently the critical path.

Workload that supports GPU-bound: 50 NCA update executions per case over
39.3M voxel-steps/case; ×898 cases = ~35.2 billion voxel-steps/epoch, doubled to
~70.5 billion if checkpointing is on.

Classification: **LIKELY GPU-bound**, based on one prior observation, not on a
measurement taken during this audit. Requires re-measurement before any
CPU-side optimization is justified on wall-clock grounds. Do not treat as fact.
