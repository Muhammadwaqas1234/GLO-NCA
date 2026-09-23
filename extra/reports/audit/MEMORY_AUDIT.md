# GPU MEMORY AUDIT

> **NAMING NOTE (added after this report was written).**
> `src/models/Model_BasicNCA3D.py` was renamed to
> `src/models/Model_GLO_NCA_Cell.py`, and the class `BasicNCA3D` to
> `GLO_NCA_Cell`. The rename was cosmetic: no behaviour, tensor shape or
> parameter changed, and production identity stayed 29,337 / 75 / 29,412,
> verified from the constructed model before and after. Any file path, class
> name or line number cited below refers to that file under its former name;
> the evidence recorded here still stands.

Analytic, from source. No GPU available in this environment — no measurement was
taken during this audit. Every number below is **computed**, labelled as such,
and must be confirmed by measurement before being relied on.

## 1. Where memory actually goes

Parameters are irrelevant: 40,656 params × 4 B ≈ **0.16 MB**. AdamW state (2
moments) ≈ 0.33 MB. EMA copy ≈ 0.16 MB. Total persistent ≈ **0.7 MB**.

**Activations dominate entirely.** Batch 1, float32, per NCA step:

| Tensor | Shape | L1 32³ | L2 96³ | L3 128³ |
|---|---|---|---|---|
| voxels | — | 32,768 | 884,736 | 2,097,152 |
| state `x` | (V, C) | 3.1 MB | 85 MB | 134 MB |
| `perceive` out (2C) | (V, 2C) | 6.3 MB | 170 MB | 268 MB |
| **`fc0` hidden (128)** | (V, 128) | **16.8 MB** | **453 MB** | **1,074 MB** |
| BN output | (V, 128) | 16.8 MB | 453 MB | 1,074 MB |
| `fc1` out | (V, C) | 3.1 MB | 85 MB | 134 MB |

**The peak is the `hidden_size=128` MLP activation at L3** — a single step's
`fc0` output is ~1.07 GB, and `bn` produces another ~1.07 GB. That one tensor
pair is larger than everything else in the model combined.

## 2. Peak without gradient checkpointing (production `v3_multilevel.yaml`)

All steps' activations are retained for backward:

| Level | Steps | Retained/step (approx, hidden+BN+state) | Subtotal |
|---|---|---|---|
| L1 32³ | 20 | ~40 MB | ~0.8 GB |
| L2 96³ | 20 | ~1.1 GB | ~22 GB |
| L3 128³ | 10 | ~2.4 GB | ~24 GB |

**Computed peak: tens of GB (order ~45 GB+).** This will not fit a 6 GB, 16 GB or
24 GB card and is marginal even on 40 GB. Classification: **COMPUTED, unmeasured.**

## 3. Peak with gradient checkpointing (`v3_multilevel_ckpt.yaml`)

Only the per-step **input** state is stored; hidden activations are recomputed in
backward. Retained ≈ (state per step) + (one step's full activation set live
during recompute):

| Level | Steps | Stored state | |
|---|---|---|---|
| L1 | 20 | ~63 MB | |
| L2 | 20 | ~1.7 GB | |
| L3 | 10 | ~1.3 GB | |
| transient recompute peak | — | ~2.4 GB | |

**Computed peak: roughly 5–7 GB.** Order-of-magnitude reduction, at the cost of
executing all 50 updates twice (~2× NCA compute).

`reports/validation/VRAM_STUDY.md` and `V3_GRADIENT_CHECKPOINTING_GPU_GATE.md`
contain prior measured figures — those, not these computed ones, should be the
citation of record.

## 4. Why it fits / where the peak occurs — direct answers

- **WHERE:** inside `BasicNCA3D.update`, at `fc0`/`bn` on the level-3 128³ volume
  (`Model_BasicNCA3D.py:161-163`).
- **WHAT tensor:** the `(1, 128, 128, 128, 128)` hidden activation — 2,097,152
  voxels × 128 hidden units × 4 B ≈ 1.07 GB, allocated twice per step (fc0 out,
  BN out).
- **WHY it fits with checkpointing:** only one step's hidden tensors are live at
  a time; the other 49 steps store just their small input state.
- **WHY it does not fit without:** all 50 steps' hidden tensors are retained.

## 5. Safe memory levers (Phase 2 candidates — NOT applied)

| Lever | Effect | Methodology impact |
|---|---|---|
| Enable `memory.gradient_checkpointing` in production | ~8× activation reduction | **NONE** — verified output/gradient/RNG identical (MODEL_AUDIT §3) |
| `zero_grad(set_to_none=True)` | ~0.16 MB | none (negligible here) |
| Mixed precision (autocast/bf16) | ~2× activation reduction + speed | **CHANGES NUMERICS** — would alter results; requires its own validation. Not recommended without a deliberate decision |
| Reduce `hidden_size` from 128 | large | **ARCHITECTURE CHANGE — forbidden** |
| Reduce L3 resolution / steps | large | **ARCHITECTURE CHANGE — forbidden** |

Only the first is both material and methodology-neutral.

## 6. Notes

- `torch.cuda.reset_peak_memory_stats()` is called once before training
  (`runner.py:415-416`) and `max_memory_allocated()` is read per epoch (`:457`)
  and at the end (`:507`). Peak VRAM is therefore genuinely recorded per run.
  Good practice — the run itself will report the real number.
- Evaluation (`metrics_eval.collect_probs:23`) runs under `torch.no_grad()`, so
  no graph is built and checkpointing correctly disables itself
  (`Model_BasicNCA3D.py:193`). Eval memory is a small fraction of training.
- `collect_probs` accumulates **every** case's full probability volume and GT on
  the **host** as numpy (`metrics_eval.py:27-29`). One `(1,128,128,128,3)` float32
  array is ~25 MB, so prob+gt per case is ~50 MB.
  - Per-epoch validation (`runner.py:444`): 200 val cases -> **~10 GB host RAM**,
    allocated and freed **every epoch** for 300 epochs.
  - Final evaluation (`runner.py:520-523`): `val_pairs` and `test_pairs` are held
    **simultaneously** (val_pairs is still referenced at `:564`
    `_write_threshold_comparison`), so 200 + 198 cases = **~20 GB host RAM live at
    once**.

  This is **host RAM, not VRAM**, so it will not show in `max_memory_allocated`.
  On a VM with less than ~24 GB RAM the final evaluation stage is an OOM risk that
  would strike only **after** the full 300-epoch training completed — the most
  expensive possible moment to fail. Classification: **CONFIRMED** code fact;
  severity **P1** on memory-constrained hosts, P2 on a large VM. Verify the
  target VM's RAM before the campaign.
