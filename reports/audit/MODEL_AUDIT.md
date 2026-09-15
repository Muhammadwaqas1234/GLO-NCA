# MODEL AUDIT — GLO-NCA V3 + BasicNCA3D + NCA update + gradient checkpointing

## 1. Architecture verification (PASS)

`Model_GLO_NCA_V3.py`, `Model_BasicNCA3D.py`.

| Requirement | Verdict | Evidence |
|---|---|---|
| Input `(B,X,Y,Z,4)` | PASS | `forward(modalities_cl)` `:170-177` |
| Output `(B,3,X,Y,Z)` | PASS | `seg_head` Conv3d(16→3) `:147`, returned `:211-212` |
| 3 levels, strict L1→L2→L3 flow | PASS | loop `:182-198`; `prev_state` chained |
| Spatial alignment | PASS | every level resized to its own `res` `:185`; all states resized to `fine_res` before fusion `:205` |
| Learnable 1×1×1 fusion | PASS | `self.fuse = Conv3d(fine_ch*3, fine_ch, 1)` `:141` |
| Sigmoid usage | PASS | model emits **logits**; sigmoid applied exactly once, inside the loss `LossFunctions.py:133` and once in eval `metrics_eval.py:27`. **No double sigmoid.** |
| Parameter count = 40,656 | PASS | independently recomputed; see PHASE1_BASELINE.md |
| Not an ensemble | PASS | one forward path, one head; levels feed forward, never averaged |

### Channel bookkeeping (verified)

- `_seed:152-159` — zeros `(B,X,Y,Z,channels)`, modalities into first 4.
- Projections `:126-130` — `prev_full(24) → nxt_state(next.channels - 4)`, i.e.
  24→20 for L1→L2 and 24→12 for L2→L3. Correct: the projection targets exactly
  the state region, and is **added** into it (`:194`), leaving the modality
  channels untouched. This is the intended nested-cascade injection.
- `level_to_fine :138-140` — each level's full width → `fine_ch=16`.

### Interpolation choices

- Modalities downsampled `trilinear` (`:185`) — appropriate for intensities.
- Projected state and fusion states resized `nearest` (`:192`, `:205`) — a
  deliberate choice for learned feature maps. Noted, not a defect.

## 2. NCA update audit (PASS with one INFO)

`BasicNCA3D.update:152-180`.

Order verified: `transpose→perceive(p0 depthwise conv→SE→spatial GC)→fc0→BN→
ReLU→dropout→fc1→stochastic mask→residual add→transpose`.

| Aspect | Verdict | Evidence |
|---|---|---|
| State init | PASS | seeded by V3 `_seed`, not by the NCA |
| Perception | PASS | `perceive:144-150` — depthwise conv (`groups=channel_n`), reflect padding, concat with identity |
| Fire rate | PASS | per-cell mask `rand(...) > fire_rate` broadcast over channels `:172-174` |
| Residual update | PASS | `x = x + dx` `:176` |
| Input-channel preservation | PASS | `forward:204` re-concatenates the ORIGINAL modality channels each step, so modalities are never overwritten — the documented V2 convention |
| Steps L1=20, L2=20, L3=10, total 50 | PASS | from config, passed at `Model_GLO_NCA_V3.py:196` |

**INFO M-01.** `update` performs 6 `transpose(1,4)` calls per step (`:158,160,162,
164,168? ,176,178`). At 50 steps that is ~300 layout swaps per forward. `transpose`
itself is a view (free), but the subsequent `Conv3d`/`BatchNorm3d` force
materialisation. This is inherent to the channels-last NCA convention inherited
unchanged from V2 and is **required for V2↔V3 comparability**. Listed as an
observation only — changing it would alter the shared `BasicNCA3D` and therefore
V2. Recommend NO CHANGE.

**INFO M-02.** `forward:203` calls `.clone()` on every step's output, then
`:204` builds a new tensor via `torch.concat`. Both allocate. Again inherited
V2 code; changing it touches V2. Recommend NO CHANGE without a separate decision.

## 3. Gradient checkpointing audit

`Model_BasicNCA3D.py:193-205`.

```python
use_ckpt = getattr(self, "use_checkpoint", False) and torch.is_grad_enabled()
...
x2 = torch.utils.checkpoint.checkpoint(
    self.update, x, fire_rate, use_reentrant=False, preserve_rng_state=True).clone()
```

| Required property | Verdict | Evidence |
|---|---|---|
| `use_reentrant=False` | PASS | `:201` |
| `preserve_rng_state=True` | PASS | `:201` |
| Correct function boundary | PASS | wraps exactly one `update` = one NCA step. Not too coarse, not too fine |
| No double checkpointing | PASS | flag lives only on `BasicNCA3D`; `GLO_NCA_V3_MultiLevel.forward` does not wrap anything itself |
| Applied to every level | PASS | `Model_GLO_NCA_V3.py:118-119` sets `nca.use_checkpoint` on all three |
| Inappropriate ops checkpointed | PASS | projections, fusion and head are outside the checkpointed region — correct, they are cheap and their activations are small |
| OFF under inference | PASS | gated on `torch.is_grad_enabled()`, so `no_grad` eval pays no recompute |
| Changes outputs? | **NO** | `preserve_rng_state=True` re-seeds the fork RNG before recompute, so the `torch.rand` fire-rate mask at `:172` is reproduced identically |
| Changes gradients? | **NO** | mathematically identical; only storage strategy differs |
| Changes RNG stream? | **NO** | checkpoint saves/restores the RNG fork, and consumes no extra draws from the global stream |

**Verdict: the checkpointing implementation is CORRECT.** No correctness issue
found, so per the audit brief the standing requirement (`checkpointing = TRUE`)
stands.

### FINDING M-03 (P1, CONFIRMED) — the production config does not enable it

`build_v3_from_config:243` reads the flag from top-level `memory.gradient_
checkpointing`, defaulting `False`. **`configs/v3_multilevel.yaml` has no
`memory:` block at all** — so the production config runs with checkpointing
**OFF**. Only `configs/v3_multilevel_ckpt.yaml` enables it, and that file
describes itself as *"Not the production config"* (`v3_multilevel_ckpt.yaml`
trailing comment).

This is a direct contradiction between the stated requirement
("checkpointing = TRUE") and the file named as production. Two files are
otherwise byte-identical, which makes it very easy to launch the wrong one.
**Which file is production must be resolved before any training run.**

Note the trade-off is real and opposite-signed: checkpointing ~halves activation
memory but roughly **doubles** NCA compute (each of the 50 updates is executed
twice). At 100% observed GPU utilisation that is a ~2× wall-clock cost. This is a
decision for you, not the audit.

## 4. Memory analysis (see MEMORY_AUDIT.md for detail)

Peak is dominated by **activations, not parameters** (40,656 params ≈ 0.16 MB).

Per-step stored activation, checkpointing OFF, batch 1, float32:

| Level | Voxels | State C | `perceive` output (2C) | ~Bytes/step |
|---|---|---|---|---|
| L1 32³ | 32,768 | 24 | 48 | ~6 MB |
| L2 96³ | 884,736 | 24 | 48 | ~170 MB |
| L3 128³ | 2,097,152 | 16 | 32 | ~268 MB |

The `fc0` hidden tensor is the true peak: `hidden_size=128` over the full volume.
At L2: 884,736 × 128 × 4 B ≈ **453 MB for a single step's hidden activation**;
at L3: 2,097,152 × 128 × 4 B ≈ **1.07 GB**. With 20 L2 steps + 10 L3 steps all
retained for backward, the graph is in the **tens of GB** range without
checkpointing. **The `hidden=128` MLP activation at L2/L3 is the memory peak** —
not the NCA state, not the fusion.

This is why checkpointing exists here, and why M-03 matters.
