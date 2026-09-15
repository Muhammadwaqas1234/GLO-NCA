# GLO-NCA V3 — Checkpointed 128³ GPU Memory Gate (clean, decisive)

**Date:** 2026-09-15 · **GPU:** NVIDIA L4 · **VRAM:** 23.66 GB usable
**Config:** `configs/v3_multilevel_ckpt.yaml` (production V3 + `memory.gradient_checkpointing: true`)
**Architecture:** UNCHANGED (params 40,656) · **Checkpointing:** OPT-IN, MEMORY-ONLY

> One clean, self-verifying measurement. Fresh `python -B` process per resolution
> (`PYTHONDONTWRITEBYTECODE=1`, `__pycache__` cleared → no stale-.pyc ambiguity),
> new results file, checkpoint-call counter + per-level flag assertions to PROVE
> checkpointing was actually active (the earlier artifact is ruled out). Synthetic
> input tensors (memory-only test; no dataset). Real CUDA, batch 1, exact
> production channels / NCA steps / resolutions.

## Integrity proof (checkpointing really active)
- `level_use_checkpoint = [true, true, true]` (all 3 levels)
- `model_gc = true`, `cfg_flag = true`
- `checkpoint_calls_forward = 50` per run (the checkpoint path executed 50 times)
- `pyc_disabled = true` (fresh `python -B`), fresh process per resolution
- Imported `Model_BasicNCA3D.py` SHA256 = `96cddf897b8c75a6265ea3e26d4c3e5a8d78ad5c2fe037dcbe80932d07a31c7e`
- Imported `Model_GLO_NCA_V3.py` SHA256 = `5335d301c3a6b88f86199c722e9467ecdc885812a2449d111cd51963b2423b13`

## Measured results (checkpointing ON)
| Resolution | forward peak | forward+backward peak | max reserved | result |
|---|---|---|---|---|
| **96³**  | 2.881 GB | **3.839 GB** | 4.297 GB | **TRUE FIT** |
| **128³** | 6.818 GB | **9.072 GB** | 10.112 GB | **TRUE FIT** |

Both comfortably below the L4's 23.66 GB (128³ uses ~9.1 GB allocated / ~10.1 GB
reserved → ~2.3–2.6× headroom).

## Comparison to the original (no checkpointing)
| Resolution | original fwd+bwd | checkpointed fwd+bwd | outcome |
|---|---|---|---|
| 96³  | OOM (>22.5 GB) | 3.84 GB | now fits |
| 128³ | OOM (>22.6 GB); est. ~87–101 GB | 9.07 GB | **now fits the L4** |

Checkpointing does NOT shrink the *original* model footprint; it provides an
equivalent-result execution (bit-identical outputs/gradients, verified earlier)
with a much smaller activation footprint.

## Result
```
GPU:                         NVIDIA L4
VRAM:                        23.66 GB

96³ forward peak:            2.881 GB
96³ forward+backward peak:   3.839 GB
96³ TRUE FIT:                PASS

128³ forward peak:           6.818 GB
128³ forward+backward peak:  9.072 GB
128³ max reserved:           10.112 GB
128³ TRUE FIT:               PASS

Checkpointing active:        YES (level flags [T,T,T]; 50 checkpoint calls; python -B)
Imported model SHA256:       Model_BasicNCA3D 96cddf89… ; Model_GLO_NCA_V3 5335d301…
Architecture:                UNCHANGED (40,656 params)
Checkpointing:               OPT-IN MEMORY-ONLY (default OFF; ckpt config opts in)

FINAL RESULT:
128³ CHECKPOINTED TRUE FIT = PASS
```

## Implication
The production V3 128³ training step (unchanged math/outputs, checkpointing ON)
**fits the existing NVIDIA L4 24 GB with large headroom** — no A100/H100/H200
needed. The 3-epoch real-data smoke test is now unblocked (to be run as a separate
controlled task).

## State
- VM `glo-nca-v3-l4`: STOP issued immediately after the two measurements (cost control).
- No dataset work, no pre-training gate, no training run in this task.
- No commit made. `configs/v3_multilevel.yaml` unchanged (checkpointing OFF by default).
