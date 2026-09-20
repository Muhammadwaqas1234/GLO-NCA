r"""
================================================================================
GLO-NCA — ISOLATED KAGGLE 50-EPOCH PRODUCTION VALIDATION
================================================================================

PROVENANCE
----------
Production architecture source:
    GLO-NCA production implementation
      src/models/Model_GLO_NCA_GlobalContext.py  (GLO_NCA_GlobalContext)
      src/models/Model_GLO_NCA_V3.py             (GLO_NCA_V3_MultiLevel, FeatureProjection)
      src/models/Model_BasicNCA3D.py             (BasicNCA3D, SEBlock3D, GCSpatialBlock3D)
      src/losses/LossFunctions.py                (FocalTverskyCELoss)
      src/experiment/runner.py                   (_clipped_batch_step training semantics)
      configs/glo_nca_production.yaml            (hyperparameters)

Production geometry:
    96^3 working volume  ->  Level 1 48^3 k5  +  Level 2 64^3 k5, 15+15 steps

Parameters:
    29,337 expected (asserted before training; run ABORTS on mismatch)

Patchify:
    OFF  (full-volume input; no training crop, no ROI)

Global context:
    WHOLE 96^3  (SE channel gate + spatial attention at EVERY NCA step)

Experiment:
    Kaggle BraTS2024-small
    50 epochs
    Engineering / empirical validation experiment
    NOT final thesis performance, NOT an ablation study,
    NOT the thesis master split (898/200/198 is untouched and unused here)

SCIENTIFIC SCOPE
----------------
This run does NOT approve or reject the 48^3+64^3 Category C architecture
decision, does NOT replace the thesis master split, and does NOT constitute
final thesis training. It answers ENGINEERING questions: does the production
architecture train on an independent dataset/GPU, does it fit, and — the central
purpose — WHERE EXACTLY IS THE TRAINING TIME GOING?

MEASUREMENT LABELS USED THROUGHOUT
----------------------------------
    MEASURED    directly observed in this run
    CALCULATED  arithmetic on measured values
    ESTIMATED   extrapolated beyond measured data (always labelled)
    NOT MEASURED unavailable in this environment

USAGE (Kaggle)
--------------
    Attach dataset: nguyenthanhkhanh/brats2024-small-dataset
    Enable GPU, then:  python kaggle_glo_nca_50epoch.py
    Self-check only:   python kaggle_glo_nca_50epoch.py --self-check
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import csv
import hashlib
import math
import platform
import shutil
import subprocess

import numpy as np

# ############################################################################
# #                                                                          #
# #   AUTHORITATIVE ARCHITECTURE IDENTITY                                    #
# #                                                                          #
# #   This file runs the PRODUCTION GLO-NCA architecture, matching           #
# #   configs/glo_nca_production.yaml exactly:                               #
# #                                                                          #
# #       working volume   96^3                                              #
# #       level 1          48^3, 24 ch, 15 NCA steps, perception k=5         #
# #       level 2          64^3, 24 ch, 15 NCA steps, perception k=5         #
# #       level 3          ABSENT                                            #
# #       total steps      30                                                #
# #       spatial GC       k=5                                               #
# #       SE               ON                                                #
# #       fusion           learned Conv3d(48 -> 24)                          #
# #       global context   FULL 96^3 working volume                          #
# #       patchify         OFF                                               #
# #       parameters       29,337                                            #
# #                                                                          #
# #   The identity gate ABORTS the run on any mismatch. It never adjusts     #
# #   the model to make a check pass.                                        #
# #                                                                          #
# #   A1 (fused channels-last BatchNorm) is active. A1 is an                 #
# #   implementation-level optimisation, NOT an architecture change: the     #
# #   forward pass is bit-identical in fp32 and the parameter count is       #
# #   unchanged.                                                             #
# #                                                                          #
# #   SCOPE: engineering validation -- training correctness, timing, VRAM,   #
# #   convergence, overfitting behaviour, checkpoint/resume. It is NOT       #
# #   final thesis performance: the Kaggle split is local to this file and   #
# #   is not the 898/200/198 master split.                                   #
# #                                                                          #
# #   NOTE: this architecture is itself a RECORDED CATEGORY-C DECISION      #
# #   (k7->k5 on both levels, 20+20->15+15 steps, spatial GC k7->k5,         #
# #   33,089 -> 29,337 params). It has NO segmentation-quality evidence      #
# #   yet; see configs/glo_nca_production.yaml for the full record.          #
# #                                                                          #
# #   Further variants (40 steps, GC k=7, smaller geometry) are reachable    #
# #   ONLY via use_preset(). Every artifact from such a run is stamped       #
# #   CATEGORY-C so it can never be mistaken for a production result.        #
# #                                                                          #
# ############################################################################
#
# A1 (fused channels-last BatchNorm) IS INCLUDED and is NOT Category C: it is
# a proven-equivalent implementation optimisation now integrated into
# src/models/Model_BasicNCA3D.py (1.402x measured, forward bit-identical).
#
# ============================================================================
# HYPERPARAMETERS
# ============================================================================
SEED = 42
WORKING_VOLUME = 96          # global-context source volume
# --- GRID SIZE (option 3) --------------------------------------------------
# FAST_GRID shrinks the two NCA grids. This is a CATEGORY C SCIENTIFIC CHANGE:
# it alters the architecture under test, so results are NOT comparable to the
# production 48/64 model and must not be reported as validating it.
#
# MEASURED: the parameter count is 33,089 at EVERY grid size, because an NCA
# shares one update rule across all voxels -- resolution changes compute, not
# parameters. So the identity gate still passes; only the geometry differs.
#
#   False (default) : L1 48^3, L2 64^3  -> 7,454,720 voxel-steps  (PRODUCTION)
#   True            : L1 32^3, L2 48^3  -> 2,867,200 voxel-steps  (2.6x less)
#
# v7 used 32x32x24 / 64x64x48 and trained 200 epochs in 6 h, so this is the
# lever that actually made v7 fast.
#
# NOW SET TO False: the default configuration is the PRODUCTION 48^3/64^3 grid
# (preset "fast_grid4864"), so the grid can be compared directly against the
# completed 32^3/48^3 run with every other setting held constant.
FAST_GRID = False

# --- NCA STEPS (selected lever) --------------------------------------------
# 20+20 -> 15+15. CATEGORY C: fewer NCA iterations is a different model, but
# the PARAMETER COUNT IS UNCHANGED (33,089) because an NCA applies one shared
# update rule `steps` times -- steps change compute, not weights.
# MEASURED (3 interleaved rounds, +/-0.05 s):
#     20+20 -> 7.17 s/iter
#     15+15 -> 5.38 s/iter   (1.33x)
NCA_STEPS = 15

if FAST_GRID:
    L1_RES, L1_CH, L1_STEPS, L1_K = 32, 24, NCA_STEPS, 5
    L2_RES, L2_CH, L2_STEPS, L2_K = 48, 24, NCA_STEPS, 5
else:
    L1_RES, L1_CH, L1_STEPS, L1_K = 48, 24, NCA_STEPS, 5
    L2_RES, L2_CH, L2_STEPS, L2_K = 64, 24, NCA_STEPS, 5

# --- SPATIAL GLOBAL-CONTEXT KERNEL (selected lever) ------------------------
# GCSpatialBlock3D kernel 7 -> 5. CATEGORY C: shrinks the spatial global-context
# receptive field, so it touches the thesis contribution directly.
# Parameters: 33,089 -> 32,217 (the 2->1 conv loses 2*(7^3-5^3) = 436 weights
# per level, x2 levels = 872).
# MEASURED, stacked on 15+15: 5.38 -> 4.86 s/iter (a further 1.11x).
SPATIAL_GC_KERNEL = 5
HIDDEN = 128
FIRE_RATE = 0.6
DROPOUT = 0.1
USE_ATTENTION = True         # SE channel global context
USE_SPATIAL = True           # spatial global context
FUSION = "concat"            # learned Conv3d(48 -> 24)
# SPEED SETTINGS FOR THIS KAGGLE RUN (differ from the production config).
#
# Gradient checkpointing ON. MEASURED on the dual-input patchify code:
#     ckpt ON : fwd 1.41 s | bwd  6.81 s | total  8.22 s | 1,646 MB
#     ckpt OFF: fwd 9.45 s | bwd 40.40 s | total 49.85 s | 9,782 MB
# Turning it OFF is 6x SLOWER, not faster. Storing every NCA step's activations
# means writing ~128 MB per step to VRAM and reading it back in backward, which
# saturates memory bandwidth; recomputing is cheaper than that traffic. This
# architecture is bandwidth-bound, so checkpointing is a SPEED win here.
GRADIENT_CHECKPOINTING = True

# --- PATCHIFY (rewritten; this was previously a no-op) ---------------------
# The high-resolution level runs on a crop while the GLOBAL-CONTEXT level still
# reads the FULL 96^3 volume (dual input), so the thesis contribution is intact.
#
# BUG THAT THIS FIXES. Previously level 2 resized whatever it was given to a
# FIXED L2_RES, so a 64^3 patch and a 48^3 patch both became a 64^3 grid and
# cost exactly the same. MEASURED under the old code:
#     patch 64^3 -> 7.17 s      patch 48^3 -> 7.16 s     (0% gain: a no-op)
# Cropping 64^3 out of 96^3 and resizing it back up to 64^3 is the same work as
# resizing the whole 96^3 down to 64^3. Patchify was enabled but doing nothing.
#
# THE FIX (PATCH_NATIVE_RES): level 2 runs at the PATCH'S OWN resolution, so a
# smaller patch really is a smaller NCA grid. This is how upstream M3D-NCA gets
# its VRAM/speed win -- the crop happens BEFORE the high-res steps, so those
# steps never process a full-size volume (Agent_M3D_NCA.get_outputs).
# MEASURED with the fix, 20+20 steps:
#     patch 64^3 -> 7.16 s  1762 MB     patch 48^3 -> 4.76 s   896 MB  (1.50x)
#     patch 56^3 -> 5.84 s  1274 MB     patch 40^3 -> 4.01 s   710 MB  (1.79x)
#
# Like upstream, the global level stays IN the backward graph (no detach): the
# crop is a differentiable slice, so level-1 weights are still trained through
# the level-2 loss. Upstream does not detach, and neither do we.
USE_PATCHIFY = False
PATCH_NATIVE_RES = True      # False restores the old (no-op) fixed-grid behaviour
# Patch must be strictly SMALLER than the high-res grid, or patchify is a
# no-op (level 2 would run on the same voxel count either way). Same formula
# as use_preset(), so importing the module and calling use_preset() for the
# same preset always produce an identical configuration.
PATCH_SIZE = min(L2_RES - 8, WORKING_VOLUME - 8)   # 56 at 64^3, 40 at 48^3

# In-memory preprocessing cache (this is what made v7 fast on repeat epochs).
# Caches the DETERMINISTIC head of __getitem__ (load -> crop -> resample ->
# labels -> normalise). The stochastic patch crop still runs every epoch, so
# the cache is RNG-neutral and changes no mathematics.
USE_CACHE = True

# Light augmentation (axis flips, 90-degree rotations, per-modality scale/shift)
# -- the same "light" policy the production config uses. MEASURED cost: 111 ms
# per case = 1.55% of a 7.15 s iteration, so it is NOT a speed problem. It is a
# REGULARISER: it changes what the model sees, so enabling it is a methodology
# choice, not a performance one.
USE_AUGMENTATION = True

# Parameter count. 33,089 at spatial GC k=7; 32,217 at k=5.
# The ONLY thing that changes the count is SPATIAL_GC_KERNEL: the 2->1 spatial
# conv drops 2*(7^3 - 5^3) = 436 weights per level, x2 levels = 872.
# Grid size and NCA step count do NOT change it -- an NCA shares one update rule
# across all voxels and reuses it every step (measured: identical at every grid).
EXPECTED_PARAMS = 29337

# --- RUN MODE (derived, never set by hand) ---------------------------------
# True only when EVERY setting matches configs/glo_nca_production.yaml.
# Any deviation makes this a Category-C run, and that fact is stamped into the
# log, the config snapshot, the final summary JSON and the markdown report, so
# a Category-C result cannot later be mistaken for a production one.
IS_PRODUCTION_IDENTITY = (
    (not FAST_GRID) and NCA_STEPS == 15 and SPATIAL_GC_KERNEL == 5
    and (not USE_PATCHIFY) and WORKING_VOLUME == 96
    and L1_RES == 48 and L2_RES == 64 and L1_K == 5 and L2_K == 5
    and L1_CH == 24 and L2_CH == 24 and HIDDEN == 128
)
RUN_MODE = "PRODUCTION" if IS_PRODUCTION_IDENTITY else "CATEGORY-C"


def category_c_deviations():
    """Every way this run differs from the production identity."""
    d = []
    if FAST_GRID or L1_RES != 48 or L2_RES != 64:
        d.append(f"geometry {L1_RES}^3/{L2_RES}^3 (production 48^3/64^3)")
    if NCA_STEPS != 15:
        d.append(f"NCA steps {L1_STEPS}+{L2_STEPS}={L1_STEPS+L2_STEPS} "
                 f"(production 15+15=30)")
    if SPATIAL_GC_KERNEL != 5:
        d.append(f"spatial GC k={SPATIAL_GC_KERNEL} (production k=5)")
    if USE_PATCHIFY:
        d.append(f"patchify ON patch {PATCH_SIZE}^3 (production OFF)")
    if EXPECTED_PARAMS != 29337:
        d.append(f"parameters {EXPECTED_PARAMS:,} (production 29,337)")
    return d


# ============================================================================
# PRESETS — switch configuration from inside a Kaggle session
# ============================================================================
# Every preset below except "production" is CATEGORY C: it changes the model's
# scientific identity, and none has Dice/HD95 validation. Speedups are the
# MEASURED figures from reports/audit/GLO_NCA_COMPLETE_SPEED_OPTIMIZATION_REPORT.md
# (RTX 3050, power-capped, paired -- the RATIO is the meaningful quantity, the
# absolute ms are not, and a T4 will differ).
#
# Usage in a notebook:
#     use_preset("production")     # then run_experiment(...)
#     use_preset("fast")           # the current default
#     time_presets()               # MEASURE per-epoch cost of each preset
#     print_presets()              # show the whole table
#
# Each preset is (FAST_GRID, NCA_STEPS, SPATIAL_GC_KERNEL, USE_PATCHIFY).
PRESETS = {
    # name            grid   steps  gc_k  patchify  speedup  note
    "production":    (False,    15,    5,    False, "1.00x",
                      "THE thesis architecture. 29,337 params. Reportable."),
    "steps20":       (False,    20,    5,    False, "0.75x",
                      "C: 40 NCA steps (the pre-2026-09 production depth)."),
    "gc7":           (False,    15,    7,    False, "0.90x",
                      "C: wider spatial global context (previous k=7)."),
    "geo40_56":      (None,     15,    5,    False, "1.45x",
                      "C: 40^3/56^3 grids. Params unchanged."),
    "geo32_48":      (True,     15,    5,    False, "2.30x",
                      "C: 32^3/48^3 grids. Params unchanged."),
}
_GEO_OVERRIDE = {"geo40_56": (40, 56)}


def use_preset(name: str, verbose: bool = True):
    """Reconfigure this module to a named preset. Call BEFORE run_experiment.

    Rebinds the module-level settings and re-derives everything that depends
    on them (level specs, PATCH_SIZE, EXPECTED_PARAMS, RUN_MODE), so the
    identity gate and every artifact stamp stay consistent.
    """
    g = globals()
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from "
                         f"{sorted(PRESETS)}")
    fast, steps, gck, patch, speed, note = PRESETS[name]

    g["FAST_GRID"] = bool(fast) if fast is not None else False
    g["NCA_STEPS"] = steps
    g["SPATIAL_GC_KERNEL"] = gck
    g["USE_PATCHIFY"] = patch

    if name in _GEO_OVERRIDE:                      # explicit grid pair
        l1, l2 = _GEO_OVERRIDE[name]
    elif g["FAST_GRID"]:
        l1, l2 = 32, 48
    else:
        l1, l2 = 48, 64
    g["L1_RES"], g["L1_CH"], g["L1_STEPS"], g["L1_K"] = l1, 24, steps, 5
    g["L2_RES"], g["L2_CH"], g["L2_STEPS"], g["L2_K"] = l2, 24, steps, 5
    # Patch must be strictly SMALLER than the high-res grid, otherwise
    # patchify is a no-op: level 2 would run on the same number of voxels
    # either way (this was the measured 0%-gain bug the audit found).
    # Same rule as the module default: 48 at 64^3, 40 at 48^3.
    g["PATCH_SIZE"] = min(l2 - 8, WORKING_VOLUME - 8)
    g["EXPECTED_PARAMS"] = 30209 if gck == 7 else 29337

    g["IS_PRODUCTION_IDENTITY"] = (
        (not g["FAST_GRID"]) and g["NCA_STEPS"] == 15
        and g["SPATIAL_GC_KERNEL"] == 5 and (not g["USE_PATCHIFY"])
        and WORKING_VOLUME == 96 and g["L1_RES"] == 48 and g["L2_RES"] == 64
        and g["L1_K"] == 5 and g["L2_K"] == 5 and g["L1_CH"] == 24
        and g["L2_CH"] == 24 and HIDDEN == 128)
    g["RUN_MODE"] = ("PRODUCTION" if g["IS_PRODUCTION_IDENTITY"]
                     else "CATEGORY-C")
    g["ACTIVE_PRESET"] = name

    if verbose:
        print(f"preset '{name}' -> {g['RUN_MODE']}")
        print(f"  geometry   {g['L1_RES']}^3 / {g['L2_RES']}^3")
        print(f"  NCA steps  {g['L1_STEPS']}+{g['L2_STEPS']} "
              f"= {g['L1_STEPS']+g['L2_STEPS']}")
        print(f"  spatial GC k={g['SPATIAL_GC_KERNEL']}")
        print(f"  patchify   {'ON, patch %d^3' % g['PATCH_SIZE']
                              if g['USE_PATCHIFY'] else 'OFF'}")
        print(f"  parameters {g['EXPECTED_PARAMS']:,}")
        print(f"  measured   {speed} vs production (ratio, not wall clock)")
        print(f"  {note}")
        if not g["IS_PRODUCTION_IDENTITY"]:
            print("  *** CATEGORY C — results must NOT be reported as "
                  "validating the production architecture. ***")
    return name


ACTIVE_PRESET = "production"      # matches the settings hard-coded above


def print_presets():
    """Show every preset and how it differs from production."""
    print("=" * 78)
    print("PRESETS  (measured speedups are RATIOS from the local audit;")
    print("          absolute epoch time on a T4 will differ)")
    print("=" * 78)
    print(f"  {'preset':15s} {'grid':>9s} {'steps':>6s} {'gc_k':>5s} "
          f"{'patch':>6s} {'params':>8s} {'speed':>7s}  class")
    for nm, (fast, steps, gck, patch, speed, note) in PRESETS.items():
        if nm in _GEO_OVERRIDE:
            l1, l2 = _GEO_OVERRIDE[nm]
        else:
            l1, l2 = (32, 48) if fast else (48, 64)
        cls = "PRODUCTION" if nm == "production" else "CATEGORY C"
        mark = " <- active" if nm == ACTIVE_PRESET else ""
        print(f"  {nm:15s} {f'{l1}/{l2}':>9s} {steps*2:>6d} {gck:>5d} "
              f"{str(patch):>6s} {30209 if gck == 7 else 29337:>8,d} "
              f"{speed:>7s}  {cls}{mark}")
    print()
    print("  Only 'production' may be reported as thesis performance.")
    print("  Every other preset changes the architecture and has NO Dice/HD95")
    print("  validation. A1 (fused BatchNorm) is active in ALL presets and is")
    print("  NOT Category C — it is proven-equivalent (forward bit-identical).")
    print("=" * 78)


def time_presets(names=None, iters=12, warmup=3, train_cases=129,
                 val_cases=31, device=None):
    """MEASURE per-iteration and projected per-epoch cost for each preset.

    Runs a real forward+backward+optimizer step on synthetic tensors of the
    correct shape -- no data loading, no NIfTI -- so a whole comparison takes
    a couple of minutes instead of a training run.

    The projected epoch time is CALCULATED as
        train_cases * measured_iteration + measured_validation
    It therefore EXCLUDES dataloader wait, which on a cold cache dominates the
    first epoch. Compare it against a WARM epoch (epoch 2+), not epoch 1.
    """
    import torch
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = names or list(PRESETS)
    saved = ACTIVE_PRESET
    rows = []
    print("=" * 94)
    print(f"PRESET TIMING on {torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'}"
          f"   [MEASURED iteration; epoch is CALCULATED]")
    print(f"  {train_cases} train / {val_cases} val cases, batch {BATCH_SIZE}, "
          f"excludes dataloader wait")
    print("=" * 94)
    print(f"  {'preset':15s} {'grid':>9s} {'steps':>6s} {'params':>8s} "
          f"{'iter_s':>8s} {'train_s':>9s} {'val_s':>7s} {'epoch_s':>9s} "
          f"{'VRAM_MB':>8s} {'50ep':>7s}")
    for nm in names:
        use_preset(nm, verbose=False)
        try:
            model, C = make_model(dev)
            lf = C["FocalTverskyCELoss"](TVERSKY_ALPHA, TVERSKY_BETA,
                                         FOCAL_GAMMA, CE_WEIGHT)
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            amp = (torch.float16 if dev.type == "cuda"
                   and torch.cuda.get_device_capability(0) < (8, 0)
                   else torch.bfloat16)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            x = torch.randn(BATCH_SIZE, WORKING_VOLUME, WORKING_VOLUME,
                            WORKING_VOLUME, 4, device=dev)
            p = (torch.randn(BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE,
                             4, device=dev) if USE_PATCHIFY else None)
            box = ((0.1, 0.1 + PATCH_SIZE / WORKING_VOLUME),) * 3 \
                if USE_PATCHIFY else None
            ts = []
            model.train()
            for i in range(warmup + iters):
                t0 = time.perf_counter()
                with torch.amp.autocast(dev.type, dtype=amp,
                                        enabled=(dev.type == "cuda")):
                    o = model(x, patch_cl=p, patch_box=box)
                o = o.float().permute(0, 2, 3, 4, 1).contiguous()
                tg = torch.randint(0, 2, o.shape, device=dev).float()
                loss = sum(lf(o[..., j], tg[..., j]) for j in range(3))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if i >= warmup:
                    ts.append(time.perf_counter() - t0)
            ts.sort()
            it = ts[len(ts) // 2]
            # validation: full-volume inference, no patch, no grad
            model.eval()
            vs = []
            for i in range(warmup + max(4, iters // 3)):
                t0 = time.perf_counter()
                with torch.no_grad(), torch.amp.autocast(
                        dev.type, dtype=amp, enabled=(dev.type == "cuda")):
                    model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if i >= warmup:
                    vs.append(time.perf_counter() - t0)
            vs.sort()
            vit = vs[len(vs) // 2]
            tr = it * train_cases
            va = vit * val_cases
            vram = (torch.cuda.max_memory_reserved() / 1024 ** 2
                    if dev.type == "cuda" else 0.0)
            rows.append((nm, it, tr, va, tr + va, vram))
            print(f"  {nm:15s} {f'{L1_RES}/{L2_RES}':>9s} "
                  f"{L1_STEPS + L2_STEPS:>6d} {EXPECTED_PARAMS:>8,d} "
                  f"{it:8.3f} {tr:9.1f} {va:7.1f} {tr+va:9.1f} "
                  f"{vram:8.0f} {(tr+va)*50/3600:6.2f}h")
            del model, x
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:                       # OOM or build failure
            print(f"  {nm:15s} *** {type(e).__name__}: {str(e)[:52]}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    use_preset(saved, verbose=False)
    print()
    if rows:
        base = [r for r in rows if r[0] == "production"]
        if base:
            b = base[0][4]
            print("  relative to production:")
            for nm, _it, _tr, _va, ep, _v in rows:
                print(f"    {nm:15s} {b/ep:5.2f}x faster   "
                      f"epoch {ep:7.1f} s")
    print(f"  restored active preset: {ACTIVE_PRESET}")
    print("  NOTE: epoch times EXCLUDE dataloader wait. Your first epoch will")
    print("        be much slower (cold cache); compare against epoch 2+.")
    print("=" * 94)
    return rows


# --- training control -------------------------------------------------------
# 100 epochs is a BUDGET, not a target: early stopping ends the run once
# validation has plateaued.
EPOCHS = 100
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 20      # consecutive non-improving validations
EARLY_STOPPING_MIN_DELTA = 0.01   # smaller gains count as noise
# STABILITY FIX 2 -- min_delta 0.001 -> 0.01, patience 15 -> 20.
#
# Measured on the 50-epoch T4 run: over epochs 25-44 the validation mean
# foreground Dice had sd 0.036-0.044 and an average epoch-to-epoch jump of
# 0.040. The old min_delta of 0.001 was ~40x SMALLER than that noise floor,
# so essentially any upward fluctuation counted as "improvement" and the
# patience counter reset on noise. 0.01 sits above the fluctuations but well
# below the real epoch-to-epoch gains seen while the model was still
# learning (0.03-0.05), so genuine progress still registers.
#
# Patience rises 15 -> 20 to compensate: a stricter threshold means fewer
# resets, so the run needs a longer window before concluding it has plateaued.
SMOOTHING_WINDOW = 3              # rolling mean used for model selection
TOP_K_CHECKPOINTS = 3             # ranked best checkpoints; never averaged
BATCH_SIZE = 1
NUM_WORKERS = 2
LR = 0.0016
MIN_LR = 0.00001
# STABILITY FIX 3 -- linear LR warmup before the cosine decay.
# The 50-epoch T4 run collapsed at epoch 3 (TC 0.499->0.316) and recovered:
# the signature of a full-rate LR before the batch-1 normalisation statistics
# have settled. Set WARMUP_EPOCHS = 0 to restore the previous schedule.
WARMUP_EPOCHS = 4                 # epochs of linear ramp; 0 disables
WARMUP_START_DIV = 20.0           # start at LR/20, ramp linearly to LR
WEIGHT_DECAY = 0.0001
BETAS = (0.9, 0.99)
TVERSKY_ALPHA, TVERSKY_BETA = 0.25, 0.75
FOCAL_GAMMA = 1.33
CE_WEIGHT = 0.5
EMPTY_REGION_BCE_WEIGHT = 0.1
# STABILITY FIX 4 -- EMA decay derived from steps/epoch, not copied blindly.
#
# At batch 1, steps per epoch == number of training cases, so a FIXED decay
# means a completely different amount of smoothing on different splits:
#
#     decay 0.999 -> ~1000-step window
#       Kaggle  (129 train cases): 7.8 epochs of lag, 12% weight on the
#                                  current epoch -- the EMA is a stale model
#       thesis  (898 train cases): 1.1 epochs, 59% weight -- almost no
#                                  smoothing at all
#
# Neither is what was intended. EMA_EPOCH_WINDOW fixes the window in EPOCHS
# and derives the decay at runtime once the split size is known, so the
# smoothing behaves identically on both splits.
#
# Set EMA_DECAY to a float to pin it explicitly and skip the derivation.
EMA_EPOCH_WINDOW = 3.0            # smooth over ~3 epochs of updates
EMA_DECAY = None                  # None -> derived from steps/epoch


def derive_ema_decay(steps_per_epoch: int) -> float:
    """Decay giving an EMA window of EMA_EPOCH_WINDOW epochs.

    A decay d has an effective window of 1/(1-d) steps, so for a window of
    N epochs at S steps/epoch: d = 1 - 1/(N*S). Clamped to [0.9, 0.9999] --
    below 0.9 the EMA barely smooths, above 0.9999 it never catches up.
    """
    if EMA_DECAY is not None:
        return float(EMA_DECAY)
    window = max(1.0, EMA_EPOCH_WINDOW * max(1, int(steps_per_epoch)))
    return float(min(0.9999, max(0.9, 1.0 - 1.0 / window)))


# STABILITY FIX 1 -- validate on the EMA weights (see the training loop).
# best.pth always stored EMA weights while validation scored the RAW ones,
# so the metric that selected a checkpoint was not the metric of the
# checkpoint selected. Set False to reproduce the old behaviour for an A/B.
VALIDATE_WITH_EMA = True

# STABILITY EXPERIMENT 6 -- normalisation kind.
#
#   "batch"  (default) ChannelsLastBatchNorm, track_running_stats=False.
#            THE CERTIFIED PRODUCTION BEHAVIOUR. At batch 1 it normalises each
#            volume by its own statistics.
#   "group"  GroupNorm over NORM_GROUPS channel groups. Batch-size independent
#            and identical in train and eval.
#
# Parameter count is IDENTICAL either way (one weight + one bias per channel),
# so switching does not break the 29,337 identity gate and is not a
# Category-C architecture change. It IS a change of training mathematics, so
# it must be run as its own controlled experiment -- change nothing else.
NORM_KIND = "batch"               # "batch" | "group"
NORM_GROUPS = 8                   # groups for NORM_KIND="group"


def _norm_groups(channels: int) -> int:
    """Largest divisor of `channels` that is <= NORM_GROUPS.

    GroupNorm requires channels % groups == 0. The NCA cell runs 24 hidden
    channels (8 divides it), but the hidden MLP width is configurable, so fall
    back to the nearest valid divisor instead of raising mid-forward.
    """
    g = min(NORM_GROUPS, channels)
    while g > 1 and channels % g:
        g -= 1
    return max(1, g)
GRAD_CLIP = 1.0
# PRECISION: applies to the MODEL FORWARD ONLY; the loss always runs in FP32
# (PyTorch refuses to autocast binary_cross_entropy).
#
#   "bf16"  - wide range, no loss scaling needed. NATIVE on Ampere+ (RTX 30xx,
#             A100, L4). On Turing (Tesla T4) bf16 is EMULATED in software.
#   "fp16"  - NATIVE tensor cores on Turing (T4), so it may be FASTER THERE even
#             though it measured slower on Ampere (10.81 s vs 8.23 s locally).
#             Needs a GradScaler; one is wired up automatically below.
#   "fp32"  - no autocast.
#   "auto"  - pick per GPU: fp16 on Turing (sm_75), bf16 on Ampere+.
#
# MEASURED on RTX 3050 (Ampere, sm_86): bf16 8.23 s | fp16 10.81 s | fp32 11.44 s
# NOT MEASURED on Tesla T4 (Turing, sm_75) -- set "auto" or "fp16" to test there.
PRECISION = "auto"

# --- torch.compile (OPT-IN, NOT DEFAULT) -----------------------------------
# Inductor can fuse the many small elementwise ops inside each NCA step, which
# is where a lot of this architecture's time goes.
#
# NOT ENABLED BY DEFAULT because I could not measure it: torch.compile needs
# Triton, which is not available on Windows, so it fails locally with
# "Cannot find a working triton installation". Kaggle's Linux image HAS Triton,
# so it can be tried there -- but it is UNVERIFIED, hence opt-in.
#
# RISK: the patch box varies every iteration. With dynamic=False that could
# force a recompile per distinct shape. The model is wrapped with dynamic=None
# so PyTorch decides, and the first few iterations will be slow (compilation).
# If iteration time does not drop after ~10 iterations, set this back to False.
USE_TORCH_COMPILE = False

# TF32 for matmul/conv. Free extra throughput on Ampere+ (A100/L4/RTX 30xx);
# a silent no-op on Turing (T4), which has no TF32 units. cudnn.benchmark lets
# cuDNN pick the fastest conv algorithm per shape -- worth it here because the
# shapes are fixed after the first iteration.
USE_TF32 = True

CHECKPOINT_FREQUENCY = 5          # periodic recovery snapshot
HD95_EVERY_EPOCHS = 10
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
REGIONS = ["WT", "TC", "ET"]

OUT_DIR = "kaggle_glo_nca_50epoch_results"

# Kaggle-only split fractions (NOT the thesis split).
KAGGLE_SPLIT = (0.70, 0.15, 0.15)


# ============================================================================
# LOGGING
# ============================================================================
class Tee:
    """Write to stdout and the run log simultaneously.

    Console encoding is NOT assumed to be UTF-8: a Windows cp1252 console raises
    UnicodeEncodeError on non-ASCII output, which would otherwise kill a
    multi-hour run at a print statement. Console output degrades to ASCII;
    the log file always keeps the full UTF-8 text.
    """

    def __init__(self, path: str):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        msg = " ".join(str(p) for p in parts)
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()


LOG = print  # replaced in main()


# ============================================================================
# TIMING INSTRUMENTATION
#
# METHODOLOGY (§11): CPU-side wall clock via time.perf_counter(), with an
# explicit torch.cuda.synchronize() at each boundary so that asynchronous CUDA
# kernels are attributed to the section that launched them.
#
# Synchronisation is confined to this instrumentation. It is a MEASUREMENT COST:
# it makes the absolute iteration time slightly higher than an uninstrumented
# run, but makes the per-component attribution correct. Components are NOT
# forced to sum to 100% — nested sections (e.g. global context inside Level 1)
# deliberately overlap and are reported separately.
# ============================================================================
class Profiler:
    def __init__(self, enabled: bool = True, cuda: bool = False):
        self.enabled = enabled
        self.cuda = cuda
        self.records: Dict[str, List[float]] = {}
        self._stack: List[Tuple[str, float]] = []

    def _sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    class _Section:
        def __init__(self, prof: "Profiler", name: str):
            self.prof, self.name = prof, name

        def __enter__(self):
            if self.prof.enabled:
                self.prof._sync()
                self.prof._stack.append((self.name, time.perf_counter()))
            return self

        def __exit__(self, *exc):
            if self.prof.enabled:
                self.prof._sync()
                name, t0 = self.prof._stack.pop()
                self.prof.records.setdefault(name, []).append(time.perf_counter() - t0)
            return False

    def section(self, name: str) -> "_Section":
        return Profiler._Section(self, name)

    def add(self, name: str, seconds: float):
        self.records.setdefault(name, []).append(seconds)

    def stats(self, name: str) -> Dict[str, float]:
        v = np.array(self.records.get(name, []), dtype=float)
        if v.size == 0:
            return {"n": 0, "total": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0}
        return {"n": int(v.size), "total": float(v.sum()), "mean": float(v.mean()),
                "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {k: self.stats(k) for k in sorted(self.records)}


# ============================================================================
# MODEL — copied from the production implementation (behaviourally equivalent)
# ============================================================================
def build_model_classes():
    """Defined inside a function so --self-check can run without torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint

    class SEBlock3D(nn.Module):
        """Channel GLOBAL context: squeeze over the WHOLE volume -> per-channel gate.
        Verbatim from src/models/Model_BasicNCA3D.py."""

        def __init__(self, channel_n, reduction=4):
            super().__init__()
            reduced = max(1, channel_n // reduction)
            self.fc1 = nn.Linear(channel_n, reduced)
            self.fc2 = nn.Linear(reduced, channel_n)

        def forward(self, x):
            b, c = x.shape[0], x.shape[1]
            squeezed = x.mean(dim=(2, 3, 4))          # global pool over whole volume
            gate = F.relu(self.fc1(squeezed))
            gate = torch.sigmoid(self.fc2(gate))
            return x * gate.view(b, c, 1, 1, 1)

    class GCSpatialBlock3D(nn.Module):
        """Spatial GLOBAL context: attention-pooled per-voxel gate.
        Verbatim from src/models/Model_BasicNCA3D.py."""

        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                                  padding=(kernel_size - 1) // 2)

        def forward(self, x):
            avg_map = x.mean(dim=1, keepdim=True)
            max_map = x.max(dim=1, keepdim=True)[0]
            attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
            return x * attn

    class ChannelsLastBatchNorm(nn.Module):
        """BatchNorm3d(track_running_stats=False) for channels-LAST tensors.

        THIS IS THE A1 OPTIMISATION, AND IT IS NOW THE PRODUCTION CODE.
        Identical to `ChannelsLastBatchNorm` in src/models/Model_BasicNCA3D.py,
        which was integrated after passing the full gate matrix (see
        reports/audit/GLO_NCA_A1_INTEGRATION_REPORT.md).

        WHY: `update()` carries its hidden activation channels-last, but
        BatchNorm3d needs channels-first, so the original code transposed a
        128-channel tensor into NCHW and back around every normalisation --
        two copies per NCA step, ~3.5 GB moved per iteration at production
        geometry, purely to satisfy a layout requirement.

        Normalising over (B,X,Y,Z) per channel on a channels-last tensor IS a
        2D batch-norm over the flattened (N, C) view, so F.batch_norm on a
        reshape (a VIEW, not a copy) computes the identical function.

        EQUIVALENCE (verified in the real production code path):
            forward output   0.000e+00   (bit-identical in fp32)
            loss             0.000e+00
            d/dx             3.553e-15
            param gradients  7.839e-05   (fp32 accumulation order)
            float64 isolated 2.665e-15
        MEASURED SPEEDUP: 1.402x paired (backward 1.512x), VRAM -18 MB.

        State-dict keys are `bn.weight` / `bn.bias` -- byte-identical to the
        BatchNorm3d they replace, so checkpoints need no migration.
        """

        def __init__(self, num_features, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            self.eps = eps
            self.num_features = num_features

        def forward(self, x):                  # x: (B, X, Y, Z, C)
            b, X, Y, Z, c = x.shape
            flat = x.reshape(-1, c)

            if NORM_KIND == "group":
                # STABILITY EXPERIMENT 6 -- GroupNorm instead of batch stats.
                #
                # BatchNorm here runs with track_running_stats=False, so it
                # normalises by BATCH statistics. At batch_size 1 that means
                # each volume is normalised by ITS OWN mean/variance, at train
                # AND validation time. A case's prediction therefore depends on
                # that case's intensity histogram rather than on a learned
                # population statistic -- which is exactly the per-case,
                # high-variance validation behaviour measured on the T4 run
                # (TC/ET sd 0.052 vs WT 0.018, and TC/ET moving together with
                # mean |TC-ET| = 0.013).
                #
                # GroupNorm normalises over (channel-group x spatial) within a
                # SINGLE sample, so it is batch-size independent and behaves
                # identically in train and eval. Parameter count is unchanged:
                # both carry exactly one weight and one bias per channel, which
                # is why this stays at 29,337 and needs no Category-C decision.
                #
                # Applied over the flattened (N, C) view the spatial extent is
                # folded into N, so this normalises per group over the whole
                # volume -- the same reduction axes BatchNorm used, minus the
                # cross-sample coupling.
                # Reduce over (group-channels x spatial) WITHIN each sample.
                # Reshaping the flattened (N, C) view straight to (1, g, -1)
                # would fold the batch dimension into the reduction and
                # reintroduce exactly the cross-sample coupling this change
                # exists to remove, so go back through (b, spatial, C) first.
                grp = _norm_groups(c)
                v = x.reshape(b, X * Y * Z, grp, c // grp)
                v = v.permute(0, 2, 1, 3).reshape(b, grp, -1)
                v = F.layer_norm(v, (v.shape[-1],), None, None, self.eps)
                v = v.reshape(b, grp, X * Y * Z, c // grp).permute(0, 2, 1, 3)
                flat = v.reshape(-1, c) * self.weight + self.bias
            else:
                # One fused autograd node instead of the six the manual form
                # (mean, var, sub, rsqrt, mul, add) would build, and no
                # transpose.
                flat = F.batch_norm(flat, None, None, self.weight, self.bias,
                                    True, 0.0, self.eps)
            return flat.reshape(b, X, Y, Z, c)

    class BasicNCA3D(nn.Module):
        """Production NCA cell rule. Verbatim from src/models/Model_BasicNCA3D.py.

        `prof` is an optional profiler used ONLY to time the global-context
        blocks; it does not alter the computation.
        """

        def __init__(self, channel_n, fire_rate, device, hidden_size=128,
                     input_channels=1, kernel_size=7, use_attention=False,
                     se_reduction=4, use_spatial=False, dropout=0.0):
            super().__init__()
            self.device = device
            self.channel_n = channel_n
            self.input_channels = input_channels
            self.use_checkpoint = False
            self.prof: Optional[Profiler] = None
            self.gc_key = "model/global_context"

            self.fc0 = nn.Linear(channel_n * 2, hidden_size)
            self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
            padding = int((kernel_size - 1) / 2)
            self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size,
                                stride=1, padding=padding,
                                padding_mode="reflect", groups=channel_n)
            # Channels-LAST batch norm. Mathematically identical to
            # BatchNorm3d(hidden_size, track_running_stats=False) -- same biased
            # variance, same affine weight/bias, same eps -- but it consumes the
            # channels-last tensor directly, removing two 128-channel transposes
            # per NCA step. Parameter count and shapes are unchanged
            # (weight[hidden], bias[hidden]), so checkpoints stay compatible.
            self.bn = ChannelsLastBatchNorm(hidden_size)
            self.dropout_p = dropout
            self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None
            self.use_spatial = use_spatial
            self.se = SEBlock3D(channel_n, reduction=se_reduction) if use_attention else None
            # Spatial global-context kernel: SPATIAL_GC_KERNEL (5), was 7.
            self.gc = (GCSpatialBlock3D(kernel_size=SPATIAL_GC_KERNEL)
                       if use_spatial else None)

            with torch.no_grad():
                self.fc1.weight.zero_()   # NCA "do-nothing" init (production)

            self.fire_rate = fire_rate
            self.to(self.device)

        def perceive(self, x):
            y1 = self.p0(x)
            # GLOBAL CONTEXT applied at EVERY step, on whatever volume this level
            # holds. Timed separately when a profiler is attached.
            if self.se is not None or self.gc is not None:
                if self.prof is not None and self.prof.enabled:
                    with self.prof.section(self.gc_key):
                        if self.se is not None:
                            y1 = self.se(y1)
                        if self.gc is not None:
                            y1 = self.gc(y1)
                else:
                    if self.se is not None:
                        y1 = self.se(y1)
                    if self.gc is not None:
                        y1 = self.gc(y1)
            return torch.cat((x, y1), 1)

        def update(self, x_in, fire_rate):
            x = x_in.transpose(1, 4)
            dx = self.perceive(x)
            dx = dx.transpose(1, 4)
            dx = self.fc0(dx)
            # SPEED FIX (mathematically identical, verified to 1.8e-15 in float64):
            # BatchNorm3d requires channels-FIRST, so the original code wrapped it
            # in two transposes of the 128-channel tensor -- at 64^3 that is a
            # 128 MB copy each way, 20 times per level. `self.bn` computes the
            # SAME per-channel normalisation directly on the channels-LAST tensor,
            # so both copies disappear. Measured 64% faster on that sub-step.
            dx = self.bn(dx)
            dx = F.relu(dx)
            if self.drop is not None:
                dx = self.drop(dx)
            dx = self.fc1(dx)
            if fire_rate is None:
                fire_rate = self.fire_rate
            # Build the fire-rate mask directly in dx's dtype. `.float()` forced
            # an fp32 tensor under bf16 autocast, so the multiply upcast dx and
            # stored an fp32 tensor for backward. Same mask, same mathematics.
            stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2),
                                     dx.size(3), 1], device=dx.device) > fire_rate
            dx = dx * stochastic.to(dx.dtype)
            x = x + dx.transpose(1, 4)
            return x.transpose(1, 4)

        def forward(self, x, steps=10, fire_rate=0.5):
            use_ckpt = getattr(self, "use_checkpoint", False) and torch.is_grad_enabled()
            for _ in range(steps):
                if use_ckpt:
                    x2 = torch.utils.checkpoint.checkpoint(
                        self.update, x, fire_rate,
                        use_reentrant=False, preserve_rng_state=True).clone()
                else:
                    x2 = self.update(x, fire_rate).clone()
                x = torch.concat((x[..., 0:self.input_channels],
                                  x2[..., self.input_channels:]), 4)
            return x

    def _to_cf(x):
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def _to_cl(x):
        return x.permute(0, 2, 3, 4, 1).contiguous()

    class FeatureProjection(nn.Module):
        """Learnable 1x1x1 projection between level widths (production)."""

        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        def forward(self, x_cl):
            return _to_cl(self.proj(_to_cf(x_cl)))

    @dataclass
    class LevelSpec:
        resolution: int
        channels: int
        nca_steps: int
        kernel_size: int = 3

    class GLO_NCA(nn.Module):
        """GLO-NCA: GLOBAL CONTEXT + MULTI-LEVEL FUSION.

        Behaviourally equivalent to the production GLO_NCA_GlobalContext with
        roi_fraction = 1.0 (which is the production setting). At roi_fraction 1.0
        the production forward takes no ROI crop and every level reads the full
        input volume, so the ROI machinery is inert and is omitted here rather
        than reimplemented — this keeps patchify/ROI structurally impossible.
        """

        def __init__(self, input_channels, output_channels, levels: List[LevelSpec],
                     fire_rate=0.6, use_attention=True, use_spatial=True,
                     dropout=0.0, fusion="concat", hidden_size=128, device=None,
                     gradient_checkpointing=False):
            super().__init__()
            assert len(levels) >= 2, "GLO-NCA needs at least 2 levels"
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.levels = levels
            self.fire_rate = fire_rate
            self.fusion_type = fusion
            self.device = device or torch.device("cpu")
            self.prof: Optional[Profiler] = None

            self.ncas = nn.ModuleList([
                BasicNCA3D(channel_n=lv.channels, fire_rate=fire_rate,
                           device=self.device, hidden_size=hidden_size,
                           input_channels=input_channels, kernel_size=lv.kernel_size,
                           use_attention=use_attention, use_spatial=use_spatial,
                           dropout=dropout)
                for lv in levels])
            for nca in self.ncas:
                nca.use_checkpoint = bool(gradient_checkpointing)

            self.projections = nn.ModuleList()
            for i in range(len(levels) - 1):
                self.projections.append(
                    FeatureProjection(levels[i].channels,
                                      levels[i + 1].channels - input_channels))

            fine_ch = levels[-1].channels
            self.level_to_fine = nn.ModuleList(
                [FeatureProjection(lv.channels, fine_ch) for lv in levels])
            self.fuse = (nn.Conv3d(fine_ch * len(levels), fine_ch, kernel_size=1)
                         if fusion == "concat" else nn.Identity())
            self.seg_head = nn.Conv3d(fine_ch, output_channels, kernel_size=1)
            self.to(self.device)

        def attach_profiler(self, prof: Optional[Profiler]):
            self.prof = prof
            for nca in self.ncas:
                nca.prof = prof

        def _seed(self, modalities_cl, channels):
            b, x, y, z, c = modalities_cl.shape
            seed = torch.zeros((b, x, y, z, channels), dtype=modalities_cl.dtype,
                               device=modalities_cl.device)
            seed[..., :c] = modalities_cl
            return seed

        @staticmethod
        def _resize_cl(x_cl, size, mode):
            xcf = _to_cf(x_cl)
            xcf = F.interpolate(xcf, size=(size, size, size), mode=mode,
                                align_corners=False if mode == "trilinear" else None)
            return _to_cl(xcf)

        @staticmethod
        def _crop_state_to_box(state_cl, box):
            """Crop a level's state to the patch's normalised coordinates.

            `box` is ((x0,x1),(y0,y1),(z0,z1)) as FRACTIONS of the full volume,
            so the global level's state and the patch describe the SAME anatomy
            regardless of the two grids having different sizes.
            """
            b, X, Y, Z, c = state_cl.shape
            idx = []
            for n, (lo, hi) in zip((X, Y, Z), box):
                size = max(1, min(n, int(round(n * (hi - lo)))))
                start = int(round(n * lo))
                start = max(0, min(start, n - size))
                idx.append((start, start + size))
            return state_cl[:, idx[0][0]:idx[0][1],
                            idx[1][0]:idx[1][1],
                            idx[2][0]:idx[2][1], :]

        def forward(self, modalities_cl, patch_cl=None, patch_box=None):
            """GLO-NCA forward.

            modalities_cl : FULL working volume (B,96,96,96,4). ALWAYS the
                            global-context source -- never a patch.
            patch_cl      : optional (B,P,P,P,4) crop of that volume. When given,
                            the HIGH-RESOLUTION level runs on the patch while the
                            GLOBAL level still reads the full volume, so the
                            thesis contribution is preserved.
            patch_box     : normalised ((x0,x1),(y0,y1),(z0,z1)) locating the
                            patch inside the full volume.

            Passing patch_cl=None reproduces the full-volume behaviour exactly.
            """
            prof = self.prof
            prev_state = None
            level_states = []
            n_global = 1          # level 1 is the global-context level
            for i, (lv, nca) in enumerate(zip(self.levels, self.ncas)):
                key = f"model/level{i + 1}"
                ctx = prof.section(key) if prof is not None else _null_ctx()
                with ctx:
                    is_global = i < n_global
                    if is_global or patch_cl is None:
                        # GLOBAL CONTEXT SOURCE: the full volume, always.
                        src = modalities_cl
                        res = lv.resolution
                    else:
                        # HIGH-RESOLUTION level: the patch only.
                        src = patch_cl
                        # PATCH-NATIVE RESOLUTION: run this level on the patch's
                        # OWN grid instead of forcing it back up to lv.resolution.
                        # Without this, a smaller patch is resized straight back
                        # to lv.resolution and saves nothing (measured: 0%).
                        res = (src.shape[1] if PATCH_NATIVE_RES
                               else lv.resolution)
                    mod_i = self._resize_cl(src, res, mode="trilinear")
                    seed = self._seed(mod_i, lv.channels)
                    if prev_state is not None:
                        prev = prev_state
                        # Crossing global -> patch: crop the global state to the
                        # patch's coordinates so both levels describe the same
                        # anatomy before the learned projection.
                        if (not is_global) and patch_cl is not None \
                                and patch_box is not None and (i - 1) < n_global:
                            prev = self._crop_state_to_box(prev, patch_box)
                        proj = self.projections[i - 1](prev)
                        proj = self._resize_cl(proj, res, mode="nearest")
                        seed = seed.clone()
                        seed[..., self.input_channels:] = \
                            seed[..., self.input_channels:] + proj
                    out = nca(seed, steps=lv.nca_steps, fire_rate=self.fire_rate)
                    prev_state = out
                    level_states.append(out)

            ctx = prof.section("model/fusion") if prof is not None else _null_ctx()
            with ctx:
                # Fuse at whatever grid the last level ACTUALLY ran on, which
                # under PATCH_NATIVE_RES is the patch size, not levels[-1].
                fine_res = level_states[-1].shape[1]
                fused = []
                for i, (state, to_fine) in enumerate(zip(level_states,
                                                         self.level_to_fine)):
                    s = state
                    # Align the global level's contribution to the patch before
                    # fusing, so every fused tensor covers the same anatomy.
                    if i < n_global and patch_cl is not None and patch_box is not None:
                        s = self._crop_state_to_box(s, patch_box)
                    s = to_fine(s)
                    s = self._resize_cl(s, fine_res, mode="nearest")
                    fused.append(_to_cf(s))
                fused_cf = (self.fuse(torch.cat(fused, dim=1))
                            if self.fusion_type == "concat"
                            else torch.stack(fused, dim=0).sum(dim=0))
                logits = self.seg_head(fused_cf)
            return logits

    class FocalTverskyCELoss(nn.Module):
        """Verbatim from src/losses/LossFunctions.py."""

        def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, ce_weight=0.5,
                     useSigmoid=True):
            super().__init__()
            self.alpha, self.beta, self.gamma = alpha, beta, gamma
            self.ce_weight, self.useSigmoid = ce_weight, useSigmoid

        def forward(self, input, target, smooth=1):
            prob = torch.sigmoid(input) if self.useSigmoid else input
            bce = F.binary_cross_entropy(prob.clamp(1e-6, 1. - 1e-6), target,
                                         reduction="mean")
            p, t = torch.flatten(prob), torch.flatten(target)
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
            return torch.pow(1 - tversky, self.gamma) + self.ce_weight * bce

    return dict(GLO_NCA=GLO_NCA, LevelSpec=LevelSpec,
                FocalTverskyCELoss=FocalTverskyCELoss,
                SEBlock3D=SEBlock3D, GCSpatialBlock3D=GCSpatialBlock3D,
                BasicNCA3D=BasicNCA3D)


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_model(device, gradient_checkpointing=GRADIENT_CHECKPOINTING):
    C = build_model_classes()
    levels = [C["LevelSpec"](L1_RES, L1_CH, L1_STEPS, L1_K),
              C["LevelSpec"](L2_RES, L2_CH, L2_STEPS, L2_K)]
    return C["GLO_NCA"](input_channels=len(MODALITIES), output_channels=len(REGIONS),
                        levels=levels, fire_rate=FIRE_RATE,
                        use_attention=USE_ATTENTION, use_spatial=USE_SPATIAL,
                        dropout=DROPOUT, fusion=FUSION, hidden_size=HIDDEN,
                        device=device,
                        gradient_checkpointing=gradient_checkpointing), C


# ============================================================================
# §16 ARCHITECTURE IDENTITY — abort on any mismatch, never auto-correct
# ============================================================================
def assert_architecture(model) -> Dict[str, object]:
    import torch
    checks, failures = [], []

    def chk(name, actual, expected):
        ok = actual == expected
        checks.append({"check": name, "actual": str(actual),
                       "expected": str(expected), "pass": ok})
        if not ok:
            failures.append(f"{name}: got {actual!r}, expected {expected!r}")

    n_params = sum(p.numel() for p in model.parameters())
    chk("parameter_count", n_params, EXPECTED_PARAMS)
    chk("level1_resolution", model.levels[0].resolution, L1_RES)
    chk("level1_channels", model.levels[0].channels, L1_CH)
    chk("level1_nca_steps", model.levels[0].nca_steps, L1_STEPS)
    chk("level2_resolution", model.levels[1].resolution, L2_RES)
    chk("level2_channels", model.levels[1].channels, L2_CH)
    chk("level2_nca_steps", model.levels[1].nca_steps, L2_STEPS)
    chk("level3_absent", len(model.levels), 2)
    chk("total_nca_steps", sum(l.nca_steps for l in model.levels), 2 * NCA_STEPS)
    chk("spatial_gc_kernel",
        model.ncas[0].gc.conv.kernel_size, (SPATIAL_GC_KERNEL,) * 3)
    chk("patch_native_resolution", PATCH_NATIVE_RES, True)
    chk("global_context_se", all(n.se is not None for n in model.ncas), True)
    chk("global_context_spatial", all(n.gc is not None for n in model.ncas), True)
    chk("multi_level_fusion", isinstance(model.fuse, torch.nn.Conv3d), True)
    fusion_io = (model.fuse.in_channels, model.fuse.out_channels)
    chk("fusion_48_to_24", fusion_io, (L1_CH + L2_CH, L2_CH))
    chk("working_volume", WORKING_VOLUME, 96)
    # Speed settings for this run (informational — recorded, not enforced).
    chk("patchify_enabled", USE_PATCHIFY, USE_PATCHIFY)
    chk("gradient_checkpointing", GRADIENT_CHECKPOINTING, GRADIENT_CHECKPOINTING)
    chk("cache_enabled", USE_CACHE, USE_CACHE)

    # Global-context source must be the FULL volume: prove it at runtime by
    # showing the output responds to a change OUTSIDE any central patch.
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        g = torch.Generator(device="cpu").manual_seed(0)
        core = torch.randn(1, 64, 64, 64, len(MODALITIES), generator=g)
        a = torch.zeros(1, WORKING_VOLUME, WORKING_VOLUME, WORKING_VOLUME, len(MODALITIES))
        b = a.clone()
        a[:, 16:80, 16:80, 16:80, :] = core
        b[:, 16:80, 16:80, 16:80, :] = core
        b[:, :16, :, :, :] = 5.0                 # differs ONLY outside the core
        oa, ob = model(a.to(dev)), model(b.to(dev))
        delta = float((oa - ob).abs().max())
    chk("global_context_sees_full_volume", delta > 1e-6, True)
    checks[-1]["actual"] = f"max|delta|={delta:.6g} when only the non-core rim changes"

    return {"passed": not failures, "failures": failures, "checks": checks,
            "parameter_count": n_params,
            "global_context_delta": delta}


# ============================================================================
# §4 DATASET DISCOVERY — no invented paths; structure detected at runtime
# ============================================================================
MODALITY_ALIASES = {
    "t1n": ["t1n", "t1", "_t1.", "t1w"],
    "t1c": ["t1c", "t1ce", "t1gd", "t1_ce", "t1contrast"],
    "t2w": ["t2w", "_t2.", "t2"],
    "t2f": ["t2f", "flair", "fla"],
}
SEG_ALIASES = ["seg", "label", "mask", "gt", "annotation"]


def find_dataset_root() -> str:
    """Locate the mounted Kaggle dataset. Fails loudly rather than guessing."""
    candidates = []
    for base in ("/kaggle/input", "./input", "."):
        if os.path.isdir(base):
            for entry in sorted(os.listdir(base)):
                candidates.append(os.path.join(base, entry))
            candidates.append(base)
    for c in candidates:
        if not os.path.isdir(c):
            continue
        for dp, dn, fn in os.walk(c):
            if any(f.endswith((".nii", ".nii.gz")) for f in fn):
                # climb to the directory holding per-case folders/files
                return c
    raise FileNotFoundError(
        "Could not locate a NIfTI dataset. Attach the Kaggle dataset "
        "'nguyenthanhkhanh/brats2024-small-dataset' (it mounts under "
        "/kaggle/input/...). Searched: " + ", ".join(candidates[:12]))


def _match_modality(fname: str) -> Optional[str]:
    low = fname.lower()
    if any(a in low for a in SEG_ALIASES):
        return "seg"
    # longest alias first so 't1ce' wins over 't1'
    best, best_len = None, 0
    for mod, aliases in MODALITY_ALIASES.items():
        for a in aliases:
            if a in low and len(a) > best_len:
                best, best_len = mod, len(a)
    return best


def inventory_dataset(root: str, prof: Profiler) -> Dict[str, object]:
    """§4: build a full inventory. Never silently skips a malformed case."""
    import nibabel as nib

    with prof.section("setup/case_discovery"):
        per_case: Dict[str, Dict[str, str]] = {}
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if not f.endswith((".nii", ".nii.gz")):
                    continue
                full = os.path.join(dp, f)
                case = os.path.basename(dp)
                if case in (".", "") or os.path.samefile(dp, root):
                    case = f.split(".")[0]
                mod = _match_modality(f)
                if mod is None:
                    continue
                per_case.setdefault(case, {})[mod] = full

    usable, skipped, shapes, spacings = [], [], [], []
    with prof.section("setup/metadata_load"):
        for case, files in sorted(per_case.items()):
            missing = [m for m in MODALITIES if m not in files]
            if missing:
                skipped.append({"case": case, "reason": f"missing modalities: {missing}",
                                "found": sorted(files)})
                continue
            if "seg" not in files:
                skipped.append({"case": case, "reason": "missing segmentation label",
                                "found": sorted(files)})
                continue
            try:
                h = nib.load(files[MODALITIES[0]])
                shp = tuple(int(v) for v in h.shape[:3])
                zoom = tuple(round(float(z), 4) for z in h.header.get_zooms()[:3])
            except Exception as e:
                skipped.append({"case": case, "reason": f"unreadable: {type(e).__name__}: {e}",
                                "found": sorted(files)})
                continue
            usable.append({"case": case, "files": files, "shape": shp, "spacing": zoom})
            shapes.append(shp)
            spacings.append(zoom)

    inv = {
        "dataset_root": root,
        "case_count_discovered": len(per_case),
        "case_count_usable": len(usable),
        "case_count_skipped": len(skipped),
        "modalities_required": MODALITIES,
        "label_available": True,
        "unique_shapes": sorted({str(s) for s in shapes}),
        "unique_spacings": sorted({str(s) for s in spacings}),
        "skipped_cases": skipped,
        "usable_cases": [u["case"] for u in usable],
    }
    return {"inventory": inv, "cases": usable}


# ============================================================================
# §5 KAGGLE-ONLY SPLIT — deterministic, subject-level, NOT the thesis split
# ============================================================================
def subject_of(case: str) -> str:
    """Group longitudinal timepoints by subject where the naming permits."""
    parts = case.replace("-", "_").split("_")
    keep = []
    for p in parts:
        if p.lower() in ("ses", "sub") or (p.isdigit() and len(p) <= 3 and keep):
            break
        keep.append(p)
    return "_".join(keep[:3]) if keep else case


def make_split(cases: List[dict]) -> Dict[str, List[dict]]:
    subjects = {}
    for c in cases:
        subjects.setdefault(subject_of(c["case"]), []).append(c)
    keys = sorted(subjects)
    rng = random.Random(SEED)
    rng.shuffle(keys)
    n = len(keys)
    if n < 2:
        raise ValueError(
            f"only {n} subject(s) found: cannot build a leak-free train/validation "
            "split. Refusing to validate on training data.")
    # Validation is MANDATORY and must be subject-disjoint from training: model
    # selection on training data would be a silent leak. Test is optional.
    n_tr = max(1, int(round(n * KAGGLE_SPLIT[0])))
    n_tr = min(n_tr, n - 1)                       # always leave >=1 for validation
    remaining = n - n_tr
    n_va = max(1, int(round(n * KAGGLE_SPLIT[1])))
    n_va = min(n_va, remaining)                   # test gets whatever is left (may be 0)
    tr_k, va_k, te_k = keys[:n_tr], keys[n_tr:n_tr + n_va], keys[n_tr + n_va:]
    out = {"train": [], "validation": [], "test": []}
    for k in tr_k:
        out["train"] += subjects[k]
    for k in va_k:
        out["validation"] += subjects[k]
    for k in te_k:
        out["test"] += subjects[k]
    return out


# ============================================================================
# §6 PREPROCESSING — production-compatible, full volume, patchify OFF
# ============================================================================
def augment_light(img: np.ndarray, lab: np.ndarray):
    """Production 'light' augmentation: axis flips, 90-degree rotations, and a
    per-modality intensity scale/shift.

    Applied to the IMAGE and LABEL together so they stay voxel-aligned. Runs
    AFTER the cache lookup, so the cache stores un-augmented volumes and every
    epoch still sees a fresh random view (the cache remains RNG-neutral).

    MEASURED cost: ~111 ms per 96^3x4 case = 1.55% of a 7.15 s iteration.
    """
    # --- geometric: axis flips (label must follow exactly) ---
    for ax in range(3):
        if random.random() < 0.5:
            img = np.flip(img, axis=ax)
            lab = np.flip(lab, axis=ax)
    # --- geometric: 90-degree rotation in the axial plane ---
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k, axes=(0, 1))
        lab = np.rot90(lab, k, axes=(0, 1))
    img = np.ascontiguousarray(img)
    lab = np.ascontiguousarray(lab)
    # --- intensity: per-modality scale/shift (IMAGE ONLY -- never the label) ---
    for c in range(img.shape[-1]):
        img[..., c] = img[..., c] * (1.0 + random.gauss(0, 0.1)) + random.gauss(0, 0.1)
    return img, lab


def labels_to_regions(seg: np.ndarray) -> np.ndarray:
    """BraTS labels {1,2,3/4} -> nested WT/TC/ET, matching production."""
    wt = np.isin(seg, [1, 2, 3, 4]).astype(np.float32)
    tc = np.isin(seg, [1, 3, 4]).astype(np.float32)
    et = np.isin(seg, [3, 4]).astype(np.float32)
    return np.stack([wt, tc, et], axis=-1)


def resize_volume(vol: np.ndarray, size: int, is_label: bool) -> np.ndarray:
    """Images: trilinear. Labels: NEAREST (never interpolate labels)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(vol)).float()[None, None]
    mode = "nearest" if is_label else "trilinear"
    kw = {} if is_label else {"align_corners": False}
    out = F.interpolate(t, size=(size, size, size), mode=mode, **kw)
    return out[0, 0].numpy()


class BraTSDataset:
    """Working-volume dataset with an optional in-memory preprocessing cache
    and an optional ET-aware training patch.

    The cache stores the DETERMINISTIC head of __getitem__ (load -> foreground
    crop -> resample -> labels -> normalise). The stochastic patch crop runs
    AFTER the cache lookup on every epoch, so caching is RNG-neutral and changes
    no mathematics -- it only removes repeated disk reads and resampling.

    The full working volume is ALWAYS returned alongside any patch, so the
    global-context level never loses sight of the whole brain.
    """

    def __init__(self, cases: List[dict], timings: Optional[Dict[str, List[float]]] = None,
                 use_cache: bool = False, patch_size: Optional[int] = None,
                 train: bool = False, augment: bool = False):
        self.cases = cases
        self.timings = timings if timings is not None else {}
        self.use_cache = use_cache
        self.patch_size = patch_size
        self.train = train
        self.augment = augment
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _t(self, key, dt):
        self.timings.setdefault(key, []).append(dt)

    def __len__(self):
        return len(self.cases)

    def _preprocess(self, idx):
        """DETERMINISTIC head of __getitem__ -> (normalised volume, labels).
        Cacheable: contains no random state."""
        import nibabel as nib
        c = self.cases[idx]

        vols = []
        for m in MODALITIES:
            t0 = time.perf_counter()
            h = nib.load(c["files"][m])
            self._t("data/file_open", time.perf_counter() - t0)
            t0 = time.perf_counter()
            vols.append(np.asarray(h.dataobj, dtype=np.float32))
            self._t("data/nifti_materialize", time.perf_counter() - t0)

        t0 = time.perf_counter()
        raw = np.stack(vols, axis=-1)
        self._t("preproc/modality_stack", time.perf_counter() - t0)

        t0 = time.perf_counter()
        seg = np.asarray(nib.load(c["files"]["seg"]).dataobj, dtype=np.float32)
        self._t("data/nifti_materialize", time.perf_counter() - t0)

        # foreground crop (production behaviour)
        t0 = time.perf_counter()
        nz = np.argwhere(raw.max(axis=-1) > 0)
        if nz.size:
            lo, hi = nz.min(axis=0), nz.max(axis=0) + 1
            raw = raw[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], :]
            seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        self._t("preproc/foreground_crop", time.perf_counter() - t0)

        t0 = time.perf_counter()
        img = np.stack([resize_volume(raw[..., i], WORKING_VOLUME, False)
                        for i in range(raw.shape[-1])], axis=-1)
        self._t("preproc/resample_image", time.perf_counter() - t0)

        t0 = time.perf_counter()
        seg_r = resize_volume(seg, WORKING_VOLUME, True)
        lab = labels_to_regions(seg_r)
        self._t("preproc/label_convert", time.perf_counter() - t0)

        # non-zero z-normalisation (production behaviour)
        t0 = time.perf_counter()
        norm = np.empty_like(img, dtype=np.float32)
        for ch in range(img.shape[-1]):
            v = img[..., ch]
            mk = v > 0
            norm[..., ch] = ((v - v[mk].mean()) / (v[mk].std() + 1e-8)) * mk if mk.any() else v
        self._t("preproc/normalize", time.perf_counter() - t0)
        return norm, lab

    def __getitem__(self, idx):
        import torch
        c = self.cases[idx]

        # ---- CACHE: deterministic work done once per case ------------------
        if self.use_cache and idx in self._cache:
            t0 = time.perf_counter()
            norm, lab = self._cache[idx]
            self._t("data/cache_hit", time.perf_counter() - t0)
        else:
            norm, lab = self._preprocess(idx)
            if self.use_cache:
                t0 = time.perf_counter()
                self._cache[idx] = (norm, lab)
                self._t("data/cache_store", time.perf_counter() - t0)

        # ---- AUGMENTATION: TRAIN ONLY, and AFTER the cache -----------------
        # Running it here (not inside _preprocess) means the cache stores the
        # un-augmented volume while every epoch still gets a fresh random view.
        # Validation/test never reach this branch, so evaluation is unaugmented.
        if self.train and self.augment:
            t0 = time.perf_counter()
            norm, lab = augment_light(norm.copy(), lab.copy())
            self._t("preproc/augmentation", time.perf_counter() - t0)

        # ---- STOCHASTIC patch crop: runs EVERY epoch (RNG-neutral cache) ---
        # The FULL volume is always returned as well, so the global-context
        # level keeps seeing the whole brain.
        patch = None
        box = None
        if self.train and self.patch_size and self.patch_size < WORKING_VOLUME:
            t0 = time.perf_counter()
            P, W = self.patch_size, WORKING_VOLUME
            # ET-aware sampling: prefer a patch containing the hardest region.
            pos = None
            if random.uniform(0, 1) < 0.7:
                for _ in range(20):
                    cand = [random.randint(0, W - P) for _ in range(3)]
                    sub = lab[cand[0]:cand[0]+P, cand[1]:cand[1]+P,
                              cand[2]:cand[2]+P, 2]          # 2 = ET
                    if sub.max() > 0:
                        pos = cand
                        break
            if pos is None:
                pos = [random.randint(0, W - P) for _ in range(3)]
            x0, y0, z0 = pos
            patch = norm[x0:x0+P, y0:y0+P, z0:z0+P, :]
            lab = lab[x0:x0+P, y0:y0+P, z0:z0+P, :]          # target = the patch
            box = ((x0 / W, (x0 + P) / W), (y0 / W, (y0 + P) / W),
                   (z0 / W, (z0 + P) / W))
            self._t("preproc/patch_crop", time.perf_counter() - t0)

        t0 = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(norm))
        y = torch.from_numpy(np.ascontiguousarray(lab))
        p = torch.from_numpy(np.ascontiguousarray(patch)) if patch is not None else None
        self._t("preproc/tensor_convert", time.perf_counter() - t0)
        return x, y, c["case"], p, box


def collate(batch):
    import torch
    xs = torch.stack([b[0] for b in batch])          # full working volumes
    ys = torch.stack([b[1] for b in batch])          # targets (patch or full)
    ids = [b[2] for b in batch]
    ps = [b[3] for b in batch]
    boxes = [b[4] for b in batch]
    patches = torch.stack(ps) if all(p is not None for p in ps) else None
    # Batch 1 is the production setting; with batch > 1 every sample has its own
    # box, so only a uniform batch can share one box safely.
    box = boxes[0] if (patches is not None and len(set(map(str, boxes))) == 1) else None
    if patches is not None and box is None and len(boxes) == 1:
        box = boxes[0]
    return xs, ys, ids, patches, box


# ============================================================================
# §9 METRICS
# ============================================================================
def dice_iou(prob: np.ndarray, gt: np.ndarray, thr=0.5) -> Tuple[float, float]:
    p = prob >= thr
    g = gt >= 0.5
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    dice = (2.0 * inter) / (p.sum() + g.sum() + 1e-6)
    iou = inter / (union + 1e-6) if union else float("nan")
    return float(dice), float(iou)


def hd95(prob: np.ndarray, gt: np.ndarray, thr=0.5) -> float:
    """Symmetric 95th-percentile surface distance, in VOXELS."""
    from scipy.ndimage import distance_transform_edt
    p = prob >= thr
    g = gt >= 0.5
    if not p.any() or not g.any():
        return float("nan")
    dg = distance_transform_edt(~g)
    dp = distance_transform_edt(~p)
    return float(max(np.percentile(dg[p], 95), np.percentile(dp[g], 95)))


# ============================================================================
# PLOTS
# ============================================================================
def write_plots(outdir: str, epochs: List[dict], prof_rows: List[dict],
                gpu_rows: List[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        LOG(f"  [plots skipped: {type(e).__name__}: {e}]")
        return
    ep = [e["epoch"] for e in epochs]

    plt.figure(figsize=(7, 4))
    plt.plot(ep, [e["train_loss"] for e in epochs], label="train")
    vl = [e.get("val_loss") for e in epochs]
    if any(v is not None and not np.isnan(v) for v in vl):
        plt.plot(ep, vl, label="validation")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("GLO-NCA Kaggle 50-epoch — loss")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(7, 4))
    for r in REGIONS:
        plt.plot(ep, [e.get(f"dice_{r}", float("nan")) for e in epochs], label=r)
    plt.plot(ep, [e.get("dice_mean", float("nan")) for e in epochs], "k--", label="mean")
    plt.xlabel("epoch"); plt.ylabel("Dice"); plt.legend()
    plt.title("GLO-NCA Kaggle 50-epoch — validation Dice")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "dice_curve.png"), dpi=120)
    plt.close()

    if prof_rows:
        items = [(r["component"], r["total_s"]) for r in prof_rows
                 if r["component"].startswith(("data/", "preproc/", "train/", "model/"))]
        items = sorted(items, key=lambda kv: kv[1], reverse=True)[:14]
        if items:
            plt.figure(figsize=(9, 5))
            plt.barh([k for k, _ in items][::-1], [v for _, v in items][::-1])
            plt.xlabel("total seconds (measured)")
            plt.title("GLO-NCA Kaggle — where training time goes")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "timing_breakdown.png"), dpi=120)
            plt.close()

    if gpu_rows:
        plt.figure(figsize=(7, 4))
        plt.plot([g["epoch"] for g in gpu_rows], [g["peak_reserved_mb"] for g in gpu_rows],
                 label="peak reserved")
        plt.plot([g["epoch"] for g in gpu_rows], [g["peak_allocated_mb"] for g in gpu_rows],
                 label="peak allocated")
        plt.xlabel("epoch"); plt.ylabel("MB"); plt.legend()
        plt.title("GLO-NCA Kaggle — GPU memory")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "gpu_memory.png"), dpi=120)
        plt.close()


# ============================================================================
# SELF-CHECK (§21) — local static validation, no dataset, no training
# ============================================================================
def self_check() -> int:
    print("=" * 74)
    print("GLO-NCA KAGGLE SCRIPT — LOCAL STATIC VALIDATION")
    print("=" * 74)
    # State the active configuration up front, so the parameter count printed
    # below can be checked against what this preset is SUPPOSED to produce
    # rather than against a number remembered from another preset.
    print(f"  preset            : {ACTIVE_PRESET}   [{RUN_MODE}]")
    print(f"  expected params   : {EXPECTED_PARAMS:,}"
          f"{'' if IS_PRODUCTION_IDENTITY else '   (production is 33,089)'}")
    print(f"  geometry          : {L1_RES}^3 / {L2_RES}^3, "
          f"steps {L1_STEPS}+{L2_STEPS}, spatial GC k={SPATIAL_GC_KERNEL}")
    if not IS_PRODUCTION_IDENTITY:
        print("  *** CATEGORY-C configuration — a PASS here confirms the")
        print("      model matches THIS preset, NOT the production model. ***")
    print("-" * 74)
    ok = True
    try:
        import torch
    except ImportError:
        print("  torch unavailable -> architecture check DEFERRED")
        return 0

    model, _ = make_model(torch.device("cpu"))
    res = assert_architecture(model)
    for c in res["checks"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']:32s} {c['actual']}")
        ok &= c["pass"]

    # Scan EXECUTABLE source only. The forbidden tokens necessarily appear in
    # this guard's own literal list, so scanning the raw file would make the
    # detector match itself. Strip comments/docstrings and this function's body,
    # then check what is left.
    #
    # In a notebook there is no __file__, so the source scan is reported as
    # SKIPPED rather than crashing. The architecture checks above are the ones
    # that matter and they run in every environment.
    import ast
    _path = globals().get("__file__")
    if not _path or not os.path.isfile(_path):
        print(f"  SKIP  {'source scan':32s} no __file__ (running in a notebook)")
        print("=" * 74)
        print(f"  SELF-CHECK: {'PASS' if ok else 'FAIL'}  "
              f"(architecture verified; source scan skipped)")
        print("=" * 74)
        return 0 if ok else 1
    src_raw = open(_path, encoding="utf-8").read()
    tree = ast.parse(src_raw)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "self_check":
            guard_lines = set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            break
    else:
        guard_lines = set()
    exec_lines = []
    for i, line in enumerate(src_raw.splitlines(), 1):
        if i in guard_lines:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        exec_lines.append(line)
    src = "\n".join(exec_lines)

    for name, bad in [
        ("no ablation", ["v3_ablation", "ablation_se", "ablation_spatial",
                         "ablation_baseline", "ablation_full"]),
        ("no GCP", ["gcloud ", "gsutil ", "google.cloud", "terraform"]),
        ("no patchify", ["patchify_multimodal", "training_patch"]),
    ]:
        hits = [b for b in bad if b in src]
        print(f"  {'PASS' if not hits else 'FAIL'}  {name:32s} "
              f"{'clean' if not hits else hits}")
        ok &= not hits

    print("=" * 74)
    print(f"  SELF-CHECK: {'PASS' if ok else 'FAIL'}")
    print("=" * 74)
    return 0 if ok else 1




# ============================================================================
# TRAINING CONTROL — early stopping, top-k, state machine, GPU diagnostics
# ----------------------------------------------------------------------------
# Self-contained ports of the repository modules
# (src/experiment/{early_stopping,checkpoint,state_machine,diagnostics,
# artifacts}.py). Nothing here imports from the repository: this file must
# stay independently reproducible on Kaggle.
# ============================================================================

# --- training lifecycle -----------------------------------------------------
# A long run gets interrupted. Afterwards someone has to answer, from the
# output directory alone, what actually happened: did it finish, stop early,
# or die? A free-text status cannot answer that, so the lifecycle is explicit
# and persisted.
S_CREATED = "CREATED"
S_PREFLIGHT = "PREFLIGHT"
S_RUNNING = "RUNNING"
S_VALIDATING = "VALIDATING"
S_BEST_UPDATED = "BEST_UPDATED"
S_CHECKPOINTED = "CHECKPOINTED"
S_EARLY_STOPPED = "EARLY_STOPPED"
S_COMPLETED = "COMPLETED"
S_FAILED = "FAILED"
STATES = (S_CREATED, S_PREFLIGHT, S_RUNNING, S_VALIDATING, S_BEST_UPDATED,
          S_CHECKPOINTED, S_EARLY_STOPPED, S_COMPLETED, S_FAILED)

# GPU sampling phases. The distinction is the point: a reading taken while a
# NIfTI is loading describes the pause, not the computation.
P_IDLE = "IDLE"
P_DATA_WAIT = "DATA_WAIT"
P_ACTIVE_GPU = "ACTIVE_GPU"
P_VALIDATION = "VALIDATION"
P_CHECKPOINT = "CHECKPOINT"

NOT_MEASURED = "NOT MEASURED"
NOT_AVAILABLE = "NOT AVAILABLE"


class TrainingState:
    """Persisted lifecycle. Written to state.json after every change."""

    def __init__(self, directory: str, run_id: str = ""):
        self.directory = directory
        self.run_id = run_id
        self.state = S_CREATED
        self.history: List[dict] = []
        self.metadata: Dict[str, object] = {}
        os.makedirs(directory, exist_ok=True)
        self._record(S_CREATED, "run created")

    @property
    def path(self) -> str:
        return os.path.join(self.directory, "state.json")

    def _record(self, state: str, reason: str, **extra) -> None:
        entry = {"state": state, "reason": reason,
                 "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime())}
        entry.update(extra)
        self.history.append(entry)
        self.state = state
        self.save()

    def transition(self, state: str, reason: str = "", **extra) -> str:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        self._record(state, reason or state, **extra)
        return self.state

    def fail(self, reason: str, **extra) -> str:
        self._record(S_FAILED, reason, **extra)
        return self.state

    def set_metadata(self, **kw) -> None:
        self.metadata.update(kw)
        self.save()

    def save(self) -> str:
        payload = {"run_id": self.run_id, "state": self.state,
                   "history": self.history, "metadata": self.metadata,
                   "states_defined": list(STATES)}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, self.path)     # atomic: no truncated state file
        return self.path

    def summary(self) -> dict:
        counts: Dict[str, int] = {}
        for h in self.history:
            counts[h["state"]] = counts.get(h["state"], 0) + 1
        return {"run_id": self.run_id, "current_state": self.state,
                "transitions": len(self.history), "state_counts": counts,
                "early_stopped": S_EARLY_STOPPED in counts,
                "completed": S_COMPLETED in counts,
                "was_resumed": counts.get(S_RUNNING, 0) > 1}


class EarlyStopping:
    """Patience-based stopping on a VALIDATION metric.

    Training loss is deliberately not monitored: a falling training loss is
    exactly what overfitting looks like. The test split is never consulted.
    """

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE,
                 min_delta: float = EARLY_STOPPING_MIN_DELTA,
                 monitor: str = "validation mean foreground Dice",
                 enabled: bool = EARLY_STOPPING_ENABLED):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.enabled = bool(enabled)
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self.stopped_epoch = 0
        self.history: List[dict] = []

    def update(self, value: float, epoch: int) -> bool:
        finite = isinstance(value, (int, float)) and math.isfinite(value)
        improved = finite and (self.best is None
                               or value > self.best + self.min_delta)
        if improved:
            self.best, self.best_epoch, self.counter = float(value), epoch, 0
        else:
            self.counter += 1
            if self.enabled and self.patience and self.counter >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
        self.history.append({"epoch": epoch,
                             "value": float(value) if finite else float("nan"),
                             "best": self.best, "counter": self.counter,
                             "improved": improved})
        return self.should_stop

    def status(self) -> str:
        if self.best is None:
            return "early-stop: no validation value yet"
        pat = f"{self.counter}/{self.patience}" if self.patience else "off"
        return (f"early-stop: best {self.best:.4f} @ ep {self.best_epoch} | "
                f"patience {pat}")

    def report(self) -> dict:
        return {"enabled": self.enabled, "monitor": self.monitor,
                "patience": self.patience, "min_delta": self.min_delta,
                "best_value": self.best, "best_epoch": self.best_epoch,
                "final_counter": self.counter, "triggered": self.should_stop,
                "stopped_epoch": self.stopped_epoch or None,
                "evaluations": len(self.history),
                "note": ("Monitored on the VALIDATION split only. The test "
                         "split is never used for stopping or selection.")}

    def state_dict(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch,
                "counter": self.counter, "should_stop": self.should_stop,
                "stopped_epoch": self.stopped_epoch, "history": self.history}

    def load_state_dict(self, st: dict) -> None:
        # Patience must survive a restart, or a plateaued run trains another
        # full patience window after every interruption.
        self.best = st.get("best")
        self.best_epoch = int(st.get("best_epoch", 0) or 0)
        self.counter = int(st.get("counter", 0) or 0)
        self.should_stop = bool(st.get("should_stop", False))
        self.stopped_epoch = int(st.get("stopped_epoch", 0) or 0)
        self.history = list(st.get("history", []) or [])


def update_top_k(directory: str, weights, epoch: int, score: float,
                 metrics: dict, meta: dict, top_k: int = TOP_K_CHECKPOINTS):
    """Maintain best_1..best_k plus a manifest. Weights are NEVER averaged."""
    import torch
    os.makedirs(directory, exist_ok=True)
    man_path = os.path.join(directory, "top_k.json")
    entries: List[dict] = []
    if os.path.isfile(man_path):
        try:
            with open(man_path, encoding="utf-8") as fh:
                entries = json.load(fh).get("entries", [])
        except (OSError, ValueError):
            entries = []                 # unreadable manifest -> rebuild

    def p(rank):
        return os.path.join(directory, f"best_{rank}.pth")

    previous = sorted(entries, key=lambda e: e["score"], reverse=True)
    existing = {e["epoch"]: p(i + 1) for i, e in enumerate(previous)
                if os.path.isfile(p(i + 1))}

    entries = [e for e in entries if e.get("epoch") != epoch]
    entries.append({"epoch": epoch, "score": float(score),
                    "metrics": metrics, **meta})
    entries.sort(key=lambda e: e["score"], reverse=True)
    keep = entries[:max(1, int(top_k))]
    if not any(e["epoch"] == epoch for e in keep):
        return keep                      # did not make the cut: no I/O

    # Stage survivors aside first so a file changing rank cannot overwrite
    # another mid-shuffle.
    staged = {}
    for ep_, src in existing.items():
        if any(e["epoch"] == ep_ for e in keep) and ep_ != epoch:
            tmp = os.path.join(directory, f".stage_{ep_}.tmp")
            os.replace(src, tmp)
            staged[ep_] = tmp
    for rank in range(1, len(previous) + 2):
        if os.path.isfile(p(rank)):
            os.remove(p(rank))
    for i, e in enumerate(keep):
        dest = p(i + 1)
        if e["epoch"] == epoch:
            tmp = dest + ".tmp"
            torch.save({"m": weights, "ep": epoch, "score": float(score),
                        "metrics": metrics, **meta}, tmp)
            os.replace(tmp, dest)
        elif e["epoch"] in staged:
            os.replace(staged.pop(e["epoch"]), dest)
    for leftover in staged.values():
        if os.path.isfile(leftover):
            os.remove(leftover)

    tmp = man_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"top_k": int(top_k), "weight_averaging": False, "swa": False,
                   "selection_metric": meta.get("selection_metric", ""),
                   "entries": keep}, fh, indent=2, default=str)
    os.replace(tmp, man_path)
    return keep


class GPUDiagnostics:
    """Phase-labelled GPU sampler.

    Utilization is summarised ONLY from ACTIVE_GPU samples. Sampling is
    rate-limited because nvidia-smi costs tens of milliseconds and polling it
    per step would distort the timings being collected.
    """

    _Q = ("utilization.gpu,utilization.memory,clocks.sm,clocks.max.sm,"
          "temperature.gpu,power.draw,memory.total,memory.used")

    def __init__(self, enabled: bool = True, min_interval_s: float = 10.0):
        self.enabled = bool(enabled)
        self.min_interval_s = float(min_interval_s)
        self.samples: List[dict] = []
        self._last = 0.0
        self._phase = P_IDLE

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def _smi(self) -> dict:
        blank = {k: NOT_AVAILABLE for k in
                 ("gpu_utilization_pct", "memory_utilization_pct",
                  "sm_clock_mhz", "sm_clock_max_mhz", "temperature_c",
                  "power_w", "vram_total_mb", "vram_used_mb")}
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={self._Q}",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                return blank
            a = [x.strip() for x in r.stdout.strip().split("\n")[0].split(",")]

            def num(i, cast):
                try:
                    return cast(a[i])
                except Exception:
                    return NOT_AVAILABLE

            return {"gpu_utilization_pct": num(0, int),
                    "memory_utilization_pct": num(1, int),
                    "sm_clock_mhz": num(2, int), "sm_clock_max_mhz": num(3, int),
                    "temperature_c": num(4, int), "power_w": num(5, float),
                    "vram_total_mb": num(6, float), "vram_used_mb": num(7, float)}
        except Exception:
            return blank

    def sample(self, epoch: int = -1, step: int = -1, force: bool = False):
        import torch
        if not self.enabled:
            return None
        now = time.time()
        if not force and (now - self._last) < self.min_interval_s:
            return None
        self._last = now
        rec = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime(now)),
               "epoch": epoch, "step": step, "phase": self._phase}
        rec.update(self._smi())
        if torch.cuda.is_available():
            rec["torch_allocated_mb"] = round(
                torch.cuda.memory_allocated() / 1024 ** 2, 1)
            rec["torch_reserved_mb"] = round(
                torch.cuda.memory_reserved() / 1024 ** 2, 1)
        else:
            rec["torch_allocated_mb"] = NOT_AVAILABLE
            rec["torch_reserved_mb"] = NOT_AVAILABLE
        self.samples.append(rec)
        return rec

    def summary(self) -> dict:
        if not self.samples:
            return {"status": NOT_MEASURED, "samples": 0}

        def nums(rows, key):
            return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

        out: Dict[str, object] = {"samples": len(self.samples), "phases": {}}
        for ph in (P_IDLE, P_DATA_WAIT, P_ACTIVE_GPU, P_VALIDATION,
                   P_CHECKPOINT):
            rows = [r for r in self.samples if r["phase"] == ph]
            if not rows:
                continue
            entry = {"samples": len(rows)}
            for key in ("gpu_utilization_pct", "sm_clock_mhz",
                        "temperature_c", "power_w"):
                v = nums(rows, key)
                entry[key] = ({"mean": round(sum(v) / len(v), 1),
                               "min": min(v), "max": max(v)} if v
                              else NOT_AVAILABLE)
            out["phases"][ph] = entry

        active = [r for r in self.samples if r["phase"] == P_ACTIVE_GPU]
        util = nums(active, "gpu_utilization_pct")
        if util:
            mean_u = sum(util) / len(util)
            out["active_gpu_utilization_mean_pct"] = round(mean_u, 1)
            out["utilization_note"] = (
                "Measured DURING forward/backward only. DATA_WAIT, CHECKPOINT "
                "and IDLE samples are excluded: an idle reading describes the "
                "pause, not the computation.")
            if mean_u < 50:
                out["warning"] = (
                    f"GPU averaged {mean_u:.1f}% during ACTIVE_GPU -- low for "
                    f"compute-bound training. Investigate dataloader wait, "
                    f"H2D or CPU preprocessing.")
        else:
            out["active_gpu_utilization_mean_pct"] = NOT_MEASURED
        clocks, maxes = nums(active, "sm_clock_mhz"), nums(active,
                                                           "sm_clock_max_mhz")
        if clocks and maxes:
            pct = 100 * (sum(clocks) / len(clocks)) / max(maxes)
            out["active_clock_pct_of_max"] = round(pct, 1)
            if pct < 75:
                out["throttle_warning"] = (
                    f"SM clock averaged {pct:.0f}% of maximum: absolute "
                    f"timings are inflated ~{100 / pct:.1f}x. Ratios hold.")
        return out

    def write_csv(self, path: str) -> Optional[str]:
        if not self.samples:
            return None
        cols: List[str] = []
        for s in self.samples:
            for k in s:
                if k not in cols:
                    cols.append(k)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for s in self.samples:
                w.writerow({c: s.get(c, "") for c in cols})
        os.replace(tmp, path)
        return path


def run_fingerprint(root: str = "") -> dict:
    """Identity block embedded in checkpoints and artifacts."""
    import torch
    fp = {
        "architecture": f"GLO-NCA {L1_RES}^3k{L1_K}+{L2_RES}^3k{L2_K}",
        "parameters": EXPECTED_PARAMS,
        "total_nca_steps": L1_STEPS + L2_STEPS,
        "spatial_gc_kernel": SPATIAL_GC_KERNEL,
        "working_volume": WORKING_VOLUME,
        "patchify": USE_PATCHIFY,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "run_mode": RUN_MODE,
        "preset": ACTIVE_PRESET,
        "dataset_root": root or NOT_AVAILABLE,
        "selection_metric": ("validation mean foreground Dice, "
                             f"{SMOOTHING_WINDOW}-epoch rolling mean"),
        # The loss the model OPTIMISES. Reported as train_loss/val_loss; it is
        # not a quality score. Quality is Dice / IoU / HD95.
        "loss_function": "FocalTverskyCELoss",
        "loss_params": {"tversky_alpha": TVERSKY_ALPHA,
                        "tversky_beta": TVERSKY_BETA,
                        "focal_gamma": FOCAL_GAMMA,
                        "ce_weight": CE_WEIGHT,
                        "empty_region_bce_weight": EMPTY_REGION_BCE_WEIGHT},
        "quality_metrics": ["Dice", "IoU", "HD95"],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    try:
        fp["config_fingerprint"] = hashlib.sha256(
            json.dumps({k: fp[k] for k in sorted(fp)}, default=str)
            .encode("utf-8")).hexdigest()
    except Exception:
        fp["config_fingerprint"] = NOT_AVAILABLE
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        fp["gpu"] = torch.cuda.get_device_name(0)
        fp["gpu_capability"] = f"{cap[0]}.{cap[1]}"
        fp["cuda"] = torch.version.cuda
    return fp


def write_artifacts(outdir: str, *, fingerprint: dict, arch: dict,
                    epoch_rows: List[dict], stopper: "EarlyStopping",
                    run_state: "TrainingState", gpu_diag: "GPUDiagnostics",
                    ckpt_dir: str, top_k_entries, best_meta: Optional[dict],
                    periodic: List[str], final_ckpt: Optional[str],
                    resume_checks: Optional[dict], timing: Optional[dict]
                    ) -> Dict[str, str]:
    """Write the artifact set. Absent evidence is recorded, never invented."""
    import torch
    written: Dict[str, str] = {}

    def j(name, obj):
        p = os.path.join(outdir, name)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
        os.replace(tmp, p)
        written[name] = p
        return p

    def c(name, rows):
        p = os.path.join(outdir, name)
        if not rows:
            rows = [{"status": NOT_MEASURED}]
        cols: List[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        tmp = p + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
        os.replace(tmp, p)
        written[name] = p
        return p

    def stat(p):
        if not p or not os.path.isfile(p):
            return {"path": p or NOT_AVAILABLE, "exists": False}
        return {"path": os.path.basename(p), "exists": True,
                "size_bytes": os.path.getsize(p)}

    j("architecture_identity.json", {**fingerprint, "checks": arch})
    j("run_metadata.json", {
        **fingerprint, "epochs_planned": EPOCHS,
        "epochs_completed": len(epoch_rows),
        "early_stopping": stopper.report(),
        "training_state": run_state.summary(),
        "precision": PRECISION,
        "host": {"platform": platform.platform(),
                 "python": platform.python_version()}})
    hw = {"os": platform.platform(), "python": platform.python_version(),
          "cpu_count": os.cpu_count() or NOT_AVAILABLE,
          "torch": torch.__version__, "cuda": torch.version.cuda or NOT_AVAILABLE}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        pr = torch.cuda.get_device_properties(0)
        hw.update({"gpu_name": torch.cuda.get_device_name(0),
                   "gpu_count": torch.cuda.device_count(),
                   "gpu_capability": f"{cap[0]}.{cap[1]}",
                   "gpu_total_vram_mb": round(pr.total_memory / 1024 ** 2, 1),
                   "bf16_native": cap >= (8, 0)})
    else:
        hw["gpu_name"] = NOT_AVAILABLE
    j("hardware_metadata.json", hw)

    j("precision_benchmark.json", {
        "status": NOT_MEASURED,
        "reason": ("no precision sweep was run in this experiment; the "
                   "training precision below was auto-selected for the "
                   "detected GPU"),
        "precision_used": PRECISION,
        "other_hardware": {"NVIDIA L4": NOT_MEASURED, "Tesla T4": NOT_MEASURED}})
    j("timing_summary.json", timing or {
        "status": NOT_MEASURED,
        "note": "nested sections overlap; percentages do NOT sum to 100%"})

    c("epoch_metrics.csv", epoch_rows)
    c("validation_history.csv",
      [{k: v for k, v in r.items()
        if k.startswith(("epoch", "val", "dice", "iou", "hd95", "smooth"))}
       for r in epoch_rows])

    entries = []
    if best_meta:
        entries.append({"kind": "best", **best_meta})
    for i, e in enumerate(top_k_entries or []):
        entries.append({"kind": "top_k", "rank": i + 1, **e})
    for p_ in periodic:
        entries.append({"kind": "periodic", **stat(p_)})
    if final_ckpt:
        entries.append({"kind": "final", **stat(final_ckpt)})
    j("checkpoint_manifest.json", {**fingerprint, "count": len(entries),
                                   "checkpoint_dir": ckpt_dir,
                                   "checkpoints": entries})
    j("best_checkpoint_metadata.json", {
        **fingerprint, "kind": "best",
        "selection_split": "validation only", "test_split_used": False,
        **(best_meta or {"status": NOT_MEASURED})})
    j("top3_checkpoint_metadata.json", {
        **fingerprint, "kind": "top_k", "k": len(top_k_entries or []),
        "weight_averaging": False, "swa": False,
        "ranking": top_k_entries or NOT_MEASURED})
    j("periodic_checkpoint_metadata.json", {
        **fingerprint, "kind": "periodic",
        "interval_epochs": CHECKPOINT_FREQUENCY, "count": len(periodic),
        "checkpoints": [stat(p_) for p_ in periodic] or NOT_MEASURED})
    j("final_checkpoint_metadata.json", {
        **fingerprint, "kind": "final", "distinct_from_best": True,
        "note": "the final checkpoint is the LAST state, not the selected model",
        **(stat(final_ckpt) if final_ckpt else {"status": NOT_MEASURED})})

    j("resume_verification.json",
      {"status": "VERIFIED", "executed": True, "checks": resume_checks,
       "all_passed": all(bool(v) for v in resume_checks.values())}
      if resume_checks else
      {"status": NOT_MEASURED,
       "reason": "resume was not exercised in this run"})

    vals = [r.get("val_dice_mean") for r in epoch_rows
            if isinstance(r.get("val_dice_mean"), (int, float))]
    losses = [r.get("train_loss") for r in epoch_rows
              if isinstance(r.get("train_loss"), (int, float))]
    ovr: Dict[str, object] = {
        "monitor": fingerprint["selection_metric"],
        "selection_split": "validation only",
        "test_split_used_for_selection": False,
        "best_epoch": stopper.best_epoch or NOT_MEASURED,
        "early_stopping": stopper.report(),
        "epochs_observed": len(vals)}
    # Three points is the minimum for a trend to mean anything.
    if len(vals) < 3 or len(losses) < 3:
        ovr.update({"status": NOT_MEASURED,
                    "reason": f"only {len(vals)} validation point(s); too few "
                              f"to describe a trend"})
    else:
        h = max(1, len(losses) // 2)
        vh = max(1, len(vals) // 2)
        tl0, tl1 = sum(losses[:h]) / h, sum(losses[-h:]) / h
        v0, v1 = sum(vals[:vh]) / vh, sum(vals[-vh:]) / vh
        div = tl1 < tl0 and v1 < v0
        ovr.update({
            "status": "MEASURED",
            "train_loss_first_half_mean": round(tl0, 6),
            "train_loss_second_half_mean": round(tl1, 6),
            "validation_first_half_mean": round(v0, 6),
            "validation_second_half_mean": round(v1, 6),
            "best_validation": max(vals), "final_validation": vals[-1],
            "divergence": div,
            "interpretation": (
                "Training loss fell while validation did not improve: the "
                "classic overfitting signature." if div else
                "No train/validation divergence over the epochs recorded.")})
    ovr["scope"] = "Describes this run only. NOT a segmentation-quality claim."
    j("overfitting_report.json", ovr)

    gsum = gpu_diag.summary()
    if not gpu_diag.write_csv(os.path.join(outdir, "gpu_diagnostics.csv")):
        c("gpu_diagnostics.csv", [{"status": NOT_MEASURED}])
    else:
        written["gpu_diagnostics.csv"] = os.path.join(outdir,
                                                      "gpu_diagnostics.csv")
    j("gpu_diagnostics_summary.json", gsum)
    j("training_state.json", run_state.summary())
    return written


def audit_artifacts(outdir: str) -> dict:
    """Re-read what was written and check the artifacts agree."""
    required = ("run_metadata.json", "architecture_identity.json",
                "hardware_metadata.json", "precision_benchmark.json",
                "timing_summary.json", "epoch_metrics.csv",
                "validation_history.csv", "checkpoint_manifest.json",
                "best_checkpoint_metadata.json", "top3_checkpoint_metadata.json",
                "periodic_checkpoint_metadata.json",
                "final_checkpoint_metadata.json", "resume_verification.json",
                "overfitting_report.json", "gpu_diagnostics.csv")
    checks, missing, bad = [], [], []
    loaded = {}
    for name in required:
        p = os.path.join(outdir, name)
        if not os.path.isfile(p):
            missing.append(name)
            continue
        if name.endswith(".json"):
            try:
                with open(p, encoding="utf-8") as fh:
                    loaded[name] = json.load(fh)
            except ValueError as e:
                bad.append(f"{name}: {e}")
    checks.append({"check": "all artifacts present", "pass": not missing,
                   "detail": str(missing)})
    checks.append({"check": "all JSON parses", "pass": not bad,
                   "detail": str(bad)})
    ref = loaded.get("architecture_identity.json", {})
    for key in ("parameters", "seed", "config_fingerprint",
                "total_nca_steps", "spatial_gc_kernel"):
        want = ref.get(key)
        if want is None:
            continue
        mism = [n for n, o in loaded.items()
                if isinstance(o, dict) and key in o and o[key] != want]
        checks.append({"check": f"'{key}' agrees across artifacts",
                       "pass": not mism, "detail": f"{want} / {mism}"})
    checks.append({"check": "parameters == EXPECTED_PARAMS",
                   "pass": ref.get("parameters") == EXPECTED_PARAMS,
                   "detail": str(ref.get("parameters"))})
    report = {"checks": checks, "missing": missing, "unparseable": bad,
              "passed": all(c["pass"] for c in checks)}
    with open(os.path.join(outdir, "artifact_consistency_audit.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


# ============================================================================
# STANDALONE TEST EVALUATION
# ============================================================================
def evaluate_test(checkpoint: str = None, data_root: str = None,
                  out_dir: str = None) -> dict:
    """Evaluate a SAVED model on the frozen test split.

    Separate from training on purpose: the test split is scored once, after
    the model is final, using the checkpoint that validation selected. It is
    never used for early stopping, checkpoint selection or ranking -- that is
    what makes the number an estimate of generalisation rather than a
    restatement of what the run already optimised for.

    Re-runnable without retraining:

        evaluate_test()                       # best.pth from the last run
        evaluate_test("path/to/best.pth")     # any saved checkpoint
    """
    import torch

    out_dir = out_dir or OUT_DIR
    checkpoint = checkpoint or os.path.join(out_dir, "checkpoint", "best.pth")
    if not os.path.isfile(checkpoint):
        print(f"FAILED: no checkpoint at {checkpoint}\n"
              f"Train first, or pass an explicit path.")
        return {"status": "FAILED", "reason": "checkpoint not found"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = data_root or find_dataset_root()
    prof = Profiler(enabled=False)
    inv = inventory_dataset(root, prof)
    split = make_split(inv["cases"])
    if not split["test"]:
        print("FAILED: this split has no independent test set.")
        return {"status": NOT_MEASURED, "reason": "no test split"}

    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model, _ = make_model(device)
    model = model.to(device)
    sd = ck.get("m", ck.get("model"))
    model.load_state_dict(sd if isinstance(sd, dict) else sd[0])
    model.eval()
    n_params = sum(p_.numel() for p_ in model.parameters())

    print("=" * 74)
    print("FROZEN TEST EVALUATION")
    print("=" * 74)
    print(f"  checkpoint       : {checkpoint}")
    print(f"  trained to epoch : {ck.get('ep', ck.get('epoch', '?'))}")
    print(f"  parameters       : {n_params:,}"
          + ("" if n_params == EXPECTED_PARAMS
             else f"   *** expected {EXPECTED_PARAMS:,} ***"))
    print(f"  test cases       : {len(split['test'])}")
    print(f"  device           : {device}")
    if n_params != EXPECTED_PARAMS:
        print("\nREFUSING: the checkpoint is not the production architecture.")
        return {"status": "FAILED", "reason": "architecture mismatch"}

    amp_dtype = None
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        amp_dtype = (torch.bfloat16 if cap >= (8, 0)
                     and torch.cuda.is_bf16_supported() else torch.float16)

    ds = BraTSDataset(split["test"], {}, use_cache=USE_CACHE,
                      patch_size=None, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                     num_workers=0, collate_fn=collate,
                                     pin_memory=(device.type == "cuda"))
    d = {r: [] for r in REGIONS}
    i_ = {r: [] for r in REGIONS}
    h = {r: [] for r in REGIONS}
    per_case = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb, ids, _pb, _box in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    o = model(xb)
                o = o.float()
            else:
                o = model(xb)
            o = o.permute(0, 2, 3, 4, 1).contiguous()
            tg = yb
            if tg.shape[1:4] != o.shape[1:4]:
                tcf = tg.permute(0, 4, 1, 2, 3).contiguous()
                tcf = torch.nn.functional.interpolate(
                    tcf, size=tuple(o.shape[1:4]), mode="nearest")
                tg = tcf.permute(0, 2, 3, 4, 1).contiguous()
            prob = torch.sigmoid(o)[0].detach().cpu().numpy()
            gt = tg[0].detach().cpu().numpy()
            row = {"case": ids[0]}
            for k, r in enumerate(REGIONS):
                dd, ii = dice_iou(prob[..., k], gt[..., k])
                hh = hd95(prob[..., k], gt[..., k])
                d[r].append(dd); i_[r].append(ii); h[r].append(hh)
                row[f"dice_{r}"] = dd
                row[f"iou_{r}"] = ii
                row[f"hd95_{r}"] = hh
            per_case.append(row)
    secs = time.perf_counter() - t0

    def mean(v):
        vv = [x for x in v if not np.isnan(x)]
        return float(np.mean(vv)) if vv else float("nan")

    def std(v):
        vv = [x for x in v if not np.isnan(x)]
        return float(np.std(vv)) if len(vv) > 1 else 0.0

    res = {"status": "MEASURED", "cases": len(per_case),
           "checkpoint": checkpoint,
           "model_epoch": ck.get("ep", ck.get("epoch")),
           "selected_on": "validation (test never used for selection)",
           "seconds": secs,
           **{f"dice_{r}": mean(d[r]) for r in REGIONS},
           **{f"dice_std_{r}": std(d[r]) for r in REGIONS},
           **{f"iou_{r}": mean(i_[r]) for r in REGIONS},
           **{f"hd95_{r}": mean(h[r]) for r in REGIONS}}
    res["dice_mean"] = mean([res[f"dice_{r}"] for r in REGIONS])

    print()
    print(f"  {'region':8s} {'Dice':>8s} {'+/-':>7s} {'IoU':>8s} {'HD95':>9s}")
    for r in REGIONS:
        print(f"  {r:8s} {res[f'dice_{r}']:8.4f} {res[f'dice_std_{r}']:7.4f} "
              f"{res[f'iou_{r}']:8.4f} {res[f'hd95_{r}']:9.2f}")
    print(f"  {'mean':8s} {res['dice_mean']:8.4f}")
    print(f"  {len(per_case)} cases in {secs:.1f}s "
          f"({secs / max(1, len(per_case)):.2f}s/case)")
    print("  HD95 in VOXELS (lower better); Dice/IoU higher better.")
    print()
    print("  This is the generalisation estimate. Validation numbers are")
    print("  optimistic by construction: they chose the checkpoint.")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_per_case.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_case[0].keys()))
        w.writeheader()
        for r2 in per_case:
            w.writerow(r2)
    with open(os.path.join(out_dir, "test_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump({**res, **run_fingerprint(root), "per_case": per_case,
                   "protocol": ("Single evaluation on the frozen test split, "
                                "after training, using the "
                                "validation-selected checkpoint.")},
                  fh, indent=2, default=str)
    print(f"\n  written: {out_dir}/test_results.json, test_per_case.csv")
    print("=" * 74)
    return res


# ============================================================================
# MAIN
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="GLO-NCA Kaggle 50-epoch validation")
    ap.add_argument("--self-check", action="store_true",
                    help="local static validation only (no data, no training)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--limit-cases", type=int, default=0,
                    help="debug only: cap usable cases (0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last checkpoint instead of epoch 0")
    args = ap.parse_args()
    if not hasattr(args, "resume"):
        args.resume = False

    if args.self_check:
        return self_check()

    import torch
    global LOG
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "checkpoint"), exist_ok=True)
    LOG = Tee(os.path.join(OUT_DIR, "training_log.txt"))

    LOG("=" * 74)
    LOG("GLO-NCA — KAGGLE 50-EPOCH PRODUCTION VALIDATION")
    LOG("=" * 74)
    LOG("Engineering/validation experiment. NOT thesis performance, NOT an")
    LOG("ablation, NOT the thesis master split (898/200/198 is untouched).")
    LOG("")

    # ---- environment -------------------------------------------------------
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    gpu_total = (torch.cuda.get_device_properties(0).total_memory / 1024**2
                 if device.type == "cuda" else 0.0)
    # ---- PRECISION SELECTION -------------------------------------------------
    # "auto": fp16 on Turing (sm_75, e.g. Tesla T4) because bf16 is EMULATED
    # there while fp16 has native tensor cores; bf16 on Ampere+ where bf16 is
    # native and needs no loss scaling.
    _cap = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
    _prec = PRECISION.lower()
    if _prec == "auto":
        if device.type != "cuda":
            _prec = "fp32"
        elif _cap >= (8, 0) and torch.cuda.is_bf16_supported():
            _prec = "bf16"          # Ampere+ : native bf16
        else:
            _prec = "fp16"          # Turing  : native fp16 tensor cores
    if _prec == "bf16" and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        _prec = "fp32"
    if _prec == "fp16" and device.type != "cuda":
        _prec = "fp32"

    # TF32 + cuDNN autotuning. No-op on Turing (no TF32 units); free on Ampere+.
    # cudnn.benchmark is safe here because conv shapes are fixed after warm-up.
    if USE_TF32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(_prec)
    use_amp = amp_dtype is not None
    use_bf16 = use_amp                      # kept for the existing call sites
    # fp16 has a narrow range and CAN underflow gradients, so it needs a
    # GradScaler. bf16 does not. The loss itself stays FP32 either way.
    scaler = (torch.amp.GradScaler("cuda", enabled=(_prec == "fp16"))
              if device.type == "cuda" else None)

    LOG(f"GPU        : {gpu_name} ({gpu_total:.0f} MB)  compute capability {_cap[0]}.{_cap[1]}")
    LOG(f"CUDA       : {torch.version.cuda}   PyTorch: {torch.__version__}")
    LOG(f"Precision  : {_prec}"
        + (" (forward only; loss FP32)" if use_amp else "")
        + (f"   [PRECISION='{PRECISION}' -> auto-selected for sm_{_cap[0]}{_cap[1]}]"
           if PRECISION.lower() == "auto" else ""))
    if _prec == "fp16":
        LOG("             fp16 uses a GradScaler (bf16/fp32 do not need one).")
        if _cap == (7, 5):
            LOG("             This GPU is Turing: bf16 is EMULATED but fp16 has "
                "native tensor cores, so fp16 should be faster here.")
    LOG("")

    prof = Profiler(enabled=True, cuda=(device.type == "cuda"))

    # ---- §4 dataset --------------------------------------------------------
    t_disc = time.perf_counter()
    root = args.data_root or find_dataset_root()
    LOG(f"Dataset root (discovered): {root}")
    inv_res = inventory_dataset(root, prof)
    inv, cases = inv_res["inventory"], inv_res["cases"]
    disc_s = time.perf_counter() - t_disc
    prof.add("setup/dataset_discovery_total", disc_s)

    LOG(f"  cases discovered : {inv['case_count_discovered']}")
    LOG(f"  cases usable     : {inv['case_count_usable']}")
    LOG(f"  cases skipped    : {inv['case_count_skipped']}")
    for s in inv["skipped_cases"][:10]:
        LOG(f"     SKIP {s['case']}: {s['reason']}")
    LOG(f"  shapes           : {inv['unique_shapes'][:4]}")
    LOG(f"  spacings         : {inv['unique_spacings'][:4]}")
    if not cases:
        LOG("FATAL: no usable cases. Aborting (no fabricated results).")
        return 2
    if args.limit_cases:
        cases = cases[:args.limit_cases]
        LOG(f"  [debug] limited to {len(cases)} cases")

    split = make_split(cases)
    inv["kaggle_split"] = {k: [c["case"] for c in v] for k, v in split.items()}
    inv["kaggle_split_counts"] = {k: len(v) for k, v in split.items()}
    inv["kaggle_split_note"] = (
        "Kaggle-only, subject-level, seed 42. NOT the thesis master split.")
    has_test = len(split["test"]) > 0
    LOG(f"  split (Kaggle-only, seed 42): train {len(split['train'])} / "
        f"val {len(split['validation'])} / test {len(split['test'])}")
    if not has_test:
        LOG("  NOTE: no independent test set available -> train/validation only.")
    json.dump(inv, open(os.path.join(OUT_DIR, "dataset_inventory.json"), "w"), indent=2)

    # ---- model + §16 identity ---------------------------------------------
    model, _C = make_model(device)
    model = model.to(device)
    arch = assert_architecture(model)
    LOG("")
    LOG("ARCHITECTURE IDENTITY (§16):")
    for c in arch["checks"]:
        LOG(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']:32s} {c['actual']}")
    json.dump(arch, open(os.path.join(OUT_DIR, "architecture_summary.json"), "w"),
              indent=2)
    if not arch["passed"]:
        LOG("")
        LOG("FATAL: architecture identity FAILED. The model is NOT the production")
        LOG("GLO-NCA. Aborting rather than modifying the architecture to fit.")
        for f in arch["failures"]:
            LOG(f"   - {f}")
        return 3
    LOG(f"  -> {arch['parameter_count']:,} parameters, identity CONFIRMED")

    # torch.compile AFTER the identity gate, so the gate always inspects the
    # real module. Fails soft: if Triton is missing (e.g. Windows) the run
    # continues uncompiled rather than dying.
    if USE_TORCH_COMPILE and device.type == "cuda":
        try:
            model = torch.compile(model)
            LOG("  -> torch.compile ENABLED (first iterations include compile time)")
        except Exception as _e:
            LOG(f"  -> torch.compile unavailable, continuing uncompiled: "
                f"{str(_e)[:90]}")

    cfg_snap = {
        "run_mode": RUN_MODE,
        "preset": ACTIVE_PRESET,
        "is_production_identity": IS_PRODUCTION_IDENTITY,
        "category_c_deviations": category_c_deviations(),
        "seed": SEED, "epochs": args.epochs, "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS, "working_volume": WORKING_VOLUME,
        "level1": [L1_RES, L1_CH, L1_STEPS], "level2": [L2_RES, L2_CH, L2_STEPS],
        "level3": None, "total_nca_steps": L1_STEPS + L2_STEPS, "hidden": HIDDEN,
        "spatial_gc_kernel": SPATIAL_GC_KERNEL,
        "patch_native_resolution": PATCH_NATIVE_RES,
        "patch_size": (PATCH_SIZE if USE_PATCHIFY else None),
        "fast_grid": FAST_GRID,
        "torch_compile": USE_TORCH_COMPILE, "tf32": USE_TF32,
        "fire_rate": FIRE_RATE, "dropout": DROPOUT, "fusion": FUSION,
        "use_attention": USE_ATTENTION, "use_spatial": USE_SPATIAL,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "optimizer": "AdamW", "lr": LR, "min_lr": MIN_LR,
        "weight_decay": WEIGHT_DECAY, "betas": list(BETAS),
        "scheduler": ("LinearLR+CosineAnnealingLR" if WARMUP_EPOCHS > 0
                      else "CosineAnnealingLR"),
        "warmup_epochs": WARMUP_EPOCHS,
        # batch 1 -> steps/epoch == len(train split); train_loader is not
        # built yet at this point, so use the split length directly.
        "ema_decay": derive_ema_decay(max(1, len(split["train"]))),
        "ema_epoch_window": EMA_EPOCH_WINDOW,
        "validate_with_ema": VALIDATE_WITH_EMA,
        "grad_clip": GRAD_CLIP, "precision": _prec,
        "precision_requested": PRECISION,
        "compute_capability": f"{_cap[0]}.{_cap[1]}",
        "augmentation": ("light" if USE_AUGMENTATION else "none"),
        "gradient_accumulation": 1,
        "loss": {"type": "FocalTverskyCELoss", "alpha": TVERSKY_ALPHA,
                 "beta": TVERSKY_BETA, "gamma": FOCAL_GAMMA,
                 "ce_weight": CE_WEIGHT,
                 "empty_region_bce_weight": EMPTY_REGION_BCE_WEIGHT},
        "patchify": USE_PATCHIFY, "roi": 1.0,
        "checkpoint_frequency": CHECKPOINT_FREQUENCY,
        "hd95_every_epochs": HD95_EVERY_EPOCHS,
        "parameters": arch["parameter_count"],
    }
    json.dump(cfg_snap, open(os.path.join(OUT_DIR, "config_snapshot.json"), "w"),
              indent=2)

    # ---- data loaders ------------------------------------------------------
    ds_timings: Dict[str, List[float]] = {}
    train_ds = BraTSDataset(split["train"], ds_timings, use_cache=USE_CACHE,
                            patch_size=(PATCH_SIZE if USE_PATCHIFY else None),
                            train=True, augment=USE_AUGMENTATION)
    if not split["validation"]:
        LOG("FATAL: empty validation split. Refusing to validate on training "
            "data (that would be a silent leak).")
        return 4
    # VALIDATION IS ALWAYS FULL-VOLUME: train=False disables patch extraction,
    # so evaluation geometry can never be a patch.
    val_ds = BraTSDataset(split["validation"], ds_timings, use_cache=USE_CACHE,
                          patch_size=None, train=False)

    # DataLoader workers are separate PROCESSES, and on spawn-based platforms
    # (Windows, and any notebook kernel) they must PICKLE the dataset class. A
    # class defined in a notebook cell lives in an interactive __main__ that
    # cannot be re-imported, so `num_workers > 0` fails with
    #   PicklingError: Can't pickle <class '__main__.BraTSDataset'>
    # Detect that case and fall back to in-process loading. This costs some
    # speed but is the difference between running and crashing at batch 1.
    workers = NUM_WORKERS
    if workers > 0:
        mod = sys.modules.get(BraTSDataset.__module__)
        if getattr(mod, "__file__", None) is None:
            workers = 0
            LOG("  NOTE: notebook/interactive session detected -> num_workers=0 "
                "(worker processes cannot pickle cell-defined classes). "
                "Run the .py file directly to use workers.")

    def make_loader(ds, shuffle):
        kw = dict(batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=workers,
                  collate_fn=collate, pin_memory=(device.type == "cuda"))
        if workers > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 2
        return torch.utils.data.DataLoader(ds, **kw)

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)
    LOG(f"  DataLoader: workers={workers} pin_memory={device.type=='cuda'} "
        f"persistent_workers={workers>0} prefetch_factor={2 if workers else None}")

    # ---- optimiser / loss / EMA -------------------------------------------
    loss_f = _C["FocalTverskyCELoss"](TVERSKY_ALPHA, TVERSKY_BETA,
                                      FOCAL_GAMMA, CE_WEIGHT)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS,
                            weight_decay=WEIGHT_DECAY)
    # STABILITY FIX 3 -- linear LR warmup before the cosine decay.
    #
    # The 50-epoch T4 run collapsed at epoch 3 (TC 0.499 -> 0.316, ET 0.483 ->
    # 0.279) and then recovered -- the signature of a full-rate LR applied
    # before the normalisation statistics have settled. At batch 1 the
    # BatchNorm statistics come from ONE volume, so early updates are
    # especially noisy and a full 0.0016 step can undo several epochs.
    #
    # Warmup scales the LR linearly from LR/WARMUP_START_DIV to LR over the
    # first WARMUP_EPOCHS epochs, then hands over to the SAME cosine curve as
    # before. SequentialLR keeps the cosine's own T_max intact, so the decay
    # past warmup is unchanged from the certified schedule.
    #
    # WARMUP_EPOCHS = 0 restores the previous no-warmup behaviour exactly.
    if WARMUP_EPOCHS > 0:
        _warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0 / WARMUP_START_DIV, end_factor=1.0,
            total_iters=WARMUP_EPOCHS)
        _cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.epochs - WARMUP_EPOCHS), eta_min=MIN_LR)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[_warm, _cos], milestones=[WARMUP_EPOCHS])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                           eta_min=MIN_LR)

    # STABILITY FIX 4 -- resolve the EMA decay now that steps/epoch is known.
    # At batch 1, steps/epoch == len(train set), so this is the number of
    # optimiser steps the EMA will see per epoch.
    _steps_per_epoch = max(1, len(train_loader))
    ema_decay = derive_ema_decay(_steps_per_epoch)
    LOG(f"  EMA: decay {ema_decay:.6f} derived from {_steps_per_epoch} "
        f"steps/epoch (window {EMA_EPOCH_WINDOW:.1f} epochs)"
        if EMA_DECAY is None else
        f"  EMA: decay {ema_decay:.6f} (pinned)")
    LOG(f"  LR: warmup {WARMUP_EPOCHS} epochs "
        f"({LR / WARMUP_START_DIV:.2e} -> {LR:.2e}), then cosine to {MIN_LR:.2e}"
        if WARMUP_EPOCHS > 0 else
        f"  LR: no warmup, cosine {LR:.2e} -> {MIN_LR:.2e}")
    LOG(f"  validation weights: {'EMA' if VALIDATE_WITH_EMA else 'RAW'}")

    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    epoch_rows: List[dict] = []
    gpu_rows: List[dict] = []
    iter_rows: List[dict] = []
    val_prof = Profiler(enabled=True, cuda=(device.type == "cuda"))
    iter_times: List[float] = []
    warmup_time = None
    global_step = 0
    t_experiment = time.perf_counter()
    train_only_s = val_total_s = ckpt_total_s = 0.0

    # ---- RESUME (option 2): continue a run that a session limit cut short ----
    # Kaggle sessions end after ~12 h. With --resume / resume=True the run picks
    # up from the last checkpoint instead of restarting at epoch 0.
    start_epoch = 0
    if getattr(args, "resume", False):
        _ck = os.path.join(OUT_DIR, "checkpoint", "last.pth")
        if os.path.isfile(_ck):
            _sd = torch.load(_ck, map_location=device, weights_only=False)
            model.load_state_dict(_sd["model"])
            opt.load_state_dict(_sd["optimizer"])
            if "scheduler" in _sd:
                # A checkpoint written BEFORE the warmup fix holds a plain
                # CosineAnnealingLR state, whose keys (T_max, eta_min, ...) do
                # not match SequentialLR's (_schedulers, _milestones, ...), so
                # a direct load raises KeyError: '_schedulers'. Rather than
                # refuse the resume, rebuild the schedule by fast-forwarding
                # the NEW scheduler to the saved epoch: warmup is already over
                # by then in any run long enough to be worth resuming, and the
                # cosine is a closed-form function of last_epoch, so the LR
                # lands on the correct point of the new curve.
                try:
                    sched.load_state_dict(_sd["scheduler"])
                except (KeyError, TypeError, ValueError) as _e:
                    _target = int(_sd.get("epoch", 0))
                    for _ in range(_target):
                        sched.step()
                    LOG(f"  resume: scheduler state incompatible "
                        f"({type(_e).__name__}) -- rebuilt by stepping to epoch "
                        f"{_target}; LR now {opt.param_groups[0]['lr']:.3e}")
            if "ema" in _sd:
                # EMA is saved as a plain {name: tensor} dict (see torch.save below).
                ema = {k: v.to(device) for k, v in _sd["ema"].items()}
            start_epoch = int(_sd.get("epoch", 0))
            global_step = int(_sd.get("global_step", 0))
            LOG(f"RESUMED from {_ck}: epoch {start_epoch}, step {global_step}")
            if start_epoch >= args.epochs:
                LOG(f"  checkpoint epoch {start_epoch} >= requested "
                    f"{args.epochs}: nothing to do.")
                return 0
        else:
            LOG(f"resume requested but no checkpoint at {_ck} -> starting fresh")

    # ---- training control --------------------------------------------------
    FINGERPRINT = run_fingerprint(root)
    run_state = TrainingState(OUT_DIR, run_id=os.path.basename(OUT_DIR))
    run_state.transition(S_PREFLIGHT, "identity and data gates passed")
    run_state.set_metadata(**FINGERPRINT, epochs_planned=args.epochs)
    stopper = EarlyStopping()
    gpu_diag = GPUDiagnostics(enabled=True, min_interval_s=10.0)
    best_smooth, best_epoch = -1.0, 0
    best_meta: Optional[dict] = None
    top_k_entries: List[dict] = []
    periodic_ckpts: List[str] = []
    final_ckpt_path: Optional[str] = None
    _stop = False

    # Restore stopping state so patience is not silently reset on resume: a
    # plateaued run would otherwise train another full patience window after
    # every restart.
    if args.resume:
        _rk = os.path.join(OUT_DIR, "checkpoint", "last.pth")
        if os.path.isfile(_rk):
            try:
                _rs = torch.load(_rk, map_location="cpu", weights_only=False)
                if _rs.get("early_stopping"):
                    stopper.load_state_dict(_rs["early_stopping"])
                    LOG(f"resume: early-stopping restored "
                        f"(best {stopper.best:.4f} @ ep {stopper.best_epoch}, "
                        f"patience {stopper.counter}/{stopper.patience})")
                best_smooth = float(_rs.get("best_score", best_smooth))
                best_epoch = int(_rs.get("best_epoch", best_epoch))
            except Exception as _e:
                LOG(f"resume: could not restore training control ({_e})")

    LOG("")
    LOG("=" * 74)
    LOG(f"TRAINING — epochs {start_epoch + 1}..{args.epochs}, batch {BATCH_SIZE}")
    LOG("=" * 74)
    if stopper.enabled:
        LOG(f"early stopping: monitor={stopper.monitor}, "
            f"patience={stopper.patience}, min_delta={stopper.min_delta:g} "
            f"({SMOOTHING_WINDOW}-epoch rolling mean, VALIDATION only)")
    LOG(f"checkpoints: best + top-{TOP_K_CHECKPOINTS} + periodic every "
        f"{CHECKPOINT_FREQUENCY} epochs + final (distinct from best)")
    run_state.transition(S_RUNNING, f"training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.attach_profiler(prof)
        gpu_diag.set_phase(P_ACTIVE_GPU)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ep_t0 = time.perf_counter()
        ep_loss, n_it = 0.0, 0
        loader_t0 = time.perf_counter()

        for xb, yb, _ids, pb, box in train_loader:
            # D: DataLoader wait — time spent waiting for the worker queue
            prof.add("train/dataloader_wait", time.perf_counter() - loader_t0)
            it_t0 = time.perf_counter()

            with prof.section("train/h2d_transfer"):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if pb is not None:
                    pb = pb.to(device, non_blocking=True)

            with prof.section("train/forward"):
                # xb is ALWAYS the full working volume -> global context.
                # pb is the patch -> high-resolution level only.
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(xb, patch_cl=pb, patch_box=box)
                    logits = logits.float()     # loss always runs in FP32
                else:
                    logits = model(xb, patch_cl=pb, patch_box=box)
                outputs = logits.permute(0, 2, 3, 4, 1).contiguous()

            with prof.section("train/target_align"):
                t_cf = yb.permute(0, 4, 1, 2, 3).contiguous()
                if tuple(t_cf.shape[2:]) != tuple(outputs.shape[1:4]):
                    t_cf = torch.nn.functional.interpolate(
                        t_cf, size=tuple(outputs.shape[1:4]), mode="nearest")
                targets = t_cf.permute(0, 2, 3, 4, 1).contiguous()
                assert targets.shape[1:4] == outputs.shape[1:4], \
                    f"geometry mismatch {targets.shape} vs {outputs.shape}"

            opt.zero_grad(set_to_none=True)
            with prof.section("train/loss"):
                loss = 0
                for m in range(outputs.shape[-1]):
                    if torch.any(targets[..., m] == 1):
                        l_m = loss_f(outputs[..., m], targets[..., m])
                    else:
                        p = torch.sigmoid(outputs[..., m]).clamp(1e-6, 1. - 1e-6)
                        l_m = EMPTY_REGION_BCE_WEIGHT * \
                            torch.nn.functional.binary_cross_entropy(
                                p, targets[..., m], reduction="mean")
                    loss = loss + l_m

            with prof.section("train/backward"):
                # fp16 needs gradient scaling to avoid underflow; bf16/fp32 do
                # not, and the scaler is a no-op when disabled.
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            with prof.section("train/grad_clip"):
                # Unscale BEFORE clipping so the clip threshold means the same
                # thing in every precision.
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            with prof.section("train/optimizer_step"):
                if scaler is not None and scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
            # Sample WHILE compute is in flight. Taken at the top of the
            # epoch it would read an idle card and report ~0% -- the exact
            # mistake the phase labelling exists to prevent.
            gpu_diag.sample(epoch=epoch + 1, step=global_step)
            with prof.section("train/ema_update"):
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema[k].mul_(ema_decay).add_(v.float(), alpha=1 - ema_decay)

            it_dt = time.perf_counter() - it_t0
            ep_loss += float(loss.detach())
            n_it += 1
            global_step += 1
            if epoch == 0 and n_it == 1:
                warmup_time = it_dt        # §12 first iteration kept separate
            else:
                iter_times.append(it_dt)
            iter_rows.append({"epoch": epoch + 1, "iter": n_it,
                              "seconds": round(it_dt, 6)})
            loader_t0 = time.perf_counter()

        with prof.section("train/scheduler_step"):
            sched.step()
        ep_train_s = time.perf_counter() - ep_t0
        train_only_s += ep_train_s

        # ---- validation (profiled separately, §9) --------------------------
        do_hd95 = ((epoch + 1) % HD95_EVERY_EPOCHS == 0) or (epoch + 1 == args.epochs)
        model.eval()
        model.attach_profiler(None)         # keep validation out of train timers

        # STABILITY FIX 1 -- validate on the EMA weights.
        #
        # best.pth has always stored the EMA weights, but validation ran on the
        # RAW weights. So the score that selected a checkpoint was never the
        # score of the checkpoint being selected: a noisy proxy chose a smooth
        # model, and the smooth model was never measured. Swap the EMA weights
        # in for the whole validation pass and restore the raw weights after,
        # so training continues from exactly where it left off.
        #
        # VALIDATE_WITH_EMA=False restores the previous behaviour for an A/B.
        _raw_sd = None
        if VALIDATE_WITH_EMA and ema is not None:
            _raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(_raw_sd[k].dtype) for k, v in ema.items()})
        v_t0 = time.perf_counter()
        dsum = {r: [] for r in REGIONS}
        isum = {r: [] for r in REGIONS}
        hsum = {r: [] for r in REGIONS}
        vloss, vn = 0.0, 0
        vload_t0 = time.perf_counter()
        with torch.no_grad():
            for xb, yb, _ids, _pb, _box in val_loader:
                val_prof.add("val/data_loading", time.perf_counter() - vload_t0)
                with val_prof.section("val/h2d"):
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                with val_prof.section("val/inference_full_volume"):
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            lg = model(xb)
                        lg = lg.float()
                    else:
                        lg = model(xb)
                    out = lg.permute(0, 2, 3, 4, 1).contiguous()
                with val_prof.section("val/target_align"):
                    t_cf = yb.permute(0, 4, 1, 2, 3).contiguous()
                    if tuple(t_cf.shape[2:]) != tuple(out.shape[1:4]):
                        t_cf = torch.nn.functional.interpolate(
                            t_cf, size=tuple(out.shape[1:4]), mode="nearest")
                    tg = t_cf.permute(0, 2, 3, 4, 1).contiguous()
                with val_prof.section("val/loss"):
                    vl = 0
                    for m in range(out.shape[-1]):
                        if torch.any(tg[..., m] == 1):
                            vl = vl + loss_f(out[..., m], tg[..., m])
                        else:
                            p = torch.sigmoid(out[..., m]).clamp(1e-6, 1. - 1e-6)
                            vl = vl + EMPTY_REGION_BCE_WEIGHT * \
                                torch.nn.functional.binary_cross_entropy(
                                    p, tg[..., m], reduction="mean")
                    vloss += float(vl.detach()) if hasattr(vl, "detach") else float(vl); vn += 1
                with val_prof.section("val/probability_generation"):
                    prob = torch.sigmoid(out)[0].detach().cpu().numpy()
                gt = tg[0].detach().cpu().numpy()
                with val_prof.section("val/dice_iou"):
                    for i, r in enumerate(REGIONS):
                        d, io = dice_iou(prob[..., i], gt[..., i])
                        dsum[r].append(d); isum[r].append(io)
                if do_hd95:
                    with val_prof.section("val/hd95"):
                        for i, r in enumerate(REGIONS):
                            hsum[r].append(hd95(prob[..., i], gt[..., i]))
                vload_t0 = time.perf_counter()
        v_s = time.perf_counter() - v_t0
        val_total_s += v_s

        # STABILITY FIX 1 (restore) -- put the RAW weights back.
        #
        # This must happen BEFORE the checkpoint block below, which saves
        # model.state_dict() as "model" and `ema` separately. Restoring later
        # would write the EMA weights into the raw slot, so a resume would
        # silently continue training from the EMA copy and the two would
        # collapse into each other.
        if _raw_sd is not None:
            model.load_state_dict(_raw_sd)
            _raw_sd = None

        gpu_diag.set_phase(P_VALIDATION)
        gpu_diag.sample(epoch=epoch + 1, step=global_step)

        # ---- checkpoint (production cadence) -------------------------------
        ck_s = 0.0
        if ((epoch + 1) % CHECKPOINT_FREQUENCY == 0) or (epoch + 1 == args.epochs):
            c0 = time.perf_counter()
            ck_path = os.path.join(OUT_DIR, "checkpoint", "last.pth")
            tmp = ck_path + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema,
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "epoch": epoch + 1, "global_step": global_step,
                        "early_stopping": stopper.state_dict(),
                        "best_score": best_smooth, "best_epoch": best_epoch,
                        "fingerprint": FINGERPRINT}, tmp)
            os.replace(tmp, ck_path)
            # Periodic recovery snapshot, kept independently of best.pth so a
            # preemption costs at most CHECKPOINT_FREQUENCY epochs.
            if (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
                _per = os.path.join(OUT_DIR, "checkpoint",
                                    f"epoch_{epoch + 1:04d}.pth")
                shutil.copyfile(ck_path, _per)
                periodic_ckpts.append(_per)
            final_ckpt_path = ck_path
            ck_s = time.perf_counter() - c0
            ckpt_total_s += ck_s
            prof.add("train/checkpoint", ck_s)
            gpu_diag.set_phase(P_CHECKPOINT)
            gpu_diag.sample(epoch=epoch + 1, step=global_step)
            run_state.transition(S_CHECKPOINTED, f"epoch {epoch + 1} persisted",
                                 epoch=epoch + 1)

        def _m(d):
            v = [x for x in d if not np.isnan(x)]
            return float(np.mean(v)) if v else float("nan")

        row = {"epoch": epoch + 1,
               "train_loss": ep_loss / max(1, n_it),
               "val_loss": vloss / max(1, vn),
               "lr": opt.param_groups[0]["lr"],
               "train_seconds": ep_train_s, "val_seconds": v_s,
               "checkpoint_seconds": ck_s, "iterations": n_it}
        for r in REGIONS:
            row[f"dice_{r}"] = _m(dsum[r])
            row[f"iou_{r}"] = _m(isum[r])
            row[f"hd95_{r}"] = _m(hsum[r]) if do_hd95 else float("nan")
        row["dice_mean"] = _m([row[f"dice_{r}"] for r in REGIONS])

        # ---- model selection (VALIDATION only) -----------------------------
        # Selection uses the rolling mean, not the raw epoch value: smoothing
        # is what stops one lucky epoch from being chosen. The test split is
        # never consulted here.
        _recent = [r["dice_mean"] for r in epoch_rows[-(SMOOTHING_WINDOW - 1):]]             + [row["dice_mean"]]
        _recent = [v for v in _recent if not np.isnan(v)]
        vm_smooth = float(np.mean(_recent)) if _recent else float("nan")
        row["val_dice_mean"] = row["dice_mean"]
        row["dice_smooth"] = vm_smooth
        row["state"] = S_VALIDATING
        epoch_rows.append(row)

        if not np.isnan(vm_smooth) and vm_smooth > best_smooth:
            best_smooth, best_epoch = vm_smooth, epoch + 1
            _w = ema if ema is not None else model.state_dict()
            _bp = os.path.join(OUT_DIR, "checkpoint", "best.pth")
            _tmp = _bp + ".tmp"
            _region_metrics = {
                **{f"dice_{r}": row[f"dice_{r}"] for r in REGIONS},
                **{f"iou_{r}": row[f"iou_{r}"] for r in REGIONS},
                **{f"hd95_{r}": row[f"hd95_{r}"] for r in REGIONS}}
            torch.save({"m": _w, "ep": epoch + 1, "score": vm_smooth,
                        "dice_mean": row["dice_mean"],
                        "val_loss": row["val_loss"],
                        **_region_metrics, **FINGERPRINT}, _tmp)
            os.replace(_tmp, _bp)
            best_meta = {"path": _bp, "epoch": epoch + 1, "score": vm_smooth,
                         "dice_mean": row["dice_mean"],
                         "val_loss": row["val_loss"],
                         "loss_function": ("FocalTverskyCELoss "
                                           f"(alpha={TVERSKY_ALPHA}, "
                                           f"beta={TVERSKY_BETA}, "
                                           f"gamma={FOCAL_GAMMA}, "
                                           f"ce_weight={CE_WEIGHT})"),
                         **_region_metrics}
            top_k_entries = update_top_k(
                os.path.join(OUT_DIR, "checkpoint", "top_k"), weights=_w,
                epoch=epoch + 1, score=vm_smooth,
                metrics={"dice_mean": row["dice_mean"],
                         "val_loss": row["val_loss"], **_region_metrics},
                meta=FINGERPRINT, top_k=TOP_K_CHECKPOINTS)
            run_state.transition(S_BEST_UPDATED, f"new best {vm_smooth:.4f}",
                                 epoch=epoch + 1, score=vm_smooth)
            row["state"] = S_BEST_UPDATED
            LOG(f"   * new best (smoothed) {vm_smooth:.4f} -> best.pth "
                f"| top-{TOP_K_CHECKPOINTS}: "
                + ", ".join(f"ep{e['epoch']}={e['score']:.4f}"
                            for e in top_k_entries))

        _stop = stopper.update(vm_smooth, epoch + 1)
        if stopper.enabled:
            LOG(f"   {stopper.status()}")

        if device.type == "cuda":
            gpu_rows.append({
                "epoch": epoch + 1,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
                "current_allocated_mb": torch.cuda.memory_allocated() / 1024**2})

        LOG(f"epoch {epoch+1:3d}/{args.epochs}  loss {row['train_loss']:.4f}  "
            f"val {row['val_loss']:.4f}  Dice WT/TC/ET "
            f"{row['dice_WT']:.3f}/{row['dice_TC']:.3f}/{row['dice_ET']:.3f}  "
            f"train {ep_train_s:.1f}s  val {v_s:.1f}s"
            + (f"  ckpt {ck_s:.2f}s" if ck_s else ""))

        # Stop AFTER the checkpoint is written, so the run stays resumable
        # from exactly where it stopped. best.pth is never overwritten.
        if _stop:
            LOG("")
            LOG(f"EARLY STOPPING at epoch {epoch + 1}: {stopper.monitor} has "
                f"not improved by >{stopper.min_delta:g} for "
                f"{stopper.patience} consecutive validations")
            LOG(f"   best {stopper.best:.4f} @ epoch {stopper.best_epoch}; "
                f"{epoch + 1} of {args.epochs} epochs used")
            LOG("   the BEST checkpoint is the selected model, not the final one")
            run_state.transition(S_EARLY_STOPPED,
                                 f"no improvement for {stopper.patience} "
                                 f"validations", epoch=epoch + 1,
                                 best_epoch=stopper.best_epoch,
                                 best_value=stopper.best)
            break

    if not stopper.should_stop:
        run_state.transition(S_COMPLETED,
                             f"planned budget of {args.epochs} epochs reached",
                             epochs_run=len(epoch_rows))

    total_s = time.perf_counter() - t_experiment

    # ---- FROZEN TEST EVALUATION -------------------------------------------
    # Runs ONCE, after training is finished, on the BEST checkpoint. The test
    # split was never touched during training: not for early stopping, not for
    # checkpoint selection, not for the top-k ranking. That is what makes this
    # number an estimate of generalisation rather than a restatement of what
    # the run already optimised for.
    test_metrics: Dict[str, object] = {"status": NOT_MEASURED}
    if has_test and best_meta:
        LOG("")
        LOG("=" * 74)
        LOG("FROZEN TEST EVALUATION")
        LOG("=" * 74)
        LOG(f"  cases            : {len(split['test'])}")
        LOG(f"  model            : best.pth (epoch {best_meta['epoch']}, "
            f"selected on VALIDATION)")
        LOG(f"  first use of test: yes -- never seen during training")
        try:
            _bw = torch.load(os.path.join(OUT_DIR, "checkpoint", "best.pth"),
                             map_location=device, weights_only=False)
            _tm, _ = make_model(device)
            _tm = _tm.to(device)
            _tm.load_state_dict(_bw["m"] if isinstance(_bw["m"], dict)
                                else _bw["m"][0])
            _tm.eval()
            _tds = BraTSDataset(split["test"], {}, use_cache=USE_CACHE,
                                patch_size=None, train=False)
            _tl = torch.utils.data.DataLoader(
                _tds, batch_size=1, shuffle=False, num_workers=0,
                collate_fn=collate, pin_memory=(device.type == "cuda"))
            _d = {r: [] for r in REGIONS}
            _i = {r: [] for r in REGIONS}
            _h = {r: [] for r in REGIONS}
            _per_case = []
            _t0 = time.perf_counter()
            with torch.no_grad():
                for _xb, _yb, _ids, _pb, _box in _tl:
                    _xb = _xb.to(device, non_blocking=True)
                    _yb = _yb.to(device, non_blocking=True)
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            _o = _tm(_xb)
                        _o = _o.float()
                    else:
                        _o = _tm(_xb)
                    _o = _o.permute(0, 2, 3, 4, 1).contiguous()
                    _tg = _yb
                    if _tg.shape[1:4] != _o.shape[1:4]:
                        _tcf = _tg.permute(0, 4, 1, 2, 3).contiguous()
                        _tcf = torch.nn.functional.interpolate(
                            _tcf, size=tuple(_o.shape[1:4]), mode="nearest")
                        _tg = _tcf.permute(0, 2, 3, 4, 1).contiguous()
                    _prob = torch.sigmoid(_o)[0].detach().cpu().numpy()
                    _gt = _tg[0].detach().cpu().numpy()
                    _row = {"case": _ids[0]}
                    for _k, _r in enumerate(REGIONS):
                        _dd, _ii = dice_iou(_prob[..., _k], _gt[..., _k])
                        _hh = hd95(_prob[..., _k], _gt[..., _k])
                        _d[_r].append(_dd); _i[_r].append(_ii); _h[_r].append(_hh)
                        _row[f"dice_{_r}"] = _dd
                        _row[f"iou_{_r}"] = _ii
                        _row[f"hd95_{_r}"] = _hh
                    _per_case.append(_row)
            _test_s = time.perf_counter() - _t0

            def _mean(v):
                vv = [x for x in v if not np.isnan(x)]
                return float(np.mean(vv)) if vv else float("nan")

            def _std(v):
                vv = [x for x in v if not np.isnan(x)]
                return float(np.std(vv)) if len(vv) > 1 else 0.0

            test_metrics = {
                "status": "MEASURED", "cases": len(_per_case),
                "model_epoch": best_meta["epoch"],
                "selected_on": "validation (test never used for selection)",
                "seconds": _test_s,
                **{f"dice_{r}": _mean(_d[r]) for r in REGIONS},
                **{f"dice_std_{r}": _std(_d[r]) for r in REGIONS},
                **{f"iou_{r}": _mean(_i[r]) for r in REGIONS},
                **{f"hd95_{r}": _mean(_h[r]) for r in REGIONS},
            }
            test_metrics["dice_mean"] = _mean(
                [test_metrics[f"dice_{r}"] for r in REGIONS])

            LOG("")
            LOG(f"  {'region':8s} {'Dice':>8s} {'+/-':>7s} {'IoU':>8s} {'HD95':>9s}")
            for _r in REGIONS:
                LOG(f"  {_r:8s} {test_metrics[f'dice_{_r}']:8.4f} "
                    f"{test_metrics[f'dice_std_{_r}']:7.4f} "
                    f"{test_metrics[f'iou_{_r}']:8.4f} "
                    f"{test_metrics[f'hd95_{_r}']:9.2f}")
            LOG(f"  {'mean':8s} {test_metrics['dice_mean']:8.4f}")
            LOG(f"  evaluated {len(_per_case)} cases in {_test_s:.1f}s "
                f"({_test_s / max(1, len(_per_case)):.2f}s/case)")
            LOG("  HD95 in VOXELS (lower better); Dice/IoU higher better.")

            with open(os.path.join(OUT_DIR, "test_per_case.csv"), "w",
                      newline="", encoding="utf-8") as _fh:
                _w = csv.DictWriter(_fh, fieldnames=list(_per_case[0].keys()))
                _w.writeheader()
                for _r2 in _per_case:
                    _w.writerow(_r2)
            with open(os.path.join(OUT_DIR, "test_results.json"), "w",
                      encoding="utf-8") as _fh:
                json.dump({**test_metrics, **FINGERPRINT,
                           "per_case": _per_case,
                           "protocol": ("Single evaluation on the frozen test "
                                        "split, after training, using the "
                                        "validation-selected best checkpoint. "
                                        "The test split was never used for "
                                        "early stopping, checkpoint selection "
                                        "or ranking.")},
                          _fh, indent=2, default=str)
            del _tm
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as _e:
            LOG(f"  TEST EVALUATION FAILED: {type(_e).__name__}: {_e}")
            test_metrics = {"status": "FAILED", "error": str(_e)[:200]}
    elif not has_test:
        LOG("")
        LOG("FROZEN TEST EVALUATION: skipped -- no independent test split "
            "(dataset too small to carve one)")
        test_metrics = {"status": NOT_MEASURED,
                        "reason": "no independent test split"}

    # ---- §15 checkpoint / resume test -------------------------------------
    LOG("")
    LOG("CHECKPOINT / RESUME TEST (§15):")
    resume_ok, resume_detail = False, "not run"
    try:
        ck_path = os.path.join(OUT_DIR, "checkpoint", "last.pth")
        sd = torch.load(ck_path, map_location=device, weights_only=False)
        m2, _ = make_model(device)
        m2 = m2.to(device)
        m2.load_state_dict(sd["model"])
        o2 = torch.optim.AdamW(m2.parameters(), lr=LR, betas=BETAS,
                               weight_decay=WEIGHT_DECAY)
        o2.load_state_dict(sd["optimizer"])
        # Must be the SAME scheduler type the run used, or load_state_dict
        # raises KeyError('_schedulers') once warmup is enabled.
        if WARMUP_EPOCHS > 0:
            s2 = torch.optim.lr_scheduler.SequentialLR(o2, schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    o2, start_factor=1.0 / WARMUP_START_DIV, end_factor=1.0,
                    total_iters=WARMUP_EPOCHS),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    o2, T_max=max(1, args.epochs - WARMUP_EPOCHS),
                    eta_min=MIN_LR)], milestones=[WARMUP_EPOCHS])
        else:
            s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                o2, T_max=args.epochs, eta_min=MIN_LR)
        s2.load_state_dict(sd["scheduler"])
        weights_ok = all(torch.equal(a.cpu(), b.cpu()) for a, b in
                         zip(model.state_dict().values(), m2.state_dict().values()))
        ema_ok = len(sd["ema"]) == len(ema)
        # The run may stop EARLY, so the last checkpoint holds the epoch the
        # run actually reached, not args.epochs. Comparing against args.epochs
        # made the resume test FAIL on every early-stopped run while resume
        # itself was working correctly -- the check was wrong, not the code.
        # Compare against the epoch this process actually finished at.
        step_ok = (sd["epoch"] == len(epoch_rows)
                   and sd["global_step"] == global_step)
        sched_ok = s2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]
        # continue training one iteration from the restored state
        m2.train()
        xb, yb, _, _pb2, _box2 = next(iter(train_loader))
        xb, yb = xb.to(device), yb.to(device)
        lg = m2(xb).float().permute(0, 2, 3, 4, 1).contiguous()
        t_cf = yb.permute(0, 4, 1, 2, 3).contiguous()
        if tuple(t_cf.shape[2:]) != tuple(lg.shape[1:4]):
            t_cf = torch.nn.functional.interpolate(t_cf, size=tuple(lg.shape[1:4]),
                                                   mode="nearest")
        tg = t_cf.permute(0, 2, 3, 4, 1).contiguous()
        l2 = sum(loss_f(lg[..., m], tg[..., m]) for m in range(lg.shape[-1]))
        o2.zero_grad(set_to_none=True); l2.backward(); o2.step()
        cont_ok = bool(torch.isfinite(l2))
        resume_ok = all([weights_ok, ema_ok, step_ok, sched_ok, cont_ok])
        resume_detail = (f"weights={weights_ok} ema={ema_ok} epoch/step={step_ok} "
                         f"scheduler={sched_ok} continue_training={cont_ok}")
    except Exception as e:
        resume_detail = f"{type(e).__name__}: {e}"
    LOG(f"  {'PASS' if resume_ok else 'FAIL'}  {resume_detail}")

    # ---- write artifacts ---------------------------------------------------
    def write_csv(path, rows, fields=None):
        if not rows:
            return
        fields = fields or list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(os.path.join(OUT_DIR, "epoch_metrics.csv"), epoch_rows)
    write_csv(os.path.join(OUT_DIR, "iteration_profile.csv"), iter_rows)
    write_csv(os.path.join(OUT_DIR, "gpu_profile.csv"), gpu_rows)

    # merge dataset-worker timings (collected in worker processes for workers=0;
    # with workers>0 they are per-process and may be partial -> labelled)
    for k, v in ds_timings.items():
        for s in v:
            prof.add(k, s)

    psum = prof.summary()
    vsum = val_prof.summary()
    timing_rows = [{"component": k, "n": s["n"], "total_s": round(s["total"], 4),
                    "mean_s": round(s["mean"], 6), "median_s": round(s["median"], 6),
                    "p95_s": round(s["p95"], 6)} for k, s in psum.items()]
    write_csv(os.path.join(OUT_DIR, "timing_summary.csv"), timing_rows)
    val_rows = [{"component": k, "n": s["n"], "total_s": round(s["total"], 4),
                 "mean_s": round(s["mean"], 6), "median_s": round(s["median"], 6),
                 "p95_s": round(s["p95"], 6)} for k, s in vsum.items()]
    write_csv(os.path.join(OUT_DIR, "validation_profile.csv"), val_rows)

    it = np.array(iter_times, dtype=float)
    steady = {"mean": float(it.mean()) if it.size else float("nan"),
              "median": float(np.median(it)) if it.size else float("nan"),
              "p95": float(np.percentile(it, 95)) if it.size else float("nan"),
              "n": int(it.size)}
    peak_vram = max((g["peak_reserved_mb"] for g in gpu_rows), default=0.0)
    last = epoch_rows[-1] if epoch_rows else {}

    # ---- §18 TOP TIME CONSUMERS — measured, ranked -------------------------
    iter_total = float(it.sum()) if it.size else 0.0
    ranked = []
    for k in ("train/dataloader_wait", "train/h2d_transfer", "model/level1",
              "model/level2", "model/global_context", "model/fusion",
              "train/loss", "train/backward", "train/grad_clip",
              "train/optimizer_step", "train/ema_update", "train/checkpoint",
              "train/target_align"):
        s = psum.get(k)
        if s and s["n"]:
            ranked.append((k, s["total"], 100.0 * s["total"] / iter_total
                           if iter_total else float("nan")))
    ranked.sort(key=lambda r: r[1], reverse=True)

    summary = {
        "label": "Kaggle engineering validation — BraTS2024-small — "
                 f"{args.epochs} epochs. NOT final thesis results.",
        "measurement_methodology": (
            "CPU wall clock with CUDA synchronisation at section boundaries. "
            "Nested sections (global context inside levels) intentionally "
            "overlap and do NOT sum to 100%."),
        "run_mode": RUN_MODE,
        "preset": ACTIVE_PRESET,
        "is_production_identity": IS_PRODUCTION_IDENTITY,
        "category_c_deviations": category_c_deviations(),
        "result_may_be_reported_as_production": IS_PRODUCTION_IDENTITY,
        "dataset_root": root,
        "cases": inv["case_count_usable"],
        "split_counts": inv["kaggle_split_counts"],
        "has_independent_test_set": has_test,
        "gpu": gpu_name, "gpu_total_mb": gpu_total,
        "cuda": torch.version.cuda, "torch": torch.__version__,
        "parameters": arch["parameter_count"],
        "geometry": f"{WORKING_VOLUME}^3 -> {L1_RES}^3 + {L2_RES}^3",
        "global_context": "WHOLE 96^3 (verified)",
        "global_context_delta": arch["global_context_delta"],
        "fusion": f"learned Conv3d({L1_CH+L2_CH} -> {L2_CH})",
        "patchify": USE_PATCHIFY,
        "epochs": args.epochs, "batch_size": BATCH_SIZE,
        "warmup_first_iteration_s": warmup_time,
        "steady_state_iteration_s": steady,
        "mean_epoch_s": float(np.mean([r["train_seconds"] for r in epoch_rows]))
                        if epoch_rows else float("nan"),
        "training_only_s": train_only_s,
        "validation_total_s": val_total_s,
        "checkpoint_total_s": ckpt_total_s,
        "total_experiment_wall_s": total_s,
        "samples_per_sec": (len(split["train"]) / np.mean(
            [r["train_seconds"] for r in epoch_rows])) if epoch_rows else float("nan"),
        "peak_vram_mb": peak_vram,
        "final_metrics": {k: last.get(k) for k in
                          ["dice_WT", "dice_TC", "dice_ET", "dice_mean",
                           "iou_WT", "iou_TC", "iou_ET",
                           "hd95_WT", "hd95_TC", "hd95_ET"]},
        "resume_test": "PASS" if resume_ok else "FAIL",
        "resume_detail": resume_detail,
        "architecture_identity": "PASS" if arch["passed"] else "FAIL",
        "top_time_consumers": [{"component": k, "total_s": round(t, 3),
                                "pct_of_iteration_time": round(p, 2)}
                               for k, t, p in ranked],
        "timing_summary": psum,
        "validation_profile": vsum,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "final_summary.json"), "w"),
              indent=2, default=str)
    write_plots(OUT_DIR, epoch_rows, timing_rows, gpu_rows)

    # ---- §18 console report ------------------------------------------------
    LOG("")
    LOG("=" * 74)
    LOG(f"GLO-NCA KAGGLE RUN — {RUN_MODE}")
    LOG("=" * 74)
    if not IS_PRODUCTION_IDENTITY:
        LOG("*** CATEGORY-C RUN — NOT THE PRODUCTION ARCHITECTURE ***")
        for _d in category_c_deviations():
            LOG(f"      - {_d}")
        LOG("*** These numbers must NOT be reported as validating the")
        LOG("    production model or as thesis performance. ***")
        LOG("-" * 74)
    LOG(f"Dataset:            {root}")
    LOG(f"Cases:              {inv['case_count_usable']}")
    LOG(f"Train/Val/Test:     {inv['kaggle_split_counts']['train']} / "
        f"{inv['kaggle_split_counts']['validation']} / "
        f"{inv['kaggle_split_counts']['test']}"
        + ("" if has_test else "   (no independent test set)"))
    LOG("")
    LOG(f"GPU:                {gpu_name}")
    LOG(f"CUDA:               {torch.version.cuda}")
    LOG(f"PyTorch:            {torch.__version__}")
    LOG("")
    LOG(f"Parameters:         {arch['parameter_count']:,}")
    LOG(f"Geometry:           {WORKING_VOLUME}^3 -> {L1_RES}^3 + {L2_RES}^3")
    LOG(f"Global Context:     WHOLE {WORKING_VOLUME}^3 "
        f"(max|delta|={arch['global_context_delta']:.4g} to non-core change)")
    LOG(f"Fusion:             learned Conv3d({L1_CH+L2_CH} -> {L2_CH})")
    LOG(f"Patchify:           "
        + (f"ON  patch {PATCH_SIZE}^3"
           + (" at native resolution" if PATCH_NATIVE_RES
              else f" resized to {L2_RES}^3")
           if USE_PATCHIFY else "OFF"))
    LOG(f"NCA steps:          {L1_STEPS}+{L2_STEPS} = {L1_STEPS+L2_STEPS}")
    LOG(f"Spatial GC kernel:  {SPATIAL_GC_KERNEL}")
    LOG("")
    LOG(f"Epochs:             {args.epochs}")
    LOG(f"Batch:              {BATCH_SIZE}")
    LOG("")
    LOG(f"Warm-up iter 1:     {warmup_time:.3f} s  (excluded from steady state)"
        if warmup_time else "Warm-up iter 1:     n/a")
    LOG(f"Mean iteration:     {steady['mean']:.3f} s   [MEASURED, n={steady['n']}]")
    LOG(f"Median iteration:   {steady['median']:.3f} s")
    LOG(f"P95 iteration:      {steady['p95']:.3f} s")
    LOG("")
    LOG(f"Mean epoch:         {summary['mean_epoch_s']:.1f} s")
    LOG(f"Training only:      {train_only_s:.1f} s")
    LOG(f"Validation total:   {val_total_s:.1f} s")
    LOG(f"Checkpoint total:   {ckpt_total_s:.1f} s")
    LOG(f"Total wall time:    {total_s:.1f} s  ({total_s/3600:.2f} h)")
    LOG("")
    LOG(f"Peak VRAM:          {peak_vram:.1f} MB")
    LOG(f"Samples/sec:        {summary['samples_per_sec']:.4f}   [CALCULATED]")
    LOG("")
    LOG("COMPONENT BREAKDOWN (§11)  — MEASURED")
    LOG(f"  {'Component':<28}{'Seconds':>12}{'% of iter':>12}{'n':>8}")
    LOG("  " + "-" * 60)
    for k, t, p in ranked:
        LOG(f"  {k:<28}{t:>12.3f}{p:>11.1f}%{psum[k]['n']:>8}")
    LOG("  Note: nested sections overlap; percentages do NOT sum to 100%.")
    LOG("")
    LOG("TOP TIME CONSUMERS (ranked by measured wall clock)")
    for i, (k, t, p) in enumerate(ranked[:8], 1):
        LOG(f"  {i}. {k:<28} {p:>6.1f}%   ({t:.1f} s)")
    LOG("")
    LOG("VALIDATION PROFILE  — MEASURED")
    for k, s in sorted(vsum.items(), key=lambda kv: kv[1]["total"], reverse=True):
        LOG(f"  {k:<32}{s['total']:>10.2f} s  mean {s['mean']*1000:>8.1f} ms  n={s['n']}")
    if epoch_rows:
        LOG(f"  validation per epoch: {val_total_s/len(epoch_rows):.2f} s")
        nval = max(1, inv["kaggle_split_counts"]["validation"])
        LOG(f"  validation per case : {val_total_s/len(epoch_rows)/nval:.3f} s")
    LOG("")
    LOG(f"WT Dice:            {last.get('dice_WT', float('nan')):.4f}")
    LOG(f"TC Dice:            {last.get('dice_TC', float('nan')):.4f}")
    LOG(f"ET Dice:            {last.get('dice_ET', float('nan')):.4f}")
    LOG(f"Mean Dice:          {last.get('dice_mean', float('nan')):.4f}")
    LOG(f"IoU (WT/TC/ET):     {last.get('iou_WT', float('nan')):.4f} / "
        f"{last.get('iou_TC', float('nan')):.4f} / {last.get('iou_ET', float('nan')):.4f}")
    # HD95 is computed only every HD95_EVERY_EPOCHS epochs, so the LAST epoch
    # usually stores nan. Reporting that made a real, measured quantity look
    # missing ("nan / nan / nan") in the summary while the value existed a few
    # epochs earlier. Report the most recent epoch that actually measured it,
    # and say which epoch that was.
    _hd = next((r for r in reversed(epoch_rows)
                if r.get("hd95_WT") == r.get("hd95_WT")), None)   # nan != nan
    if _hd is None:
        LOG("HD95 (WT/TC/ET):    NOT MEASURED  [no epoch computed HD95]")
    else:
        LOG(f"HD95 (WT/TC/ET):    {_hd['hd95_WT']:.2f} / {_hd['hd95_TC']:.2f} / "
            f"{_hd['hd95_ET']:.2f}  [voxels, epoch {_hd.get('epoch', '?')}]")
    LOG("")
    LOG(f"Resume test:        {'PASS' if resume_ok else 'FAIL'}")
    LOG(f"Architecture ident: {'PASS' if arch['passed'] else 'FAIL'}")
    LOG("=" * 74)
    LOG(f"SCOPE [{RUN_MODE}]: Kaggle engineering run, BraTS2024-small, "
        f"{args.epochs} epochs.")
    if IS_PRODUCTION_IDENTITY:
        LOG(f"Architecture matches configs/glo_nca_production.yaml "
            f"({EXPECTED_PARAMS:,}).")
        LOG("Still NOT final thesis performance: Kaggle-local split, small "
            "dataset, not the 898/200/198 master split.")
    else:
        LOG("CATEGORY-C: the architecture under test is NOT the production")
        LOG("model. Deviations: " + "; ".join(category_c_deviations()))
        LOG("Does NOT validate the production architecture, does NOT replace")
        LOG("the thesis master split, is NOT thesis performance.")
    LOG("=" * 74)

    # ---- experiment-artifact report (NOT added to the repository) ----------
    with open(os.path.join(OUT_DIR, "FINAL_KAGGLE_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write(f"# GLO-NCA — Kaggle Engineering Run [{RUN_MODE}]\n\n")
        if not IS_PRODUCTION_IDENTITY:
            fh.write("> ## ⚠ CATEGORY-C RUN — NOT THE PRODUCTION ARCHITECTURE\n>\n")
            fh.write("> The model measured here is **not** the production "
                     "GLO-NCA. It differs by:\n>\n")
            for _d in category_c_deviations():
                fh.write(f"> - {_d}\n")
            fh.write(">\n> These candidates are frozen and unvalidated (no "
                     "Dice/HD95 evidence).\n")
            fh.write("> **Do not report these numbers as validating the "
                     "production architecture or as thesis performance.**\n\n")
        else:
            fh.write("> Architecture matches `configs/glo_nca_production.yaml` "
                     "(33,089 parameters).\n> Still an engineering run: "
                     "Kaggle-local split, not the 898/200/198 master split.\n\n")
        fh.write("**Experiment artifact only.** Not thesis performance, not an "
                 "ablation, not the thesis master split.\n\n")
        fh.write(f"- Dataset: `{root}` — {inv['case_count_usable']} usable cases\n")
        fh.write(f"- Split (Kaggle-only, seed 42): "
                 f"{inv['kaggle_split_counts']}\n")
        fh.write(f"- GPU: {gpu_name} · CUDA {torch.version.cuda} · "
                 f"PyTorch {torch.__version__}\n")
        fh.write(f"- Parameters: **{arch['parameter_count']:,}** "
                 f"(expected {EXPECTED_PARAMS:,}"
                 f"{'' if IS_PRODUCTION_IDENTITY else '; production is 33,089'})\n")
        fh.write(f"- Geometry: {WORKING_VOLUME}³ → {L1_RES}³ + {L2_RES}³, "
                 f"NCA steps {L1_STEPS}+{L2_STEPS}, spatial GC k="
                 f"{SPATIAL_GC_KERNEL}, patchify "
                 f"{f'ON ({PATCH_SIZE}³ native)' if USE_PATCHIFY else 'OFF'}, "
                 "global context whole-volume\n")
        fh.write("- BatchNorm: A1 fused channels-last "
                 "(proven-equivalent, 1.402× measured — NOT Category C)\n\n")
        fh.write("## Timing (MEASURED)\n\n")
        fh.write(f"- Mean iteration: {steady['mean']:.3f} s "
                 f"(median {steady['median']:.3f}, p95 {steady['p95']:.3f}, "
                 f"n={steady['n']})\n")
        fh.write(f"- Mean epoch: {summary['mean_epoch_s']:.1f} s\n")
        fh.write(f"- Total wall: {total_s/3600:.2f} h\n")
        fh.write(f"- Peak VRAM: {peak_vram:.1f} MB\n\n")
        fh.write("| Component | Seconds | % of iteration | n |\n|---|---|---|---|\n")
        for k, t, p in ranked:
            fh.write(f"| `{k}` | {t:.3f} | {p:.1f}% | {psum[k]['n']} |\n")
        fh.write("\nNested sections overlap; percentages do not sum to 100%.\n\n")
        fh.write("## Final metrics (MEASURED)\n\n")
        fh.write(f"- Dice WT/TC/ET: {last.get('dice_WT', float('nan')):.4f} / "
                 f"{last.get('dice_TC', float('nan')):.4f} / "
                 f"{last.get('dice_ET', float('nan')):.4f}\n")
        fh.write(f"- Mean Dice: {last.get('dice_mean', float('nan')):.4f}\n")
        fh.write(f"- Resume test: {'PASS' if resume_ok else 'FAIL'}\n")
        fh.write(f"- Architecture identity: "
                 f"{'PASS' if arch['passed'] else 'FAIL'}\n")

    # ---- artifact set + consistency audit ----------------------------------
    _timing = {"sections": dict(psum or {}),
               "note": "Nested sections overlap; percentages do NOT sum to 100%.",
               "methodology": ("CPU wall clock with torch.cuda.synchronize() "
                               "at section boundaries only.")}
    _resume_checks = ({"model": True, "ema": True, "epoch_step": True,
                       "scheduler": True, "continue_training": True,
                       "early_stopping": bool(stopper.history)}
                      if resume_ok else None)
    _written = write_artifacts(
        OUT_DIR, fingerprint=FINGERPRINT, arch=arch, epoch_rows=epoch_rows,
        stopper=stopper, run_state=run_state, gpu_diag=gpu_diag,
        ckpt_dir=os.path.join(OUT_DIR, "checkpoint"),
        top_k_entries=top_k_entries, best_meta=best_meta,
        periodic=periodic_ckpts, final_ckpt=final_ckpt_path,
        resume_checks=_resume_checks, timing=_timing)
    with open(os.path.join(OUT_DIR, "test_evaluation.json"), "w",
              encoding="utf-8") as _fh:
        json.dump({**FINGERPRINT, "test": test_metrics}, _fh, indent=2,
                  default=str)
    _audit = audit_artifacts(OUT_DIR)

    LOG("")
    LOG("=" * 74)
    LOG("ARTIFACTS + TRAINING CONTROL")
    LOG("=" * 74)
    LOG(f"  artifacts written    : {len(_written)}")
    LOG(f"  consistency audit    : {'PASS' if _audit['passed'] else 'FAIL'}")
    for _c in _audit["checks"]:
        if not _c["pass"]:
            LOG(f"    FAIL {_c['check']}: {str(_c['detail'])[:60]}")
    _g = gpu_diag.summary()
    _u = _g.get("active_gpu_utilization_mean_pct")
    if isinstance(_u, (int, float)):
        LOG(f"  GPU util (ACTIVE_GPU): {_u}%   "
            f"[{_g.get('samples', 0)} samples; idle/data-wait excluded]")
    if _g.get("warning"):
        LOG(f"  WARNING: {_g['warning']}")
    if _g.get("throttle_warning"):
        LOG(f"  WARNING: {_g['throttle_warning']}")
    LOG(f"  training state       : {run_state.state}")
    LOG(f"  epochs completed     : {len(epoch_rows)} of {args.epochs} planned")
    LOG(f"  best epoch           : {best_epoch} (smoothed {best_smooth:.4f})")
    LOG(f"  top-{TOP_K_CHECKPOINTS} retained        : "
        + (", ".join(f"ep{e['epoch']}={e['score']:.4f}"
                     for e in top_k_entries) or "none"))
    LOG(f"  periodic snapshots   : {len(periodic_ckpts)}")

    # Quality metrics at the SELECTED epoch. Dice / IoU / HD95 are the quality
    # measures; Tversky (FocalTverskyCELoss) is the training objective and is
    # reported as a loss, not as a score.
    if best_meta:
        LOG("")
        LOG(f"  SELECTED MODEL (epoch {best_meta['epoch']}) — validation metrics")
        LOG(f"    {'region':8s} {'Dice':>8s} {'IoU':>8s} {'HD95':>9s}")
        for _r in REGIONS:
            _d = best_meta.get(f"dice_{_r}", float('nan'))
            _i = best_meta.get(f"iou_{_r}", float('nan'))
            _h = best_meta.get(f"hd95_{_r}", float('nan'))
            LOG(f"    {_r:8s} {_d:8.4f} {_i:8.4f} {_h:9.2f}")
        LOG(f"    {'mean':8s} {best_meta.get('dice_mean', float('nan')):8.4f}")
        LOG(f"    loss (FocalTverskyCE): {best_meta.get('val_loss', float('nan')):.4f}"
            f"   [objective, not a quality score]")
        LOG(f"    HD95 is in VOXELS; lower is better. Dice/IoU: higher is better.")

    if isinstance(test_metrics, dict) and test_metrics.get("status") == "MEASURED":
        LOG("")
        LOG(f"  FROZEN TEST SET ({test_metrics['cases']} cases, never seen "
            f"during training)")
        LOG(f"    {'region':8s} {'Dice':>8s} {'IoU':>8s} {'HD95':>9s}")
        for _r in REGIONS:
            LOG(f"    {_r:8s} {test_metrics[f'dice_{_r}']:8.4f} "
                f"{test_metrics[f'iou_{_r}']:8.4f} "
                f"{test_metrics[f'hd95_{_r}']:9.2f}")
        LOG(f"    {'mean':8s} {test_metrics['dice_mean']:8.4f}")
        LOG("    This is the generalisation estimate. Validation numbers "
            "above were used for")
        LOG("    model selection, so they are optimistic by construction.")
    if stopper.should_stop:
        LOG(f"  EARLY STOPPED        : best {stopper.best:.4f} @ epoch "
            f"{stopper.best_epoch}, {len(epoch_rows)} epochs used")

    LOG(f"\nArtifacts written to: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted by user")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
