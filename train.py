r"""
================================================================================
GLO-NCA -- clean training entry point (v7 recipe, GCP / Docker ready)
================================================================================
Global Context-Aware Neural Cellular Automata for multi-modal (BraTS) brain
tumour segmentation. This is the canonical, professional entry point used for
the full-dataset (882-case) training on GCP; the ``kaggle_v*.py`` scripts are
kept as the experiment history that led here.

This script is the v7 recipe with the two decisions from the 200-case runs
baked in:
  * ensemble + TTA are REMOVED  -- they washed out the small TC/ET regions
    (plain 0.766 beat ensemble+TTA 0.738 on the 200-case test), so inference
    is a single clean forward pass.
  * the winning preprocessing (foreground crop + non-zero z-norm), the spatial
    global-context block, ET-aware patch sampling, Focal-Tversky loss, light
    dropout, EMA and gradient-norm clipping are all kept.

It uses the real ``Dataset_NiiGz_3D_BraTS`` (driven by config flags) instead of
duplicating the crop/normalise logic in a subclass, and does proper
gradient-NORM clipping (not a per-element hook that lingers on the parameters).

Paths are read from the environment so the same image runs on Kaggle, a laptop
or a GCP VM without editing code:

    DATA_ROOT   BraTS root: one sub-folder per patient (auto-detected if unset)
    OUT_DIR     where checkpoints / metrics / curves are written (default ./out)
    EPOCHS, N_PATIENTS, SEED, BATCH_SIZE, NUM_WORKERS   optional overrides

Run locally:
    DATA_ROOT=/path/to/BraTS OUT_DIR=./out python train.py

Run in Docker (see Dockerfile / README):
    docker run --gpus all -e DATA_ROOT=/data -e OUT_DIR=/out \
        -v /mnt/brats:/data -v /mnt/ckpt:/out glo-nca
================================================================================
"""
import os
import sys
import time
import json
import math
import random

import numpy as np
import torch

# Make ``src`` importable whether run from the repo root or elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import FocalTverskyCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_GLO_NCA import Agent_GLO_NCA
from src.agents.Agent import iou_score, hd95_score


# =============================================================================
# Configuration (env-overridable; defaults are the v7 recipe)
# =============================================================================
def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


EPOCHS      = _env_int("EPOCHS", 150)
N_PATIENTS  = _env_int("N_PATIENTS", 0) or None   # 0/unset -> use all patients
SEED        = _env_int("SEED", 42)
BATCH_SIZE  = _env_int("BATCH_SIZE", 1)
NUM_WORKERS = _env_int("NUM_WORKERS", 4)

# --- model / cascade (v5+ architecture) ---
CHANNEL_N   = 24
HIDDEN      = 128
USE_SPATIAL = True                       # spatial global-context block (GC)
STEPS       = [20, 20]                    # inference steps per level (low, high)
FIRE_RATE   = 0.6

# High-res patch size is an env knob so you can test VRAM headroom without
# editing code: PATCH=64 (laptop / Kaggle-safe), 96 (balanced), 128 (max context,
# GCP-class GPU only -- can OOM a 6 GB card). The low-res level is always the
# high-res size halved. Z is kept ~0.75*XY (BraTS volumes are thinner in Z).
_PATCH = _env_int("PATCH", 96)
_PATCH_TABLE = {
    64:  [[32, 32, 24], [64, 64, 48]],
    96:  [[48, 48, 32], [96, 96, 64]],
    128: [[64, 64, 48], [128, 128, 96]],
}
INPUT_SIZE = _PATCH_TABLE.get(_PATCH, _PATCH_TABLE[96])

# --- training ---
LR_START     = 16e-4
LR_MIN       = 1e-5
TVERSKY_BETA = 0.75                       # beta>alpha -> penalise missed tumour
FOCAL_GAMMA  = 1.33                       # focus loss on the hard ET region
DROPOUT      = 0.1
USE_EMA      = True
EMA_DECAY    = 0.999
GRAD_CLIP    = 1.0                        # gradient-NORM clip (0 disables)
BEST_WINDOW  = 3                          # epochs averaged for best-model choice

# --- preprocessing (the v3->v4 win) ---
USE_FOREGROUND_CROP = True
USE_NONZERO_NORM    = True
USE_AUG             = True                # anti-overfit; safe at 882-case scale
AUG_LEVEL           = os.environ.get("AUG_LEVEL", "heavy")  # 'light' | 'heavy'
PRIORITIZE_REGION   = 2                   # bias patches toward ET (rarest)
PRIORITIZE_MASKS    = 0.7                 # P(patch must contain the region)

REGIONS    = ["WT", "TC", "ET"]
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

OUT_DIR = os.environ.get("OUT_DIR", os.path.join(_HERE, "out"))


# =============================================================================
# Data root discovery
# =============================================================================
def find_data_root():
    r"""Resolve the BraTS root. Prefer $DATA_ROOT; else auto-detect the first
    directory that contains >=2 patient sub-folders holding .nii/.nii.gz files.
    """
    env = os.environ.get("DATA_ROOT")
    if env and os.path.isdir(env):
        return env, sum(os.path.isdir(os.path.join(env, d)) for d in os.listdir(env))
    for base in ("/kaggle/input", "/data", _HERE):
        if not os.path.isdir(base):
            continue
        for root, dirs, _ in os.walk(base):
            c = 0
            for d in dirs:
                try:
                    if any(f.endswith((".nii", ".nii.gz"))
                           for f in os.listdir(os.path.join(root, d))):
                        c += 1
                except OSError:
                    pass
            if c >= 2:
                return root, c
    return None, 0


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id):
    r"""Seed each DataLoader worker so the random patch sampling is reproducible
    even with num_workers > 0 (otherwise every worker shares the same RNG state
    and the run is not reproducible)."""
    s = (SEED + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def make_split(data_root, seed):
    pats = sorted(d for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d)))
    random.Random(seed).shuffle(pats)
    if N_PATIENTS:
        pats = pats[:N_PATIENTS]
    n = len(pats)
    a, b = int(n * 0.70), int(n * 0.15)
    return pats[:a], pats[a:a + b], pats[a + b:]


# =============================================================================
# Evaluation (single clean forward pass -- no ensemble, no TTA)
# =============================================================================
def _collect_probs(agent, dataset, state):
    r"""Run one clean forward pass over ``state`` and return the per-case
    (probability, ground-truth) volume pairs.

    The 200-case study showed the stochastic pseudo-ensemble + flip-TTA lowered
    TC/ET, so inference is a single forward pass; the only stochasticity left is
    the NCA fire-rate, which we do not average over. Collecting probs once lets
    us both score and tune the decision threshold without re-running the model.
    """
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    pairs = []
    with torch.no_grad():
        for data in loader:
            data = agent.prepare_data(data, eval=True)
            out, targets = agent.get_outputs(data, full_img=True)
            prob = torch.sigmoid(out).detach().cpu().numpy()
            gt = targets.detach().cpu().numpy()
            pairs.append((prob, gt))
    agent.exp.set_model_state("train")
    return pairs


def _score(pairs, thresholds):
    r"""Dice / mIoU / HD95 per region from collected (prob, gt) pairs, using a
    per-region decision threshold ``thresholds`` (dict region -> float)."""
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    for prob, gt in pairs:
        for i, r in enumerate(REGIONS):
            th = thresholds[r]
            p, t = prob[..., i], gt[..., i]
            inter = np.logical_and(p >= th, t >= 0.5).sum()
            denom = (p >= th).sum() + (t >= 0.5).sum() + 1e-6
            acc[r]["dice"].append((2 * inter) / denom)
            acc[r]["iou"].append(iou_score(p, t, threshold=th))
            acc[r]["hd95"].append(hd95_score(p, t, threshold=th))
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])),
                  "iou": float(np.mean(acc[r]["iou"])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out


def evaluate(agent, dataset, state, thresholds=None):
    r"""Per-region Dice / mIoU / HD95 on ``state`` (single forward pass).
        #Args
            thresholds: optional dict region->float; defaults to 0.5 each.
    """
    if thresholds is None:
        thresholds = {r: 0.5 for r in REGIONS}
    return _score(_collect_probs(agent, dataset, state), thresholds)


def tune_thresholds(pairs, grid=None):
    r"""Pick the per-region decision threshold that maximises mean Dice on the
    given (prob, gt) pairs. Tuned on VALIDATION only, then applied to test --
    a standard, legitimate post-processing step. Tiny regions (ET/TC) often peak
    below 0.5, so this recovers Dice at zero training cost.
    """
    if grid is None:
        grid = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    best = {}
    for i, r in enumerate(REGIONS):
        best_th, best_d = 0.5, -1.0
        for th in grid:
            dices = []
            for prob, gt in pairs:
                p, t = prob[..., i], gt[..., i]
                inter = np.logical_and(p >= th, t >= 0.5).sum()
                denom = (p >= th).sum() + (t >= 0.5).sum() + 1e-6
                dices.append((2 * inter) / denom)
            d = float(np.mean(dices))
            if d > best_d:
                best_d, best_th = d, th
        best[r] = best_th
    return best


# =============================================================================
# Diagnostics -- turn the run into a "what helped / what to change" report
# =============================================================================
def _diagnose(hist, ck, val_pairs, test_pairs, test_05, test, thresholds, peak):
    r"""Summarise the run into decision signals with plain verdicts, so a cheap
    Kaggle run tells you what to keep before paying for the full GCP run.
    Returns {"lines": [...printable...], "verdict": str, plus raw numbers}.
    """
    lines, raw = [], {}

    # 1) Overfitting: gap between the best VAL mean and the TEST mean (@0.5).
    val_best = max(hist["val_mean"]) if hist["val_mean"] else 0.0
    test_mean_05 = float(np.mean([test_05[r]["dice"] for r in REGIONS]))
    gap = val_best - test_mean_05
    raw["val_best"], raw["test_mean_05"], raw["val_test_gap"] = val_best, test_mean_05, gap
    tag = "GOOD" if gap < 0.03 else ("OK" if gap < 0.06 else "WATCH: overfitting / small-val noise")
    lines.append(f"[overfit ] val_best {val_best:.3f} -> test {test_mean_05:.3f} "
                 f"(gap {gap:+.3f})  -> {tag}")

    # 2) Threshold tuning: did moving off 0.5 help, and where?
    per = []
    for r in REGIONS:
        d = test[r]["dice"] - test_05[r]["dice"]
        per.append(f"{r}{d:+.3f}@{thresholds[r]:.2f}")
    tmean = float(np.mean([test[r]["dice"] for r in REGIONS])) - test_mean_05
    raw["threshold_gain_mean"] = tmean
    tag = "GOOD: keep tuned thresholds" if tmean > 0.005 else "OK: 0.5 is already fine"
    lines.append(f"[thresh  ] tuning gain {tmean:+.3f} ({', '.join(per)})  -> {tag}")

    # 3) Convergence: is val still rising at the end (undertrained) or plateaued?
    vm = hist["val_mean"]
    if len(vm) >= 10:
        late = np.mean(vm[-5:]) - np.mean(vm[-10:-5])
        raw["late_slope"] = float(late)
        if late > 0.01:
            tag = "WATCH: still rising -> train MORE epochs"
        elif late < -0.02:
            tag = "WATCH: val falling -> overfit late, best-epoch earlier / fewer epochs"
        else:
            tag = "GOOD: plateaued (epoch budget about right)"
        lines.append(f"[converge] last-5 vs prev-5 val change {late:+.3f}  -> {tag}")

    # 4) Per-region weakness: which region is dragging the mean?
    worst = min(REGIONS, key=lambda r: test[r]["dice"])
    raw["worst_region"] = worst
    lines.append(f"[weakest ] {worst} is lowest at {test[worst]['dice']:.3f}  "
                 f"-> focus next tweaks here (sampling / loss weight)")

    # 5) Best epoch position -- very early best on many epochs = noise/overfit.
    if hist["epoch"]:
        frac = ck["ep"] / max(hist["epoch"])
        raw["best_epoch_frac"] = float(frac)
        tag = ("WATCH: best came early -> likely lucky/overfit" if frac < 0.4
               else "GOOD: best in the mature phase")
        lines.append(f"[bestep  ] best @ {ck['ep']}/{max(hist['epoch'])} "
                     f"({frac*100:.0f}%)  -> {tag}")

    # 6) VRAM headroom -- tells you if a bigger patch is feasible next.
    if peak > 0:
        tag = ("room to grow: try larger PATCH" if peak < 8 else
               "near a 16 GB budget" if peak < 14 else "WATCH: close to OOM")
        lines.append(f"[vram    ] peak {peak:.2f} GB  -> {tag}")

    # Overall verdict.
    if gap < 0.03 and (len(vm) < 10 or -0.02 <= raw.get("late_slope", 0) <= 0.01):
        verdict = f"Healthy run. Weakest region = {worst}. Config looks GCP-ready."
    elif gap >= 0.06:
        verdict = ("Large val->test gap: small-val noise or overfitting. On 882 "
                   "cases this should shrink; if not, add regularisation / more data.")
    else:
        verdict = (f"Usable. Address the WATCH lines above (weakest = {worst}) "
                   f"before the full GCP run.")
    return {"lines": lines, "verdict": verdict, "raw": raw}


# =============================================================================
# Training
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    data_root, nf = find_data_root()
    assert data_root, ("BraTS dataset not found. Set DATA_ROOT to the folder "
                       "that holds one sub-folder per patient.")
    print(f"DATA_ROOT = {data_root} ({nf} patients)")

    tr, va, te = make_split(data_root, SEED)
    print(f"\nSplit -> train {len(tr)} | val {len(va)} | test {len(te)}")
    print(f"GLO-NCA: ch={CHANNEL_N} hidden={HIDDEN} steps={STEPS} fire={FIRE_RATE} "
          f"patch={INPUT_SIZE[-1]} beta={TVERSKY_BETA} gamma={FOCAL_GAMMA} "
          f"aug={USE_AUG}/{AUG_LEVEL} ema={USE_EMA} clip={GRAD_CLIP} ep={EPOCHS}", flush=True)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = [{
        "img_path": data_root, "label_path": data_root,
        "model_path": os.path.join(OUT_DIR, "m"),
        "device": str(dev), "unlock_CPU": True,
        "optimizer": "adamw", "lr": LR_START, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": 10 ** 9, "evaluate_interval": 10 ** 9, "n_epoch": EPOCHS,
        "batch_size": BATCH_SIZE, "batch_duplication": 1,
        "channel_n": CHANNEL_N, "inference_steps": STEPS, "cell_fire_rate": FIRE_RATE,
        "input_channels": 4, "output_channels": 3, "hidden_size": HIDDEN,
        "train_model": 1, "use_attention": True,
        "input_size": INPUT_SIZE, "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        # Preprocessing flags now drive the REAL dataset (no subclass override).
        "foreground_crop": USE_FOREGROUND_CROP, "nonzero_norm": USE_NONZERO_NORM,
        "augment": USE_AUG, "augment_level": AUG_LEVEL,
        "patchify": True, "priotize_masks": PRIORITIZE_MASKS,
        "prioritize_region": PRIORITIZE_REGION,
    }]

    ds = Dataset_NiiGz_3D_BraTS()
    ds.MODALITIES = MODALITIES
    ca = [
        BasicNCA3D(CHANNEL_N, FIRE_RATE, dev, HIDDEN, kernel_size=7, input_channels=4,
                   use_attention=True, use_spatial=USE_SPATIAL, dropout=DROPOUT),
        BasicNCA3D(CHANNEL_N, FIRE_RATE, dev, HIDDEN, kernel_size=3, input_channels=4,
                   use_attention=True, use_spatial=USE_SPATIAL, dropout=DROPOUT),
    ]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, ds, ca, agent)
    ds.set_experiment(exp)

    def entry(p):
        return (p, p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")

    # Cosine LR over the whole run.
    spe = max(1, math.ceil(len(tr) / BATCH_SIZE))
    total = EPOCHS * spe
    agent.scheduler = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=LR_MIN)
        for opt in agent.optimizer
    ]

    loss_f = FocalTverskyCELoss(alpha=1 - TVERSKY_BETA, beta=TVERSKY_BETA,
                                gamma=FOCAL_GAMMA, ce_weight=0.5)

    # --- EMA: shadow copy of the (floating-point) weights ---
    # BatchNorm here uses track_running_stats=False, so there are no running
    # buffers to average -- EMA touches only learnable parameters.
    ema = None
    if USE_EMA:
        ema = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in ca]

    def ema_update():
        if ema is None:
            return
        for e, m in zip(ema, ca):
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    e[k].mul_(EMA_DECAY).add_(v.detach(), alpha=1 - EMA_DECAY)

    n_params = sum(p.numel() for m in ca for p in m.parameters())
    print("Trainable params:", n_params, "| device:", dev, flush=True)

    hist = {k: [] for k in ("epoch", "loss", "lr", "val_mean", "val_WT", "val_TC", "val_ET")}
    best, best_path = -1.0, os.path.join(OUT_DIR, "best.pth")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    print("Loading + caching volumes (slow first pass)...", flush=True)
    for ep in range(EPOCHS):
        losses = []
        loader = torch.utils.data.DataLoader(
            ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            pin_memory=(dev.type == "cuda"), worker_init_fn=worker_init_fn)
        for i, data in enumerate(loader):
            # Proper gradient-NORM clipping: run the forward/backward through the
            # agent WITHOUT its optimizer step, clip, then step. We reimplement
            # the multi-optimizer step here so the clip is a real clip_grad_norm_
            # (the old per-element register_hook lingered on the parameters).
            r = _clipped_batch_step(agent, data, loss_f, GRAD_CLIP)
            ema_update()
            if r:
                losses.append(sum(r.values()))
            if ep == 0 and (i + 1) % 20 == 0:
                print(f"  [epoch 1] {i+1}/{spe} ({time.time()-t0:.0f}s)...", flush=True)

        cur_lr = agent.optimizer[0].param_groups[0]["lr"]
        val = evaluate(agent, ds, "val")
        vm = float(np.mean([val[r]["dice"] for r in REGIONS]))
        hist["epoch"].append(ep + 1)
        hist["loss"].append(float(np.mean(losses)) if losses else 0.0)
        hist["lr"].append(cur_lr)
        hist["val_mean"].append(vm)
        for r in REGIONS:
            hist[f"val_{r}"].append(val[r]["dice"])
        # Smoothed val: mean of this and the previous BEST_WINDOW-1 epochs. The
        # per-epoch val on a small split is spiky (a lucky single epoch can win
        # on noise, then generalise worse -- the observed val->test gap). Saving
        # on the rolling mean picks a genuinely stable point, not a spike.
        recent = hist["val_mean"][-BEST_WINDOW:]
        vm_smooth = float(np.mean(recent))
        print(f"ep {ep+1}/{EPOCHS} | lr {cur_lr:.2e} | loss {hist['loss'][-1]:.3f} | "
              f"val mean {vm:.3f} (WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} "
              f"ET {val['ET']['dice']:.3f}) | smooth {vm_smooth:.3f}", flush=True)
        if vm_smooth > best:
            best = vm_smooth
            save_states = ema if ema is not None else [m.state_dict() for m in ca]
            torch.save({"m": save_states, "ep": ep + 1,
                        "val_mean": vm, "val_smooth": vm_smooth}, best_path)
            print(f"   * new best (smoothed) {vm_smooth:.3f} saved", flush=True)

    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0

    # --- Final test: single clean forward pass (best/EMA weights) ---
    ck = torch.load(best_path, map_location=dev)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)

    # Tune the per-region decision threshold on VALIDATION, then apply it to the
    # held-out test set (thresholds never see the test labels -- a legitimate
    # post-processing step). Report both 0.5 and tuned so the gain is explicit.
    val_pairs = _collect_probs(agent, ds, "val")
    thresholds = tune_thresholds(val_pairs)
    test_pairs = _collect_probs(agent, ds, "test")
    test_05 = _score(test_pairs, {r: 0.5 for r in REGIONS})
    test = _score(test_pairs, thresholds)

    def _print_table(title, res):
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print(f"{'region':<8}{'Dice':<12}{'mIoU':<12}{'HD95(vox)':<12}")
        print("-" * 60)
        for r in REGIONS:
            print(f"{r:<8}{res[r]['dice']:<12.4f}{res[r]['iou']:<12.4f}{res[r]['hd95']:<12.3f}")
        m = float(np.mean([res[r]['dice'] for r in REGIONS]))
        print("-" * 60)
        print(f"{'mean':<8}{m:<12.4f}")
        return m

    _print_table(f"GLO-NCA -- FINAL TEST @ 0.5 (best @ epoch {ck['ep']})", test_05)
    th_str = ", ".join(f"{r}={thresholds[r]:.2f}" for r in REGIONS)
    mean = _print_table(f"GLO-NCA -- FINAL TEST @ tuned thresholds ({th_str})", test)
    print(f"\ntrain time {train_time:.0f}s | peak VRAM {peak:.2f} GB | params {n_params}")

    # =====================================================================
    # DIAGNOSTIC REPORT -- read this to decide what helped before spending on
    # a full GCP run. Each line ends in a plain verdict (GOOD / OK / WATCH).
    # =====================================================================
    diag = _diagnose(hist, ck, val_pairs, test_pairs, test_05, test, thresholds, peak)
    print("\n" + "#" * 60)
    print("# DIAGNOSTIC REPORT  (what helped / what to change)")
    print("#" * 60)
    print(f"run: patch={INPUT_SIZE[-1]} aug={USE_AUG}/{AUG_LEVEL} steps={STEPS} "
          f"ema={USE_EMA} clip={GRAD_CLIP} params={n_params} peakVRAM={peak:.2f}GB")
    print("-" * 60)
    for line in diag["lines"]:
        print(line)
    print("-" * 60)
    print("VERDICT:", diag["verdict"])
    print("#" * 60, flush=True)

    json.dump({"test": test, "test_at_0.5": test_05, "thresholds": thresholds,
               "history": hist, "best_epoch": ck["ep"], "diagnostics": diag,
               "params": n_params, "train_time": train_time, "peak_vram": peak,
               "config": {"steps": STEPS, "fire": FIRE_RATE, "patch": INPUT_SIZE,
                          "beta": TVERSKY_BETA, "gamma": FOCAL_GAMMA, "aug": USE_AUG,
                          "aug_level": AUG_LEVEL, "n_patients": N_PATIENTS or nf}},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2, default=str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        a1.plot(hist["epoch"], hist["loss"], color="crimson")
        a1.set_title("Loss"); a1.set_xlabel("epoch")
        for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
            a2.plot(hist["epoch"], hist[f"val_{r}"], label=f"val {r}", color=c)
        a2.plot(hist["epoch"], hist["val_mean"], "--k", label="mean")
        a2.set_title("Validation Dice"); a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "curves.png"), dpi=130)
    except Exception as e:  # plotting must never crash a headless GCP run
        print("plot skipped:", e)
    print("Saved to", OUT_DIR, flush=True)


def _clipped_batch_step(agent, data, loss_f, grad_clip):
    r"""One multi-NCA training step with proper gradient-NORM clipping.

    Mirrors ``Agent_Multi_NCA.batch_step`` (per-region loss, multi-optimizer)
    but inserts ``clip_grad_norm_`` between backward and step. Kept here rather
    than in the agent so the framework code stays untouched and the clip is
    opt-in for this entry point.
    """
    data = agent.prepare_data(data)
    outputs, targets = agent.get_outputs(data)
    for opt in agent.optimizer:
        opt.zero_grad()
    loss = 0
    loss_ret = {}
    # Always compute the per-region loss, even when the region is absent from
    # this patch. The old framework guarded this with `if 1 in targets[...,m]`,
    # which gave a region zero gradient on empty patches -- so the model was
    # never taught "this area is NOT tumour", inflating false positives on the
    # tiny ET region. Focal-Tversky is well-defined on an empty target (the
    # smooth term keeps it finite), so include every region every step.
    for m in range(outputs.shape[-1]):
        loss_loc = loss_f(outputs[..., m], targets[..., m])
        loss = loss + loss_loc
        loss_ret[m] = loss_loc.item()
    if loss != 0:
        loss.backward()
        if grad_clip and grad_clip > 0:
            for net in agent.model:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        for opt, sch in zip(agent.optimizer, agent.scheduler):
            opt.step()
            sch.step()
    return loss_ret


if __name__ == "__main__":
    main()
