r"""Evaluation and threshold tuning: one forward pass, per-region Dice/mIoU/HD95, validation-only tuning."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.agents.Agent import iou_score, hd95_score

REGIONS = ["WT", "TC", "ET"]


def collect_probs(agent, dataset, state) -> List[Tuple[np.ndarray, np.ndarray]]:
    """One forward pass over ``state``; returns per-case (prob, gt). Profiled separately from training."""
    from src.profiling import get_profiler
    prof = get_profiler()

    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    pairs = []
    with torch.no_grad():
        for data in loader:
            with prof.section(f"validation/{state}_prepare", cuda=True):
                data = agent.prepare_data(data, eval=True)
            with prof.section(f"validation/{state}_inference", cuda=True):
                out, targets = agent.get_outputs(data, full_img=True)
            # Sigmoid + host copy is a sync point; timed separately from inference.
            with prof.section(f"validation/{state}_probabilities", cuda=True):
                prob = torch.sigmoid(out).detach().cpu().numpy()
            # Store binary GT as uint8 (bit-exact for gt >= 0.5, saves host RAM at 128³).
            # Probabilities stay float32: float16 flips threshold decisions.
            with prof.section(f"validation/{state}_gt_to_host", cuda=True):
                gt = (targets.detach().cpu().numpy() >= 0.5).astype(np.uint8)
            pairs.append((prob, gt))
    agent.exp.set_model_state("train")
    return pairs


def score(pairs, thresholds: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Per-region Dice/mIoU/HD95 at the given per-region thresholds."""
    from src.profiling import get_profiler
    prof = get_profiler()
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    for prob, gt in pairs:
        for i, r in enumerate(REGIONS):
            th = thresholds[r]
            p, t = prob[..., i], gt[..., i]
            with prof.section("validation/threshold_dice"):
                inter = np.logical_and(p >= th, t >= 0.5).sum()
                denom = (p >= th).sum() + (t >= 0.5).sum() + 1e-6
                acc[r]["dice"].append((2 * inter) / denom)
            with prof.section("validation/iou"):
                acc[r]["iou"].append(iou_score(p, t, threshold=th))
            with prof.section("validation/hd95"):
                acc[r]["hd95"].append(hd95_score(p, t, threshold=th))
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])),
                  "iou": float(np.mean(acc[r]["iou"])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out


def score_per_case(pairs, thresholds: Dict[str, float]):
    r"""Per-case metrics with the same definitions as ``score``: {region: {"dice", "iou", "hd95": [...]}}."""
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    for prob, gt in pairs:
        for i, r in enumerate(REGIONS):
            th = thresholds[r]
            p, t = prob[..., i], gt[..., i]
            inter = np.logical_and(p >= th, t >= 0.5).sum()
            denom = (p >= th).sum() + (t >= 0.5).sum() + 1e-6
            acc[r]["dice"].append(float((2 * inter) / denom))
            acc[r]["iou"].append(float(iou_score(p, t, threshold=th)))
            acc[r]["hd95"].append(float(hd95_score(p, t, threshold=th)))
    return acc


def evaluate(agent, dataset, state, thresholds=None):
    if thresholds is None:
        thresholds = {r: 0.5 for r in REGIONS}
    return score(collect_probs(agent, dataset, state), thresholds)


def tune_thresholds(pairs, grid=None) -> Dict[str, float]:
    """Best per-region threshold by mean Dice on validation pairs only."""
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
