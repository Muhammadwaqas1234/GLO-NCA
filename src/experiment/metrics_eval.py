r"""Evaluation + threshold tuning, extracted verbatim (behaviour-preserving)
from the original train.py so the runner reuses the EXACT methodology:
single clean forward pass, per-region Dice/mIoU/HD95, val-only threshold tuning.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.agents.Agent import iou_score, hd95_score

REGIONS = ["WT", "TC", "ET"]


def collect_probs(agent, dataset, state) -> List[Tuple[np.ndarray, np.ndarray]]:
    """One clean forward pass over ``state``; returns per-case (prob, gt)."""
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


def score(pairs, thresholds: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Per-region Dice/mIoU/HD95 at the given per-region thresholds."""
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
    if thresholds is None:
        thresholds = {r: 0.5 for r in REGIONS}
    return score(collect_probs(agent, dataset, state), thresholds)


def tune_thresholds(pairs, grid=None) -> Dict[str, float]:
    """Best per-region threshold by mean Dice on VALIDATION pairs (test never
    seen). Same grid and logic as the original."""
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
