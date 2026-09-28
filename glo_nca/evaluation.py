"""Evaluation: Dice, mIoU and HD95 at a fixed 0.5 threshold on full volumes."""
from __future__ import annotations

import math

import numpy as np
import torch

from src.agents.Agent import iou_score, hd95_score

from .config import REGIONS


def evaluate(agent, dataset, state, ensemble=1, tta=False):
    """Evaluate one split; ensemble > 1 averages stochastic passes and tta adds axis flips."""
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    flips = [None]
    if tta:
        flips = [None, (1,), (2,), (3,)]   # identity + flip each spatial axis
    with torch.no_grad():
        for data in loader:
            data = agent.prepare_data(data, eval=True)
            probs = []
            for fl in flips:
                idx, inp, tgt = data
                if fl is not None:
                    inp = torch.flip(inp, dims=fl)
                for _ in range(ensemble):
                    out, targets = agent.get_outputs((idx, inp, tgt), full_img=True)
                    p = torch.sigmoid(out)
                    if fl is not None:
                        p = torch.flip(p, dims=fl)         # un-flip prediction
                    probs.append(p.detach().cpu().numpy())
            prob = np.mean(probs, axis=0)
            gt = targets.detach().cpu().numpy()
            for i, r in enumerate(REGIONS):
                p, t = prob[..., i], gt[..., i]
                inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
                acc[r]["dice"].append((2 * inter) / ((p >= 0.5).sum() + (t >= 0.5).sum() + 1e-6))
                acc[r]["iou"].append(iou_score(p, t)); acc[r]["hd95"].append(hd95_score(p, t))
    agent.exp.set_model_state("train")
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])), "iou": float(np.mean(acc[r]["iou"])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out
