r"""Connected-component post-processing, applied at evaluation only.

Production minimum component sizes: WT 50 / TC 5 / ET 0. ET filtering is off
because a threshold of 10 reduced a real scattered ET lesion to one voxel.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

REGIONS = ["WT", "TC", "ET"]

PRODUCTION_MIN_COMPONENT: Dict[str, int] = {"WT": 50, "TC": 5, "ET": 0}


def min_component_config(cfg=None) -> Dict[str, int]:
    """Per-region thresholds from config, falling back to the production set."""
    if cfg is None:
        return dict(PRODUCTION_MIN_COMPONENT)
    section = cfg.section("evaluation") or {}
    raw = section.get("postprocessing") or {}
    sizes = raw.get("min_component_voxels") or {}
    return {r: int(sizes.get(r, PRODUCTION_MIN_COMPONENT[r])) for r in REGIONS}


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop components smaller than ``min_voxels``; the largest is always kept. No-op without SciPy."""
    if min_voxels <= 0 or not mask.any():
        return mask
    try:
        from scipy.ndimage import label
    except Exception:
        return mask
    lab, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep_largest = int(sizes.argmax())
    out = np.zeros_like(mask, dtype=bool)
    for c in range(1, n + 1):
        if sizes[c] >= min_voxels or c == keep_largest:
            out |= (lab == c)
    return out


def postprocess_probs(prob: np.ndarray, thresholds: Dict[str, float],
                      min_component: Dict[str, int]) -> np.ndarray:
    """Zero probabilities of removed components per region; ``prob`` is (X, Y, Z, R)."""
    if not any(int(min_component.get(r, 0)) > 0 for r in REGIONS):
        return prob
    out = prob.copy()
    for i, r in enumerate(REGIONS):
        k = int(min_component.get(r, 0))
        if k <= 0:
            continue
        m = out[..., i] >= float(thresholds.get(r, 0.5))
        kept = remove_small_components(m, k)
        out[..., i] = np.where(m & ~kept, 0.0, out[..., i])
    return out


def apply_to_pairs(pairs: List[Tuple[np.ndarray, np.ndarray]],
                   thresholds: Dict[str, float],
                   min_component: Dict[str, int]):
    """Post-process every (prob, gt) pair; ground truth is never modified."""
    if not any(int(min_component.get(r, 0)) > 0 for r in REGIONS):
        return pairs
    return [(postprocess_probs(p, thresholds, min_component), gt)
            for p, gt in pairs]
