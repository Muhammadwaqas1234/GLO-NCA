"""Lesion-size stratified evaluation (analysis only).

Groups existing per-case rows by ground-truth lesion size and reports the same
metrics per stratum. No extra model pass; no effect on training, thresholds,
checkpoint selection or stopping.

Sizes are in resampled voxels of the 128³ grid (per-case scale factor, not
mm³), so the bins are analysis bins, not clinical categories. Strata use
ground truth only, never predictions.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import numpy as np

REGIONS = ["WT", "TC", "ET"]

# Analysis bins in resampled 128³ voxels; upper edge exclusive, last bin unbounded.
DEFAULT_STRATA: List[tuple] = [
    ("small", 0, 100),
    ("medium", 100, 1000),
    ("large", 1000, None),
]

# Empty ground truth gets its own stratum so it cannot distort size-stratum Dice.
ABSENT = "absent_gt"


def stratum_of(gt_voxels: int, strata: Sequence[tuple] = DEFAULT_STRATA) -> str:
    """Name the stratum a ground-truth voxel count falls into."""
    if int(gt_voxels) <= 0:
        return ABSENT
    for name, lo, hi in strata:
        if gt_voxels >= lo and (hi is None or gt_voxels < hi):
            return name
    return strata[-1][0]


def _mean(values: Sequence[float]):
    finite = [v for v in values if v == v]
    return float(np.mean(finite)) if finite else None


def stratify(rows: Sequence[dict],
             strata: Sequence[tuple] = DEFAULT_STRATA) -> List[dict]:
    """One row per (region, stratum); HD95 averages valid cases only and never substitutes zero."""
    names = [s[0] for s in strata] + [ABSENT]
    out: List[dict] = []

    for r in REGIONS:
        buckets: Dict[str, List[dict]] = {n: [] for n in names}
        for row in rows:
            buckets[stratum_of(row.get(f"gt_vox_{r}", 0), strata)].append(row)

        for name in names:
            group = buckets[name]
            dice = [row[f"dice_{r}"] for row in group]
            iou = [row[f"iou_{r}"] for row in group]
            hd = [row[f"hd95_{r}"] for row in group]
            hd_valid = [v for v in hd if v == v]

            empty_pred = sum(1 for row in group
                             if int(row.get(f"pred_vox_{r}", 0)) == 0)
            empty_gt = sum(1 for row in group
                           if int(row.get(f"gt_vox_{r}", 0)) == 0)
            both_empty = sum(1 for row in group
                             if int(row.get(f"pred_vox_{r}", 0)) == 0
                             and int(row.get(f"gt_vox_{r}", 0)) == 0)

            gt_sizes = [int(row.get(f"gt_vox_{r}", 0)) for row in group
                        if int(row.get(f"gt_vox_{r}", 0)) > 0]

            out.append({
                "region": r,
                "stratum": name,
                "gt_voxel_range": _range_label(name, strata),
                "cases": len(group),
                "gt_voxels_median": (float(np.median(gt_sizes))
                                     if gt_sizes else None),
                "dice_mean": _mean(dice),
                "dice_std": (float(np.std([d for d in dice if d == d]))
                             if any(d == d for d in dice) else None),
                "iou_mean": _mean(iou),
                "hd95_mean_over_valid": _mean(hd_valid),
                "hd95_valid_cases": len(hd_valid),
                "hd95_invalid_cases": len(hd) - len(hd_valid),
                "empty_gt_cases": empty_gt,
                "empty_prediction_cases": empty_pred,
                "both_empty_cases": both_empty,
            })
    return out


def _range_label(name: str, strata: Sequence[tuple]) -> str:
    if name == ABSENT:
        return "gt == 0"
    for sname, lo, hi in strata:
        if sname == name:
            return f"[{lo}, {'inf' if hi is None else hi})"
    return ""


COLUMNS = ["region", "stratum", "gt_voxel_range", "cases", "gt_voxels_median",
           "dice_mean", "dice_std", "iou_mean", "hd95_mean_over_valid",
           "hd95_valid_cases", "hd95_invalid_cases", "empty_gt_cases",
           "empty_prediction_cases", "both_empty_cases"]


def write_csv(path: str, rows: Sequence[dict]) -> str:
    """Machine-readable output for thesis analysis. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c))
                             for c in COLUMNS})
    return path
