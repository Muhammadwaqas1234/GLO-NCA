r"""Per-case diagnostics for GLO-NCA validation and frozen test evaluation.

Metric definitions are reused from ``metrics_eval`` rather than reimplemented,
so a per-case row and the aggregate score can never disagree. What this module
adds is the accounting an examiner needs: which case a number came from,
how much foreground the ground truth and the prediction each contained, and
whether HD95 was genuinely computable.

HD95 is undefined when either mask is empty -- a surface distance needs two
surfaces. Those cases are recorded as undefined WITH A REASON and excluded
from the HD95 mean; they are never silently replaced by zero.

Metrics come from the primary segmentation logits only. Deep-supervision
auxiliary heads exist to shape training and never enter a reported metric.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .metrics_eval import REGIONS
from src.agents.Agent import hd95_score, iou_score

FIELDS: Tuple[str, ...] = (
    "case_id", "subject_id", "split",
    "gt_vox_WT", "pred_vox_WT", "gt_vox_TC", "pred_vox_TC",
    "gt_vox_ET", "pred_vox_ET",
    "dice_WT", "dice_TC", "dice_ET", "dice_mean",
    "iou_WT", "iou_TC", "iou_ET",
    "hd95_WT", "hd95_TC", "hd95_ET", "hd95_mean",
    "metric_valid", "failure_reason",
)


def _subject_of(case_id: str) -> str:
    """Base subject id: the case id minus its trailing timepoint."""
    return case_id.rsplit("-", 1)[0] if "-" in case_id else case_id


def _hd95_status(pred_mask: np.ndarray, gt_mask: np.ndarray) -> str:
    if not gt_mask.any() and not pred_mask.any():
        return "both_empty"
    if not gt_mask.any():
        return "empty_gt"
    if not pred_mask.any():
        return "empty_prediction"
    return "ok"


def build_rows(pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
               case_ids: Sequence[str], split: str,
               thresholds: Dict[str, float]) -> List[dict]:
    """One diagnostic row per case.

    ``pairs`` are (probability, ground truth) in ``REGIONS`` order, exactly as
    ``metrics_eval.collect_probs`` returns them and after any post-processing
    the caller applied.
    """
    rows: List[dict] = []
    for idx, (prob, gt) in enumerate(pairs):
        cid = case_ids[idx] if idx < len(case_ids) else f"case_{idx:04d}"
        row: dict = {"case_id": cid, "subject_id": _subject_of(cid),
                     "split": split}
        dices, hd_values, reasons = [], [], []

        for i, r in enumerate(REGIONS):
            th = float(thresholds.get(r, 0.5))
            p, t = prob[..., i], gt[..., i]
            p_mask, t_mask = p >= th, t >= 0.5

            row[f"gt_vox_{r}"] = int(t_mask.sum())
            row[f"pred_vox_{r}"] = int(p_mask.sum())

            inter = np.logical_and(p_mask, t_mask).sum()
            denom = p_mask.sum() + t_mask.sum() + 1e-6
            dice = float((2 * inter) / denom)
            row[f"dice_{r}"] = dice
            dices.append(dice)
            row[f"iou_{r}"] = float(iou_score(p, t, threshold=th))

            status = _hd95_status(p_mask, t_mask)
            if status == "ok":
                try:
                    value = float(hd95_score(p, t, threshold=th))
                except Exception as exc:                     # pragma: no cover
                    value = float("nan")
                    status = f"error:{type(exc).__name__}"
                if value != value:
                    status = "undefined"
            else:
                value = float("nan")
            row[f"hd95_{r}"] = value
            if value == value:
                hd_values.append(value)
            else:
                reasons.append(f"{r}:{status}")

        row["dice_mean"] = float(np.mean(dices)) if dices else float("nan")
        row["hd95_mean"] = float(np.mean(hd_values)) if hd_values else float("nan")
        row["metric_valid"] = int(not reasons)
        row["failure_reason"] = ";".join(reasons)
        rows.append(row)
    return rows


def summarise(rows: Sequence[dict]) -> dict:
    """Aggregate counts, including explicit HD95 validity accounting."""
    out: dict = {"cases_evaluated": len(rows), "regions": {}}
    for r in REGIONS:
        hd = [row[f"hd95_{r}"] for row in rows]
        valid = [v for v in hd if v == v]
        dice = [row[f"dice_{r}"] for row in rows]
        reasons: Dict[str, int] = {}
        for row in rows:
            for part in (row["failure_reason"] or "").split(";"):
                if part.startswith(f"{r}:"):
                    reasons[part.split(":", 1)[1]] = \
                        reasons.get(part.split(":", 1)[1], 0) + 1
        out["regions"][r] = {
            "dice_mean": float(np.mean(dice)) if dice else None,
            "hd95_valid_cases": len(valid),
            "hd95_invalid_cases": len(hd) - len(valid),
            "hd95_mean_over_valid": float(np.mean(valid)) if valid else None,
            "hd95_invalid_reasons": reasons,
        }
    out["cases_fully_valid"] = sum(int(row["metric_valid"]) for row in rows)
    return out


def write_csv(path: str, rows: Sequence[dict]) -> str:
    """Write rows in a deterministic column order. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    os.replace(tmp, path)
    return path
