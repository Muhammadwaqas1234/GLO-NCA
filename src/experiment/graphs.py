r"""Generate thesis-useful graphs from the recorded CSV metrics.

Everything is read back from metrics/*.csv (never hard-coded), so the graphs
always reflect exactly what was logged. Plotting is best-effort and headless
(Agg backend) so it never crashes a GCP run.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

REGIONS = ["WT", "TC", "ET"]


def _read_csv(path: str) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {}
    if not os.path.exists(path):
        return cols
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for field in reader.fieldnames or []:
            cols[field] = []
        for row in reader:
            for k, v in row.items():
                try:
                    cols[k].append(float(v))
                except (ValueError, TypeError):
                    cols[k].append(float("nan"))
    return cols


def generate(ws) -> List[str]:
    """Create graphs/*.png from metrics/train.csv + metrics/validation.csv.
    Returns the list of files written."""
    written: List[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return written

    train = _read_csv(ws.path("metrics", "train.csv"))
    val = _read_csv(ws.path("metrics", "validation.csv"))
    ep = val.get("epoch") or train.get("epoch") or []

    def _save(fig, name):
        p = ws.path("graphs", name)
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 1) Loss (train) + val mean dice on a twin axis for context.
    if ep and train.get("loss"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train["epoch"], train["loss"], color="crimson", label="train loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training loss")
        ax.grid(alpha=.3); ax.legend()
        _save(fig, "loss.png")

    # 2) Dice per region + mean.
    if ep and all(f"dice_{r}" in val for r in REGIONS):
        fig, ax = plt.subplots(figsize=(8, 5))
        for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
            ax.plot(val["epoch"], val[f"dice_{r}"], label=f"val {r}", color=c)
        if "dice_mean" in val:
            ax.plot(val["epoch"], val["dice_mean"], "--k", label="mean")
        ax.set_xlabel("epoch"); ax.set_ylabel("Dice"); ax.set_ylim(0, 1)
        ax.set_title("Validation Dice"); ax.grid(alpha=.3); ax.legend()
        _save(fig, "dice.png")

    # 3) mIoU per region.
    if ep and all(f"iou_{r}" in val for r in REGIONS):
        fig, ax = plt.subplots(figsize=(8, 5))
        for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
            ax.plot(val["epoch"], val[f"iou_{r}"], label=f"val {r}", color=c)
        ax.set_xlabel("epoch"); ax.set_ylabel("mIoU"); ax.set_ylim(0, 1)
        ax.set_title("Validation mIoU"); ax.grid(alpha=.3); ax.legend()
        _save(fig, "miou.png")

    # 4) HD95 per region (voxels).
    if ep and all(f"hd95_{r}" in val for r in REGIONS):
        fig, ax = plt.subplots(figsize=(8, 5))
        for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
            ax.plot(val["epoch"], val[f"hd95_{r}"], label=f"val {r}", color=c)
        ax.set_xlabel("epoch"); ax.set_ylabel("HD95 (vox)")
        ax.set_title("Validation HD95 (voxels)"); ax.grid(alpha=.3); ax.legend()
        _save(fig, "hd95.png")

    # 5) Learning rate.
    if ep and train.get("lr"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train["epoch"], train["lr"], color="darkorange")
        ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
        ax.set_title("Learning-rate schedule"); ax.grid(alpha=.3)
        _save(fig, "learning_rate.png")

    # 6) GPU memory, if recorded.
    if ep and train.get("gpu_mem_gb") and any(
            v == v and v > 0 for v in train["gpu_mem_gb"]):  # any non-nan >0
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train["epoch"], train["gpu_mem_gb"], color="teal")
        ax.set_xlabel("epoch"); ax.set_ylabel("GPU memory (GB)")
        ax.set_title("GPU memory (peak per epoch)"); ax.grid(alpha=.3)
        _save(fig, "gpu_memory.png")

    return written
