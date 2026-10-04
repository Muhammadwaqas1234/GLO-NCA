r"""Publication-quality figures: training curves, test summary, per-case spread and examples."""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")   # headless on GCP
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .config import REGIONS

# Okabe-Ito colour-blind-safe palette.
COLORS = {"WT": "#0072B2", "TC": "#E69F00", "ET": "#009E73", "mean": "#222222",
          "loss": "#D55E00", "lr": "#56B4E9", "gt": "#FFD700", "pred": "#E8185D"}
NAMES = {"WT": "Whole tumour", "TC": "Tumour core", "ET": "Enhancing tumour"}

STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 100,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}


def _save(fig, out_dir, name):
    r"""Save a figure as 300 dpi PNG and vector PDF under out_dir/figures."""
    folder = os.path.join(out_dir, "figures")
    os.makedirs(folder, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"), **kw)
    plt.close(fig)


def _valid(xs, ys):
    r"""Pairs where y is a number (skips epochs without validation)."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and not (isinstance(y, float) and math.isnan(y))]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_training(hist, best_epoch, out_dir, title="GLO-NCA cascade"):
    r"""Loss and learning rate (left) and validation Dice per region with the best epoch (right)."""
    with plt.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"wspace": 0.42})
        a1.plot(hist["epoch"], hist["loss"], color=COLORS["loss"], lw=1.6, label="Training loss")
        a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.set_title("Training loss and learning rate")
        lr_ax = a1.twinx()
        exp = int(math.floor(math.log10(max(hist["lr"])))) if hist["lr"] and max(hist["lr"]) > 0 else 0
        lr_ax.plot(hist["epoch"], [v / 10 ** exp for v in hist["lr"]], color=COLORS["lr"], lw=1.2, ls="--")
        lr_ax.set_ylabel(f"Learning rate (×10$^{{{exp}}}$)" if exp else "Learning rate"); lr_ax.grid(False)
        lr_ax.spines["right"].set_visible(True)
        for ax in (a1, a2):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        a1.legend(handles=[Line2D([], [], color=COLORS["loss"], lw=1.6, label="Training loss"),
                           Line2D([], [], color=COLORS["lr"], lw=1.2, ls="--", label="Learning rate")],
                  loc="upper right")

        for r in REGIONS:
            x, y = _valid(hist["epoch"], hist[f"val_{r}"])
            a2.plot(x, y, color=COLORS[r], lw=1.4, marker="o", ms=2.5, label=NAMES[r])
        x, y = _valid(hist["epoch"], hist["val_mean"])
        a2.plot(x, y, color=COLORS["mean"], lw=1.8, ls="--", label="Mean")
        if best_epoch in hist["epoch"]:
            best = hist["val_mean"][hist["epoch"].index(best_epoch)]
            a2.axvline(best_epoch, color="#888888", lw=0.9, ls=":")
            a2.annotate(f"best epoch {best_epoch}\nmean Dice {best:.3f}", (best_epoch, best),
                        xytext=(8, -28), textcoords="offset points", fontsize=8.5,
                        arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.8})
        a2.set_xlabel("Epoch"); a2.set_ylabel("Dice"); a2.set_ylim(0, 1)
        a2.set_title("Validation Dice"); a2.legend(loc="lower right", ncol=2)
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.03)
        _save(fig, out_dir, "training_curves")


def plot_test_summary(settings, out_dir):
    r"""Grouped Dice bars per region for each test setting, and HD95 for the final setting."""
    names = list(settings)
    with plt.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
        groups = REGIONS + ["mean"]
        width = 0.8 / len(names)
        shades = [0.45, 0.72, 1.0][-len(names):] if len(names) <= 3 else np.linspace(0.4, 1, len(names))
        for i, (name, alpha) in enumerate(zip(names, shades)):
            res = settings[name]
            vals = [res[r]["dice"] for r in REGIONS] + [float(np.mean([res[r]["dice"] for r in REGIONS]))]
            xs = np.arange(len(groups)) + (i - (len(names) - 1) / 2) * width
            bars = a1.bar(xs, vals, width * 0.92, color=[COLORS[g] for g in groups], alpha=alpha,
                          edgecolor="white", linewidth=0.6, label=name)
            for b, v in zip(bars, vals):
                a1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=90 if len(names) > 2 else 0)
        a1.set_xticks(range(len(groups)), [NAMES.get(g, "Mean") for g in groups])
        a1.set_ylim(0, 1.08); a1.set_ylabel("Dice"); a1.set_title("Test Dice by region")
        a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=a) for a in shades],
                  labels=names, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(names))

        final = settings[names[-1]]
        hd = [final[r]["hd95"] for r in REGIONS]
        bars = a2.bar(range(len(REGIONS)), [0 if math.isnan(v) else v for v in hd],
                      color=[COLORS[r] for r in REGIONS], width=0.6, edgecolor="white")
        for b, v in zip(bars, hd):
            a2.text(b.get_x() + b.get_width() / 2, (0 if math.isnan(v) else v) * 1.02 + 0.05,
                    "n/a" if math.isnan(v) else f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
        a2.set_ylim(0, max([v for v in hd if not math.isnan(v)] or [1]) * 1.18)
        a2.set_xticks(range(len(REGIONS)), [NAMES[r] for r in REGIONS])
        a2.set_ylabel("HD95 (voxels, lower is better)"); a2.set_title(f"Test HD95 ({names[-1].split(' (')[0].lower()})")
        _save(fig, out_dir, "test_summary")


def plot_per_case(rows, out_dir, setting="test"):
    r"""Box plots of per-case Dice with every case shown, so failures and spread are visible."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [[row[r] for row in rows] for r in REGIONS]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "lw": 1.4})
        rng = np.random.default_rng(0)
        for i, (r, patch) in enumerate(zip(REGIONS, bp["boxes"]), 1):
            patch.set_facecolor(COLORS[r]); patch.set_alpha(0.25); patch.set_edgecolor(COLORS[r])
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(data[i - 1])), data[i - 1], s=9,
                       color=COLORS[r], alpha=0.7, edgecolors="none", zorder=3)
            ax.scatter([i], [np.mean(data[i - 1])], marker="D", s=28, color="white",
                       edgecolors="black", zorder=4)
        ax.set_xticks(range(1, len(REGIONS) + 1),
                      [f"{NAMES[r]}\nmedian {np.median(d):.3f}" for r, d in zip(REGIONS, data)])
        ax.set_ylim(-0.03, 1.03); ax.set_ylabel("Dice per case")
        ax.set_title(f"Per-case Dice on the {setting} set (n = {len(rows)}); diamond = mean")
        _save(fig, out_dir, f"{setting}_per_case")


def plot_examples(examples, out_dir, setting="test"):
    r"""Axial slices with ground-truth and predicted outlines for the given (label, case, img, gt, prob)."""
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(len(examples), len(REGIONS) + 1,
                                 figsize=(2.6 * (len(REGIONS) + 1), 2.7 * len(examples)), squeeze=False)
        for row, (label, case, img, gt, prob) in zip(axes, examples):
            z = int(np.argmax(gt[..., 0].sum(axis=(0, 1))))   # slice with the most tumour
            base = np.rot90(img[:, :, z])
            brain = base[base != 0]
            lo, hi = np.percentile(brain, (1, 99)) if brain.size else (0, 1)
            base = np.ma.masked_where(base == 0, base)        # background drawn black
            for ax in row:
                ax.set_facecolor("black")
            row[0].imshow(base, cmap="gray", vmin=lo, vmax=hi)
            row[0].set_title(f"{label}: {case}", fontsize=8.5, loc="left")
            for ax, (i, r) in zip(row[1:], enumerate(REGIONS)):
                ax.imshow(base, cmap="gray", vmin=lo, vmax=hi)
                t, p = np.rot90(gt[:, :, z, i]), np.rot90(prob[:, :, z, i] >= 0.5)
                if t.any():
                    ax.contour(t, levels=[0.5], colors=COLORS["gt"], linewidths=1.1)
                if p.any():
                    ax.contour(p, levels=[0.5], colors=COLORS["pred"], linewidths=1.1, linestyles="--")
                inter = np.logical_and(gt[..., i] >= 0.5, prob[..., i] >= 0.5).sum()
                dice = 2 * inter / ((gt[..., i] >= 0.5).sum() + (prob[..., i] >= 0.5).sum() + 1e-6)
                ax.set_title(f"{NAMES[r]} (case Dice {dice:.3f})", fontsize=8.5)
            for ax in row:
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
        fig.legend(handles=[Line2D([], [], color=COLORS["gt"], lw=1.4, label="Ground truth"),
                            Line2D([], [], color=COLORS["pred"], lw=1.4, ls="--", label="Prediction")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Example {setting} segmentations (best, median and worst case)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        _save(fig, out_dir, f"{setting}_examples")
