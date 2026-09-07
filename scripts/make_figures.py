#!/usr/bin/env python
r"""Generate thesis figures from saved experiment data (CSV/JSON only -- no
hard-coded metric values). Per-experiment training/val curves are already made
by the runner (graphs/); this adds the ablation + region comparison figures.

Usage:
    python scripts/make_figures.py --ablation-csv reports/ablation_results.csv \
        --final experiments/GLO-NCA-V2-final-... --out reports/thesis/figures
"""
import argparse
import csv
import json
import os


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-csv", default=None)
    ap.add_argument("--final", default=None, help="final experiment dir")
    ap.add_argument("--out", default="reports/thesis/figures")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable: {exc}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    written = []

    def save(fig, name):
        p = os.path.join(args.out, name)
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        written.append(p)

    REGIONS = ["WT", "TC", "ET"]

    # --- ablation figures ---
    if args.ablation_csv and os.path.exists(args.ablation_csv):
        rows = _read_csv(args.ablation_csv)
        labels = [r["experiment"] for r in rows]
        mean_dice = [float(r["mean_Dice"]) for r in rows]
        params = [int(float(r["parameters"])) for r in rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, mean_dice, color="#4c72b0")
        ax.set_ylabel("mean Dice"); ax.set_ylim(0, 1)
        ax.set_title("Ablation: mean Dice"); ax.grid(axis="y", alpha=.3)
        save(fig, "ablation_mean_dice.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, params, color="#55a868")
        ax.set_ylabel("parameters"); ax.set_title("Ablation: parameter count")
        ax.grid(axis="y", alpha=.3)
        save(fig, "ablation_parameters.png")

        # parameter-efficiency trade-off (params vs mean Dice)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(params, mean_dice, s=60)
        for x, y, l in zip(params, mean_dice, labels):
            ax.annotate(l, (x, y), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel("parameters"); ax.set_ylabel("mean Dice")
        ax.set_title("Parameter–performance trade-off"); ax.grid(alpha=.3)
        save(fig, "parameter_efficiency.png")

        # Dice / mIoU / HD95 by region across configs
        for metric, ylab, fname, ylim in [
                ("Dice", "Dice", "region_dice.png", (0, 1)),
                ("mIoU", "mIoU", "region_miou.png", (0, 1)),
                ("HD95_vox", "HD95 (vox)", "region_hd95.png", None)]:
            fig, ax = plt.subplots(figsize=(8, 4))
            width = 0.2
            for j, reg in enumerate(REGIONS):
                vals = [float(r[f"{reg}_{metric}"]) for r in rows]
                xs = [k + j * width for k in range(len(rows))]
                ax.bar(xs, vals, width=width, label=reg)
            ax.set_xticks([k + width for k in range(len(rows))])
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylab)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(f"{ylab} by region across configurations")
            ax.legend(); ax.grid(axis="y", alpha=.3)
            save(fig, fname)

    if not written:
        print("no figures produced (no ablation CSV given / matplotlib issue).")
    else:
        print("figures written:")
        for p in written:
            print("  " + p)
    print("Per-experiment loss/dice/miou/hd95/lr curves are in each "
          "experiment's graphs/ dir (produced by the runner).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
