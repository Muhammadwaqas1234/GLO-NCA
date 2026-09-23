#!/usr/bin/env python
r"""Assemble the ablation results table from finished experiment directories.

Reads each experiment's reports/results.json + experiment_manifest.json (NO
hard-coded numbers) and writes reports/ablation_results.csv plus a small
interpretation (Full - Baseline etc., reported as ABSOLUTE improvements only --
no significance claims).

Usage:
    python scripts/make_ablation_table.py \
        --baseline experiments/GLO-NCA-V2-ablation-baseline-... \
        --se       experiments/GLO-NCA-V2-ablation-se-... \
        --spatial  experiments/GLO-NCA-V2-ablation-spatial-... \
        --full     experiments/GLO-NCA-V2-ablation-full-... \
        --out      reports/ablation_results.csv
"""
import argparse
import csv
import json
import os

REGIONS = ["WT", "TC", "ET"]


def _load(exp_dir):
    rp = os.path.join(exp_dir, "reports", "results.json")
    if not os.path.exists(rp):
        raise SystemExit(f"FAIL: results.json not found for {exp_dir}. Has this "
                         f"experiment finished? (expected {rp})")
    with open(rp, encoding="utf-8") as fh:
        res = json.load(fh)
    man = {}
    mpath = os.path.join(exp_dir, "experiment_manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            man = json.load(fh)
    return res, man


def _row(label, exp_dir):
    res, man = _load(exp_dir)
    test = res["test"]  # tuned-threshold test metrics (frozen)
    model = man.get("model", {})
    mean_dice = sum(test[r]["dice"] for r in REGIONS) / 3
    mean_iou = sum(test[r]["iou"] for r in REGIONS) / 3
    hd = [test[r]["hd95"] for r in REGIONS if test[r]["hd95"] == test[r]["hd95"]]
    mean_hd = sum(hd) / len(hd) if hd else float("nan")
    return {
        "experiment": label,
        "attention": model.get("se_enabled"),
        "spatial_context": model.get("spatial_gc_enabled"),
        "parameters": res.get("params", man.get("model", {}).get("total_parameters")),
        "mean_Dice": round(mean_dice, 4),
        "WT_Dice": round(test["WT"]["dice"], 4),
        "TC_Dice": round(test["TC"]["dice"], 4),
        "ET_Dice": round(test["ET"]["dice"], 4),
        "mean_mIoU": round(mean_iou, 4),
        "WT_mIoU": round(test["WT"]["iou"], 4),
        "TC_mIoU": round(test["TC"]["iou"], 4),
        "ET_mIoU": round(test["ET"]["iou"], 4),
        "mean_HD95_vox": round(mean_hd, 3),
        "WT_HD95_vox": round(test["WT"]["hd95"], 3),
        "TC_HD95_vox": round(test["TC"]["hd95"], 3),
        "ET_HD95_vox": round(test["ET"]["hd95"], 3),
        "training_time": res.get("train_time"),
        "peak_vram": res.get("peak_vram"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--se", required=True)
    ap.add_argument("--spatial", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", default="reports/ablation_results.csv")
    args = ap.parse_args()

    rows = [_row("A0_baseline", args.baseline), _row("A1_SE", args.se),
            _row("A2_spatial", args.spatial), _row("A3_full", args.full)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    # Absolute improvements (NO significance claims).
    base, full = rows[0], rows[3]
    print("\nAbsolute improvements (Full - Baseline):")
    for k in ["mean_Dice", "WT_Dice", "TC_Dice", "ET_Dice", "mean_mIoU"]:
        print(f"  {k}: {full[k] - base[k]:+.4f}")
    print(f"  parameters: {full['parameters']} vs baseline {base['parameters']} "
          f"(+{full['parameters'] - base['parameters']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
