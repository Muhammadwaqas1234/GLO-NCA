#!/usr/bin/env python
r"""Generate thesis-ready tables from the FINAL experiment directory (and,
optionally, the ablation CSV). Reads only saved experiment outputs -- no
hard-coded numbers, no fabricated baselines.

Writes into <final_exp>/reports/thesis/:
    dataset_table.csv, model_complexity.csv, ablation_table.csv (if provided),
    final_metrics.csv, threshold_table.csv, statistical_summary.csv,
    training_summary.csv, experiment_summary.json, final_summary.md

Usage:
    python scripts/make_thesis_tables.py --final experiments/GLO-NCA-V2-final-... \
        [--ablation-csv reports/ablation_results.csv]
"""
import argparse
import csv
import json
import os
import shutil

REGIONS = ["WT", "TC", "ET"]


def _read_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="final experiment directory")
    ap.add_argument("--ablation-csv", default=None)
    args = ap.parse_args()

    exp = args.final
    res = _read_json(os.path.join(exp, "reports", "results.json"))
    man = _read_json(os.path.join(exp, "experiment_manifest.json"))
    stats_path = os.path.join(exp, "reports", "statistical_summary.json")
    stats = _read_json(stats_path) if os.path.exists(stats_path) else {}

    out = os.path.join(exp, "reports", "thesis")
    os.makedirs(out, exist_ok=True)

    def wcsv(name, header, rows):
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)

    counts = man.get("dataset", {}).get("counts", {})
    # Table 1 -- dataset split
    wcsv("dataset_table.csv", ["split", "count"],
         [["train", counts.get("train")], ["validation", counts.get("val")],
          ["test", counts.get("test")],
          ["total", sum(v for v in counts.values() if isinstance(v, int))]])

    # Table 2 -- model complexity
    model = man.get("model", {})
    wcsv("model_complexity.csv",
         ["model", "total_parameters", "trainable_parameters", "patch_size", "peak_vram_gb"],
         [["GLO-NCA-V2", model.get("total_parameters"),
           model.get("trainable_parameters"),
           man.get("training", {}).get("patch_size"), res.get("peak_vram")]])

    # Table 3 -- ablation (copied if provided)
    if args.ablation_csv and os.path.exists(args.ablation_csv):
        shutil.copy(args.ablation_csv, os.path.join(out, "ablation_table.csv"))

    # Table 4 -- final metrics (tuned thresholds) + @0.5
    test, test05 = res["test"], res["test_at_0.5"]
    rows = []
    for r in REGIONS:
        rows.append([r, f"{test[r]['dice']:.4f}", f"{test05[r]['dice']:.4f}",
                     f"{test[r]['iou']:.4f}", f"{test[r]['hd95']:.3f}"])
    mean = lambda d, k: sum(d[r][k] for r in REGIONS) / 3
    rows.append(["mean", f"{mean(test,'dice'):.4f}", f"{mean(test05,'dice'):.4f}",
                 f"{mean(test,'iou'):.4f}", ""])
    wcsv("final_metrics.csv",
         ["region", "dice_tuned", "dice_at_0.5", "miou", "hd95_vox"], rows)

    # threshold table (copy the runner's comparison if present)
    tc = os.path.join(exp, "reports", "threshold_comparison.csv")
    if os.path.exists(tc):
        shutil.copy(tc, os.path.join(out, "threshold_table.csv"))

    # statistical summary (mean/median/std/CI) if present
    if stats:
        srows = []
        for r in REGIONS:
            for m in ("dice", "iou", "hd95"):
                s = stats[r][m]
                srows.append([r, m, f"{s['mean']:.4f}", f"{s['median']:.4f}",
                              f"{s['std']:.4f}", s["n"],
                              f"{s['ci95_lo']:.4f}", f"{s['ci95_hi']:.4f}",
                              s.get("ci_method")])
        wcsv("statistical_summary.csv",
             ["region", "metric", "mean", "median", "std", "n",
              "ci95_lo", "ci95_hi", "ci_method"], srows)

    # training summary
    wcsv("training_summary.csv",
         ["field", "value"],
         [["best_epoch", res.get("best_epoch")],
          ["training_time_s", res.get("train_time")],
          ["peak_vram_gb", res.get("peak_vram")],
          ["epochs", man.get("training", {}).get("epochs")],
          ["gpu", (man.get("gpu") or {}).get("device_name", "N/A")]])

    # experiment summary json
    summary = {
        "experiment_id": man.get("experiment_id"),
        "git": man.get("git"), "seed": man.get("seed"),
        "dataset": man.get("dataset"), "model": man.get("model"),
        "training": man.get("training"), "loss": man.get("loss"),
        "best_epoch": res.get("best_epoch"),
        "thresholds": res.get("thresholds"),
        "test_tuned": test, "test_at_0.5": test05,
        "peak_vram": res.get("peak_vram"), "train_time": res.get("train_time"),
    }
    with open(os.path.join(out, "experiment_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    _write_summary_md(out, man, res, stats)
    print(f"thesis tables written to {out}")
    return 0


def _write_summary_md(out, man, res, stats):
    test = res["test"]
    mean = sum(test[r]["dice"] for r in REGIONS) / 3
    counts = man.get("dataset", {}).get("counts", {})
    with open(os.path.join(out, "final_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("# GLO-NCA V2 — Final Experiment Record\n\n")
        fh.write(f"- Experiment ID: `{man.get('experiment_id')}`\n")
        fh.write(f"- Git commit: `{(man.get('git') or {}).get('commit','N/A')}`\n")
        fh.write(f"- Seed: {man.get('seed')}\n")
        fh.write(f"- Dataset split: train {counts.get('train')} / "
                 f"val {counts.get('val')} / test {counts.get('test')}\n")
        fh.write(f"- Parameters: {man.get('model',{}).get('total_parameters')}\n")
        fh.write(f"- Patch: {man.get('training',{}).get('patch_size')}³ | "
                 f"epochs: {man.get('training',{}).get('epochs')} | "
                 f"aug: {man.get('training',{}).get('augmentation')}\n")
        fh.write(f"- Best epoch (validation, smoothed): {res.get('best_epoch')}\n")
        fh.write(f"- GPU: {(man.get('gpu') or {}).get('device_name','N/A')} | "
                 f"peak VRAM: {res.get('peak_vram')} GB | "
                 f"train time: {res.get('train_time')} s\n\n")
        fh.write("## Test (frozen validation-tuned thresholds)\n\n")
        fh.write("| Region | Dice | mIoU | HD95(vox) |\n|---|---|---|---|\n")
        for r in REGIONS:
            fh.write(f"| {r} | {test[r]['dice']:.4f} | {test[r]['iou']:.4f} "
                     f"| {test[r]['hd95']:.3f} |\n")
        fh.write(f"| **mean** | **{mean:.4f}** | | |\n\n")
        fh.write("HD95 is in VOXELS on the resampled grid (not mm).\n")
        fh.write("Thresholds tuned on validation only; test evaluated once, frozen.\n")


if __name__ == "__main__":
    raise SystemExit(main())
