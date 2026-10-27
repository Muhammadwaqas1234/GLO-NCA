r"""Final report: test table, results JSON, per-case CSV and figures."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from . import plots
from .config import REGIONS


def show(lab, t):
    r"""Print one row of the Dice table."""
    m = np.mean([t[r]['dice'] for r in REGIONS])
    print(f"{lab:<24}{t['WT']['dice']:<8.3f}{t['TC']['dice']:<8.3f}{t['ET']['dice']:<8.3f}{m:<8.3f}")


def report_tuned(tuned):
    r"""Print the tuned post-processing result (thresholds chosen on validation)."""
    th, mc = tuned["thresholds"], tuned["min_component_voxels"]
    print("-" * 60)
    print("Tuned post-processing (chosen on validation, applied once to test):")
    print("thresholds " + " ".join(f"{r}={th[r]:.2f}" for r in REGIONS)
          + " | min component " + " ".join(f"{r}={mc[r]}" for r in REGIONS)
          + f" | full resolution {tuned['full_resolution']}"
          + f" | BraTS empty-region scoring {tuned.get('brats_empty', False)}")
    if tuned["val_tuned"]:
        show("val", {r: {"dice": tuned["val_tuned"][r]} for r in REGIONS})
    show("test (tuned)", tuned["test"])
    for r in REGIONS:
        t = tuned["test"][r]
        print(f"{r:<8}{t['dice']:<12.4f}{t['iou']:<12.4f}{t['hd95']:<12.3f}")
    test = tuned["test"]
    if all("hd95_penalised" in test[r] for r in REGIONS):
        print("HD95 over all cases (374 mm for a missed or false region): "
              + " | ".join(f"{r} median {test[r]['hd95_penalised_median']:.2f} "
                           f"(IQR {test[r]['hd95_penalised_q1']:.2f}-{test[r]['hd95_penalised_q3']:.2f})" for r in REGIONS))
    if all("lesion_dice" in test[r] for r in REGIONS):
        print("Lesion-wise (BraTS 2023 form): " + " | ".join(
            f"{r} Dice {test[r]['lesion_dice']:.4f} HD95 {test[r]['lesion_hd95']:.2f} "
            f"(lesions {test[r]['lesions']}, missed {test[r]['lesions_missed']}, "
            f"false {test[r]['lesions_false_positive']})" for r in REGIONS))


def final_report(cfg, out_dir, ck_ep, test_plain, test, hist, n_params, train_time, peak, tuned=None,
                 per_case=None):
    r"""Print the test table and write results.json and the per-case CSV."""
    print("\n" + "=" * 60)
    print(f"GLO-NCA cascade - FINAL TEST (best @ epoch {ck_ep})")
    print("=" * 60)
    print(f"{'setting':<24}{'WT':<8}{'TC':<8}{'ET':<8}{'mean':<8}")
    print("-" * 60)
    show("plain", test_plain)
    show(f"+ensemble x{cfg.ENSEMBLE_N} +TTA" if cfg.USE_TTA else f"+ensemble x{cfg.ENSEMBLE_N}", test)
    print("-" * 60)
    print("Full metrics (ensemble+TTA):")
    print(f"{'region':<8}{'Dice':<12}{'mIoU':<12}{'HD95':<12}")
    for r in REGIONS:
        print(f"{r:<8}{test[r]['dice']:<12.4f}{test[r]['iou']:<12.4f}{test[r]['hd95']:<12.3f}")
    mean = np.mean([test[r]['dice'] for r in REGIONS])
    print("-" * 60)
    print(f"{'mean':<8}{mean:<12.4f}")
    print(f"train time {train_time:.0f}s | peak VRAM {peak:.2f} GB | params {n_params}")
    if tuned is not None:
        report_tuned(tuned)

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump({"test": test, "test_plain": test_plain, "history": hist, "best_epoch": ck_ep,
                   "test_tuned": tuned, "params": n_params, "train_time": train_time, "peak_vram": peak,
                   "config": {"steps": cfg.STEPS, "fire": cfg.SKIP_PROB, "patch": cfg.INPUT_SIZE,
                              "beta": cfg.TVERSKY_BETA, "aug": cfg.USE_AUG,
                              "augment": cfg.AUGMENT, "region_weights": cfg.REGION_WEIGHTS,
                              "split_seed": cfg.SPLIT_SEED}},
                  fh, indent=2, default=str)

    if per_case:
        save_per_case(per_case, out_dir)
    print("Saved to", out_dir)


def save_per_case(rows, out_dir, name="test_per_case.csv"):
    r"""Write per-case Dice (one row per case) as CSV, with HD95, its status and lesion-wise scores when scored."""
    extra = [f"{r}_{k}" for r in REGIONS for k in ("hd95", "hd95_status")] if rows and "WT_hd95" in rows[0] else []
    lesion = ("lesion_dice", "lesion_hd95", "lesions", "lesions_missed", "lesions_false_positive")
    extra += [f"{r}_{k}" for r in REGIONS for k in lesion] if rows and "WT_lesion_dice" in rows[0] else []
    with open(os.path.join(out_dir, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case"] + REGIONS + ["mean"] + extra)
        for row in rows:
            w.writerow([row["case"]] + [f"{row[r]:.6f}" for r in REGIONS]
                       + [f"{np.mean([row[r] for r in REGIONS]):.6f}"]
                       + [f"{row[k]:.4f}" if isinstance(row[k], float) else row[k] for k in extra])


def load_per_case(out_dir, name="test_per_case.csv"):
    r"""Read the per-case CSV written by save_per_case, or None when absent."""
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        return [{"case": row["case"], **{r: float(row[r]) for r in REGIONS}} for row in csv.DictReader(fh)]


def write_figures(out_dir, hist, best_epoch, settings, per_case=None, examples=None, title="GLO-NCA cascade",
                  hd95_label="HD95 (voxels, lower is better)"):
    r"""Draw every available figure into out_dir/figures; a plotting error is reported, never raised."""
    jobs = [("training curves", lambda: plots.plot_training(hist, best_epoch, out_dir, title)),
            ("test summary", lambda: plots.plot_test_summary(settings, out_dir, hd95_label=hd95_label))]
    if per_case:
        jobs.append(("per-case Dice", lambda: plots.plot_per_case(per_case, out_dir)))
    if examples:
        jobs.append(("example segmentations", lambda: plots.plot_examples(examples, out_dir)))
    for name, job in jobs:
        try:
            job()
        except Exception as exc:   # figures must never cost the results of a finished run
            print(f"WARNING: figure '{name}' was not drawn: {exc}")
    print("Figures saved to", os.path.join(out_dir, "figures"))
