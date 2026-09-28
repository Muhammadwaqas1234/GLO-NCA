"""Final report: test table, results JSON and training curves."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")   # headless on GCP; the saved figure is identical
import matplotlib.pyplot as plt
import numpy as np

from .config import REGIONS


def show(lab, t):
    m = np.mean([t[r]['dice'] for r in REGIONS])
    print(f"{lab:<24}{t['WT']['dice']:<8.3f}{t['TC']['dice']:<8.3f}{t['ET']['dice']:<8.3f}{m:<8.3f}")


def final_report(C, out_dir, ck_ep, test_plain, test, hist, n_params, train_time, peak):
    print("\n" + "=" * 60)
    print(f"GLO-NCA cascade - FINAL TEST (best @ epoch {ck_ep})")
    print("=" * 60)
    print(f"{'setting':<24}{'WT':<8}{'TC':<8}{'ET':<8}{'mean':<8}")
    print("-" * 60)
    show("plain", test_plain)
    show(f"+ensemble x{C.ENSEMBLE_N} +TTA" if C.USE_TTA else f"+ensemble x{C.ENSEMBLE_N}", test)
    print("-" * 60)
    print("Full metrics (ensemble+TTA):")
    print(f"{'region':<8}{'Dice':<12}{'mIoU':<12}{'HD95':<12}")
    for r in REGIONS:
        print(f"{r:<8}{test[r]['dice']:<12.4f}{test[r]['iou']:<12.4f}{test[r]['hd95']:<12.3f}")
    mean = np.mean([test[r]['dice'] for r in REGIONS])
    print("-" * 60)
    print(f"{'mean':<8}{mean:<12.4f}")
    print(f"train time {train_time:.0f}s | peak VRAM {peak:.2f} GB | params {n_params}")

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump({"test": test, "test_plain": test_plain, "history": hist, "best_epoch": ck_ep,
                   "params": n_params, "train_time": train_time, "peak_vram": peak,
                   "config": {"steps": C.STEPS, "fire": C.FIRE_RATE, "patch": C.INPUT_SIZE,
                              "beta": C.TVERSKY_BETA, "aug": C.USE_AUG}},
                  fh, indent=2, default=str)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(hist["epoch"], hist["loss"], color="crimson"); a1.set_title("Loss"); a1.set_xlabel("epoch")
    for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
        a2.plot(hist["epoch"], hist[f"val_{r}"], label=f"val {r}", color=c)
    a2.plot(hist["epoch"], hist["val_mean"], "--k", label="mean")
    a2.set_title("Validation Dice"); a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130); plt.close(fig)
    print("Saved to", out_dir)
