r"""Diagnostic report -- extracted verbatim (behaviour-preserving) from the
original train.py. Turns a finished run into GOOD/OK/WATCH decision signals.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

REGIONS = ["WT", "TC", "ET"]


def diagnose(hist: Dict[str, list], best_epoch: int,
             test_05: Dict[str, Any], test: Dict[str, Any],
             thresholds: Dict[str, float], peak: float) -> Dict[str, Any]:
    lines, raw = [], {}

    val_best = max(hist["val_mean"]) if hist["val_mean"] else 0.0
    test_mean_05 = float(np.mean([test_05[r]["dice"] for r in REGIONS]))
    gap = val_best - test_mean_05
    raw["val_best"], raw["test_mean_05"], raw["val_test_gap"] = val_best, test_mean_05, gap
    tag = "GOOD" if gap < 0.03 else ("OK" if gap < 0.06 else "WATCH: overfitting / small-val noise")
    lines.append(f"[overfit ] val_best {val_best:.3f} -> test {test_mean_05:.3f} "
                 f"(gap {gap:+.3f})  -> {tag}")

    per = []
    for r in REGIONS:
        d = test[r]["dice"] - test_05[r]["dice"]
        per.append(f"{r}{d:+.3f}@{thresholds[r]:.2f}")
    tmean = float(np.mean([test[r]["dice"] for r in REGIONS])) - test_mean_05
    raw["threshold_gain_mean"] = tmean
    tag = "GOOD: keep tuned thresholds" if tmean > 0.005 else "OK: 0.5 is already fine"
    lines.append(f"[thresh  ] tuning gain {tmean:+.3f} ({', '.join(per)})  -> {tag}")

    vm = hist["val_mean"]
    if len(vm) >= 10:
        late = np.mean(vm[-5:]) - np.mean(vm[-10:-5])
        raw["late_slope"] = float(late)
        if late > 0.01:
            tag = "WATCH: still rising -> train MORE epochs"
        elif late < -0.02:
            tag = "WATCH: val falling -> overfit late, best-epoch earlier / fewer epochs"
        else:
            tag = "GOOD: plateaued (epoch budget about right)"
        lines.append(f"[converge] last-5 vs prev-5 val change {late:+.3f}  -> {tag}")

    worst = min(REGIONS, key=lambda r: test[r]["dice"])
    raw["worst_region"] = worst
    lines.append(f"[weakest ] {worst} is lowest at {test[worst]['dice']:.3f}  "
                 f"-> focus next tweaks here (sampling / loss weight)")

    if hist["epoch"]:
        frac = best_epoch / max(hist["epoch"])
        raw["best_epoch_frac"] = float(frac)
        tag = ("WATCH: best came early -> likely lucky/overfit" if frac < 0.4
               else "GOOD: best in the mature phase")
        lines.append(f"[bestep  ] best @ {best_epoch}/{max(hist['epoch'])} "
                     f"({frac*100:.0f}%)  -> {tag}")

    if peak > 0:
        tag = ("room to grow: try larger PATCH" if peak < 8 else
               "near a 16 GB budget" if peak < 14 else "WATCH: close to OOM")
        lines.append(f"[vram    ] peak {peak:.2f} GB  -> {tag}")

    if gap < 0.03 and (len(vm) < 10 or -0.02 <= raw.get("late_slope", 0) <= 0.01):
        verdict = f"Healthy run. Weakest region = {worst}. Config looks GCP-ready."
    elif gap >= 0.06:
        verdict = ("Large val->test gap: small-val noise or overfitting. On more "
                   "data this should shrink; if not, add regularisation / more data.")
    else:
        verdict = (f"Usable. Address the WATCH lines above (weakest = {worst}) "
                   f"before the full run.")
    return {"lines": lines, "verdict": verdict, "raw": raw}
