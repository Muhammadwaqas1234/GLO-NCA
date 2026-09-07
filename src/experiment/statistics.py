r"""Per-case statistics for thesis reporting: mean / median / std, and an
optional bootstrap 95% CI over test cases. Honest by construction -- the CI is
labelled as bootstrap (not analytical) and records its method + seed + samples.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

REGIONS = ["WT", "TC", "ET"]
METRICS = ["dice", "iou", "hd95"]


def _clean(values: List[float]) -> List[float]:
    """Drop NaNs (HD95 is NaN when exactly one mask is empty -- undefined)."""
    return [v for v in values if not (isinstance(v, float) and math.isnan(v))]


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> Dict[str, float]:
    """Bootstrap percentile CI for the mean of ``values``. Returns lo/hi and the
    method metadata. Not an analytical CI -- resampling over cases."""
    vals = _clean(values)
    if len(vals) < 2:
        return {"lo": float("nan"), "hi": float("nan"),
                "method": "bootstrap-percentile", "n_boot": n_boot,
                "seed": seed, "n_cases": len(vals)}
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"lo": lo, "hi": hi, "method": "bootstrap-percentile",
            "n_boot": n_boot, "seed": seed, "n_cases": len(vals)}


def summarize_per_case(per_case: Dict[str, Dict[str, List[float]]],
                       n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Turn per-case arrays into mean/median/std/CI per region+metric."""
    out: Dict[str, Any] = {}
    for r in REGIONS:
        out[r] = {}
        for m in METRICS:
            vals = _clean(per_case[r][m])
            if vals:
                ci = bootstrap_ci(per_case[r][m], n_boot=n_boot, seed=seed)
                out[r][m] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                    "ci95_lo": ci["lo"], "ci95_hi": ci["hi"],
                    "ci_method": ci["method"], "ci_n_boot": ci["n_boot"],
                    "ci_seed": ci["seed"],
                }
            else:
                out[r][m] = {"mean": float("nan"), "median": float("nan"),
                             "std": float("nan"), "n": 0,
                             "ci95_lo": float("nan"), "ci95_hi": float("nan"),
                             "ci_method": "bootstrap-percentile",
                             "ci_n_boot": n_boot, "ci_seed": seed}
    return out
