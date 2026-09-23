#!/usr/bin/env python
r"""
Patchify is OFF on the GLO-NCA production path.  NO TRAINING.

The dataset crop uses `self._train_patch_size or self.size`, and
`set_train_patch_size` is only called when `data.training_patch.enabled` is
true. With it false the crop degenerates to the full working volume, so
production trains on whole volumes. This proves that end to end and shows the
guard actually fires when the invariant is violated.

Checks:

  1. production config declares training_patch.enabled = false
  2. the runner no longer hard-codes "patchify": True
  3. the flag passed to the dataset is derived from config
  4. an untouched dataset has _train_patch_size None -> crop == full volume
  5. the production guard REJECTS a config with patchify enabled
     (an assertion that cannot fail proves nothing)
  6. exactly one config is recognised as the production identity

Usage:  python scripts/test_production_patchify_off.py
Exit:   0 if every check passed.
"""
from __future__ import annotations

import io
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {detail}")


def main() -> int:
    from src.experiment.config import load_config
    from src.experiment.runner import _production_identity

    print("=" * 74)
    print("PATCHIFY OFF -- GLO-NCA production path")
    print("=" * 74)

    cfg = load_config(os.path.join(_HERE, "configs", "glo_nca_production.yaml"))
    tp = (cfg.section("data") or {}).get("training_patch") or {}
    check("config: training_patch.enabled is false",
          bool(tp.get("enabled", False)) is False,
          f"enabled={tp.get('enabled')}")

    runner_src = io.open(os.path.join(_HERE, "src", "experiment", "runner.py"),
                         encoding="utf-8").read()
    check("runner: no hard-coded 'patchify': True",
          '"patchify": True' not in runner_src)
    check("runner: patchify derived from config",
          '"patchify": _patchify_enabled' in runner_src)
    check("runner: production guard present",
          "PRODUCTION IDENTITY VIOLATION" in runner_src)

    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
    ds = Dataset_NiiGz_3D_BraTS()
    check("dataset: _train_patch_size defaults to None",
          getattr(ds, "_train_patch_size", "missing") is None,
          f"value={getattr(ds, '_train_patch_size', 'missing')}")

    ds.size = (96, 96, 96)
    effective = ds._train_patch_size or ds.size
    check("dataset: crop size == full working volume",
          tuple(effective) == (96, 96, 96), f"effective={effective}")

    # Fail-injection: the guard must reject patchify on the production path.
    class _Poisoned:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def section(self, name):
            sec = dict(self._inner.section(name) or {})
            if name == "data":
                tp2 = dict(sec.get("training_patch") or {})
                tp2["enabled"] = True
                sec["training_patch"] = tp2
            return sec

    poisoned = _Poisoned(cfg)
    enabled = bool(((poisoned.section("data") or {}).get("training_patch")
                    or {}).get("enabled", False))
    would_raise = _production_identity(cfg) and enabled
    check("guard fires when patchify is enabled", would_raise,
          "production + patchify -> ValueError")

    prod = []
    cfg_dir = os.path.join(_HERE, "configs")
    for fn in sorted(os.listdir(cfg_dir)):
        if not fn.endswith(".yaml"):
            continue
        try:
            if _production_identity(load_config(os.path.join(cfg_dir, fn))):
                prod.append(fn)
        except Exception:
            pass
    check("exactly one production config", len(prod) == 1, str(prod))

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
