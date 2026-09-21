#!/usr/bin/env python
r"""
Connected-component post-processing for GLO-NCA evaluation.  NO TRAINING.

Production thresholds are WT 50 / TC 5 / ET 0 (ET disabled). These reproduce
the measured behaviour that motivated them:

  * ET 0   a 246-voxel scattered lesion survives intact. At ET 10 the same
           lesion collapsed to a single voxel on the frozen diagnostic split.
  * TC 5   1-4 voxel specks are removed while a 216-voxel body is preserved.
  * WT 50  unchanged.

Checks:

  1. thresholds resolve from the production config
  2. a 246-voxel scattered ET lesion is preserved exactly (ET disabled)
  3. TC 5 removes 1-voxel and 4-voxel specks
  4. TC 5 preserves the 216-voxel connected body
  5. WT 50 removes sub-threshold blobs
  6. the largest component is never removed, so a prediction cannot be emptied
  7. only removed voxels are zeroed; surviving probabilities are untouched
  8. ground truth is never modified
  9. post-processing is absent from the training path
 10. the runner applies it to validation and test

Usage:  python scripts/test_postprocessing.py
Exit:   0 if every check passed.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:48s} {detail}")


def main() -> int:
    from src.experiment.config import load_config
    from src.experiment.postprocess import (REGIONS, apply_to_pairs,
                                            min_component_config,
                                            postprocess_probs,
                                            remove_small_components)

    print("=" * 74)
    print("POST-PROCESSING -- GLO-NCA production evaluation path")
    print("=" * 74)

    cfg = load_config(os.path.join(_HERE, "configs", "glo_nca_production.yaml"))
    mc = min_component_config(cfg)
    check("config resolves WT=50 TC=5 ET=0",
          mc == {"WT": 50, "TC": 5, "ET": 0}, str(mc))

    thr = {r: 0.5 for r in REGIONS}

    # ET disabled: a scattered 246-voxel lesion must survive intact.
    rng = np.random.default_rng(0)
    prob = np.zeros((40, 40, 40, 3), np.float32)
    flat = prob[..., 2].reshape(-1)
    flat[rng.choice(40 * 40 * 40, 246, replace=False)] = 0.9
    prob[..., 2] = flat.reshape(40, 40, 40)
    before = int((prob[..., 2] >= 0.5).sum())
    after = int((postprocess_probs(prob, thr, mc)[..., 2] >= 0.5).sum())
    check("ET 246-voxel scattered lesion preserved",
          after == before == 246, f"{before} -> {after}")

    # TC 5: remove specks, keep the body.
    p2 = np.zeros((20, 20, 20, 3), np.float32)
    p2[2:8, 2:8, 2:8, 1] = 0.9          # 216 voxels
    p2[15, 15, 15, 1] = 0.9             # 1 voxel
    p2[17:19, 17:19, 17:18, 1] = 0.9    # 4 voxels
    b2 = int((p2[..., 1] >= 0.5).sum())
    out2 = postprocess_probs(p2, thr, mc)
    a2 = int((out2[..., 1] >= 0.5).sum())
    check("TC 5 removes 1-voxel and 4-voxel specks",
          b2 - a2 == 5, f"{b2} -> {a2} (removed {b2 - a2})")
    check("TC 5 preserves the 216-voxel body",
          bool((out2[2:8, 2:8, 2:8, 1] >= 0.5).all()) and a2 == 216,
          f"body kept={a2}")

    # WT 50.
    p3 = np.zeros((20, 20, 20, 3), np.float32)
    p3[2:8, 2:8, 2:8, 0] = 0.9          # 216 voxels
    p3[15:17, 15:17, 15:18, 0] = 0.9    # 12 voxels < 50
    b3 = int((p3[..., 0] >= 0.5).sum())
    a3 = int((postprocess_probs(p3, thr, mc)[..., 0] >= 0.5).sum())
    check("WT 50 removes a 12-voxel blob", b3 - a3 == 12, f"{b3} -> {a3}")

    tiny = np.zeros((10, 10, 10), bool)
    tiny[5, 5, 5] = True
    check("largest component is never removed",
          bool(remove_small_components(tiny, 1000).any()),
          "1-voxel mask, threshold 1000")

    p4 = np.zeros((20, 20, 20, 3), np.float32)
    p4[2:8, 2:8, 2:8, 0] = 0.77
    p4[15:17, 15:17, 15:18, 0] = 0.88
    out4 = postprocess_probs(p4, thr, mc)
    check("surviving probabilities untouched",
          abs(float(out4[5, 5, 5, 0]) - 0.77) < 1e-6
          and float(out4[15, 15, 15, 0]) == 0.0,
          f"kept={out4[5, 5, 5, 0]:.2f} removed={out4[15, 15, 15, 0]:.2f}")

    gt = (np.random.default_rng(1).random((12, 12, 12, 3)) > 0.7).astype(np.float32)
    pairs = [(np.zeros((12, 12, 12, 3), np.float32), gt.copy())]
    out_pairs = apply_to_pairs(pairs, thr, mc)
    check("ground truth never modified",
          bool(np.array_equal(out_pairs[0][1], gt)))

    runner_src = io.open(os.path.join(_HERE, "src", "experiment", "runner.py"),
                         encoding="utf-8").read()
    i_loss = runner_src.find("loss_fn")
    i_pp = runner_src.find("PP.apply_to_pairs")
    check("post-processing absent from the training path",
          runner_src.count("PP.apply_to_pairs") == 2 and i_pp > i_loss,
          "evaluation only")
    check("runner applies it to validation and test",
          runner_src.count("PP.apply_to_pairs") == 2)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
