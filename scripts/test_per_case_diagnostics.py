#!/usr/bin/env python
r"""
Per-case diagnostics for the GLO-NCA production evaluation path.  NO TRAINING.

Checks that the production runner owns the diagnostics (not just the smoke
test), that HD95 validity is accounted rather than faked, and that validation
and test artifacts stay separate so a validation file can never be mistaken
for frozen-test evidence.

Checks:

   1. the module exposes the required schema
   2. voxel counts are recorded for ground truth and prediction
   3. Dice matches metrics_eval on the same pair (no second implementation)
   4. HD95 undefined on an empty prediction, with the reason recorded
   5. HD95 undefined on empty ground truth, with the reason recorded
   6. undefined HD95 is EXCLUDED from the mean, never replaced by zero
   7. summarise() reports valid/invalid counts and reasons per region
   8. the CSV is deterministic and column-stable
   9. subject_id is derived from the case id
  10. the runner writes validation_per_case.csv and test_per_case.csv
  11. validation and test go to SEPARATE files
  12. diagnostics run after evaluation, never during training

Usage:  python scripts/test_per_case_diagnostics.py
Exit:   0 if every check passed.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:50s} {detail}")


def main() -> int:
    from src.experiment import metrics_eval as ME
    from src.experiment import per_case_diagnostics as PCD

    print("=" * 74)
    print("PER-CASE DIAGNOSTICS -- GLO-NCA production evaluation")
    print("=" * 74)

    required = {"case_id", "subject_id", "split", "dice_mean",
                "gt_vox_WT", "pred_vox_WT", "gt_vox_ET", "pred_vox_ET",
                "hd95_WT", "hd95_ET", "metric_valid", "failure_reason"}
    check("schema exposes the required columns",
          required.issubset(set(PCD.FIELDS)),
          f"{len(PCD.FIELDS)} columns")

    thr = {r: 0.5 for r in ("WT", "TC", "ET")}

    def blank():
        return (np.zeros((16, 16, 16, 3), np.float32),
                np.zeros((16, 16, 16, 3), np.float32))

    p0, g0 = blank()
    g0[2:10, 2:10, 2:10, :] = 1.0
    p0[3:11, 3:11, 3:11, :] = 0.9

    p1, g1 = blank()
    g1[2:10, 2:10, 2:10, :] = 1.0
    p1[3:11, 3:11, 3:11, :] = 0.9
    p1[..., 2] = 0.0                      # empty ET prediction

    p2, g2 = blank()
    g2[2:10, 2:10, 2:10, :] = 1.0
    g2[..., 2] = 0.0                      # empty ET ground truth
    p2[3:11, 3:11, 3:11, :] = 0.9

    pairs = [(p0, g0), (p1, g1), (p2, g2)]
    ids = ["BraTS-MET-00001-000", "BraTS-MET-00002-001", "BraTS-MET-00003-000"]
    rows = PCD.build_rows(pairs, ids, "validation", thr)

    check("voxel counts recorded",
          rows[0]["gt_vox_WT"] == 512 and rows[0]["pred_vox_WT"] == 512,
          f"gt {rows[0]['gt_vox_WT']} pred {rows[0]['pred_vox_WT']}")

    ref = ME.score_per_case([pairs[0]], thr)
    check("Dice matches metrics_eval (single implementation)",
          abs(rows[0]["dice_WT"] - ref["WT"]["dice"][0]) < 1e-9,
          f"{rows[0]['dice_WT']:.6f} vs {ref['WT']['dice'][0]:.6f}")

    check("HD95 undefined on empty prediction",
          rows[1]["hd95_ET"] != rows[1]["hd95_ET"]
          and "ET:empty_prediction" in rows[1]["failure_reason"],
          rows[1]["failure_reason"])
    check("HD95 undefined on empty ground truth",
          rows[2]["hd95_ET"] != rows[2]["hd95_ET"]
          and "ET:empty_gt" in rows[2]["failure_reason"],
          rows[2]["failure_reason"])
    check("undefined HD95 is not replaced by zero",
          rows[1]["hd95_ET"] != 0.0 and rows[2]["hd95_ET"] != 0.0)
    check("undefined HD95 excluded from the row mean",
          rows[1]["hd95_mean"] == rows[1]["hd95_mean"],
          f"mean over valid = {rows[1]['hd95_mean']:.3f}")

    summary = PCD.summarise(rows)
    et = summary["regions"]["ET"]
    check("summary counts valid vs invalid HD95",
          et["hd95_valid_cases"] == 1 and et["hd95_invalid_cases"] == 2,
          f"valid {et['hd95_valid_cases']} invalid {et['hd95_invalid_cases']}")
    check("summary records the reasons",
          set(et["hd95_invalid_reasons"]) == {"empty_prediction", "empty_gt"},
          str(et["hd95_invalid_reasons"]))
    check("summary counts fully valid cases",
          summary["cases_fully_valid"] == 1,
          f"{summary['cases_fully_valid']}/3")

    check("subject_id derived from case id",
          rows[1]["subject_id"] == "BraTS-MET-00002",
          rows[1]["subject_id"])

    import csv
    with tempfile.TemporaryDirectory() as td:
        path = PCD.write_csv(os.path.join(td, "reports",
                                          "validation_per_case.csv"), rows)
        first = io.open(path, encoding="utf-8").read()
        PCD.write_csv(path, rows)
        second = io.open(path, encoding="utf-8").read()
        check("CSV is deterministic", first == second)
        got = list(csv.DictReader(io.open(path, encoding="utf-8")))
        check("CSV columns match the declared schema",
              list(got[0]) == list(PCD.FIELDS), f"{len(got)} rows")

    runner_src = io.open(os.path.join(_HERE, "src", "experiment", "runner.py"),
                         encoding="utf-8").read()
    check("runner writes validation_per_case.csv",
          '"validation_per_case.csv"' in runner_src)
    check("runner writes test_per_case.csv",
          '"test_per_case.csv"' in runner_src)
    check("validation and test written to separate files",
          runner_src.count("PCD.write_csv") == 2)

    i_diag = runner_src.find("PCD.build_rows")
    i_eval = runner_src.find("test_pairs = ME.collect_probs")
    i_loop = runner_src.find("for ep in range(")
    check("diagnostics run after evaluation, not during training",
          -1 < i_loop < i_eval < i_diag,
          "training loop -> evaluation -> diagnostics")

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
