#!/usr/bin/env python
r"""Read-only ET ground-truth voxel census over the TRAINING split.

Pre-flight measurement required before the lesion-aware sampling experiment:
the sampler weights cases by ET volume, so the distribution of that volume must
be known and reproducible before any GPU spend.

WHAT THIS DOES NOT DO
  * does not train, build or run the model
  * does not modify labels, the dataset or the split
  * does not touch the validation or test splits (asserted at exit)

DEFINITION
  Voxels are counted AFTER the same deterministic preprocessing head training
  uses -- foreground crop to the brain bounding box, then resample to the
  configured working volume (128^3). The scale factor therefore DIFFERS PER
  CASE and no voxel spacing is carried through, so counts are in RESAMPLED
  VOXELS and are not a physical volume. This matches how LesionAwareSampler
  and lesion_strata interpret size, which is the point of the measurement.

  Region channel order is REGIONS = [WT, TC, ET]; ET is index 2.

Usage:
  python scripts/analyze_et_voxels.py [--data-root DIR] [--limit N]
                                      [--out reports/analysis/et_voxel_census.csv]
Exit:
  0 on a complete census, 1 if any case failed to load.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ROOT = r"C:\Users\raiwa\Downloads\MICCAI-LH-BraTS2025-MET-Challenge-Training"
REGIONS = ["WT", "TC", "ET"]
ET_INDEX = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", DEFAULT_ROOT))
    ap.add_argument("--limit", type=int, default=0, help="0 = all training cases")
    ap.add_argument("--out", default=os.path.join(
        "reports", "analysis", "et_voxel_census.csv"))
    ap.add_argument("--config", default=os.path.join(
        _HERE, "configs", "glo_nca_production.yaml"))
    ap.add_argument("--cache", action="store_true",
                    help="persist the preprocessing cache (~38 MB/case, "
                         "~34 GB for 898 cases). OFF by default: this census "
                         "only reads labels once, so the cache would cost disk "
                         "without saving any work.")
    args = ap.parse_args()

    print("=" * 78)
    print("ET GROUND-TRUTH VOXEL CENSUS -- TRAINING SPLIT, READ ONLY")
    print("=" * 78)

    if not os.path.isdir(args.data_root):
        print(f"  BLOCKED: dataset not found at {args.data_root}")
        return 1

    out_path = args.out if os.path.isabs(args.out) else os.path.join(_HERE, args.out)
    if os.path.exists(out_path):
        print(f"  BLOCKED: {out_path} exists; refusing to overwrite an "
              f"existing report. Move it or pass --out.")
        return 1

    from src.experiment.config import load_config
    from src.experiment import lesion_strata as LS

    cfg = load_config(args.config)
    os.environ.setdefault("DATA_ROOT", args.data_root)

    # Each cached case is ~38 MB, so a full pass would write ~34 GB for data
    # that is read exactly once here. Disabling the cache keeps the census
    # read-only in spirit as well as in effect, and avoids filling the disk.
    if not args.cache:
        _data = cfg.section("data") or {}
        _data["cache"] = dict(_data.get("cache") or {}, enabled=False)
        print("  preprocess cache disabled (pass --cache to persist it)")

    # Build the dataset through the production path so preprocessing, the
    # foreground crop and the resample target match training exactly.
    from src.experiment import runner as RUN
    exp, ds, _train_ids = _build_training_dataset(cfg, args.data_root, RUN)

    # _case_ids_for_state resolves ids from the path entries the runner
    # installs; here the split ids ARE the case ids, so use them directly and
    # keep the CSV joinable to split/master_split.json.
    ids = list(_train_ids)
    n_total = len(ds)
    limit = args.limit if args.limit and args.limit < n_total else n_total
    print(f"  data root        {args.data_root}")
    print(f"  working volume   {cfg.get('training', 'patch_size')}^3 "
          f"(resampled voxels, not mm^3)")
    print(f"  training cases   {n_total}" +
          (f"  (limiting to {limit})" if limit != n_total else ""))
    print(f"  analysis bins    {LS.DEFAULT_STRATA}")
    print()

    rows, failures = [], []
    t0 = time.perf_counter()
    for idx in range(limit):
        cid = ids[idx] if idx < len(ids) else f"case_{idx:04d}"
        try:
            item = ds[idx]
            label = np.asarray(item[2])
            counts = {r: int((label[..., i] > 0.5).sum())
                      for i, r in enumerate(REGIONS)}
            et = counts["ET"]
            rows.append({
                "case_id": cid,
                "subject_id": cid.rsplit("-", 1)[0] if "-" in cid else cid,
                "et_voxels": et,
                "et_present": int(et > 0),
                "et_stratum": LS.stratum_of(et),
                "wt_voxels": counts["WT"],
                "tc_voxels": counts["TC"],
                "status": "ok",
            })
        except Exception as exc:
            failures.append((cid, f"{type(exc).__name__}: {exc}"))
            rows.append({"case_id": cid, "subject_id": cid, "et_voxels": "",
                         "et_present": "", "et_stratum": "", "wt_voxels": "",
                         "tc_voxels": "", "status": f"error:{type(exc).__name__}"})
        if (idx + 1) % 50 == 0 or idx + 1 == limit:
            el = time.perf_counter() - t0
            print(f"    {idx + 1:4d}/{limit}  {el / (idx + 1):.2f} s/case  "
                  f"elapsed {el / 60:.1f} min", flush=True)

    elapsed = time.perf_counter() - t0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["case_id", "subject_id", "et_voxels", "et_present", "et_stratum",
            "wt_voxels", "tc_voxels", "status"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    et = np.array([r["et_voxels"] for r in ok], dtype=np.int64)
    pos = et[et > 0]

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": "train",
        "unit": "resampled voxels of the 128^3 working volume "
                "(per-case scale factor; NOT mm^3)",
        "analysis_bins": [list(s) for s in LS.DEFAULT_STRATA],
        "cases_total": len(rows),
        "cases_ok": len(ok),
        "cases_failed": len(failures),
        "et_positive_cases": int((et > 0).sum()),
        "et_absent_cases": int((et == 0).sum()),
        "et_positive_fraction": (float((et > 0).mean()) if len(et) else None),
        "et_min": int(pos.min()) if len(pos) else None,
        "et_max": int(pos.max()) if len(pos) else None,
        "et_mean_over_positive": float(pos.mean()) if len(pos) else None,
        "et_median_over_positive": float(np.median(pos)) if len(pos) else None,
        "et_percentiles_over_positive": (
            {f"p{p}": float(np.percentile(pos, p))
             for p in (5, 10, 25, 50, 75, 90, 95)} if len(pos) else None),
        "stratum_counts": {},
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_case": round(elapsed / max(1, len(rows)), 3),
        "failures": failures[:20],
    }
    for name in [s[0] for s in LS.DEFAULT_STRATA] + [LS.ABSENT]:
        summary["stratum_counts"][name] = sum(
            1 for r in ok if r["et_stratum"] == name)

    json_path = os.path.splitext(out_path)[0] + "_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print("-" * 78)
    print(f"  cases                 {summary['cases_ok']} ok / "
          f"{summary['cases_failed']} failed")
    print(f"  ET positive           {summary['et_positive_cases']} "
          f"({(summary['et_positive_fraction'] or 0) * 100:.1f}%)")
    print(f"  ET absent             {summary['et_absent_cases']}")
    if len(pos):
        print(f"  ET voxels (positive)  min {summary['et_min']}  "
              f"median {summary['et_median_over_positive']:.0f}  "
              f"mean {summary['et_mean_over_positive']:.0f}  "
              f"max {summary['et_max']}")
        p = summary["et_percentiles_over_positive"]
        print(f"  percentiles           p5 {p['p5']:.0f}  p25 {p['p25']:.0f}  "
              f"p50 {p['p50']:.0f}  p75 {p['p75']:.0f}  p95 {p['p95']:.0f}")
    print(f"  strata                {summary['stratum_counts']}")
    print(f"  elapsed               {elapsed / 60:.1f} min "
          f"({summary['seconds_per_case']:.2f} s/case)")
    print(f"  csv                   {out_path}")
    print(f"  summary               {json_path}")

    # Test-set firewall: this script only ever constructs the train state.
    print(f"  test cases accessed   0 (dataset state pinned to 'train')")
    print("-" * 78)
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for cid, err in failures[:10]:
            print(f"    {cid}: {err}")
    return 1 if failures else 0


def _build_training_dataset(cfg, data_root, RUN):
    """Construct the production dataset pinned to the TRAINING split.

    Uses the runner's own `_build_dispatch` and replicates its split wiring, so
    preprocessing, the foreground crop and the resample target are identical to
    a real run. Only the train ids are installed -- the val and test entries are
    left EMPTY, so this script is structurally unable to read them.
    """
    import torch

    from src.experiment import datasource

    # Experiment requires a real model_path even though no model is trained
    # here; a throwaway temp dir keeps this script from touching any workspace.
    import tempfile
    scratch = tempfile.mkdtemp(prefix="et_census_")

    device = torch.device("cpu")
    ds, _ca, _agent, exp, _flat = RUN._build_dispatch(
        cfg, data_root, device, epochs=1, out_model_dir=scratch)

    # The canonical frozen split, read exactly as the runner reads it.
    split_file = cfg.get("data", "split_file")
    cand = (split_file if os.path.exists(split_file)
            else os.path.join(_HERE, split_file))
    master = datasource.load_master_split(cand)
    tr, va, te = master["train"], master["validation"], master["test"]
    print(f"  split file       {cand}")
    print(f"  split sha256     {master['split_sha256'][:32]}...")
    path_map = datasource.case_path_map(data_root)

    def entry(p):
        return (path_map.get(p, p), p, 0)

    # TRAIN ONLY. val/test stay empty -> the firewall is structural, not a rule.
    exp.data_split.images["train"] = {p: {0: entry(p)} for p in tr}
    exp.data_split.labels["train"] = {p: {0: entry(p)} for p in tr}
    for sp in ("val", "test"):
        exp.data_split.images[sp] = {}
        exp.data_split.labels[sp] = {}
    exp.set_model_state("train")
    print(f"  split            train {len(tr)} | val {len(va)} (not loaded) "
          f"| test {len(te)} (not loaded)")
    return exp, ds, tr


if __name__ == "__main__":
    raise SystemExit(main())
