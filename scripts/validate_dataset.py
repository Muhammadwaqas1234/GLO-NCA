#!/usr/bin/env python
r"""Standalone dataset validator.

Usage:
    python scripts/validate_dataset.py --root /path/to/BraTS
    python scripts/validate_dataset.py --config configs/gcp_full.yaml
    python scripts/validate_dataset.py --root /path/to/BraTS --report out.json

Exits 0 on PASS, 1 on FAIL -- so it can gate a CI or a training launcher.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.experiment.dataset_validation import validate_dataset, summarize
from src.experiment.datasource import resolve_data_root


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a BraTS-style dataset.")
    ap.add_argument("--root", default=None, help="dataset root (one folder per patient)")
    ap.add_argument("--config", default=None, help="YAML config to read dataset.root/modalities from")
    ap.add_argument("--limit", type=int, default=None, help="validate only the first N patients")
    ap.add_argument("--report", default=None, help="write the JSON report to this path")
    args = ap.parse_args()

    modalities = None
    root = args.root
    if args.config:
        from src.experiment.config import load_config
        cfg = load_config(args.config)
        modalities = cfg.get("dataset", "modalities")
        root = root or cfg.get("dataset", "root")

    root = resolve_data_root(root)
    report = validate_dataset(root, modalities=modalities, limit=args.limit)
    print(summarize(report))

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nreport written to {args.report}")

    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
