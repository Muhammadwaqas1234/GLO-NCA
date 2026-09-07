#!/usr/bin/env python
r"""Create the ONE canonical master patient split, shared by A0/A1/A2/A3/Final.

Generates split/master_split.json from the VALIDATED dataset using seed 42 and
the established 70/15/15 proportions. Patient IDs only -- never image data.
Refuses to overwrite an existing master split unless --force is given, and fails
loudly on any integrity problem (overlap, missing coverage, empty dataset).

Usage:
    python scripts/create_master_split.py --data-root "$VM_DATA_DIR"
    python scripts/create_master_split.py --data-root ... --out split/master_split.json
    python scripts/create_master_split.py --data-root ... --force   # overwrite
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.experiment import datasource
from src.experiment.dataset_validation import validate_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=os.path.join("split", "master_split.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the dataset validator (NOT recommended)")
    ap.add_argument("--force", action="store_true", help="overwrite existing split")
    args = ap.parse_args()

    root = datasource.resolve_data_root(args.data_root)
    if not root:
        print(f"FAIL: dataset root not found: {args.data_root}")
        return 1

    if os.path.exists(args.out) and not args.force:
        print(f"FAIL: master split already exists at {args.out}. "
              f"Refusing to overwrite (use --force to replace). "
              f"A stable split is required for reproducibility.")
        return 1

    if not args.skip_validate:
        rep = validate_dataset(root, limit=None)
        if rep["result"] != "PASS":
            print(f"FAIL: dataset validation did not PASS ({rep['result']}). "
                  f"Fix the dataset before creating the master split.")
            return 1

    try:
        master = datasource.build_master_split(root, seed=args.seed)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2)

    print("MASTER SPLIT CREATED")
    print(f"  file:                {args.out}")
    print(f"  split_version:       {master['split_version']}")
    print(f"  seed:                {master['seed']}")
    print(f"  dataset case count:  {master['dataset_case_count']}")
    print(f"  train count:         {master['train_count']}")
    print(f"  validation count:    {master['val_count']}")
    print(f"  test count:          {master['test_count']}")
    print(f"  split fingerprint:   {master['split_sha256']}")
    print(f"  patient_id_hash:     {master['patient_id_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
