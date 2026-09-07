#!/usr/bin/env python
r"""Create a one-time dataset identity manifest (Phase 3 sec. 10).

Captures: name, root, case count, case IDs, per-patient file list, modality
names, label mapping, and a lightweight content fingerprint (sha256 over the
sorted (relative-path, size) list -- NOT over file bytes, to avoid rehashing
gigabytes every run; documented as such).

Usage:
    python scripts/dataset_identity.py --root /path/to/BraTS --out dataset_identity.json
"""
import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
LABEL_MAPPING = {"note": "raw BraTS labels -> nested regions",
                 "WT": "labels {1,2,3/4}", "TC": "labels {1,3/4}",
                 "ET": "label {4} or {3} (year-dependent)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", default="BraTS")
    ap.add_argument("--out", default="dataset_identity.json")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"FAIL: dataset root not found: {root}")
        return 1

    cases = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    per_case = {}
    fingerprint_items = []
    for c in cases:
        cdir = os.path.join(root, c)
        files = sorted(f for f in os.listdir(cdir)
                       if f.endswith((".nii", ".nii.gz")))
        per_case[c] = files
        for f in files:
            rel = f"{c}/{f}"
            size = os.path.getsize(os.path.join(cdir, f))
            fingerprint_items.append(f"{rel}:{size}")

    fp = hashlib.sha256("\n".join(fingerprint_items).encode()).hexdigest()
    manifest = {
        "dataset_name": args.name,
        "root": os.path.abspath(root),
        "case_count": len(cases),
        "case_ids": cases,
        "modalities": MODALITIES,
        "label_mapping": LABEL_MAPPING,
        "fingerprint_sha256": fp,
        "fingerprint_method": "sha256 over sorted '<case>/<file>:<bytes>' list "
                              "(structure+size, not file content)",
        "files_per_case": per_case,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"dataset identity: {len(cases)} cases, fingerprint {fp[:16]}...")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
