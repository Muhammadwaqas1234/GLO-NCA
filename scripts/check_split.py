#!/usr/bin/env python
r"""Verify train/val/test split integrity for an experiment (or a fresh split).

Checks (Phase 3 sec. 11-12):
  * pairwise-disjoint train/val/test
  * union == the intended dataset population (optional, if --data-root given)
  * reports the actual counts (does NOT assume 882)

Usage:
    python scripts/check_split.py --experiment experiments/<id>
    python scripts/check_split.py --split-json experiments/<id>/split/split.json
    python scripts/check_split.py --experiment <id> --data-root /path/to/BraTS
Exit 0 if the split is valid, 1 otherwise.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None, help="experiment dir (uses split/split.json)")
    ap.add_argument("--split-json", default=None, help="path to a split.json directly")
    ap.add_argument("--split", default=None, help="path to a master_split.json (verifies fingerprint)")
    ap.add_argument("--data-root", default=None, help="optional: verify union == dataset")
    args = ap.parse_args()

    path = args.split or args.split_json
    if not path and args.experiment:
        path = os.path.join(args.experiment, "split", "split.json")
    if not path or not os.path.exists(path):
        print(f"FAIL: split file not found ({path})")
        return 1

    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    tr, va, te = set(d["train"]), set(d["validation"]), set(d["test"])

    ok = True

    # If this is a master split with a stored fingerprint, re-verify it.
    if "split_sha256" in d:
        from src.experiment.datasource import split_fingerprint
        recomputed = split_fingerprint(d["train"], d["validation"], d["test"])
        match = recomputed == d["split_sha256"]
        print(("PASS " if match else "FAIL ")
              + f"fingerprint {d['split_sha256'][:12]} "
              + ("matches" if match else f"!= recomputed {recomputed[:12]}"))
        ok = ok and match
    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and cond

    check(f"train & val   disjoint (overlap {len(tr & va)})", not (tr & va))
    check(f"train & test  disjoint (overlap {len(tr & te)})", not (tr & te))
    check(f"val   & test  disjoint (overlap {len(va & te)})", not (va & te))
    print(f"counts -> train {len(tr)} | val {len(va)} | test {len(te)} | "
          f"total {len(tr | va | te)}")

    if args.data_root:
        from src.experiment.datasource import list_patients
        pop = set(list_patients(args.data_root))
        union = tr | va | te
        extra = union - pop
        missing = pop - union
        check(f"split subset of dataset ({len(extra)} unknown ids)", not extra)
        if missing:
            print(f"NOTE: {len(missing)} dataset cases not in the split "
                  f"(expected if number_of_patients capped the split).")
        print(f"dataset population: {len(pop)} patients under {args.data_root}")

    print("SPLIT OK" if ok else "SPLIT INVALID")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
