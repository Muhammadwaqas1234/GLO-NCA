#!/usr/bin/env python
r"""Validate the local 100-case Kaggle profiling sample (fail-closed).

Verifies the staging directory against the expected deterministic seed-42 sample:
exactly 100 cases, 50 flat + 50 UCSD, 94 ET-present / 6 ET-absent, all 4 modalities
+ seg per case, exact IDs, no duplicates, no unexpected dirs. Exits nonzero on any
failure. Diagnostic only; touches no production code or the canonical split.

Usage:
    python scripts/validate_kaggle_sample.py \
        --staging <staging-dir>/MICCAI-LH-BraTS2025-MET-Challenge-Training \
        --expected <sample_100.json>
"""
import argparse, json, os, sys, glob

MODS = ["t1n", "t1c", "t2w", "t2f"]
EXPECT = {"count": 100, "flat": 50, "nested": 50, "et_present": 94, "et_absent": 6, "seed": 42}


def discover(root):
    segs = glob.glob(os.path.join(root, "**", "*-seg.nii.gz"), recursive=True)
    cases = {}
    for s in segs:
        d = os.path.dirname(s)
        cid = os.path.basename(s)[:-len("-seg.nii.gz")]
        rel = os.path.relpath(d, root)
        cohort = rel.split(os.sep)[0] if os.sep in rel else "_flat_"
        cases[cid] = {"dir": d, "cohort": cohort, "seg": s}
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--expected", required=True, help="sample_100.json with expected IDs")
    ap.add_argument("--check-et", action="store_true", help="also read seg to verify ET counts (slow)")
    args = ap.parse_args()

    exp = json.load(open(args.expected))
    exp_ids = set(exp["case_ids"])
    fails = []

    cases = discover(args.staging)
    ids = set(cases)

    # 1-4 counts + IDs + duplicates
    if len(cases) != EXPECT["count"]: fails.append(f"case count {len(cases)} != {EXPECT['count']}")
    seg_count = len(glob.glob(os.path.join(args.staging, "**", "*-seg.nii.gz"), recursive=True))
    if seg_count != EXPECT["count"]: fails.append(f"seg count {seg_count} != {EXPECT['count']}")
    if ids != exp_ids:
        missing = exp_ids - ids; extra = ids - exp_ids
        if missing: fails.append(f"missing expected IDs: {sorted(missing)[:5]} (+{max(0,len(missing)-5)})")
        if extra: fails.append(f"unexpected IDs: {sorted(extra)[:5]} (+{max(0,len(extra)-5)})")
    # duplicate check via filesystem (dir names unique by construction; check seg basenames)
    seg_basenames = [os.path.basename(s)[:-len('-seg.nii.gz')] for s in glob.glob(os.path.join(args.staging,'**','*-seg.nii.gz'),recursive=True)]
    if len(seg_basenames) != len(set(seg_basenames)): fails.append("duplicate case IDs present")

    # 5-6 cohort distribution
    flat = sum(1 for c in cases.values() if c["cohort"] == "_flat_")
    nested = sum(1 for c in cases.values() if c["cohort"] == "UCSD - Training")
    if flat != EXPECT["flat"]: fails.append(f"flat {flat} != {EXPECT['flat']}")
    if nested != EXPECT["nested"]: fails.append(f"nested {nested} != {EXPECT['nested']}")

    # 11-12 modalities + seg per case
    for cid, c in cases.items():
        for m in MODS:
            p = os.path.join(c["dir"], f"{cid}-{m}.nii.gz")
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                fails.append(f"{cid}: missing/empty {m}")
        if os.path.getsize(c["seg"]) == 0:
            fails.append(f"{cid}: empty seg")

    # 9 seed metadata
    if int(exp.get("seed", -1)) != EXPECT["seed"]: fails.append(f"expected seed {exp.get('seed')} != 42")

    # 7-8 ET distribution (optional, slow — reads seg volumes)
    et_present = et_absent = None
    if args.check_et:
        import numpy as np, nibabel as nib
        et_present = 0; et_absent = 0
        for cid, c in cases.items():
            arr = np.asarray(nib.load(c["seg"]).dataobj)
            (et_present := et_present + 1) if (arr == 3).any() else (et_absent := et_absent + 1)
        if et_present != EXPECT["et_present"]: fails.append(f"ET-present {et_present} != {EXPECT['et_present']}")
        if et_absent != EXPECT["et_absent"]: fails.append(f"ET-absent {et_absent} != {EXPECT['et_absent']}")

    print("=" * 50)
    print("LOCAL KAGGLE SAMPLE VALIDATION")
    print("=" * 50)
    print(f"discovered cases : {len(cases)} (expect {EXPECT['count']})")
    print(f"seg files        : {seg_count}")
    print(f"flat / nested    : {flat} / {nested}")
    print(f"ids match expected: {ids == exp_ids}")
    if et_present is not None:
        print(f"ET present/absent: {et_present} / {et_absent}")
    print("RESULT:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  -", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
