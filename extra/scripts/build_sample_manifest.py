#!/usr/bin/env python
r"""Build the authoritative 100-case sample manifest with cryptographic integrity.
Diagnostic only; does not touch production code or the canonical split.

Records per case: id, cohort, relative_path, the 4 modality paths + seg path, file
sizes, SHA256 per file, ET present/absent, shape. Plus totals + a manifest SHA256
computed over the sorted (case_id, per-file sha256) tuples.
"""
import argparse, json, os, sys, glob, hashlib
import numpy as np, nibabel as nib

MODS = ["t1n", "t1c", "t2w", "t2f"]


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sha", action="store_true", help="compute per-file SHA256 (slow)")
    args = ap.parse_args()

    root = args.staging
    segs = sorted(glob.glob(os.path.join(root, "**", "*-seg.nii.gz"), recursive=True))
    cases = []
    for s in segs:
        d = os.path.dirname(s); cid = os.path.basename(s)[:-len("-seg.nii.gz")]
        rel = os.path.relpath(d, root)
        cohort = rel.split(os.sep)[0] if os.sep in rel else "_flat_"
        rec = {"case_id": cid, "cohort": cohort, "relative_path": rel.replace(os.sep, "/")}
        for m in MODS + ["seg"]:
            p = os.path.join(d, f"{cid}-{m}.nii.gz")
            rec[f"{m}_path"] = os.path.relpath(p, root).replace(os.sep, "/")
            rec[f"{m}_size"] = os.path.getsize(p) if os.path.exists(p) else None
            if args.sha and os.path.exists(p):
                rec[f"{m}_sha256"] = sha256(p)
        arr = np.asarray(nib.load(s).dataobj)
        rec["et_present"] = bool((arr == 3).any())
        rec["shape"] = list(int(x) for x in arr.shape)
        cases.append(rec)

    cases.sort(key=lambda c: c["case_id"])
    et_p = sum(1 for c in cases if c["et_present"])
    flat = sum(1 for c in cases if c["cohort"] == "_flat_")
    nested = sum(1 for c in cases if c["cohort"] == "UCSD - Training")

    # manifest SHA over sorted (id, seg_sha or seg_size) — deterministic identity
    hasher = hashlib.sha256()
    for c in cases:
        key = c.get("seg_sha256") or str(c.get("seg_size"))
        hasher.update((c["case_id"] + ":" + key).encode())
    manifest_sha = hasher.hexdigest()

    manifest = {
        "seed": 42, "total_cases": len(cases), "flat_cases": flat, "nested_cases": nested,
        "et_present": et_p, "et_absent": len(cases) - et_p,
        "case_ids": [c["case_id"] for c in cases],
        "case_ids_sha256": hashlib.sha256("".join(c["case_id"] for c in cases).encode()).hexdigest(),
        "manifest_sha256": manifest_sha,
        "cases": cases,
    }
    json.dump(manifest, open(args.out, "w"), indent=2)
    print("wrote", args.out)
    print(f"cases={len(cases)} flat={flat} nested={nested} ET+={et_p} ET-={len(cases)-et_p}")
    print("case_ids_sha256:", manifest["case_ids_sha256"])
    print("manifest_sha256:", manifest_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
