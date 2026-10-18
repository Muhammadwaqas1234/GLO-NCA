r"""Paired comparison of two runs on the same cases: patient-cluster bootstrap CIs and Wilcoxon tests."""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

REGIONS = ["WT", "TC", "ET"]


def load_scores(path):
    r"""Per-case Dice from a run folder, a test_per_case.csv or an evaluate.py progress file (.jsonl)."""
    if os.path.isdir(path):
        path = os.path.join(path, "test_per_case.csv")
    out = {}
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if "case" in row and row.get("split", "test") == "test":
                    out[row["case"]] = {r: float(row["metrics"][r]["dice"]) for r in REGIONS}
    else:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[row["case"]] = {r: float(row[r]) for r in REGIONS}
    return out


def patient_of(case):
    r"""Patient id: BraTS 2023 names end in a scan number (BraTS-GLI-00001-001); other names are one scan each."""
    m = re.match(r"^(.*-\d+)-\d+$", case)
    return m.group(1) if m else case


def holm(pvalues):
    r"""Holm-Bonferroni adjusted p-values, in the input order; undefined (NaN) p-values stay NaN."""
    valid = [i for i, p in enumerate(pvalues) if not np.isnan(p)]
    adj, running = [float("nan")] * len(pvalues), 0.0
    for rank, i in enumerate(sorted(valid, key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (len(valid) - rank) * pvalues[i]))
        adj[i] = running
    return adj


def compare(a, b, boot=10000, seed=0):
    r"""Paired statistics of B minus A per region and for the mean over regions."""
    from scipy.stats import wilcoxon

    cases = sorted(set(a) & set(b))
    patients = sorted({patient_of(c) for c in cases})
    idx = {p: [i for i, c in enumerate(cases) if patient_of(c) == p] for p in patients}
    score = lambda s, c, k: s[c][k] if k in REGIONS else float(np.mean([s[c][r] for r in REGIONS]))
    draws = np.random.default_rng(seed).integers(0, len(patients), size=(boot, len(patients)))
    sizes = np.array([len(idx[p]) for p in patients])
    rows = {}
    for k in ["mean"] + REGIONS:
        d = np.array([score(b, c, k) - score(a, c, k) for c in cases])
        sums = np.array([d[idx[p]].sum() for p in patients])
        # Cluster bootstrap: resample patients and keep every scan of each drawn patient.
        lo, hi = np.percentile(sums[draws].sum(axis=1) / sizes[draws].sum(axis=1), [2.5, 97.5])
        per_patient = sums / sizes
        nz = per_patient[per_patient != 0]
        rows[k] = {"mean_A": float(np.mean([score(a, c, k) for c in cases])),
                   "mean_B": float(np.mean([score(b, c, k) for c in cases])),
                   "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                   "ci95": [float(lo), float(hi)],
                   "p_wilcoxon": float(wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan"),
                   "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
    # Mean Dice is the primary endpoint; WT/TC/ET are secondary and Holm-corrected among themselves.
    rows["mean"]["p_holm"] = rows["mean"]["p_wilcoxon"]
    for r, p in zip(REGIONS, holm([rows[r]["p_wilcoxon"] for r in REGIONS])):
        rows[r]["p_holm"] = p
    return cases, patients, rows


def main(argv=None) -> int:
    r"""Compare run B against run A and print a table; optionally save JSON."""
    ap = argparse.ArgumentParser(description="Paired per-case comparison of two runs (difference = B - A).")
    ap.add_argument("a", help="baseline: run folder, test_per_case.csv or evaluate progress .jsonl")
    ap.add_argument("b", help="proposed: same formats as a")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", help="write the results as JSON")
    args = ap.parse_args(argv)

    a, b = load_scores(args.a), load_scores(args.b)
    cases, patients, rows = compare(a, b, args.boot, args.seed)
    if not cases:
        print("FAILED: the two runs share no cases.")
        return 2
    if len(cases) != len(a) or len(cases) != len(b):
        print(f"note: only the {len(cases)} shared cases are compared (A {len(a)}, B {len(b)})")
    print(f"{len(cases)} cases, {len(patients)} patients | difference = B - A | {args.boot} cluster bootstraps")
    print(f"{'':6}{'A':>8}{'B':>8}{'diff':>9}{'median':>9}{'95% CI':>22}{'p':>9}{'p Holm':>9}{'B>A':>6}{'B<A':>6}")
    for k in ["mean"] + REGIONS:
        r = rows[k]
        print(f"{k:6}{r['mean_A']:8.4f}{r['mean_B']:8.4f}{r['mean_diff']:+9.4f}{r['median_diff']:+9.4f}"
              f"  [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]{r['p_wilcoxon']:9.4f}{r['p_holm']:9.4f}"
              f"{r['better']:6d}{r['worse']:6d}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"a": args.a, "b": args.b, "cases": len(cases), "patients": len(patients),
                       "bootstrap": args.boot, "results": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
