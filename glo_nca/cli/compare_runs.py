r"""Paired comparison of two runs on the same cases: patient-cluster bootstrap CIs and Wilcoxon tests."""
import argparse
import csv
import json
import os
import sys

import numpy as np

from glo_nca.lesion import penalised_hd95

REGIONS = ["WT", "TC", "ET"]


# Spec fields that define what a per-case score means; thresholds and seeds may differ between runs.
SPEC_KEYS = ("metric_version", "full_resolution", "empty_region", "nested_regions", "hd95", "lesion_wise")


def load_spec(path):
    r"""Scoring rules of a per-case file: the run's test_per_case_spec.json or a progress file's settings."""
    if os.path.isdir(path):
        path = os.path.join(path, "test_per_case.csv")
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            s = json.loads(fh.readline()).get("settings", {})
        return {"full_resolution": s.get("full_resolution"), "nested_regions": s.get("nested_regions"),
                "empty_region": "brats" if s.get("brats_empty") else "v7",
                "hd95": (s.get("hd95_convention") if s.get("hd95_surface") else "legacy"),
                "metric_version": s.get("metric_version"), "lesion_wise": bool(s.get("lesion_wise"))}
    side = path[:-len(".csv")] + "_spec.json"        # test_per_case.csv -> test_per_case_spec.json
    if not os.path.exists(side):
        return None
    spec = json.load(open(side, encoding="utf-8"))
    out = {k: spec.get(k) for k in SPEC_KEYS}
    out["empty_region"] = "brats" if str(out["empty_region"]).startswith("brats") else "v7"
    out["lesion_wise"] = bool(out["lesion_wise"])
    out["hd95"] = "legacy" if "legacy" in str(out["hd95"]) else str(out["hd95"]).split(",")[1].strip().split()[0]
    return out


def check_specs(paths, allow_mixed=False):
    r"""Fail unless every input has a scoring spec and all specs agree (or mixing is explicitly allowed)."""
    specs = {p: load_spec(p) for p in paths}
    missing = [p for p, s in specs.items() if s is None]
    if missing and not allow_mixed:
        raise ValueError("no scoring specification for " + ", ".join(missing)
                         + " (scored by older code?); use --allow-mixed-specs for an exploratory comparison")
    known = {p: s for p, s in specs.items() if s is not None}
    if len({json.dumps(s, sort_keys=True) for s in known.values()}) > 1 and not allow_mixed:
        raise ValueError("the inputs were scored under different rules: "
                         + "; ".join(f"{p}: {s}" for p, s in known.items()) + " (use --allow-mixed-specs)")
    return specs


def load_scores(path, metric="dice"):
    r"""Per-case Dice from a run folder, a test_per_case.csv or an evaluate.py progress file (.jsonl)."""
    if os.path.isdir(path):
        path = os.path.join(path, "test_per_case.csv")
    out = {}
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if "case" in row and row.get("split", "test") == "test":
                    out[row["case"]] = {r: _jsonl_value(row["metrics"][r], metric) for r in REGIONS}
    else:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[row["case"]] = {r: _csv_value(row, r, metric) for r in REGIONS}
    return out


def _jsonl_value(m, metric):
    r"""One per-case value from an evaluate.py progress row."""
    if metric == "dice":
        return float(m["dice"])
    if metric.startswith("lesion_"):
        return float(m["lesion"][metric[len("lesion_"):]])
    return penalised_hd95(float(m["hd95"]), m.get("status", "defined"))


def _csv_value(row, region, metric):
    r"""One per-case value from a per-case CSV; HD95 uses the BraTS rule for empty masks."""
    if metric == "dice":
        return float(row[region])
    if metric == "hd95":
        return penalised_hd95(float(row[f"{region}_hd95"]), row.get(f"{region}_hd95_status", "defined"))
    return float(row[f"{region}_{metric}"])


def patient_of(case):
    r"""Patient id (same rule as the split): BraTS 2023+ names end in a scan number; others are one scan each."""
    from glo_nca.data import patient_of as _patient_of
    return _patient_of(case)


def holm(pvalues):
    r"""Holm-Bonferroni adjusted p-values, in the input order; undefined (NaN) p-values stay NaN."""
    valid = [i for i, p in enumerate(pvalues) if not np.isnan(p)]
    adj, running = [float("nan")] * len(pvalues), 0.0
    for rank, i in enumerate(sorted(valid, key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (len(valid) - rank) * pvalues[i]))
        adj[i] = running
    return adj


def compare(a, b, boot=10000, seed=0):
    r"""Paired statistics of B minus A per region and for the mean over regions.

    ``a`` and ``b`` are per-case score dictionaries, or lists of them (one per training seed, paired by
    position). The estimand is the mean over patients of the patient-level difference: the scans of a
    patient are averaged first, and the bootstrap interval and the Wilcoxon test both use these patient
    values. With several seeds, each seed's effect is reported and the interval resamples patients and
    seeds (two-level bootstrap); the Wilcoxon test uses each patient's difference averaged over seeds.
    """
    from scipy.stats import wilcoxon

    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    if len(a) != len(b):
        raise ValueError("A and B need the same number of seeds (paired by position)")
    cases = sorted(set.intersection(*[set(x) for x in a + b]))
    patients = sorted({patient_of(c) for c in cases})
    idx = {p: [c for c in cases if patient_of(c) == p] for p in patients}
    score = lambda s, c, k: s[c][k] if k in REGIONS else float(np.mean([s[c][r] for r in REGIONS]))
    patient_score = lambda s, p, k: float(np.mean([score(s, c, k) for c in idx[p]]))
    rng = np.random.default_rng(seed)
    p_draws = rng.integers(0, len(patients), size=(boot, len(patients)))
    s_draws = rng.integers(0, len(a), size=(boot, len(a)))
    rows = {}
    for k in ["mean"] + REGIONS:
        A = np.array([[patient_score(sa, p, k) for p in patients] for sa in a])     # seeds x patients
        B = np.array([[patient_score(sb, p, k) for p in patients] for sb in b])
        D = B - A
        boots = np.array([D[np.ix_(s_draws[i], p_draws[i])].mean() for i in range(boot)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        per_patient = D.mean(axis=0)
        nz = per_patient[per_patient != 0]
        rows[k] = {"mean_A": float(A.mean()), "mean_B": float(B.mean()),
                   "mean_diff": float(D.mean()), "median_diff": float(np.median(per_patient)),
                   "ci95": [float(lo), float(hi)],
                   "p_wilcoxon": float(wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan"),
                   "better": int((per_patient > 0).sum()), "worse": int((per_patient < 0).sum()),
                   "seed_effects": [float(x) for x in D.mean(axis=1)]}
    # Mean Dice is the primary endpoint; WT/TC/ET are secondary and Holm-corrected among themselves.
    # Correction across several architectural contrasts is done over their primary p-values (holm()).
    rows["mean"]["p_holm"] = rows["mean"]["p_wilcoxon"]
    for r, p in zip(REGIONS, holm([rows[r]["p_wilcoxon"] for r in REGIONS])):
        rows[r]["p_holm"] = p
    return cases, patients, rows


def main(argv=None) -> int:
    r"""Compare run B against run A and print a table; optionally save JSON."""
    ap = argparse.ArgumentParser(description="Paired per-case comparison of two runs (difference = B - A).")
    ap.add_argument("a", nargs="?", help="baseline: run folder, test_per_case.csv or evaluate progress "
                                         ".jsonl; several seeds as a comma-separated list")
    ap.add_argument("b", nargs="?", help="proposed: same formats as a, with the same number of seeds")
    ap.add_argument("--allow-mixed-specs", action="store_true",
                    help="compare inputs scored under different or unknown rules (exploratory; recorded)")
    ap.add_argument("--contrast", action="append", default=[], metavar="NAME=A::B",
                    help="one contrast of a family (repeatable); A and B as above, separated by ::. The primary mean-Dice "
                         "p-values are Holm-corrected across all contrasts")
    ap.add_argument("--metric", default="dice", choices=["dice", "lesion_dice", "hd95", "lesion_hd95"],
                    help="per-case metric to compare (default dice); HD95 (lower is better) uses 374 mm for a missed "
                         "or false region")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", help="write the results as JSON")
    args = ap.parse_args(argv)
    if args.contrast:
        return _family(args)
    if not (args.a and args.b):
        ap.error("give A and B, or one or more --contrast")

    try:
        check_specs(args.a.split(",") + args.b.split(","), args.allow_mixed_specs)
    except ValueError as exc:
        print(f"FAILED: {exc}")
        return 2
    a = [load_scores(x, args.metric) for x in args.a.split(",")]
    b = [load_scores(x, args.metric) for x in args.b.split(",")]
    if len(a) != len(b):
        print(f"FAILED: A has {len(a)} seed(s), B has {len(b)}; pair them by position.")
        return 2
    cases, patients, rows = compare(a, b, args.boot, args.seed)
    if not cases:
        print("FAILED: the runs share no cases.")
        return 2
    if any(len(cases) != len(x) for x in a + b):
        print(f"note: only the {len(cases)} cases shared by every run are compared")
    print(f"{len(cases)} cases, {len(patients)} patients, {len(a)} seed(s) | difference = B - A, "
          f"patient-level | {args.boot} bootstraps over patients" + (" and seeds" if len(a) > 1 else ""))
    print(f"{'':6}{'A':>8}{'B':>8}{'diff':>9}{'median':>9}{'95% CI':>22}{'p':>9}{'p Holm':>9}{'B>A':>6}{'B<A':>6}")
    for k in ["mean"] + REGIONS:
        r = rows[k]
        print(f"{k:6}{r['mean_A']:8.4f}{r['mean_B']:8.4f}{r['mean_diff']:+9.4f}{r['median_diff']:+9.4f}"
              f"  [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]{r['p_wilcoxon']:9.4f}{r['p_holm']:9.4f}"
              f"{r['better']:6d}{r['worse']:6d}")
    if len(a) > 1:
        print("seed-wise mean-Dice effects: " + ", ".join(f"{x:+.4f}" for x in rows["mean"]["seed_effects"]))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"a": args.a, "b": args.b, "seeds": len(a), "cases": len(cases), "patients": len(patients),
                       "estimand": "mean over patients of the patient-level difference (scans averaged per patient)",
                       "bootstrap": args.boot, "results": rows}, fh, indent=2)
    return 0


def _family(args) -> int:
    r"""Several contrasts: each one's table, then Holm correction of the primary p-values across them."""
    results = {}
    for item in args.contrast:
        name, _, pair = item.partition("=")
        a_spec, _, b_spec = pair.partition("::")
        try:
            check_specs(a_spec.split(",") + b_spec.split(","), args.allow_mixed_specs)
        except ValueError as exc:
            print(f"FAILED: {exc}")
            return 2
        a = [load_scores(x, args.metric) for x in a_spec.split(",")]
        b = [load_scores(x, args.metric) for x in b_spec.split(",")]
        if len(a) != len(b) or not a_spec or not b_spec:
            print(f"FAILED: contrast {name} needs A::B with the same number of seeds.")
            return 2
        cases, patients, rows = compare(a, b, args.boot, args.seed)
        results[name] = {"cases": len(cases), "patients": len(patients), "seeds": len(a), "results": rows}
    adjusted = holm([results[n]["results"]["mean"]["p_wilcoxon"] for n in results])
    print(f"{'contrast':24}{'diff':>9}{'95% CI':>22}{'p':>9}{'p Holm (family)':>17}")
    for (name, res), p_adj in zip(results.items(), adjusted):
        r = res["results"]["mean"]
        res["p_holm_family"] = p_adj
        print(f"{name:24}{r['mean_diff']:+9.4f}  [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
              f"{r['p_wilcoxon']:9.4f}{p_adj:17.4f}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump({"family": [c.partition("=")[0] for c in args.contrast], "bootstrap": args.boot,
                       "correction": "Holm across the family on the primary mean-Dice p-values",
                       "contrasts": results}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
