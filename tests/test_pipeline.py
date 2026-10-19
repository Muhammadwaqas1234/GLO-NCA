#!/usr/bin/env python
r"""End-to-end check on synthetic data: train the fixed and plain configs, predict, re-score, compare, benchmark."""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from glo_nca.cli import analyze, benchmark, compare_runs, evaluate, predict, train  # noqa: E402

SMALL = ["--override", "training.epochs=2", "--override", "data.n_patients=8",
         "--override", "evaluation.ensemble_n=1", "--override", "training.workers=0",
         "--override", "training.val_every=1", "--override", "model.input_size=[[16,16,16],[32,32,32]]",
         "--override", "data.modalities=[t1n,t1c,t2w,t2f]"]
R = []


def check(name, ok, detail=""):
    r"""Record and print one check result."""
    R.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}", flush=True)


def main() -> int:
    r"""Run the pipeline in a temporary folder; returns 0 when every check passes."""
    work = tempfile.mkdtemp(prefix="glo_pipeline_")
    try:
        data, runs = os.path.join(work, "syn"), os.path.join(work, "runs")
        subprocess.run([sys.executable, os.path.join(REPO, "tests", "make_synthetic_data.py"), data,
                        "--cases", "12"], check=True, capture_output=True)
        for name in ("fixed", "fixed_plain"):
            rc = train.main(["--config", os.path.join(REPO, "configs", "experiments", f"brats2021_res96_{name}.yaml"),
                             "--data-root", data, "--output", runs, "--experiment-id", name, *SMALL])
            status = json.load(open(os.path.join(runs, name, "status.json")))
            check(f"train {name}: completes and tests", rc == 0 and status["state"] == "completed",
                  status["state"])
        fixed = os.path.join(runs, "fixed")
        with open(os.path.join(fixed, "test_per_case.csv"), newline="") as fh:
            n_test = len(list(csv.DictReader(fh)))
        check("per-case test scores written", n_test > 0, f"{n_test} cases")

        out = os.path.join(work, "pred")
        rc = predict.main(["--run", fixed, "--input", data, "--output", out, "--limit", "2", "--save-probs",
                           os.path.join(work, "probs")])
        rebuilt = os.path.join(work, "rebuilt")
        rc2 = predict.main(["--from-probs", os.path.join(work, "probs"), "--output", rebuilt])
        same = rc == 0 and rc2 == 0 and sorted(os.listdir(out)) == sorted(os.listdir(rebuilt))
        if same:
            import nibabel as nib
            import numpy as np
            same = all(np.array_equal(nib.load(os.path.join(out, f)).get_fdata(),
                                      nib.load(os.path.join(rebuilt, f)).get_fdata()) for f in os.listdir(out))
        check("predict: label maps, identical from saved probabilities", same, f"{len(os.listdir(out))} cases")

        rc = evaluate.main(["--run", fixed, "--data-root", data, "--splits", "test", "--nested-regions",
                            "--hd95-surface", "--full-resolution", "--brats-empty",
                            "--output", os.path.join(work, "eval.json")])
        res = json.load(open(os.path.join(work, "eval.json")))
        check("evaluate: nested regions, surface HD95 at full resolution",
              rc == 0 and res.get("nested_regions") is True, f"keys {sorted(res)[:4]}")

        rc = analyze.main(["decompose", "--run", fixed, "--data-root", data, "--limit", "2",
                           "--output", os.path.join(work, "dec.json")])
        dec = json.load(open(os.path.join(work, "dec.json")))
        check("analyze decompose: six stages, per-case CSV",
              rc == 0 and len(dec["summary"]) == 6
              and os.path.exists(os.path.join(work, "dec_per_case.csv")), f"{dec['cases']} cases")
        rc = analyze.main(["steps", "--run", fixed, "--data-root", data, "--limit", "2", "--steps", "2,4",
                           "--ensemble-n", "1", "--no-tta", "--output", os.path.join(work, "steps.json")])
        st = json.load(open(os.path.join(work, "steps.json")))
        check("analyze steps: one row per step count", rc == 0 and set(st["results"]) == {"2", "4"})
        case = sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d)))[0]
        rc = analyze.main(["erf", "--run", fixed, "--case", os.path.join(data, case), "--steps", "2,8",
                           "--passes", "1", "--output", os.path.join(work, "erf.json")])
        er = json.load(open(os.path.join(work, "erf.json")))["results"]
        check("analyze erf: influence radius grows with steps", rc == 0
              and er["8"]["r90_mm"] >= er["2"]["r90_mm"], f"r90 {er['2']['r90_mm']:.0f} -> {er['8']['r90_mm']:.0f} mm")

        rc = compare_runs.main([os.path.join(runs, "fixed_plain"), fixed, "--boot", "200",
                                "--output", os.path.join(work, "cmp.json")])
        cmp = json.load(open(os.path.join(work, "cmp.json")))
        check("compare_runs: paired table for all regions", rc == 0 and set(cmp["results"]) ==
              {"mean", "WT", "TC", "ET"}, f"{cmp['cases']} cases")

        rc = benchmark.main(["--run", fixed, "--device", "cpu", "--repeats", "2", "--warmup", "1",
                             "--output", os.path.join(work, "bench.json")])
        bench = json.load(open(os.path.join(work, "bench.json")))
        check("benchmark: parameters and latency", rc == 0 and bench["parameters"] == 30138
              and bench["latency_median_s"] > 0, f"{bench['latency_median_s']} s")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n  {sum(R)}/{len(R)} passed")
    return 0 if all(R) else 1


if __name__ == "__main__":
    sys.exit(main())
