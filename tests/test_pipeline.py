#!/usr/bin/env python
r"""End-to-end check on synthetic data: train GLO-NCA, the ablations and the single-level config, predict, re-score, compare, benchmark."""
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
        for name in ("fixed", "fixed_single", "fixed_noglobal", "glonca"):
            rc = train.main(["--config", os.path.join(REPO, "configs", "experiments", f"brats2021_res96_{name}.yaml"),
                             "--data-root", data, "--output", runs, "--experiment-id", name, *SMALL,
                             *(["--override", "data.modalities=[none,t1c,none,none]"] if name == "fixed_single" else [])])
            status = json.load(open(os.path.join(runs, name, "status.json")))
            check(f"train {name}: completes and tests", rc == 0 and status["state"] == "completed",
                  status["state"])
        fixed = os.path.join(runs, "fixed")
        # A full training checkpoint with the old layer names (models and EMA) resumes and completes.
        import torch
        old_run = os.path.join(runs, "old_names")
        rc = train.main(["--config", os.path.join(REPO, "configs", "experiments", "brats2021_res96_fixed.yaml"),
                         "--data-root", data, "--output", runs, "--experiment-id", "old_names", "--stop-after-epoch", "1",
                         *SMALL])
        from glo_nca.model import LEGACY_LAYER_NAMES
        reverse = {v: k for k, v in LEGACY_LAYER_NAMES.items()}
        old = lambda k: next((o + k[len(n):] for n, o in reverse.items() if k.startswith(n)), k)
        ck = torch.load(os.path.join(old_run, "last.pth"), weights_only=False)
        ck["models"] = [{old(k): v for k, v in sd.items()} for sd in ck["models"]]
        ck["ema"] = [{old(k): v for k, v in sd.items()} for sd in ck["ema"]]
        torch.save(ck, os.path.join(old_run, "last.pth"))
        rc2 = train.main(["--resume", old_run, "--data-root", data])
        state = json.load(open(os.path.join(old_run, "status.json")))["state"]
        check("resume: old-name checkpoint (models and EMA) trains, saves and completes",
              rc == 0 and rc2 == 0 and state == "completed")

        with open(os.path.join(fixed, "test_per_case.csv"), newline="") as fh:
            n_test = len(list(csv.DictReader(fh)))
        check("per-case test scores written", n_test > 0, f"{n_test} cases")

        # Parity: the saved fixed config, run as-is, writes per-case scores from its own final scoring.
        res = json.load(open(os.path.join(fixed, "results.json")))
        tuned = res["test_tuned"]
        with open(os.path.join(fixed, "test_per_case.csv"), newline="") as fh:
            rows = list(csv.DictReader(fh))
        parity = all(abs(sum(float(r[g]) for r in rows) / len(rows) - tuned["test"][g]["dice"]) < 1e-6
                     for g in ("WT", "TC", "ET"))
        spec = tuned.get("evaluation_spec", {})
        check("fixed config: final table, per-case CSV and spec agree", parity and spec.get("full_resolution")
              and spec.get("nested_regions") and "medpy" in spec.get("hd95", ""), spec.get("hd95", ""))

        # Stale results: a changed config makes a resumed evaluation refuse.
        ev = ["--run", fixed, "--data-root", data, "--splits", "test", "--no-tune-thresholds",
              "--full-resolution", "--brats-empty",
              "--hd95-surface", "--nested-regions", "--output", os.path.join(work, "resume.json")]
        rc1, rc2 = evaluate.main(ev), evaluate.main(ev)
        cfg = os.path.join(fixed, "config.yaml")
        original = open(cfg, encoding="utf-8").read()
        open(cfg, "w", encoding="utf-8").write(original + "\n# changed\n")
        rc3 = evaluate.main(ev)
        open(cfg, "w", encoding="utf-8").write(original)
        check("evaluate: resumes unchanged inputs, refuses a changed config", (rc1, rc2, rc3) == (0, 0, 2))

        # Data identity: an edited scan under the same path invalidates the progress file.
        seg = next(os.path.join(data, c, f) for c in sorted(os.listdir(data)) if os.path.isdir(os.path.join(data, c))
                   for f in os.listdir(os.path.join(data, c)) if f.endswith("seg.nii.gz"))
        st = os.stat(seg)
        os.utime(seg, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        rc4 = evaluate.main(ev)
        check("evaluate: refuses resumed scores after a scan changed", rc4 == 2)

        # One effective protocol: overrides are executed and recorded identically.
        ov = ["--run", fixed, "--data-root", data, "--splits", "test", "--no-tune-thresholds", "--ensemble-n", "1",
              "--no-tta", "--case-seed", "7", "--output", os.path.join(work, "ov.json")]
        rc = evaluate.main(ov)
        out_spec = json.load(open(os.path.join(work, "ov.json")))["evaluation_spec"]
        head_spec = json.loads(open(os.path.join(work, "ov.json.progress.jsonl")).readline())["settings"]["evaluation_spec"]
        check("evaluate: executed overrides, progress header and saved spec agree",
              rc == 0 and all(spec["ensemble_n"] == 1 and spec["tta"] is False and spec["case_seed"] == 7
                              and spec["tuned"] is False for spec in (out_spec, head_spec)))

        # Ensemble of two runs with threshold tuning: predictions aligned by case ID.
        rc = evaluate.main(["--run", fixed, "--run", os.path.join(runs, "glonca"), "--data-root", data,
                            "--tune-thresholds", "--ensemble-n", "1", "--no-tta", "--no-hd95",
                            "--output", os.path.join(work, "ens.json")])
        ens = json.load(open(os.path.join(work, "ens.json")))
        check("evaluate: tuned two-model ensemble aligned by case", rc == 0 and ens["test_mean_dice"] is not None)

        out = os.path.join(work, "pred")
        rc = predict.main(["--run", fixed, "--input", data, "--output", out, "--limit", "2", "--save-probs",
                           os.path.join(work, "probs")])
        rebuilt = os.path.join(work, "rebuilt")
        rc2 = predict.main(["--from-probs", os.path.join(work, "probs"), "--output", rebuilt])
        maps = lambda d: sorted(f for f in os.listdir(d) if f.endswith(".nii.gz"))
        same = rc == 0 and rc2 == 0 and maps(out) == maps(rebuilt) and len(maps(out)) == 2
        if same:
            import nibabel as nib
            import numpy as np
            same = all(np.array_equal(nib.load(os.path.join(out, f)).get_fdata(),
                                      nib.load(os.path.join(rebuilt, f)).get_fdata()) for f in maps(out))
        check("predict: label maps, identical from saved probabilities", same, f"{len(maps(out))} cases")
        rc = predict.main(["--run", fixed, "--input", data, "--output", out, "--limit", "2",
                           "--thresholds", "WT=0.3"])
        check("predict: refuses to mix outputs of other settings", rc == 2)
        other = os.path.join(work, "syn_copy")
        shutil.copytree(data, other)
        rc_in = predict.main(["--run", fixed, "--input", other, "--output", out, "--limit", "2"])
        rc_fp = predict.main(["--from-probs", os.path.join(work, "probs"), "--output", rebuilt,
                              "--thresholds", "WT=0.9"])
        check("predict: refuses another input folder and a mismatched --from-probs decode", (rc_in, rc_fp) == (2, 2))

        rc = evaluate.main(["--run", fixed, "--data-root", data, "--splits", "test", "--no-tune-thresholds",
                            "--nested-regions",
                            "--hd95-surface", "--full-resolution", "--brats-empty",
                            "--output", os.path.join(work, "eval.json")])
        res = json.load(open(os.path.join(work, "eval.json")))
        check("evaluate: nested regions, surface HD95 at full resolution",
              rc == 0 and res.get("nested_regions") is True, f"keys {sorted(res)[:4]}")
        rc = evaluate.main(["--run", fixed, "--data-root", data, "--splits", "test",
                            "--output", os.path.join(work, "default.json")])
        spec = json.load(open(os.path.join(work, "default.json")))["evaluation_spec"]
        check("evaluate: without flags follows the run's evaluation spec",
              rc == 0 and spec["full_resolution"] and spec["nested_regions"] and "medpy" in spec["hd95"])
        # Lesion-wise scoring (tuned path): summary, per-case CSV with spec, and a paired lesion-wise comparison.
        rc = evaluate.main(["--run", fixed, "--data-root", data, "--lesion-wise",
                            "--output", os.path.join(work, "lesion.json")])
        lres = json.load(open(os.path.join(work, "lesion.json")))
        lcsv = os.path.join(work, "lesion_per_case.csv")
        cmp_rc = compare_runs.main([lcsv, lcsv, "--metric", "lesion_dice", "--boot", "50"])
        check("evaluate --lesion-wise: summary, per-case CSV, compare_runs --metric lesion_dice",
              rc == 0 and "lesion_dice" in lres["test"]["ET"] and lres["evaluation_spec"]["lesion_wise"]
              and os.path.exists(lcsv) and cmp_rc == 0)
        rc = evaluate.main(["--run", fixed, "--data-root", data, "--splits", "test", "--legacy",
                            "--output", os.path.join(work, "legacy.json")])
        mixed = compare_runs.main([fixed, os.path.join(work, "legacy.json.progress.jsonl"), "--boot", "50"])
        check("compare_runs: refuses inputs scored under different rules", rc == 0 and mixed == 2)

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
        cases2 = [os.path.join(data, c) for c in sorted(d for d in os.listdir(data)
                                                        if os.path.isdir(os.path.join(data, d)))[:2]]
        rc = analyze.main(["erf", "--run", fixed, "--case", *cases2, "--steps", "2,8",
                           "--passes", "1", "--output", os.path.join(work, "erf.json")])
        er = json.load(open(os.path.join(work, "erf.json")))
        targets = {c: er["per_case"][c]["target_voxel"] for c in er["per_case"]}
        check("analyze erf: fixed target per case, summary over cases", rc == 0
              and er["summary"]["8"]["r90_mm"]["cases"] == 2 and len(targets) == 2,
              f"r90 {er['summary']['2']['r90_mm']['mean']:.0f} -> {er['summary']['8']['r90_mm']['mean']:.0f} mm")

        rc = analyze.main(["modalities", "--run", fixed, "--data-root", data, "--limit", "2", "--ensemble-n", "1",
                           "--no-tta", "--output", os.path.join(work, "lomo.json")])
        lomo = json.load(open(os.path.join(work, "lomo.json")))["results"]
        check("analyze modalities: all and one row per missing modality", rc == 0 and len(lomo) == 5)
        rc = analyze.main(["manifest", "--data-root", data, "--run", fixed, "--output", os.path.join(work, "man.csv")])
        with open(os.path.join(work, "man.csv"), newline="") as fh:
            man = list(csv.DictReader(fh))
        check("analyze manifest: one row per case with splits", rc == 0 and len(man) == 12
              and {r["split"] for r in man} >= {"train", "test"}, f"{len(man)} cases")
        rc = analyze.main(["perturb", "--run", fixed, "--case", os.path.join(data, case), "--distances", "8,16",
                           "--positions", "1", "--passes", "1", "--output", os.path.join(work, "pert.json")])
        check("analyze perturb: effect per distance", rc in (0, 2))
        hist = open(os.path.join(fixed, "history.csv")).readline()
        check("history: live and EMA validation both recorded", "val_other_mean" in hist)

        rc = compare_runs.main([os.path.join(runs, "fixed_noglobal"), os.path.join(runs, "glonca"), "--boot", "200",
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
