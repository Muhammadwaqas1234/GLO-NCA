r"""Re-evaluate one run or an ensemble of runs; resumable, with optional tuned post-processing."""
import argparse
import json
import os
import sys


def _parse_thresholds(text):
    r"""Parse 'WT=0.65,TC=0.65,ET=0.6' into per-region thresholds (default 0.5)."""
    out = {"WT": 0.5, "TC": 0.5, "ET": 0.5}
    for item in filter(None, (text or "").split(",")):
        region, _, value = item.partition("=")
        if region.strip() not in out:
            raise ValueError(f"unknown region in --thresholds: {region}")
        out[region.strip()] = float(value)
    return out


def _parse_min_component(text):
    r"""Parse 'WT=0,TC=50,ET=50' into a per-region minimum component size."""
    out = {"WT": 0, "TC": 0, "ET": 0}
    for item in filter(None, (text or "").split(",")):
        region, _, value = item.partition("=")
        if region.strip() not in out:
            raise ValueError(f"unknown region in --min-component: {region}")
        out[region.strip()] = int(value)
    return out


def file_sha256(path):
    r"""SHA-256 of a file, read in blocks."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Settings that change the inputs a model sees; ensemble members must agree on all of them.
# Their inverse transform back to the scan follows from the crop, the working size and the resampling,
# so INPUT_SIZE and USE_FOREGROUND_CROP also guarantee the same inverse mapping.
_INPUT_KEYS = ("INPUT_SIZE", "MODALITIES", "ET_LABELS", "USE_FOREGROUND_CROP", "USE_NONZERO_NORM", "BRAIN_MASK",
               "LEGACY_PATCHIFY")


def main(argv=None) -> int:
    r"""Re-evaluate one run or an ensemble of runs and write the result JSON."""
    ap = argparse.ArgumentParser(description="Re-evaluate one run or an ensemble of runs.")
    ap.add_argument("--run", action="append", required=True, metavar="RUN_DIR",
                    help="run folder with config.yaml, split.json and best.pth (repeatable)")
    ap.add_argument("--data-root", required=True, help="folder that holds the BraTS case folders")
    ap.add_argument("--output", default=None, help="result JSON (default: <first run>/evaluation.json)")
    ap.add_argument("--tune-thresholds", action=argparse.BooleanOptionalAction, default=None,
                    help="choose per-region threshold and clean-up size on validation (default: the run's setting)")
    ap.add_argument("--min-component", default="", metavar="WT=0,TC=50,ET=50",
                    help="fixed clean-up: drop predicted components below this size (default: the run's "
                         "setting; ignored when tuning)")
    ap.add_argument("--full-resolution", action=argparse.BooleanOptionalAction, default=None,
                    help="score in the original scan space (default: the run's evaluation.full_resolution)")
    ap.add_argument("--ensemble-n", type=int, default=None,
                    help="stochastic passes per model (default: each run's evaluation.ensemble_n)")
    ap.add_argument("--no-tta", action="store_true", help="disable flip test-time augmentation")
    ap.add_argument("--case-seed", type=int, default=None,
                    help="seed the stochastic passes per case (base + hash of the case name), for exact repeats")
    ap.add_argument("--brats-empty", action=argparse.BooleanOptionalAction, default=None,
                    help="score an empty prediction of an absent region as Dice 1 (default: the run's setting)")
    ap.add_argument("--no-hd95", action="store_true",
                    help="skip HD95 (the slowest metric at full resolution); Dice and IoU only")
    ap.add_argument("--hd95-surface", action=argparse.BooleanOptionalAction, default=None,
                    help="surface-based HD95, in mm at full resolution (default: the run's setting)")
    ap.add_argument("--hd95-convention", default=None, choices=["medpy", "max_directed"],
                    help="medpy: one percentile over both directions; max_directed: larger directed "
                         "percentile (default: the run's setting)")
    ap.add_argument("--nested-regions", action=argparse.BooleanOptionalAction, default=None,
                    help="score the final nested masks, as predict.py writes them (default: the run's setting)")
    ap.add_argument("--legacy", action="store_true",
                    help="the original scoring: working grid, v7 empty rule, independent masks, voxel HD95")
    ap.add_argument("--thresholds", default="", metavar="WT=0.65,TC=0.65,ET=0.6",
                    help="fixed per-region thresholds, e.g. those a run chose on validation (ignored when tuning)")
    ap.add_argument("--splits", default="val,test", help="splits to score without tuning (default val,test)")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    from glo_nca.config import REGIONS, load_config
    from glo_nca.data import find_data_root
    from glo_nca.evaluation import collect_probs, final_evaluation, score_streaming
    from glo_nca.metrics import METRIC_VERSION
    from glo_nca.reporting import report_tuned
    from glo_nca.trainer import build, code_identity, set_seed

    runs = [os.path.abspath(r) for r in args.run]
    splits = []
    for r in runs:
        with open(os.path.join(r, "split.json"), encoding="utf-8") as fh:
            s = json.load(fh)
        splits.append((s["train"], s["validation"], s["test"]))
    if any(s != splits[0] for s in splits[1:]):
        print("FAILED: the runs use different splits; an ensemble needs one shared split.json.")
        return 2
    data_root, _ = find_data_root(args.data_root)
    if not data_root:
        print(f"FAILED: BraTS not found under {args.data_root}")
        return 2

    models, datasets, ds, C0 = [], [], None, None
    for r in runs:
        cfg = load_config(os.path.join(r, "config.yaml"))
        if C0 is not None:
            differ = [k for k in _INPUT_KEYS if getattr(cfg, k) != getattr(C0, k)]
            if differ:
                print(f"FAILED: ensemble members differ in {', '.join(differ)}.")
                return 2
        C0 = C0 or cfg
        set_seed(cfg.SEED)
        ds_r, cells, agent, dev = build(cfg, data_root, r, splits[0])
        ds = ds or ds_r
        datasets.append(ds_r)      # each model reads its own dataset, in its own split state
        if tuple(ds_r.size) != tuple(ds.size) or ds_r.images_path != ds.images_path:
            print("FAILED: ensemble members differ in working size or data root.")
            return 2
        ck = torch.load(os.path.join(r, "best.pth"), map_location=dev, weights_only=False)
        for m, sd in zip(cells, ck["m"]):
            m.load_state_dict(sd)
        n = args.ensemble_n if args.ensemble_n is not None else cfg.ENSEMBLE_N
        tta = cfg.USE_TTA and not args.no_tta
        print(f"{os.path.basename(r)}: best epoch {ck['ep']} (val {ck['val_mean']:.3f}), "
              f"ensemble {n}, tta {tta}", flush=True)
        if args.case_seed is not None:
            agent.case_seed = args.case_seed
        models.append((agent, n, tta))
    # One effective protocol: the run's config with every command-line override applied. It drives the
    # inference, the progress identity and the evaluation_spec written with the results.
    pick = lambda flag, run_value: (False if args.legacy else run_value) if flag is None else flag
    C0.ENSEMBLE_N, C0.USE_TTA, C0.CASE_SEED = models[0][1], models[0][2], models[0][0].case_seed
    C0.TUNE_THRESHOLDS = pick(args.tune_thresholds, C0.TUNE_THRESHOLDS)
    if args.min_component or args.legacy:
        C0.MIN_COMPONENT = _parse_min_component(args.min_component)
    C0.FULL_RESOLUTION_EVAL = pick(args.full_resolution, C0.FULL_RESOLUTION_EVAL)
    C0.BRATS_EMPTY = pick(args.brats_empty, C0.BRATS_EMPTY)
    C0.NESTED_REGIONS = pick(args.nested_regions, C0.NESTED_REGIONS)
    C0.SURFACE_HD95 = pick(args.hd95_surface, C0.SURFACE_HD95)
    C0.HD95_CONVENTION = args.hd95_convention or C0.HD95_CONVENTION
    args.tune_thresholds, args.full_resolution, args.brats_empty = (C0.TUNE_THRESHOLDS, C0.FULL_RESOLUTION_EVAL,
                                                                     C0.BRATS_EMPTY)
    args.nested_regions, args.hd95_surface, args.hd95_convention = (C0.NESTED_REGIONS, C0.SURFACE_HD95,
                                                                     C0.HD95_CONVENTION)
    print(f"scoring: tuned {C0.TUNE_THRESHOLDS}, full resolution {C0.FULL_RESOLUTION_EVAL}, BraTS empty rule "
          f"{C0.BRATS_EMPTY}, nested {C0.NESTED_REGIONS}, surface HD95 {C0.SURFACE_HD95} ({C0.HD95_CONVENTION}), "
          f"ensemble {C0.ENSEMBLE_N}, tta {C0.USE_TTA}, case seed {C0.CASE_SEED}", flush=True)
    from glo_nca.cli.predict import folder_fingerprint
    from glo_nca.evaluation import evaluation_spec
    # Identity of everything that determines the scores, so a resumed or repeated evaluation can be checked.
    identity = {"runs": [{"run": r, "best_pth_sha256": file_sha256(os.path.join(r, "best.pth")),
                          "config_sha256": file_sha256(os.path.join(r, "config.yaml")),
                          "split_sha256": file_sha256(os.path.join(r, "split.json"))} for r in runs],
                "data_root": os.path.abspath(data_root),
                "data_fingerprint": folder_fingerprint(data_root, (".nii", ".nii.gz")),
                "metric_version": METRIC_VERSION,
                "code": code_identity(), "case_seeds": [a.case_seed for a, _, _ in models]}

    out = args.output or os.path.join(runs[0], "evaluation.json")
    if args.tune_thresholds:
        # Tuning needs every validation probability map at once.
        cases = {}
        for state in ("val", "test"):
            per_model = [collect_probs(agent, d, state, ensemble=n, tta=tta, dtype=np.float32)
                         for (agent, n, tta), d in zip(models, datasets)]
            # Align by case ID, never by position, and require the same cases from every model.
            by_case = [{c: (p, g) for c, p, g in m} for m in per_model]
            ids = sorted(by_case[0])
            if any(sorted(b) != ids for b in by_case[1:]):
                print(f"FAILED: the models returned different {state} cases.")
                return 2
            cases[state] = [(c, np.mean([b[c][0].astype(np.float32) for b in by_case], axis=0),
                             by_case[0][c][1]) for c in ids]
        tuned = final_evaluation(C0, ds, cases["val"], cases["test"])
    else:
        # Fixed post-processing: score case by case with a resumable progress file.
        thresholds = _parse_thresholds(args.thresholds)
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        progress = out + ".progress.jsonl"
        settings = {**identity, "evaluation_spec": evaluation_spec(C0, thresholds, C0.MIN_COMPONENT),
                    "full_resolution": args.full_resolution, "hd95": not args.no_hd95,
                    "brats_empty": args.brats_empty, "hd95_surface": args.hd95_surface,
                    "hd95_convention": args.hd95_convention, "nested_regions": args.nested_regions,
                    "thresholds": thresholds, "min_component": C0.MIN_COMPONENT,
                    "models": [(n, t) for _, n, t in models]}
        if os.path.exists(progress):
            with open(progress, encoding="utf-8") as fh:
                first = json.loads(fh.readline() or "{}")
            if first.get("settings") != json.loads(json.dumps(settings)):
                print(f"FAILED: {progress} was written for other inputs or settings (checkpoint, config, "
                      f"split, data, metric version, code or options); delete it to start over.")
                return 2
        else:
            with open(progress, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"settings": settings}) + "\n")
        res = {state: score_streaming(models, ds, state, progress, thresholds, C0.MIN_COMPONENT,
                                      args.full_resolution, not args.no_hd95, args.brats_empty,
                                      args.hd95_surface and args.hd95_convention, args.nested_regions)
               for state in splits}
        tuned = {"thresholds": thresholds, "min_component_voxels": C0.MIN_COMPONENT,
                 "full_resolution": args.full_resolution, "brats_empty": args.brats_empty,
                 "hd95_surface": args.hd95_surface, "nested_regions": args.nested_regions,
                 "val_tuned": {r: res["val"][r]["dice"] for r in REGIONS} if "val" in res else None,
                 "test": res.get("test")}
    if tuned["test"] is not None:
        report_tuned(tuned)
    mean = float(np.mean([tuned["test"][r]["dice"] for r in REGIONS])) if tuned["test"] else None
    val_mean = float(np.mean([tuned["val_tuned"][r] for r in REGIONS])) if tuned["val_tuned"] else None
    print(f"val mean Dice {val_mean} | test mean Dice {mean} | {len(runs)} model(s)")

    out = args.output or os.path.join(runs[0], "evaluation.json")
    with open(out, "w", encoding="utf-8") as fh:
        # The spec of what was executed (after every override), with the thresholds actually used.
        tuned["evaluation_spec"] = evaluation_spec(C0, tuned.get("thresholds"), tuned.get("min_component_voxels"))
        json.dump({"runs": runs, "identity": identity, **tuned, "val_mean_dice": val_mean,
                   "test_mean_dice": mean}, fh, indent=2, default=str)
    print("Saved to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
