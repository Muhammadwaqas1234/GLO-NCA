#!/usr/bin/env python
r"""Re-evaluate one run or an ensemble of runs; resumable, with optional tuned post-processing."""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _parse_min_component(text):
    r"""Parse 'WT=0,TC=50,ET=50' into a per-region minimum component size."""
    out = {"WT": 0, "TC": 0, "ET": 0}
    for item in filter(None, (text or "").split(",")):
        region, _, value = item.partition("=")
        if region.strip() not in out:
            raise ValueError(f"unknown region in --min-component: {region}")
        out[region.strip()] = int(value)
    return out


def main() -> int:
    r"""Re-evaluate one run or an ensemble of runs and write the result JSON."""
    ap = argparse.ArgumentParser(description="Re-evaluate one run or an ensemble of runs.")
    ap.add_argument("--run", action="append", required=True, metavar="RUN_DIR",
                    help="run folder with config.yaml, split.json and best.pth (repeatable)")
    ap.add_argument("--data-root", required=True, help="folder that holds the BraTS case folders")
    ap.add_argument("--output", default=None, help="result JSON (default: <first run>/evaluation.json)")
    ap.add_argument("--tune-thresholds", action="store_true",
                    help="choose per-region threshold and clean-up size on validation")
    ap.add_argument("--min-component", default="", metavar="WT=0,TC=50,ET=50",
                    help="fixed clean-up: drop predicted components below this size (ignored when tuning)")
    ap.add_argument("--full-resolution", action="store_true",
                    help="score in the original scan space instead of the working resolution")
    ap.add_argument("--ensemble-n", type=int, default=None,
                    help="stochastic passes per model (default: each run's evaluation.ensemble_n)")
    ap.add_argument("--no-tta", action="store_true", help="disable flip test-time augmentation")
    ap.add_argument("--no-hd95", action="store_true",
                    help="skip HD95 (the slowest metric at full resolution); Dice and IoU only")
    args = ap.parse_args()

    import numpy as np
    import torch

    from glo_nca.config import REGIONS, load_config
    from glo_nca.data import find_data_root
    from glo_nca.evaluation import collect_probs, improved_evaluation, score_streaming
    from glo_nca.reporting import report_tuned
    from glo_nca.trainer import build, set_seed

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

    models, ds, C0 = [], None, None
    for r in runs:
        C = load_config(os.path.join(r, "config.yaml"))
        if C0 is not None and C.INPUT_SIZE != C0.INPUT_SIZE:
            print("FAILED: ensemble members differ in working resolution.")
            return 2
        C0 = C0 or C
        set_seed(C.SEED)
        ds_r, ca, agent, dev = build(C, data_root, r, splits[0])
        ds = ds or ds_r
        ck = torch.load(os.path.join(r, "best.pth"), map_location=dev, weights_only=False)
        for m, sd in zip(ca, ck["m"]):
            m.load_state_dict(sd)
        n = args.ensemble_n if args.ensemble_n is not None else C.ENSEMBLE_N
        tta = C.USE_TTA and not args.no_tta
        print(f"{os.path.basename(r)}: best epoch {ck['ep']} (val {ck['val_mean']:.3f}), "
              f"ensemble {n}, tta {tta}", flush=True)
        models.append((agent, n, tta))

    C0.TUNE_THRESHOLDS = args.tune_thresholds
    C0.MIN_COMPONENT = _parse_min_component(args.min_component)
    C0.FULL_RESOLUTION_EVAL = args.full_resolution
    out = args.output or os.path.join(runs[0], "evaluation.json")
    if args.tune_thresholds:
        # Tuning needs every validation probability map at once.
        cases = {}
        for state in ("val", "test"):
            per_model = [collect_probs(agent, ds, state, ensemble=n, tta=tta) for agent, n, tta in models]
            cases[state] = [(c, np.mean([m[i][1].astype(np.float32) for m in per_model], axis=0)
                             .astype(np.float16), g) for i, (c, _, g) in enumerate(per_model[0])]
        tuned = improved_evaluation(C0, ds, cases["val"], cases["test"])
    else:
        # Fixed post-processing: score case by case with a resumable progress file.
        thresholds = {r: 0.5 for r in REGIONS}
        progress = out + ".progress.jsonl"
        settings = {"runs": runs, "full_resolution": args.full_resolution, "hd95": not args.no_hd95,
                    "min_component": C0.MIN_COMPONENT, "models": [(n, t) for _, n, t in models]}
        if os.path.exists(progress):
            with open(progress, encoding="utf-8") as fh:
                first = json.loads(fh.readline() or "{}")
            if first.get("settings") != json.loads(json.dumps(settings)):
                print(f"FAILED: {progress} was written with other settings; delete it to start over.")
                return 2
        else:
            with open(progress, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"settings": settings}) + "\n")
        res = {state: score_streaming(models, ds, state, progress, thresholds, C0.MIN_COMPONENT,
                                      args.full_resolution, not args.no_hd95)
               for state in ("val", "test")}
        tuned = {"thresholds": thresholds, "min_component_voxels": C0.MIN_COMPONENT,
                 "full_resolution": args.full_resolution,
                 "val_tuned": {r: res["val"][r]["dice"] for r in REGIONS}, "test": res["test"]}
    report_tuned(tuned)
    mean = float(np.mean([tuned["test"][r]["dice"] for r in REGIONS]))
    val_mean = float(np.mean([tuned["val_tuned"][r] for r in REGIONS]))
    print(f"val mean Dice {val_mean:.4f} | test mean Dice {mean:.4f} | {len(runs)} model(s)")

    out = args.output or os.path.join(runs[0], "evaluation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"runs": runs, **tuned, "val_mean_dice": val_mean, "test_mean_dice": mean}, fh, indent=2, default=str)
    print("Saved to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
