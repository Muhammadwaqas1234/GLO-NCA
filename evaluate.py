#!/usr/bin/env python
r"""Re-evaluate trained runs: tuned thresholds, component clean-up, full resolution, run ensembles.

One --run re-scores that model; several --run folders average their probabilities (an ensemble
of independently trained models). All runs must share the same split.json. Post-processing is
tuned on the validation split only and the test split is scored once.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _parse_min_component(text):
    out = {"WT": 0, "TC": 0, "ET": 0}
    for item in filter(None, (text or "").split(",")):
        region, _, value = item.partition("=")
        if region.strip() not in out:
            raise ValueError(f"unknown region in --min-component: {region}")
        out[region.strip()] = int(value)
    return out


def main() -> int:
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
    args = ap.parse_args()

    import numpy as np
    import torch

    from glo_nca.config import REGIONS, load_config
    from glo_nca.data import find_data_root
    from glo_nca.evaluation import collect_probs, improved_evaluation
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

    sums, ds, C0 = {"val": None, "test": None}, None, None
    for r in runs:
        C = load_config(os.path.join(r, "config.yaml"))
        C0 = C0 or C
        set_seed(C.SEED)
        ds, ca, agent, dev = build(C, data_root, r, splits[0])
        ck = torch.load(os.path.join(r, "best.pth"), map_location=dev, weights_only=False)
        for m, sd in zip(ca, ck["m"]):
            m.load_state_dict(sd)
        n = args.ensemble_n if args.ensemble_n is not None else C.ENSEMBLE_N
        tta = C.USE_TTA and not args.no_tta
        print(f"{os.path.basename(r)}: best epoch {ck['ep']} (val {ck['val_mean']:.3f}), "
              f"ensemble {n}, tta {tta}", flush=True)
        for state in ("val", "test"):
            cases = collect_probs(agent, ds, state, ensemble=n, tta=tta)
            if sums[state] is None:
                sums[state] = [(c, p.astype(np.float32), g) for c, p, g in cases]
            else:
                if [(c, p.shape) for c, p, _ in cases] != [(c, s.shape) for c, s, _ in sums[state]]:
                    print("FAILED: ensemble members differ in cases or working resolution.")
                    return 2
                sums[state] = [(c, s + p.astype(np.float32), g)
                               for (c, s, g), (c2, p, _) in zip(sums[state], cases)]
    cases = {k: [(c, (p / len(runs)).astype(np.float16), g) for c, p, g in v] for k, v in sums.items()}

    C0.TUNE_THRESHOLDS = args.tune_thresholds
    C0.MIN_COMPONENT = _parse_min_component(args.min_component)
    C0.FULL_RESOLUTION_EVAL = args.full_resolution
    tuned = improved_evaluation(C0, ds, cases["val"], cases["test"])
    report_tuned(tuned)
    mean = float(np.mean([tuned["test"][r]["dice"] for r in REGIONS]))
    print(f"test mean Dice {mean:.4f} over {len(cases['test'])} cases, {len(runs)} model(s)")

    out = args.output or os.path.join(runs[0], "evaluation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"runs": runs, **tuned, "test_mean_dice": mean}, fh, indent=2, default=str)
    print("Saved to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
