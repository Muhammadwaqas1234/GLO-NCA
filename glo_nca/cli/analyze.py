r"""Analyses of a trained run: inference decomposition, iteration budget and effective receptive field."""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

STAGES = ["single pass", "+ ensemble x4", "+ flips", "+ thresholds", "+ clean-up", "+ ET rule"]


def _load(run, data_root, splits=None):
    r"""Config, dataset, levels and agent of a run with best.pth loaded, in inference mode."""
    import torch

    from glo_nca.config import load_config
    from glo_nca.trainer import build, set_seed
    C = load_config(os.path.join(run, "config.yaml"))
    set_seed(C.SEED)
    ds, ca, agent, dev = build(C, data_root, run, splits or ([], [], []))
    ck = torch.load(os.path.join(run, "best.pth"), map_location=dev, weights_only=False)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)
    return C, ds, ca, agent, dev


def _test_split(run):
    r"""The run's saved (train, validation, test) split."""
    with open(os.path.join(run, "split.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    return s["train"], s["validation"], s["test"]


def _metrics(masks, gt, spacing):
    r"""BraTS Dice (empty prediction of an absent region scores 1) and surface HD95 in mm per region."""
    from glo_nca.config import REGIONS
    from glo_nca.evaluation import _dice
    from glo_nca.metrics import hd95_surface
    out = {}
    for i, r in enumerate(REGIONS):
        t = gt[..., i] > 0
        out[r] = {"dice": float(_dice(masks[r], t, True)),
                  "hd95": float(hd95_surface(masks[r], t, spacing=spacing))}
    return out


def _summary(rows):
    r"""Mean Dice and HD95 (over defined cases, with their count) per region and stage."""
    from glo_nca.config import REGIONS
    out = {}
    for stage in STAGES:
        sr = [r for r in rows if r["stage"] == stage]
        if not sr:
            continue
        out[stage] = {}
        for g in REGIONS:
            hd = [r[g]["hd95"] for r in sr if not np.isnan(r[g]["hd95"])]
            out[stage][g] = {"dice": float(np.mean([r[g]["dice"] for r in sr])),
                             "hd95": float(np.mean(hd)) if hd else float("nan"), "hd95_cases": len(hd)}
        out[stage]["mean_dice"] = float(np.mean([out[stage][g]["dice"] for g in REGIONS]))
    return out


def decompose(args) -> int:
    r"""Score the same seeded passes after each inference stage, on one split at full resolution."""
    import torch

    from glo_nca.config import REGIONS
    from glo_nca.data import find_data_root
    from glo_nca.evaluation import (_parse_regions, decode_regions, predict_passes, scan_spacing,
                                    to_full_resolution)
    run = os.path.abspath(args.run)
    data_root, _ = find_data_root(args.data_root)
    C, ds, _, agent, _ = _load(run, data_root, _test_split(run))
    agent.case_seed = args.case_seed
    agent.exp.set_model_state(args.split)
    th = _parse_regions(args.thresholds, 0.5, float)
    mc = _parse_regions(args.min_component, 0, int)
    half, none = {r: 0.5 for r in REGIONS}, {r: 0 for r in REGIONS}
    et_rule = tuple(float(x) for x in args.et_to_ncr.split(",")) if args.et_to_ncr else None
    keys = list(ds.images_list)[:args.limit]
    rows = []
    with torch.no_grad():
        for n, key in enumerate(keys, 1):
            data = torch.utils.data.default_collate([ds[list(ds.images_list).index(key)]])
            case, passes, _ = predict_passes(agent, data, args.ensemble_n, True)
            spacing = scan_spacing(ds, case)
            probs = {"single pass": passes[0], "+ ensemble x4": np.mean(passes[:args.ensemble_n], axis=0),
                     "+ flips": np.mean(passes, axis=0)}
            full, gt = {}, None
            for k, p in probs.items():
                full[k], gt = to_full_resolution(ds, case, p)
            settings = [("single pass", full["single pass"], half, none, None),
                        ("+ ensemble x4", full["+ ensemble x4"], half, none, None),
                        ("+ flips", full["+ flips"], half, none, None),
                        ("+ thresholds", full["+ flips"], th, none, None),
                        ("+ clean-up", full["+ flips"], th, mc, None)]
            if et_rule:
                settings.append(("+ ET rule", full["+ flips"], th, mc, et_rule))
            for stage, p, t, m, rule in settings:
                rows.append({"case": case, "stage": stage, "seed": agent.last_seed,
                             **_metrics(decode_regions(p, t, m, rule), gt, spacing)})
            if n % 10 == 0 or n == len(keys):
                print(f"  {n}/{len(keys)} cases", flush=True)
    summary = _summary(rows)
    _print_table(summary, "Inference decomposition")
    _save(args.output, {"run": run, "split": args.split, "cases": len(keys), "ensemble_n": args.ensemble_n,
                        "thresholds": th, "min_component": mc, "et_rule": et_rule,
                        "case_seed": args.case_seed, "summary": summary}, rows)
    return 0


def steps(args) -> int:
    r"""Accuracy and per-case latency for several NCA step counts (same number on both levels)."""
    import torch

    from glo_nca.config import REGIONS
    from glo_nca.data import find_data_root
    from glo_nca.evaluation import _parse_regions, decode_regions, predict_case, scan_spacing, to_full_resolution
    run = os.path.abspath(args.run)
    data_root, _ = find_data_root(args.data_root)
    C, ds, _, agent, dev = _load(run, data_root, _test_split(run))
    agent.case_seed = args.case_seed
    agent.exp.set_model_state(args.split)
    th = _parse_regions(args.thresholds, 0.5, float)
    mc = _parse_regions(args.min_component, 0, int)
    keys = list(ds.images_list)[:args.limit]
    results = {}
    for s in [int(x) for x in args.steps.split(",")]:
        agent.exp.config["inference_steps"] = [s, s]
        dice, times = {r: [] for r in REGIONS}, []
        with torch.no_grad():
            for key in keys:
                data = torch.utils.data.default_collate([ds[list(ds.images_list).index(key)]])
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                case, prob, _ = predict_case(agent, data, args.ensemble_n, not args.no_tta)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                full, gt = to_full_resolution(ds, case, prob)
                m = _metrics(decode_regions(full, th, mc), gt, scan_spacing(ds, case))
                for r in REGIONS:
                    dice[r].append(m[r]["dice"])
        results[s] = {**{r: float(np.mean(dice[r])) for r in REGIONS},
                      "mean_dice": float(np.mean([np.mean(dice[r]) for r in REGIONS])),
                      "latency_median_s": float(np.median(times))}
        print(f"steps {s:3d}+{s:<3d} | mean Dice {results[s]['mean_dice']:.4f} "
              f"(WT {results[s]['WT']:.3f} TC {results[s]['TC']:.3f} ET {results[s]['ET']:.3f}) | "
              f"{results[s]['latency_median_s']:.2f} s/case", flush=True)
    agent.exp.config["inference_steps"] = C.STEPS
    _save(args.output, {"run": run, "split": args.split, "cases": len(keys), "trained_steps": C.STEPS,
                        "ensemble_n": args.ensemble_n, "tta": not args.no_tta, "results": results}, None)
    return 0


def erf(args) -> int:
    r"""Effective receptive field: input-gradient magnitude of the output at one tumour voxel."""
    import torch

    from glo_nca.config import REGIONS
    run = os.path.abspath(args.run)
    folder = os.path.abspath(args.case)
    case = os.path.basename(folder)
    C, ds, ca, agent, dev = _load(run, os.path.dirname(folder))
    if args.modalities:
        ds.MODALITIES = [m.strip() for m in args.modalities.split(",")]
    for m in ca:
        m.grad_checkpoint = True          # memory: recompute each step in backward
    agent.exp.set_model_state("test")
    files = [ds._find_modality_file(folder, case, m) for m in ds.MODALITIES]
    raw = np.stack([ds.load_item(f) for f in files], axis=-1)
    bbox = ds._foreground_bbox(raw)
    x0, x1, y0, y1, z0, z1 = bbox
    img = ds.normalize(ds.resize_image(raw[x0:x1, y0:y1, z0:z1]))
    # Working voxel size in mm (scans are 1 mm isotropic; the crop is resized to the working cube).
    vox_mm = np.array([(x1 - x0), (y1 - y0), (z1 - z0)]) / np.array(ds.size)
    region = REGIONS.index(args.region)
    results = {}
    for s in [int(x) for x in args.steps.split(",")]:
        agent.exp.config["inference_steps"] = [s, s]
        grads = []
        for k in range(args.passes):
            torch.manual_seed(k)
            x = torch.from_numpy(img[None].astype(np.float32)).to(dev).requires_grad_(True)
            seed = agent.make_seed(x)
            out, _ = agent.get_outputs((["_" + case + "_0"], seed, torch.zeros((1, *ds.size, 3), device=dev)),
                                       full_img=True, keep_grad=True)
            if k == 0:
                # Target: the voxel of highest predicted probability for the region (a tumour voxel).
                target = np.unravel_index(int(torch.argmax(out[0, ..., region]).item()), tuple(ds.size))
            out[0, target[0], target[1], target[2], region].backward()
            grads.append(x.grad[0].abs().sum(-1).cpu().numpy())
        g = np.mean(grads, axis=0)
        idx = np.indices(g.shape).reshape(3, -1).T
        dist = np.sqrt((((idx - np.array(target)) * vox_mm) ** 2).sum(1))
        order = np.argsort(dist)
        cum = np.cumsum(g.ravel()[order]) / max(g.sum(), 1e-12)
        radius = {f"r{int(q * 100)}_mm": float(dist[order][np.searchsorted(cum, q)]) for q in (0.5, 0.9, 0.95)}
        far = float(g.ravel()[dist > args.far_mm].sum() / max(g.sum(), 1e-12))
        results[s] = {**radius, f"share_beyond_{int(args.far_mm)}mm": far, "target_voxel": [int(v) for v in target]}
        print(f"steps {s:3d}+{s:<3d} | influence radius 50% {radius['r50_mm']:.1f} mm, 90% {radius['r90_mm']:.1f} mm, "
              f"95% {radius['r95_mm']:.1f} mm | beyond {args.far_mm:.0f} mm {far:.1%}", flush=True)
    _save(args.output, {"run": run, "case": case, "region": args.region, "passes": args.passes,
                        "working_voxel_mm": vox_mm.tolist(), "trained_steps": C.STEPS, "results": results}, None)
    return 0


def overlap(args) -> int:
    r"""Check a BraTS ID mapping table for evaluation cases that were in the run's training or validation split."""
    import re
    path = args.mapping
    if path.lower().endswith((".xlsx", ".xls")):
        import pandas as pd
        rows = pd.read_excel(path).astype(str).values.tolist()
    else:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
    with open(os.path.join(args.run, "split.json"), encoding="utf-8") as fh:
        split = json.load(fh)
    where = {c: name for name in ("train", "validation", "test") for c in split[name]}
    # Each row links IDs of one patient across releases; keep the BraTS 2021 ID found in the row.
    ids_2021 = re.compile(r"BraTS2021_\d{5}")
    evaluated = None
    if args.cases:
        with open(args.cases, newline="", encoding="utf-8-sig") as fh:
            evaluated = {row[0] for row in csv.reader(fh) if row}
    hits = {"train": [], "validation": [], "test": []}
    checked = 0
    for row in rows:
        text = " ".join(str(x) for x in row)
        other = [x for x in re.findall(r"BraTS-GLI-\d{5}-\d{3}", text)]
        if not other or (evaluated is not None and not any(o in evaluated for o in other)):
            continue
        checked += 1
        for b in ids_2021.findall(text):
            if b in where:
                hits[where[b]].append((other[0] if other else "", b))
    print(f"{checked} mapped evaluation cases checked against {len(where)} split cases")
    for name, h in hits.items():
        print(f"  in {name:10s}: {len(h)}" + (f"  e.g. {h[:3]}" if h else ""))
    _save(args.output, {"mapping": path, "checked": checked,
                        "overlap": {k: [list(x) for x in v] for k, v in hits.items()}}, None)
    return 1 if hits["train"] or hits["validation"] else 0


def _print_table(summary, title):
    r"""Print a stage table: Dice and HD95 per region and mean Dice."""
    print(f"\n{title}")
    print(f"{'':16}{'WT':>8}{'TC':>8}{'ET':>8}{'mean':>8}{'HD95 WT':>10}{'TC':>7}{'ET':>7}")
    for stage, s in summary.items():
        print(f"{stage:16}{s['WT']['dice']:8.4f}{s['TC']['dice']:8.4f}{s['ET']['dice']:8.4f}{s['mean_dice']:8.4f}"
              f"{s['WT']['hd95']:10.2f}{s['TC']['hd95']:7.2f}{s['ET']['hd95']:7.2f}")


def _save(path, result, rows):
    r"""Write the result JSON and, for per-case rows, a CSV next to it."""
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    if rows:
        with open(os.path.splitext(path)[0] + "_per_case.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["case", "stage", "seed"] + [f"{r}_{k}" for r in ("WT", "TC", "ET") for k in ("dice", "hd95")])
            for row in rows:
                w.writerow([row["case"], row["stage"], row["seed"]]
                           + [row[r][k] for r in ("WT", "TC", "ET") for k in ("dice", "hd95")])
    print("Saved to", path)


def main(argv=None) -> int:
    r"""Parse the sub-command and run it."""
    ap = argparse.ArgumentParser(description="Analyses of a trained GLO-NCA run.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", required=True, help="run folder with config.yaml, split.json and best.pth")
    common.add_argument("--output", help="write the results as JSON")
    common.add_argument("--case-seed", type=int, default=0, help="base seed of the stochastic passes")
    scoring = argparse.ArgumentParser(add_help=False)
    scoring.add_argument("--data-root", required=True)
    scoring.add_argument("--split", default="test", choices=["val", "test"])
    scoring.add_argument("--limit", type=int, default=None, help="only the first N cases")
    scoring.add_argument("--ensemble-n", type=int, default=4)
    scoring.add_argument("--thresholds", default="WT=0.65,TC=0.65,ET=0.6")
    scoring.add_argument("--min-component", default="WT=400,TC=25,ET=400")

    d = sub.add_parser("decompose", parents=[common, scoring],
                       help="score single pass, +ensemble, +flips, +thresholds, +clean-up, +ET rule")
    d.add_argument("--et-to-ncr", default="0.25,5000", help="ET rule RATIO,VOXELS; empty to skip")
    s = sub.add_parser("steps", parents=[common, scoring], help="accuracy and latency per NCA step count")
    s.add_argument("--steps", default="5,10,20,40")
    s.add_argument("--no-tta", action="store_true")
    e = sub.add_parser("erf", parents=[common], help="effective receptive field at one tumour voxel")
    e.add_argument("--case", required=True, help="one case folder with the four modalities")
    e.add_argument("--modalities", default="", help="modality suffixes if they differ from the run's")
    e.add_argument("--region", default="WT", choices=["WT", "TC", "ET"])
    e.add_argument("--steps", default="20")
    e.add_argument("--passes", type=int, default=3, help="stochastic passes averaged")
    e.add_argument("--far-mm", type=float, default=30.0)
    o = sub.add_parser("overlap", help="evaluation cases that appear in the run's training or validation split")
    o.add_argument("--run", required=True, help="run folder with split.json")
    o.add_argument("--mapping", required=True, help="BraTS ID mapping table (.csv or .xlsx) linking releases")
    o.add_argument("--cases", default="", help="optional CSV whose first column lists the evaluated case IDs")
    o.add_argument("--output", help="write the result as JSON")
    args = ap.parse_args(argv)
    return {"decompose": decompose, "steps": steps, "erf": erf, "overlap": overlap}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
