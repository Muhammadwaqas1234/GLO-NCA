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
    rows, qc = [], []
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
            # Uncertainty on the working grid: spread of the passes and low confidence inside the tumour.
            mean_p = np.mean(passes, axis=0)
            tumour = mean_p[..., 0] >= 0.5
            final = rows[-1]
            qc.append({"case": case, "mean_dice": float(np.mean([final[r]["dice"] for r in REGIONS])),
                       "pass_std": float(np.std(passes, axis=0)[tumour].mean()) if tumour.any() else 0.0,
                       "low_confidence": float(1 - np.maximum(mean_p, 1 - mean_p)[tumour].mean())
                       if tumour.any() else 0.0})
            if n % 10 == 0 or n == len(keys):
                print(f"  {n}/{len(keys)} cases", flush=True)
    summary = _summary(rows)
    _print_table(summary, "Inference decomposition")
    quality = _quality_control(qc, args.failure_dice)
    print(f"Uncertainty QC (failure: mean Dice < {args.failure_dice}, {quality['failures']} of {len(qc)}): "
          f"pass spread AUROC {quality['pass_std']['auroc_failure']:.3f}, "
          f"low confidence AUROC {quality['low_confidence']['auroc_failure']:.3f}")
    _save(args.output, {"run": run, "split": args.split, "cases": len(keys), "ensemble_n": args.ensemble_n,
                        "thresholds": th, "min_component": mc, "et_rule": et_rule,
                        "case_seed": args.case_seed, "summary": summary, "quality_control": quality,
                        "per_case_uncertainty": qc}, rows)
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


def _auroc(score, positive):
    r"""Area under the ROC curve of a score for detecting positives (Mann-Whitney, ties counted half)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def _quality_control(qc, failure_dice):
    r"""How well per-case uncertainty flags poor cases: AUROC, Spearman rho and a risk-coverage table."""
    from scipy.stats import spearmanr
    dice = np.array([q["mean_dice"] for q in qc])
    fail = dice < failure_dice
    out = {"failure_dice": failure_dice, "failures": int(fail.sum()), "cases": len(qc)}
    for k in ("pass_std", "low_confidence"):
        u = np.array([q[k] for q in qc])
        rho = spearmanr(u, dice).correlation if len(qc) > 2 else float("nan")
        # Risk-coverage: keep the most certain cases first; risk is 1 - mean Dice of the kept cases.
        order = np.argsort(u)
        cover = {f"{c}%": float(1 - dice[order[:max(1, int(round(len(qc) * c / 100)))]].mean())
                 for c in (50, 80, 100)}
        out[k] = {"auroc_failure": _auroc(u, fail), "spearman_vs_dice": float(rho), "risk_at_coverage": cover}
    return out


def manifest(args) -> int:
    r"""One row per case: split, labels present, region volumes, ET presence, spacing and shape."""
    import nibabel as nib

    from glo_nca.data import Dataset_BraTS_Foreground, find_data_root
    data_root, _ = find_data_root(args.data_root)
    split = {}
    if args.run:
        for name, ids in zip(("train", "validation", "test"), _test_split(os.path.abspath(args.run))):
            split.update({c: name for c in ids})
    ds = Dataset_BraTS_Foreground.__new__(Dataset_BraTS_Foreground)
    ds.ET_LABELS = tuple(int(x) for x in args.et_labels.split(","))
    cases = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))
    rows = []
    for n, case in enumerate(cases, 1):
        folder = os.path.join(data_root, case)
        try:
            img = nib.load(ds._find_modality_file(folder, case, "seg"))
        except FileNotFoundError:
            rows.append({"case": case, "split": split.get(case, ""), "excluded": "no segmentation"})
            continue
        seg = np.asanyarray(img.dataobj).astype(np.int16)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        ml = float(np.prod(zooms)) / 1000.0
        regions = ds._labels_to_regions(seg) > 0
        rows.append({"case": case, "split": split.get(case, ""), "excluded": "",
                     "labels": " ".join(str(v) for v in np.unique(seg)), "shape": "x".join(map(str, seg.shape)),
                     "spacing_mm": "x".join(f"{z:g}" for z in zooms),
                     "wt_ml": round(regions[..., 0].sum() * ml, 2), "tc_ml": round(regions[..., 1].sum() * ml, 2),
                     "et_ml": round(regions[..., 2].sum() * ml, 2), "et_present": bool(regions[..., 2].any())})
        if n % 100 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)} cases", flush=True)
    summary = {}
    for name in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == name and not r["excluded"]]
        if rs:
            summary[name or "unsplit"] = {
                "cases": len(rs), "et_absent": sum(not r["et_present"] for r in rs),
                "median_wt_ml": float(np.median([r["wt_ml"] for r in rs])),
                "median_et_ml": float(np.median([r["et_ml"] for r in rs])),
                "small_et_below_1ml": sum(0 < r["et_ml"] < 1 for r in rs)}
    summary["excluded"] = sum(bool(r["excluded"]) for r in rows)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if args.output:
        keys = ["case", "split", "excluded", "labels", "shape", "spacing_mm", "wt_ml", "tc_ml", "et_ml", "et_present"]
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        with open(os.path.splitext(args.output)[0] + "_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)
        print("Saved to", args.output)
    return 0


def modalities(args) -> int:
    r"""Leave-one-modality-out at inference: each modality in turn is set to zero (missing)."""
    import torch

    from glo_nca.config import REGIONS
    from glo_nca.data import find_data_root
    from glo_nca.evaluation import _parse_regions, decode_regions, predict_case, scan_spacing, to_full_resolution
    run = os.path.abspath(args.run)
    data_root, _ = find_data_root(args.data_root)
    C, ds, _, agent, _ = _load(run, data_root, _test_split(run))
    agent.case_seed = args.case_seed
    agent.exp.set_model_state(args.split)
    th = _parse_regions(args.thresholds, 0.5, float)
    mc = _parse_regions(args.min_component, 0, int)
    keys = list(ds.images_list)[:args.limit]
    variants = ["all"] + [f"without {m}" for m in C.MODALITIES]
    dice = {v: {r: [] for r in REGIONS} for v in variants}
    with torch.no_grad():
        for n, key in enumerate(keys, 1):
            sample = ds[list(ds.images_list).index(key)]
            for j, v in enumerate(variants):
                img = sample[1].copy()
                if j:
                    img[..., j - 1] = 0.0          # this modality missing
                data = torch.utils.data.default_collate([(sample[0], img, sample[2])])
                case, prob, _ = predict_case(agent, data, args.ensemble_n, not args.no_tta)
                full, gt = to_full_resolution(ds, case, prob)
                m = _metrics(decode_regions(full, th, mc), gt, scan_spacing(ds, case))
                for r in REGIONS:
                    dice[v][r].append(m[r]["dice"])
            if n % 10 == 0 or n == len(keys):
                print(f"  {n}/{len(keys)} cases", flush=True)
    results = {}
    for v in variants:
        results[v] = {r: float(np.mean(dice[v][r])) for r in REGIONS}
        results[v]["mean_dice"] = float(np.mean([results[v][r] for r in REGIONS]))
        drop = results["all"]["mean_dice"] - results[v]["mean_dice"]
        print(f"{v:20s} | mean Dice {results[v]['mean_dice']:.4f} (WT {results[v]['WT']:.3f} TC {results[v]['TC']:.3f} "
              f"ET {results[v]['ET']:.3f}) | drop {drop:+.4f}", flush=True)
    _save(args.output, {"run": run, "split": args.split, "cases": len(keys), "ensemble_n": args.ensemble_n,
                        "tta": not args.no_tta, "results": results}, None)
    return 0


def perturb(args) -> int:
    r"""Remote perturbation: change of the tumour prediction when a noise cube is placed at a distance."""
    import torch

    run = os.path.abspath(args.run)
    folder = os.path.abspath(args.case)
    case = os.path.basename(folder)
    C, ds, _, agent, dev = _load(run, os.path.dirname(folder))
    if args.modalities:
        ds.MODALITIES = [m.strip() for m in args.modalities.split(",")]
    agent.exp.set_model_state("test")
    raw = np.stack([ds.load_item(ds._find_modality_file(folder, case, m)) for m in ds.MODALITIES], axis=-1)
    x0, x1, y0, y1, z0, z1 = ds._foreground_bbox(raw)
    img = ds.normalize(ds.resize_image(raw[x0:x1, y0:y1, z0:z1]))
    vox_mm = np.array([(x1 - x0), (y1 - y0), (z1 - z0)]) / np.array(ds.size)
    brain = np.any(img != 0, axis=-1)

    def run_once(volume, seed):
        torch.manual_seed(seed)
        x = torch.from_numpy(volume[None].astype(np.float32)).to(dev)
        with torch.no_grad():
            out, _ = agent.get_outputs((["_" + case + "_0"], agent.make_seed(x),
                                        torch.zeros((1, *ds.size, 3), device=dev)), full_img=True)
        return torch.sigmoid(out[0, ..., 0]).cpu().numpy()      # whole-tumour probability

    base = [run_once(img, k) for k in range(args.passes)]
    tumour = np.mean(base, axis=0) >= 0.5
    if not tumour.any():
        print("FAILED: no tumour predicted in this case.")
        return 2
    centre = np.argwhere(tumour).mean(0)
    idx = np.argwhere(brain & ~tumour)
    dist = np.sqrt((((idx - centre) * vox_mm) ** 2).sum(1))
    rng = np.random.default_rng(0)
    half = args.cube // 2
    results = {}
    for d in [float(x) for x in args.distances.split(",")]:
        cand = idx[np.abs(dist - d) < 3.0]
        if len(cand) == 0:
            continue
        effects = []
        for c in cand[rng.choice(len(cand), size=min(args.positions, len(cand)), replace=False)]:
            vol = img.copy()
            sl = tuple(slice(max(int(c[a]) - half, 0), int(c[a]) + half) for a in range(3))
            vol[sl] += rng.normal(0, args.noise, size=vol[sl].shape).astype(np.float32)
            pert = [run_once(vol, k) for k in range(args.passes)]
            effects.append(float(np.mean([np.abs(p - b)[tumour].mean() for p, b in zip(pert, base)])))
        results[d] = {"mean_abs_change": float(np.mean(effects)), "positions": len(effects)}
        print(f"noise cube at {d:5.0f} mm from the tumour centre | mean |change| of tumour probability "
              f"{results[d]['mean_abs_change']:.2e} ({len(effects)} positions)", flush=True)
    _save(args.output, {"run": run, "case": case, "cube_voxels": args.cube, "noise_sd": args.noise,
                        "passes": args.passes, "working_voxel_mm": vox_mm.tolist(), "results": results}, None)
    return 0


def _json_default(o):
    r"""JSON for NumPy scalars and arrays."""
    return o.item() if hasattr(o, "item") else o.tolist() if hasattr(o, "tolist") else str(o)


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
        json.dump(result, fh, indent=2, default=_json_default)
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
    d.add_argument("--failure-dice", type=float, default=0.8, help="a case with mean Dice below this is a failure")
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
    md = sub.add_parser("modalities", parents=[common, scoring], help="leave-one-modality-out at inference")
    md.add_argument("--no-tta", action="store_true")
    mf = sub.add_parser("manifest", help="per-case table: split, labels, region volumes, ET presence")
    mf.add_argument("--data-root", required=True)
    mf.add_argument("--run", default="", help="run folder whose split.json assigns the splits")
    mf.add_argument("--et-labels", default="3,4")
    mf.add_argument("--output", help="CSV path; a _summary.json is written next to it")
    pt = sub.add_parser("perturb", parents=[common], help="remote perturbation: effect of distant noise")
    pt.add_argument("--case", required=True, help="one case folder with the four modalities")
    pt.add_argument("--modalities", default="")
    pt.add_argument("--distances", default="20,40,60,80", help="mm from the predicted tumour centre")
    pt.add_argument("--cube", type=int, default=8, help="noise cube edge in working voxels")
    pt.add_argument("--noise", type=float, default=1.0, help="noise standard deviation (z-score units)")
    pt.add_argument("--positions", type=int, default=4, help="cube positions per distance")
    pt.add_argument("--passes", type=int, default=2)
    o = sub.add_parser("overlap", help="evaluation cases that appear in the run's training or validation split")
    o.add_argument("--run", required=True, help="run folder with split.json")
    o.add_argument("--mapping", required=True, help="BraTS ID mapping table (.csv or .xlsx) linking releases")
    o.add_argument("--cases", default="", help="optional CSV whose first column lists the evaluated case IDs")
    o.add_argument("--output", help="write the result as JSON")
    args = ap.parse_args(argv)
    return {"decompose": decompose, "steps": steps, "erf": erf, "overlap": overlap, "modalities": modalities,
            "manifest": manifest, "perturb": perturb}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
