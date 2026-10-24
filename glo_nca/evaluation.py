r"""Evaluation: Dice, mIoU and HD95 per region, plain and with tuned post-processing."""
from __future__ import annotations

import json
import math
import os
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import hd95_voxel_legacy, hd95_surface, iou_score

from .config import REGIONS


def evaluate(agent, dataset, state, ensemble=1, tta=False, workers=0, per_case=None, surface_hd95=False,
             nested=False):
    r"""Evaluate one split on the working grid at threshold 0.5; ensemble > 1 averages passes, tta adds flips.

    nested scores the final nested masks (TC includes ET, WT includes TC); per_case collects case Dice.
    """
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers)
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    with torch.no_grad():
        for data in loader:
            case, prob, gt = predict_case(agent, data, ensemble, tta)
            if nested:
                half = {r: 0.5 for r in REGIONS}
                masks = decode_regions(prob, half, {r: 0 for r in REGIONS})
                prob = np.stack([masks[r] for r in REGIONS], axis=-1).astype(np.float32)
            # Keep the batch axis: HD95 is computed on (1, X, Y, Z) arrays as in the reference.
            prob, gt = prob[None], gt[None].astype(np.float32)
            for i, r in enumerate(REGIONS):
                p, t = prob[..., i], gt[..., i]
                inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
                acc[r]["dice"].append((2 * inter) / ((p >= 0.5).sum() + (t >= 0.5).sum() + 1e-6))
                acc[r]["iou"].append(iou_score(p, t))
                acc[r]["hd95"].append(hd95_surface(p[0] >= 0.5, t[0] >= 0.5, convention=_conv(surface_hd95))
                                      if surface_hd95 else hd95_voxel_legacy(p, t))
            if per_case is not None:
                per_case.append({"case": case, **{r: float(acc[r]["dice"][-1]) for r in REGIONS}})
    agent.exp.set_model_state("train")
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])), "iou": float(np.mean(acc[r]["iou"])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out


# Threshold tuning, small-component removal and full-resolution scoring.

THRESHOLD_GRID = [round(0.30 + 0.05 * i, 2) for i in range(9)]
MIN_COMPONENT_GRID = [0, 10, 25, 50, 100, 200, 400]   # voxels in the space being scored


def case_id(img_id):
    r"""Case folder name from a dataset id of the form ``_<case>_0``."""
    return img_id[1:-2]




def _parse_regions(text, default, cast):
    r"""Parse 'WT=a,TC=b,ET=c' into a per-region dictionary."""
    out = {"WT": default, "TC": default, "ET": default}
    for item in filter(None, (text or "").split(",")):
        region, _, value = item.partition("=")
        if region.strip() not in out:
            raise ValueError(f"unknown region: {region}")
        out[region.strip()] = cast(value)
    return out


def case_seed(base, case):
    r"""Seed of one case's stochastic passes: the base seed plus a stable hash of the case name."""
    return (int(base) + zlib.crc32(case.encode())) % 2**31


def predict_case(agent, data, ensemble=1, tta=False):
    r"""Mean probability map for one collated case: (case, prob float32, gt bool).

    With agent.case_seed set, pass k of a case is seeded with case_seed(base, case) + k, so stochastic
    predictions are reproducible case by case; the seed is kept in agent.last_seed.
    """
    case, probs, gt = predict_passes(agent, data, ensemble, tta)
    return case, np.mean(probs, axis=0), gt


def predict_passes(agent, data, ensemble=1, tta=False):
    r"""Every stochastic pass of one case, flips outermost: (case, [prob per pass], gt bool).

    With tta, the first ``ensemble`` passes are the un-flipped ones, so their mean is the ensemble
    without flips and the first pass alone is a single pass.
    """
    data = agent.prepare_data(data, eval=True)
    idx, inp0, tgt = data
    flips = [None, (1,), (2,), (3,)] if tta else [None]
    seed = case_seed(agent.case_seed, case_id(idx[0])) if getattr(agent, "case_seed", None) is not None else None
    agent.last_seed = seed
    if seed is None:
        return _passes(agent, idx, inp0, tgt, flips, ensemble, None)
    # Seeded passes must not change the training random sequence: fork the CPU and CUDA generators.
    devices = [inp0.device] if inp0.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        return _passes(agent, idx, inp0, tgt, flips, ensemble, seed)


def _passes(agent, idx, inp0, tgt, flips, ensemble, seed):
    r"""Run every flip and stochastic pass of one prepared case; pass k is seeded with seed + k."""
    probs = []
    k = 0
    for fl in flips:
        inp = torch.flip(inp0, dims=fl) if fl is not None else inp0
        for _ in range(ensemble):
            if seed is not None:
                torch.manual_seed(seed + k)
            k += 1
            out, targets = agent.get_outputs((idx, inp, tgt), full_img=True)
            p = torch.sigmoid(out)
            if fl is not None:
                p = torch.flip(p, dims=fl)
            probs.append(p.detach().cpu().numpy()[0])
    gt = targets.detach().cpu().numpy()[0] >= 0.5
    return case_id(idx[0]), probs, gt


def collect_probs(agent, dataset, state, ensemble=1, tta=False, dtype=np.float16):
    r"""Mean probability map per case as a list of (case, prob, gt uint8); float16 saves memory."""
    if dataset is not agent.exp.dataset:
        raise ValueError("collect_probs needs the agent's own dataset, whose split state it sets")
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    cases = []
    with torch.no_grad():
        for data in loader:
            case, prob, gt = predict_case(agent, data, ensemble, tta)
            cases.append((case, prob.astype(dtype), gt.astype(np.uint8)))
    agent.exp.set_model_state("train")
    return cases


def decode_regions(prob, thresholds, min_component, et_rule=None, nested=True):
    r"""Final masks: threshold, clean up, optional ET relabel, then nest ET in TC and TC in WT."""
    masks = {r: remove_small_components(prob[..., i] >= thresholds[r], min_component[r])
             for i, r in enumerate(REGIONS)}
    if et_rule:
        masks = relabel_small_et(masks, et_rule[0], et_rule[1])
    if nested:
        # The BraTS label map stores nested regions: TC includes ET, WT includes TC.
        masks["TC"] = masks["TC"] | masks["ET"]
        masks["WT"] = masks["WT"] | masks["TC"]
    return masks


def masks_to_label(masks, et_label=3):
    r"""BraTS label map from nested masks: 2 edema, 1 necrotic core, et_label enhancing tumour."""
    label = np.zeros(masks["WT"].shape, np.uint8)
    label[masks["WT"]] = 2
    label[masks["TC"]] = 1
    label[masks["ET"]] = et_label
    return label


def _case_metrics(prob, gt, thresholds, min_component, with_hd95=True, empty_one=False,
                  surface_hd95=False, nested=False, spacing=None):
    r"""Per-region Dice, IoU and (optionally) HD95 for one case; surface_hd95 uses the BraTS definition."""
    out = {}
    masks = decode_regions(prob, thresholds, min_component, nested=nested)
    for i, r in enumerate(REGIONS):
        p = masks[r]
        t = gt[..., i] > 0
        pf, tf = p.astype(np.float32), t.astype(np.float32)
        out[r] = {"dice": float(_dice(p, t, empty_one)), "iou": float(iou_score(pf, tf)),
                  "hd95": float("nan") if not with_hd95
                  else hd95_surface(p, t, spacing=spacing or (1.0, 1.0, 1.0), convention=_conv(surface_hd95))
                  if surface_hd95
                  else float(hd95_voxel_legacy(pf, tf))}
    return out


def _mean_metrics(rows):
    r"""Average per-case metrics per region; HD95 over cases where it is defined (count kept)."""
    out = {}
    for r in REGIONS:
        hd = [x[r]["hd95"] for x in rows if not math.isnan(x[r]["hd95"])]
        out[r] = {"dice": float(np.mean([x[r]["dice"] for x in rows])),
                  "iou": float(np.mean([x[r]["iou"] for x in rows])),
                  "hd95": float(np.mean(hd)) if hd else float("nan"),
                  "hd95_median": float(np.median(hd)) if hd else float("nan"),
                  "hd95_cases": len(hd), "cases": len(rows)}
    return out


def score_streaming(models, dataset, state, progress_path, thresholds, min_component,
                    full_resolution=False, with_hd95=True, empty_one=False, surface_hd95=False,
                    nested=False):
    r"""Score one split case by case into a resumable progress file, averaging ``models``."""
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue   # a line cut short by an interruption
                if row.get("split") == state:
                    done[row["case"]] = row["metrics"]
    for agent, _, _ in models:
        agent.exp.set_model_state(state)
    keys = list(dataset.images_list)
    todo = [(i, k) for i, k in enumerate(keys) if k[0] not in done]
    print(f"{state}: {len(keys)} cases, {len(keys) - len(todo)} already scored", flush=True)
    with torch.no_grad(), open(progress_path, "a", encoding="utf-8") as fh:
        for n, (i, key) in enumerate(todo, 1):
            data = torch.utils.data.default_collate([dataset[i]])
            preds = [predict_case(agent, data, ens, tta) for agent, ens, tta in models]
            case, gt = preds[0][0], preds[0][2]
            prob = np.mean([p for _, p, _ in preds], axis=0)
            spacing = None
            if full_resolution:
                prob, gt = to_full_resolution(dataset, case, prob)
                spacing = scan_spacing(dataset, case)
            metrics = _case_metrics(prob, gt, thresholds, min_component, with_hd95, empty_one,
                                    surface_hd95, nested, spacing)
            row = {"split": state, "case": case, "metrics": metrics}
            if getattr(models[0][0], "last_seed", None) is not None:
                row["seed"] = models[0][0].last_seed
            fh.write(json.dumps(row) + "\n")
            fh.flush(); os.fsync(fh.fileno())
            done[case] = metrics
            if n % 20 == 0 or n == len(todo):
                print(f"  {state}: {len(keys) - len(todo) + n}/{len(keys)}", flush=True)
    for agent, _, _ in models:
        agent.exp.set_model_state("train")
    return _mean_metrics([done[k[0]] for k in keys])


def remove_small_components(mask, min_voxels):
    r"""Drop connected components (26-connectivity) smaller than ``min_voxels``."""
    if min_voxels <= 0 or not mask.any():
        return mask
    lab, sizes = _components(mask)
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[lab]


def relabel_small_et(masks, max_ratio, max_voxels):
    r"""Move a small predicted ET (absolute and relative to TC) into the necrotic core."""
    et = int(masks["ET"].sum())
    tc = int((masks["TC"] | masks["ET"]).sum())
    if et and et < max_voxels and et < max_ratio * tc:
        masks = dict(masks)
        masks["TC"] = masks["TC"] | masks["ET"]
        masks["ET"] = np.zeros_like(masks["ET"])
    return masks


def to_full_resolution(dataset, case, prob):
    r"""Resample a working-resolution prob map into the original scan; returns (prob, gt)."""
    folder = os.path.join(dataset.images_path, case)
    raw = np.stack([dataset.load_item(dataset._find_modality_file(folder, case, m))
                    for m in dataset.MODALITIES], axis=-1)
    seg = dataset.load_item(dataset._find_modality_file(folder, case, dataset.SEG_SUFFIX))
    gt = dataset._labels_to_regions(seg).astype(np.uint8)
    bbox = dataset._foreground_bbox(raw) if dataset.use_foreground_crop else None
    return paste_to_scan(prob, bbox, seg.shape), gt


def paste_to_scan(prob, bbox, shape):
    r"""Resample a working-resolution prob map into its crop box of a scan of the given shape."""
    if bbox is None:
        bbox = (0, shape[0], 0, shape[1], 0, shape[2])
    x0, x1, y0, y1, z0, z1 = bbox
    t = torch.from_numpy(prob.astype(np.float32)).permute(3, 0, 1, 2)[None]
    up = F.interpolate(t, size=(x1 - x0, y1 - y0, z1 - z0), mode="trilinear", align_corners=False)
    full = np.zeros(tuple(shape) + (prob.shape[-1],), np.float32)
    full[x0:x1, y0:y1, z0:z1] = up[0].permute(1, 2, 3, 0).numpy()
    return full


def scan_spacing(dataset, case):
    r"""Voxel spacing in mm of a case's original scan, from its segmentation header."""
    import nibabel as nib
    folder = os.path.join(dataset.images_path, case)
    path = dataset._find_modality_file(folder, case, dataset.SEG_SUFFIX)
    return tuple(float(z) for z in nib.load(path).header.get_zooms()[:3])


def _conv(surface_hd95):
    r"""HD95 convention from a surface_hd95 setting (True means the medpy convention)."""
    return surface_hd95 if isinstance(surface_hd95, str) else "medpy"


def _pairs(cases, dataset=None, full_resolution=False):
    r"""Yield (case, prob, gt, spacing); at full resolution, in the original scan and its spacing in mm."""
    for case, prob, gt in cases:
        if full_resolution:
            yield (case, *to_full_resolution(dataset, case, prob), scan_spacing(dataset, case))
        else:
            yield case, prob.astype(np.float32), gt, None


def _dice(p, t, empty_one=False):
    r"""Dice of two boolean masks; empty_one scores an empty prediction of an empty region as 1 (BraTS)."""
    if empty_one and not p.any() and not t.any():
        return 1.0
    # Otherwise the same formula as evaluate(), so results stay comparable with the plain metric.
    return (2 * np.logical_and(p, t).sum()) / (p.sum() + t.sum() + 1e-6)


def _components(mask):
    r"""Connected-component labels (26-connectivity) and the size of each label."""
    from scipy import ndimage
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    return lab, np.bincount(lab.ravel())


def _tune_regions(cases, dataset, full_resolution, grid, size_grid, empty_one, regions, inner):
    r"""Dice table per candidate for the given regions; inner adds the chosen masks of nested regions."""
    table = {r: {(th, sz): [] for th in grid for sz in size_grid} for r in regions}
    for _, prob, gt, _ in _pairs(cases, dataset, full_resolution):
        for r in regions:
            i = REGIONS.index(r)
            t = gt[..., i] > 0
            # Nested scoring: the region's final mask also contains its chosen inner regions.
            extra = np.zeros(t.shape, bool)
            for q, (th_q, sz_q) in inner.get(r, {}).items():
                extra |= remove_small_components(prob[..., REGIONS.index(q)] >= th_q, sz_q)
            for th in grid:
                mask = prob[..., i] >= th
                lab, sizes = _components(mask) if mask.any() else (None, None)
                for sz in size_grid:
                    if lab is None or sz <= 0:
                        p = mask
                    else:
                        keep = sizes >= sz; keep[0] = False
                        p = keep[lab]
                    table[r][(th, sz)].append(_dice(p | extra, t, empty_one))
    return table


def tune_thresholds(cases, dataset=None, full_resolution=False, grid=THRESHOLD_GRID,
                    size_grid=MIN_COMPONENT_GRID, empty_one=False, nested=False):
    r"""Pick per-region threshold and clean-up size that maximise validation Dice.

    With nested, regions are tuned inside out (ET, then TC with ET, then WT with TC and ET), so the
    choice is scored on the same final masks as decode_regions.
    """
    pick = lambda tab: max(tab, key=lambda k: (np.mean(tab[k]), -k[1]))
    if not nested:
        table = _tune_regions(cases, dataset, full_resolution, grid, size_grid, empty_one, REGIONS, {})
    else:
        table, chosen = {}, {}
        for r, inner in (("ET", []), ("TC", ["ET"]), ("WT", ["TC", "ET"])):
            table.update(_tune_regions(cases, dataset, full_resolution, grid, size_grid, empty_one,
                                       [r], {r: {q: chosen[q] for q in inner}}))
            chosen[r] = pick(table[r])
    best = {r: pick(table[r]) for r in REGIONS}
    thresholds = {r: best[r][0] for r in REGIONS}
    sizes = {r: int(best[r][1]) for r in REGIONS}
    val = {r: float(np.mean(table[r][best[r]])) for r in REGIONS}
    return thresholds, sizes, val


def score(cases, thresholds, min_component, dataset=None, full_resolution=False, empty_one=False,
          nested=False, surface_hd95=False, per_case=None):
    r"""Dice, mIoU and HD95 per region with per-region thresholds and component clean-up.

    surface_hd95 is False (legacy voxel HD95), True (surface, medpy convention) or a convention name;
    per_case, when given, receives one row per case with the same values.
    """
    acc = {r: {"dice": [], "iou": [], "hd95": [], "status": []} for r in REGIONS}
    for case, prob, gt, spacing in _pairs(cases, dataset, full_resolution):
        masks = decode_regions(prob, thresholds, min_component, nested=nested)
        for i, r in enumerate(REGIONS):
            p = masks[r]
            t = gt[..., i] > 0
            pf, tf = p.astype(np.float32), t.astype(np.float32)
            acc[r]["dice"].append(_dice(p, t, empty_one))
            acc[r]["iou"].append(iou_score(pf, tf))
            acc[r]["hd95"].append(hd95_surface(p, t, spacing=spacing or (1.0, 1.0, 1.0),
                                               convention=_conv(surface_hd95))
                                  if surface_hd95 else hd95_voxel_legacy(pf, tf))
            acc[r]["status"].append(hd95_status(p, t))
        if per_case is not None:
            per_case.append({"case": case, **{r: float(acc[r]["dice"][-1]) for r in REGIONS},
                             **{f"{r}_hd95": float(acc[r]["hd95"][-1]) for r in REGIONS},
                             **{f"{r}_hd95_status": acc[r]["status"][-1] for r in REGIONS}})
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        status = acc[r]["status"]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])), "iou": float(np.mean(acc[r]["iou"])),
                  # Conditional on the cases where HD95 is defined; the failures are counted separately.
                  "hd95": float(np.mean(hd)) if hd else float("nan"),
                  "hd95_median": float(np.median(hd)) if hd else float("nan"),
                  "cases": len(status), "hd95_cases": len(hd),
                  "hd95_missed": status.count("missed"), "hd95_false_positive": status.count("false_positive"),
                  "both_empty": status.count("both_empty")}
    return out


def hd95_status(pred, target):
    r"""Why HD95 is or is not defined: 'defined', 'both_empty', 'missed' or 'false_positive'."""
    p, t = bool(np.any(pred)), bool(np.any(target))
    if p and t:
        return "defined"
    if not p and not t:
        return "both_empty"
    return "missed" if t else "false_positive"


def final_evaluation(cfg, dataset, val_cases, test_cases):
    r"""Tune post-processing on validation (if enabled), then score the test cases once."""
    thresholds, min_component = {r: 0.5 for r in REGIONS}, dict(cfg.MIN_COMPONENT)
    val_tuned = None
    if cfg.TUNE_THRESHOLDS:
        thresholds, min_component, val_tuned = tune_thresholds(
            val_cases, dataset, cfg.FULL_RESOLUTION_EVAL, empty_one=cfg.BRATS_EMPTY, nested=cfg.NESTED_REGIONS)
    else:
        val = score(val_cases, thresholds, min_component, dataset, cfg.FULL_RESOLUTION_EVAL, cfg.BRATS_EMPTY,
                    cfg.NESTED_REGIONS, cfg.hd95)
        val_tuned = {r: val[r]["dice"] for r in REGIONS}
    test_rows = []
    test = score(test_cases, thresholds, min_component, dataset, cfg.FULL_RESOLUTION_EVAL, cfg.BRATS_EMPTY,
                 cfg.NESTED_REGIONS, cfg.hd95, per_case=test_rows)
    return {"thresholds": thresholds, "min_component_voxels": min_component,
            "evaluation_spec": evaluation_spec(cfg, thresholds, min_component),
            "full_resolution": cfg.FULL_RESOLUTION_EVAL, "brats_empty": cfg.BRATS_EMPTY,
            "nested_regions": cfg.NESTED_REGIONS, "surface_hd95": cfg.SURFACE_HD95,
            "val_tuned": val_tuned, "test": test, "test_per_case": test_rows}


def evaluation_spec(cfg, thresholds=None, min_component=None):
    r"""The complete scoring specification, written next to every result."""
    from .metrics import METRIC_VERSION
    return {"metric_version": METRIC_VERSION, "full_resolution": cfg.FULL_RESOLUTION_EVAL,
            "empty_region": "brats (empty prediction of an absent region scores 1)" if cfg.BRATS_EMPTY
            else "v7 (absent region scores 0)",
            "nested_regions": cfg.NESTED_REGIONS,
            "hd95": (f"surface, {cfg.HD95_CONVENTION} convention, "
                     + ("mm from scan spacing" if cfg.FULL_RESOLUTION_EVAL else "working-grid voxels"))
            if cfg.SURFACE_HD95 else "legacy voxel-based", "hd95_one_empty": "undefined: excluded from the conditional mean, counted as missed or false_positive",
            "tuned": cfg.TUNE_THRESHOLDS, "thresholds": thresholds, "min_component_voxels": min_component,
            "ensemble_n": cfg.ENSEMBLE_N, "tta": cfg.USE_TTA, "case_seed": cfg.CASE_SEED,
            "rows": "single pass and ensemble at threshold 0.5, then tuned post-processing, all with this spec",
            "validation_selection": "working grid, threshold 0.5"
            + (", nested masks" if cfg.NESTED_REGIONS else "") + (f", surface HD95 ({cfg.HD95_CONVENTION})"
                                                                if cfg.SURFACE_HD95 else "")}
