r"""Evaluation: Dice, mIoU and HD95 per region, plain and with tuned post-processing."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.Agent import iou_score, hd95_score

from .config import REGIONS


def evaluate(agent, dataset, state, ensemble=1, tta=False, workers=0):
    r"""Evaluate one split; ensemble > 1 averages stochastic passes and tta adds axis flips."""
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers)
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    flips = [None]
    if tta:
        flips = [None, (1,), (2,), (3,)]   # identity + flip each spatial axis
    with torch.no_grad():
        for data in loader:
            data = agent.prepare_data(data, eval=True)
            probs = []
            for fl in flips:
                idx, inp, tgt = data
                if fl is not None:
                    inp = torch.flip(inp, dims=fl)
                for _ in range(ensemble):
                    out, targets = agent.get_outputs((idx, inp, tgt), full_img=True)
                    p = torch.sigmoid(out)
                    if fl is not None:
                        p = torch.flip(p, dims=fl)         # un-flip prediction
                    probs.append(p.detach().cpu().numpy())
            prob = np.mean(probs, axis=0)
            gt = targets.detach().cpu().numpy()
            for i, r in enumerate(REGIONS):
                p, t = prob[..., i], gt[..., i]
                inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
                acc[r]["dice"].append((2 * inter) / ((p >= 0.5).sum() + (t >= 0.5).sum() + 1e-6))
                acc[r]["iou"].append(iou_score(p, t)); acc[r]["hd95"].append(hd95_score(p, t))
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


def predict_case(agent, data, ensemble=1, tta=False):
    r"""Mean probability map for one collated case: (case, prob float32, gt bool)."""
    data = agent.prepare_data(data, eval=True)
    idx, inp0, tgt = data
    flips = [None, (1,), (2,), (3,)] if tta else [None]
    probs = []
    for fl in flips:
        inp = torch.flip(inp0, dims=fl) if fl is not None else inp0
        for _ in range(ensemble):
            out, targets = agent.get_outputs((idx, inp, tgt), full_img=True)
            p = torch.sigmoid(out)
            if fl is not None:
                p = torch.flip(p, dims=fl)
            probs.append(p.detach().cpu().numpy())
    prob = np.mean(probs, axis=0)[0]
    gt = targets.detach().cpu().numpy()[0] >= 0.5
    return case_id(idx[0]), prob, gt


def collect_probs(agent, dataset, state, ensemble=1, tta=False):
    r"""Mean probability map per case as a list of (case, prob float16, gt uint8)."""
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    cases = []
    with torch.no_grad():
        for data in loader:
            case, prob, gt = predict_case(agent, data, ensemble, tta)
            cases.append((case, prob.astype(np.float16), gt.astype(np.uint8)))
    agent.exp.set_model_state("train")
    return cases


def _case_metrics(prob, gt, thresholds, min_component, with_hd95=True):
    r"""Per-region Dice, IoU and (optionally) HD95 for one case."""
    out = {}
    for i, r in enumerate(REGIONS):
        p = remove_small_components(prob[..., i] >= thresholds[r], min_component[r])
        t = gt[..., i] > 0
        pf, tf = p.astype(np.float32), t.astype(np.float32)
        out[r] = {"dice": float(_dice(p, t)), "iou": float(iou_score(pf, tf)),
                  "hd95": float(hd95_score(pf, tf)) if with_hd95 else float("nan")}
    return out


def _mean_metrics(rows):
    r"""Average per-case metrics per region, ignoring undefined HD95 values."""
    out = {}
    for r in REGIONS:
        hd = [x[r]["hd95"] for x in rows if not math.isnan(x[r]["hd95"])]
        out[r] = {"dice": float(np.mean([x[r]["dice"] for x in rows])),
                  "iou": float(np.mean([x[r]["iou"] for x in rows])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out


def score_streaming(models, dataset, state, progress_path, thresholds, min_component,
                    full_resolution=False, with_hd95=True):
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
            if full_resolution:
                prob, gt = to_full_resolution(dataset, case, prob)
            metrics = _case_metrics(prob, gt, thresholds, min_component, with_hd95)
            fh.write(json.dumps({"split": state, "case": case, "metrics": metrics}) + "\n")
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


def to_full_resolution(dataset, case, prob):
    r"""Resample a working-resolution prob map into the original scan; returns (prob, gt)."""
    folder = os.path.join(dataset.images_path, case)
    raw = np.stack([dataset.load_item(dataset._find_modality_file(folder, case, m))
                    for m in dataset.MODALITIES], axis=-1)
    seg = dataset.load_item(dataset._find_modality_file(folder, case, dataset.SEG_SUFFIX))
    gt = dataset._labels_to_regions(seg).astype(np.uint8)
    bbox = dataset._foreground_bbox(raw) if dataset.use_foreground_crop else None
    if bbox is None:
        bbox = (0, seg.shape[0], 0, seg.shape[1], 0, seg.shape[2])
    x0, x1, y0, y1, z0, z1 = bbox
    t = torch.from_numpy(prob.astype(np.float32)).permute(3, 0, 1, 2)[None]
    up = F.interpolate(t, size=(x1 - x0, y1 - y0, z1 - z0), mode="trilinear", align_corners=False)
    full = np.zeros(seg.shape + (prob.shape[-1],), np.float32)
    full[x0:x1, y0:y1, z0:z1] = up[0].permute(1, 2, 3, 0).numpy()
    return full, gt


def _pairs(cases, dataset=None, full_resolution=False):
    r"""Yield (prob, gt) per case, resampled to the original scan when requested."""
    for case, prob, gt in cases:
        if full_resolution:
            yield to_full_resolution(dataset, case, prob)
        else:
            yield prob.astype(np.float32), gt


def _dice(p, t):
    # Same formula as evaluate(), so results stay comparable with the plain metric.
    r"""Dice coefficient of two boolean masks."""
    return (2 * np.logical_and(p, t).sum()) / (p.sum() + t.sum() + 1e-6)


def _components(mask):
    r"""Connected-component labels (26-connectivity) and the size of each label."""
    from scipy import ndimage
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    return lab, np.bincount(lab.ravel())


def tune_thresholds(cases, dataset=None, full_resolution=False, grid=THRESHOLD_GRID,
                    size_grid=MIN_COMPONENT_GRID):
    r"""Pick per-region threshold and clean-up size that maximise validation Dice."""
    table = {r: {(th, sz): [] for th in grid for sz in size_grid} for r in REGIONS}
    for prob, gt in _pairs(cases, dataset, full_resolution):
        for i, r in enumerate(REGIONS):
            t = gt[..., i] > 0
            for th in grid:
                mask = prob[..., i] >= th
                lab, sizes = _components(mask) if mask.any() else (None, None)
                for sz in size_grid:
                    if lab is None or sz <= 0:
                        p = mask
                    else:
                        keep = sizes >= sz; keep[0] = False
                        p = keep[lab]
                    table[r][(th, sz)].append(_dice(p, t))
    best = {r: max(table[r], key=lambda k: (np.mean(table[r][k]), -k[1])) for r in REGIONS}
    thresholds = {r: best[r][0] for r in REGIONS}
    sizes = {r: int(best[r][1]) for r in REGIONS}
    val = {r: float(np.mean(table[r][best[r]])) for r in REGIONS}
    return thresholds, sizes, val


def score(cases, thresholds, min_component, dataset=None, full_resolution=False):
    r"""Dice, mIoU and HD95 per region with per-region thresholds and component clean-up."""
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    for prob, gt in _pairs(cases, dataset, full_resolution):
        for i, r in enumerate(REGIONS):
            p = remove_small_components(prob[..., i] >= thresholds[r], min_component[r])
            t = gt[..., i] > 0
            pf, tf = p.astype(np.float32), t.astype(np.float32)
            acc[r]["dice"].append(_dice(p, t))
            acc[r]["iou"].append(iou_score(pf, tf)); acc[r]["hd95"].append(hd95_score(pf, tf))
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])), "iou": float(np.mean(acc[r]["iou"])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out


def improved_evaluation(C, dataset, val_cases, test_cases):
    r"""Tune post-processing on validation (if enabled), then score the test cases once."""
    thresholds, min_component = {r: 0.5 for r in REGIONS}, dict(C.MIN_COMPONENT)
    val_tuned = None
    if C.TUNE_THRESHOLDS:
        thresholds, min_component, val_tuned = tune_thresholds(val_cases, dataset,
                                                               C.FULL_RESOLUTION_EVAL)
    else:
        val = score(val_cases, thresholds, min_component, dataset, C.FULL_RESOLUTION_EVAL)
        val_tuned = {r: val[r]["dice"] for r in REGIONS}
    test = score(test_cases, thresholds, min_component, dataset, C.FULL_RESOLUTION_EVAL)
    return {"thresholds": thresholds, "min_component_voxels": min_component,
            "full_resolution": C.FULL_RESOLUTION_EVAL, "val_tuned": val_tuned, "test": test}
