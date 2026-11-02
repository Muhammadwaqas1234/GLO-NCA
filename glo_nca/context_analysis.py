r"""Analyses of how GLO-NCA uses global context.

- knock-out: switch off one context mechanism of a trained model at inference and score the test cases;
- attention: where the attention-pooled global block looks, and how much of it falls on the tumour;
- false-positive distance: predicted tumour components without ground truth and their distance to it;
- figures: HD95 distributions of several runs, and example cases chosen by a fixed rule.
"""
import contextlib
import csv

import numpy as np
import torch
from scipy import ndimage

REGIONS = ("WT", "TC", "ET")
_CUBE = np.ones((3, 3, 3), bool)

# Mechanisms that can be switched off at inference: name -> parts.
KNOCKOUTS = {"global block": {"global_context"}, "SE": {"se"}, "spatial gate": {"spatial_gate"},
             "coarse levels": {"coarse"}, "all context": {"global_context", "se", "spatial_gate", "coarse"}}


def knockout_variants(cfg):
    r"""The knock-outs that apply to this model's configuration, after the intact model."""
    present = {"global block": cfg.USE_GLOBAL, "SE": cfg.USE_SE, "spatial gate": cfg.USE_SPATIAL,
               "coarse levels": cfg.LEVELS > 1, "all context": True}
    return ["intact"] + [k for k in KNOCKOUTS if present[k]]


@contextlib.contextmanager
def knocked_out(cells, parts):
    r"""Temporarily remove ``parts`` from the cells; ``coarse`` makes every coarse level pass on zeros."""
    saved = []
    for i, cell in enumerate(cells):
        state = {a: getattr(cell, a) for a in ("se", "spatial_gate", "global_context")}
        saved.append(state)
        for a in state:
            if a in parts:
                setattr(cell, a, None)
        if "coarse" in parts and i < len(cells) - 1:
            # The finer level then receives no hidden state from this level: no coarse information.
            cell.forward = lambda x, steps=10, skip_prob=None: torch.zeros_like(x)
    try:
        yield
    finally:
        for cell, state in zip(cells, saved):
            for a, v in state.items():
                setattr(cell, a, v)
            cell.__dict__.pop("forward", None)


@contextlib.contextmanager
def recording_attention(cells):
    r"""Record the attention maps of every global-context block; yields {level: list of maps}."""
    records = {}
    for i, cell in enumerate(cells):
        if cell.global_context is not None:
            records[i] = cell.global_context.record = []
    try:
        yield records
    finally:
        for cell in cells:
            if cell.global_context is not None:
                cell.global_context.record = None


def attention_on_tumour(attn, wt):
    r"""Share of the attention weight on whole-tumour voxels and its ratio to the tumour's volume share.

    ``attn`` is one level's map (X, Y, Z) summing to 1; ``wt`` the ground truth at full working size, max-pooled
    to the level's size. A ratio above 1 means the block looks at the tumour more than at random.
    """
    factor = [w // a for w, a in zip(wt.shape, attn.shape)]
    t = torch.from_numpy(wt.astype(np.float32))[None, None]
    t = torch.nn.functional.max_pool3d(t, kernel_size=factor, stride=factor)[0, 0].numpy() > 0
    if not t.any():
        return float("nan"), float("nan")
    mass = float(attn[t].sum())
    return mass, mass / float(t.mean())


def false_positive_components(pred, gt, spacing):
    r"""Predicted components without any ground-truth voxel: count, volume (ml) and distance (mm) to the tumour.

    ``pred`` and ``gt`` are whole-tumour masks in the scan; with an empty ground truth every component is
    false and its distance is undefined (NaN).
    """
    lab, n = ndimage.label(pred, structure=_CUBE)
    if n == 0:
        return []
    voxel_ml = float(np.prod(spacing)) / 1000.0
    dist = ndimage.distance_transform_edt(~gt, sampling=spacing) if gt.any() else None
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        if (gt[sl] & comp).any():
            continue
        d = float(dist[sl][comp].min()) if dist is not None else float("nan")
        out.append({"volume_ml": float(comp.sum()) * voxel_ml, "distance_mm": d})
    return out


def summarise_false_positives(rows, far_mm=20.0):
    r"""Per-run summary of false-positive components over cases."""
    comps = [c for r in rows for c in r["components"]]
    dists = np.array([c["distance_mm"] for c in comps if not np.isnan(c["distance_mm"])])
    return {"cases": len(rows), "cases_with_false_components": int(sum(bool(r["components"]) for r in rows)),
            "false_components": len(comps), "false_components_per_case": len(comps) / max(len(rows), 1),
            "false_volume_ml_mean_per_case": float(np.mean([sum(c["volume_ml"] for c in r["components"]) for r in rows]))
            if rows else float("nan"),
            f"false_components_beyond_{far_mm:g}mm": int((dists > far_mm).sum()),
            "distance_mm_median": float(np.median(dists)) if len(dists) else float("nan"),
            "distance_mm_q90": float(np.percentile(dists, 90)) if len(dists) else float("nan")}


def rule_cases(diff, quantiles=(0.5, 0.9)):
    r"""Cases whose difference is closest to the given quantiles (a fixed rule, not a hand-picked choice)."""
    items = [(c, d) for c, d in diff.items() if not np.isnan(d)]
    if not items:
        return []
    values = np.array([d for _, d in items])
    out = []
    for q in quantiles:
        target = np.quantile(values, q)
        case = min(items, key=lambda cd: abs(cd[1] - target))[0]
        if case not in out:
            out.append(case)
    return out


def read_per_case(path, metric="hd95"):
    r"""Per-case values {case: {region: value}} from a per-case CSV (HD95 with 374 mm for a missed or false region)."""
    from .lesion import penalised_hd95
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if metric == "hd95":
                out[row["case"]] = {r: penalised_hd95(float(row[f"{r}_hd95"]), row.get(f"{r}_hd95_status", "defined"))
                                    for r in REGIONS}
            else:
                out[row["case"]] = {r: float(row[r if metric == "dice" else f"{r}_{metric}"]) for r in REGIONS}
    return out


def plot_hd95(runs, out_path):
    r"""Box plots of per-case HD95 per region for several runs: {label: {case: {region: value}}}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, r in zip(axes, REGIONS):
        data = [[v[r] for v in vals.values()] for vals in runs.values()]
        ax.boxplot(data, showfliers=True, flierprops={"markersize": 3})
        ax.set_xticks(range(1, len(runs) + 1), list(runs), rotation=20, ha="right", fontsize=8)
        ax.set_title(r)
        ax.set_yscale("symlog", linthresh=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("HD95 (mm; 374 for a missed or false region)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_examples(cases, image_of, label_of, preds, out_path):
    r"""One row per case: image, ground truth and each run's prediction on the slice with most tumour.

    ``image_of(case)`` returns the scan used as background, ``label_of(case)`` the ground-truth label map,
    and ``preds`` maps a run label to a function returning that run's label map for the case.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 2 + len(preds)
    fig, axes = plt.subplots(len(cases), cols, figsize=(2.6 * cols, 2.6 * len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        img, gt = image_of(case), label_of(case)
        z = int(np.argmax((gt > 0).sum(axis=(0, 1))))
        panels = [("image", None), ("ground truth", gt)] + [(name, f(case)) for name, f in preds.items()]
        for col, (title, lab) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.rot90(img[..., z]), cmap="gray")
            if lab is not None:
                over = np.ma.masked_equal(np.rot90(lab[..., z]).astype(float), 0)
                ax.imshow(over, cmap="jet", alpha=0.45, vmin=0, vmax=4)
            ax.set_title(title if row == 0 else "", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(case, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
