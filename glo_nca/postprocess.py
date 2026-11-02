r"""Post-processing of the final masks, chosen on the validation split only.

Three steps, each off unless the validation search selects it:

- ``wt_keep_ratio``: keep the whole-tumour components of at least this fraction of the largest one
  (removes isolated false-positive spots); TC and ET are kept inside the remaining whole tumour.
- ``tc_fill_holes``: fill enclosed holes of the tumour core.
- ``et_rule``: (ratio, voxels) relabels a small predicted ET as necrotic core (see relabel_small_et).

The search (``tune_postprocess``) adds one step at a time and keeps it only when it raises the mean
lesion-wise Dice of the validation cases (the BraTS 2023 ranking metric) by more than ``min_gain``.
"""
import numpy as np
from scipy import ndimage

from .lesion import lesion_wise

REGIONS = ("WT", "TC", "ET")
NONE = {"wt_keep_ratio": 0.0, "tc_fill_holes": False, "et_rule": None}
WT_KEEP_GRID = (0.05, 0.1, 0.2, 0.5)
ET_RULE_GRID = tuple((r, v) for r in (0.1, 0.25, 0.5) for v in (500, 2000, 5000))
_CUBE = np.ones((3, 3, 3), bool)


def keep_main_tumour(mask, ratio):
    r"""Components (26-connected) of at least ``ratio`` times the largest component's size."""
    if ratio <= 0 or not mask.any():
        return mask
    lab, n = ndimage.label(mask, structure=_CUBE)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= ratio * sizes.max()
    keep[0] = False
    return keep[lab]


def apply_postprocess(masks, post):
    r"""Apply the selected steps to nested final masks (WT ⊇ TC ⊇ ET); returns new masks."""
    if not post:
        return masks
    m = dict(masks)
    if post.get("et_rule"):
        from .evaluation import relabel_small_et
        m = relabel_small_et(m, *post["et_rule"])
    if post.get("tc_fill_holes") and m["TC"].any():
        m["TC"] = ndimage.binary_fill_holes(m["TC"])
    if post.get("wt_keep_ratio", 0) > 0:
        m["WT"] = keep_main_tumour(m["WT"], post["wt_keep_ratio"])
    # Keep the nesting of the BraTS label map.
    m["TC"] = (m["TC"] | m["ET"]) & m["WT"]
    m["ET"] = m["ET"] & m["TC"]
    return m


def _objective(decoded, post):
    r"""Mean lesion-wise Dice over cases and regions for one post-processing setting."""
    vals = []
    for masks, gt, spacing in decoded:
        m = apply_postprocess(masks, post)
        vals.extend(lesion_wise(m[r], gt[..., i] > 0, spacing=spacing)["dice"] for i, r in enumerate(REGIONS))
    return float(np.mean(vals))


def tune_postprocess(decoded, min_gain=0.001):
    r"""Greedy validation search; ``decoded`` is [(nested masks, gt, spacing)] of the validation cases.

    Returns (chosen setting, objective of every candidate tried) so the search is reported in full.
    """
    best = dict(NONE)
    trace = {"none": _objective(decoded, best)}
    score = trace["none"]
    for key, grid in (("wt_keep_ratio", WT_KEEP_GRID), ("tc_fill_holes", (True,)), ("et_rule", ET_RULE_GRID)):
        cand = {v: _objective(decoded, {**best, key: v}) for v in grid}
        trace.update({f"{key}={v}": s for v, s in cand.items()})
        v, s = max(cand.items(), key=lambda kv: kv[1])
        if s > score + min_gain:
            best[key], score = v, s
    return best, trace
