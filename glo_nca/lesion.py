r"""Lesion-wise Dice and HD95 in the form of the BraTS 2023 challenge evaluation.

Every ground-truth lesion (26-connected component) is scored on its own: the predicted components that
overlap the lesion dilated by ``dilation`` voxels are its match. A lesion without a match scores Dice 0
and HD95 ``penalty``; every predicted component that matches no lesion is a false positive with the
same scores. The case score is the sum over lesions divided by (lesions + false positives). Lesions
smaller than ``min_volume`` voxels are not scored, and predicted components that match them are not
counted as false positives. A case without lesions and without predictions scores Dice 1 and HD95 0.
"""
import numpy as np
from scipy import ndimage

from .metrics import hd95_surface

DILATION = 3            # voxels of 26-connected dilation used to match predictions to a lesion
MIN_VOLUME = 50         # ground-truth lesions below this size (voxels) are not scored
PENALTY = 374.0         # HD95 (mm) of a missed lesion or a false-positive component

_CUBE = np.ones((3, 3, 3), bool)


def lesion_wise(pred, target, spacing=(1.0, 1.0, 1.0), convention="medpy", dilation=DILATION,
                min_volume=MIN_VOLUME, penalty=PENALTY):
    r"""Lesion-wise scores of one region of one case.

    Returns a dict with ``dice``, ``hd95`` and the counts ``lesions`` (scored ground-truth lesions),
    ``detected``, ``missed`` and ``false_positive``.
    """
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    gt_lab, n_gt = ndimage.label(target, structure=_CUBE)
    pred_lab, n_pred = ndimage.label(pred, structure=_CUBE)
    matched = set()
    dice, hd, detected, missed = [], [], 0, 0
    for k in range(1, n_gt + 1):
        lesion = gt_lab == k
        hits = np.unique(pred_lab[ndimage.binary_dilation(lesion, structure=_CUBE, iterations=dilation)])
        hits = hits[hits > 0]
        matched.update(int(h) for h in hits)
        if lesion.sum() < min_volume:
            continue
        if hits.size == 0:
            dice.append(0.0); hd.append(penalty); missed += 1
            continue
        match = np.isin(pred_lab, hits)
        inter = np.logical_and(match, lesion).sum()
        dice.append(float(2.0 * inter / (match.sum() + lesion.sum())))
        hd.append(float(hd95_surface(match, lesion, spacing=spacing, convention=convention)))
        detected += 1
    false_positive = n_pred - len(matched)
    n = len(dice) + false_positive
    if n == 0:
        return {"dice": 1.0, "hd95": 0.0, "lesions": 0, "detected": 0, "missed": 0, "false_positive": 0}
    return {"dice": float(sum(dice) / n), "hd95": float((sum(hd) + penalty * false_positive) / n),
            "lesions": len(dice), "detected": detected, "missed": missed, "false_positive": int(false_positive)}


def penalised_hd95(value, status, penalty=PENALTY):
    r"""Case HD95 with the BraTS rule for empty masks: 0 if both are empty, ``penalty`` if one is."""
    if status == "both_empty":
        return 0.0
    if status in ("missed", "false_positive"):
        return penalty
    return float(value)
