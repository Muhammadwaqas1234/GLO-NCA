r"""Segmentation metrics: IoU and 95th-percentile Hausdorff distance."""
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    _SCIPY_AVAILABLE = True
except Exception:  # scipy optional; HD95 simply returns NaN if missing
    _SCIPY_AVAILABLE = False


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    r"""Intersection-over-Union (Jaccard) for one binary region."""
    pred_bin = (pred >= threshold).astype(np.uint8)
    target_bin = (target >= 0.5).astype(np.uint8)
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    if union == 0:
        # Both empty: perfect agreement.
        return 1.0
    return float((intersection + smooth) / (union + smooth))


def _surface_distances(a, b):
    r"""Distances from every voxel of ``a`` to the nearest voxel of ``b``."""
    return distance_transform_edt(~b)[a]


def hd95_score(pred, target, threshold=0.5):
    r"""95th-percentile Hausdorff distance (in voxels) for one binary region."""
    if not _SCIPY_AVAILABLE:
        return float("nan")
    pred_bin = (pred >= threshold)
    target_bin = (target >= 0.5)
    if pred_bin.sum() == 0 and target_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return float("nan")
    all_d = np.concatenate([_surface_distances(pred_bin, target_bin),
                            _surface_distances(target_bin, pred_bin)])
    return float(np.percentile(all_d, 95))
