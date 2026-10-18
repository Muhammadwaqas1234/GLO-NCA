r"""Segmentation metrics: IoU and 95th-percentile Hausdorff distance (voxel-based and surface-based)."""
import numpy as np

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
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


def _surface(mask):
    r"""Boundary voxels of a boolean mask (voxels with a 6-connected background neighbour)."""
    return mask & ~binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)


def hd95_surface(pred, target, spacing=(1.0, 1.0, 1.0)):
    r"""Surface HD95 in physical units (BraTS/medpy): 0 if both masks are empty, NaN if one is."""
    if not _SCIPY_AVAILABLE:
        return float("nan")
    p, t = np.asarray(pred, bool), np.asarray(target, bool)
    if not p.any() and not t.any():
        return 0.0
    if not p.any() or not t.any():
        return float("nan")
    sp, st = _surface(p), _surface(t)
    d_pt = distance_transform_edt(~st, sampling=spacing)[sp]
    d_tp = distance_transform_edt(~sp, sampling=spacing)[st]
    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))
