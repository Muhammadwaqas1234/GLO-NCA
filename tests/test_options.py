#!/usr/bin/env python
r"""Unit checks for the optional improvements (no data or GPU needed)."""
import os
import random
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from glo_nca.config import load_config  # noqa: E402
from glo_nca.data import Dataset_BraTS_Foreground  # noqa: E402
from glo_nca.evaluation import remove_small_components, score, to_full_resolution, tune_thresholds  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    r"""Record and print one check result."""
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{detail}")


def test_config():
    r"""Defaults, base inheritance and the fixed split seed."""
    base = load_config(os.path.join(_ROOT, "configs", "glo_nca_cascade_brats2021.yaml"))
    check("defaults reproduce the original recipe",
          base.SPLIT_SEED == base.SEED and not base.AUGMENT and base.REGION_WEIGHTS == [1, 1, 1]
          and not base.improved_eval)
    exp = load_config(os.path.join(_ROOT, "configs", "experiments", "brats2021_improved.yaml"))
    check("base: inheritance merges nested sections",
          exp.MODALITIES == base.MODALITIES and exp.INPUT_SIZE[-1] == [96, 96, 96]
          and exp.TVERSKY_BETA == 0.65 and exp.CHANNEL_N == base.CHANNEL_N)
    ovr = load_config(os.path.join(_ROOT, "configs", "experiments", "brats2021_improved.yaml"),
                      ["experiment.seed=7"])
    check("model seed changes, split seed stays fixed", ovr.SEED == 7 and ovr.SPLIT_SEED == 42)


def test_augmentation():
    r"""Augmentation keeps labels aligned and background at zero."""
    random.seed(0)
    img = np.random.rand(16, 16, 8, 4).astype(np.float32)
    label = (img[..., :3] > 0.5).astype(np.float32)
    ok = True
    for _ in range(20):
        a, b = Dataset_BraTS_Foreground._augment_spatial(img, label)
        ok &= np.array_equal(b, (a[..., :3] > 0.5).astype(np.float32)) and a.flags["C_CONTIGUOUS"]
    check("spatial augmentation keeps image and label aligned", bool(ok))
    z = img.copy(); z[:4] = 0
    out = Dataset_BraTS_Foreground._augment_intensity(z)
    check("intensity augmentation leaves background at zero",
          bool((out[:4] == 0).all() and not np.allclose(out[4:], z[4:])))


def test_components():
    r"""Small components are removed, large ones kept."""
    m = np.zeros((20, 20, 20), bool); m[2:8, 2:8, 2:8] = True; m[15, 15, 15] = True
    r = remove_small_components(m, 10)
    check("small components are removed, large ones kept", bool(r.sum() == 216 and not r[15, 15, 15]))


def test_tuning():
    r"""Threshold and clean-up tuning on synthetic cases."""
    rng = np.random.default_rng(0)
    cases = []
    for i in range(4):
        gt = np.zeros((16, 16, 16, 3), np.uint8); gt[4:12, 4:12, 4:12] = 1
        prob = np.where(gt > 0, 0.4, 0.1) + rng.uniform(0, 0.05, gt.shape)
        prob[1, 1, 1] = 0.9   # isolated false positive
        cases.append((f"c{i}", prob.astype(np.float16), gt))
    th, sizes, val = tune_thresholds(cases)
    check("tuning finds the lower threshold and a clean-up size",
          all(th[r] <= 0.4 and sizes[r] > 0 and val[r] > 0.99 for r in th), f"{th} {sizes}")
    res = score(cases, th, sizes)
    check("scoring with tuned settings", all(res[r]["dice"] > 0.99 for r in res))


class _FakeDS:
    r"""Minimal dataset stand-in for the full-resolution test."""
    MODALITIES = ["a", "b"]
    SEG_SUFFIX = "seg"
    use_foreground_crop = True
    _foreground_bbox = staticmethod(Dataset_BraTS_Foreground._foreground_bbox)
    _labels_to_regions = staticmethod(Dataset_BraTS_Foreground._labels_to_regions)

    def __init__(self, root, vols):
        r"""Store the fake volumes."""
        self.images_path, self.vols = root, vols

    def _find_modality_file(self, folder, case, m):
        r"""Return the modality key itself."""
        return m

    def load_item(self, key):
        r"""Return the stored volume."""
        return self.vols[key]


def test_full_resolution():
    r"""Predictions map back to the original scan space."""
    raw = np.zeros((40, 40, 30)); raw[10:30, 5:35, 4:24] = 1.0
    seg = np.zeros_like(raw); seg[15:25, 10:30, 8:20] = 2
    ds = _FakeDS(tempfile.gettempdir(), {"a": raw, "b": raw, "seg": seg})
    prob = np.zeros((10, 15, 10, 3), np.float32)
    prob[2:8, 2:13, 2:8, 0] = 1.0   # WT in the cropped, downsampled frame
    full, gt = to_full_resolution(ds, "case", prob)
    wt = full[..., 0] >= 0.5
    dice = 2 * (wt & (gt[..., 0] > 0)).sum() / (wt.sum() + gt[..., 0].sum())
    check("full resolution maps back to the original scan",
          full.shape == (40, 40, 30, 3) and not wt[:10].any() and dice > 0.8, f"dice {dice:.3f}")


def test_early_stop():
    r"""Early stopping counts only gains above min_delta."""
    from glo_nca.trainer import last_gain_epoch
    nan = float("nan")
    h = {"epoch": [1, 2, 3, 4, 5, 6], "val_mean": [0.60, nan, 0.65, 0.655, 0.66, 0.70]}
    check("early stop: last gain above min_delta 0.01",
          last_gain_epoch(h, 0.01) == 6, f"{last_gain_epoch(h, 0.01)}")
    h2 = {"epoch": [1, 2, 3, 4], "val_mean": [0.60, 0.605, 0.608, 0.609]}
    check("early stop: small gains do not reset patience", last_gain_epoch(h2, 0.01) == 1)


def test_brats_empty():
    r"""BraTS scoring gives 1 to an empty prediction of an absent region; the default gives 0."""
    from glo_nca.evaluation import _dice
    empty = np.zeros((4, 4, 4), bool)
    full = np.ones((4, 4, 4), bool)
    check("empty region: v7 formula 0, BraTS scoring 1",
          _dice(empty, empty) == 0 and _dice(empty, empty, empty_one=True) == 1.0
          and _dice(full, empty, empty_one=True) == 0)


def test_hd95_surface():
    r"""Surface HD95: identical masks give 0, a cube shifted by 2 voxels gives 2, empty cases follow BraTS."""
    from src.utils.metrics import hd95_surface
    a = np.zeros((30, 30, 30), bool); a[5:15, 5:15, 5:15] = True
    b = np.zeros_like(a); b[7:17, 5:15, 5:15] = True
    e = np.zeros_like(a)
    ok = (hd95_surface(a, a) == 0.0 and abs(hd95_surface(a, b) - 2.0) < 1e-6
          and hd95_surface(e, e) == 0.0 and np.isnan(hd95_surface(a, e))
          and abs(hd95_surface(a, b, spacing=(2.0, 1.0, 1.0)) - 4.0) < 1e-6)
    check("surface HD95: known distances, spacing and empty cases", ok,
          f"shift 2 -> {hd95_surface(a, b):.2f}")


if __name__ == "__main__":
    for t in (test_config, test_augmentation, test_components, test_tuning, test_full_resolution, test_early_stop, test_brats_empty, test_hd95_surface):
        t()
    print(f"\n  {sum(RESULTS)}/{len(RESULTS)} passed")
    raise SystemExit(0 if all(RESULTS) else 1)
