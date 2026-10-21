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
from glo_nca.data import BraTSDataset, labels_to_regions  # noqa: E402
from glo_nca.evaluation import remove_small_components, score, to_full_resolution, tune_thresholds  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    r"""Record and print one check result."""
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{detail}")


def test_config():
    r"""Defaults, base inheritance and the fixed split seed."""
    base = load_config(os.path.join(_ROOT, "configs", "glo_nca_cascade_brats2021.yaml"))
    check("defaults reproduce the original recipe",
          base.SPLIT_SEED == base.SEED and not base.AUGMENT and base.REGION_WEIGHTS == [1, 1, 1]
          and not base.uses_final_scoring)
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
        a, b = BraTSDataset._augment_spatial(img, label)
        ok &= np.array_equal(b, (a[..., :3] > 0.5).astype(np.float32)) and a.flags["C_CONTIGUOUS"]
    check("spatial augmentation keeps image and label aligned", bool(ok))
    z = img.copy(); z[:4] = 0
    out = BraTSDataset._augment_intensity(z)
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
    _foreground_bbox = staticmethod(BraTSDataset._foreground_bbox)
    _labels_to_regions = staticmethod(lambda seg: labels_to_regions(seg))

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
    from glo_nca.metrics import hd95_surface
    a = np.zeros((30, 30, 30), bool); a[5:15, 5:15, 5:15] = True
    b = np.zeros_like(a); b[7:17, 5:15, 5:15] = True
    e = np.zeros_like(a)
    ok = (hd95_surface(a, a) == 0.0 and abs(hd95_surface(a, b) - 2.0) < 1e-6
          and hd95_surface(e, e) == 0.0 and np.isnan(hd95_surface(a, e))
          and abs(hd95_surface(a, b, spacing=(2.0, 1.0, 1.0)) - 4.0) < 1e-6)
    check("surface HD95: known distances, spacing and empty cases", ok,
          f"shift 2 -> {hd95_surface(a, b):.2f}")


def test_review_options():
    r"""Review fixes are off by default, on in the fixed config; update_prob maps to the skip rate."""
    base = load_config(os.path.join(_ROOT, "configs", "experiments", "brats2021_res96_long.yaml"))
    fixed = load_config(os.path.join(_ROOT, "configs", "experiments", "brats2021_res96_fixed.yaml"))
    plain = load_config(os.path.join(_ROOT, "configs", "experiments", "brats2021_res96_fixed_plain.yaml"))
    check("review fixes off by default (reported run unchanged)",
          base.USE_SE and base.USE_SPATIAL and not (base.USE_GLOBAL or base.ALL_REGION_LOSS
          or base.VALIDATE_EMA or base.CROP_ET_PROB or base.NESTED_REGIONS or base.SURFACE_HD95))
    check("fixed config: all fixes on, same recipe otherwise",
          fixed.ALL_REGION_LOSS and fixed.VALIDATE_EMA and fixed.CROP_ET_PROB == 0.5
          and fixed.NESTED_REGIONS and fixed.SURFACE_HD95 and abs(fixed.SKIP_PROB - 0.6) < 1e-9
          and fixed.EPOCHS == base.EPOCHS and fixed.INPUT_SIZE == base.INPUT_SIZE
          and fixed.SPLIT_SEED == base.SPLIT_SEED and fixed.SEED == base.SEED)
    check("plain control differs only in SE and spatial gate",
          not plain.USE_SE and not plain.USE_SPATIAL
          and {k: v for k, v in vars(plain).items() if k not in ("NAME", "USE_SE", "USE_SPATIAL", "raw")}
          == {k: v for k, v in vars(fixed).items() if k not in ("NAME", "USE_SE", "USE_SPATIAL", "raw")})


def test_all_region_loss():
    r"""With all_regions, positive ET logits on an ET-free crop are penalised and get a gradient."""
    import torch
    from glo_nca.agent import CascadeAgent
    from glo_nca.losses import FocalTverskyCELoss

    class _Exp:
        def get_from_config(self, key):
            return 0 if key == "train_model" else None

    class _Opt:
        def zero_grad(self): pass
        def step(self): pass

    out = torch.full((1, 8, 8, 8, 3), 3.0, requires_grad=True)
    tgt = torch.zeros((1, 8, 8, 8, 3)); tgt[..., 0:2] = 1.0       # WT and TC present, ET absent
    found = {}
    for all_regions in (False, True):
        agent = CascadeAgent.__new__(CascadeAgent)
        agent.exp, agent.optimizer, agent.scheduler = _Exp(), [_Opt()], [_Opt()]
        agent.region_weights, agent.all_regions = None, all_regions
        agent.prepare_data = lambda d: d
        agent.get_outputs = lambda d: (out, tgt)
        out.grad = None
        losses = agent.batch_step(None, FocalTverskyCELoss(alpha=0.25, beta=0.75))
        found[all_regions] = (2 in losses, out.grad is not None and bool(out.grad[..., 2].abs().sum() > 0))
    check("absent ET: no loss before, loss and gradient with all_regions",
          found[False] == (False, False) and found[True] == (True, True))


def test_et_crop():
    r"""ET-biased crop origins contain enhancing tumour and stay inside the volume."""
    import torch
    from glo_nca.agent import CascadeAgent
    t = torch.zeros((40, 40, 40, 3)); t[35:38, 2:4, 30:33, 2] = 1
    ok = True
    for _ in range(50):
        x, y, z = CascadeAgent._et_crop(t, (16, 16, 16))
        ok &= 0 <= x <= 24 and 0 <= y <= 24 and 0 <= z <= 24
        ok &= bool(t[x:x+16, y:y+16, z:z+16, 2].any())
    check("ET crop contains ET; none without ET",
          ok and CascadeAgent._et_crop(torch.zeros((40, 40, 40, 3)), (16, 16, 16)) is None)


def test_global_block():
    r"""Full cell, BatchNorm, batch size 1: the global branch changes the output and receives gradients."""
    import torch
    from glo_nca.model import SpatialGate3D, GLONCACell
    changes, grads = {}, {}
    for norm in ("batch", "layer"):
        torch.manual_seed(11)
        cell = GLONCACell(12, 0.0, "cpu", 32, kernel_size=3, input_channels=4, use_global=True, norm=norm)
        torch.nn.init.normal_(cell.global_context.transform[-1].weight, std=0.5)
        torch.nn.init.normal_(cell.fc_out.weight, std=0.1)
        x = torch.randn(1, 8, 8, 8, 12)
        torch.manual_seed(1); on = cell(x, steps=2)
        branch, cell.global_context = cell.global_context, None              # negative control: the branch bypassed
        torch.manual_seed(1); off = cell(x, steps=2)
        torch.manual_seed(1); again = cell(x, steps=2)
        cell.global_context = branch
        torch.manual_seed(1); out = cell(x, steps=2)
        out[..., 4:7].sum().backward()
        changes[norm] = ((on - off).abs().max().item(), (off - again).abs().max().item())
        grads[norm] = branch.transform[-1].weight.grad.norm().item()
    # The spatial gate stays local: a distant voxel does not change it.
    torch.manual_seed(0)
    s = SpatialGate3D(7)
    y = torch.randn(1, 24, 20, 20, 20); far = y.clone(); far[..., 19, 19, 19] += 50.0
    with torch.no_grad():
        local = (s(far) - s(y))[..., 0, 0, 0].abs().max().item()
    ok = all(c[0] > 1e-2 and c[1] == 0 for c in changes.values()) and all(g > 1e-3 for g in grads.values())
    check("global branch survives BatchNorm in the full cell (batch 1)", ok and local == 0,
          f"change {changes['batch'][0]:.2f}, grad {grads['batch']:.1e}, control {changes['batch'][1]:.0f}")


def test_decoder():
    r"""Exported label maps reload to exactly the nested masks the evaluation scores."""
    from glo_nca.evaluation import decode_regions, masks_to_label
    rng = np.random.default_rng(0)
    prob = rng.random((20, 20, 20, 3)).astype(np.float32)
    th, mc = {"WT": 0.5, "TC": 0.6, "ET": 0.7}, {"WT": 0, "TC": 0, "ET": 0}
    masks = decode_regions(prob, th, mc)
    back = labels_to_regions(masks_to_label(masks, 3)) > 0
    raw = decode_regions(prob, th, mc, nested=False)
    legacy = np.zeros((20, 20, 20), np.uint8)       # the earlier predict.py label map
    legacy[raw["WT"]] = 2; legacy[raw["TC"]] = 1; legacy[raw["ET"]] = 3
    check("decoder: label map round trip equals scored masks",
          all(np.array_equal(back[..., i], masks[r]) for i, r in enumerate(["WT", "TC", "ET"])))
    check("decoder: exported label maps unchanged", np.array_equal(legacy, masks_to_label(masks, 3)),
          f"nesting adds {int((masks['WT'] & ~raw['WT']).sum())} WT voxels")


def test_with_weights():
    r"""EMA validation scores the given weights and restores the live ones afterwards."""
    import torch
    from glo_nca.trainer import with_weights
    m = torch.nn.Linear(2, 1)
    live = {k: v.clone() for k, v in m.state_dict().items()}
    ema = {k: torch.full_like(v, 7.0) for k, v in m.state_dict().items()}
    seen = with_weights([m], [ema], lambda: m.weight.detach().clone())
    check("EMA validation uses EMA weights, restores live weights",
          bool((seen == 7.0).all()) and all(torch.equal(m.state_dict()[k], live[k]) for k in live))


def test_local_norm():
    r"""Per-voxel LayerNorm keeps a plain cell strictly local; BatchNorm uses volume statistics."""
    import torch
    from glo_nca.model import GLONCACell
    torch.manual_seed(0)
    x = torch.randn(1, 16, 16, 16, 24)
    far = x.clone(); far[0, 15, 15, 15, 4:] += 5.0
    effect = {}
    for norm in ("batch", "layer"):
        cell = GLONCACell(24, 0.0, "cpu", 32, kernel_size=3, input_channels=4, norm=norm)
        torch.nn.init.normal_(cell.fc_out.weight, std=0.1)
        with torch.no_grad():
            torch.manual_seed(1); a = cell.update(x, 0.0)
            torch.manual_seed(1); b = cell.update(far, 0.0)
        effect[norm] = (a - b)[0, 0, 0, 0].abs().max().item()
    check("layer norm: one step is strictly local, batch norm is not",
          effect["layer"] == 0.0 and effect["batch"] > 0, f"batch {effect['batch']:.2e}")


def test_bce_logits():
    r"""BCE on logits matches the clamped version for moderate logits and stays finite for large ones."""
    import torch
    from glo_nca.losses import FocalTverskyCELoss
    z = torch.linspace(-4, 4, 100); t = (z > 0.3).float()
    old = FocalTverskyCELoss(alpha=0.25, beta=0.75)(z, t)
    new = FocalTverskyCELoss(alpha=0.25, beta=0.75, bce_logits=True)(z, t)
    big = torch.tensor([60.0, -60.0], requires_grad=True)
    loss = FocalTverskyCELoss(bce_logits=True)(big, torch.tensor([0.0, 1.0]))
    loss.backward()
    check("BCE on logits: same value, finite gradient at extreme logits",
          abs(old.item() - new.item()) < 1e-4 and bool(torch.isfinite(big.grad).all()) and big.grad.abs().sum() > 0)


def test_brain_mask():
    r"""Brain-mask normalisation: zero outside the brain, zero mean and unit variance inside."""
    ds = BraTSDataset.__new__(BraTSDataset)
    ds.size, ds.brain_mask, ds.use_nonzero_norm = (20, 20, 20), True, True
    raw = np.zeros((30, 30, 30, 2), np.float32)
    raw[5:25, 5:25, 5:25] = np.random.default_rng(0).random((20, 20, 20, 2)) * 100 + 1
    img = ds.normalize(ds.resize_image(raw))
    inside = np.any(img != 0, axis=-1)
    ok = (abs(img[inside].mean()) < 1e-4 and abs(img[..., 0][inside].std() - 1) < 1e-3
          and not img[0, 0, 0].any() and inside.sum() > 0)
    check("brain mask: background zero, brain normalised", bool(ok))


def test_nested_tuning_and_spacing():
    r"""Nested tuning scores the final masks; surface HD95 scales with the voxel spacing."""
    from glo_nca.evaluation import _case_metrics
    gt = np.zeros((20, 20, 20, 3), np.uint8)
    gt[4:16, 4:16, 4:16, 0] = 1; gt[6:14, 6:14, 6:14, 1] = 1; gt[8:12, 8:12, 8:12, 2] = 1
    prob = gt.astype(np.float32) * 0.9
    prob[..., 1] *= 0.5            # TC probability low; nesting with ET still yields part of TC
    cases = [("c", prob, gt)]
    _, _, val_plain = tune_thresholds(cases)
    _, _, val_nested = tune_thresholds(cases, nested=True)
    th, mc = {"WT": 0.5, "TC": 0.5, "ET": 0.5}, {"WT": 0, "TC": 0, "ET": 0}
    shift = np.zeros_like(prob); shift[2:, ..., :] = prob[:-2]
    hd1 = _case_metrics(shift, gt, th, mc, surface_hd95=True)["WT"]["hd95"]
    hd2 = _case_metrics(shift, gt, th, mc, surface_hd95=True, spacing=(2.0, 1.0, 1.0))["WT"]["hd95"]
    check("nested tuning returns scores on final masks", val_nested["TC"] >= val_plain["TC"] - 1e-9)
    check("HD95 in mm uses the scan spacing", abs(hd1 - 2.0) < 1e-6 and abs(hd2 - 4.0) < 1e-6,
          f"{hd1:.1f} -> {hd2:.1f}")


def test_compare_runs():
    r"""Paired comparison: patient-level estimand, scans averaged per patient, several seeds."""
    from glo_nca.cli.compare_runs import compare, patient_of
    rng = np.random.default_rng(0)
    a = {f"BraTS-GLI-{i:05d}-{j:03d}": {r: float(rng.random()) for r in ["WT", "TC", "ET"]}
         for i in range(30) for j in range(1 + (i % 3 == 0))}          # some patients have two scans
    b = {c: {r: v + 0.05 for r, v in s.items()} for c, s in a.items()}
    b2 = {c: {r: v + 0.03 for r, v in s.items()} for c, s in a.items()}
    _, patients, same = compare(a, a, boot=200)
    _, _, up = compare(a, b, boot=200)
    _, _, seeds = compare([a, a], [b, b2], boot=300)
    check("compare_runs: patient estimand, seed-wise effects",
          same["mean"]["mean_diff"] == 0 and abs(up["mean"]["mean_diff"] - 0.05) < 1e-9
          and up["mean"]["ci95"][0] > 0.04 and len(patients) == 30
          and np.allclose(seeds["mean"]["seed_effects"], [0.05, 0.03]) and abs(seeds["mean"]["mean_diff"] - 0.04) < 1e-9
          and seeds["mean"]["ci95"][0] <= 0.03 + 1e-9 and seeds["mean"]["ci95"][1] >= 0.05 - 1e-9
          and patient_of("BraTS2021_00001") == "BraTS2021_00001")


def test_labels():
    r"""The configured ET labels reach the production loader; unknown labels are rejected."""
    from glo_nca.trainer import build
    reg = labels_to_regions(np.array([0, 1, 2, 3, 4]))
    ok = reg[:, 2].tolist() == [0, 0, 0, 1, 1] and reg[:, 1].tolist() == [0, 1, 0, 1, 1]
    rejects = lambda f: (lambda: (f(), False))() if False else _raises(f)
    cfg = load_config(os.path.join(_ROOT, "configs", "glo_nca_cascade_brats2021.yaml"), ["data.et_labels=[3]"])
    with tempfile.TemporaryDirectory() as out:
        ds, _, _, _ = build(cfg, out, out, ([], [], []))
        configured = ds._labels_to_regions(np.array([0, 3]))[:, 2].tolist() == [0, 1]
        post = rejects(lambda: ds._labels_to_regions(np.array([0, 4])))
    check("labels: configured ET labels used by build(); unknown labels rejected",
          ok and configured and post and rejects(lambda: labels_to_regions(np.array([0, 5])))
          and rejects(lambda: labels_to_regions(np.array([0.0, 1.5]))))


def _raises(fn):
    r"""True when fn raises ValueError."""
    try:
        fn()
    except ValueError:
        return True
    return False


def test_saved_split():
    r"""A resumed run reads split.json instead of rebuilding the split from the folders."""
    import json
    from glo_nca.trainer import _load_split
    cfg = load_config(os.path.join(_ROOT, "configs", "glo_nca_cascade_brats2021.yaml"))
    with tempfile.TemporaryDirectory() as root:
        data, out = os.path.join(root, "data"), os.path.join(root, "run")
        for i in range(10):
            os.makedirs(os.path.join(data, f"case{i}"))
            open(os.path.join(data, f"case{i}", "x.nii.gz"), "w").close()
        os.makedirs(out)
        fresh = _load_split(cfg, data, out, resume=False)
        saved = {"train": ["a"], "validation": ["b"], "test": ["c"], "seed": 0}
        json.dump(saved, open(os.path.join(out, "split.json"), "w"))
        resumed = _load_split(cfg, data, out, resume=True)
    check("resume reuses the saved split.json",
          resumed == (["a"], ["b"], ["c"]) and sum(len(x) for x in fresh) == 10)


def test_cache_fingerprint():
    r"""The disk-cache key changes when a source scan changes."""
    import time
    ds = BraTSDataset.__new__(BraTSDataset)
    ds.size, ds.use_foreground_crop, ds.MODALITIES = (8, 8, 8), True, ["t1"]
    ds.ET_LABELS, ds.brain_mask, ds.cache_dir = (3, 4), False, "c"
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "case1"))
        for m in ("t1", "seg"):
            open(os.path.join(root, "case1", f"case1_{m}.nii.gz"), "wb").write(b"x")
        ds.images_path = root
        before = ds._cache_path("case1")
        time.sleep(1.1)
        open(os.path.join(root, "case1", "case1_t1.nii.gz"), "wb").write(b"xy")
        after = ds._cache_path("case1")
    check("cache key follows the source files", before != after)


def test_grad_checkpoint():
    r"""Activation checkpointing gives the same loss and gradients as the plain forward pass."""
    import torch
    from glo_nca.model import GLONCACell
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        cell = GLONCACell(24, 0.6, "cpu", 32, kernel_size=3, input_channels=4, use_se=True,
                            use_spatial=True)
        torch.nn.init.normal_(cell.fc_out.weight, std=0.05)
        cell.grad_checkpoint = ckpt
        torch.manual_seed(1)
        out = cell(torch.randn(1, 8, 8, 8, 24), steps=4, skip_prob=0.6)
        out[..., 4:7].sum().backward()
        grads.append(torch.cat([q.grad.flatten() for q in cell.parameters() if q.grad is not None]))
    check("activation checkpointing: identical gradients", torch.allclose(grads[0], grads[1], atol=1e-6),
          f"max diff {(grads[0] - grads[1]).abs().max().item():.1e}")


def test_bf16_and_case_seed():
    r"""bf16 autocast trains with a finite float32 loss; seeded inference passes repeat exactly."""
    import torch
    from glo_nca.evaluation import case_seed
    from glo_nca.model import GLONCACell
    cell = GLONCACell(24, 0.6, "cpu", 32, kernel_size=3, input_channels=4)
    x = torch.randn(1, 8, 8, 8, 24)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = cell(x, steps=2, skip_prob=0.6)
    loss = out.float()[..., 4:7].mean()
    loss.backward()
    ok_amp = loss.dtype == torch.float32 and bool(torch.isfinite(loss))
    runs = []
    for _ in range(2):
        torch.manual_seed(case_seed(7, "BraTS2021_00001"))
        with torch.no_grad():
            runs.append(cell(x, steps=3, skip_prob=0.6))
    ok_seed = torch.equal(runs[0], runs[1]) and case_seed(7, "a") != case_seed(7, "b")
    check("bf16 training step; seeded passes repeat exactly", ok_amp and ok_seed)


def test_run_info():
    r"""run_info.json records code, software and settings, one entry per start or resume."""
    import json
    from glo_nca.trainer import write_run_info
    cfg = load_config(os.path.join(_ROOT, "configs", "experiments", "brats2021_res96_fixed.yaml"))
    with tempfile.TemporaryDirectory() as out:
        write_run_info(out, cfg, False)
        write_run_info(out, cfg, True)
        info = json.load(open(os.path.join(out, "run_info.json")))
    check("run_info: commit, torch and seed per start/resume",
          len(info) == 2 and info[1]["resume"] and "torch" in info[0] and info[0]["seed"] == cfg.SEED,
          f"commit {str(info[0]['git_commit'])[:8]}")


def test_hd95_reference():
    r"""HD95 conventions: medpy matches medpy itself; max_directed matches a brute-force reference."""
    from scipy.ndimage import binary_erosion, generate_binary_structure
    from scipy.spatial import cKDTree
    from glo_nca.metrics import hd95_surface

    def directed(a, b, spacing):
        st = generate_binary_structure(3, 1)
        sa = np.argwhere(a & ~binary_erosion(a, st, border_value=0)) * spacing
        sb = np.argwhere(b & ~binary_erosion(b, st, border_value=0)) * spacing
        return cKDTree(sb).query(sa)[0], cKDTree(sa).query(sb)[0]

    try:
        from medpy.metric.binary import hd95 as medpy_hd95
    except ImportError:
        medpy_hd95 = None
    rng = np.random.default_rng(3)
    worst = {"medpy": 0.0, "max_directed": 0.0}
    differ = False
    for _ in range(6):
        a = np.zeros((40, 40, 30), bool); b = np.zeros_like(a)
        for m, n in ((a, 3), (b, 1)):               # asymmetric: different surface sizes
            for _ in range(n):
                c = rng.integers(8, 30, 3); r = rng.integers(3, 8, 3)
                m[c[0]-r[0]:c[0]+r[0], c[1]-r[1]:c[1]+r[1], max(c[2]-r[2], 0):c[2]+r[2]] = True
        spacing = np.array([1.0, 0.8, 1.5])             # anisotropic
        d_ab, d_ba = directed(a, b, spacing)
        pooled = np.percentile(np.concatenate([d_ab, d_ba]), 95)
        if medpy_hd95 is not None:
            pooled = medpy_hd95(a, b, voxelspacing=tuple(spacing), connectivity=1)
        maxd = max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))
        worst["medpy"] = max(worst["medpy"], abs(hd95_surface(a, b, tuple(spacing), "medpy") - pooled))
        worst["max_directed"] = max(worst["max_directed"],
                                    abs(hd95_surface(a, b, tuple(spacing), "max_directed") - maxd))
        differ |= abs(pooled - maxd) > 1e-3
    check("HD95 conventions match medpy and a directed reference", max(worst.values()) < 1e-6 and differ,
          f"reference {'medpy' if medpy_hd95 else 'brute force'}; max diff {max(worst.values()):.1e}")


def test_ablation_configs():
    r"""Ablation configs: matched recipe, intended differences and parameter counts."""
    from glo_nca.model import GLONCACell
    load = lambda n: load_config(os.path.join(_ROOT, "configs", "experiments", f"brats2021_res96_{n}.yaml"))
    full, plain, wide, single = load("fixed"), load("fixed_plain"), load("fixed_plain_wide"), load("fixed_single")
    se, spatial = load("fixed_se"), load("fixed_spatial")
    count = lambda cfg: sum(p.numel() for k in (7, 3) for p in GLONCACell(
        cfg.CHANNEL_N, cfg.SKIP_PROB, "cpu", cfg.HIDDEN, kernel_size=k, input_channels=len(cfg.MODALITIES),
        use_se=cfg.USE_SE, use_spatial=cfg.USE_SPATIAL, use_global=cfg.USE_GLOBAL).parameters())
    n = {k: count(cfg) for k, cfg in (("full", full), ("plain", plain), ("wide", wide), ("single", single))}
    same = lambda a, b, skip: ({k: v for k, v in vars(a).items() if k not in skip | {"NAME", "raw"}}
                               == {k: v for k, v in vars(b).items() if k not in skip | {"NAME", "raw"}})
    ok = (same(full, single, {"MODALITIES"}) and same(plain, wide, {"HIDDEN"})
          and n["full"] == 30138 and n["plain"] == 28128 and abs(n["wide"] - n["full"]) < 100
          and single.MODALITIES == ["t1ce"] and not full.LEGACY_PATCHIFY
          and same(full, se, {"USE_SPATIAL"}) and same(full, spatial, {"USE_SE"})
          and full.uses_final_scoring and full.FULL_RESOLUTION_EVAL and full.ENSEMBLE_N == 4)
    check("ablation configs: one difference each, matched capacity", ok,
          f"full {n['full']}, plain {n['plain']}, wide {n['wide']}, single {n['single']}")


def test_rng_untouched():
    r"""Seeded evaluation repeats exactly and leaves the caller's random state unchanged."""
    import torch
    from glo_nca.evaluation import predict_passes
    from glo_nca.trainer import build
    cfg = load_config(os.path.join(_ROOT, "configs", "glo_nca_cascade_brats2021.yaml"),
                    ["model.input_size=[[8,8,8],[16,16,16]]", "evaluation.case_seed=3"])
    with tempfile.TemporaryDirectory() as out:
        _, cells, agent, _ = build(cfg, out, out, ([], [], []))
    for m in cells:
        torch.nn.init.normal_(m.fc_out.weight, std=0.05)
    agent.exp.set_model_state("test")
    data = (["_c_0"], torch.randn(1, 16, 16, 16, 4), torch.zeros(1, 16, 16, 16, 3))
    torch.manual_seed(123)
    before = torch.get_rng_state()
    with torch.no_grad():
        first = predict_passes(agent, data, 2, False)[1]
        second = predict_passes(agent, data, 2, False)[1]
    check("seeded evaluation: repeatable, caller RNG unchanged",
          torch.equal(before, torch.get_rng_state()) and all(np.array_equal(x, y) for x, y in zip(first, second)))


def test_forward_default():
    r"""Cell.forward without skip_prob uses the configured skip probability."""
    import torch
    from glo_nca.model import GLONCACell
    cell = GLONCACell(8, 0.6, "cpu", 16, kernel_size=3, input_channels=2)
    torch.nn.init.normal_(cell.fc_out.weight, std=1.0)
    x = torch.randn(1, 24, 24, 24, 8)
    torch.manual_seed(0)
    with torch.no_grad():
        moved = (cell(x, steps=1) - x)[..., 2:].abs().sum(-1) > 0
    share = moved.float().mean().item()
    check("forward default: configured skip 0.6 -> about 40% of cells update", abs(share - 0.4) < 0.03,
          f"{share:.1%}")


def test_overlap_coverage():
    r"""The overlap check fails when the mapping does not cover every evaluated case."""
    import csv as _csv, json as _json
    from glo_nca.cli import analyze
    with tempfile.TemporaryDirectory() as d:
        _json.dump({"train": ["BraTS2021_00001"], "validation": [], "test": []}, open(f"{d}/split.json", "w"))
        with open(f"{d}/map.csv", "w", newline="") as fh:
            w = _csv.writer(fh); w.writerow(["BraTS2023", "BraTS2021"])
            w.writerow(["BraTS-GLI-01667-000", "BraTS2021_00009"])
        with open(f"{d}/cases.csv", "w", newline="") as fh:
            w = _csv.writer(fh); w.writerow(["BraTS-GLI-01667-000"]); w.writerow(["BraTS-GLI-01668-000"])
        incomplete = analyze.main(["overlap", "--run", d, "--mapping", f"{d}/map.csv", "--cases", f"{d}/cases.csv"])
        with open(f"{d}/cases.csv", "w", newline="") as fh:
            _csv.writer(fh).writerow(["BraTS-GLI-01667-000"])
        clean = analyze.main(["overlap", "--run", d, "--mapping", f"{d}/map.csv", "--cases", f"{d}/cases.csv"])
    check("overlap: incomplete mapping fails, covered and disjoint passes", incomplete == 2 and clean == 0)


def test_patient_split():
    r"""Patient-grouped split: identical to the folder shuffle for one scan per patient; scans kept together."""
    import random as _random
    from glo_nca.data import make_split
    with tempfile.TemporaryDirectory() as root:
        names = [f"BraTS2021_{i:05d}" for i in range(40)]
        for n in names:
            os.makedirs(os.path.join(root, n)); open(os.path.join(root, n, "x.nii.gz"), "w").close()
        old = sorted(names); _random.Random(42).shuffle(old)
        a, b = int(40 * 0.7), int(40 * 0.15)
        same = make_split(root, 42) == (old[:a], old[a:a + b], old[a + b:])
    with tempfile.TemporaryDirectory() as root:
        for i in range(20):
            for j in range(1 + (i % 2)):
                d = os.path.join(root, f"BraTS-GLI-{i:05d}-{j:03d}"); os.makedirs(d)
                open(os.path.join(d, "x.nii.gz"), "w").close()
        tr, va, te = make_split(root, 1, by_patient=True)
        side = {c[:15]: k for k, part in enumerate((tr, va, te)) for c in part}
        together = all(side[c[:15]] == k for k, part in enumerate((tr, va, te)) for c in part)
    check("split by patient: same split for one scan each, scans of a patient together", same and together)


def test_preflight():
    r"""Preflight flags a missing modality, mismatched affines and a duplicate scan; clean data passes."""
    import nibabel as nib
    from glo_nca.cli import analyze

    def case(root, name, vol, affine=np.eye(4), skip=None):
        d = os.path.join(root, name); os.makedirs(d)
        for m in ("t1", "t1ce", "t2", "flair", "seg"):
            if m == skip:
                continue
            data = (vol > 0.5).astype(np.int16) if m == "seg" else vol
            nib.save(nib.Nifti1Image(data, affine if m == "t2" else np.eye(4)), os.path.join(d, f"{name}_{m}.nii.gz"))

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as root:
        case(root, "BraTS2021_00001", rng.random((8, 8, 8)).astype(np.float32))
        case(root, "BraTS2021_00002", rng.random((8, 8, 8)).astype(np.float32))
        clean = analyze.main(["preflight", "--data-root", root])
        dup = rng.random((8, 8, 8)).astype(np.float32)
        case(root, "BraTS2021_00003", dup)
        case(root, "BraTS2021_00004", dup)                                       # duplicate scan
        case(root, "BraTS2021_00005", rng.random((8, 8, 8)).astype(np.float32), skip="flair")
        shifted = np.eye(4); shifted[0, 3] = 5.0
        case(root, "BraTS2021_00006", rng.random((8, 8, 8)).astype(np.float32), affine=shifted)
        out = os.path.join(root, "pre.csv")
        bad = analyze.main(["preflight", "--data-root", root, "--output", out])
        import csv as _csv
        problems = {r["case"]: r["problems"] for r in _csv.DictReader(open(out, newline=""))}
    check("preflight: clean data passes; duplicate, missing file and affine mismatch fail",
          clean == 0 and bad == 2 and "duplicate" in problems["BraTS2021_00004"]
          and "missing flair" in problems["BraTS2021_00005"] and "affines" in problems["BraTS2021_00006"])


def test_contrast_family():
    r"""Several contrasts: Holm correction across their primary p-values."""
    import csv as _csv, json as _json
    from glo_nca.cli import compare_runs
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        def write(name, shift):
            with open(os.path.join(d, name), "w", newline="") as fh:
                w = _csv.writer(fh); w.writerow(["case", "WT", "TC", "ET", "mean"])
                for i in range(40):
                    v = [float(x) for x in rng.random(3) * 0.1 + 0.7 + shift]
                    w.writerow([f"BraTS2021_{i:05d}", *v, sum(v) / 3])
        write("a.csv", 0.0); write("b.csv", 0.05); write("c.csv", 0.0)
        rc = compare_runs.main(["--contrast", f"big={d}/a.csv::{d}/b.csv", "--contrast", f"none={d}/a.csv::{d}/c.csv",
                                "--boot", "200", "--output", f"{d}/fam.json"])
        fam = _json.load(open(f"{d}/fam.json"))["contrasts"]
    check("contrast family: Holm across contrasts", rc == 0 and fam["big"]["p_holm_family"] >= fam["big"]["results"]["mean"]["p_wilcoxon"]
          and fam["big"]["p_holm_family"] < 0.05)


def test_code_identity():
    r"""The code identity reports the commit and whether the checkout has uncommitted changes."""
    from glo_nca.trainer import code_identity
    ident = code_identity()
    check("code identity: commit and dirty state", bool(ident["commit"]) and ident["dirty"] in (True, False),
          f"dirty {ident['dirty']}")


def test_resume_config_changes():
    r"""Resume detects config changes; only worker and cache settings pass without permission."""
    from glo_nca.trainer import RESUME_ALLOWED, config_changes
    old = {"training": {"epochs": 150, "workers": 4}, "loss": {"all_regions": False}}
    new = {"training": {"epochs": 150, "workers": 8}, "loss": {"all_regions": True}}
    ch = config_changes(old, new)
    blocked = sorted(k for k in ch if k not in RESUME_ALLOWED)
    check("resume: config changes found, training changes need permission",
          ch == {"loss.all_regions": [False, True], "training.workers": [4, 8]} and blocked == ["loss.all_regions"])


def test_legacy_checkpoint_names():
    r"""Checkpoints written with the old layer names load into the renamed cell with identical values."""
    import torch
    from glo_nca.model import LEGACY_LAYER_NAMES, GLONCACell
    cell = GLONCACell(24, 0.6, "cpu", 32, kernel_size=3, input_channels=4, use_se=True, use_spatial=True,
                      use_global=True)
    new = {k: torch.randn_like(v) for k, v in cell.state_dict().items()}
    reverse = {v: k for k, v in LEGACY_LAYER_NAMES.items()}
    def old_name(k):
        for n, o in reverse.items():
            if k.startswith(n):
                return o + k[len(n):]
        return k
    legacy = {old_name(k): v.clone() for k, v in new.items()}
    cell.load_state_dict(legacy)                                  # strict
    check("old checkpoint layer names load with identical values",
          any(k != old_name(k) for k in new) and all(torch.equal(cell.state_dict()[k], v) for k, v in new.items()))


if __name__ == "__main__":
    for t in (test_config, test_augmentation, test_components, test_tuning, test_full_resolution, test_early_stop, test_brats_empty, test_hd95_surface,
              test_review_options, test_all_region_loss, test_et_crop, test_global_block, test_decoder,
              test_with_weights, test_labels, test_saved_split, test_local_norm, test_bce_logits,
              test_brain_mask, test_nested_tuning_and_spacing, test_compare_runs,
              test_cache_fingerprint, test_grad_checkpoint, test_bf16_and_case_seed, test_run_info,
              test_hd95_reference, test_ablation_configs, test_rng_untouched, test_forward_default,
              test_overlap_coverage, test_patient_split, test_preflight, test_contrast_family,
              test_code_identity, test_resume_config_changes, test_legacy_checkpoint_names):
        t()
    print(f"\n  {sum(RESULTS)}/{len(RESULTS)} passed")
    raise SystemExit(0 if all(RESULTS) else 1)
