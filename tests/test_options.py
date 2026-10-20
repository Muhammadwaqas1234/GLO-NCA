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
          and fixed.NESTED_REGIONS and fixed.SURFACE_HD95 and abs(fixed.FIRE_RATE - 0.6) < 1e-9
          and fixed.EPOCHS == base.EPOCHS and fixed.INPUT_SIZE == base.INPUT_SIZE
          and fixed.SPLIT_SEED == base.SPLIT_SEED and fixed.SEED == base.SEED)
    check("plain control differs only in SE and spatial gate",
          not plain.USE_SE and not plain.USE_SPATIAL
          and {k: v for k, v in vars(plain).items() if k not in ("NAME", "USE_SE", "USE_SPATIAL", "raw")}
          == {k: v for k, v in vars(fixed).items() if k not in ("NAME", "USE_SE", "USE_SPATIAL", "raw")})


def test_all_region_loss():
    r"""With all_regions, positive ET logits on an ET-free crop are penalised and get a gradient."""
    import torch
    from glo_nca.agent import Agent_GLO_NCA
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
        agent = Agent_GLO_NCA.__new__(Agent_GLO_NCA)
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
    from glo_nca.agent import Agent_GLO_NCA
    t = torch.zeros((40, 40, 40, 3)); t[35:38, 2:4, 30:33, 2] = 1
    ok = True
    for _ in range(50):
        x, y, z = Agent_GLO_NCA._et_crop(t, (16, 16, 16))
        ok &= 0 <= x <= 24 and 0 <= y <= 24 and 0 <= z <= 24
        ok &= bool(t[x:x+16, y:y+16, z:z+16, 2].any())
    check("ET crop contains ET; none without ET",
          ok and Agent_GLO_NCA._et_crop(torch.zeros((40, 40, 40, 3)), (16, 16, 16)) is None)


def test_global_block():
    r"""The global block starts as identity and reaches every voxel; the spatial gate stays local."""
    import torch
    from glo_nca.model import GCSpatialBlock3D, GLO_NCA_Cell, GlobalContextBlock3D
    torch.manual_seed(0)
    x = torch.randn(1, 24, 20, 20, 20)
    far = x.clone(); far[..., 19, 19, 19] += 50.0                # perturb one distant voxel
    g = GlobalContextBlock3D(24)
    identity = torch.allclose(g(x), x)
    torch.nn.init.normal_(g.transform[-1].weight); torch.nn.init.normal_(g.key.weight)
    s = GCSpatialBlock3D(7)
    with torch.no_grad():
        g_far = (g(far) - g(x))[..., 0, 0, 0].abs().max().item()
        s_far = (s(far) - s(x))[..., 0, 0, 0].abs().max().item()
    count = lambda **kw: sum(p.numel() for k in (7, 3) for p in GLO_NCA_Cell(
        24, 0.6, "cpu", 128, kernel_size=k, input_channels=4, **kw).parameters())
    glo, plain = count(use_attention=True, use_spatial=True), count()
    check("global block: identity at start, reaches a distant voxel", identity and g_far > 1e-3 and s_far == 0,
          f"far effect {g_far:.3f} vs gate {s_far:.0f}")
    check("parameters: GLO-NCA 30,138, plain 28,128", glo == 30138 and plain == 28128,
          f"global +{count(use_attention=True, use_spatial=True, use_global=True) - glo}")


def test_decoder():
    r"""Exported label maps reload to exactly the nested masks the evaluation scores."""
    from glo_nca.evaluation import decode_regions, masks_to_label
    rng = np.random.default_rng(0)
    prob = rng.random((20, 20, 20, 3)).astype(np.float32)
    th, mc = {"WT": 0.5, "TC": 0.6, "ET": 0.7}, {"WT": 0, "TC": 0, "ET": 0}
    masks = decode_regions(prob, th, mc)
    back = Dataset_BraTS_Foreground._labels_to_regions(masks_to_label(masks, 3)) > 0
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
    from glo_nca.model import GLO_NCA_Cell
    torch.manual_seed(0)
    x = torch.randn(1, 16, 16, 16, 24)
    far = x.clone(); far[0, 15, 15, 15, 4:] += 5.0
    effect = {}
    for norm in ("batch", "layer"):
        cell = GLO_NCA_Cell(24, 0.0, "cpu", 32, kernel_size=3, input_channels=4, norm=norm)
        torch.nn.init.normal_(cell.fc1.weight, std=0.1)
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
    ds = Dataset_BraTS_Foreground.__new__(Dataset_BraTS_Foreground)
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
    r"""Paired comparison: identical runs give zero difference, patients are clustered by scan."""
    from glo_nca.cli.compare_runs import compare, patient_of
    rng = np.random.default_rng(0)
    a = {f"BraTS-GLI-{i:05d}-{j:03d}": {r: float(rng.random()) for r in ["WT", "TC", "ET"]}
         for i in range(30) for j in range(2)}
    b = {c: {r: v + 0.05 for r, v in s.items()} for c, s in a.items()}
    _, patients, same = compare(a, a, boot=200)
    _, _, up = compare(a, b, boot=200)
    check("compare_runs: zero for identical runs, +0.05 detected, 30 patients",
          same["mean"]["mean_diff"] == 0 and abs(up["mean"]["mean_diff"] - 0.05) < 1e-9
          and up["mean"]["ci95"][0] > 0.04 and len(patients) == 30
          and patient_of("BraTS2021_00001") == "BraTS2021_00001")


def test_labels():
    r"""Known BraTS labels map to nested regions; an unknown label stops loading."""
    reg = Dataset_BraTS_Foreground._labels_to_regions(np.array([0, 1, 2, 3, 4]))
    ok = reg[:, 2].tolist() == [0, 0, 0, 1, 1] and reg[:, 1].tolist() == [0, 1, 0, 1, 1]
    try:
        Dataset_BraTS_Foreground._labels_to_regions(np.array([0, 5])); raised = False
    except ValueError:
        raised = True

    class _Post(Dataset_BraTS_Foreground):
        ET_LABELS = (3,)
    try:
        _Post._labels_to_regions(np.array([0, 4])); post = False
    except ValueError:
        post = True   # label 4 is not silently merged into ET when only 3 is ET
    check("labels: 3/4 are ET by default, unknown labels rejected", ok and raised and post)


def test_saved_split():
    r"""A resumed run reads split.json instead of rebuilding the split from the folders."""
    import json
    from glo_nca.trainer import _load_split
    C = load_config(os.path.join(_ROOT, "configs", "glo_nca_cascade_brats2021.yaml"))
    with tempfile.TemporaryDirectory() as root:
        data, out = os.path.join(root, "data"), os.path.join(root, "run")
        for i in range(10):
            os.makedirs(os.path.join(data, f"case{i}"))
            open(os.path.join(data, f"case{i}", "x.nii.gz"), "w").close()
        os.makedirs(out)
        fresh = _load_split(C, data, out, resume=False)
        saved = {"train": ["a"], "validation": ["b"], "test": ["c"], "seed": 0}
        json.dump(saved, open(os.path.join(out, "split.json"), "w"))
        resumed = _load_split(C, data, out, resume=True)
    check("resume reuses the saved split.json",
          resumed == (["a"], ["b"], ["c"]) and sum(len(x) for x in fresh) == 10)


def test_cache_fingerprint():
    r"""The disk-cache key changes when a source scan changes."""
    import time
    ds = Dataset_BraTS_Foreground.__new__(Dataset_BraTS_Foreground)
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
    from glo_nca.model import GLO_NCA_Cell
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        cell = GLO_NCA_Cell(24, 0.6, "cpu", 32, kernel_size=3, input_channels=4, use_attention=True,
                            use_spatial=True)
        torch.nn.init.normal_(cell.fc1.weight, std=0.05)
        cell.grad_checkpoint = ckpt
        torch.manual_seed(1)
        out = cell(torch.randn(1, 8, 8, 8, 24), steps=4, fire_rate=0.6)
        out[..., 4:7].sum().backward()
        grads.append(torch.cat([q.grad.flatten() for q in cell.parameters() if q.grad is not None]))
    check("activation checkpointing: identical gradients", torch.allclose(grads[0], grads[1], atol=1e-6),
          f"max diff {(grads[0] - grads[1]).abs().max().item():.1e}")


def test_bf16_and_case_seed():
    r"""bf16 autocast trains with a finite float32 loss; seeded inference passes repeat exactly."""
    import torch
    from glo_nca.evaluation import case_seed
    from glo_nca.model import GLO_NCA_Cell
    cell = GLO_NCA_Cell(24, 0.6, "cpu", 32, kernel_size=3, input_channels=4)
    x = torch.randn(1, 8, 8, 8, 24)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = cell(x, steps=2, fire_rate=0.6)
    loss = out.float()[..., 4:7].mean()
    loss.backward()
    ok_amp = loss.dtype == torch.float32 and bool(torch.isfinite(loss))
    runs = []
    for _ in range(2):
        torch.manual_seed(case_seed(7, "BraTS2021_00001"))
        with torch.no_grad():
            runs.append(cell(x, steps=3, fire_rate=0.6))
    ok_seed = torch.equal(runs[0], runs[1]) and case_seed(7, "a") != case_seed(7, "b")
    check("bf16 training step; seeded passes repeat exactly", ok_amp and ok_seed)


def test_run_info():
    r"""run_info.json records code, software and settings, one entry per start or resume."""
    import json
    from glo_nca.trainer import write_run_info
    C = load_config(os.path.join(_ROOT, "configs", "experiments", "brats2021_res96_fixed.yaml"))
    with tempfile.TemporaryDirectory() as out:
        write_run_info(out, C, False)
        write_run_info(out, C, True)
        info = json.load(open(os.path.join(out, "run_info.json")))
    check("run_info: commit, torch and seed per start/resume",
          len(info) == 2 and info[1]["resume"] and "torch" in info[0] and info[0]["seed"] == C.SEED,
          f"commit {str(info[0]['git_commit'])[:8]}")


def test_hd95_reference():
    r"""Surface HD95 equals an independent brute-force computation (medpy definition) on random shapes."""
    from scipy.ndimage import binary_erosion, generate_binary_structure
    from scipy.spatial import cKDTree
    from glo_nca.metrics import hd95_surface

    def reference(a, b, spacing):
        # medpy: border = mask minus its erosion (connectivity 1); HD95 = max of both directed 95th percentiles.
        st = generate_binary_structure(3, 1)
        sa = np.argwhere(a & ~binary_erosion(a, st, border_value=0)) * spacing
        sb = np.argwhere(b & ~binary_erosion(b, st, border_value=0)) * spacing
        d_ab = cKDTree(sb).query(sa)[0]
        d_ba = cKDTree(sa).query(sb)[0]
        return max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))

    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(6):
        a = np.zeros((40, 40, 30), bool); b = np.zeros_like(a)
        for m in (a, b):
            for _ in range(3):
                c = rng.integers(8, 30, 3); r = rng.integers(3, 7, 3)
                m[c[0]-r[0]:c[0]+r[0], c[1]-r[1]:c[1]+r[1], max(c[2]-r[2], 0):c[2]+r[2]] = True
        spacing = np.array([1.0, 1.0, 1.5])
        worst = max(worst, abs(hd95_surface(a, b, spacing=tuple(spacing)) - reference(a, b, spacing)))
    check("surface HD95 matches an independent reference", worst < 1e-6, f"max diff {worst:.1e}")


def test_ablation_configs():
    r"""Ablation configs: matched recipe, intended differences and parameter counts."""
    import torch
    from glo_nca.model import GLO_NCA_Cell
    load = lambda n: load_config(os.path.join(_ROOT, "configs", "experiments", f"brats2021_res96_{n}.yaml"))
    full, plain, wide, single = load("fixed"), load("fixed_plain"), load("fixed_plain_wide"), load("fixed_single")
    count = lambda C: sum(p.numel() for k in (7, 3) for p in GLO_NCA_Cell(
        C.CHANNEL_N, C.FIRE_RATE, "cpu", C.HIDDEN, kernel_size=k, input_channels=len(C.MODALITIES),
        use_attention=C.USE_SE, use_spatial=C.USE_SPATIAL, use_global=C.USE_GLOBAL).parameters())
    n = {k: count(C) for k, C in (("full", full), ("plain", plain), ("wide", wide), ("single", single))}
    same = lambda a, b, skip: ({k: v for k, v in vars(a).items() if k not in skip | {"NAME", "raw"}}
                               == {k: v for k, v in vars(b).items() if k not in skip | {"NAME", "raw"}})
    ok = (same(full, single, {"MODALITIES"}) and same(plain, wide, {"HIDDEN"})
          and n["full"] == 30138 and n["plain"] == 28128 and abs(n["wide"] - n["full"]) < 100
          and single.MODALITIES == ["t1ce"] and not full.LEGACY_PATCHIFY)
    check("ablation configs: one difference each, matched capacity", ok,
          f"full {n['full']}, plain {n['plain']}, wide {n['wide']}, single {n['single']}")


if __name__ == "__main__":
    for t in (test_config, test_augmentation, test_components, test_tuning, test_full_resolution, test_early_stop, test_brats_empty, test_hd95_surface,
              test_review_options, test_all_region_loss, test_et_crop, test_global_block, test_decoder,
              test_with_weights, test_labels, test_saved_split, test_local_norm, test_bce_logits,
              test_brain_mask, test_nested_tuning_and_spacing, test_compare_runs,
              test_cache_fingerprint, test_grad_checkpoint, test_bf16_and_case_seed, test_run_info,
              test_hd95_reference, test_ablation_configs):
        t()
    print(f"\n  {sum(RESULTS)}/{len(RESULTS)} passed")
    raise SystemExit(0 if all(RESULTS) else 1)
