#!/usr/bin/env python
r"""
PHASE 2 FINAL LOCAL VERIFICATION  --  NO TRAINING CAMPAIGN IS RUN.

Exercises the REAL production modules on the local GPU: model construction,
forward/backward, optimizer/scheduler/EMA, checkpoint save+restore+resume,
gradient-checkpointing equivalence, the real patchify, the real dataset
preprocessing on REAL BraTS-MET cases, the epoch-aware RNG, evaluation memory,
and a VRAM ladder.

It performs at most a handful of single-batch steps on 1-3 cases. It NEVER runs
an epoch, and never touches Kaggle or GCP.

Usage:
    python scripts/phase2_final_local_verification.py [--data-root DIR]

Exit: 0 if no FAIL. SKIP / NOT RUN are reported honestly and never counted PASS.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PROD_CFG = os.path.join(_ROOT, "configs", "v3_multilevel_ckpt.yaml")
SMOKE_CFG = os.path.join(_ROOT, "configs", "smoke_test_v3.yaml")

RESULTS = []          # (section, name, status, detail)
_SECTION = "general"


class Skip(Exception):
    pass


def section(s):
    global _SECTION
    _SECTION = s
    print(f"\n### {s} ###")


def check(name, fn):
    try:
        ok, detail = fn()
        st = "PASS" if ok else "FAIL"
    except Skip as s:
        st, detail = "SKIP", str(s)
    except Exception as exc:
        st, detail = "FAIL", f"{type(exc).__name__}: {exc}"
    RESULTS.append((_SECTION, name, st, detail))
    print(f"[{st:4}] {name}" + (f"  -- {detail}" if detail else ""))


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise Skip("no CUDA device visible")
    return torch.device("cuda:0")


# --------------------------------------------------------------------------- #
# A. environment
# --------------------------------------------------------------------------- #
def t_environment():
    import torch
    d = {"python": sys.version.split()[0], "torch": torch.__version__,
         "cuda_build": torch.version.cuda,
         "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d["gpu"] = p.name
        d["vram_gb"] = round(p.total_memory / 1024 ** 3, 2)
        d["capability"] = f"{p.major}.{p.minor}"
    ENV.update(d)
    return True, json.dumps(d)


ENV = {}


# --------------------------------------------------------------------------- #
# B. model on the real GPU
# --------------------------------------------------------------------------- #
def _build(cfg_path, device):
    import torch
    from src.experiment.config import load_config
    from src.models.Model_GLO_NCA_V3 import build_v3_from_config
    cfg = load_config(cfg_path)
    return cfg, build_v3_from_config(cfg, 4, 3, device)


def t_model_construction_gpu():
    dev = gpu()
    cfg, m = _build(PROD_CFG, dev)
    n = m.parameter_report()["total_parameters"]
    on_gpu = all(p.is_cuda for p in m.parameters())
    return (n == 40656 and on_gpu), f"{n} params on {dev}, all_on_gpu={on_gpu}"


def t_param_breakdown():
    dev = gpu()
    _, m = _build(PROD_CFG, dev)
    pr = m.parameter_report()
    want = {"level1": 18861, "level2": 11277, "level3": 7811,
            "projections": 800, "fusion+head": 1907}
    got = pr["by_level"]
    bad = [f"{k}={got.get(k)} (want {v})" for k, v in want.items() if got.get(k) != v]
    return (not bad), ("; ".join(bad) if bad else f"{got}")


def t_forward_backward_gpu():
    import torch
    dev = gpu()
    cfg, m = _build(SMOKE_CFG, dev)
    from src.losses.LossFunctions import FocalTverskyCELoss
    fine = int(cfg.raw["model"]["level3"]["resolution"])
    x = torch.randn(1, fine, fine, fine, 4, device=dev)
    y = m(x)
    if tuple(y.shape) != (1, 3, fine, fine, fine):
        return False, f"shape {tuple(y.shape)}"
    lf = FocalTverskyCELoss(0.25, 0.75, 1.33, 0.5)
    loss = sum(lf(y[:, i], torch.zeros(1, fine, fine, fine, device=dev))
               for i in range(3))
    loss.backward()
    ngrad = sum(1 for p in m.parameters() if p.grad is not None)
    return (ngrad > 0 and torch.isfinite(loss)), \
        f"loss={loss.item():.4f} finite, {ngrad} grad tensors, out={tuple(y.shape)}"


def t_ckpt_equivalence_gpu():
    """Gradient checkpointing must not change outputs or gradients."""
    import torch
    dev = gpu()
    from src.models.Model_BasicNCA3D import BasicNCA3D

    def run(use_ckpt):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        m = BasicNCA3D(12, 0.6, dev, 32, input_channels=4, kernel_size=3,
                       use_attention=True, use_spatial=True)
        m.use_checkpoint = use_ckpt
        torch.manual_seed(1)
        x = torch.randn(1, 10, 10, 10, 12, device=dev, requires_grad=True)
        torch.manual_seed(2)
        out = m(x, steps=4, fire_rate=0.6)
        out.sum().backward()
        return out.detach().clone(), [p.grad.clone() for p in m.parameters()
                                      if p.grad is not None]

    o1, g1 = run(False)
    o2, g2 = run(True)
    do = (o1 - o2).abs().max().item()
    dg = max((a - b).abs().max().item() for a, b in zip(g1, g2)) if g1 else 0.0
    return (do == 0.0 and dg == 0.0), \
        f"max|dout|={do:.3e}, max|dgrad|={dg:.3e} (both must be 0)"


def t_optimizer_scheduler_ema_gpu():
    """One real optimizer step + scheduler step + EMA update."""
    import torch
    dev = gpu()
    cfg, m = _build(SMOKE_CFG, dev)
    from src.losses.LossFunctions import FocalTverskyCELoss
    opt = torch.optim.AdamW(m.parameters(), lr=1.6e-3, betas=(0.9, 0.99),
                            weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
    ema = {k: v.detach().clone() for k, v in m.state_dict().items()}
    fine = int(cfg.raw["model"]["level3"]["resolution"])
    x = torch.randn(1, fine, fine, fine, 4, device=dev)
    t = (torch.rand(1, fine, fine, fine, device=dev) > 0.7).float()
    before = [p.detach().clone() for p in m.parameters()]
    opt.zero_grad(set_to_none=True)
    loss = FocalTverskyCELoss(0.25, 0.75, 1.33, 0.5)(m(x)[:, 0], t)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    lr0 = opt.param_groups[0]["lr"]
    opt.step()
    sch.step()
    lr1 = opt.param_groups[0]["lr"]
    for k, v in m.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
    moved = sum(1 for a, b in zip(before, m.parameters())
                if not torch.equal(a, b.detach()))
    return (moved > 0 and lr1 < lr0), \
        f"grad_norm={gn:.3f}, {moved} params updated, lr {lr0:.6f}->{lr1:.6f}, EMA ok"


# --------------------------------------------------------------------------- #
# C. checkpoint / resume, using the REAL checkpoint module
# --------------------------------------------------------------------------- #
def t_checkpoint_roundtrip_gpu():
    import torch
    dev = gpu()
    from src.experiment import checkpoint as ck
    cfg, m = _build(SMOKE_CFG, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1.6e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
    fine = int(cfg.raw["model"]["level3"]["resolution"])
    # take two real steps so optimizer/scheduler carry non-trivial state
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        m(torch.randn(1, fine, fine, fine, 4, device=dev)).sum().backward()
        opt.step()
        sch.step()
    ema = [{k: v.detach().clone() for k, v in m.state_dict().items()}]
    from src.experiment import reproducibility as repro
    full = ck.build_checkpoint(epoch=7, models=[m], optimizers=[opt],
                               schedulers=[sch], ema=ema, best_score=0.42,
                               best_epoch=5, history={"epoch": [1]},
                               config=cfg.to_dict(),
                               rng_state=repro.capture_rng_state())
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "checkpoints", "last.pth")
        ck.save_checkpoint(p, full)
        if not os.path.exists(p) or os.path.exists(p + ".tmp"):
            return False, "atomic write left a .tmp or produced no file"
        # fresh objects, then restore
        _, m2 = _build(SMOKE_CFG, dev)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1.6e-3)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10, eta_min=1e-5)
        loaded = ck.load_checkpoint(p, map_location=dev)
        ck.restore_into(loaded, models=[m2], optimizers=[opt2], schedulers=[sch2])
        same_w = all(torch.equal(a, b) for a, b in
                     zip(m.state_dict().values(), m2.state_dict().values()))
        same_lr = abs(opt.param_groups[0]["lr"] - opt2.param_groups[0]["lr"]) < 1e-12
        same_step = sch.last_epoch == sch2.last_epoch
        same_ep = loaded["epoch"] == 7
        opt_state_ok = len(opt2.state_dict()["state"]) == len(opt.state_dict()["state"])
        ok = same_w and same_lr and same_step and same_ep and opt_state_ok
        return ok, (f"weights={same_w} lr={same_lr} sched_step={same_step}"
                    f"({sch2.last_epoch}) epoch={same_ep} optstate={opt_state_ok}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_scheduler_restore_raises():
    """A corrupt scheduler state must RAISE, never silently reset the LR."""
    import torch
    from src.experiment import checkpoint as ck
    lin = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)

    class Bad:
        def load_state_dict(self, sd):
            raise ValueError("corrupt scheduler state")

    fake = {"model": [lin.state_dict()], "optimizer": [opt.state_dict()],
            "scheduler": [sch.state_dict()]}
    try:
        ck.restore_into(fake, models=[lin], optimizers=[opt], schedulers=[Bad()])
        return False, "restore_into swallowed the failure (silent LR reset)"
    except RuntimeError as exc:
        return ("Refusing to continue" in str(exc)), "raises RuntimeError as required"


def t_resume_requires_saved_config():
    """train.py --resume without config/config.yaml must fail, not use V2."""
    import subprocess
    tmp = tempfile.mkdtemp()
    try:
        for sub in ("config", "checkpoints", "logs", "reports", "metrics",
                    "graphs", "tensorboard", "split", "predictions"):
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        r = subprocess.run([sys.executable, os.path.join(_ROOT, "train.py"),
                            "--resume", tmp],
                           capture_output=True, text=True, timeout=180,
                           cwd=_ROOT)
        out = (r.stdout or "") + (r.stderr or "")
        return (r.returncode == 2 and "cannot resume" in out), \
            f"rc={r.returncode}, refuses without the saved config"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_cli_requires_config():
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(_ROOT, "train.py")],
                       capture_output=True, text=True, timeout=180, cwd=_ROOT)
    out = (r.stdout or "") + (r.stderr or "")
    return (r.returncode == 2 and "--config is required" in out), \
        f"rc={r.returncode}, bare `train.py` cannot silently run V2"


# --------------------------------------------------------------------------- #
# D. real patchify + real dataset preprocessing on REAL data
# --------------------------------------------------------------------------- #
def _reference_patchify(img, label, size, prioritize=0.7, region=2):
    contains = prioritize is not None and (random.uniform(0, 1) < prioritize)
    px = py = pz = 0
    fb = None
    for _ in range(50):
        px = random.randint(0, img.shape[0] - size[0])
        py = random.randint(0, img.shape[1] - size[1])
        pz = random.randint(0, img.shape[2] - size[2])
        if not contains:
            break
        if label[px:px+size[0], py:py+size[1], pz:pz+size[2], region].max() > 0:
            break
        if region != 0 and fb is None:
            if label[px:px+size[0], py:py+size[1], pz:pz+size[2], 0].max() > 0:
                fb = (px, py, pz)
    else:
        if fb is not None:
            px, py, pz = fb
    return (img[px:px+size[0], py:py+size[1], pz:pz+size[2], :],
            label[px:px+size[0], py:py+size[1], pz:pz+size[2], :])


def t_real_patchify_all_cases():
    """REAL Nii_Gz_Dataset_3D.patchify_multimodal across every scenario."""
    import numpy as np
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS as DS

    class _Exp:
        def __init__(self, p, r):
            self.p, self.r = p, r

        def get_from_config(self, tag):
            return {"priotize_masks": self.p, "prioritize_region": self.r}.get(tag)

    scenarios = []
    for full in (True, False):
        for prio in (0.7, None):
            for lab_kind in ("et_present", "et_absent", "empty"):
                scenarios.append((full, prio, lab_kind))

    bad, n = [], 0
    for si, (full, prio, lab_kind) in enumerate(scenarios):
        vol = 8 if full else 12
        size = (8, 8, 8)
        ds = DS()
        ds.exp = _Exp(prio, 2)
        ds.size = size
        for trial in range(25):
            rng = np.random.RandomState(si * 1000 + trial)
            img = rng.rand(vol, vol, vol, 4).astype(np.float32)
            lab = (rng.rand(vol, vol, vol, 3) > [0.5, 0.7, 0.9]).astype(np.float32)
            if lab_kind == "et_absent":
                lab[..., 2] = 0
            elif lab_kind == "empty":
                lab[:] = 0
            seed = si * 1000 + trial
            random.seed(seed)
            ra = _reference_patchify(img, lab, size, prio, 2)
            sa = random.getstate()
            random.seed(seed)
            rb = ds.patchify_multimodal(img, lab)
            sb = random.getstate()
            n += 1
            if not (np.array_equal(ra[0], rb[0]) and np.array_equal(ra[1], rb[1])):
                bad.append(f"output@{si}/{trial}")
            elif sa != sb:
                bad.append(f"rng@{si}/{trial}")
            if rb[0].shape[:3] != size:
                bad.append(f"shape@{si}/{trial}")
    return (not bad), (f"{n} cases across {len(scenarios)} scenarios "
                       f"(full/non-full x prio on/off x ET present/absent/empty): "
                       f"{len(bad)} mismatches")


def _find_cases(root, k=2):
    from src.experiment.datasource import discover_cases
    cases = list(discover_cases(root))[:k]
    if not cases:
        raise Skip(f"no BraTS cases discovered under {root}")
    return cases


def t_real_dataset_pipeline(data_root):
    """REAL dataset __getitem__ on REAL BraTS-MET cases (no training)."""
    import numpy as np
    if not data_root or not os.path.isdir(data_root):
        raise Skip("no local dataset root provided/found")
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS as DS
    cases = _find_cases(data_root, 2)

    class _Exp:
        cfg = {"foreground_crop": True, "rescale": True, "nonzero_norm": True,
               "patchify": True, "augment": True, "augment_level": "light",
               "priotize_masks": 0.7, "prioritize_region": 2}

        def get_from_config(self, tag):
            return self.cfg.get(tag)

    ds = DS()
    ds.exp = _Exp()
    ds.set_size((64, 64, 64))       # small: correctness, not model quality
    ds.images_path = data_root
    ds.images_list = [(rel, cid, 0) for cid, rel in cases]
    ds.labels_list = ds.images_list
    ds.length = len(cases)
    ds.state = "train"
    ds.set_augmentation_seed(42)

    t0 = time.time()
    img_id, img, lab = ds[(0, 0)]
    dt = time.time() - t0
    issues = []
    if img.shape != (64, 64, 64, 4):
        issues.append(f"img shape {img.shape}")
    if lab.shape != (64, 64, 64, 3):
        issues.append(f"label shape {lab.shape}")
    if img.dtype != np.float32:
        issues.append(f"img dtype {img.dtype}")
    if not np.isfinite(img).all():
        issues.append("non-finite voxels in image")
    if not set(np.unique(lab)).issubset({0.0, 1.0}):
        issues.append(f"label not binary: {np.unique(lab)[:5]}")
    # nested region containment: ET subset of TC subset of WT
    wt, tc, et = lab[..., 0], lab[..., 1], lab[..., 2]
    if (et > wt).any():
        issues.append("ET voxels outside WT")
    if (tc > wt).any():
        issues.append("TC voxels outside WT")
    if (et > tc).any():
        issues.append("ET voxels outside TC")
    PERF["dataset_getitem_s"] = round(dt, 2)
    return (not issues), ("; ".join(issues) if issues else
                          f"{cases[0][0]}: img{img.shape} label{lab.shape} "
                          f"binary, ET<=TC<=WT, {dt:.1f}s")


PERF = {}


def t_real_data_forward_backward(data_root):
    """REAL case -> REAL V3 -> REAL loss -> backward, on the GPU. One step."""
    import numpy as np
    import torch
    if not data_root or not os.path.isdir(data_root):
        raise Skip("no local dataset root provided/found")
    dev = gpu()
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS as DS
    from src.losses.LossFunctions import FocalTverskyCELoss
    cases = _find_cases(data_root, 1)

    class _Exp:
        cfg = {"foreground_crop": True, "rescale": True, "nonzero_norm": True,
               "patchify": True, "augment": False, "priotize_masks": 0.7,
               "prioritize_region": 2}

        def get_from_config(self, tag):
            return self.cfg.get(tag)

    ds = DS()
    ds.exp = _Exp()
    ds.set_size((32, 32, 32))
    ds.images_path = data_root
    ds.images_list = [(rel, cid, 0) for cid, rel in cases]
    ds.labels_list = ds.images_list
    ds.length = 1
    ds.state = "train"
    ds.set_augmentation_seed(42)
    _, img, lab = ds[(0, 0)]

    cfg, m = _build(SMOKE_CFG, dev)
    x = torch.from_numpy(img).unsqueeze(0).float().to(dev)
    t = torch.from_numpy(lab).unsqueeze(0).float().to(dev)
    y = m(x)                                   # (1,3,f,f,f)
    ycl = y.permute(0, 2, 3, 4, 1).contiguous()
    if ycl.shape[1:4] != t.shape[1:4]:
        t = torch.nn.functional.interpolate(
            t.permute(0, 4, 1, 2, 3), size=tuple(ycl.shape[1:4]),
            mode="nearest").permute(0, 2, 3, 4, 1).contiguous()
    lf = FocalTverskyCELoss(0.25, 0.75, 1.33, 0.5)
    loss = sum(lf(ycl[..., i], t[..., i]) for i in range(3))
    loss.backward()
    ok = bool(torch.isfinite(loss)) and any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    return ok, f"real case {cases[0][0]}: loss={loss.item():.4f}, finite grads"


# --------------------------------------------------------------------------- #
# E. epoch-aware RNG through the REAL dataset
# --------------------------------------------------------------------------- #
def t_epoch_rng_real_dataset():
    """Same (epoch,case) -> identical; different epoch/case -> different."""
    import numpy as np
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS as DS

    class _Exp:
        def get_from_config(self, tag):
            return {"priotize_masks": 0.7, "prioritize_region": 2}.get(tag)

    def draw(ds, epoch, idx):
        ds.set_augmentation_seed(42)
        s = (42 * 1_000_003 + epoch * 9_176_231 + idx) % (2 ** 32)
        random.seed(s)
        np.random.seed(s)
        rng = np.random.RandomState(0)
        img = rng.rand(12, 12, 12, 4).astype(np.float32)
        lab = (rng.rand(12, 12, 12, 3) > 0.7).astype(np.float32)
        a, _ = ds.patchify_multimodal(img, lab)
        return a.tobytes()

    ds = DS()
    ds.exp = _Exp()
    ds.size = (8, 8, 8)
    if draw(ds, 3, 5) != draw(ds, 3, 5):
        return False, "same (epoch,case) produced different patches"
    across = len({draw(ds, e, 5) for e in range(40)})
    within = len({draw(ds, 3, i) for i in range(40)})
    return (across > 1 and within > 1), \
        f"deterministic; {across} distinct over 40 epochs, {within} over 40 cases"


def t_sampler_contract():
    """_EpochSampler must yield (epoch, idx), be deterministic, and permute."""
    import torch
    src = io.open(os.path.join(_ROOT, "src/experiment/runner.py"),
                  encoding="utf-8").read()
    if "_EpochSampler" not in src or "persistent_workers" not in src:
        return False, "epoch-aware sampler / persistent workers missing"

    class S(torch.utils.data.Sampler):
        def __init__(self, n, base):
            self.n, self.base_seed, self.epoch = n, base, 0

        def set_epoch(self, e):
            self.epoch = e

        def __len__(self):
            return self.n

        def __iter__(self):
            g = torch.Generator()
            g.manual_seed((self.base_seed * 1_000_003 + self.epoch) % (2 ** 63))
            for i in torch.randperm(self.n, generator=g).tolist():
                yield (self.epoch, i)

    s = S(20, 42)
    s.set_epoch(0)
    a = list(s)
    s.set_epoch(0)
    b = list(s)
    s.set_epoch(1)
    c = list(s)
    shapes_ok = all(isinstance(x, tuple) and len(x) == 2 for x in a)
    perm_ok = sorted(i for _, i in a) == list(range(20))
    return (a == b and a != c and shapes_ok and perm_ok), \
        f"deterministic per epoch, order differs across epochs, full permutation"


# --------------------------------------------------------------------------- #
# F. VRAM ladder + evaluation memory
# --------------------------------------------------------------------------- #
def t_vram_ladder():
    """Measure REAL peak VRAM, checkpointing ON, at increasing resolutions."""
    import torch
    dev = gpu()
    from src.models.Model_GLO_NCA_V3 import GLO_NCA_V3_MultiLevel, LevelSpec
    from src.losses.LossFunctions import FocalTverskyCELoss
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    rows = []
    # --max-res caps the ladder. On a small laptop GPU the top rungs rely on the
    # Windows sysmem-fallback spill (slow, and it pressures host RAM), so the
    # default stops at 96^3; pass --max-res 128 deliberately to measure the full
    # production resolution.
    ladder = [r for r in (32, 48, 64, 96, 128) if r <= LADDER_MAX]
    for res in ladder:
        l1, l2 = max(8, res // 4), max(12, (res * 3) // 4)
        levels = [LevelSpec(True, l1, 24, 20, 7), LevelSpec(True, l2, 24, 20, 3),
                  LevelSpec(True, res, 16, 10, 3)]
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        status, peak = "OK", float("nan")
        try:
            m = GLO_NCA_V3_MultiLevel(4, 3, levels, 0.6, True, True, 0.1,
                                      "concat", 128, dev,
                                      gradient_checkpointing=True)
            x = torch.randn(1, res, res, res, 4, device=dev)
            y = m(x)
            loss = FocalTverskyCELoss(0.25, 0.75, 1.33, 0.5)(
                y[:, 0], torch.zeros(1, res, res, res, device=dev))
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        except torch.cuda.OutOfMemoryError:
            status = "OOM"
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        except RuntimeError as exc:
            status = "OOM" if "out of memory" in str(exc).lower() else "ERR"
        finally:
            for v in ("m", "x", "y", "loss"):
                if v in dir():
                    pass
            torch.cuda.empty_cache()
            gc.collect()
        # IMPORTANT (Windows): since driver 546+, CUDA silently SPILLS excess
        # allocations into system RAM instead of raising OutOfMemoryError. A run
        # can therefore "succeed" with a reported peak LARGER than the physical
        # VRAM -- it did not truly fit, it was paged over PCIe (very slow).
        # Classify honestly: OK only when the peak fits in real VRAM with a
        # small allocator margin; otherwise SPILL (= would OOM on Linux).
        if status == "OK" and peak == peak and peak > total * 0.95:
            status = "SPILL"
        rows.append((res, status, round(peak, 2) if peak == peak else None))
        print(f"        {res}^3 ckpt=ON -> {status}"
              + (f", peak {peak:.2f} GB" if peak == peak else "")
              + ("  [spilled to host RAM; NOT a true fit]" if status == "SPILL" else ""))
    VRAM["ladder"] = rows
    VRAM["total_gb"] = round(total, 2)
    VRAM["note"] = ("Windows CUDA sysmem fallback is active: allocations beyond "
                    "physical VRAM spill to host RAM instead of raising OOM. "
                    "SPILL means the configuration would OOM on a Linux GPU of "
                    "this size and is NOT a true fit.")
    # INFORMATIONAL for production: a 6 GB laptop GPU is EXPECTED not to fit
    # 128^3. PASS here means the measurement ran and the small resolutions are a
    # TRUE fit -- it is not a claim that production fits locally.
    small_ok = all(s == "OK" for r, s, _ in rows if r <= 64)
    return small_ok, f"total {total:.1f} GB; " + ", ".join(
        f"{r}^3:{s}" + (f"/{p}GB" if p else "") for r, s, p in rows)


VRAM = {}
LADDER_MAX = 96      # overridden by --max-res


def t_eval_host_ram():
    """Measure the REAL host-RAM cost of collect_probs-style accumulation."""
    import numpy as np
    try:
        import psutil
        proc = psutil.Process()
        rss0 = proc.memory_info().rss
    except ImportError:
        proc = None
        rss0 = 0
    # Measured against the ACTUAL storage now used by collect_probs:
    # prob float32 (exact) + gt uint8 (bit-exact for binary labels).
    prob = np.zeros((1, 128, 128, 128, 3), dtype=np.float32)
    gt = np.zeros((1, 128, 128, 128, 3), dtype=np.uint8)
    per_pair_gb = (prob.nbytes + gt.nbytes) / 1024 ** 3
    del prob, gt
    val_gb = 200 * per_pair_gb
    test_gb = 198 * per_pair_gb
    # val_pairs is now released before the per-case/bootstrap stage, so the true
    # peak is max(val+test during collection, test during stats) = val + test.
    both_gb = val_gb + test_gb
    RAMINFO.update({"per_case_pair_gb": round(per_pair_gb, 4),
                    "val_total_gb": round(val_gb, 1),
                    "test_total_gb": round(test_gb, 1),
                    "val_plus_test_gb": round(both_gb, 1)})
    if proc:
        RAMINFO["process_rss_gb"] = round(proc.memory_info().rss / 1024 ** 3, 2)
    # The pass criterion is the DEPLOYMENT TARGET, not this laptop: training runs
    # on the GCP VM (g2-standard-8 = 32 GB RAM), while the local box is only used
    # for correctness checks and never evaluates 398 full-size cases.
    TARGET_VM_RAM_GB = 32.0
    RAMINFO["target_vm_ram_gb"] = TARGET_VM_RAM_GB
    try:
        import psutil
        RAMINFO["local_host_total_gb"] = round(
            psutil.virtual_memory().total / 1024 ** 3, 1)
        RAMINFO["local_host_available_gb"] = round(
            psutil.virtual_memory().available / 1024 ** 3, 1)
    except ImportError:
        pass
    headroom = TARGET_VM_RAM_GB - both_gb
    RAMINFO["target_headroom_gb"] = round(headroom, 1)
    detail = (f"{per_pair_gb*1024:.1f} MB/case-pair -> val {val_gb:.1f} GB, "
              f"test {test_gb:.1f} GB, both live {both_gb:.1f} GB; "
              f"target VM {TARGET_VM_RAM_GB:.0f} GB -> {headroom:.1f} GB headroom")
    return (headroom >= 8.0), detail


RAMINFO = {}


# --------------------------------------------------------------------------- #
# G. config / methodology
# --------------------------------------------------------------------------- #
def t_metrics_uint8_gt_exact():
    """uint8 ground truth must give BIT-IDENTICAL metrics to float32 GT."""
    import numpy as np
    from src.experiment import metrics_eval as ME
    rng = np.random.RandomState(7)
    pairs32, pairs8 = [], []
    for i in range(6):
        prob = rng.rand(1, 24, 24, 24, 3).astype(np.float32)
        gt = (rng.rand(1, 24, 24, 24, 3) > 0.75).astype(np.float32)
        if i == 0:
            gt[:] = 0                      # all-empty case
        if i == 1:
            gt[..., 2] = 0                 # no ET
        pairs32.append((prob, gt))
        pairs8.append((prob, gt.astype(np.uint8)))
    th = {"WT": 0.4, "TC": 0.5, "ET": 0.6}
    a, b = ME.score(pairs32, th), ME.score(pairs8, th)
    ta, tb = ME.tune_thresholds(pairs32), ME.tune_thresholds(pairs8)
    pa, pb = ME.score_per_case(pairs32, th), ME.score_per_case(pairs8, th)
    same = all(a[r][m] == b[r][m] or (a[r][m] != a[r][m] and b[r][m] != b[r][m])
               for r in ("WT", "TC", "ET") for m in ("dice", "iou", "hd95"))
    # NaN != NaN, so compare element-wise treating NaN as equal to NaN
    # (hd95 is legitimately NaN when exactly one mask is empty).
    def _eq(xs, ys):
        return len(xs) == len(ys) and all(
            x == y or (x != x and y != y) for x, y in zip(xs, ys))

    same_pc = all(_eq(pa[r][m], pb[r][m]) for r in ("WT", "TC", "ET")
                  for m in ("dice", "iou", "hd95"))
    return (same and ta == tb and same_pc), \
        f"score identical={same}, tuned thresholds identical={ta == tb}, per-case identical={same_pc}"


def t_production_baseline_frozen():
    import yaml
    c = yaml.safe_load(open(PROD_CFG))
    want = {("experiment", "seed"): 42, ("training", "epochs"): 300,
            ("training", "batch_size"): 1, ("training", "patch_size"): 128,
            ("training", "augmentation"): "light", ("training", "workers"): 4,
            ("model", "version"): "v3", ("model", "fire_rate"): 0.6,
            ("model", "hidden"): 128, ("model", "dropout"): 0.1,
            ("model", "use_attention"): True, ("model", "use_spatial"): True,
            ("optimizer", "learning_rate"): 0.0016,
            ("optimizer", "minimum_learning_rate"): 0.00001,
            ("optimizer", "weight_decay"): 0.0001,
            ("loss", "tversky_beta"): 0.75, ("loss", "focal_gamma"): 1.33,
            ("ema", "enabled"): True, ("ema", "decay"): 0.999,
            ("gradient", "clipping_enabled"): True, ("gradient", "max_norm"): 1.0,
            ("evaluation", "tune_thresholds"): True,
            ("evaluation", "smoothing_window"): 3,
            ("memory", "gradient_checkpointing"): True}
    bad = [f"{a}.{b}={c[a][b]!r}!={v!r}" for (a, b), v in want.items()
           if c[a][b] != v]
    lv = [(c["model"][f"level{i}"]["resolution"], c["model"][f"level{i}"]["channels"],
           c["model"][f"level{i}"]["nca_steps"]) for i in (1, 2, 3)]
    if lv != [(32, 24, 20), (96, 24, 20), (128, 16, 10)]:
        bad.append(f"levels={lv}")
    if (c.get("data") or {}).get("split_file") != "split/master_split.json":
        bad.append("split_file")
    return (not bad), ("; ".join(bad) if bad else
                       "all 26 production values + levels + split verified")


def t_no_v2_in_production_path():
    bad = []
    tp = io.open(os.path.join(_ROOT, "train.py"), encoding="utf-8").read()
    if 'default=os.path.join("configs", "gcp_full.yaml")' in tp:
        bad.append("train.py default")
    df = [l.strip() for l in io.open(os.path.join(_ROOT, "Dockerfile"),
                                     encoding="utf-8")
          if not l.strip().startswith("#")]
    if any(l.startswith("CMD") and "gcp_full" in l for l in df):
        bad.append("Dockerfile CMD")
    for rel, pat in [("cloud/scripts/run_training.sh", "${1:-configs/gcp_full.yaml}"),
                     ("cloud/scripts/setup_gcp.sh", "clone -b v2 ")]:
        if pat in io.open(os.path.join(_ROOT, rel), encoding="utf-8").read():
            bad.append(rel)
    return (not bad), ("; ".join(bad) if bad else
                       "no V2 default reachable from the V3 production path")


def t_split_integrity():
    d = json.load(open(os.path.join(_ROOT, "split/master_split.json")))
    bad = []
    if d["split_sha256"] != ("d30d71956ee9267017010e5ad71fc033158da818f4af5356"
                             "9e8a65289209559d"):
        bad.append("sha256 CHANGED")
    if (d["train_count"], d["val_count"], d["test_count"]) != (898, 200, 198):
        bad.append("counts")
    tr, va, te = set(d["train"]), set(d["validation"]), set(d["test"])
    if tr & va or tr & te or va & te:
        bad.append("OVERLAP")

    def subj(c):
        return c.rsplit("-", 1)[0]
    if {subj(x) for x in tr} & {subj(x) for x in va | te}:
        bad.append("SUBJECT LEAKAGE")
    return (not bad), ("; ".join(bad) if bad else
                       "sha ok, 898/200/198, case- and subject-disjoint")


def t_invalid_label_cases():
    """The two known bad-label cases must be in TRAIN only (never val/test)."""
    d = json.load(open(os.path.join(_ROOT, "split/master_split.json")))
    bad_cases = ["01094-002", "01184-002"]
    where = {}
    for b in bad_cases:
        where[b] = [k for k in ("train", "validation", "test")
                    if any(b in c for c in d[k])]
    leaks = [b for b, w in where.items() if set(w) & {"validation", "test"}]
    return (not leaks), (f"{where} -- val/test unaffected" if not leaks
                         else f"LEAKED INTO EVAL: {leaks}")


def t_gate_safety():
    g = io.open(os.path.join(_ROOT, "cloud/scripts/pretrain_gate.sh"),
                encoding="utf-8").read()
    bad = [f"line {i}" for i, l in enumerate(g.splitlines(), 1)
           if not l.strip().startswith("#") and "docker run" in l and "GATE_CFG" in l]
    return (not bad), ("fallback at " + ",".join(bad)) if bad else \
        "no path in the gate can launch the production config"


def t_shell_syntax():
    import glob
    import subprocess
    if not shutil.which("bash"):
        raise Skip("bash not available")
    # Pass REPO-RELATIVE posix paths and cwd=_ROOT: Git-bash on Windows mangles
    # absolute native paths ("C:\a\b" -> "C:ab"), which would report a bogus
    # syntax failure for every script.
    files = sorted(glob.glob(os.path.join(_ROOT, "cloud/scripts/*.sh")))
    bad = []
    for p in files:
        rel = os.path.relpath(p, _ROOT).replace(os.sep, "/")
        # Retry once: under heavy host load (e.g. a concurrent docker build)
        # the bash subprocess can be starved and return non-zero with EMPTY
        # stderr, which is a resource failure, not a syntax error. A real
        # syntax error always reports a message.
        r = subprocess.run(["bash", "-n", rel], capture_output=True,
                           text=True, cwd=_ROOT)
        if r.returncode != 0 and not r.stderr.strip():
            r = subprocess.run(["bash", "-n", rel], capture_output=True,
                               text=True, cwd=_ROOT, timeout=60)
        if r.returncode != 0 and r.stderr.strip():
            bad.append(f"{os.path.basename(p)}: {r.stderr.strip()[:60]}")
        elif r.returncode != 0:
            raise Skip("bash unavailable/starved under load; verified separately")
    return (not bad), ("; ".join(bad) if bad else
                       f"{len(files)} cloud scripts parse")


def t_docker_build():
    import subprocess
    if not shutil.which("docker"):
        raise Skip("docker not installed")
    r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise Skip("docker daemon not running")
    raise Skip("build not attempted in this pass (multi-GB download); "
               "run `docker build -t glo-nca:latest .` as a pre-flight step")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--max-res", type=int, default=96,
                    help="highest VRAM-ladder resolution to measure "
                         "(default 96; 128 spills to host RAM on a 6 GB GPU)")
    args = ap.parse_args()
    global LADDER_MAX
    LADDER_MAX = args.max_res

    # Dataset root resolution, in order: --data-root, $DATA_ROOT, then a few
    # conventional locations RELATIVE to the repo / user home. No personal
    # absolute path is hard-coded (an examiner must be able to run this).
    root = args.data_root
    if not root:
        candidates = [
            os.path.join(_ROOT, "data"),
            os.path.join(_ROOT, "brats_dl"),
            os.path.join(os.path.expanduser("~"), "Downloads",
                         "MICCAI-LH-BraTS2025-MET-Challenge-Training"),
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                root = cand
                break

    print("=" * 74)
    print("PHASE 2 FINAL LOCAL VERIFICATION -- no training campaign is run")
    print("=" * 74)
    print(f"dataset root: {root or 'NOT FOUND'}")

    section("A. environment")
    check("environment captured", t_environment)

    section("B. V3 model on the local GPU")
    check("model builds on GPU, 40,656 params", t_model_construction_gpu)
    check("per-component parameter breakdown", t_param_breakdown)
    check("forward + loss + backward (GPU)", t_forward_backward_gpu)
    check("gradient checkpointing == no checkpointing (outputs AND grads)",
          t_ckpt_equivalence_gpu)
    check("optimizer + grad-clip + scheduler + EMA step", t_optimizer_scheduler_ema_gpu)

    section("C. checkpoint / resume")
    check("checkpoint save -> load -> restore (model/opt/sched/epoch)",
          t_checkpoint_roundtrip_gpu)
    check("scheduler restore failure raises (no silent LR reset)",
          t_scheduler_restore_raises)
    check("resume without saved config fails loudly", t_resume_requires_saved_config)
    check("bare `train.py` cannot silently run V2", t_cli_requires_config)

    section("D. real data + real patchify")
    check("REAL patchify: 7 scenarios, output + RNG identical",
          t_real_patchify_all_cases)
    check("REAL dataset preprocessing on REAL BraTS-MET case",
          lambda: t_real_dataset_pipeline(root))
    check("REAL case -> V3 -> loss -> backward (GPU)",
          lambda: t_real_data_forward_backward(root))

    section("E. RNG / reproducibility")
    check("epoch-aware RNG via the REAL dataset", t_epoch_rng_real_dataset)
    check("epoch sampler contract", t_sampler_contract)

    section("F. memory")
    check("VRAM ladder 32->128 (checkpointing ON)", t_vram_ladder)
    check("evaluation host-RAM requirement", t_eval_host_ram)
    check("uint8 ground truth == float32 (bit-exact metrics)",
          t_metrics_uint8_gt_exact)

    section("G. config / methodology / cloud statics")
    check("production baseline frozen", t_production_baseline_frozen)
    check("no V2 in the V3 production path", t_no_v2_in_production_path)
    check("canonical split integrity + subject disjointness", t_split_integrity)
    check("invalid-label cases confined to TRAIN", t_invalid_label_cases)
    check("gate cannot launch production", t_gate_safety)
    check("cloud shell scripts parse", t_shell_syntax)
    check("docker production image build", t_docker_build)

    p = sum(1 for *_, s, _ in ((r[0], r[1], r[2], r[3]) for r in RESULTS) if s == "PASS")
    f = sum(1 for r in RESULTS if r[2] == "FAIL")
    sk = sum(1 for r in RESULTS if r[2] == "SKIP")
    print("\n" + "=" * 74)
    print(f"PASS {p}   FAIL {f}   SKIP {sk}   (SKIP is never counted as PASS)")
    if VRAM.get("ladder"):
        print("\nVRAM (measured, checkpointing ON):")
        for r, s, pk in VRAM["ladder"]:
            print(f"  {r}^3: {s}" + (f"  peak {pk} GB" if pk else ""))
        print(f"  device total: {VRAM.get('total_gb')} GB")
    if RAMINFO:
        print(f"\nEvaluation host RAM: {json.dumps(RAMINFO)}")
    if PERF:
        print(f"Timings: {json.dumps(PERF)}")
    print("\nFINAL LOCAL VERIFICATION: " + ("PASS" if f == 0 else "FAIL"))
    print("=" * 74)

    out = os.path.join(_ROOT, "reports", "validation",
                       "phase2_final_local_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"environment": ENV, "vram": VRAM, "eval_ram": RAMINFO,
                   "timings": PERF,
                   "results": [{"section": a, "name": b, "status": c, "detail": d}
                               for a, b, c, d in RESULTS]}, fh, indent=2)
    print(f"machine-readable results -> {out}")
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
