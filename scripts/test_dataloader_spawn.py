#!/usr/bin/env python
r"""
DataLoader / multiprocessing-spawn regression test.  NO TRAINING.

Guards the two latent production bugs fixed in commit 7476e18, both of which
were fatal for `workers > 0` under the SPAWN start method (the default on
Windows, and selectable anywhere):

  A. ``Experiment`` held a ``SummaryWriter`` (thread lock) reachable from the
     dataset  ->  "TypeError: cannot pickle '_thread.lock' object"
  B. ``_worker_init`` was a nested closure and ``_EpochSampler`` a local class
     ->  "AttributeError: Can't get local object 'run.<locals>._worker_init'"

Production runs `workers: 4`, so either bug meant production could not start on
a spawn platform.

This test uses the REAL production classes -- the real Dataset, the real
Experiment, the real ``_WorkerInit`` / ``_EpochSampler`` and a real
``torch.utils.data.DataLoader``. It deliberately does NOT mock them: a mock
would re-introduce exactly the "tests test a reimplementation" defect the Phase 1
audit flagged.

It exercises workers = 0, 1 and 2, and forces a genuine spawn context so the
result is not a fork-only claim.

Usage:  python scripts/test_dataloader_spawn.py
Exit:   0 if every executed check passed; SKIP/NOT RUN are never counted PASS.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS = []


class Skip(Exception):
    pass


def check(name, fn):
    try:
        ok, detail = fn()
        st = "PASS" if ok else "FAIL"
    except Skip as s:
        st, detail = "SKIP", str(s)
    except Exception as exc:
        st, detail = "FAIL", f"{type(exc).__name__}: {exc}"
    RESULTS.append((name, st, detail))
    print(f"[{st:4}] {name}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# A synthetic on-disk dataset: real NIfTI files, tiny volumes. The REAL Dataset
# class reads them, so the real loading/preprocessing path is exercised without
# needing the full BraTS cohort.
# --------------------------------------------------------------------------- #
def _make_cases(root, n=4, size=12):
    import nibabel as nib
    mods = ["t1n", "t1c", "t2w", "t2f"]
    ids = []
    for i in range(n):
        cid = f"BraTS-MET-9{i:04d}-000"
        d = os.path.join(root, cid)
        os.makedirs(d, exist_ok=True)
        rng = np.random.RandomState(i)
        for m in mods:
            vol = rng.rand(size, size, size).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, np.eye(4)),
                     os.path.join(d, f"{cid}-{m}.nii.gz"))
        seg = np.zeros((size, size, size), dtype=np.float32)
        seg[2:6, 2:6, 2:6] = 1      # NCR
        seg[4:8, 4:8, 4:8] = 2      # ED
        seg[5:7, 5:7, 5:7] = 4      # ET
        nib.save(nib.Nifti1Image(seg, np.eye(4)),
                 os.path.join(d, f"{cid}-seg.nii.gz"))
        ids.append(cid)
    return ids


class _Exp:
    """Minimal stand-in for the config surface the Dataset reads.

    NOTE: the REAL Experiment is exercised separately by
    ``t_real_experiment_picklable`` -- this one only supplies config values so a
    DataLoader can be built without a full experiment workspace."""

    CFG = {"foreground_crop": True, "rescale": True, "nonzero_norm": True,
           "patchify": True, "augment": True, "augment_level": "light",
           "priotize_masks": 0.7, "prioritize_region": 2}

    def get_from_config(self, tag):
        return self.CFG.get(tag)


class _AgentStub:
    """Module-level (therefore picklable) minimal agent for Experiment wiring."""

    def __init__(self):
        import torch
        self.model = [torch.nn.Linear(2, 2)]

    def set_exp(self, exp):
        self.exp = exp


def _build_dataset(root, ids, size=8):
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS as DS
    ds = DS()
    ds.exp = _Exp()
    ds.set_size((size, size, size))
    ds.images_path = root
    ds.images_list = [(c, c, 0) for c in ids]
    ds.labels_list = ds.images_list
    ds.length = len(ids)
    ds.state = "train"
    ds.set_augmentation_seed(42)
    return ds


def _loader(ds, workers, epoch=0, ctx=None):
    import torch
    from src.experiment.runner import _WorkerInit, _EpochSampler
    sampler = _EpochSampler(len(ds), 42)
    sampler.set_epoch(epoch)
    kw = dict(sampler=sampler, batch_size=1, num_workers=workers,
              worker_init_fn=_WorkerInit(42))
    if workers > 0:
        kw["persistent_workers"] = True      # production setting
        kw["prefetch_factor"] = 2
        if ctx is not None:
            kw["multiprocessing_context"] = ctx
    return torch.utils.data.DataLoader(ds, **kw)


def _drain(loader, limit=4):
    """Read `limit` batches, then SHUT THE LOADER DOWN deterministically.

    With ``persistent_workers=True`` the workers outlive the iterator, so
    abandoning it and letting the GC finalise it later produces a Windows
    teardown race:
        OSError: [WinError 6] The handle is invalid
    seen as a flaky "worker exited unexpectedly". That is a test-harness
    lifetime issue, not a defect in the dataset or sampler -- production drains
    its loader fully. We therefore release the iterator and the loader's workers
    explicitly so the result is deterministic rather than timing-dependent.
    """
    out = []
    it = iter(loader)
    try:
        for _ in range(limit):
            try:
                cid, img, lab = next(it)
            except StopIteration:
                break
            out.append((cid[0], np.asarray(img).copy(), np.asarray(lab).copy()))
    finally:
        del it
        shutdown = getattr(loader, "_iterator", None)
        if shutdown is not None and hasattr(shutdown, "_shutdown_workers"):
            try:
                shutdown._shutdown_workers()
            except Exception:
                pass
        loader._iterator = None
    return out


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def t_platform():
    return True, (f"default start method = {mp.get_start_method()}; "
                  f"spawn available = {'spawn' in mp.get_all_start_methods()}")


def t_worker_objects_picklable():
    """Bug B guard: both must be importable AND picklable at module scope."""
    from src.experiment.runner import _WorkerInit, _EpochSampler
    w = pickle.loads(pickle.dumps(_WorkerInit(42)))
    s = pickle.loads(pickle.dumps(_EpochSampler(10, 42)))
    s.set_epoch(3)
    idx = list(s)
    shapes_ok = all(isinstance(x, tuple) and len(x) == 2 for x in idx)
    perm_ok = sorted(i for _, i in idx) == list(range(10))
    return (shapes_ok and perm_ok and w.base_seed == 42), \
        "module-level, picklable, yields (epoch, index), full permutation"


def t_real_experiment_picklable():
    """Bug A guard: the REAL Experiment must survive pickling (writer dropped)."""
    import tempfile
    import torch
    from src.utils.Experiment import Experiment
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS as DS
    # Experiment discovers cases from img_path, so it needs a real cohort on
    # disk -- give it a tiny synthetic one rather than an empty directory.
    tmp = tempfile.mkdtemp(prefix="glonca_exp_")
    _make_cases(tmp, n=2, size=8)
    cfg = [{"img_path": tmp, "label_path": tmp, "model_path": tmp,
            "device": "cpu", "unlock_CPU": True, "optimizer": "adamw",
            "lr": 1e-3, "lr_gamma": 0.9999, "betas": (0.9, 0.99),
            "weight_decay": 1e-4, "save_interval": 10 ** 9,
            "evaluate_interval": 10 ** 9, "n_epoch": 1, "batch_size": 1,
            "batch_duplication": 1, "channel_n": 16, "inference_steps": 10,
            "cell_fire_rate": 0.6, "input_channels": 4, "output_channels": 3,
            "hidden_size": 64, "train_model": 0, "use_attention": True,
            "input_size": [[8, 8, 8]], "scale_factor": 2,
            "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True,
            "rescale": True, "foreground_crop": True, "nonzero_norm": True,
            "augment": False, "patchify": True, "priotize_masks": 0.7,
            "prioritize_region": 2}]

    ds = DS()
    agent = _AgentStub()
    exp = Experiment(cfg, ds, agent.model, agent)
    had_writer = getattr(exp, "writer", None) is not None
    exp2 = pickle.loads(pickle.dumps(exp))          # would raise before the fix
    return (had_writer and getattr(exp2, "writer", "missing") is None), \
        (f"real Experiment pickles; live writer present={had_writer}, "
         "dropped in the pickled copy")


def t_workers(root, ids, workers, ctx=None):
    ds = _build_dataset(root, ids)
    batches = _drain(_loader(ds, workers, ctx=ctx), limit=4)
    if len(batches) != 4:
        return False, f"expected 4 batches, got {len(batches)}"
    img = batches[0][1]
    lab = batches[0][2]
    if img.shape != (1, 8, 8, 8, 4) or lab.shape != (1, 8, 8, 8, 3):
        return False, f"shapes {img.shape} / {lab.shape}"
    # ET subset TC subset WT must hold through the worker path too
    wt, tc, et = lab[..., 0], lab[..., 1], lab[..., 2]
    if (et > tc).any() or (tc > wt).any():
        return False, "ET/TC/WT nesting violated"
    tag = f"spawn ctx, workers={workers}" if ctx is not None else f"workers={workers}"
    return True, f"{tag}: 4 batches, shapes ok, ET<=TC<=WT"


def t_determinism_same_epoch(root, ids, workers):
    """Same seed + same epoch must reproduce the same batches exactly."""
    a = _drain(_loader(_build_dataset(root, ids), workers, epoch=0), limit=4)
    b = _drain(_loader(_build_dataset(root, ids), workers, epoch=0), limit=4)
    same_ids = [x[0] for x in a] == [x[0] for x in b]
    same_img = all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    same_lab = all(np.array_equal(x[2], y[2]) for x, y in zip(a, b))
    return (same_ids and same_img and same_lab), \
        f"workers={workers}: ids={same_ids} images={same_img} labels={same_lab}"


def t_epoch_streams_differ(root, ids, workers):
    """Different epochs must NOT replay the same augmentation stream.

    This is the Phase 1 R-01 fix: before it, every epoch produced an identical
    augmentation sequence."""
    a = _drain(_loader(_build_dataset(root, ids), workers, epoch=0), limit=4)
    b = _drain(_loader(_build_dataset(root, ids), workers, epoch=1), limit=4)
    order_differs = [x[0] for x in a] != [x[0] for x in b]
    content_differs = any(not np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    return (order_differs or content_differs), \
        (f"workers={workers}: order differs={order_differs}, "
         f"content differs={content_differs}")


def t_case_streams_independent(root, ids, workers):
    """Within one epoch, different cases must get independent streams."""
    got = _drain(_loader(_build_dataset(root, ids), workers, epoch=0),
                 limit=min(4, len(ids)))
    uniq = {x[0] for x in got}
    distinct_imgs = len({x[1].tobytes() for x in got})
    return (len(uniq) == len(got) and distinct_imgs == len(got)), \
        f"workers={workers}: {len(uniq)} distinct cases, {distinct_imgs} distinct tensors"


def main():
    import tempfile
    print("=" * 74)
    print("DATALOADER / SPAWN REGRESSION  --  no training is performed")
    print("=" * 74)

    check("platform / start method", t_platform)
    check("bug B: _WorkerInit + _EpochSampler are module-level and picklable",
          t_worker_objects_picklable)
    check("bug A: real Experiment pickles (SummaryWriter dropped)",
          t_real_experiment_picklable)

    tmp = tempfile.mkdtemp(prefix="glonca_spawn_")
    try:
        ids = _make_cases(tmp, n=4, size=12)
        print(f"\n-- synthetic NIfTI cohort: {len(ids)} cases at {tmp} --\n")

        for w in (0, 1, 2):
            check(f"real DataLoader end-to-end, workers={w}",
                  lambda w=w: t_workers(tmp, ids, w))

        # Force a genuine spawn context so this is not a fork-only claim.
        if "spawn" in mp.get_all_start_methods():
            ctx = mp.get_context("spawn")
            for w in (1, 2):
                check(f"FORCED spawn context, workers={w}",
                      lambda w=w, c=ctx: t_workers(tmp, ids, w, ctx=c))
        else:
            check("FORCED spawn context",
                  lambda: (_ for _ in ()).throw(Skip("spawn unavailable")))

        for w in (0, 2):
            check(f"determinism: same seed+epoch reproduces, workers={w}",
                  lambda w=w: t_determinism_same_epoch(tmp, ids, w))
            check(f"epoch streams differ across epochs, workers={w}",
                  lambda w=w: t_epoch_streams_differ(tmp, ids, w))
            check(f"case streams independent within an epoch, workers={w}",
                  lambda w=w: t_case_streams_independent(tmp, ids, w))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    p = sum(1 for _, s, _ in RESULTS if s == "PASS")
    f = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    sk = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print("\n" + "=" * 74)
    print(f"PASS {p}   FAIL {f}   SKIP {sk}   (SKIP is never counted as PASS)")
    print("DATALOADER/SPAWN REGRESSION: " + ("PASS" if f == 0 else "FAIL"))
    print("=" * 74)
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
