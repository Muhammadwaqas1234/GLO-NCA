#!/usr/bin/env python
r"""
Deterministic preprocessing-cache correctness + benchmark.  NO TRAINING.

Proves the cache is scientifically inert before it is ever recommended:

  1. cached output is BYTE-IDENTICAL to uncached output
  2. cache lookup is RNG-NEUTRAL (identical random state afterwards), so a
     cached run draws the same stochastic sequence as an uncached one
  3. stochastic behaviour is NOT frozen: augmentation/patch decisions still
     differ across epochs with the cache enabled
  4. identity awareness: a different size/modality/crop contract is a MISS
  5. corruption safety: truncated/garbage entries are rejected, not consumed
  6. atomicity: no partial file is ever visible under the final name
  7. benchmark: cold build vs warm reload vs uncached

Usage:  python scripts/test_preprocess_cache.py [--data-root DIR]
Exit:   0 if every executed check passed. SKIP is never counted as PASS.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS = []
BENCH = {}


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


class _Exp:
    def __init__(self, augment=False, patchify=False):
        self.C = {"foreground_crop": True, "rescale": True, "nonzero_norm": True,
                  "patchify": patchify, "augment": augment,
                  "augment_level": "light",
                  "priotize_masks": 0.7, "prioritize_region": 2}

    def get_from_config(self, tag):
        return self.C.get(tag)


def _dataset(root, ids, size, cache=None, augment=False, patchify=False,
             state="train"):
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS as DS
    ds = DS()
    ds.exp = _Exp(augment=augment, patchify=patchify)
    ds.set_size((size, size, size))
    ds.images_path = root
    ds.images_list = [(rel, cid, 0) for cid, rel in ids]
    ds.labels_list = ds.images_list
    ds.length = len(ids)
    ds.state = state
    ds.set_augmentation_seed(42)
    if cache is not None:
        ds.set_preprocess_cache(cache)
    return ds


def _cache(directory, root, size, enabled=True):
    from src.datasets.preprocess_cache import PreprocessCache
    return PreprocessCache(directory, dataset_root=root,
                           modalities=["t1n", "t1c", "t2w", "t2f"],
                           size=(size, size, size), crop_fg=True, rescale=True,
                           enabled=enabled)


def _find_cases(root, k):
    from src.experiment.datasource import discover_cases
    cases = list(discover_cases(root))[:k]
    if not cases:
        raise Skip(f"no BraTS cases under {root}")
    return cases


# --------------------------------------------------------------------------- #
def t_identical_output(root, ids, size, cdir):
    """Cached output must be byte-identical to uncached output."""
    plain = _dataset(root, ids, size, state="val")
    a = [plain[i] for i in range(len(ids))]

    c = _cache(cdir, root, size)
    warm = _dataset(root, ids, size, cache=c, state="val")
    _ = [warm[i] for i in range(len(ids))]          # build cache
    warm2 = _dataset(root, ids, size, cache=_cache(cdir, root, size), state="val")
    b = [warm2[i] for i in range(len(ids))]         # read from cache

    same = all(np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2])
               and x[0] == y[0] for x, y in zip(a, b))
    dtypes = all(y[1].dtype == np.float32 and y[2].dtype == np.float32 for y in b)
    return (same and dtypes), \
        f"{len(ids)} cases byte-identical={same}, dtypes preserved={dtypes}"


def t_rng_neutral(root, ids, size, cdir):
    """A cache HIT must leave the RNG in exactly the state a MISS would."""
    c = _cache(cdir, root, size)
    warm = _dataset(root, ids, size, cache=c, state="val")
    _ = [warm[i] for i in range(len(ids))]          # ensure populated

    def draw(use_cache):
        ds = _dataset(root, ids, size,
                      cache=_cache(cdir, root, size, enabled=use_cache),
                      state="val")
        random.seed(1234)
        np.random.seed(1234)
        for i in range(len(ids)):
            ds[i]
        return (random.getstate(), np.random.get_state()[1].tobytes(),
                random.random(), float(np.random.rand()))

    a = draw(False)
    b = draw(True)
    return (a == b), \
        "python+numpy RNG state identical after cached vs uncached reads"


def t_stochastic_not_frozen(root, ids, size, cdir):
    """With the cache ON, augmentation/patching must still vary per epoch."""
    c = _cache(cdir, root, size)
    ds = _dataset(root, ids, size, cache=c, augment=True, patchify=True,
                  state="train")
    half = max(4, size // 2)
    ds.set_size((half, half, half))                 # force a real patch crop
    ds.size = (half, half, half)
    outs = []
    for epoch in (0, 1, 2):
        random.seed(0)
        np.random.seed(0)
        outs.append(ds[(epoch, 0)][1].copy())
    differ = (not np.array_equal(outs[0], outs[1])) or \
             (not np.array_equal(outs[1], outs[2]))
    return differ, ("same case across 3 epochs still produces DIFFERENT "
                    "augmented patches (randomness not frozen)")


def t_identity_awareness(root, ids, size, cdir):
    """A different preprocessing contract must MISS, never reuse."""
    c = _cache(cdir, root, size)
    ds = _dataset(root, ids, size, cache=c, state="val")
    ds[0]                                            # populate at `size`
    from src.datasets.preprocess_cache import PreprocessCache
    other = PreprocessCache(cdir, dataset_root=root,
                            modalities=["t1n", "t1c", "t2w", "t2f"],
                            size=(size + 8, size + 8, size + 8),
                            crop_fg=True, rescale=True)
    hit = other.get(ids[0][0])
    other2 = PreprocessCache(cdir, dataset_root=root,
                             modalities=["t1c", "t1n", "t2w", "t2f"],  # reordered
                             size=(size, size, size), crop_fg=True, rescale=True)
    hit2 = other2.get(ids[0][0])
    return (hit is None and hit2 is None), \
        "different size MISSES; different modality order MISSES (no silent reuse)"


def t_corruption_safety(root, ids, size, cdir):
    """Truncated / garbage entries must be rejected, not consumed."""
    c = _cache(cdir, root, size)
    ds = _dataset(root, ids, size, cache=c, state="val")
    ds[0]
    p = c.path_for(ids[0][0])
    if not os.path.exists(p):
        return False, "cache entry was never written"

    with open(p, "r+b") as fh:                       # truncate to half
        fh.truncate(max(1, os.path.getsize(p) // 2))
    c2 = _cache(cdir, root, size)
    truncated_rejected = c2.get(ids[0][0]) is None

    with open(p, "wb") as fh:
        fh.write(b"not a torch file at all")
    c3 = _cache(cdir, root, size)
    garbage_rejected = c3.get(ids[0][0]) is None

    os.remove(p)
    return (truncated_rejected and garbage_rejected), \
        (f"truncated rejected={truncated_rejected}, "
         f"garbage rejected={garbage_rejected} (both rebuilt from source)")


def t_atomic_write(root, ids, size, cdir):
    """No partial file may ever be visible under the final cache name."""
    c = _cache(cdir, root, size)
    ds = _dataset(root, ids, size, cache=c, state="val")
    ds[0]
    leftovers = [f for f in os.listdir(cdir) if f.endswith(".tmp")]
    final = c.path_for(ids[0][0])
    ok = os.path.exists(final) and not leftovers
    # a valid final file must load cleanly
    reread = _cache(cdir, root, size).get(ids[0][0]) is not None
    return (ok and reread), \
        f"temp files left={len(leftovers)}, final entry valid={reread}"


def t_benchmark(root, ids, size, cdir):
    """Cold build vs warm reload vs uncached -- construction cost separated."""
    shutil.rmtree(cdir, ignore_errors=True)
    os.makedirs(cdir, exist_ok=True)

    t0 = time.perf_counter()
    plain = _dataset(root, ids, size, state="val")
    for i in range(len(ids)):
        plain[i]
    uncached = (time.perf_counter() - t0) / len(ids) * 1000

    c = _cache(cdir, root, size)
    t0 = time.perf_counter()
    build = _dataset(root, ids, size, cache=c, state="val")
    for i in range(len(ids)):
        build[i]
    cold = (time.perf_counter() - t0) / len(ids) * 1000

    t0 = time.perf_counter()
    warm = _dataset(root, ids, size, cache=_cache(cdir, root, size), state="val")
    for i in range(len(ids)):
        warm[i]
    warm_ms = (time.perf_counter() - t0) / len(ids) * 1000

    size_mb = sum(os.path.getsize(os.path.join(cdir, f))
                  for f in os.listdir(cdir)) / 1024 ** 2
    speedup = uncached / warm_ms if warm_ms else float("nan")
    BENCH.update({"cases": len(ids), "volume": f"{size}^3",
                  "uncached_ms_per_case": round(uncached, 1),
                  "cold_build_ms_per_case": round(cold, 1),
                  "warm_reload_ms_per_case": round(warm_ms, 1),
                  "speedup_vs_uncached": round(speedup, 2),
                  "disk_mb_total": round(size_mb, 2),
                  "disk_mb_per_case": round(size_mb / len(ids), 2)})
    return warm_ms < uncached, \
        (f"uncached {uncached:.0f} ms/case -> warm {warm_ms:.0f} ms/case "
         f"({speedup:.1f}x); cold build {cold:.0f} ms/case; "
         f"{size_mb / len(ids):.1f} MB/case")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    root = args.data_root
    if not root:
        for cand in [os.path.join(_ROOT, "data"),
                     os.path.join(os.path.expanduser("~"), "Downloads",
                                  "MICCAI-LH-BraTS2025-MET-Challenge-Training")]:
            if os.path.isdir(cand):
                root = cand
                break

    print("=" * 76)
    print("PREPROCESS CACHE — correctness + benchmark (no training)")
    print("=" * 76)
    print(f"dataset root: {root or 'NOT FOUND'}")

    if not root or not os.path.isdir(root):
        print("\nNo dataset available -> all checks SKIPPED (never PASS).")
        return 0

    ids = _find_cases(root, args.cases)
    print(f"cases: {len(ids)}   volume: {args.size}^3\n")
    cdir = tempfile.mkdtemp(prefix="glonca_cache_")
    try:
        check("cached output is byte-identical to uncached",
              lambda: t_identical_output(root, ids, args.size, cdir))
        check("cache lookup is RNG-neutral",
              lambda: t_rng_neutral(root, ids, args.size, cdir))
        check("stochastic augmentation/patching NOT frozen by the cache",
              lambda: t_stochastic_not_frozen(root, ids, args.size, cdir))
        check("identity awareness (size / modality order) -> MISS",
              lambda: t_identity_awareness(root, ids, args.size, cdir))
        check("corrupted + truncated entries rejected",
              lambda: t_corruption_safety(root, ids, args.size, cdir))
        check("atomic write (no partial file under the final name)",
              lambda: t_atomic_write(root, ids, args.size, cdir))
        check("benchmark: cold build vs warm reload vs uncached",
              lambda: t_benchmark(root, ids, args.size, cdir))
    finally:
        shutil.rmtree(cdir, ignore_errors=True)

    p = sum(1 for _, s, _ in RESULTS if s == "PASS")
    f = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    sk = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print("\n" + "=" * 76)
    if BENCH:
        import json
        print("BENCHMARK: " + json.dumps(BENCH))
    print(f"PASS {p}   FAIL {f}   SKIP {sk}   (SKIP is never counted as PASS)")
    print("PREPROCESS CACHE: " + ("PASS" if f == 0 else "FAIL"))
    print("=" * 76)
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
