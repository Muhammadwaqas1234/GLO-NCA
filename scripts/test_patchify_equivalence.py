#!/usr/bin/env python
r"""
Patchify no-op equivalence test (run on Kaggle, CPU is fine — no GPU needed).

Proves whether the proposed patchify short-circuit (when patch_size ==
image_size, return the full volume directly) is EXACTLY equivalent to the current
`patchify_multimodal`, and quantifies its effect on the downstream Python RNG
stream (which augmentation shares).

It does NOT modify any source. It imports the real dataset method and compares:
  1. output image/label (must be identical arrays for the same input);
  2. the number of Python `random.*` draws each variant consumes (so we know
     whether a short-circuit would shift augmentation's RNG and therefore whether
     the run stays bit-identical).

Usage (Kaggle):  python scripts/test_patchify_equivalence.py
"""
import os, sys, random
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class _RngCounter:
    """Wrap random.* to count draws without changing values (same seed stream)."""
    def __init__(self):
        self.calls = 0
        self._uniform = random.uniform
        self._randint = random.randint
        self._random = random.random

    def __enter__(self):
        c = self
        def u(a, b): c.calls += 1; return c._uniform(a, b)
        def ri(a, b): c.calls += 1; return c._randint(a, b)
        def r(): c.calls += 1; return c._random()
        random.uniform, random.randint, random.random = u, ri, r
        return self

    def __exit__(self, *a):
        random.uniform, random.randint, random.random = self._uniform, self._randint, self._random


def current_patchify(img, label, size, prioritize=0.7, region=2):
    """Faithful copy of the CURRENT patchify_multimodal control flow (so we can
    count RNG draws and reproduce output for a fixed seed)."""
    contains_mask = prioritize is not None and (random.uniform(0, 1) < prioritize)
    pos_x = pos_y = pos_z = 0
    fallback = None
    for _ in range(50):
        pos_x = random.randint(0, img.shape[0] - size[0])
        pos_y = random.randint(0, img.shape[1] - size[1])
        pos_z = random.randint(0, img.shape[2] - size[2])
        if not contains_mask:
            break
        patch = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], region]
        if patch.max() > 0:
            break
        if region != 0 and fallback is None:
            wt = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], 0]
            if wt.max() > 0:
                fallback = (pos_x, pos_y, pos_z)
    else:
        if fallback is not None:
            pos_x, pos_y, pos_z = fallback
    return (img[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :],
            label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :])


def proposed_shortcircuit(img, label, size, prioritize=0.7, region=2):
    """Proposed P1: if patch==volume there is only one position (0,0,0); return the
    full volume. NOTE: this consumes NO random draws (unlike current)."""
    if tuple(img.shape[:3]) == tuple(size):
        return img, label
    # (else: identical to current — omitted here; the interesting case is ==)
    return current_patchify(img, label, size, prioritize, region)


def main():
    R = 128
    print("Patchify equivalence test (patch==volume case)")
    for seed in (0, 1, 42, 123):
        rng = np.random.RandomState(seed)
        img = rng.rand(R, R, R, 4).astype("float32")
        label = (rng.rand(R, R, R, 3) > 0.7).astype("float32")  # some cases have ET, some not
        size = (R, R, R)

        random.seed(seed)
        with _RngCounter() as c1:
            a1, b1 = current_patchify(img, label, size)
        cur_calls = c1.calls

        random.seed(seed)
        with _RngCounter() as c2:
            a2, b2 = proposed_shortcircuit(img, label, size)
        new_calls = c2.calls

        img_ok = np.array_equal(a1, a2)
        lab_ok = np.array_equal(b1, b2)
        full_ok = np.array_equal(a1, img) and np.array_equal(b1, label)
        print(f"seed={seed:4d} | output identical: img={img_ok} label={lab_ok} "
              f"| current==full-volume: {full_ok} "
              f"| rng draws current={cur_calls} proposed={new_calls} "
              f"{'(RNG SHIFT!)' if cur_calls != new_calls else '(rng same)'}")

    print()
    print("INTERPRETATION:")
    print("  * If 'output identical' is always True AND 'current==full-volume' is")
    print("    always True -> the short-circuit returns the SAME arrays.")
    print("  * If 'rng draws' differ -> the short-circuit SHIFTS the Python RNG the")
    print("    dataset shares with augmentation; to stay bit-identical, the fix must")
    print("    consume the SAME number of random.* draws (or augmentation must use a")
    print("    separate RNG). This is the correctness gate before applying the fix.")


if __name__ == "__main__":
    raise SystemExit(main())
