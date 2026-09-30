#!/usr/bin/env python
r"""Write a small synthetic BraTS-style dataset (brain, oedema, core, enhancing tumour) for tests."""
import argparse
import os

import nibabel as nib
import numpy as np

NAMING = {
    "met": (["t1n", "t1c", "t2w", "t2f"], 3),     # BraTS 2023+ names; ET is label 3
    "2021": (["t1", "t1ce", "t2", "flair"], 4),   # BraTS 2021 names; ET is label 4
}


def _ellipsoid(shape, centre, radii):
    r"""Boolean ellipsoid mask."""
    grid = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum(((g - c) / r) ** 2 for g, c, r in zip(grid, centre, radii)) <= 1.0


def make_case(rng, shape, et_label):
    r"""One case: four modalities and a segmentation with nested tumour regions."""
    centre = np.array(shape) / 2
    brain = _ellipsoid(shape, centre + rng.uniform(-2, 2, 3), np.array(shape) * rng.uniform(0.36, 0.42, 3))
    tc = centre + rng.uniform(-6, 6, 3)
    r = rng.uniform(4, 8)
    oedema = _ellipsoid(shape, tc, (r * 1.8,) * 3) & brain
    core = _ellipsoid(shape, tc, (r,) * 3) & brain
    enhancing = core & ~_ellipsoid(shape, tc, (r * 0.6,) * 3)
    seg = np.zeros(shape, np.uint8)
    seg[oedema] = 2
    seg[core] = 1
    seg[enhancing] = et_label
    vols = []
    for contrast in (1.0, 1.6, 1.3, 1.9):
        v = np.where(brain, 100 + rng.normal(0, 8, shape), 0.0)
        v[oedema] *= 1.2 * contrast / 1.5
        v[enhancing] *= contrast
        vols.append(np.clip(v, 0, None).astype(np.float32))
    return vols, seg


def main():
    r"""Write ``--cases`` synthetic cases under ``out``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output folder (one sub-folder per case)")
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--naming", choices=sorted(NAMING), default="met")
    ap.add_argument("--shape", type=int, nargs=3, default=(56, 56, 44))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mods, et_label = NAMING[args.naming]
    rng = np.random.default_rng(args.seed)
    affine = np.eye(4)
    for i in range(args.cases):
        case = f"BraTS-SYN-{i:05d}-000"
        folder = os.path.join(args.out, case)
        os.makedirs(folder, exist_ok=True)
        vols, seg = make_case(rng, tuple(args.shape), et_label)
        for m, v in zip(mods, vols):
            nib.save(nib.Nifti1Image(v, affine), os.path.join(folder, f"{case}-{m}.nii.gz"))
        nib.save(nib.Nifti1Image(seg, affine), os.path.join(folder, f"{case}-seg.nii.gz"))
    print(f"wrote {args.cases} cases ({args.naming} naming) to {args.out}")


if __name__ == "__main__":
    main()
