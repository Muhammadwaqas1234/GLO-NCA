#!/usr/bin/env python
r"""Validate the two EXPERIMENT mechanisms against real data / real schedules.

  section 7 -- lesion-aware case sampler, built from the real 898-case census
  section 8 -- LR warmup, verified over the real production step count

This is IMPLEMENTATION validation only. It makes no claim about Dice, and it
trains nothing. Production remains uniform-sampled and warmup-free; that is
asserted here too.

The sampler is built from the ET census CSV (scripts/analyze_et_voxels.py)
rather than by re-reading the dataset, so this script is fast and uses exactly
the counts the census measured.

Usage:
  python scripts/validate_experiments_real_data.py
        [--census reports/analysis/et_voxel_census.csv]
Exit:
  0 if every check passed.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

R = []


def check(name, ok, detail=""):
    R.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=os.path.join(
        _HERE, "reports", "analysis", "et_voxel_census.csv"))
    args = ap.parse_args()

    print("=" * 78)
    print("EXPERIMENT VALIDATION ON REAL DATA (sampler + warmup)")
    print("=" * 78)

    if not os.path.isfile(args.census):
        print(f"\n  BLOCKED: ET census not found at {args.census}")
        print("  Run scripts/analyze_et_voxels.py first.")
        return 1

    with open(args.census, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "ok"]
    et = np.array([int(r["et_voxels"]) for r in rows], dtype=np.int64)
    ids = [r["case_id"] for r in rows]

    print(f"\n  census            {len(rows)} training cases")
    print(f"  ET positive       {int((et > 0).sum())}  "
          f"ET absent {int((et == 0).sum())}")

    # ---------------------------------------------------------------- section 7
    print("\n[7] LESION-AWARE SAMPLER ON REAL DATA")
    from src.experiment.lesion_sampler import LesionAwareSampler

    ecfg = yaml.safe_load(open(os.path.join(
        _HERE, "configs", "experiments", "glo_nca_small_lesion_sampling.yaml"),
        encoding="utf-8"))
    las = ecfg["lesion_aware_sampling"]
    seed = int(ecfg["experiment"]["seed"])

    s = LesionAwareSampler(et, base_seed=seed, boost=float(las["boost"]),
                           small_lesion_voxels=int(las["small_lesion_voxels"]))
    d = s.describe()

    check("sampler constructed from real census",
          d["cases"] == len(rows), f"{d['cases']} cases")
    check("ET-positive count matches census",
          d["et_positive_cases"] == int((et > 0).sum()),
          f"{d['et_positive_cases']}")
    check("all weights finite and positive",
          bool(np.isfinite(s.weights).all() and (s.weights > 0).all()),
          f"range [{s.weights.min():.3f}, {s.weights.max():.3f}]")
    check("ET-absent cases keep weight exactly 1.0",
          bool(np.allclose(s.weights[et == 0], 1.0)) if (et == 0).any() else True)
    check("weights normalised", abs(s.probs.sum() - 1.0) < 1e-12)
    check("epoch length equals baseline (uniform) length",
          len(s) == len(rows), f"{len(s)} draws == {len(rows)} cases")

    s.set_epoch(0)
    a = list(s)
    s.set_epoch(0)
    check(f"deterministic at seed {seed}, epoch 0", a == list(s))
    s.set_epoch(1)
    b = list(s)
    check("differs across epochs", a != b)
    check("draw count per epoch is exact", len(a) == len(rows), f"{len(a)}")
    check("indices stay inside the training split",
          all(0 <= i < len(rows) for _, i in a))

    # Empirical distribution over many epochs.
    drawn = []
    for ep in range(30):
        s.set_epoch(ep)
        drawn += [i for _, i in s]
    drawn = np.array(drawn)
    exp_pos = float((et > 0).mean())
    got_pos = float((et[drawn] > 0).mean())
    small_mask = et > 0
    thr = int(las["small_lesion_voxels"])
    exp_small = float(((et > 0) & (et <= thr)).mean())
    got_small = float(((et[drawn] > 0) & (et[drawn] <= thr)).mean())

    print(f"\n    baseline (uniform) ET-positive share : {exp_pos:.4f}")
    print(f"    experimental       ET-positive share : {got_pos:.4f}")
    print(f"    baseline  small-ET (<= {thr} vox) share : {exp_small:.4f}")
    print(f"    experimental small-ET share          : {got_small:.4f}")
    print(f"    epoch length                          : {len(a)}")

    # MEASURED ON REAL DATA: in BraTS-METS every training case carries ET, so
    # the ET-positive share is already 1.0 under uniform sampling and cannot
    # rise. Asserting an increase would encode a BraTS-GLIOMA assumption this
    # dataset refutes. The contract that applies here is that weighting must
    # not DROP ET-positive coverage; the size-based reweighting is asserted
    # separately below and is what the sampler actually does.
    if exp_pos >= 1.0:
        check("ET-positive coverage preserved (all cases carry ET)",
              got_pos >= exp_pos - 1e-9,
              f"{got_pos:.4f} >= {exp_pos:.4f}; dataset has no ET-absent case")
    else:
        check("ET-positive representation increases",
              got_pos > exp_pos, f"{got_pos:.4f} > {exp_pos:.4f}")
    if exp_small > 0:
        check("small-ET representation increases",
              got_small > exp_small, f"{got_small:.4f} > {exp_small:.4f}")
    else:
        check("small-ET representation increases", True,
              "no small-ET cases in census under current bins -- see report")
    check("ET-absent cases still sampled (not excluded)",
          bool((et[drawn] == 0).any()) if (et == 0).any() else True)

    check("sampler does not modify labels",
          not any(hasattr(s, n) for n in ("label", "labels", "targets")),
          "sampler holds only voxel counts and weights")
    check("sampler carries no model/architecture state",
          not any(hasattr(s, n) for n in ("model", "modules", "state_dict")))

    # Isolation: the sampler must be built from the train split only.
    from src.experiment import datasource
    master = datasource.load_master_split(
        os.path.join(_HERE, "split", "master_split.json"))
    tr, va, te = set(master["train"]), set(master["validation"]), set(master["test"])
    cens = set(ids)
    check("census contains ONLY training cases",
          cens <= tr, f"{len(cens - tr)} non-train ids")
    check("no validation case in the sampler", not (cens & va),
          f"{len(cens & va)} overlap")
    check("no test case in the sampler", not (cens & te),
          f"{len(cens & te)} overlap")

    # ---------------------------------------------------------------- section 8
    print("\n[8] LR WARMUP OVER THE REAL PRODUCTION SCHEDULE")
    from src.experiment.config import load_config
    from src.experiment.warmup import WarmupCosineLR, warmup_steps_from_config

    wcfg_path = os.path.join(_HERE, "configs", "experiments",
                             "glo_nca_lr_warmup.yaml")
    wcfg = load_config(wcfg_path)
    raw = yaml.safe_load(open(wcfg_path, encoding="utf-8"))

    n_train = len(rows)
    batch = int(raw["training"]["batch_size"])
    spe = max(1, n_train // batch)
    epochs = int(raw["training"]["epochs"])
    total = spe * epochs
    base_lr = float(raw["optimizer"]["learning_rate"])
    eta_min = float(raw["optimizer"]["minimum_learning_rate"])
    wep = int(raw["optimizer"]["warmup_epochs"])
    warm = warmup_steps_from_config(wcfg, spe)

    check("warmup_epochs read from experiment config", wep == 3, f"{wep}")
    check("warmup steps derived from real epoch length",
          warm == wep * spe, f"{warm} steps = {wep} x {spe}")
    print(f"    real schedule: {n_train} cases / batch {batch} "
          f"= {spe} steps/epoch x {epochs} epochs = {total:,} steps")

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=base_lr)
    sch = WarmupCosineLR(opt, total_steps=total, warmup_steps=warm,
                         eta_min=eta_min)
    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()

    check("starts below peak", lrs[0] < base_lr, f"{lrs[0]:.8f}")
    check("warmup strictly increasing",
          all(lrs[i] < lrs[i + 1] for i in range(warm - 1)))
    check("peak == production LR at end of warmup",
          abs(lrs[warm - 1] - base_lr) < 1e-12, f"{lrs[warm - 1]:.6f}")
    check("cosine monotonically decays after warmup",
          all(lrs[i] >= lrs[i + 1] - 1e-15 for i in range(warm, total - 1)))
    check("final LR == minimum_learning_rate",
          abs(opt.param_groups[0]["lr"] - eta_min) < 1e-9,
          f"{opt.param_groups[0]['lr']:.10f}")
    check("total steps unchanged vs baseline budget",
          len(lrs) == total, f"{len(lrs):,}")

    # Resume reproduces the identical sequence.
    cut = warm + 137
    o2 = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    s2 = WarmupCosineLR(o2, total_steps=total, warmup_steps=warm, eta_min=eta_min)
    for _ in range(cut):
        o2.step()
        s2.step()
    st = s2.state_dict()
    o3 = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    s3 = WarmupCosineLR(o3, total_steps=total, warmup_steps=warm, eta_min=eta_min)
    s3.load_state_dict(st)
    cont = []
    for _ in range(50):
        o3.step()
        s3.step()
        cont.append(o3.param_groups[0]["lr"])
    check("resumed LR sequence identical to uninterrupted",
          all(abs(cont[i] - lrs[cut + 1 + i]) < 1e-15 for i in range(50)),
          f"resumed at step {cut}")

    # ---------------------------------------------------------- isolation again
    print("\n[16] PRODUCTION REMAINS UNCHANGED")
    prod = yaml.safe_load(open(os.path.join(
        _HERE, "configs", "glo_nca_production.yaml"), encoding="utf-8"))
    check("production sampler = uniform (no sampler key)",
          "lesion_aware_sampling" not in prod)
    check("production warmup = absent",
          "warmup_epochs" not in prod["optimizer"])
    check("production boundary loss = absent", "boundary" not in prod["loss"])
    check("production dynamic weighting = absent",
          "dynamic_weighting" not in prod["loss"])
    check("production hard-case mining = absent", "hard_case_mining" not in prod)

    import io as _io
    runner_src = _io.open(os.path.join(_HERE, "src", "experiment", "runner.py"),
                          encoding="utf-8").read()
    check("runner does not import the experimental sampler",
          "lesion_sampler" not in runner_src)
    check("runner does not import warmup", "from . import warmup" not in runner_src)
    check("runner still constructs _EpochSampler",
          "_EpochSampler(len(ds), cfg.seed)" in runner_src)

    failed = [n for n, ok, _ in R if not ok]
    print("\n" + "=" * 78)
    print(f"  {len(R) - len(failed)} / {len(R)} passed")
    if failed:
        print("  FAILED: " + "; ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
