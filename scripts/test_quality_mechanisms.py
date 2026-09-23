#!/usr/bin/env python
"""Tests for the three mechanisms implemented in the quality-mechanism audit:

  * lesion-size stratified evaluation  (production, evaluation-only)
  * lesion-aware case sampling         (experiment)
  * linear LR warmup                   (experiment)

Plus the production-baseline invariants that must survive all of them.
Repository convention: standalone script, no pytest.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

R = []


def check(name, ok, detail=""):
    R.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:54s} {detail}")


def _row(cid, gt, pred, dice, hd, iou=0.5, reason=""):
    r = {"case_id": cid, "subject_id": cid, "split": "val",
         "dice_mean": dice, "hd95_mean": hd, "metric_valid": int(not reason),
         "failure_reason": reason}
    for i, reg in enumerate(["WT", "TC", "ET"]):
        r[f"gt_vox_{reg}"] = gt[i]
        r[f"pred_vox_{reg}"] = pred[i]
        r[f"dice_{reg}"] = dice
        r[f"iou_{reg}"] = iou
        r[f"hd95_{reg}"] = hd
    return r


def test_strata():
    print("\n[1] LESION-SIZE STRATIFIED EVALUATION")
    from src.experiment import lesion_strata as LS

    check("small bin", LS.stratum_of(50) == "small", LS.stratum_of(50))
    check("medium bin", LS.stratum_of(500) == "medium", LS.stratum_of(500))
    check("large bin", LS.stratum_of(50000) == "large", LS.stratum_of(50000))
    check("boundary 100 -> medium", LS.stratum_of(100) == "medium")
    check("boundary 99 -> small", LS.stratum_of(99) == "small")
    check("empty GT is its own stratum, not 'small'",
          LS.stratum_of(0) == LS.ABSENT, LS.stratum_of(0))

    rows = [
        _row("a", [5000, 800, 50], [5000, 800, 50], 0.9, 3.0),
        _row("b", [5000, 800, 500], [5000, 800, 500], 0.7, 5.0),
        _row("c", [5000, 800, 0], [5000, 800, 0], 0.4,
             float("nan"), reason="ET:both_empty"),
        _row("d", [5000, 800, 80], [5000, 800, 0], 0.1,
             float("nan"), reason="ET:empty_prediction"),
    ]
    out = LS.stratify(rows)
    et = {r["stratum"]: r for r in out if r["region"] == "ET"}

    check("all regions x strata emitted", len(out) == 3 * 4, f"{len(out)} rows")
    check("ET small stratum counts 2 cases", et["small"]["cases"] == 2,
          f"{et['small']['cases']}")
    check("ET medium stratum counts 1 case", et["medium"]["cases"] == 1)
    check("empty-GT case excluded from size strata",
          et[LS.ABSENT]["cases"] == 1 and et[LS.ABSENT]["empty_gt_cases"] == 1)
    check("HD95 averaged over VALID only, never zero-filled",
          et["small"]["hd95_valid_cases"] == 1
          and et["small"]["hd95_invalid_cases"] == 1
          and abs(et["small"]["hd95_mean_over_valid"] - 3.0) < 1e-9,
          f"valid={et['small']['hd95_valid_cases']} "
          f"mean={et['small']['hd95_mean_over_valid']}")
    check("empty prediction counted", et["small"]["empty_prediction_cases"] == 1)
    check("Dice mean per stratum correct",
          abs(et["small"]["dice_mean"] - 0.5) < 1e-9,
          f"{et['small']['dice_mean']}")
    check("stratification uses GT not prediction",
          et["small"]["cases"] == 2,
          "case d has empty pred, 80 gt voxels -> still small")

    tmp = tempfile.mkdtemp()
    p = LS.write_csv(os.path.join(tmp, "r", "s.csv"), out)
    import csv as _csv
    with open(p, encoding="utf-8") as fh:
        got = list(_csv.DictReader(fh))
    check("CSV written with full schema",
          len(got) == 12 and list(got[0].keys()) == LS.COLUMNS)
    absent_idx = [i for i, g in enumerate(got)
                  if g["region"] == "ET" and g["stratum"] == LS.ABSENT][0]
    check("undefined HD95 blank in CSV, not 0",
          got[absent_idx]["hd95_mean_over_valid"] == "")
    check("configurable bins honoured",
          LS.stratum_of(50, [("tiny", 0, 10), ("rest", 10, None)]) == "rest")


def test_sampler():
    print("\n[2] SMALL-LESION-AWARE CASE SAMPLING (experiment)")
    from src.experiment.lesion_sampler import LesionAwareSampler

    et = [0, 0, 50, 5000, 0, 80, 200, 0, 0, 10]
    s = LesionAwareSampler(et, base_seed=42, boost=2.0, small_lesion_voxels=100)

    check("constructed", isinstance(s, torch.utils.data.Sampler))
    check("epoch length == dataset length (budget unchanged)",
          len(s) == len(et) and len(list(s)) == len(et), f"{len(s)}")
    # Synthetic fixture: unlike the real BraTS-METS cohort it DOES contain
    # ET-absent cases, which exercises the zero-volume branch.
    check("ET-absent cases keep weight 1.0",
          all(abs(s.weights[i] - 1.0) < 1e-9 for i in (0, 1, 4, 7, 8)))
    check("small-lesion case boosted above ET-absent",
          s.weights[2] > s.weights[0], f"{s.weights[2]:.3f} > 1.0")
    check("smallest lesion gets the largest boost",
          s.weights[9] >= s.weights[2] >= s.weights[6] >= s.weights[3],
          f"{s.weights[9]:.2f} {s.weights[2]:.2f} "
          f"{s.weights[6]:.2f} {s.weights[3]:.2f}")
    check("boost is bounded by config", s.weights.max() <= 2.0 + 1e-9,
          f"max {s.weights.max():.3f}")
    check("weights normalised to a distribution",
          abs(s.probs.sum() - 1.0) < 1e-12)
    check("all weights strictly positive (no case excluded)",
          bool((s.weights > 0).all()))

    s.set_epoch(0)
    a = list(s)
    s.set_epoch(0)
    b = list(s)
    check("deterministic for a fixed (seed, epoch)", a == b)
    s.set_epoch(1)
    c = list(s)
    check("differs across epochs", a != c)
    s2 = LesionAwareSampler(et, base_seed=7, boost=2.0, small_lesion_voxels=100)
    s2.set_epoch(0)
    check("differs across seeds", list(s2) != a)
    check("yields (epoch, index) like _EpochSampler",
          isinstance(a[0], tuple) and len(a[0]) == 2 and a[0][0] == 0)
    check("indices in range", all(0 <= i < len(et) for _, i in a))

    draws = []
    for ep in range(60):
        s.set_epoch(ep)
        draws += [i for _, i in s]
    # The mechanism is SMALL-LESION oversampling. This synthetic fixture has
    # ET-absent cases, so the ET-positive share could also rise; on the real
    # cohort it cannot (every case has ET) and only the size effect applies.
    # Assert the size effect, which holds in both situations.
    thr = 100
    small = sum(1 for i in draws if 0 < et[i] <= thr) / len(draws)
    base_small = sum(1 for v in et if 0 < v <= thr) / len(et)
    check("small-lesion cases drawn more often than uniform",
          small > base_small, f"{small:.3f} vs uniform {base_small:.3f}")

    flat = LesionAwareSampler([0, 0, 0], 42)
    check("all-ET-absent degrades to uniform",
          bool(np.allclose(flat.probs, 1 / 3)))
    d = s.describe()
    check("describe() reports runtime facts",
          d["et_positive_cases"] == 5 and d["with_replacement"] is True
          and d["samples_per_epoch"] == len(et))


def test_warmup():
    print("\n[3] LR WARMUP (experiment)")
    from src.experiment.warmup import WarmupCosineLR

    base_lr, eta_min = 0.0016, 0.00001
    spe, epochs, wep = 10, 100, 3
    total, warm = spe * epochs, spe * wep
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=base_lr)
    sch = WarmupCosineLR(opt, total_steps=total, warmup_steps=warm,
                         eta_min=eta_min)

    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()

    check("LR at step 0 is below peak (warmup active)", lrs[0] < base_lr,
          f"{lrs[0]:.6f}")
    check("warmup rises monotonically",
          all(lrs[i] < lrs[i + 1] for i in range(warm - 1)))
    check("peak LR reached exactly at end of warmup",
          abs(lrs[warm - 1] - base_lr) < 1e-9, f"{lrs[warm - 1]:.6f}")
    check("production peak LR preserved", abs(max(lrs) - base_lr) < 1e-9)
    check("cosine decay after warmup",
          all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(warm, total - 1)))
    final = opt.param_groups[0]["lr"]
    check("final LR reaches eta_min", abs(final - eta_min) < 1e-6,
          f"{final:.8f}")
    check("step count unchanged (budget preserved)", len(lrs) == total)

    opt2 = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    s2 = WarmupCosineLR(opt2, total_steps=total, warmup_steps=warm,
                        eta_min=eta_min)
    l2 = []
    for _ in range(total):
        l2.append(opt2.param_groups[0]["lr"])
        opt2.step()
        s2.step()
    check("deterministic across constructions", lrs == l2)

    opt3 = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    s3 = WarmupCosineLR(opt3, total_steps=total, warmup_steps=warm,
                        eta_min=eta_min)
    for _ in range(50):
        opt3.step()
        s3.step()
    st = s3.state_dict()
    opt4 = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    s4 = WarmupCosineLR(opt4, total_steps=total, warmup_steps=warm,
                        eta_min=eta_min)
    s4.load_state_dict(st)
    # load_state_dict restores scheduler state but does not push the LR into
    # the optimizer until the next step() -- standard PyTorch behaviour, and
    # what the runner relies on. The contract to assert is that the RESUMED
    # schedule continues to produce the same LR sequence as the uninterrupted
    # one, not that the optimizer is mutated by the load itself.
    check("resume restores scheduler position", s4.last_epoch == 50,
          f"last_epoch {s4.last_epoch}")
    check("resumed LR matches uninterrupted run at the same step",
          abs(s4.get_lr()[0] - lrs[50]) < 1e-12,
          f"{s4.get_lr()[0]:.10f} vs {lrs[50]:.10f}")
    resumed = []
    for _ in range(20):
        opt4.step()
        s4.step()
        resumed.append(opt4.param_groups[0]["lr"])
    check("resumed schedule continues identically",
          all(abs(resumed[i] - lrs[51 + i]) < 1e-12 for i in range(20)))

    opt5 = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    WarmupCosineLR(opt5, total_steps=total, warmup_steps=0, eta_min=eta_min)
    check("warmup_steps=0 starts at peak (baseline-equivalent)",
          abs(opt5.param_groups[0]["lr"] - base_lr) < 1e-9)

    try:
        WarmupCosineLR(opt, total_steps=10, warmup_steps=20, eta_min=eta_min)
        check("rejects warmup >= total", False, "no error raised")
    except ValueError:
        check("rejects warmup >= total", True, "ValueError")


def test_baseline_intact():
    print("\n[4] PRODUCTION BASELINE INVARIANTS")
    prod_path = os.path.join(_HERE, "configs", "glo_nca_production.yaml")
    prod = yaml.safe_load(open(prod_path, encoding="utf-8"))

    check("patchify still OFF",
          prod["data"]["training_patch"]["enabled"] is False)
    check("working volume still 128", prod["training"]["patch_size"] == 128)
    check("Tversky beta still 0.60",
          abs(prod["loss"]["tversky_beta"] - 0.60) < 1e-9)
    check("deep supervision weight still 0.4",
          abs(prod["model"]["deep_supervision"]["weight"] - 0.4) < 1e-9)
    check("EMA decay still 0.999", abs(prod["ema"]["decay"] - 0.999) < 1e-9)
    check("LR still 0.0016",
          abs(prod["optimizer"]["learning_rate"] - 0.0016) < 1e-12)
    check("early stopping still 15 / 0.01",
          prod["training"]["early_stopping"]["patience"] == 15
          and abs(prod["training"]["early_stopping"]["min_delta"] - 0.01) < 1e-9)
    check("post-processing still WT50/TC5/ET0",
          prod["evaluation"]["postprocessing"]["min_component_voxels"]
          == {"WT": 50, "TC": 5, "ET": 0})
    check("seed still 42", prod["experiment"]["seed"] == 42)

    check("NO lesion_aware_sampling in production",
          "lesion_aware_sampling" not in prod)
    check("NO warmup_epochs in production",
          "warmup_epochs" not in prod["optimizer"])
    check("NO dynamic loss weighting in production",
          "dynamic_weighting" not in prod["loss"])
    check("NO boundary loss in production", "boundary" not in prod["loss"])
    check("NO hard-case mining in production", "hard_case_mining" not in prod)

    check("dead sampling keys annotated as INERT",
          "INERT IN PRODUCTION" in open(prod_path, encoding="utf-8").read())

    exp_dir = os.path.join(_HERE, "configs", "experiments")
    present = sorted(f for f in os.listdir(exp_dir) if f.endswith(".yaml"))
    check("only IMPLEMENTED experiments have configs",
          present == ["glo_nca_lr_warmup.yaml",
                      "glo_nca_small_lesion_sampling.yaml"],
          ", ".join(present))

    for fn, key in (("glo_nca_small_lesion_sampling.yaml", "lesion_aware_sampling"),
                    ("glo_nca_lr_warmup.yaml", None)):
        e = yaml.safe_load(open(os.path.join(exp_dir, fn), encoding="utf-8"))
        same = all([
            e["training"]["patch_size"] == prod["training"]["patch_size"],
            e["loss"] == prod["loss"],
            e["model"] == prod["model"],
            e["experiment"]["seed"] == prod["experiment"]["seed"],
            e["data"]["training_patch"]["enabled"] is False,
            e["training"]["epochs"] == prod["training"]["epochs"],
            e["ema"] == prod["ema"],
        ])
        check(f"{fn}: differs from baseline ONLY by its mechanism", same)
        if key:
            check(f"{fn}: validation/test stay uniform",
                  e[key]["apply_to_validation"] is False
                  and e[key]["apply_to_test"] is False)


def main():
    print("=" * 78)
    print("QUALITY MECHANISM TESTS")
    print("=" * 78)
    test_strata()
    test_sampler()
    test_warmup()
    test_baseline_intact()
    failed = [n for n, ok, _ in R if not ok]
    print("\n" + "=" * 78)
    print(f"  {len(R) - len(failed)} / {len(R)} passed")
    if failed:
        print("  FAILED: " + "; ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
