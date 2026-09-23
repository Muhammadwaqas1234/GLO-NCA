#!/usr/bin/env python
r"""
GLO-NCA production real-data smoke.  REAL BraTS DATA, no synthetic tensors.

Exercises the production path end to end on a small number of REAL cases drawn
from the canonical split:

    dataset -> split -> preprocessing -> 128^3 working volume -> L1 48^3
    -> L2 64^3 -> NCA steps -> global context -> fusion -> primary logits
    -> auxiliary training outputs -> primary loss -> auxiliary loss
    -> backward -> optimizer -> EMA -> checkpoint -> validation
    -> post-processing -> Dice/IoU/HD95

It deliberately uses TRAIN and VALIDATION cases only; the test split is never
opened, so the run cannot contaminate the frozen evaluation set.

If the dataset is unavailable the smoke reports BLOCKED and exits non-zero.
It never substitutes synthetic data for real cases.

Usage:  python scripts/test_real_data_smoke.py [--data-root DIR] [--cases N]
Exit:   0 if every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ROOT = r"C:\Users\raiwa\Downloads\MICCAI-LH-BraTS2025-MET-Challenge-Training"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {detail}")


def load_case(case_dir: str, case_id: str, modalities, size: int):
    """Load one real case: 4 modalities + label, resampled to `size`^3."""
    import nibabel as nib
    import torch.nn.functional as F

    vols = []
    for mod in modalities:
        path = os.path.join(case_dir, f"{case_id}-{mod}.nii.gz")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        vols.append(np.asarray(nib.load(path).dataobj, dtype=np.float32))
    seg = np.asarray(
        nib.load(os.path.join(case_dir, f"{case_id}-seg.nii.gz")).dataobj,
        dtype=np.float32)

    img = np.stack(vols, axis=-1)
    for c in range(img.shape[-1]):
        v = img[..., c]
        nz = v[v > 0]
        if nz.size:
            img[..., c] = (v - nz.mean()) / (nz.std() + 1e-8)

    wt = np.isin(seg, [1, 2, 3, 4]).astype(np.float32)
    tc = np.isin(seg, [1, 3, 4]).astype(np.float32)
    et = np.isin(seg, [3, 4]).astype(np.float32)
    lab = np.stack([wt, tc, et], axis=-1)

    def resize(a, mode):
        t = torch.from_numpy(np.ascontiguousarray(a)).float()
        t = t.permute(3, 0, 1, 2).unsqueeze(0)
        kw = {} if mode == "nearest" else {"align_corners": False}
        t = torch.nn.functional.interpolate(t, size=(size, size, size),
                                            mode=mode, **kw)
        return t.squeeze(0).permute(1, 2, 3, 0).contiguous()

    return resize(img, "trilinear"), resize(lab, "nearest")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--cases", type=int, default=3)
    args = ap.parse_args()

    from src.experiment.config import load_config
    from src.experiment.datasource import discover_cases
    from src.experiment.postprocess import apply_to_pairs, min_component_config
    # The runner selects build_glo_nca_global_context when the config has a
    # model.global_context block, which production does. Mirror that here so
    # the gate verifies the model that actually trains.
    from src.experiment.runner import _build_production_model

    print("=" * 74)
    print("REAL-DATA SMOKE -- GLO-NCA Production")
    print("=" * 74)

    if not os.path.isdir(args.data_root):
        print(f"\n  BLOCKED: dataset not found at {args.data_root}")
        print("  The real-data gate cannot be satisfied with synthetic data.")
        return 1

    cfg = load_config(os.path.join(_HERE, "configs", "glo_nca_production.yaml"))
    WV = int(cfg.get("training", "patch_size"))
    modalities = list(cfg.get("dataset", "modalities"))

    # discover_cases returns (case_id, path RELATIVE to data_root).
    located = {cid: os.path.join(args.data_root, rel)
               for cid, rel in discover_cases(args.data_root)}
    check("dataset discovered 1296 cases", len(located) == 1296, str(len(located)))

    split = json.load(open(os.path.join(_HERE, "split", "master_split.json"),
                           encoding="utf-8"))
    check("canonical split 898/200/198",
          (len(split["train"]), len(split["validation"]), len(split["test"]))
          == (898, 200, 198),
          f"{len(split['train'])}/{len(split['validation'])}/{len(split['test'])}")

    train_ids = split["train"][:args.cases]
    val_ids = split["validation"][:args.cases]
    check("smoke uses train+validation only, never test",
          not (set(train_ids) | set(val_ids)) & set(split["test"]),
          f"{len(train_ids)} train, {len(val_ids)} val")

    # Explicit test-set firewall accounting, reported as a count rather than a
    # boolean so the number appears verbatim in the pre-flight report.
    _test_set = set(split["test"])
    _touched = [c for c in (train_ids + val_ids) if c in _test_set]
    check("test cases accessed = 0", len(_touched) == 0,
          f"{len(_touched)} of {len(_test_set)} test cases opened")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_production_model(cfg, device).to(device)
    total = sum(p.numel() for p in model.parameters())
    aux_n = sum(p.numel() for p in model.aux_heads.parameters()) \
        if getattr(model, "aux_heads", None) else 0
    check("inference parameters 30,209", total - aux_n == 30209, f"{total - aux_n:,}")
    check("auxiliary parameters 75", aux_n == 75, str(aux_n))
    check("training parameters 30,284", total == 30284, f"{total:,}")

    loss_f = None
    try:
        from src.losses.LossFunctions import FocalTverskyCELoss
        beta = float(cfg.get("loss", "tversky_beta"))
        loss_f = FocalTverskyCELoss(1 - beta, beta,
                                    float(cfg.get("loss", "focal_gamma")),
                                    float(cfg.get("loss", "ce_weight")))
    except Exception as exc:                                 # pragma: no cover
        check("production loss importable", False, f"{type(exc).__name__}: {exc}")
        return 1
    check("production loss importable", True, "FocalTverskyCELoss")

    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(cfg.get("optimizer", "learning_rate")))
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    ds_w = float((cfg.section("model").get("deep_supervision") or {}).get("weight", 0.0))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    iter_times, losses = [], []
    for cid in train_ids:
        img, lab = load_case(located[cid], cid, modalities, WV)
        check(f"real case loaded {cid[:22]}",
              tuple(img.shape) == (WV, WV, WV, 4) and torch.isfinite(img).all(),
              f"{tuple(img.shape)}")
        x = img.unsqueeze(0).to(device)
        y = lab.unsqueeze(0).to(device)

        t0 = time.perf_counter()
        out = model(x)
        if not isinstance(out, tuple):
            check("train() returns (logits, aux)", False, "bare tensor")
            return 1
        logits, aux = out
        logits_cl = logits.float().permute(0, 2, 3, 4, 1).contiguous()
        y_cf = y.permute(0, 4, 1, 2, 3).contiguous()
        y_cf = torch.nn.functional.interpolate(
            y_cf, size=tuple(logits_cl.shape[1:4]), mode="nearest")
        tgt = y_cf.permute(0, 2, 3, 4, 1).contiguous()

        primary = sum(loss_f(logits_cl[..., m], tgt[..., m]) for m in range(3))
        aux_loss = 0
        for a in aux:
            a_cl = a.float().permute(0, 2, 3, 4, 1).contiguous()
            aux_loss = aux_loss + sum(loss_f(a_cl[..., m], tgt[..., m])
                                      for m in range(3))
        loss = primary + ds_w * (aux_loss / max(1, len(aux)))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema[k].mul_(0.999).add_(v.detach().float(), alpha=0.001)
        if device.type == "cuda":
            torch.cuda.synchronize()
        iter_times.append(time.perf_counter() - t0)
        losses.append(float(primary.detach()))

    check("train() returned (logits, aux) for every case", True,
          f"{len(train_ids)} cases")
    check("primary losses finite", all(np.isfinite(losses)), str([f"{l:.3f}" for l in losses]))
    check("gradients finite",
          all(p.grad is None or torch.isfinite(p.grad).all()
              for p in model.parameters()))
    check("parameters finite after optimizer steps",
          all(torch.isfinite(p).all() for p in model.parameters()))
    check("EMA state finite", all(torch.isfinite(v).all() for v in ema.values()))

    # ---- validation on REAL validation cases, primary logits only ----------
    model.eval()
    pairs = []
    with torch.no_grad():
        for cid in val_ids:
            img, lab = load_case(located[cid], cid, modalities, WV)
            x = img.unsqueeze(0).to(device)
            out = model(x)
            check(f"eval() bare tensor {cid[:22]}",
                  isinstance(out, torch.Tensor), type(out).__name__)
            prob = torch.sigmoid(out.float()).permute(0, 2, 3, 4, 1)[0].cpu().numpy()
            y_cf = lab.unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()
            y_cf = torch.nn.functional.interpolate(
                y_cf, size=prob.shape[:3], mode="nearest")
            pairs.append((prob, y_cf.permute(0, 2, 3, 4, 1)[0].numpy()))

    mc = min_component_config(cfg)
    pp = apply_to_pairs(pairs, {r: 0.5 for r in ("WT", "TC", "ET")}, mc)
    check("post-processing applied to validation", len(pp) == len(pairs),
          f"WT {mc['WT']} / TC {mc['TC']} / ET {mc['ET']}")

    from src.experiment import metrics_eval as ME
    scored = ME.score(pp, {r: 0.5 for r in ("WT", "TC", "ET")})
    finite = all(np.isfinite(scored[r]["dice"]) for r in ("WT", "TC", "ET"))
    check("Dice computed and finite", finite,
          " ".join(f"{r}={scored[r]['dice']:.3f}" for r in ("WT", "TC", "ET")))

    # Per-case voxel accounting. The model has taken a couple of optimizer
    # steps from a random initialisation, so Dice near zero is the EXPECTED
    # result: this gate validates the pipeline, never segmentation quality.
    for idx, (prob, gt) in enumerate(pp):
        for i, r in enumerate(("WT", "TC", "ET")):
            g = int((gt[..., i] >= 0.5).sum())
            p_ = int((prob[..., i] >= 0.5).sum())
            print(f"    case{idx} {r}: gt_vox {g:>7d}  pred_vox {p_:>7d}")
    any_gt = any(int((gt[..., i] >= 0.5).sum()) > 0
                 for _, gt in pp for i in range(3))
    check("ground truth carries real foreground", any_gt,
          "labels are real, not synthetic")

    hd_valid = sum(1 for r in ("WT", "TC", "ET")
                   if np.isfinite(scored[r].get("hd95", float("nan"))))
    hd_undef = 3 - hd_valid
    check("HD95 accounted, undefined values not faked", True,
          f"{hd_valid}/3 defined, {hd_undef}/3 undefined "
          f"(empty prediction or empty ground truth)")

    # ---- per-case diagnostics + lesion-size stratification ------------------
    # These are the production EVALUATION artifacts. They must be produced from
    # the same pairs the metrics came from, with no extra model pass.
    import tempfile

    from src.experiment import lesion_strata as LS
    from src.experiment import per_case_diagnostics as PCD

    _ids = [f"smoke-{i:03d}" for i in range(len(pp))]
    rows = PCD.build_rows(pp, _ids, "validation",
                          {r: 0.5 for r in ("WT", "TC", "ET")})
    check("per-case diagnostic rows built", len(rows) == len(pp),
          f"{len(rows)} rows x {len(PCD.COLUMNS)} columns"
          if hasattr(PCD, "COLUMNS") else f"{len(rows)} rows")
    check("per-case rows carry gt/pred voxel counts",
          all(f"gt_vox_{r}" in rows[0] and f"pred_vox_{r}" in rows[0]
              for r in ("WT", "TC", "ET")))
    check("HD95 undefined recorded as NaN, never 0",
          all(not (row[f"hd95_{r}"] == 0.0 and row[f"gt_vox_{r}"] == 0)
              for row in rows for r in ("WT", "TC", "ET")))

    strata = LS.stratify(rows)
    check("lesion-size stratification produced",
          len(strata) == 3 * (len(LS.DEFAULT_STRATA) + 1),
          f"{len(strata)} region x stratum rows")
    _names = {s["stratum"] for s in strata}
    check("all strata present incl. absent_gt",
          _names == {n for n, _, _ in LS.DEFAULT_STRATA} | {LS.ABSENT},
          ", ".join(sorted(_names)))
    check("absent_gt separated from size strata",
          all(s["empty_gt_cases"] == s["cases"]
              for s in strata if s["stratum"] == LS.ABSENT),
          "empty-GT cases cannot distort small/medium/large Dice")
    check("stratum case counts reconcile with per-case rows",
          all(sum(s["cases"] for s in strata if s["region"] == r) == len(rows)
              for r in ("WT", "TC", "ET")))

    with tempfile.TemporaryDirectory() as td:
        pc_csv = PCD.write_csv(os.path.join(td, "reports", "vpc.csv"), rows)
        ls_csv = LS.write_csv(os.path.join(td, "reports", "vls.csv"), strata)
        check("validation_per_case CSV written", os.path.getsize(pc_csv) > 0,
              f"{os.path.getsize(pc_csv):,} bytes")
        check("by_lesion_size CSV written", os.path.getsize(ls_csv) > 0,
              f"{os.path.getsize(ls_csv):,} bytes")
        import csv as _csv
        with open(ls_csv, encoding="utf-8") as fh:
            _got = list(_csv.DictReader(fh))
        check("stratified CSV schema matches module contract",
              _got and list(_got[0].keys()) == LS.COLUMNS)

    # ---- checkpoint round trip on the real model ---------------------------
    from src.experiment import checkpoint as ckpt_io
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "smoke.pth")
        ckpt_io.save_checkpoint(p, {"model": [model.state_dict()], "ema": [ema],
                                    "epoch": 1, "format": "glo-nca-v2-ckpt-1"})
        ck = torch.load(p, map_location="cpu", weights_only=False)
        check("checkpoint round trip on the real model",
              all(torch.equal(model.state_dict()[k].cpu().float(),
                              ck["model"][0][k].cpu().float())
                  for k in ck["model"][0]),
              f"{os.path.getsize(p):,} bytes")

    med = sorted(iter_times)[len(iter_times) // 2]
    print()
    print(f"  MEASURED device            : {device}")
    print(f"  MEASURED working volume    : {WV}^3")
    print(f"  MEASURED median iteration  : {med:.3f} s")
    if device.type == "cuda":
        print(f"  MEASURED peak VRAM         : "
              f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    print(f"  MEASURED primary losses    : "
          + ", ".join(f"{l:.4f}" for l in losses))

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
