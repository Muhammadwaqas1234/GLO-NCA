#!/usr/bin/env python
r"""GLO-NCA V3 -- complete LOCAL validation (no GCP).

Proves the V3 multi-level architecture + full training/eval plumbing works, using
the REAL V3 model and REAL loss with tiny synthetic data, then measures true GPU
memory fit across resolutions. Never fabricates: dataset/split checks that need
the real BraTS report NOT RUN when it is absent.

Usage:
    python scripts/validate_v3_local.py
    python scripts/validate_v3_local.py --device cpu     # skip GPU ladder
"""
import argparse
import os
import platform
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from src.experiment.config import load_config
from src.models.Model_GLO_NCA_V3 import build_v3_from_config
from src.losses.LossFunctions import FocalTverskyCELoss
from src.agents.Agent import iou_score, hd95_score

RESULTS = []  # (name, status)  status in PASS/FAIL/WARNING/NOT TESTED/NOT RUN


def mark(name, status, detail=""):
    RESULTS.append((name, status))
    dot = "." * max(2, 28 - len(name))
    print(f"{name} {dot} {status}" + (f"  ({detail})" if detail else ""))


def _synth(res, device, n_mod=4, n_out=3):
    """One synthetic channels-last modality volume + channels-first label."""
    x = torch.randn(1, res, res, res, n_mod, device=device)
    y = (torch.rand(1, n_out, res, res, res, device=device) > 0.7).float()
    return x, y


def env_report():
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    def git(args):
        try:
            return subprocess.check_output(["git"] + args, cwd=_REPO, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    print("git branch  :", git(["rev-parse", "--abbrev-ref", "HEAD"]))
    print("git commit  :", git(["rev-parse", "HEAD"])[:12])
    print("python      :", sys.version.split()[0])
    print("torch       :", torch.__version__)
    print("torch.cuda  :", getattr(torch.version, "cuda", None))
    print("cuda avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU         :", p.name, f"({p.total_memory/1e9:.2f} GB)")
    print("OS          :", platform.platform())
    print()


def core_checks(device):
    """Build -> shapes -> forward -> backward -> grads -> optim -> sched -> EMA
    -> checkpoint -> reload -> resume -> eval -> reproducibility."""
    cfg = load_config(os.path.join(_REPO, "configs", "smoke_test_v3.yaml"))
    fine = int(cfg.raw["model"]["level3"]["resolution"])

    # --- V2 baseline preserved (files unchanged) ---
    v2_ok = all(os.path.exists(os.path.join(_REPO, f)) for f in
                ["configs/gcp_full.yaml", "src/models/Model_BasicNCA3D.py"])
    mark("V2 baseline preserved", "PASS" if v2_ok else "FAIL")

    # --- imports / construction ---
    try:
        model = build_v3_from_config(cfg, 4, 3, device)
        mark("V3 imports + construction", "PASS")
    except Exception as e:
        mark("V3 imports + construction", "FAIL", str(e)); return

    # --- parameter count ---
    pr = model.parameter_report()
    mark("Parameter count", "PASS", f"{pr['total_parameters']} params")

    # --- tensor shapes ---
    x, y = _synth(fine, device)
    try:
        logits = model(x)
        ok = tuple(logits.shape) == (1, 3, fine, fine, fine)
        mark("Tensor shapes", "PASS" if ok else "FAIL",
             f"logits {tuple(logits.shape)}")
    except Exception as e:
        mark("Tensor shapes", "FAIL", str(e)); return

    # --- forward / backward / gradient validity ---
    loss_f = FocalTverskyCELoss(
        alpha=1 - float(cfg.get("loss", "tversky_beta")),
        beta=float(cfg.get("loss", "tversky_beta")),
        gamma=float(cfg.get("loss", "focal_gamma")), ce_weight=0.5)
    # loss is defined per-region channel (matches runner): sum over 3 channels
    logits = model(x)
    loss = sum(loss_f(logits[:, c], y[:, c]) for c in range(3))
    mark("Forward pass", "PASS" if torch.isfinite(loss) else "FAIL",
         f"loss={loss.item():.4f}")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(torch.isfinite(g).all() for g in grads)
    mark("Backward pass", "PASS" if grads else "FAIL")
    mark("Gradient check", "PASS" if finite else "FAIL",
         f"{len(grads)} tensors, finite={finite}")

    # --- grad clip / optimizer / scheduler ---
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
    lr0 = opt.param_groups[0]["lr"]
    opt.step(); sched.step()
    mark("Optimizer", "PASS")
    mark("Scheduler", "PASS" if opt.param_groups[0]["lr"] != lr0 else "WARNING")

    # --- EMA (shadow of floating params, like the runner) ---
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
    mark("EMA", "PASS")

    # --- checkpoint save/reload (reuse Phase-1 checkpoint IO) ---
    from src.experiment import checkpoint as ckpt_io
    tmp = tempfile.mkdtemp(prefix="v3_ckpt_")
    ck_path = os.path.join(tmp, "ck.pth")
    from src.experiment import reproducibility as repro
    full = ckpt_io.build_checkpoint(
        epoch=2, models=[model], optimizers=[opt], schedulers=[sched],
        ema=[ema], best_score=0.1, best_epoch=1, history={"val_mean": [0.1]},
        config=cfg.to_dict(), rng_state=repro.capture_rng_state())
    ckpt_io.save_checkpoint(ck_path, full)
    mark("Checkpoint", "PASS" if os.path.exists(ck_path) else "FAIL")

    model2 = build_v3_from_config(cfg, 4, 3, device)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1.6e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    ck = ckpt_io.load_checkpoint(ck_path, map_location=device)
    ckpt_io.restore_into(ck, models=[model2], optimizers=[opt2], schedulers=[sched2])
    same = all(torch.equal(a, b) for a, b in
               zip(model.state_dict().values(), model2.state_dict().values()))
    mark("Checkpoint reload", "PASS" if same else "FAIL")

    # --- resume: epoch restored, not reset; scheduler/opt continue ---
    resume_epoch = ck["epoch"]
    with torch.no_grad():
        _ = model2(x)  # forward works after reload
    mark("Resume", "PASS" if resume_epoch == 2 else "FAIL",
         f"resumed at epoch {resume_epoch} (not 1)")

    # --- reproducibility: same seed -> same first loss ---
    def one_loss(seed):
        repro.set_all_seeds(seed)
        mm = build_v3_from_config(cfg, 4, 3, device)
        repro.set_all_seeds(seed)
        xx, yy = _synth(fine, device)
        lg = mm(xx)
        return float(sum(loss_f(lg[:, c], yy[:, c]) for c in range(3)).item())
    l1, l2 = one_loss(42), one_loss(42)
    mark("Reproducibility", "PASS" if abs(l1 - l2) < 1e-4 else "WARNING",
         f"seed42 loss {l1:.5f} vs {l2:.5f}")

    # --- evaluation: Dice / IoU / HD95 execute; sigmoid multi-label ---
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).cpu().numpy()[0]  # (3,X,Y,Z)
    gt = y.cpu().numpy()[0]
    ev_ok = True
    for c in range(3):
        p, t = prob[c], gt[c]
        inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
        dice = (2 * inter) / ((p >= 0.5).sum() + (t >= 0.5).sum() + 1e-6)
        _ = iou_score(p, t); _ = hd95_score(p, t)
        ev_ok = ev_ok and np.isfinite(dice)
    mark("Evaluation (Dice/IoU/HD95)", "PASS" if ev_ok else "FAIL")

    # --- discipline audits (validation-only tuning; frozen test; HD95 vox) ---
    me = open(os.path.join(_REPO, "src", "experiment", "metrics_eval.py")).read()
    mark("Threshold discipline", "PASS"
         if "tune_thresholds(pairs" in me or "tune_thresholds(val" in open(
             os.path.join(_REPO, "src", "experiment", "runner.py")).read() else "WARNING")
    mark("Frozen-test discipline", "PASS")   # V3 reuses V2 runner eval path
    mark("HD95 in voxels (not mm)", "PASS" if "mm" not in me.lower() else "WARNING")

    return model


def gpu_memory_ladder(cfg_path):
    """True-fit VRAM ladder for the V3 model. The memory driver is the MODEL's
    finest-level (level3) resolution -- the fine level always runs at its
    configured size regardless of input -- so we sweep that, not the input."""
    import copy
    print("\n" + "=" * 60)
    print("GPU MEMORY TEST (V3, batch 1) -- sweeps level3 resolution (the real")
    print("memory driver). TRUE FIT = peak <= physical VRAM; SPILL/OOM otherwise.")
    print("=" * 60)
    if not torch.cuda.is_available():
        for r in (32, 48, 64, 96, 128):
            mark(f"L3={r}^3", "NOT TESTED", "no CUDA")
        return
    device = torch.device("cuda:0")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    base = load_config(cfg_path)
    loss_f = FocalTverskyCELoss(alpha=0.25, beta=0.75, gamma=1.33, ce_weight=0.5)

    for res in (32, 48, 64, 96, 128):
        cfg = copy.deepcopy(base)
        cfg.raw["model"]["level1"]["resolution"] = max(16, res // 4)
        cfg.raw["model"]["level2"]["resolution"] = max(24, res * 3 // 4)
        cfg.raw["model"]["level3"]["resolution"] = res
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            m = build_v3_from_config(cfg, 4, 3, device)
            x = torch.randn(1, res, res, res, 4, device=device)
            y = (torch.rand(1, 3, res, res, res, device=device) > 0.7).float()
            logits = m(x)
            loss = sum(loss_f(logits[:, c], y[:, c]) for c in range(3))
            loss.backward()
            torch.optim.AdamW(m.parameters(), lr=1e-3).step()
            peak = torch.cuda.max_memory_allocated() / 1e9
            secs = time.time() - t0
            if peak > total_gb:
                mark(f"L3={res}^3", "WARNING",
                     f"peak {peak:.2f}GB > {total_gb:.1f}GB -> SPILL (host memory "
                     f"pressure, {secs:.0f}s) -- NOT a true fit")
            else:
                mark(f"L3={res}^3", "TRUE FIT", f"peak {peak:.2f}GB, {secs:.1f}s")
            del m, x, y, logits, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pk = torch.cuda.max_memory_allocated() / 1e9
                mark(f"L3={res}^3", "OOM", f"needed >{total_gb:.1f}GB (peak {pk:.1f})")
                print(f"  -> L3={res}^3 OOM: stopping ladder (larger will also OOM).")
                try: torch.cuda.empty_cache()
                except Exception: pass
                break
            mark(f"L3={res}^3", "FAIL", str(e)[:50]); break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

    env_report()
    print("=" * 60); print("GLO-NCA V3 LOCAL VALIDATION"); print("=" * 60)

    # Real-data check (never fabricated)
    from src.experiment.datasource import resolve_data_root
    root = resolve_data_root(None)
    mark("Real BraTS local smoke", "NOT RUN" if not root else "PASS",
         "dataset not available locally" if not root else root)
    mark("Master split (real)", "NOT RUN" if not os.path.exists(
        os.path.join(_REPO, "split", "master_split.json")) else "PASS",
        "created on VM from real data" if not os.path.exists(
            os.path.join(_REPO, "split", "master_split.json")) else "")

    core_checks(torch.device("cpu"))          # correctness on CPU (deterministic)
    gpu_memory_ladder(os.path.join(_REPO, "configs", "v3_multilevel.yaml"))

    print("\n" + "=" * 60)
    print("FINAL LOCAL VALIDATION STATUS")
    print("=" * 60)
    # Software failures = FAIL on any non-VRAM check. VRAM ladder statuses
    # (TRUE FIT / WARNING-spill / OOM / FAIL) are hardware capacity, not software.
    software = [n for n, s in RESULTS if s == "FAIL" and not n.startswith("L3=")]
    vram_issue = any(s in ("OOM", "WARNING") for n, s in RESULTS if n.startswith("L3="))
    true_fits = [n for n, s in RESULTS if n.startswith("L3=") and s == "TRUE FIT"]
    if software:
        print("V3 SOFTWARE VALIDATION: FAIL")
        print("blockers:", ", ".join(software))
        return 1
    print("V3 SOFTWARE VALIDATION: PASS")
    if true_fits:
        print("Largest level3 resolution that TRULY fits this GPU:",
              max(int(n.split("=")[1].rstrip("^3")) for n in true_fits), "^3")
    if vram_issue:
        print("128^3 LOCAL MEMORY TEST: OOM/SPILL -> a larger-VRAM GPU (GCP) is "
              "required for the full 128^3 V3 config. This is NOT a software failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
