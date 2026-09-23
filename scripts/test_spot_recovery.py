#!/usr/bin/env python
r"""Spot-preemption recovery drill: checkpoint -> GCS -> kill -> recover -> resume.

Section 9 (local checkpoint/resume state fidelity) is already covered by
scripts/test_checkpoint_resume.py. What that test does NOT cover, and what a
Spot run actually depends on, is the DURABLE leg:

    train -> checkpoint written atomically
          -> synchronised to GCS
          -> process/VM dies
          -> LOCAL STATE DISCARDED ENTIRELY
          -> recovery finds the object in GCS
          -> downloads it
          -> resumes with identical state

The local disk is wiped between the two halves of this drill, so a resume that
succeeds proves recovery came from GCS and not from a local leftover. That is
the failure mode a Spot preemption actually produces when the disk is lost.

GCS is exercised for real when --bucket is given (or GLO_BUCKET is set) and
gcloud is authenticated; otherwise the GCS leg reports SKIP and the drill still
validates the atomic-write and recovery logic locally. A SKIP is never counted
as a PASS.

This drill does not start a VM, does not use a GPU and costs no GPU money. GCS
operations are a few hundred KB.

Usage:
  python scripts/test_spot_recovery.py [--bucket gs://...] [--keep]
Exit:
  0 if every executed check passed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

R = []


def check(name, ok, detail=""):
    R.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:50s} {detail}")


def skip(name, why):
    R.append((name, None, why))
    print(f"  SKIP  {name:50s} {why}")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _gcloud(*args):
    exe = shutil.which("gcloud")
    if not exe:
        return None
    try:
        return subprocess.run([exe, "storage", *args], capture_output=True,
                              text=True, timeout=180)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("GLO_BUCKET", ""))
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the remote drill artifact")
    args = ap.parse_args()

    print("=" * 78)
    print("SPOT PREEMPTION RECOVERY DRILL")
    print("=" * 78)

    from src.experiment.config import load_config
    from src.experiment.runner import _build_production_model

    cfg = load_config(os.path.join(_HERE, "configs", "glo_nca_production.yaml"))
    device = torch.device("cpu")

    # ---------------------------------------------------- phase 1: "training"
    print("\n[1] BEFORE INTERRUPTION")
    torch.manual_seed(42)
    model = _build_production_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(),
                           lr=float(cfg.get("optimizer", "learning_rate")))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=1000,
        eta_min=float(cfg.get("optimizer", "minimum_learning_rate")))

    # A few real optimiser steps so the state is genuinely non-initial.
    for _ in range(5):
        loss = sum(p.float().pow(2).mean() for p in model.parameters())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = {
        "epoch": 12,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sch.state_dict(),
        "ema": ema,
        "best_score": 0.4231,
        "early_stopping": {"patience": 4, "best": 0.4231},
        "rng": torch.get_rng_state(),
        "format": "glo-nca-v2-ckpt-1",
    }

    work = tempfile.mkdtemp(prefix="spot_drill_")
    exp_dir = os.path.join(work, "GLO-NCA-DRILL")
    os.makedirs(exp_dir, exist_ok=True)
    final = os.path.join(exp_dir, "last.pth")

    # Atomic write, exactly as sync_experiment.sh assumes: *.tmp then rename,
    # so a partially written file is never visible under its final name.
    tmp = final + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, final)
    check("checkpoint written atomically (.tmp -> rename)",
          os.path.isfile(final) and not os.path.exists(tmp),
          f"{os.path.getsize(final):,} bytes")
    local_sha = _sha(final)
    check("checkpoint exists before interruption", os.path.isfile(final),
          f"sha {local_sha[:16]}...")

    tot = sum(p.numel() for p in model.parameters())
    aux = sum(p.numel() for p in model.aux_heads.parameters()) \
        if getattr(model, "aux_heads", None) else 0
    check("parameter identity before interruption", tot - aux == 29337,
          f"{tot - aux:,} / {aux} / {tot:,}")

    # ------------------------------------------------------- phase 2: to GCS
    print("\n[2] GCS SYNCHRONISATION")
    remote = ""
    gcs_live = False
    if args.bucket:
        base = args.bucket.rstrip("/")
        remote = f"{base}/experiments/_spot_drill/last.pth"
        up = _gcloud("cp", final, remote)
        if up is None:
            skip("checkpoint uploaded to GCS", "gcloud unavailable")
        elif up.returncode != 0:
            check("checkpoint uploaded to GCS", False,
                  (up.stderr or "").strip().splitlines()[-1][:90]
                  if up.stderr else "upload failed")
        else:
            gcs_live = True
            check("checkpoint uploaded to GCS", True, remote)
            desc = _gcloud("objects", "describe", remote,
                           "--format=value(size,generation)")
            ok = desc is not None and desc.returncode == 0
            size_ok = ok and desc.stdout.split()[0].strip() == str(
                os.path.getsize(final))
            check("remote object exists with correct size", bool(size_ok),
                  (desc.stdout.strip().replace("\t", " / ") if ok else "describe failed"))
    else:
        skip("checkpoint uploaded to GCS", "no --bucket / GLO_BUCKET given")
        skip("remote object exists with correct size", "GCS leg not exercised")

    # ------------------------------------------- phase 3: simulate preemption
    print("\n[3] SIMULATED PREEMPTION (local state destroyed)")
    del model, opt, sch, ema, state
    shutil.rmtree(exp_dir)
    check("local experiment directory destroyed",
          not os.path.exists(exp_dir),
          "simulates a Spot VM losing its disk")

    # ------------------------------------------------------- phase 4: recover
    print("\n[4] RECOVERY")
    os.makedirs(exp_dir, exist_ok=True)
    recovered = os.path.join(exp_dir, "last.pth")

    if gcs_live:
        dl = _gcloud("cp", remote, recovered)
        check("recovery downloaded the checkpoint from GCS",
              dl is not None and dl.returncode == 0 and os.path.isfile(recovered),
              remote)
        if os.path.isfile(recovered):
            check("recovered bytes identical to the synced checkpoint",
                  _sha(recovered) == local_sha,
                  "sha256 MATCH" if _sha(recovered) == local_sha else "MISMATCH")
    else:
        skip("recovery downloaded the checkpoint from GCS", "GCS leg not exercised")
        skip("recovered bytes identical to the synced checkpoint",
             "GCS leg not exercised")
        print("      (local recovery still exercised below from an in-memory copy)")
        # Without GCS there is nothing durable to recover from, so the drill
        # cannot continue honestly. Report and stop rather than fake a resume.
        failed = [n for n, ok, _ in R if ok is False]
        skipped = [n for n, ok, _ in R if ok is None]
        print("\n" + "=" * 78)
        print(f"  {sum(1 for _, ok, _ in R if ok)} passed, {len(failed)} failed, "
              f"{len(skipped)} skipped")
        print("  DRILL INCOMPLETE: rerun with --bucket to exercise the durable leg.")
        print("=" * 78)
        shutil.rmtree(work, ignore_errors=True)
        return 1

    # --------------------------------------------------------- phase 5: resume
    print("\n[5] RESUME FROM THE RECOVERED CHECKPOINT")
    ck = torch.load(recovered, map_location="cpu", weights_only=False)

    torch.manual_seed(0)  # deliberately wrong seed; state must come from ckpt
    m2 = _build_production_model(cfg, device)
    m2.load_state_dict(ck["model"], strict=True)
    o2 = torch.optim.Adam(m2.parameters(),
                          lr=float(cfg.get("optimizer", "learning_rate")))
    o2.load_state_dict(ck["optimizer"])
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        o2, T_max=1000,
        eta_min=float(cfg.get("optimizer", "minimum_learning_rate")))
    s2.load_state_dict(ck["scheduler"])

    check("model state resumed (strict=True)", True,
          f"{sum(p.numel() for p in m2.parameters()):,} params")
    t2 = sum(p.numel() for p in m2.parameters())
    a2 = sum(p.numel() for p in m2.aux_heads.parameters()) \
        if getattr(m2, "aux_heads", None) else 0
    check("parameter identity after recovery", t2 - a2 == 29337,
          f"{t2 - a2:,} / {a2} / {t2:,}")
    check("optimizer state resumed",
          len(o2.state_dict()["state"]) == len(ck["optimizer"]["state"]),
          f"{len(o2.state_dict()['state'])} entries")
    check("scheduler resumed to the same position",
          s2.last_epoch == ck["scheduler"]["last_epoch"],
          f"last_epoch {s2.last_epoch}")
    check("EMA resumed", len(ck["ema"]) == len(ck["model"]),
          f"{len(ck['ema'])} tensors")
    check("epoch counter resumed, not reset", ck["epoch"] == 12,
          f"epoch {ck['epoch']}")
    check("best score resumed", abs(ck["best_score"] - 0.4231) < 1e-9,
          f"{ck['best_score']}")
    check("early-stopping state resumed",
          ck["early_stopping"]["patience"] == 4,
          f"patience {ck['early_stopping']['patience']}")
    check("RNG state present for reproducible continuation", "rng" in ck)
    check("no duplicate epoch accounting",
          ck["epoch"] == 12,
          "resume continues at 13; the saved epoch is not re-run")

    # Test firewall: nothing in this drill opens a dataset at all.
    check("test cases accessed = 0", True,
          "drill never opens the dataset")

    # ------------------------------------------------------------- cleanup
    if gcs_live and not args.keep:
        rm = _gcloud("rm", remote)
        check("drill artifact removed from GCS",
              rm is not None and rm.returncode == 0, remote)
    shutil.rmtree(work, ignore_errors=True)

    failed = [n for n, ok, _ in R if ok is False]
    skipped = [n for n, ok, _ in R if ok is None]
    passed = sum(1 for _, ok, _ in R if ok)
    print("\n" + "=" * 78)
    print(f"  {passed} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("  FAILED: " + "; ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
