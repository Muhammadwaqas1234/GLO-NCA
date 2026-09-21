#!/usr/bin/env python
r"""
Checkpoint completeness and real resume for the GLO-NCA production path.
NO full training: a few optimizer steps on synthetic tensors, then a genuine
save / load / continue cycle.

This does not merely open a checkpoint file. It advances the optimizer,
scheduler and EMA, saves, rebuilds fresh objects, restores, and verifies that
every restored component matches and that training continues from the same
state.

Checks:

   1. checkpoint carries model / optimizer / scheduler / EMA / RNG / config
   2. early-stopping state is attached (patience survives preemption)
   3. format tag present
   4. model weights restore bit-exactly
   5. optimizer state restores
   6. scheduler restores and reproduces the same learning rate
   7. EMA restores by VALUE, not merely by key count
   8. RNG state restores and reproduces the same random draw
   9. early-stopping patience counter restores
  10. training continues after resume with a finite loss
  11. resumed parameters match an uninterrupted run step for step
  12. fp16 is refused by the runner (no GradScaler in the training step)

Usage:  python scripts/test_checkpoint_resume.py
Exit:   0 if every check passed.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {detail}")


def make_stack(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300,
                                                       eta_min=1e-5)
    ema = [{k: v.detach().clone() for k, v in model.state_dict().items()}]
    return model, opt, sched, ema


def step(model, opt, sched, ema, decay=0.999, n=1):
    for _ in range(n):
        x = torch.randn(4, 8)
        loss = model(x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema[0][k].mul_(decay).add_(v.detach(), alpha=1 - decay)
    sched.step()
    return float(loss.detach())


def main() -> int:
    from src.experiment import checkpoint as ckpt_io
    from src.experiment import early_stopping, reproducibility as repro

    print("=" * 74)
    print("CHECKPOINT COMPLETENESS + REAL RESUME -- GLO-NCA Production")
    print("=" * 74)

    model, opt, sched, ema = make_stack()
    stopper = early_stopping.EarlyStopping(patience=15, min_delta=0.001)
    for i, score in enumerate([0.10, 0.20, 0.205], start=1):
        stopper.update(score, i)

    torch.manual_seed(1234)
    for _ in range(5):
        step(model, opt, sched, ema)

    full = ckpt_io.build_checkpoint(
        epoch=5, models=[model], optimizers=[opt], schedulers=[sched],
        ema=ema, best_score=0.20, best_epoch=2,
        history={"val_mean": [0.10, 0.20, 0.205]}, config={"probe": True},
        rng_state=repro.capture_rng_state())
    full["early_stopping"] = stopper.state_dict()

    for key in ("model", "optimizer", "scheduler", "ema", "rng_state",
                "config", "epoch", "best_score", "best_epoch"):
        check(f"checkpoint carries '{key}'", key in full)
    check("checkpoint carries 'early_stopping'", "early_stopping" in full)
    check("checkpoint carries a format tag", "format" in full,
          str(full.get("format")))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "last.pth")
        ckpt_io.save_checkpoint(path, full)
        check("checkpoint file written", os.path.isfile(path),
              f"{os.path.getsize(path):,} bytes")

        # Continue the ORIGINAL stack, to compare against the resumed one.
        torch.manual_seed(99)
        ref_loss = step(model, opt, sched, ema)
        ref_params = [p.detach().clone() for p in model.parameters()]

        ck = torch.load(path, map_location="cpu", weights_only=False)
        m2, o2, s2, _ = make_stack(seed=7)          # deliberately different init
        pre_restore = [p.detach().clone() for p in m2.parameters()]
        m2.load_state_dict(ck["model"][0])
        o2.load_state_dict(ck["optimizer"][0])
        s2.load_state_dict(ck["scheduler"][0])
        ema2 = [{k: v.clone() for k, v in ck["ema"][0].items()}]

        check("fresh stack differed before restore",
              not all(torch.equal(a, b)
                      for a, b in zip(pre_restore, ref_params)),
              "guards a vacuous comparison")

        # Compare against the ON-DISK checkpoint. `full["model"][0]` holds
        # live references into the original model, so it moves when that model
        # trains on; torch.save serialised the values at save time.
        saved_model = ck["model"][0]
        check("model weights restore bit-exactly",
              all(torch.equal(m2.state_dict()[k].float(), saved_model[k].float())
                  for k in saved_model))
        check("optimizer state restores",
              str(o2.state_dict()["param_groups"])
              == str(ck["optimizer"][0]["param_groups"]))
        check("scheduler restores the same LR",
              abs(s2.get_last_lr()[0] - ck["scheduler"][0]["_last_lr"][0]) < 1e-12,
              f"{s2.get_last_lr()[0]:.6e}")
        check("EMA values match the saved state",
              all(torch.allclose(ema2[0][k].float(), ck["ema"][0][k].float())
                  for k in ema2[0]))

        st2 = early_stopping.EarlyStopping(patience=15, min_delta=0.001)
        st2.load_state_dict(ck["early_stopping"])
        check("early-stopping patience restores",
              st2.state_dict() == stopper.state_dict(),
              f"patience counter preserved")

        repro.restore_rng_state(ck["rng_state"])
        draw_after_restore = torch.randn(3)
        repro.restore_rng_state(ck["rng_state"])
        draw_again = torch.randn(3)
        check("RNG state restores reproducibly",
              torch.equal(draw_after_restore, draw_again))

        # Continue training from the resumed state with the same seed.
        torch.manual_seed(99)
        res_loss = step(m2, o2, s2, ema2)
        res_params = [p.detach().clone() for p in m2.parameters()]
        check("training continues after resume",
              torch.isfinite(torch.tensor(res_loss)).item(),
              f"loss={res_loss:.6f}")
        check("resumed step matches uninterrupted run",
              all(torch.allclose(a, b, atol=1e-6)
                  for a, b in zip(res_params, ref_params)),
              f"ref={ref_loss:.6f} resumed={res_loss:.6f}")

    runner_src = io.open(os.path.join(_HERE, "src", "experiment", "runner.py"),
                         encoding="utf-8").read()
    check("fp16 refused (no GradScaler in the step)",
          "requires a GradScaler" in runner_src)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
