#!/usr/bin/env python
r"""
EMA validation correctness for the GLO-NCA production path.  NO TRAINING.

Validation must score the SAME weights that are saved as `best`, otherwise the
selected checkpoint is not the model that produced the selected metric. The
runner previously evaluated the raw model (`ME.evaluate(agent, ...)`) while
saving the EMA state (`weights = ema if ema is not None`).

Checks:

  1. raw and EMA states are genuinely distinguishable (guards a vacuous suite)
  2. swapping EMA in actually changes the validation output
  3. the raw weights are restored EXACTLY afterwards
  4. the saved `best` weights equal the state validation scored
  5. the swap does not disturb optimizer state or gradients
  6. EMA disabled is a no-op
  7. the real runner wires swap/restore in the correct ORDER, i.e. the raw
     weights are back before `best` and the full checkpoint are written

Usage:  python scripts/test_ema_validation.py
Exit:   0 if every check passed.
"""
from __future__ import annotations

import copy
import io
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} {detail}")


def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def make_swappers(ca, ema):
    """Mirror of the runner's EMA swap helpers."""

    def swap_in():
        if ema is None:
            return None
        raw = [{k: v.detach().clone() for k, v in m.state_dict().items()}
               for m in ca]
        for m, e in zip(ca, ema):
            m.load_state_dict({k: v.to(m.state_dict()[k].dtype)
                               for k, v in e.items()})
        return raw

    def restore(raw):
        if raw is None:
            return
        for m, r in zip(ca, raw):
            m.load_state_dict(r)

    return swap_in, restore


def main() -> int:
    print("=" * 74)
    print("EMA VALIDATION CORRECTNESS -- GLO-NCA production path")
    print("=" * 74)

    model = tiny_model()
    ca = [model]
    ema = [{k: v.detach().clone() for k, v in model.state_dict().items()}]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    diff = max((model.state_dict()[k] - ema[0][k]).abs().max().item()
               for k in ema[0])
    check("raw and EMA states differ", diff > 0.5, f"max|d|={diff:.3f}")

    x = torch.randn(8, 4)
    raw_out = model(x).detach().clone()
    swap_in, restore = make_swappers(ca, ema)
    raw_states = swap_in()
    ema_out = model(x).detach().clone()
    check("validation output changes under EMA",
          not torch.allclose(raw_out, ema_out),
          f"max|d|={(raw_out - ema_out).abs().max().item():.3f}")

    evaluated = {k: v.detach().clone() for k, v in model.state_dict().items()}
    restore(raw_states)
    check("raw weights restored exactly",
          torch.allclose(model(x), raw_out))

    check("saved best == evaluated state",
          all(torch.allclose(ema[0][k].float(), evaluated[k].float())
              for k in ema[0]))

    model2 = tiny_model()
    opt = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    model2(torch.randn(8, 4)).sum().backward()
    opt.step()
    ema2 = [{k: v.detach().clone() * 0.5
             for k, v in model2.state_dict().items()}]
    opt_before = copy.deepcopy(opt.state_dict())
    grads_before = [p.grad.detach().clone() for p in model2.parameters()
                    if p.grad is not None]
    s2, r2 = make_swappers([model2], ema2)
    r2(s2())
    grads_after = [p.grad.detach().clone() for p in model2.parameters()
                   if p.grad is not None]
    check("optimizer state untouched",
          str(opt.state_dict()) == str(opt_before))
    check("gradients untouched",
          len(grads_after) == len(grads_before)
          and all(torch.allclose(a, b)
                  for a, b in zip(grads_after, grads_before)))

    model3 = tiny_model()
    s3, r3 = make_swappers([model3], None)
    before = model3(x).detach().clone()
    r3(s3())
    check("EMA disabled is a no-op", torch.allclose(model3(x), before))

    src = io.open(os.path.join(_HERE, "src", "experiment", "runner.py"),
                  encoding="utf-8").read()
    i_swap = src.find("_raw_states = _swap_in_ema()")
    i_restore = src.find("_restore_raw(_raw_states)")
    i_best = src.find("weights = ema if ema is not None")
    i_full = src.find("full = ckpt_io.build_checkpoint(")
    check("runner wires the EMA swap", i_swap != -1 and i_restore != -1)
    check("swap precedes restore", -1 < i_swap < i_restore)
    check("restore precedes best-checkpoint save", -1 < i_restore < i_best)
    check("restore precedes full checkpoint", -1 < i_restore < i_full)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
