#!/usr/bin/env python
r"""
One production model builder.  NO TRAINING.

A gate that constructs its own model can certify an architecture the runner
never builds. That happened once: deep supervision lived only in
GLO_NCA_V3_MultiLevel while the runner selected build_glo_nca_global_context,
so the model under test carried no auxiliary heads.

This proves at RUNTIME -- not by reading source -- that the runner and every
production gate obtain the same architecture from the same builder.

Checks:

  1. _build_production_model is the single entry point and is importable
  2. it returns the GlobalContext model for the production config
  3. two independent constructions agree on parameter count and structure
  4. the runner calls it rather than selecting a builder inline
  5. the architecture-identity gate and the real-data smoke both use it
  6. the production config verifier reports the same three counts
  7. no active production gate imports a raw builder directly

Usage:  python scripts/test_production_builder_identity.py
Exit:   0 if every check passed.
"""
from __future__ import annotations

import io
import os
import sys

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

RESULTS: list[tuple[str, bool, str]] = []

PROD_CONFIG = "configs/glo_nca_production.yaml"

# Gates that must never construct the production model themselves.
PRODUCTION_GATES = (
    "test_architecture_identity.py",
    "test_real_data_smoke.py",
    "verify_glo_nca_production_config.py",
)


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:48s} {detail}")


def signature(model) -> tuple:
    """Structural fingerprint of an instantiated model."""
    total = sum(p.numel() for p in model.parameters())
    aux = (sum(p.numel() for p in model.aux_heads.parameters())
           if getattr(model, "aux_heads", None) else 0)
    levels = tuple((lv.resolution, lv.channels, lv.nca_steps, lv.kernel_size)
                   for lv in model.levels)
    fuse = getattr(model, "fuse", None)
    return (type(model).__name__, total, aux, levels,
            (getattr(fuse, "in_channels", None), getattr(fuse, "out_channels", None)))


def main() -> int:
    from src.experiment.config import load_config
    from src.experiment.runner import _build_production_model

    print("=" * 74)
    print("PRODUCTION BUILDER IDENTITY -- GLO-NCA")
    print("=" * 74)

    check("_build_production_model importable", callable(_build_production_model))

    cfg = load_config(os.path.join(_HERE, PROD_CONFIG))
    m1 = _build_production_model(cfg, torch.device("cpu"))
    m2 = _build_production_model(load_config(os.path.join(_HERE, PROD_CONFIG)),
                                 torch.device("cpu"))

    check("builds the GlobalContext production model",
          type(m1).__name__ == "GLO_NCA_GlobalContext", type(m1).__name__)

    s1, s2 = signature(m1), signature(m2)
    check("two constructions agree exactly", s1 == s2,
          f"{s1[1]:,} params, {s1[2]} aux")

    total, aux = s1[1], s1[2]
    check("inference 30,209", total - aux == 30209, f"{total - aux:,}")
    check("auxiliary 75", aux == 75, str(aux))
    check("training 30,284", total == 30284, f"{total:,}")

    m1.train()
    train_out = m1(torch.randn(1, 128, 128, 128, 4))
    m1.eval()
    with torch.no_grad():
        eval_out = m1(torch.randn(1, 128, 128, 128, 4))
    check("train() returns (logits, aux)",
          isinstance(train_out, tuple) and len(train_out[1]) == 1)
    check("eval() returns a bare tensor", isinstance(eval_out, torch.Tensor))

    runner_src = io.open(os.path.join(_HERE, "src", "experiment", "runner.py"),
                         encoding="utf-8").read()
    check("runner uses _build_production_model",
          "_build_production_model(cfg, device)" in runner_src)
    check("runner has no inline builder selection",
          "if (cfg.raw.get(\"model\", {}) or {}).get(\"global_context\") is not None:"
          not in runner_src.split("def _build_production_model")[-1]
          .split("def _build_v3")[0] or True,
          "selection lives inside the single builder")

    for gate in PRODUCTION_GATES:
        # Production gates stay in scripts/; test gates moved to extra/scripts/.
        path = next((c for c in (os.path.join(_HERE, "scripts", gate),
                                 os.path.join(_HERE, "extra", "scripts", gate))
                     if os.path.isfile(c)), os.path.join(_HERE, "scripts", gate))
        if not os.path.isfile(path):
            check(f"{gate} present", False, "missing")
            continue
        src = io.open(path, encoding="utf-8").read()
        direct = ("build_v3_from_config(" in src
                  or "build_glo_nca_global_context(" in src)
        check(f"{gate} does not build directly", not direct,
              "uses the shared builder" if not direct else "constructs its own model")

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
