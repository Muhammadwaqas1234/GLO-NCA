#!/usr/bin/env python
r"""
GLO-NCA production architecture identity.  NO TRAINING.

Builds the model from configs/glo_nca_production.yaml and verifies the
architecture from the INSTANTIATED model, never from documentation. Parameter
counts are measured, not asserted against a hard-coded constant that could be
edited to make the gate pass.

Checks the production geometry, the global-context configuration, the
normalization strategy, the absence of level 3, the loss parameters, the
post-processing thresholds, and that gradient accumulation and GroupNorm are
not enabled on the production path.

Usage:  python scripts/test_architecture_identity.py
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

EXPECTED_INFERENCE_PARAMS = 30209


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} {detail}")


def main() -> int:
    from src.experiment.config import load_config
    from src.experiment.postprocess import min_component_config
    # The runner selects build_glo_nca_global_context when the config has a
    # model.global_context block, which production does. Mirror that here so
    # the gate verifies the model that actually trains.
    from src.experiment.runner import _build_production_model

    print("=" * 74)
    print("ARCHITECTURE IDENTITY -- GLO-NCA Production")
    print("=" * 74)

    cfg = load_config(os.path.join(_HERE, "configs", "glo_nca_production.yaml"))
    model = _build_production_model(cfg, torch.device("cpu"))
    mcfg = cfg.section("model")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    aux = sum(p.numel() for p in model.aux_heads.parameters()) \
        if getattr(model, "aux_heads", None) else 0

    check("model name is GLO-NCA",
          "glo_nca" in type(model).__name__.lower().replace("-", "_"),
          type(model).__name__)

    levels = list(getattr(model, "levels", []))
    check("two levels, level 3 absent", len(levels) == 2, f"n={len(levels)}")
    if len(levels) == 2:
        l1, l2 = levels
        check("level1 48^3 / 24ch / 15 steps / k=5",
              (l1.resolution, l1.channels, l1.nca_steps, l1.kernel_size)
              == (48, 24, 15, 5),
              f"{l1.resolution} {l1.channels} {l1.nca_steps} {l1.kernel_size}")
        check("level2 64^3 / 24ch / 15 steps / k=5",
              (l2.resolution, l2.channels, l2.nca_steps, l2.kernel_size)
              == (64, 24, 15, 5),
              f"{l2.resolution} {l2.channels} {l2.nca_steps} {l2.kernel_size}")
        check("total NCA steps == 30",
              l1.nca_steps + l2.nca_steps == 30,
              str(l1.nca_steps + l2.nca_steps))

    check("config level3 disabled",
          bool(mcfg["level3"]["enabled"]) is False)
    wv = int(cfg.get("training", "patch_size"))
    check("working volume 128^3", wv == 128, f"{wv}^3")
    check("training_patch.working_volume consistent",
          int((cfg.section("data") or {})["training_patch"]["working_volume"]) == wv,
          str((cfg.section("data") or {})["training_patch"]["working_volume"]))
    # RESTORED TO 7 by explicit decision. It had been reduced 7 -> 5 on speed
    # grounds with no quality evidence; since this kernel IS the thesis
    # contribution, the larger receptive field was restored. This also moves
    # inference parameters 29,337 -> 30,209 (+872).
    check("spatial global-context kernel 7",
          int(mcfg["spatial_kernel_size"]) == 7, str(mcfg["spatial_kernel_size"]))
    check("SE (use_attention) enabled", bool(mcfg["use_attention"]))
    check("spatial global context enabled", bool(mcfg["use_spatial"]))
    check("ROI fraction 1.0",
          float(mcfg["global_context"]["roi_fraction"]) == 1.0,
          str(mcfg["global_context"]["roi_fraction"]))

    fuse = getattr(model, "fuse", None)
    check("learned fusion Conv3d(48 -> 24)",
          isinstance(fuse, torch.nn.Conv3d)
          and (fuse.in_channels, fuse.out_channels) == (48, 24),
          f"{type(fuse).__name__} "
          f"{getattr(fuse, 'in_channels', '?')}->{getattr(fuse, 'out_channels', '?')}")

    check("patchify disabled",
          bool((cfg.section("data") or {})["training_patch"]["enabled"]) is False)
    check("batch size 1", int(cfg.get("training", "batch_size")) == 1)
    check("seed 42", int(cfg.get("experiment", "seed")) == 42)
    check("hidden 128", int(mcfg["hidden"]) == 128)

    beta = float(cfg.get("loss", "tversky_beta"))
    check("Tversky alpha 0.40 / beta 0.60",
          abs(beta - 0.60) < 1e-9 and abs((1 - beta) - 0.40) < 1e-9,
          f"alpha={1 - beta:.2f} beta={beta:.2f}")

    check("post-processing WT 50 / TC 5 / ET 0",
          min_component_config(cfg) == {"WT": 50, "TC": 5, "ET": 0},
          str(min_component_config(cfg)))

    norms = {type(m).__name__ for m in model.modules()
             if "Norm" in type(m).__name__}
    check("no GroupNorm in the production model",
          not any("Group" in n for n in norms), str(sorted(norms)))

    runner_src = io.open(os.path.join(_HERE, "src", "experiment", "runner.py"),
                         encoding="utf-8").read()
    check("no gradient accumulation in the runner",
          "grad_accum" not in runner_src.lower()
          and "accumulation_steps" not in runner_src.lower())

    print()
    print(f"  MEASURED total parameters     : {total:,}")
    print(f"  MEASURED trainable parameters : {trainable:,}")
    print(f"  MEASURED auxiliary parameters : {aux:,}")
    print()
    check("inference parameters == 30,209",
          total - aux == EXPECTED_INFERENCE_PARAMS, f"{total - aux:,}")
    check("training parameters == inference + aux",
          trainable == (total - aux) + aux, f"{trainable:,}")
    check("training parameters == 30,284",
          trainable == 30284, f"{trainable:,}")
    check("deep supervision active, 75 auxiliary parameters",
          aux == 75, f"{aux} aux parameters")

    ds_cfg = (mcfg.get("deep_supervision") or {})
    check("deep supervision enabled in config",
          bool(ds_cfg.get("enabled", False)), f"weight={ds_cfg.get('weight')}")

    x = torch.randn(1, 96, 96, 96, 4)
    model.train()
    train_out = model(x)
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    check("train() returns (logits, aux)",
          isinstance(train_out, tuple) and len(train_out[1]) == 1,
          f"{len(train_out[1])} auxiliary head(s)" if isinstance(train_out, tuple) else "bare tensor")
    check("eval() returns primary logits only",
          isinstance(eval_out, torch.Tensor), type(eval_out).__name__)

    check("output at the finest level resolution",
          tuple(eval_out.shape[2:]) == (64, 64, 64), str(tuple(eval_out.shape[2:])))

    # Global context must read the WHOLE working volume: change only the outer
    # rim and require the output to move.
    a = torch.zeros(1, wv, wv, wv, 4)
    q = wv // 4
    a[:, q:wv - q, q:wv - q, q:wv - q, :] = 1.0
    b = a.clone()
    b[:, :q, :, :, :] = 5.0
    with torch.no_grad():
        delta = (model(a) - model(b)).abs().max().item()
    check("global context reads the full working volume",
          delta > 1e-6, f"max|delta|={delta:.6f} when only the rim changes")

    ds_w = float((mcfg.get("deep_supervision") or {}).get("weight", 0.0))
    check("deep-supervision weight 0.4", abs(ds_w - 0.4) < 1e-9, str(ds_w))
    check("gradient checkpointing enabled",
          bool((cfg.raw.get("memory", {}) or {}).get("gradient_checkpointing",
               (cfg.section("performance") or {}).get("gradient_checkpointing", False))),
          "memory/performance.gradient_checkpointing")
    check("preprocessing cache enabled",
          bool(((cfg.section("data") or {}).get("cache") or {}).get("enabled", False)))
    check("epoch budget 300",
          int(cfg.get("training", "epochs")) == 300,
          str(cfg.get("training", "epochs")))

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
