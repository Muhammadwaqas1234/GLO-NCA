#!/usr/bin/env python
r"""Legacy GPU memory gate for three-level V3 configs (not used for the production configuration).

Sweeps the finest-level resolution, derives the lower levels from it, and classifies
each peak as TRUE FIT (<= physical VRAM), SPILL (> VRAM) or OOM. It cannot build the
two-level production geometry (L1 48³, L2 64³); pretrain_gate.sh runs it only for legacy
configs and uses verify_glo_nca_production_config.py plus the real-data smoke for production.

Usage (legacy configs only, on the GPU VM):
    python scripts/gpu_memory_gate_v3.py --config <legacy_v3_config.yaml> --resolutions 96,128
"""
import argparse
import copy
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from src.experiment.config import load_config
from src.models.Model_GLO_NCA_V3 import build_v3_from_config
from src.losses.LossFunctions import FocalTverskyCELoss


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LEGACY three-level V3 GPU memory gate. Does NOT validate the "
                    "two-level GLO-NCA production configuration; for that use "
                    "scripts/verify_glo_nca_production_config.py.")
    ap.add_argument("--config", default=None,
                    help="a LEGACY three-level V3 config (required; no default)")
    ap.add_argument("--resolutions", default="32,48,64,96,128",
                    help="comma list of legacy level3 resolutions to probe")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    res_list = [int(r) for r in args.resolutions.split(",") if r.strip()]

    # Refuse anything but an explicit legacy config: this tool always builds a
    # derived three-level model, which is not the production architecture.
    if not args.config:
        print("LEGACY V3 GPU MEMORY GATE: REFUSED -- no --config given.")
        print("This legacy tool needs an explicit three-level V3 config. The two-level")
        print("production GLO-NCA is verified by scripts/verify_glo_nca_production_config.py.")
        return 3
    base = load_config(args.config)
    model_cfg = base.raw.get("model", {}) or {}
    level3 = model_cfg.get("level3") or {}
    if model_cfg.get("global_context") is not None or not level3.get("enabled", False):
        print(f"LEGACY V3 GPU MEMORY GATE: REFUSED -- {args.config} is not a legacy")
        print("three-level config (model.global_context set or level3 not enabled).")
        print("Measuring it here would build a different, three-level architecture.")
        print("The production GLO-NCA is verified by scripts/verify_glo_nca_production_config.py.")
        return 3

    if not torch.cuda.is_available():
        print("GPU MEMORY GATE: NOT TESTED (no CUDA GPU on this machine).")
        print("Run this on the target GCP GPU. Local CPU cannot measure VRAM fit.")
        return 2

    device = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / 1e9
    print("=" * 64)
    print(f"LEGACY V3 GPU MEMORY GATE  |  GPU: {p.name}  VRAM: {total_gb:.1f} GB")
    print(f"config: {args.config}  batch: {args.batch}")
    print("sweeping legacy level3 resolution (the activation-memory driver)")
    print("=" * 64)

    loss_f = FocalTverskyCELoss(
        alpha=1 - float(base.get("loss", "tversky_beta")),
        beta=float(base.get("loss", "tversky_beta")),
        gamma=float(base.get("loss", "focal_gamma")), ce_weight=0.5)

    results = {}
    true_fit = []
    for res in res_list:
        cfg = copy.deepcopy(base)
        # Legacy nested ratio: lower levels derived from the finest level.
        cfg.raw["model"]["level1"]["resolution"] = max(16, res // 4)
        cfg.raw["model"]["level2"]["resolution"] = max(24, res * 3 // 4)
        cfg.raw["model"]["level3"]["resolution"] = res
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            m = build_v3_from_config(cfg, 4, 3, device)
            x = torch.randn(args.batch, res, res, res, 4, device=device)
            y = (torch.rand(args.batch, 3, res, res, res, device=device) > 0.7).float()
            logits = m(x)
            loss = sum(loss_f(logits[:, c], y[:, c]) for c in range(3))
            loss.backward()
            torch.optim.AdamW(m.parameters(), lr=1e-3).step()
            peak = torch.cuda.max_memory_allocated() / 1e9
            secs = time.time() - t0
            if peak > total_gb:
                verdict = "SPILL"
                print(f"L3={res:>3}^3 : SPILL     peak {peak:5.2f}GB > {total_gb:.1f}GB "
                      f"(host-memory pressure, {secs:.0f}s) -- NOT a true fit")
            else:
                verdict = "TRUE FIT"
                true_fit.append(res)
                print(f"L3={res:>3}^3 : TRUE FIT  peak {peak:5.2f}GB  ({secs:.1f}s)")
            results[res] = (verdict, peak)
            del m, x, y, logits, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results[res] = ("OOM", None)
                print(f"L3={res:>3}^3 : OOM       needed > {total_gb:.1f}GB")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # larger resolutions will also OOM
                print(f"  -> stopping sweep at first OOM (larger will also OOM).")
                break
            print(f"L3={res:>3}^3 : ERROR    {str(e)[:60]}")
            results[res] = ("ERROR", None)
            break

    print("=" * 64)
    legacy = [r for r in (96, 128) if r in res_list]
    legacy_fit = all(results.get(r, ("", None))[0] == "TRUE FIT" for r in legacy) and legacy
    print("Largest TRUE FIT legacy level3 on this GPU:",
          f"{max(true_fit)}^3" if true_fit else "none")
    for r in legacy:
        v = results.get(r, ("NOT REACHED", None))[0]
        print(f"Legacy V3 {r}^3: {v}")
    if legacy and legacy_fit:
        print("\nLEGACY V3 GPU MEMORY GATE: PASS (legacy 96^3 + 128^3 TRUE FIT)")
        return 0
    if legacy:
        print("\nLEGACY V3 GPU MEMORY GATE: FAIL for the legacy resolutions on this GPU.")
        return 1
    print("\nLEGACY V3 GPU MEMORY GATE: informational (legacy 96/128 not in sweep).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
