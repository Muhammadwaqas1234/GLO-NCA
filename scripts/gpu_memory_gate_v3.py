#!/usr/bin/env python
r"""GLO-NCA V3 -- production GPU memory gate (run ON THE TARGET GPU, e.g. GCP).

Measures whether the V3 production path fits the *actual* GPU at the thesis
resolutions. It sweeps the model's finest-level (level3) resolution -- the real
activation-memory driver -- through the production values and classifies each:

    TRUE FIT  : peak allocated <= physical VRAM
    SPILL     : peak > physical VRAM (host-memory pressure; NOT a real fit)
    OOM       : allocation failed

It does NOT train and NOT modify the architecture. If 96^3/128^3 OOM on the
chosen GPU, that is reported as a hardware-capacity result (pick a larger GPU),
never silently worked around.

Usage (on the GPU VM):
    python scripts/gpu_memory_gate_v3.py --config configs/v3_multilevel.yaml
    python scripts/gpu_memory_gate_v3.py --resolutions 96,128     # production only
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("configs", "v3_multilevel.yaml"))
    ap.add_argument("--resolutions", default="32,48,64,96,128",
                    help="comma list of level3 resolutions to probe")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    res_list = [int(r) for r in args.resolutions.split(",") if r.strip()]

    if not torch.cuda.is_available():
        print("GPU MEMORY GATE: NOT TESTED (no CUDA GPU on this machine).")
        print("Run this on the target GCP GPU. Local CPU cannot measure VRAM fit.")
        return 2

    device = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / 1e9
    print("=" * 64)
    print(f"V3 GPU MEMORY GATE  |  GPU: {p.name}  VRAM: {total_gb:.1f} GB")
    print(f"config: {args.config}  batch: {args.batch}")
    print("sweeping level3 resolution (the real activation-memory driver)")
    print("=" * 64)

    base = load_config(args.config)
    loss_f = FocalTverskyCELoss(
        alpha=1 - float(base.get("loss", "tversky_beta")),
        beta=float(base.get("loss", "tversky_beta")),
        gamma=float(base.get("loss", "focal_gamma")), ce_weight=0.5)

    results = {}
    true_fit = []
    for res in res_list:
        cfg = copy.deepcopy(base)
        # keep the nested multi-scale ratio; level3 drives memory
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
    prod = [r for r in (96, 128) if r in res_list]
    prod_fit = all(results.get(r, ("", None))[0] == "TRUE FIT" for r in prod) and prod
    print("Largest TRUE FIT level3 on this GPU:",
          f"{max(true_fit)}^3" if true_fit else "none")
    for r in prod:
        v = results.get(r, ("NOT REACHED", None))[0]
        print(f"Production {r}^3: {v}")
    if prod and prod_fit:
        print("\nV3 GPU MEMORY GATE: PASS (production 96^3 + 128^3 TRUE FIT)")
        return 0
    if prod:
        print("\nV3 GPU MEMORY GATE: FAIL for production resolution on this GPU.")
        print("Pick a larger-VRAM GPU. Do NOT reduce the thesis architecture.")
        return 1
    print("\nV3 GPU MEMORY GATE: informational (production 96/128 not in sweep).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
