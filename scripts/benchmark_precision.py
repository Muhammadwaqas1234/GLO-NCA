#!/usr/bin/env python3
r"""Measure FP32 / BF16 / FP16 on the production GLO-NCA model.

Answers one engineering question: which precision is fastest and numerically
sound on *this* machine. It is not evidence about segmentation quality, and
it changes nothing about the model -- architecture, optimizer, loss, NCA
steps, kernels and split are read from the production config and left alone.

Support is probed, never assumed. BF16 needs compute capability 8.0+ (it is
emulated in software on Turing, so a "works" result there is not a
recommendation), and a mode that cannot run is recorded as NOT SUPPORTED
rather than quietly skipped.

Every number printed is measured. Nothing here is estimated.

    python scripts/benchmark_precision.py [--config PATH] [--iters N]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


def _device_report(torch) -> Dict[str, Any]:
    """Everything about the device that could explain a timing result."""
    rep: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else "NOT AVAILABLE"),
    }
    if not torch.cuda.is_available():
        rep["device"] = "cpu"
        return rep
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    rep.update({
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "total_vram_mb": round(props.total_memory / 1024 ** 2, 1),
        "multiprocessors": props.multi_processor_count,
        # bf16 is NATIVE only on Ampere (8.0) and later. On Turing it runs but
        # is emulated, so "supported" there does not mean "fast".
        "bf16_native": cap >= (8, 0),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "fp16_tensor_cores": cap >= (7, 0),
    })
    return rep


def _gpu_clocks() -> Dict[str, Any]:
    """Clock/thermal state. A throttled GPU invalidates absolute timings."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,clocks.mem,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        a = [x.strip() for x in r.stdout.strip().split("\n")[0].split(",")]

        def num(v, cast):
            try:
                return cast(v)
            except Exception:
                return "NOT AVAILABLE"

        sm, sm_max = num(a[0], int), num(a[1], int)
        out = {"sm_clock_mhz": sm, "sm_clock_max_mhz": sm_max,
               "memory_clock_mhz": num(a[2], int),
               "temperature_c": num(a[3], int),
               "power_w": num(a[4], float)}
        if isinstance(sm, int) and isinstance(sm_max, int) and sm_max:
            out["clock_pct_of_max"] = round(100 * sm / sm_max, 1)
            out["throttled"] = sm < 0.75 * sm_max
        return out
    except Exception:
        return {"status": "NOT AVAILABLE (nvidia-smi absent or failed)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/glo_nca_production.yaml")
    ap.add_argument("--iters", type=int, default=8,
                    help="timed iterations per precision (default 8)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="untimed warm-up iterations (default 3)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="output JSON (default reports/benchmarks/"
                         "precision_benchmark.json)")
    args = ap.parse_args()

    import torch
    import yaml
    from src.experiment.config import Config
    from src.experiment.runner import _build_production_model
    from src.losses.LossFunctions import FocalTverskyCELoss

    cfg_path = os.path.join(_ROOT, args.config)
    if not os.path.isfile(cfg_path):
        print(f"FAILED: config not found: {cfg_path}")
        return 2
    raw = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg = Config(raw=raw)

    print("=" * 84)
    print("GLO-NCA PRECISION BENCHMARK")
    print("=" * 84)
    dev_rep = _device_report(torch)
    for k, v in dev_rep.items():
        print(f"  {k:24s} {v}")
    clocks = _gpu_clocks()
    print()
    for k, v in clocks.items():
        print(f"  {k:24s} {v}")
    if clocks.get("throttled"):
        print("\n  *** GPU IS THROTTLED. Absolute times are inflated; the")
        print("      RATIO between precisions remains the usable result. ***")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_production_model(cfg, device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_aux = (sum(p.numel() for p in model.aux_heads.parameters())
             if getattr(model, "aux_heads", None) else 0)

    # Identity first: a benchmark of the wrong model is worse than none.
    m_ = raw.get("model", {})
    ident = {
        "parameters": n_params,
        "level1": [model.levels[0].resolution, model.levels[0].channels,
                   model.levels[0].nca_steps, model.levels[0].kernel_size],
        "level2": [model.levels[1].resolution, model.levels[1].channels,
                   model.levels[1].nca_steps, model.levels[1].kernel_size],
        "level3_absent": len(model.levels) == 2,
        "total_nca_steps": sum(l.nca_steps for l in model.levels),
        "spatial_gc_kernel": int(m_.get("spatial_kernel_size", 7)),
        "patchify": bool((raw.get("data", {}).get("training_patch") or {})
                         .get("enabled", False)),
        "roi_fraction": model.roi_fraction,
    }
    print()
    print(f"  architecture             {n_params:,} params | "
          f"L1 {ident['level1'][0]}^3 k{ident['level1'][3]} | "
          f"L2 {ident['level2'][0]}^3 k{ident['level2'][3]} | "
          f"{ident['total_nca_steps']} steps | GC k{ident['spatial_gc_kernel']}")
    if (n_params - n_aux, n_aux) != (30209, 75):
        print(f"\nFAILED: expected 30,209 inference + 75 auxiliary parameters, "
              f"built {n_params - n_aux:,} + {n_aux}.")
        print("Refusing to benchmark a model that is not the production one.")
        return 3

    vol = int((raw.get("data", {}).get("training_patch") or {})
              .get("working_volume", 96))
    fine = model.levels[-1].resolution
    loss_fn = FocalTverskyCELoss(
        alpha=1.0 - float(cfg.get("loss", "tversky_beta")),
        beta=float(cfg.get("loss", "tversky_beta")),
        gamma=float(cfg.get("loss", "focal_gamma")),
        ce_weight=float(cfg.get("loss", "ce_weight", 0.5)))

    # Candidate precisions. fp32 always runs; the half formats are probed.
    candidates: List[Dict[str, Any]] = [
        {"name": "fp32", "dtype": None, "supported": True, "reason": ""},
    ]
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        candidates.append({
            "name": "bf16", "dtype": torch.bfloat16,
            "supported": bool(torch.cuda.is_bf16_supported()),
            "reason": ("native (compute capability >= 8.0)" if cap >= (8, 0)
                       else "EMULATED on this device (capability < 8.0): it "
                            "runs, but a good result here is not a "
                            "recommendation")})
        candidates.append({
            "name": "fp16", "dtype": torch.float16,
            "supported": cap >= (7, 0),
            "reason": ("native tensor cores" if cap >= (7, 0)
                       else "no fp16 tensor cores on this device")})
    else:
        for nm in ("bf16", "fp16"):
            candidates.append({"name": nm, "dtype": None, "supported": False,
                               "reason": "NOT SUPPORTED: no CUDA device"})

    print()
    print("=" * 84)
    print(f"MEASURED  (batch {args.batch}, {args.warmup} warm-up + "
          f"{args.iters} timed iterations each)")
    print("=" * 84)
    print(f"  {'precision':10s} {'fwd_ms':>9s} {'bwd_ms':>9s} {'iter_ms':>9s} "
          f"{'std':>7s} {'alloc_MB':>9s} {'resv_MB':>9s} {'finite':>7s}")

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        if not cand["supported"]:
            print(f"  {cand['name']:10s} {'NOT SUPPORTED':>46s}   {cand['reason']}")
            results.append({"precision": cand["name"], "status": "NOT SUPPORTED",
                            "reason": cand["reason"]})
            continue
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            opt = torch.optim.AdamW(model.parameters(),
                                    lr=float(cfg.get("optimizer",
                                                     "learning_rate")))
            x = torch.randn(args.batch, vol, vol, vol, 4, device=device)
            tgt = torch.randint(0, 2, (args.batch, fine, fine, fine, 3),
                                device=device).float()
            scaler = (torch.amp.GradScaler("cuda")
                      if cand["name"] == "fp16" and device.type == "cuda"
                      else None)
            fwd, bwd, finite = [], [], True
            model.train()
            for i in range(args.warmup + args.iters):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.amp.autocast(device.type, dtype=cand["dtype"],
                                        enabled=cand["dtype"] is not None):
                    out = model(x)
                out = out.float().permute(0, 2, 3, 4, 1).contiguous()
                loss = sum(loss_fn(out[..., j], tgt[..., j]) for j in range(3))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                if not torch.isfinite(loss).item():
                    finite = False
                if i >= args.warmup:
                    fwd.append((t1 - t0) * 1000)
                    bwd.append((t2 - t1) * 1000)
            tot = [a + b for a, b in zip(fwd, bwd)]
            alloc = (torch.cuda.max_memory_allocated() / 1024 ** 2
                     if device.type == "cuda" else "NOT AVAILABLE")
            resv = (torch.cuda.max_memory_reserved() / 1024 ** 2
                    if device.type == "cuda" else "NOT AVAILABLE")
            rec = {
                "precision": cand["name"], "status": "MEASURED",
                "note": cand["reason"],
                "forward_ms": {"mean": statistics.mean(fwd),
                               "median": statistics.median(fwd)},
                "backward_ms": {"mean": statistics.mean(bwd),
                                "median": statistics.median(bwd)},
                "iteration_ms": {
                    "mean": statistics.mean(tot),
                    "median": statistics.median(tot),
                    "std": statistics.pstdev(tot) if len(tot) > 1 else 0.0,
                    "min": min(tot), "max": max(tot)},
                "samples_per_s": args.batch / (statistics.mean(tot) / 1000),
                "peak_allocated_mb": alloc, "peak_reserved_mb": resv,
                "numerically_finite": finite,
                "iterations": len(tot), "warmup": args.warmup,
            }
            results.append(rec)
            a_s = f"{alloc:9.0f}" if isinstance(alloc, float) else f"{'n/a':>9s}"
            r_s = f"{resv:9.0f}" if isinstance(resv, float) else f"{'n/a':>9s}"
            print(f"  {cand['name']:10s} {statistics.median(fwd):8.1f} "
                  f"{statistics.median(bwd):8.1f} {statistics.median(tot):8.1f} "
                  f"{rec['iteration_ms']['std']:6.1f} {a_s} {r_s} "
                  f"{str(finite):>7s}")
            del opt, x, tgt
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  {cand['name']:10s} {'OOM':>46s}")
            results.append({"precision": cand["name"], "status": "NOT MEASURED",
                            "reason": "CUDA out of memory"})
            torch.cuda.empty_cache()
        except Exception as exc:                       # noqa: BLE001
            print(f"  {cand['name']:10s} FAILED: {str(exc)[:50]}")
            results.append({"precision": cand["name"], "status": "NOT MEASURED",
                            "reason": str(exc)[:200]})

    measured = [r for r in results if r.get("status") == "MEASURED"
                and r.get("numerically_finite")]
    recommendation, rationale = "NOT DETERMINED", "no precision measured cleanly"
    if measured:
        best = min(measured, key=lambda r: r["iteration_ms"]["median"])
        recommendation = best["precision"]
        rationale = (f"fastest numerically-valid precision measured on "
                     f"{dev_rep.get('device')}: "
                     f"{best['iteration_ms']['median']:.1f} ms/iteration")
        if clocks.get("throttled"):
            rationale += " (GPU throttled; ratio valid, absolute inflated)"

    print()
    print(f"  RECOMMENDED: {recommendation}  -- {rationale}")
    print("  This is a SPEED result only. It says nothing about segmentation")
    print("  quality, and it changes no scientific setting.")

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": args.config,
        "architecture_identity": ident,
        "device": dev_rep,
        "gpu_clocks_at_start": clocks,
        "batch_size": args.batch,
        "input_geometry": f"{vol}^3 x 4 modalities",
        "output_geometry": f"{fine}^3 x 3 regions",
        "results": results,
        "recommended_precision": recommendation,
        "recommendation_rationale": rationale,
        "scope": ("Engineering speed benchmark. NOT evidence of segmentation "
                  "quality. Architecture, optimizer, loss, NCA steps, kernels "
                  "and split are unchanged."),
    }
    out = args.out or os.path.join(_ROOT, "reports", "benchmarks",
                                   "precision_benchmark.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n  written: {os.path.relpath(out, _ROOT)}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
