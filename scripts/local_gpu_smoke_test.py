#!/usr/bin/env python
r"""Phase 3.2 local GPU/CUDA validation for GLO-NCA.

Proves the REAL GLO-NCA model + REAL loss actually run on CUDA, and measures how
large a patch the local GPU can hold. Only the INPUT is synthetic; the model,
loss, optimizer and checkpoint path are the repository's real ones.

It does NOT train to convergence, does NOT touch GCP, and produces NO research
results (synthetic input -> the loss value is meaningless, and is not reported
as a metric).

Usage:
    python scripts/local_gpu_smoke_test.py                 # default patches
    python scripts/local_gpu_smoke_test.py --patches 32 48 64 96
    python scripts/local_gpu_smoke_test.py --device cpu    # force CPU (debug)

Exit 0 only if CUDA was actually used and at least one patch size passed the
full forward/backward/optimizer/checkpoint round-trip; non-zero otherwise.
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import FocalTverskyCELoss

# V2 model defaults (resolved from the canonical config, not hard-coded here).
from src.experiment.config import load_config

# PATCH -> two-level cascade sizes (same mapping the runner uses).
PATCH_TABLE = {
    32:  [[16, 16, 12], [32, 32, 24]],
    48:  [[24, 24, 16], [48, 48, 32]],
    64:  [[32, 32, 24], [64, 64, 48]],
    96:  [[48, 48, 32], [96, 96, 64]],
    128: [[64, 64, 48], [128, 128, 96]],
}


def gpu_diagnostics():
    print("=" * 64)
    print("GPU / CUDA DIAGNOSTICS")
    print("=" * 64)
    print(f"PyTorch version      : {torch.__version__}")
    print(f"torch.version.cuda   : {torch.version.cuda}")
    avail = torch.cuda.is_available()
    print(f"cuda.is_available()  : {avail}")
    if not avail:
        print(f"cuda.device_count()  : 0")
        print("\nCUDA NOT AVAILABLE")
        # diagnose the most likely cause
        if torch.version.cuda is None:
            print("Likely cause: CPU-ONLY PyTorch build installed "
                  "(torch.version.cuda is None). Install a CUDA wheel, e.g.:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("PyTorch is a CUDA build but no GPU is visible. Check the "
                  "NVIDIA driver / that a GPU is present (nvidia-smi).")
        return False
    print(f"cuda.device_count()  : {torch.cuda.device_count()}")
    p = torch.cuda.get_device_properties(0)
    print(f"device name          : {p.name}")
    print(f"total GPU memory     : {p.total_memory / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()
    print(f"allocated (start)    : {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print(f"reserved  (start)    : {torch.cuda.memory_reserved()/1e6:.1f} MB")
    return True


def build_real_model(device, cfg):
    """The REAL two-level GLO-NCA cascade with V2 defaults from the config."""
    ch = int(cfg.get("model", "channel_n"))
    hidden = int(cfg.get("model", "hidden"))
    fire = float(cfg.get("model", "fire_rate"))
    attn = bool(cfg.get("model", "use_attention"))
    spatial = bool(cfg.get("model", "use_spatial"))
    dropout = float(cfg.get("model", "dropout"))
    ca = [
        BasicNCA3D(ch, fire, device, hidden, kernel_size=7, input_channels=4,
                   use_attention=attn, use_spatial=spatial, dropout=dropout),
        BasicNCA3D(ch, fire, device, hidden, kernel_size=3, input_channels=4,
                   use_attention=attn, use_spatial=spatial, dropout=dropout),
    ]
    return ca, ch


def run_one_patch(patch, device, cfg, steps):
    """Full forward/backward/optimizer/checkpoint round-trip at one patch size,
    using the REAL high-res NCA level. Returns a result dict (never raises for
    OOM -- reports it)."""
    hi = PATCH_TABLE[patch][-1]  # high-res level size [X,Y,Z]
    res = {"patch": patch, "batch": 1, "forward": "FAIL", "backward": "FAIL",
           "optim": "FAIL", "ckpt": "FAIL", "peak_alloc_gb": None,
           "peak_reserved_gb": None, "oom": False, "seconds": None, "result": "FAIL"}
    ca, ch = build_real_model(device, cfg)
    net = ca[1]  # high-res level bears the memory cost at full patch resolution
    loss_f = FocalTverskyCELoss(
        alpha=1 - float(cfg.get("loss", "tversky_beta")),
        beta=float(cfg.get("loss", "tversky_beta")),
        gamma=float(cfg.get("loss", "focal_gamma")), ce_weight=0.5)
    opt = torch.optim.AdamW(net.parameters(),
                            lr=float(cfg.get("optimizer", "learning_rate")))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    try:
        # seed tensor in NCA channels-last layout (B, X, Y, Z, channel_n)
        x = torch.zeros((1, hi[0], hi[1], hi[2], ch), dtype=torch.float32, device=device)
        x[..., :4] = torch.rand((1, hi[0], hi[1], hi[2], 4), device=device)  # 4 modalities
        target = (torch.rand((1, hi[0], hi[1], hi[2], 3), device=device) > 0.5).float()

        out = net(x, steps=steps[-1], fire_rate=float(cfg.get("model", "fire_rate")))
        pred = out[..., 4:7]  # 3 output channels (WT/TC/ET), same slice as the agent
        res["forward"] = "PASS"

        loss = 0
        for m in range(3):
            loss = loss + loss_f(pred[..., m], target[..., m])
        loss.backward()
        res["backward"] = "PASS"

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        res["optim"] = "PASS"

        # checkpoint round-trip (torch save/load; same mechanism as the runner)
        import tempfile
        ck_path = os.path.join(tempfile.gettempdir(), f"glo_smoke_{patch}.pth")
        torch.save({"m": net.state_dict()}, ck_path)
        sd = torch.load(ck_path, map_location=device, weights_only=False)
        net.load_state_dict(sd["m"])
        with torch.no_grad():
            _ = net(x, steps=1, fire_rate=0.5)  # forward after reload
        os.remove(ck_path)
        res["ckpt"] = "PASS"
        res["result"] = "PASS"
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            res["oom"] = True
            res["result"] = "OOM"
            print(f"  patch {patch}^3: CUDA OUT OF MEMORY")
        else:
            res["result"] = f"ERROR: {exc}"
            print(f"  patch {patch}^3: ERROR {exc}")
    res["seconds"] = round(time.time() - t0, 1)

    # peak-memory read (guarded: a hard OOM can corrupt the CUDA context)
    if device.type == "cuda":
        try:
            res["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            res["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        except Exception:
            pass
        # A "PASS" whose peak exceeds physical VRAM did NOT truly fit -- Windows
        # WDDM can spill to shared host memory (very slow). Flag it honestly.
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if (res["result"] == "PASS" and res["peak_alloc_gb"] is not None
                and res["peak_alloc_gb"] > total_gb):
            res["result"] = "PASS*"  # ran, but oversubscribed VRAM (shared-mem spill)
            res["spilled"] = True

    # confirm real device placement
    try:
        res["param_device"] = str(next(net.parameters()).device)
    except Exception:
        res["param_device"] = "n/a"
    res["input_device"] = str(x.device) if res.get("forward") == "PASS" else "n/a"
    try:
        del ca, net, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, nargs="+", default=[32, 48, 64, 96])
    ap.add_argument("--device", default=None, help="force 'cpu' or 'cuda:0'")
    # Defaults to the PRODUCTION config. The previous default was
    # configs/gcp_full.yaml -- the V2 baseline -- so running this bare
    # measured an architecture the project no longer trains.
    ap.add_argument("--config",
                    default=os.path.join("configs", "glo_nca_production.yaml"))
    args = ap.parse_args()

    cuda_ok = gpu_diagnostics()
    forced_cpu = args.device == "cpu"
    if not cuda_ok and not forced_cpu:
        print("\nLOCAL GPU VALIDATION: BLOCKED (CUDA unavailable)")
        return 2

    device = torch.device(args.device) if args.device else torch.device("cuda:0")
    cfg = load_config(os.path.join(_REPO, args.config))
    steps = list(cfg.get("model", "steps"))
    print(f"\nDevice: {device.type}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 64)
    print("REAL GLO-NCA MODEL + LOSS -- per-patch round trip (batch 1)")
    print("=" * 64)
    rows = []
    for patch in sorted(args.patches):
        if patch not in PATCH_TABLE:
            print(f"  skip patch {patch} (not in table)"); continue
        r = run_one_patch(patch, device, cfg, steps)
        rows.append(r)
        vram = f"{r['peak_alloc_gb']}GB" if r["peak_alloc_gb"] is not None else "n/a"
        print(f"  {patch:>3}^3 | fwd {r['forward']:<4} | bwd {r['backward']:<4} | "
              f"opt {r['optim']:<4} | ckpt {r['ckpt']:<4} | peak {vram:<8} | "
              f"{r['seconds']}s | {r['result']}")
        if r["result"] == "OOM":
            print(f"  -> stopping at {patch}^3 (OOM); larger patches will also OOM.")
            break
        if r.get("spilled"):
            print(f"  -> {patch}^3 ran but peak VRAM ({r['peak_alloc_gb']}GB) exceeds "
                  f"the GPU's physical memory: it spilled to shared host memory "
                  f"(slow). NOT a true fit; stopping escalation.")
            break

    # device-placement confirmation (proves CUDA was really used)
    first_pass = next((r for r in rows if r["result"] in ("PASS", "PASS*")), None)
    print("\n" + "=" * 64)
    print("DEVICE PLACEMENT")
    print("=" * 64)
    if first_pass:
        print(f"Parameter device : {first_pass['param_device']}")
        print(f"Input device     : {first_pass['input_device']}")

    print("\n" + "=" * 64)
    print("MEMORY TABLE")
    print("=" * 64)
    print(f"{'Patch':<7}{'Batch':<7}{'Forward':<9}{'Backward':<10}"
          f"{'PeakVRAM':<12}{'Result':<8}")
    for r in rows:
        vram = f"{r['peak_alloc_gb']} GB" if r["peak_alloc_gb"] is not None else "n/a"
        print(f"{str(r['patch'])+'^3':<7}{r['batch']:<7}{r['forward']:<9}"
              f"{r['backward']:<10}{vram:<12}{r['result']:<8}")

    # A true fit = PASS whose peak stayed within physical VRAM (not PASS* spill).
    true_fit = [r for r in rows if r["result"] == "PASS"]
    ran_cuda = [r for r in rows if r["result"] in ("PASS", "PASS*")]
    used_cuda = (device.type == "cuda"
                 and any(r.get("param_device", "").startswith("cuda") for r in ran_cuda))
    print("\n" + "=" * 64)
    print("Legend: PASS = ran within physical VRAM | PASS* = ran but spilled to "
          "shared host memory (not a real fit) | OOM = out of memory")
    if forced_cpu:
        print("LOCAL GPU VALIDATION: NOT APPLICABLE (forced CPU debug run)")
        return 0
    if used_cuda and true_fit:
        largest = max(r["patch"] for r in true_fit)
        print(f"\nLOCAL GPU VALIDATION: PASS (CUDA confirmed)")
        print(f"Largest patch that TRULY fits this {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB GPU: {largest}^3")
        return 0
    if used_cuda:
        print("\nLOCAL GPU VALIDATION: PASS (CUDA confirmed) but NO patch fit "
              "within physical VRAM (all spilled/OOM). Use a bigger GPU for training.")
        return 0
    print("\nLOCAL GPU VALIDATION: BLOCKED (CUDA not actually used)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
