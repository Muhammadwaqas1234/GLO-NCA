#!/usr/bin/env python
r"""
PHASE 3 — GLO-NCA V3 local GPU memory audit.   LOCAL ONLY. NO TRAINING. NO GCP.

Characterises GPU/CPU memory and runtime of the REAL production V3 model across a
resolution sweep (32/48/64/96/128) so the practical local hardware boundary is
established from measurement rather than assumption.

DESIGN NOTES
------------
* **Real model.** Uses ``GLO_NCA_V3_MultiLevel`` built from the production config
  (``configs/historical/v3_multilevel_ckpt.yaml``): production channels (24/24/16), hidden
  128, fire rate 0.6, SE + spatial GC, gradient checkpointing ON, batch 1.
  No toy model, no architecture change.

* **What the sweep varies.** Only the FINEST level's resolution is swept; levels
  1 and 2 keep their production values (32^3 and 96^3). At ``--res 128`` the
  configuration is therefore EXACTLY production (32/96/128, 20+20+10 = 50 steps).
  Lower rungs are diagnostic points on the same production architecture, not
  alternative models. Level 2 is clamped to not exceed the finest level so the
  hierarchy stays coarse->fine.

* **Process isolation.** Each resolution runs in a FRESH subprocess. A CUDA OOM
  can leave an allocator in a degraded state, and Windows sysmem fallback can
  leave the driver under pressure; isolation guarantees one rung cannot bias or
  crash the next. The parent only aggregates JSON.

* **Windows safety.** The CUDA driver on Windows spills beyond physical VRAM into
  host RAM instead of raising OOM. A peak above device VRAM is therefore a
  SPILL, never a FIT. Each child is given a wall-clock timeout and the parent
  refuses to launch a rung if free host RAM is below a safety floor.

* **Timing.** ``torch.cuda.synchronize()`` brackets every timed region; CUDA is
  asynchronous and unsynchronised wall-clock would measure kernel launch only.

Reproduce:
    python scripts/memory_audit_v3.py
    python scripts/memory_audit_v3.py --resolutions 32,48,64,96,128 --repeats 3
    python scripts/memory_audit_v3.py --res 128 --mode fwdbwd --child   # one rung

Outputs:
    reports/phase3_memory_audit.json      machine-readable raw results
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PROD_CFG = os.path.join(_ROOT, "configs", "v3_multilevel_ckpt.yaml")

# Safety floors. The 128^3 rung is the dangerous one: on a 6 GB card it spills to
# host RAM, so the parent refuses to start a rung without headroom, and kills a
# child that exceeds the time budget rather than letting the machine thrash.
MIN_FREE_HOST_GB = 2.0
CHILD_TIMEOUT_S = {32: 300, 48: 300, 64: 420, 96: 600, 128: 900}


# --------------------------------------------------------------------------- #
# GPU utilisation sampling (background thread, nvidia-smi)
# --------------------------------------------------------------------------- #
class _UtilSampler:
    """Samples `nvidia-smi` utilisation in a thread. Reports NOT RELIABLY
    MEASURED rather than inventing a number when sampling is unavailable or the
    window is too short to collect meaningful samples."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples: List[int] = []
        self._stop = False
        self._t = None
        self.available = False

    def _run(self):
        while not self._stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if out.returncode == 0:
                    v = out.stdout.strip().splitlines()[0].strip()
                    self.samples.append(int(v))
                    self.available = True
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        import threading
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        if self._t:
            self._t.join(timeout=3)
        return False

    def result(self) -> Dict[str, Any]:
        # Fewer than 3 samples over a short benchmark cannot characterise
        # utilisation; say so instead of reporting a misleading single reading.
        if not self.available or len(self.samples) < 3:
            return {"status": "NOT RELIABLY MEASURED",
                    "samples": len(self.samples),
                    "note": "benchmark shorter than the sampling window, or "
                            "nvidia-smi unavailable"}
        return {"status": "MEASURED", "samples": len(self.samples),
                "mean_pct": round(sum(self.samples) / len(self.samples), 1),
                "max_pct": max(self.samples), "min_pct": min(self.samples),
                "method": "nvidia-smi utilization.gpu polled every "
                          f"{self.interval * 1000:.0f} ms in a background thread"}


# --------------------------------------------------------------------------- #
# child: one resolution, one mode
# --------------------------------------------------------------------------- #
def run_one(res: int, mode: str, repeats: int) -> Dict[str, Any]:
    """Execute ONE rung in this process. Returns a result dict (never raises)."""
    import gc
    import torch
    from src.experiment.config import load_config
    from src.models.Model_GLO_NCA_V3 import GLO_NCA_V3_MultiLevel, LevelSpec
    from src.losses.LossFunctions import FocalTverskyCELoss
    from src.profiling.memory import MemorySampler

    rec: Dict[str, Any] = {"resolution": res, "mode": mode,
                           "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime())}
    mem = MemorySampler(enabled=True)
    total_vram_mb = mem.device_total_mb()
    rec["device_total_vram_mb"] = total_vram_mb

    if not torch.cuda.is_available():
        rec.update(status="SKIP", error="no CUDA device")
        return rec

    dev = torch.device("cuda:0")

    # ---- production model geometry -------------------------------------
    cfg = load_config(PROD_CFG)
    m = cfg.raw["model"]
    l1r = int(m["level1"]["resolution"])          # 32  (production, fixed)
    l2r = min(int(m["level2"]["resolution"]), res)  # 96, clamped <= finest
    levels = [
        LevelSpec(True, l1r, int(m["level1"]["channels"]),
                  int(m["level1"]["nca_steps"]), int(m["level1"]["kernel_size"])),
        LevelSpec(True, l2r, int(m["level2"]["channels"]),
                  int(m["level2"]["nca_steps"]), int(m["level2"]["kernel_size"])),
        LevelSpec(True, res, int(m["level3"]["channels"]),
                  int(m["level3"]["nca_steps"]), int(m["level3"]["kernel_size"])),
    ]
    rec["levels"] = [{"resolution": lv.resolution, "channels": lv.channels,
                      "nca_steps": lv.nca_steps, "kernel_size": lv.kernel_size}
                     for lv in levels]
    rec["total_nca_steps"] = sum(lv.nca_steps for lv in levels)
    rec["gradient_checkpointing"] = bool(
        (cfg.raw.get("memory") or {}).get("gradient_checkpointing", False))
    rec["batch_size"] = 1
    rec["is_exact_production_geometry"] = (
        l1r == 32 and l2r == 96 and res == 128 and rec["total_nca_steps"] == 50)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    rec["host_before"] = mem.sample()

    fwd_ms: List[float] = []
    bwd_ms: List[float] = []
    status = "FIT"
    err = None

    try:
        torch.manual_seed(42)
        model = GLO_NCA_V3_MultiLevel(
            4, 3, levels,
            fire_rate=float(m["fire_rate"]),
            use_attention=bool(m["use_attention"]),
            use_spatial=bool(m["use_spatial"]),
            dropout=float(m["dropout"]),
            fusion=str((m.get("feature_fusion") or {}).get("type", "concat")),
            hidden_size=int(m["hidden"]), device=dev,
            gradient_checkpointing=rec["gradient_checkpointing"])
        rec["parameters"] = sum(p.numel() for p in model.parameters())

        # Deterministic synthetic input at the true pipeline shape
        # (B, X, Y, Z, 4) -- no dataset needed for a memory characterisation.
        torch.manual_seed(1234)
        x = torch.randn(1, res, res, res, 4, device=dev)
        target = (torch.rand(1, res, res, res, device=dev) > 0.85).float()
        lossf = FocalTverskyCELoss(0.25, 0.75, 1.33, 0.5)

        with _UtilSampler() as util:
            for i in range(repeats):
                if mode == "forward":
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        y = model(x)
                    torch.cuda.synchronize()
                    fwd_ms.append((time.perf_counter() - t0) * 1000)
                else:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    y = model(x)
                    torch.cuda.synchronize()
                    fwd_ms.append((time.perf_counter() - t0) * 1000)

                    loss = lossf(y[:, 0], target)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    loss.backward()
                    torch.cuda.synchronize()
                    bwd_ms.append((time.perf_counter() - t1) * 1000)
                    model.zero_grad(set_to_none=True)
                del y
            rec["gpu_utilization"] = util.result()

        rec["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
        rec["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 2)
        rec["allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 ** 2, 2)
        rec["reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 ** 2, 2)

    except torch.cuda.OutOfMemoryError as exc:
        status, err = "OOM", f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
        rec["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 2)
    except RuntimeError as exc:
        oom = "out of memory" in str(exc).lower()
        status = "OOM" if oom else "ERROR"
        err = f"{type(exc).__name__}: {str(exc)[:300]}"
        with contextlib.suppress(Exception):
            rec["peak_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
            rec["peak_reserved_mb"] = round(
                torch.cuda.max_memory_reserved() / 1024 ** 2, 2)
    except Exception as exc:
        status, err = "ERROR", f"{type(exc).__name__}: {str(exc)[:300]}"

    rec["host_after"] = mem.sample()

    # ---- FIT / SPILL classification ------------------------------------
    # A peak above ~95% of physical VRAM means the Windows driver paged into
    # host RAM instead of raising OOM. Completing is NOT the same as fitting.
    peak = rec.get("peak_reserved_mb")
    if status == "FIT" and peak is not None and total_vram_mb:
        if peak > total_vram_mb * 0.95:
            status = "SPILL"
            rec["spill_reason"] = (
                f"peak reserved {peak:.0f} MB exceeds 95% of device VRAM "
                f"({total_vram_mb:.0f} MB) -- host-RAM fallback, not a true fit")
    rec["status"] = status
    if err:
        rec["error"] = err

    def _stats(v):
        if not v:
            return None
        s = sorted(v)
        return {"n": len(v), "mean_ms": round(sum(v) / len(v), 2),
                "median_ms": round(s[len(s) // 2], 2),
                "min_ms": round(s[0], 2), "max_ms": round(s[-1], 2)}

    rec["forward"] = _stats(fwd_ms)
    rec["backward"] = _stats(bwd_ms)

    # ---- verified release ----------------------------------------------
    with contextlib.suppress(Exception):
        del model, x, target
    rec["release"] = mem.release()
    return rec


# --------------------------------------------------------------------------- #
# parent: orchestrate isolated child processes
# --------------------------------------------------------------------------- #
def _env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
    }
    with contextlib.suppress(Exception):
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=_ROOT).stdout.strip()
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_total_vram_mb"] = round(p.total_memory / 1024 ** 2, 2)
            info["capability"] = f"{p.major}.{p.minor}"
    except Exception as exc:
        info["torch_error"] = str(exc)
    with contextlib.suppress(Exception):
        info["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip().splitlines()[0]
    with contextlib.suppress(Exception):
        import psutil
        vm = psutil.virtual_memory()
        info["host_ram_total_gb"] = round(vm.total / 1024 ** 3, 2)
        info["host_ram_available_gb"] = round(vm.available / 1024 ** 3, 2)
    return info


def _free_host_gb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return None


def _launch(res: int, mode: str, repeats: int) -> Dict[str, Any]:
    """Run one rung in a FRESH process, with a timeout and a host-RAM guard."""
    free = _free_host_gb()
    if free is not None and free < MIN_FREE_HOST_GB:
        return {"resolution": res, "mode": mode, "status": "ABORTED",
                "error": f"only {free:.2f} GB host RAM free (< "
                         f"{MIN_FREE_HOST_GB} GB safety floor); refusing to "
                         "risk system-memory exhaustion",
                "host_free_gb_before": round(free, 2)}

    timeout = CHILD_TIMEOUT_S.get(res, 600)
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           "--res", str(res), "--mode", mode, "--repeats", str(repeats)]
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd=_ROOT)
    except subprocess.TimeoutExpired:
        return {"resolution": res, "mode": mode, "status": "TIMEOUT/ABORTED",
                "error": f"exceeded the {timeout}s safety budget and was killed",
                "wall_s": round(time.perf_counter() - t0, 1)}
    wall = time.perf_counter() - t0

    for line in reversed(out.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            rec = json.loads(line[len("__RESULT__"):])
            rec["child_wall_s"] = round(wall, 1)
            return rec
    return {"resolution": res, "mode": mode, "status": "ERROR",
            "error": "child produced no result",
            "stderr_tail": out.stderr.strip()[-400:],
            "child_wall_s": round(wall, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GLO-NCA V3 local GPU memory audit (no training, no GCP).")
    ap.add_argument("--resolutions", default="32,48,64,96,128")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(_ROOT, "reports",
                                                  "phase3_memory_audit.json"))
    # child-mode flags
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--res", type=int)
    ap.add_argument("--mode", choices=["forward", "fwdbwd"], default="fwdbwd")
    args = ap.parse_args()

    if args.child:
        rec = run_one(args.res, args.mode, args.repeats)
        print("__RESULT__" + json.dumps(rec))
        return 0

    res_list = [int(r) for r in args.resolutions.split(",") if r.strip()]
    print("=" * 78)
    print("PHASE 3 — GLO-NCA V3 LOCAL GPU MEMORY AUDIT")
    print("no training · no GCP · each resolution in an isolated process")
    print("=" * 78)
    env = _env()
    print(f"GPU    : {env.get('gpu', 'N/A')}  "
          f"({env.get('gpu_total_vram_mb', 0) / 1024:.1f} GB)")
    print(f"torch  : {env.get('torch')} (CUDA {env.get('cuda_build')})")
    print(f"host   : {env.get('host_ram_total_gb')} GB total, "
          f"{env.get('host_ram_available_gb')} GB free")
    print(f"commit : {env.get('git_commit', 'n/a')[:12]}")
    print()

    results = []
    for res in res_list:
        for mode in ("forward", "fwdbwd"):
            print(f"  [{res:>3}^3 {mode:<7}] running ...", end="", flush=True)
            rec = _launch(res, mode, args.repeats)
            results.append(rec)
            st = rec.get("status", "?")
            pk = rec.get("peak_reserved_mb")
            extra = f" peak_reserved={pk:.0f} MB" if pk else ""
            f = (rec.get("forward") or {}).get("median_ms")
            b = (rec.get("backward") or {}).get("median_ms")
            tm = (f" fwd={f:.0f}ms" if f else "") + (f" bwd={b:.0f}ms" if b else "")
            print(f"\r  [{res:>3}^3 {mode:<7}] {st:<14}{extra}{tm}")
            if rec.get("error"):
                print(f"        └─ {rec['error'][:110]}")

    payload = {"phase": "PHASE3_MEMORY_AUDIT", "environment": env,
               "production_config": PROD_CFG,
               "safety": {"min_free_host_gb": MIN_FREE_HOST_GB,
                          "child_timeout_s": CHILD_TIMEOUT_S,
                          "process_isolation": True},
               "repeats_per_rung": args.repeats,
               "results": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nraw results -> {args.out}")

    bad = [r for r in results if r.get("status") in ("ERROR",)]
    print("STATUS: " + ("completed" if not bad else
                        f"completed with {len(bad)} ERROR rung(s)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
