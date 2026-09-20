r"""GPU diagnostics and training-phase timing.

Two problems this module exists to avoid.

**Sampling the wrong moment.** A GPU reading taken while Python is loading a
NIfTI or between benchmark iterations shows an idle card -- low clock, ~0%
utilization -- and says nothing about training. Every sample here carries the
phase it was taken in, so an idle reading can never be mistaken for evidence
about compute.

**Measuring the measurement.** CUDA is asynchronous, so timing a GPU section
requires a synchronize, and synchronizing inside the hot loop would change
what is being measured. Timing here happens at section boundaries only, and
`PhaseTimer` reports the cost it added so that cost is visible rather than
assumed.

Anything the hardware does not expose is recorded as ``NOT AVAILABLE``. No
value in this module is estimated or inferred.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# Training phases a sample can belong to. The distinction is the point: a
# reading taken in DATA_WAIT describes the dataloader, not the model.
IDLE = "IDLE"
DATA_WAIT = "DATA_WAIT"
ACTIVE_GPU = "ACTIVE_GPU"
VALIDATION = "VALIDATION"
CHECKPOINT = "CHECKPOINT"
PHASES = (IDLE, DATA_WAIT, ACTIVE_GPU, VALIDATION, CHECKPOINT)

_NA = "NOT AVAILABLE"

_QUERY = ("utilization.gpu,utilization.memory,clocks.sm,clocks.max.sm,"
          "clocks.mem,temperature.gpu,power.draw,memory.total,memory.used,"
          "memory.free")


def _nvidia_smi() -> Dict[str, Any]:
    """One nvidia-smi sample. Every field degrades to NOT AVAILABLE."""
    blank = {k: _NA for k in (
        "gpu_utilization_pct", "memory_utilization_pct", "sm_clock_mhz",
        "sm_clock_max_mhz", "memory_clock_mhz", "temperature_c", "power_w",
        "vram_total_mb", "vram_used_mb", "vram_free_mb")}
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return blank
        a = [x.strip() for x in r.stdout.strip().split("\n")[0].split(",")]

        def num(i, cast):
            try:
                return cast(a[i])
            except Exception:
                return _NA

        return {
            "gpu_utilization_pct": num(0, int),
            "memory_utilization_pct": num(1, int),
            "sm_clock_mhz": num(2, int),
            "sm_clock_max_mhz": num(3, int),
            "memory_clock_mhz": num(4, int),
            "temperature_c": num(5, int),
            "power_w": num(6, float),
            "vram_total_mb": num(7, float),
            "vram_used_mb": num(8, float),
            "vram_free_mb": num(9, float),
        }
    except Exception:
        return blank


class GPUDiagnostics:
    """Phase-labelled GPU sampler.

    Sampling is rate-limited (``min_interval_s``) because nvidia-smi costs
    tens of milliseconds -- polling it every step would distort the very
    timings the run is trying to measure.
    """

    def __init__(self, enabled: bool = True, min_interval_s: float = 5.0):
        self.enabled = bool(enabled)
        self.min_interval_s = float(min_interval_s)
        self.samples: List[Dict[str, Any]] = []
        self._last = 0.0
        self._phase = IDLE
        self._torch = None
        try:
            import torch
            self._torch = torch if torch.cuda.is_available() else None
        except Exception:
            self._torch = None

    def set_phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        self._phase = phase

    @contextmanager
    def phase(self, phase: str):
        """Mark a region as a phase, restoring the previous one on exit."""
        prev = self._phase
        self.set_phase(phase)
        try:
            yield self
        finally:
            self._phase = prev

    def sample(self, *, epoch: int = -1, step: int = -1,
               force: bool = False) -> Optional[Dict[str, Any]]:
        """Take one sample if the rate limit allows (or ``force``)."""
        if not self.enabled:
            return None
        now = time.time()
        if not force and (now - self._last) < self.min_interval_s:
            return None
        self._last = now
        rec: Dict[str, Any] = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime(now)),
            "epoch": epoch, "step": step, "phase": self._phase,
        }
        rec.update(_nvidia_smi())
        if self._torch is not None:
            rec["torch_allocated_mb"] = round(
                self._torch.cuda.memory_allocated() / 1024 ** 2, 1)
            rec["torch_reserved_mb"] = round(
                self._torch.cuda.memory_reserved() / 1024 ** 2, 1)
            rec["torch_max_allocated_mb"] = round(
                self._torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        else:
            for k in ("torch_allocated_mb", "torch_reserved_mb",
                      "torch_max_allocated_mb"):
                rec[k] = _NA
        self.samples.append(rec)
        return rec

    # ------------------------------------------------------------- reporting
    def summary(self) -> Dict[str, Any]:
        """Per-phase summary. Utilization is reported ONLY for ACTIVE_GPU."""
        if not self.samples:
            return {"status": "NOT MEASURED", "samples": 0}

        def nums(rows, key):
            return [r[key] for r in rows
                    if isinstance(r.get(key), (int, float))]

        out: Dict[str, Any] = {"samples": len(self.samples), "phases": {}}
        for ph in PHASES:
            rows = [r for r in self.samples if r["phase"] == ph]
            if not rows:
                continue
            entry: Dict[str, Any] = {"samples": len(rows)}
            for key, label in (("gpu_utilization_pct", "gpu_util_pct"),
                               ("sm_clock_mhz", "sm_clock_mhz"),
                               ("temperature_c", "temperature_c"),
                               ("power_w", "power_w"),
                               ("torch_allocated_mb", "torch_allocated_mb")):
                vals = nums(rows, key)
                entry[label] = ({"mean": round(statistics.mean(vals), 1),
                                 "min": min(vals), "max": max(vals)}
                                if vals else _NA)
            out["phases"][ph] = entry

        active = [r for r in self.samples if r["phase"] == ACTIVE_GPU]
        util = nums(active, "gpu_utilization_pct")
        if util:
            mean_util = statistics.mean(util)
            out["active_gpu_utilization_mean_pct"] = round(mean_util, 1)
            out["utilization_note"] = (
                "Measured DURING forward/backward only. Samples taken in "
                "DATA_WAIT, CHECKPOINT or IDLE are excluded: an idle reading "
                "describes the pause, not the computation.")
            if mean_util < 50:
                out["warning"] = (
                    f"GPU averaged {mean_util:.1f}% during ACTIVE_GPU. That is "
                    f"low for compute-bound training; investigate dataloader "
                    f"wait, H2D, CPU preprocessing or synchronisation before "
                    f"changing any scientific setting.")
        else:
            out["active_gpu_utilization_mean_pct"] = _NA

        clocks = nums(active, "sm_clock_mhz")
        maxes = nums(active, "sm_clock_max_mhz")
        if clocks and maxes:
            pct = 100 * statistics.mean(clocks) / max(maxes)
            out["active_clock_pct_of_max"] = round(pct, 1)
            if pct < 75:
                out["throttle_warning"] = (
                    f"SM clock averaged {pct:.0f}% of maximum during active "
                    f"compute: absolute timings are inflated by roughly "
                    f"{100 / pct:.1f}x. Ratios remain valid.")
        return out

    def write_csv(self, path: str) -> Optional[str]:
        if not self.samples:
            return None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cols = list(self.samples[0].keys())
        for s in self.samples:                 # union, in case a field appears late
            for k in s:
                if k not in cols:
                    cols.append(k)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for s in self.samples:
                w.writerow({c: s.get(c, "") for c in cols})
        os.replace(tmp, path)
        return path


class PhaseTimer:
    """Wall-clock timing of named sections, with explicit CUDA boundaries.

    ``cuda=True`` synchronises at entry and exit so async kernels are charged
    to the section that launched them. That synchronisation is itself a cost,
    so it is counted and reported: an instrument that hides its own overhead
    cannot be trusted.
    """

    def __init__(self, enabled: bool = True, cuda: bool = False):
        self.enabled = bool(enabled)
        self.records: Dict[str, List[float]] = {}
        self._sync_overhead_s = 0.0
        self._sync_calls = 0
        self._torch = None
        if cuda:
            try:
                import torch
                self._torch = torch if torch.cuda.is_available() else None
            except Exception:
                self._torch = None

    def _sync(self) -> None:
        if self._torch is None:
            return
        t = time.perf_counter()
        self._torch.cuda.synchronize()
        self._sync_overhead_s += time.perf_counter() - t
        self._sync_calls += 1

    @contextmanager
    def section(self, name: str, cuda: bool = True):
        if not self.enabled:
            yield
            return
        if cuda:
            self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                self._sync()
            self.records.setdefault(name, []).append(time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self.records.setdefault(name, []).append(float(seconds))

    def summary(self, total_key: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sections": {},
            "synchronisation_overhead_s": round(self._sync_overhead_s, 4),
            "synchronisation_calls": self._sync_calls,
            "note": ("Sections NEST (e.g. model/level1 inside train/forward), "
                     "so percentages deliberately do NOT sum to 100%."),
        }
        total = None
        if total_key and total_key in self.records:
            total = sum(self.records[total_key])
        for name, vals in sorted(self.records.items(),
                                 key=lambda kv: -sum(kv[1])):
            s = sum(vals)
            entry = {
                "total_s": round(s, 4), "count": len(vals),
                "mean_ms": round(1000 * s / len(vals), 3),
                "median_ms": round(1000 * statistics.median(vals), 3),
                "min_ms": round(1000 * min(vals), 3),
                "max_ms": round(1000 * max(vals), 3),
                "std_ms": round(1000 * statistics.pstdev(vals), 3)
                if len(vals) > 1 else 0.0,
            }
            if total:
                entry["pct_of_total"] = round(100 * s / total, 2)
            out["sections"][name] = entry
        return out

    def write_json(self, path: str, extra: Optional[Dict[str, Any]] = None,
                   total_key: Optional[str] = None) -> str:
        payload = self.summary(total_key=total_key)
        if extra:
            payload.update(extra)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, path)
        return path
