r"""CUDA-aware section timing with a true no-op disabled mode.

Disabled (default): section() returns a shared no-op context with no CUDA sync.
Enabled: sections marked cuda=True are synchronised so GPU work is included; CPU-only
sections are not. Worker-side timings are plain floats and never touch CUDA.
"""
from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# no-op fast path
# --------------------------------------------------------------------------- #
class _NullCtx:
    """Zero-cost context manager reused for every disabled section."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_CTX = _NullCtx()


class NullProfiler:
    """Profiler that measures nothing. Used whenever profiling is disabled."""
    enabled = False

    def section(self, name: str, cuda: bool = False):
        return _NULL_CTX

    def record(self, name: str, ms: float) -> None:
        pass

    def mark_iteration(self, index: int) -> None:
        pass

    def is_profiled_iteration(self) -> bool:
        return False

    def should_stop(self) -> bool:
        return False

    def snapshot(self) -> Dict[str, Any]:
        return {}


# --------------------------------------------------------------------------- #
# real profiler
# --------------------------------------------------------------------------- #
class Profiler:
    """Per-section timings grouped per iteration; profiler-warmup iterations (CUDA init,
    cuDNN autotune, cold cache) are excluded from the statistics. Unrelated to LR warmup.
    """

    enabled = True

    def __init__(self, *, warmup: int = 5, iterations: int = 20,
                 cuda: bool = True, memory: bool = True,
                 out_dir: Optional[str] = None):
        self.warmup = int(warmup)
        self.iterations = int(iterations)
        self.cuda_enabled = bool(cuda)
        self.memory_enabled = bool(memory)
        self.out_dir = out_dir

        self._iter = -1                       # -1 => outside the iteration loop
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._current: Dict[str, float] = {}
        self.iteration_records: List[Dict[str, Any]] = []
        self._torch = None
        self._cuda_ok = False
        if self.cuda_enabled:
            try:
                import torch
                self._torch = torch
                self._cuda_ok = torch.cuda.is_available()
            except Exception:
                self._cuda_ok = False

    # ----------------------------------------------------------------- timing
    @contextlib.contextmanager
    def _timed(self, name: str, cuda: bool):
        sync = cuda and self._cuda_ok
        if sync:
            self._torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                # Without this the timer stops at kernel LAUNCH, not completion.
                self._torch.cuda.synchronize()
            self.record(name, (time.perf_counter() - t0) * 1000.0)

    def section(self, name: str, cuda: bool = False):
        """Time a named block; cuda=True synchronises so GPU work is included."""
        return self._timed(name, cuda)

    def record(self, name: str, ms: float) -> None:
        """Record a measurement taken elsewhere (e.g. in a DataLoader worker)."""
        self._samples[name].append(ms)
        self._current[name] = self._current.get(name, 0.0) + ms

    # Pickling: spawned DataLoader workers may pickle this object; the torch handle is
    # dropped and restored, so worker sections time CPU work only.
    def __getstate__(self):
        st = self.__dict__.copy()
        st["_torch"] = None
        st["_cuda_ok"] = False
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._torch = None
        self._cuda_ok = False

    # -------------------------------------------------------------- iteration
    def mark_iteration(self, index: int) -> None:
        """Close the previous iteration's record and open a new one."""
        if self._current:
            self.iteration_records.append(
                {"iteration": self._iter,
                 "warmup": self._iter < self.warmup,
                 **{k: round(v, 4) for k, v in self._current.items()}})
        self._current = {}
        self._iter = index

    def is_profiled_iteration(self) -> bool:
        """True once the profiler warmup is over (this iteration counts)."""
        return self._iter >= self.warmup

    def should_stop(self) -> bool:
        """True when profiler warmup + profiled iterations are complete."""
        return self._iter >= (self.warmup + self.iterations - 1)

    def flush(self) -> None:
        self.mark_iteration(self._iter)

    def all_samples(self) -> Dict[str, List[float]]:
        """Every recorded sample, including out-of-iteration ones (validation, worker stages)."""
        return {k: list(v) for k, v in self._samples.items()}

    def out_of_band_statistics(self) -> Dict[str, Dict[str, float]]:
        """Stats for sections that never appear in an iteration record."""
        in_band = set()
        for rec in self.iteration_records:
            in_band.update(k for k in rec if k not in ("iteration", "warmup"))
        out: Dict[str, Dict[str, float]] = {}
        for name, vals in sorted(self._samples.items()):
            if name in in_band or not vals:
                continue
            out[name] = {"count": len(vals),
                         "mean_ms": round(statistics.fmean(vals), 4),
                         "median_ms": round(statistics.median(vals), 4),
                         "min_ms": round(min(vals), 4),
                         "max_ms": round(max(vals), 4),
                         "total_ms": round(sum(vals), 4)}
        return out

    # ---------------------------------------------------------------- results
    def statistics(self) -> Dict[str, Dict[str, float]]:
        """Per-section stats over post-warmup profiled iterations only."""
        post: Dict[str, List[float]] = defaultdict(list)
        for rec in self.iteration_records:
            if rec.get("warmup"):
                continue
            for k, v in rec.items():
                if k not in ("iteration", "warmup"):
                    post[k].append(float(v))

        out: Dict[str, Dict[str, float]] = {}
        for name, vals in sorted(post.items()):
            if not vals:
                continue
            s = {"count": len(vals),
                 "mean_ms": round(statistics.fmean(vals), 4),
                 "median_ms": round(statistics.median(vals), 4),
                 "min_ms": round(min(vals), 4),
                 "max_ms": round(max(vals), 4)}
            # A percentile from 1-2 samples is noise dressed as a statistic.
            if len(vals) >= 5:
                s["p95_ms"] = round(
                    statistics.quantiles(vals, n=20)[18], 4)
            else:
                s["p95_ms"] = None
            out[name] = s
        return out

    def snapshot(self) -> Dict[str, Any]:
        return {"warmup_iterations": self.warmup,
                "profiled_iterations": self.iterations,
                "cuda_timing": self._cuda_ok,
                "iterations": self.iteration_records,
                "statistics": self.statistics(),
                "out_of_band": self.out_of_band_statistics()}

    def write(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        """Write raw JSONL traces + aggregated JSON stats."""
        d = out_dir or self.out_dir
        if not d:
            return {}
        os.makedirs(os.path.join(d, "traces"), exist_ok=True)
        os.makedirs(os.path.join(d, "metrics"), exist_ok=True)
        jsonl = os.path.join(d, "traces", "iterations.jsonl")
        with open(jsonl, "w", encoding="utf-8") as fh:
            for rec in self.iteration_records:
                fh.write(json.dumps(rec) + "\n")
        stats = os.path.join(d, "metrics", "statistics.json")
        with open(stats, "w", encoding="utf-8") as fh:
            json.dump(self.snapshot(), fh, indent=2)
        return {"traces": jsonl, "statistics": stats}


# --------------------------------------------------------------------------- #
# module-level accessor
# --------------------------------------------------------------------------- #
_ACTIVE: Any = NullProfiler()


def get_profiler():
    return _ACTIVE


def set_profiler(p) -> None:
    global _ACTIVE
    _ACTIVE = p


def reset_profiler() -> None:
    set_profiler(NullProfiler())


def configure(cfg, out_dir: Optional[str] = None):
    """Build a profiler from the config's profiling section; absent or disabled -> NullProfiler."""
    try:
        sec = cfg.section("profiling") if hasattr(cfg, "section") else None
    except Exception:
        sec = None
    sec = sec or {}
    if not bool(sec.get("enabled", False)):
        reset_profiler()
        return _ACTIVE
    p = Profiler(warmup=int(sec.get("warmup_iterations", 5)),
                 iterations=int(sec.get("iterations", 20)),
                 cuda=bool(sec.get("cuda", True)),
                 memory=bool(sec.get("memory", True)),
                 out_dir=out_dir)
    set_profiler(p)
    return p
