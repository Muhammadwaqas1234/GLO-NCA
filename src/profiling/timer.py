r"""Core profiler: CUDA-aware section timing with a true-no-op disabled mode.

Design constraints (Phase 2):
  * Disabled by default; when disabled, ``section()`` returns a shared no-op
    context manager and performs **no CUDA synchronisation whatsoever**, so
    production training behaviour and timing are unchanged.
  * CUDA is asynchronous: a bare ``perf_counter()`` around a GPU call measures
    only the launch, not the work. When ``cuda=True`` and a section is marked
    ``cuda=True`` we synchronise around it so the number means what it says.
    Sections that are pure CPU (disk, NIfTI, preprocessing) are NOT
    synchronised, so we do not manufacture sync points that production lacks.
  * Worker-safe: dataset-side timings are accumulated into plain floats that
    survive being pickled back from a DataLoader worker; nothing CUDA-related
    is touched inside a worker.
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
    """Collects per-section timings, grouped per iteration.

    ``warmup`` iterations are measured but excluded from the reported
    statistics -- the first iterations include CUDA context creation, cuDNN
    autotuning and cold page-cache effects and are never representative.
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
        """Time a named block. ``cuda=True`` synchronises around it so the GPU
        work is actually included (CUDA is async)."""
        return self._timed(name, cuda)

    def record(self, name: str, ms: float) -> None:
        """Record a measurement directly (used for worker-side timings that were
        measured in another process and shipped back with the batch)."""
        self._samples[name].append(ms)
        self._current[name] = self._current.get(name, 0.0) + ms

    # ------------------------------------------------------------- pickling
    # A Profiler can be reached from the Dataset (which holds no reference, but
    # the module-level singleton is imported inside __getitem__). On Windows,
    # DataLoader workers are SPAWNED and the dataset is pickled, so any
    # unpicklable attribute here becomes "cannot pickle '_thread.lock'". The
    # torch module handle is exactly such an object, so it is dropped on pickle
    # and restored on unpickle. Worker-side sections then time CPU work only
    # (correct: a worker must never touch CUDA anyway).
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
        """True once warmup is over (i.e. this iteration counts)."""
        return self._iter >= self.warmup

    def should_stop(self) -> bool:
        """True when warmup + profiled iterations are complete."""
        return self._iter >= (self.warmup + self.iterations - 1)

    def flush(self) -> None:
        self.mark_iteration(self._iter)

    def all_samples(self) -> Dict[str, List[float]]:
        """EVERY sample ever recorded, including sections that fall outside the
        per-iteration window (validation runs after the training loop, and
        dataset stages measured in DataLoader workers arrive out-of-band)."""
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
        """Per-section stats over POST-WARMUP iterations only."""
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
    """Build a profiler from a Config's ``profiling:`` section.

    Absent section or ``enabled: false`` => NullProfiler (production default).
    """
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
