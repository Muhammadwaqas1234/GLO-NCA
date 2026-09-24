r"""Memory sampling for profiling; keeps host and GPU quantities separate.

  process_rss_mb    this process's resident host RAM
  system_used_mb    whole-machine host RAM in use
  gpu_allocated_mb  tensors held by the caching allocator
  gpu_reserved_mb   memory reserved from the driver
  gpu_peak_mb       max allocated since the last reset

Host fields need psutil (None without it). On Windows a GPU peak above physical VRAM
is a spill to host RAM, not a fit; classify_peak() reports that.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False


class MemorySampler:
    """Samples host + GPU memory. Safe to construct when CUDA is unavailable."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._torch = None
        self._cuda = False
        try:
            import torch
            self._torch = torch
            self._cuda = torch.cuda.is_available()
        except Exception:
            self._cuda = False
        self._proc = psutil.Process() if _PSUTIL else None

    # Drop unpicklable handles when crossing a process boundary; re-acquire lazily.
    def __getstate__(self):
        st = self.__dict__.copy()
        st["_torch"] = None
        st["_proc"] = None
        st["_cuda"] = False
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._torch = None
        self._cuda = False
        self._proc = psutil.Process() if _PSUTIL else None

    # ------------------------------------------------------------------ query
    def sample(self) -> Dict[str, Optional[float]]:
        if not self.enabled:
            return {}
        out: Dict[str, Optional[float]] = {}
        if self._proc is not None:
            out["process_rss_mb"] = round(self._proc.memory_info().rss / 1024 ** 2, 2)
            vm = psutil.virtual_memory()
            out["system_used_mb"] = round((vm.total - vm.available) / 1024 ** 2, 2)
            out["system_available_mb"] = round(vm.available / 1024 ** 2, 2)
        else:
            out["process_rss_mb"] = None
            out["system_used_mb"] = None
            out["system_available_mb"] = None
        if self._cuda:
            t = self._torch
            out["gpu_allocated_mb"] = round(t.cuda.memory_allocated() / 1024 ** 2, 2)
            out["gpu_reserved_mb"] = round(t.cuda.memory_reserved() / 1024 ** 2, 2)
            out["gpu_peak_mb"] = round(t.cuda.max_memory_allocated() / 1024 ** 2, 2)
        else:
            out["gpu_allocated_mb"] = None
            out["gpu_reserved_mb"] = None
            out["gpu_peak_mb"] = None
        return out

    def reset_peak(self) -> None:
        if self.enabled and self._cuda:
            self._torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------ diagnostics
    def device_total_mb(self) -> Optional[float]:
        if not self._cuda:
            return None
        p = self._torch.cuda.get_device_properties(0)
        return round(p.total_memory / 1024 ** 2, 2)

    def classify_peak(self, peak_mb: Optional[float]) -> str:
        """FIT / SPILL / UNKNOWN; a peak above physical VRAM is a spill, never a fit."""
        total = self.device_total_mb()
        if peak_mb is None or total is None:
            return "UNKNOWN"
        return "SPILL" if peak_mb > total * 0.95 else "FIT"

    def release(self) -> Dict[str, Any]:
        """Sample, empty_cache, sample again; report the actual before/after numbers."""
        before = self.sample()
        if self._cuda:
            import gc
            gc.collect()
            self._torch.cuda.empty_cache()
            self._torch.cuda.synchronize()
        after = self.sample()
        return {"before": before, "after": after,
                "device_total_mb": self.device_total_mb()}
