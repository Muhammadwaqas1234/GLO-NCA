r"""Memory sampling for Phase 2 profiling.

Deliberately keeps FOUR distinct quantities apart, because conflating them is
how misleading memory claims get made:

  * ``process_rss_mb``   -- this Python process's resident set (host RAM)
  * ``system_used_mb``   -- whole-machine host RAM in use
  * ``gpu_allocated_mb`` -- tensors currently held by the caching allocator
  * ``gpu_reserved_mb``  -- memory the allocator has reserved from the driver
  * ``gpu_peak_mb``      -- max allocated since the last peak reset

GPU numbers come from ``torch.cuda`` and are only meaningful on the CUDA device.
Host numbers need ``psutil``; when it is absent the host fields are reported as
``None`` rather than guessed.

NOTE on Windows: the CUDA driver may spill beyond physical VRAM into host RAM
instead of raising OOM. A ``gpu_peak_mb`` above the device's physical VRAM is
therefore a SPILL, not a fit -- ``classify_peak()`` makes that explicit.
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

    # Same spawn-safety rule as Profiler: the torch handle and the psutil
    # Process object are not picklable, so drop them if this ever crosses a
    # process boundary and re-acquire lazily in the child.
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
        """FIT / SPILL / UNKNOWN. A peak above physical VRAM means the driver
        paged to host RAM (Windows sysmem fallback) -- never call that a fit."""
        total = self.device_total_mb()
        if peak_mb is None or total is None:
            return "UNKNOWN"
        return "SPILL" if peak_mb > total * 0.95 else "FIT"

    def release(self) -> Dict[str, Any]:
        """Verified memory release: sample, empty_cache, sample again.

        Reports the ACTUAL numbers before and after rather than asserting that
        ``empty_cache()`` worked."""
        before = self.sample()
        if self._cuda:
            import gc
            gc.collect()
            self._torch.cuda.empty_cache()
            self._torch.cuda.synchronize()
        after = self.sample()
        return {"before": before, "after": after,
                "device_total_mb": self.device_total_mb()}
