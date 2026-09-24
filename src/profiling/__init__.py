r"""GLO-NCA profiling: observational only, disabled by default.

When disabled every hook is a no-op with no CUDA synchronisation, so training is unaffected.

    from src.profiling import get_profiler, configure
    configure(cfg)                      # reads the profiling: config section
    prof = get_profiler()
    with prof.section("model/level1"):
        ...
"""
from .timer import (Profiler, NullProfiler, get_profiler, configure,
                    set_profiler, reset_profiler)
from .memory import MemorySampler
from .report import write_reports

__all__ = ["Profiler", "NullProfiler", "get_profiler", "configure",
           "set_profiler", "reset_profiler", "MemorySampler", "write_reports"]
