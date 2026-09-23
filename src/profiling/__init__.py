r"""GLO-NCA profiling / tracing (Phase 2).

OBSERVATIONAL ONLY. Nothing in this package changes training behaviour: no
architecture, resolution, NCA-step, loss, optimizer, EMA, split, threshold or
checkpoint-policy change. It only measures.

Disabled by default. When disabled every hook is a no-op costing one attribute
lookup, and -- critically -- it inserts **no CUDA synchronisation**, so normal
production training is unaffected.

Typical use::

    from src.profiling import get_profiler, configure

    configure(cfg)                      # reads the `profiling:` config section
    prof = get_profiler()
    with prof.section("model/level1"):
        ...

See ``extra/reports/PHASE2_PROFILING_REPORT.md`` for the generated analysis.
"""
from .timer import (Profiler, NullProfiler, get_profiler, configure,
                    set_profiler, reset_profiler)
from .memory import MemorySampler
from .report import write_reports

__all__ = ["Profiler", "NullProfiler", "get_profiler", "configure",
           "set_profiler", "reset_profiler", "MemorySampler", "write_reports"]
