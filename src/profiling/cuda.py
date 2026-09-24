r"""Bounded torch.profiler integration: a short window of iterations exported as a Chrome trace.

Returns a null context when disabled or unavailable.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Optional


@contextlib.contextmanager
def _null():
    yield None


def torch_profiler(out_dir: str, *, enabled: bool = False,
                   active: int = 3, warmup: int = 1, wait: int = 1,
                   record_shapes: bool = True, profile_memory: bool = True,
                   with_stack: bool = False):
    """Bounded profiler context; yields the profiler (call .step() per iteration) or None."""
    if not enabled:
        return _null()
    try:
        import torch
        from torch.profiler import (profile, ProfilerActivity, schedule,
                                    tensorboard_trace_handler)
    except Exception:
        return _null()

    acts = [ProfilerActivity.CPU]
    try:
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
    except Exception:
        pass

    d = os.path.join(out_dir, "profiler")
    os.makedirs(d, exist_ok=True)

    return profile(
        activities=acts,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler(d),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )


def device_info() -> dict:
    """GPU identity + capacity, or a clear 'no CUDA' marker."""
    try:
        import torch
    except Exception:
        return {"cuda_available": False, "reason": "torch not importable"}
    if not torch.cuda.is_available():
        return {"cuda_available": False, "torch": torch.__version__}
    p = torch.cuda.get_device_properties(0)
    return {"cuda_available": True,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": p.name,
            "vram_total_mb": round(p.total_memory / 1024 ** 2, 2),
            "capability": f"{p.major}.{p.minor}"}
