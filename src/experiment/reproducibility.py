r"""Seeding and RNG-state capture/restore.

Runs are reproducible in practice and a resume continues the same RNG stream;
full bit-exact CUDA determinism is not enforced (see ``describe()``).
"""
from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    """Snapshot every RNG state so a checkpoint can resume the same stream."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`capture_rng_state` (best-effort)."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["torch_cuda"]])
        except Exception:
            pass  # GPU config changed between runs; skip


def _as_byte_tensor(x: Any) -> torch.Tensor:
    """Coerce a reloaded torch RNG state to a uint8 CPU tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.uint8, device="cpu")
    return torch.tensor(x, dtype=torch.uint8)


def describe() -> str:
    return ("Seeds set for Python/NumPy/PyTorch(+CUDA) and RNG states are "
            "captured in every checkpoint and restored on resume. Full bit-exact "
            "determinism is NOT enforced (cudnn kept in default mode), so minor "
            "run-to-run variation on GPU is possible.")
