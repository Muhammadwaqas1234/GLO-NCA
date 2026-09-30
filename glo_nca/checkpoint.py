r"""Full training checkpoints, so a preempted run resumes where it stopped."""
from __future__ import annotations

import os
import random

import numpy as np
import torch

FORMAT = "glo-nca-cascade-ckpt-1"


def capture_rng():
    r"""Snapshot the Python, NumPy and torch random states."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    r"""Restore random states saved by capture_rng."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG states must be CPU byte tensors (the checkpoint may be loaded onto the GPU).
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_atomic(obj, path):
    r"""Write to <path>.tmp, then rename, so a crash never leaves a partial file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_last(epoch, models, agent, ema, best, hist, train_seconds, config_raw):
    r"""Assemble the full resume checkpoint written after every epoch."""
    return {"format": FORMAT, "epoch": epoch,
            "models": [m.state_dict() for m in models],
            "optimizers": [o.state_dict() for o in agent.optimizer],
            "schedulers": [s.state_dict() for s in agent.scheduler],
            "ema": ema, "best": best, "history": hist,
            "train_seconds": train_seconds, "config": config_raw,
            "rng": capture_rng()}


def load_last(path, models, agent, device):
    r"""Restore models, optimizers and schedulers in place; return the checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path}: not a GLO-NCA cascade checkpoint (format {ck.get('format')!r})")
    for m, sd in zip(models, ck["models"]):
        m.load_state_dict(sd)
    for o, sd in zip(agent.optimizer, ck["optimizers"]):
        o.load_state_dict(sd)
    for s, sd in zip(agent.scheduler, ck["schedulers"]):
        s.load_state_dict(sd)
    return ck
