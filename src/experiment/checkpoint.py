r"""Robust checkpointing: a checkpoint holds the COMPLETE training state so a
run can resume exactly (model, optimizer, scheduler, EMA, best score, epoch,
config and RNG states). ``best`` and ``last`` are kept strictly separate.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch


def build_checkpoint(*, epoch: int, models: List[torch.nn.Module],
                     optimizers: List[torch.optim.Optimizer],
                     schedulers: List[Any],
                     ema: Optional[List[Dict[str, torch.Tensor]]],
                     best_score: float, best_epoch: int,
                     history: Dict[str, list], config: Dict[str, Any],
                     rng_state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a full-state checkpoint dict."""
    return {
        "epoch": epoch,
        "model": [m.state_dict() for m in models],
        "optimizer": [o.state_dict() for o in optimizers],
        "scheduler": [s.state_dict() for s in schedulers],
        "ema": ema,  # list of state_dicts or None
        "best_score": best_score,
        "best_epoch": best_epoch,
        "history": history,
        "config": config,
        "rng_state": rng_state,
        "format": "glo-nca-v2-ckpt-1",
    }


def save_checkpoint(path: str, ckpt: Dict[str, Any]) -> None:
    """Atomically save a checkpoint (write to .tmp then rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def save_best_weights(path: str, weights: List[Dict[str, torch.Tensor]],
                      epoch: int, val_mean: float, val_smooth: float) -> None:
    """Save the compact best-model file (EMA or raw weights) used for testing.
    Kept in the SAME format the original train.py wrote, so downstream test code
    is unchanged: {"m": [...], "ep": int, "val_mean": ..., "val_smooth": ...}."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"m": weights, "ep": epoch,
                "val_mean": val_mean, "val_smooth": val_smooth}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    # weights_only=False: our checkpoints embed numpy/torch RNG states (not just
    # tensors), and they are produced by this codebase, so they are trusted.
    # PyTorch >= 2.6 defaults weights_only=True and would reject them otherwise.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        return torch.load(path, map_location=map_location)


def restore_into(ckpt: Dict[str, Any], *, models: List[torch.nn.Module],
                 optimizers: List[torch.optim.Optimizer],
                 schedulers: List[Any]) -> None:
    """Restore model/optimizer/scheduler states from a full checkpoint."""
    for m, sd in zip(models, ckpt["model"]):
        m.load_state_dict(sd)
    for o, sd in zip(optimizers, ckpt.get("optimizer", [])):
        o.load_state_dict(sd)
    for s, sd in zip(schedulers, ckpt.get("scheduler", [])):
        try:
            s.load_state_dict(sd)
        except Exception:
            pass  # scheduler shape can shift if epochs changed; keep going
