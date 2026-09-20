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
    # Phase 2 (P1): this was a bare `except: pass` with NO logging. If scheduler
    # restore failed, the run silently continued with a freshly-built
    # CosineAnnealingLR at step 0 -- i.e. the learning rate jumped back to its
    # initial value in the middle of a 300-epoch campaign, corrupting the run
    # with no warning anywhere. The failure is now surfaced loudly (and is
    # visible afterwards in metrics/train.csv's `lr` column).
    failures = []
    for i, (s, sd) in enumerate(zip(schedulers, ckpt.get("scheduler", []))):
        try:
            s.load_state_dict(sd)
        except Exception as exc:
            failures.append(f"scheduler[{i}]: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "checkpoint restore FAILED for: " + "; ".join(failures) + ".\n"
            "Refusing to continue: the LR schedule would silently restart from "
            "step 0 mid-campaign. This normally means the checkpoint was written "
            "by a config with different `training.epochs` (T_max). Resume with "
            "the ORIGINAL config, or start a new experiment deliberately.")


# ---------------------------------------------------------------- top-K best
# Best-checkpoint retention. `save_best_weights` above keeps the single best
# file in the original format and is unchanged; the helpers here maintain an
# ADDITIONAL ranked set (best_1 .. best_k) plus a JSON manifest.
#
# Weights are never averaged and no SWA is performed: the ranked files are
# retained for inspection and for a later, explicitly approved experiment.


def _best_path(directory: str, rank: int) -> str:
    return os.path.join(directory, f"best_{rank}.pth")


def update_top_k(directory: str, *, weights: List[Dict[str, torch.Tensor]],
                 epoch: int, score: float, metrics: Dict[str, Any],
                 meta: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """Insert one candidate into the ranked best-K set, highest score first.

    Returns the manifest (a list, best first). The candidate is written only if
    it makes the top K, so a run that never improves performs no extra I/O.
    Files are rewritten in rank order, which keeps `best_1.pth` always the best
    regardless of the order in which candidates arrived.
    """
    import json

    os.makedirs(directory, exist_ok=True)
    manifest_path = os.path.join(directory, "top_k.json")
    entries: List[Dict[str, Any]] = []
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                entries = json.load(fh).get("entries", [])
        except (OSError, ValueError):
            entries = []          # unreadable manifest -> rebuild from scratch

    # Rank BEFORE this candidate is considered. The manifest order is the
    # authority for where each retained epoch's file currently lives, so the
    # mapping epoch -> existing file must be built from the PREVIOUS ranking,
    # never from the new one.
    previous = sorted(entries, key=lambda e: e["score"], reverse=True)
    existing = {}
    for i, e in enumerate(previous):
        p = _best_path(directory, i + 1)
        if os.path.isfile(p):
            existing[e["epoch"]] = p

    entries = [e for e in entries if e.get("epoch") != epoch]
    entries.append({"epoch": epoch, "score": float(score),
                    "metrics": metrics, **meta})
    entries.sort(key=lambda e: e["score"], reverse=True)
    keep = entries[:max(1, int(top_k))]
    if not any(e["epoch"] == epoch for e in keep):
        return keep               # candidate did not make the cut: no write

    # Move every surviving file aside first, so a file that changes rank can
    # never overwrite another before it has been relocated.
    staged = {}
    for ep_, src in existing.items():
        if any(e["epoch"] == ep_ for e in keep) and ep_ != epoch:
            tmp = os.path.join(directory, f".stage_{ep_}.tmp")
            os.replace(src, tmp)
            staged[ep_] = tmp

    # Any file still sitting at a best_* path is no longer retained.
    for rank in range(1, len(previous) + 2):
        stale = _best_path(directory, rank)
        if os.path.isfile(stale):
            os.remove(stale)

    for i, e in enumerate(keep):
        dest = _best_path(directory, i + 1)
        if e["epoch"] == epoch:
            tmp = dest + ".tmp"
            torch.save({"m": weights, "ep": epoch, "score": float(score),
                        "metrics": metrics, **meta}, tmp)
            os.replace(tmp, dest)
        elif e["epoch"] in staged:
            os.replace(staged.pop(e["epoch"]), dest)

    for leftover in staged.values():       # defensive: never leak temp files
        if os.path.isfile(leftover):
            os.remove(leftover)

    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"top_k": int(top_k), "selection_metric": meta.get(
            "selection_metric", "validation mean foreground Dice"),
            "entries": keep}, fh, indent=2, default=str)
    os.replace(tmp, manifest_path)
    return keep


def provenance_metadata(*, config: Dict[str, Any],
                        split_sha: str = "", dataset_root: str = "") -> Dict[str, Any]:
    """Identity/provenance block embedded in best checkpoints.

    Captures what is needed to reproduce a saved model: code revision, config
    fingerprint, dataset and split identity, and the software/hardware it was
    produced on. Every field degrades to a sentinel rather than raising, so
    checkpointing can never fail because provenance is unavailable.
    """
    import hashlib
    import json
    import platform
    import subprocess
    import sys

    def _git(*args: str) -> str:
        try:
            out = subprocess.run(("git",) + args, capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else "unavailable"
        except Exception:
            return "unavailable"

    try:
        fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    except Exception:
        fingerprint = "unavailable"

    meta: Dict[str, Any] = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_fingerprint": fingerprint,
        "split_sha256": split_sha or "unavailable",
        "dataset_root": dataset_root or "unavailable",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "selection_metric": "validation mean foreground Dice "
                            "(mean over WT/TC/ET, 3-epoch rolling mean)",
    }
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
            meta.update({
                "gpu": torch.cuda.get_device_name(0),
                "gpu_capability": f"{cap[0]}.{cap[1]}",
                "cuda": torch.version.cuda,
            })
        except Exception:
            pass
    return meta
