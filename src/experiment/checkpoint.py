r"""Full-state checkpointing (model, optimizer, scheduler, EMA, epoch, config, RNG); best and last kept separate."""
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
    """Save the compact best-model file: {"m": [...], "ep", "val_mean", "val_smooth"}."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"m": weights, "ep": epoch,
                "val_mean": val_mean, "val_smooth": val_smooth}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    # weights_only=False: our checkpoints embed RNG states and are produced by this codebase.
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
    # Fail loudly: a silently reset scheduler would restart the LR mid-run.
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


# Top-K best: a ranked best_1..best_k set plus a manifest; weights are never averaged.


def _best_path(directory: str, rank: int) -> str:
    return os.path.join(directory, f"best_{rank}.pth")


def update_top_k(directory: str, *, weights: List[Dict[str, torch.Tensor]],
                 epoch: int, score: float, metrics: Dict[str, Any],
                 meta: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """Insert a candidate into the ranked best-K set (highest first); written only if it makes the top K."""
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

    # Map epoch -> existing file from the previous ranking, not the new one.
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

    # Move survivors aside first so a re-ranked file never overwrites another.
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
    """Provenance block for best checkpoints; each field falls back to a sentinel, never raises."""
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
