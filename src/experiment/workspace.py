r"""Experiment directory management.

Creates a unique, timestamped experiment directory with the full Phase 1 layout
and never overwrites a previous run. Also owns ``status.json`` (a live progress
file) and helpers to write the manifest / reports.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

# Sub-directories created for every experiment.
SUBDIRS = [
    "checkpoints", "checkpoints/periodic",
    "metrics", "graphs", "tensorboard", "logs",
    "config", "split", "reports", "predictions",
]

STATUS_STATES = (
    "initializing", "validating", "training",
    "evaluating", "packaging", "completed", "failed",
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class Workspace:
    """A single experiment's directory tree and its status file."""

    root: str
    experiment_id: str

    # --------------------------------------------------------------- creation
    @classmethod
    def create(cls, base: str, name: str) -> "Workspace":
        """Create a fresh experiment directory: base/<name>-<timestamp>/."""
        experiment_id = f"{name}-{_timestamp()}"
        root = os.path.join(base, experiment_id)
        # Extremely unlikely, but guarantee uniqueness rather than overwrite.
        suffix = 1
        while os.path.exists(root):
            experiment_id = f"{name}-{_timestamp()}-{suffix}"
            root = os.path.join(base, experiment_id)
            suffix += 1
        for sub in SUBDIRS:
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        return cls(root=root, experiment_id=experiment_id)

    @classmethod
    def open_existing(cls, root: str) -> "Workspace":
        """Reopen an existing experiment directory (for --resume)."""
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Experiment directory not found: {root}")
        for sub in SUBDIRS:  # tolerate older dirs missing a folder
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        return cls(root=root, experiment_id=os.path.basename(root.rstrip(os.sep)))

    # ------------------------------------------------------------------ paths
    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    @property
    def best_ckpt(self) -> str:
        return self.path("checkpoints", "best.pth")

    @property
    def last_ckpt(self) -> str:
        return self.path("checkpoints", "last.pth")

    def periodic_ckpt(self, epoch: int) -> str:
        return self.path("checkpoints", "periodic", f"epoch_{epoch:04d}.pth")

    @property
    def status_path(self) -> str:
        return self.path("status.json")

    @property
    def manifest_path(self) -> str:
        return self.path("experiment_manifest.json")

    # ----------------------------------------------------------------- status
    def write_status(self, state: str, **extra: Any) -> None:
        """Atomically write status.json with the current state and progress."""
        assert state in STATUS_STATES, f"unknown status {state!r}"
        payload: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "state": state,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        payload.update(extra)
        self._atomic_json(self.status_path, payload)

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        self._atomic_json(self.manifest_path, manifest)

    def write_json(self, rel_path: str, obj: Any) -> None:
        self._atomic_json(self.path(rel_path), obj)

    @staticmethod
    def _atomic_json(path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
        os.replace(tmp, path)
