r"""Configuration loading for GLO-NCA V2 experiments.

Loads a YAML config into a plain, validated ``Config`` object. The defaults in
the YAML files under ``configs/`` are byte-for-byte the current V2 defaults, so
loading ``gcp_full.yaml`` reproduces the established methodology exactly.

The only derivation performed here is the two-level cascade ``input_size`` from
``training.patch_size``, using the SAME mapping the original ``train.py`` used
(64 / 96 / 128), so nothing about the model changes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# The exact PATCH -> [[low-res],[high-res]] mapping from the original train.py.
PATCH_TABLE: Dict[int, List[List[int]]] = {
    64:  [[32, 32, 24], [64, 64, 48]],
    96:  [[48, 48, 32], [96, 96, 64]],
    128: [[64, 64, 48], [128, 128, 96]],
}


@dataclass
class Config:
    """Parsed experiment configuration. Holds the raw nested dict plus a few
    convenience accessors; nothing here mutates the research defaults."""

    raw: Dict[str, Any]
    path: Optional[str] = None

    # ------------------------------------------------------------------ helpers
    def section(self, name: str) -> Dict[str, Any]:
        return self.raw.get(name, {}) or {}

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.section(section).get(key, default)

    # ------------------------------------------------------------ derived values
    @property
    def input_size(self) -> List[List[int]]:
        """Two-level cascade sizes derived from training.patch_size."""
        patch = int(self.get("training", "patch_size", 96))
        return copy.deepcopy(PATCH_TABLE.get(patch, PATCH_TABLE[96]))

    @property
    def seed(self) -> int:
        return int(self.get("experiment", "seed", 42))

    @property
    def split_seed(self) -> int:
        return int(self.get("dataset", "split_seed", self.seed))

    @property
    def name(self) -> str:
        return str(self.get("experiment", "name", "GLO-NCA-V2"))

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)


_REQUIRED_SECTIONS = [
    "experiment", "dataset", "model", "training",
    "optimizer", "loss", "ema", "gradient", "evaluation", "logging",
]


def load_config(path: str) -> Config:
    """Load and lightly validate a YAML config file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path!r} did not parse to a mapping.")

    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"Config {path!r} is missing sections: {missing}")

    patch = int(raw.get("training", {}).get("patch_size", 96))
    if patch not in PATCH_TABLE:
        raise ValueError(
            f"training.patch_size={patch} is not one of {sorted(PATCH_TABLE)}. "
            "A different patch size is a new experiment, not an auto-adjustment.")

    aug = str(raw.get("training", {}).get("augmentation", "heavy"))
    if aug not in ("none", "light", "heavy"):
        raise ValueError(f"training.augmentation={aug!r} must be none|light|heavy.")

    return Config(raw=raw, path=path)
