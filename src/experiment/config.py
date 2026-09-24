r"""Configuration loading for GLO-NCA experiments.

Loads a YAML file into a validated ``Config``. The only derivation is the
legacy V2 cascade ``input_size``; the production two-level model derives its
per-level resolutions itself.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# Legacy V2: patch size -> [[low-res], [high-res]] cascade sizes.
PATCH_TABLE: Dict[int, List[List[int]]] = {
    64:  [[32, 32, 24], [64, 64, 48]],
    96:  [[48, 48, 32], [96, 96, 64]],
    128: [[64, 64, 48], [128, 128, 96]],
}


@dataclass
class Config:
    """Parsed experiment configuration: the raw nested dict plus read-only accessors."""

    raw: Dict[str, Any]
    path: Optional[str] = None

    # Helpers.
    def section(self, name: str) -> Dict[str, Any]:
        return self.raw.get(name, {}) or {}

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.section(section).get(key, default)

    # Derived values.
    @property
    def input_size(self) -> List[List[int]]:
        """Legacy V2 cascade sizes derived from training.patch_size."""
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
        return str(self.get("experiment", "name", "GLO-NCA"))

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

    # PATCH_TABLE applies to legacy V2 only; the two-level (v3) model accepts any positive patch_size.
    is_v3 = str(raw.get("model", {}).get("version", "")).lower() == "v3"
    patch = int(raw.get("training", {}).get("patch_size", 96))
    if not is_v3 and patch not in PATCH_TABLE:
        raise ValueError(
            f"training.patch_size={patch} is not one of {sorted(PATCH_TABLE)}. "
            "A different patch size is a new experiment, not an auto-adjustment.")
    if is_v3 and patch <= 0:
        raise ValueError(f"V3 training.patch_size must be positive, got {patch}.")

    aug = str(raw.get("training", {}).get("augmentation", "heavy"))
    if aug not in ("none", "light", "heavy"):
        raise ValueError(f"training.augmentation={aug!r} must be none|light|heavy.")

    return Config(raw=raw, path=path)
