"""Load the YAML config into the training constants."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

REGIONS = ["WT", "TC", "ET"]


@dataclass
class CascadeConfig:
    NAME: str
    SEED: int
    DATA_BASE: str
    N_PATIENTS: Optional[int]
    MODALITIES: List[str]
    USE_FOREGROUND_CROP: bool
    USE_NONZERO_NORM: bool
    USE_AUG: bool
    CHANNEL_N: int
    HIDDEN: int
    USE_SPATIAL: bool
    STEPS: List[int]
    FIRE_RATE: float
    DROPOUT: float
    INPUT_SIZE: List[List[int]]
    PRIORITIZE_REGION: int
    EPOCHS: int
    BATCH_SIZE: int
    NUM_WORKERS: int
    DEVICE: str
    LR_START: float
    LR_MIN: float
    GRAD_CLIP: float
    USE_EMA: bool
    EMA_DECAY: float
    TVERSKY_BETA: float
    FOCAL_GAMMA: float
    ENSEMBLE_N: int
    USE_TTA: bool
    # Optional improvements; the defaults reproduce the original recipe exactly.
    SPLIT_SEED: int = 0
    AUGMENT: bool = False
    REGION_WEIGHTS: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    TUNE_THRESHOLDS: bool = False
    MIN_COMPONENT: Dict[str, int] = field(default_factory=lambda: {"WT": 0, "TC": 0, "ET": 0})
    FULL_RESOLUTION_EVAL: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def improved_eval(self) -> bool:
        """True when any evaluation improvement is enabled."""
        return self.TUNE_THRESHOLDS or self.FULL_RESOLUTION_EVAL or any(self.MIN_COMPONENT.values())


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: str) -> Dict[str, Any]:
    """Read a config, resolving an optional ``base:`` file relative to it."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("base", None)
    if base:
        return _merge(_read(os.path.join(os.path.dirname(path), base)), raw)
    return raw


def _set(raw: Dict[str, Any], dotted: str, value: str) -> None:
    """Apply one ``section.key=value`` override; the value is parsed as YAML."""
    section, _, key = dotted.partition(".")
    if not key or section not in raw:
        raise KeyError(f"unknown config key: {dotted}")
    raw[section][key] = yaml.safe_load(value)


def load_config(path: str, overrides: Optional[List[str]] = None) -> CascadeConfig:
    raw = _read(path)
    for item in overrides or []:
        dotted, _, value = item.partition("=")
        _set(raw, dotted.strip(), value.strip())
    e, d, m, t, l, v = (raw["experiment"], raw["data"], raw["model"],
                        raw["training"], raw["loss"], raw["evaluation"])
    min_comp = {r: int((v.get("min_component_voxels") or {}).get(r, 0)) for r in REGIONS}
    return CascadeConfig(
        NAME=str(e["name"]), SEED=int(e["seed"]),
        DATA_BASE=str(d["base"]), N_PATIENTS=d["n_patients"],
        MODALITIES=list(d["modalities"]),
        USE_FOREGROUND_CROP=bool(d["foreground_crop"]),
        USE_NONZERO_NORM=bool(d["nonzero_norm"]), USE_AUG=bool(d["augmentation"]),
        CHANNEL_N=int(m["channel_n"]), HIDDEN=int(m["hidden"]),
        USE_SPATIAL=bool(m["use_spatial"]), STEPS=list(m["steps"]),
        FIRE_RATE=float(m["fire_rate"]), DROPOUT=float(m["dropout"]),
        INPUT_SIZE=[list(s) for s in m["input_size"]],
        PRIORITIZE_REGION=int(m["prioritize_region"]),
        EPOCHS=int(t["epochs"]), BATCH_SIZE=int(t["batch_size"]),
        NUM_WORKERS=int(t["workers"]), DEVICE=str(t["device"]),
        LR_START=float(t["lr_start"]), LR_MIN=float(t["lr_min"]),
        GRAD_CLIP=float(t["grad_clip"]), USE_EMA=bool(t["ema"]),
        EMA_DECAY=float(t["ema_decay"]),
        TVERSKY_BETA=float(l["tversky_beta"]), FOCAL_GAMMA=float(l["focal_gamma"]),
        ENSEMBLE_N=int(v["ensemble_n"]), USE_TTA=bool(v["tta"]),
        SPLIT_SEED=int(e.get("split_seed") if e.get("split_seed") is not None else e["seed"]),
        AUGMENT=bool(d.get("augment", False)),
        REGION_WEIGHTS=[float(x) for x in l.get("region_weights", [1.0, 1.0, 1.0])],
        TUNE_THRESHOLDS=bool(v.get("tune_thresholds", False)),
        MIN_COMPONENT=min_comp,
        FULL_RESOLUTION_EVAL=bool(v.get("full_resolution", False)),
        raw=raw,
    )
