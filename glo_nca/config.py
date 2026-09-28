"""Load the YAML config into the training constants."""
from __future__ import annotations

import copy
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
    raw: Dict[str, Any] = field(default_factory=dict)


def _set(raw: Dict[str, Any], dotted: str, value: str) -> None:
    """Apply one ``section.key=value`` override; the value is parsed as YAML."""
    section, _, key = dotted.partition(".")
    if not key or section not in raw or key not in raw[section]:
        raise KeyError(f"unknown config key: {dotted}")
    raw[section][key] = yaml.safe_load(value)


def load_config(path: str, overrides: Optional[List[str]] = None) -> CascadeConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        dotted, _, value = item.partition("=")
        _set(raw, dotted.strip(), value.strip())
    e, d, m, t, l, v = (raw["experiment"], raw["data"], raw["model"],
                        raw["training"], raw["loss"], raw["evaluation"])
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
        raw=raw,
    )
