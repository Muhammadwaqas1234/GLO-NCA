r"""Dataset root discovery and the seeded patient split.

Extracted verbatim (behaviour-preserving) from the original train.py so the
validator and the trainer share one implementation. The split is the SAME
seeded 70/15/15 split; this module additionally lets the split be *materialised*
to files and *reloaded* so a resumed run uses identical patient IDs.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple


def resolve_data_root(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the BraTS root. Priority: explicit arg -> $DATA_ROOT -> auto-detect
    under /kaggle/input, /data, cwd. Returns None if nothing plausible is found.
    (Same logic as the original find_data_root, minus the patient count.)"""
    if explicit and os.path.isdir(explicit):
        return explicit
    env = os.environ.get("DATA_ROOT")
    if env and os.path.isdir(env):
        return env
    for base in ("/kaggle/input", "/data", os.getcwd()):
        if not os.path.isdir(base):
            continue
        for root, dirs, _ in os.walk(base):
            count = 0
            for d in dirs:
                try:
                    if any(f.endswith((".nii", ".nii.gz"))
                           for f in os.listdir(os.path.join(root, d))):
                        count += 1
                except OSError:
                    pass
            if count >= 2:
                return root
    return None


def list_patients(data_root: str) -> List[str]:
    return sorted(d for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d)))


def make_split(data_root: str, seed: int, n_patients: int = 0
               ) -> Tuple[List[str], List[str], List[str]]:
    """Seeded 70/15/15 train/val/test split -- identical to the original."""
    pats = list_patients(data_root)
    random.Random(seed).shuffle(pats)
    if n_patients:
        pats = pats[:n_patients]
    n = len(pats)
    a, b = int(n * 0.70), int(n * 0.15)
    return pats[:a], pats[a:a + b], pats[a + b:]


def write_split(ws, train: List[str], val: List[str], test: List[str]) -> None:
    """Materialise the split to split/{train,validation,test}.txt + split.json."""
    for name, ids in (("train", train), ("validation", val), ("test", test)):
        with open(ws.path("split", f"{name}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(ids) + ("\n" if ids else ""))
    ws.write_json(os.path.join("split", "split.json"),
                  {"train": train, "validation": val, "test": test,
                   "counts": {"train": len(train), "val": len(val), "test": len(test)}})


def read_split(ws) -> Optional[Tuple[List[str], List[str], List[str]]]:
    """Reload a previously materialised split (used on --resume). None if absent."""
    path = ws.path("split", "split.json")
    if not os.path.exists(path):
        return None
    import json
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return d["train"], d["validation"], d["test"]
