r"""Dataset root discovery and the seeded patient split.

Extracted verbatim (behaviour-preserving) from the original train.py so the
validator and the trainer share one implementation. The split is the SAME
seeded 70/15/15 split; this module additionally lets the split be *materialised*
to files and *reloaded* so a resumed run uses identical patient IDs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Tuple

SPLIT_VERSION = "GLO-NCA-V2-master-v1"


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
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return d["train"], d["validation"], d["test"]


# =============================================================================
# Master split -- ONE canonical patient-level split shared by every experiment.
# =============================================================================
def split_fingerprint(train: List[str], val: List[str], test: List[str]) -> str:
    """Deterministic sha256 over the canonical (sorted-within-partition) split.
    Order of partitions is fixed; IDs are sorted so the fingerprint depends only
    on which patient is in which partition, not on list order."""
    canon = json.dumps({"train": sorted(train), "val": sorted(val),
                        "test": sorted(test)}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def patient_id_hash(all_ids: List[str]) -> str:
    return hashlib.sha256("\n".join(sorted(all_ids)).encode()).hexdigest()


def build_master_split(data_root: str, seed: int = 42) -> Dict:
    """Build the canonical split dict from the dataset. Fails loudly on any
    integrity problem. Patient IDs only -- never image data."""
    pats = list_patients(data_root)
    if not pats:
        raise ValueError(f"no patient folders found under {data_root!r}")
    tr, va, te = make_split(data_root, seed, n_patients=0)

    # integrity: disjoint + union == population
    s_tr, s_va, s_te, pop = set(tr), set(va), set(te), set(pats)
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        raise ValueError("split partitions overlap -- refusing to write master split")
    if (s_tr | s_va | s_te) != pop:
        missing = pop - (s_tr | s_va | s_te)
        extra = (s_tr | s_va | s_te) - pop
        raise ValueError(f"split does not cover dataset (missing={len(missing)}, "
                         f"unknown={len(extra)})")

    return {
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "train": tr, "validation": va, "test": te,
        "train_count": len(tr), "val_count": len(va), "test_count": len(te),
        "dataset_case_count": len(pats),
        "split_sha256": split_fingerprint(tr, va, te),
        "patient_id_hash": patient_id_hash(pats),
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat(timespec="seconds"),
    }


def load_master_split(path: str) -> Dict:
    """Load a master split file. Fails loudly if absent or structurally invalid."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"configured master split not found: {path}. Create it once with "
            f"scripts/create_master_split.py -- it is NOT auto-regenerated.")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    for k in ("train", "validation", "test", "split_sha256"):
        if k not in d:
            raise ValueError(f"master split {path} missing key {k!r}")
    # re-verify the stored fingerprint matches the stored IDs (tamper check)
    recomputed = split_fingerprint(d["train"], d["validation"], d["test"])
    if recomputed != d["split_sha256"]:
        raise ValueError(f"master split fingerprint mismatch in {path} "
                         f"(stored {d['split_sha256'][:12]}, recomputed "
                         f"{recomputed[:12]}) -- file altered? refusing to use.")
    return d
