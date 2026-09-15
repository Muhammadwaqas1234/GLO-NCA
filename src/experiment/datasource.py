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
import re
from typing import Dict, List, Optional, Tuple

SPLIT_VERSION = "GLO-NCA-V2-master-v1"

# Required per-case files (modern BraTS names). A directory counts as a valid
# case only if it contains all four modalities + a segmentation. This is the
# same requirement the dataset loader / validator enforce, kept here so case
# DISCOVERY and case VALIDATION agree.
REQUIRED_SUFFIXES = ("t1n", "t1c", "t2w", "t2f", "seg")

# BraTS-MET case id = subject + longitudinal timepoint, e.g. BraTS-MET-00559-002.
# The trailing "-<timepoint>" is stripped to obtain the base SUBJECT id, so the
# master split can be made SUBJECT-disjoint (no timepoint of one subject leaks
# across train/val/test). For datasets without this pattern (e.g. flat BraTS-2020
# folders) the case id has no such suffix and is its own subject -> behaviour is
# identical to the original single-timepoint split.
_CASE_RE = re.compile(r"^BraTS-MET-\d+-\d+$")
_TIMEPOINT_RE = re.compile(r"-\d+$")


def _has_required_files(folder: str) -> bool:
    """True if ``folder`` holds all required modality/seg files (either sep/ext)."""
    try:
        names = [f.lower() for f in os.listdir(folder)]
    except OSError:
        return False
    for suf in REQUIRED_SUFFIXES:
        endings = tuple(f"{sep}{suf}{ext}" for sep in ("_", "-")
                        for ext in (".nii.gz", ".nii"))
        if not any(n.endswith(endings) for n in names):
            return False
    return True


def discover_cases(data_root: str) -> List[Tuple[str, str]]:
    """Recursively discover VALID case directories under ``data_root``.

    A directory is a case iff it directly contains all four modalities + seg.
    Container directories (e.g. ``UCSD - Training/``) hold no such files and are
    therefore never returned as cases; their valid sub-directories are. Returns
    a deterministically sorted list of ``(case_id, rel_path)`` where ``case_id``
    is the leaf folder name and ``rel_path`` is the path relative to
    ``data_root`` (== ``case_id`` for a flat dataset). Fails loudly on duplicate
    case ids (a real risk when two cohorts are merged).
    """
    found: List[Tuple[str, str]] = []
    for dirpath, _dirs, _files in os.walk(data_root):
        if dirpath == data_root:
            continue
        if _has_required_files(dirpath):
            case_id = os.path.basename(dirpath)
            rel = os.path.relpath(dirpath, data_root).replace("\\", "/")
            found.append((case_id, rel))
    found.sort(key=lambda t: t[0])
    ids = [c for c, _ in found]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise ValueError(
            f"duplicate case ids across the dataset: {dups[:10]}"
            f"{' ...' if len(dups) > 10 else ''}. Refusing to build an ambiguous "
            f"case set (a case id must resolve to exactly one folder).")
    return found


def discover_case_ids(data_root: str) -> List[str]:
    """Sorted list of valid case ids (see :func:`discover_cases`)."""
    return [c for c, _ in discover_cases(data_root)]


def case_path_map(data_root: str) -> Dict[str, str]:
    """Map each valid case id -> its path RELATIVE to ``data_root``."""
    return {c: rel for c, rel in discover_cases(data_root)}


def subject_of(case_id: str) -> str:
    """Base subject id for a BraTS-MET case id (strips the timepoint suffix).

    ``BraTS-MET-00559-002`` -> ``BraTS-MET-00559``. Ids that do not match the
    BraTS-MET ``<subject>-<timepoint>`` pattern are returned unchanged (so each
    is treated as its own subject) -- keeping flat single-timepoint datasets
    behaving exactly as before.
    """
    if _CASE_RE.match(case_id):
        return _TIMEPOINT_RE.sub("", case_id)
    return case_id


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
    """Sorted valid case ids under ``data_root`` (recursive, files-validated).

    For a flat single-timepoint dataset this returns exactly the top-level case
    folders as before; for a nested multi-cohort dataset (e.g. BraTS-MET with a
    ``UCSD - Training/`` sub-cohort) it additionally finds the nested cases and
    never returns container folders that lack case files.
    """
    return discover_case_ids(data_root)


def make_split(data_root: str, seed: int, n_patients: int = 0
               ) -> Tuple[List[str], List[str], List[str]]:
    """Seeded 70/15/15 SUBJECT-disjoint train/val/test split.

    Cases are grouped by base subject id (:func:`subject_of`) and whole subjects
    are assigned to a partition, so no two timepoints of the same subject land in
    different partitions (prevents temporal leakage). For a flat single-timepoint
    dataset every case is its own subject, so this reduces to the original
    per-case 70/15/15 split. Deterministic for a given seed.
    """
    cases = list_patients(data_root)
    if n_patients:
        cases = cases[:n_patients]
    # group cases by subject, deterministically
    subjects: Dict[str, List[str]] = {}
    for c in cases:
        subjects.setdefault(subject_of(c), []).append(c)
    subj_ids = sorted(subjects)
    random.Random(seed).shuffle(subj_ids)          # shuffle SUBJECTS, not cases
    n = len(subj_ids)
    a, b = int(n * 0.70), int(n * 0.15)
    parts = (subj_ids[:a], subj_ids[a:a + b], subj_ids[a + b:])
    out = []
    for group in parts:
        ids = sorted(c for s in group for c in subjects[s])
        out.append(ids)
    return out[0], out[1], out[2]


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
    integrity problem. Case IDs only -- never image data.

    The split is SUBJECT-disjoint: all longitudinal timepoints of one subject
    stay in the same partition (see :func:`make_split`). This is verified here as
    a hard gate before the split is returned, so a leaky split can never be
    written.
    """
    pats = list_patients(data_root)
    if not pats:
        raise ValueError(f"no valid case folders found under {data_root!r}")
    tr, va, te = make_split(data_root, seed, n_patients=0)

    # integrity: case-disjoint + union == population
    s_tr, s_va, s_te, pop = set(tr), set(va), set(te), set(pats)
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        raise ValueError("split partitions overlap -- refusing to write master split")
    if (s_tr | s_va | s_te) != pop:
        missing = pop - (s_tr | s_va | s_te)
        extra = (s_tr | s_va | s_te) - pop
        raise ValueError(f"split does not cover dataset (missing={len(missing)}, "
                         f"unknown={len(extra)})")

    # HARD LEAKAGE GATE: subjects must not straddle partitions.
    subj_tr = {subject_of(c) for c in tr}
    subj_va = {subject_of(c) for c in va}
    subj_te = {subject_of(c) for c in te}
    leak = (subj_tr & subj_va) | (subj_tr & subj_te) | (subj_va & subj_te)
    if leak:
        raise ValueError(f"SUBJECT LEAKAGE: {len(leak)} subject(s) appear in more "
                         f"than one partition, e.g. {sorted(leak)[:5]}. Refusing "
                         f"to write a temporally-leaky master split.")

    n_subjects = len(subj_tr | subj_va | subj_te)
    return {
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "grouping": "subject-disjoint (base id = case id minus -<timepoint>)",
        "train": tr, "validation": va, "test": te,
        "train_count": len(tr), "val_count": len(va), "test_count": len(te),
        "dataset_case_count": len(pats),
        "subject_count": n_subjects,
        "train_subject_count": len(subj_tr),
        "val_subject_count": len(subj_va),
        "test_subject_count": len(subj_te),
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
