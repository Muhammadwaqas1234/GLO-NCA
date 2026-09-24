r"""Dataset validation before training: files, readability, dimensions, NaN/Inf, labels and duplicates."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# BraTS raw label values we accept (0 bg, 1 NCR, 2 ED, 3/4 ET depending on year).
ALLOWED_SEG_LABELS = {0, 1, 2, 3, 4}
DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
SEG_SUFFIX = "seg"


def _find_file(folder: str, patient: str, suffix: str) -> Optional[str]:
    """Locate a modality/seg file tolerant to BraTS naming (mirrors the loader)."""
    for sep in ("_", "-"):
        for ext in (".nii.gz", ".nii"):
            cand = os.path.join(folder, f"{patient}{sep}{suffix}{ext}")
            if os.path.exists(cand):
                return cand
    endings = tuple(f"{sep}{suffix}{ext}"
                    for sep in ("_", "-") for ext in (".nii.gz", ".nii"))
    try:
        for f in os.listdir(folder):
            if f.lower().endswith(endings):
                return os.path.join(folder, f)
    except OSError:
        pass
    return None


def _load(path: str):
    import nibabel as nib
    return nib.load(path).get_fdata()


def validate_patient(folder: str, patient: str,
                     modalities: List[str],
                     allowed_seg_labels: Optional[set] = None) -> Dict[str, Any]:
    """Validate one patient folder; ``allowed_seg_labels`` widens labels for this case only (per policy)."""
    allowed = allowed_seg_labels or ALLOWED_SEG_LABELS
    errors: List[str] = []
    shapes: Dict[str, Tuple[int, ...]] = {}
    tolerated_labels: List[int] = []   # stray labels accepted by explicit policy

    for mod in modalities + [SEG_SUFFIX]:
        path = _find_file(folder, patient, mod)
        if path is None:
            errors.append(f"missing modality/seg: {mod}")
            continue
        if os.path.getsize(path) == 0:
            errors.append(f"empty file: {os.path.basename(path)}")
            continue
        try:
            vol = _load(path)
        except Exception as exc:
            errors.append(f"unreadable {mod}: {exc}")
            continue
        shapes[mod] = tuple(vol.shape)
        if mod == SEG_SUFFIX:
            uniq = set(np.unique(vol).astype(int).tolist())
            bad = uniq - allowed
            if bad:
                errors.append(f"unexpected seg labels: {sorted(bad)}")
            # Record accepted stray labels separately from the dimension-check shapes.
            tolerated_labels.extend(sorted((uniq - ALLOWED_SEG_LABELS) & allowed))
        else:
            if np.isnan(vol).any():
                errors.append(f"NaN values in {mod}")
            if np.isinf(vol).any():
                errors.append(f"Inf values in {mod}")

    # Dimension consistency across all present volumes.
    if len(set(shapes.values())) > 1:
        errors.append(f"inconsistent dimensions: {shapes}")

    out = {"patient": patient, "ok": not errors,
           "errors": errors, "shapes": shapes}
    if tolerated_labels:
        out["tolerated_seg_labels"] = sorted(set(tolerated_labels))
    return out


def _validation_workers() -> int:
    """Threads for the dataset scan. GLO_VALIDATE_WORKERS overrides; 1 disables."""
    import multiprocessing
    raw_value = os.environ.get("GLO_VALIDATE_WORKERS")
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            pass
    try:
        return max(1, min(16, multiprocessing.cpu_count()))
    except Exception:
        return 1


def validate_dataset(root: str,
                     modalities: Optional[List[str]] = None,
                     limit: Optional[int] = None,
                     policy: Optional[Any] = None) -> Dict[str, Any]:
    """Validate every patient under ``root``; returns a report with ``result`` PASS or FAIL."""
    modalities = modalities or DEFAULT_MODALITIES
    if not root or not os.path.isdir(root):
        return {"result": "FAIL", "root": root,
                "fatal": f"dataset root not found: {root}", "patients": []}

    # Recursive case discovery (finds nested cohorts); falls back to a flat scan.
    try:
        from src.experiment.datasource import discover_cases
        cases = discover_cases(root)  # [(case_id, rel_path)] sorted by id
    except Exception:
        cases = sorted((d, d) for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
    if limit:
        cases = cases[:limit]

    top_level = sum(1 for _cid, rel in cases if "/" not in rel)
    nested = len(cases) - top_level

    # Duplicate case ids (case-insensitive).
    lowered = [cid.lower() for cid, _ in cases]
    dups = sorted({c for c in lowered if lowered.count(c) > 1})

    if policy is not None:
        policy.assert_known_cases({cid for cid, _ in cases})

    todo = [(cid, rel) for cid, rel in cases
            if policy is None or not policy.is_excluded(cid)]

    # Validate cases in parallel; result order is restored, so the report matches a serial scan.
    workers = _validation_workers()
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(
                lambda item: validate_patient(
                    os.path.join(root, item[1]), item[0], modalities,
                    allowed_seg_labels=(policy.allowed_labels_for(item[0])
                                        if policy is not None else None)),
                todo))
    else:
        results = [
            validate_patient(os.path.join(root, rel), cid, modalities,
                             allowed_seg_labels=(policy.allowed_labels_for(cid)
                                                 if policy is not None else None))
            for cid, rel in todo
        ]
    n_ok = sum(r["ok"] for r in results)
    n_bad = len(results) - n_ok

    passed = (n_bad == 0 and not dups and len(results) > 0)
    return {
        "result": "PASS" if passed else "FAIL",
        "root": root,
        "modalities": modalities,
        "patient_count": len(results),
        "top_level_cases": top_level,
        "nested_cases": nested,
        "passed_patients": n_ok,
        "failed_patients": n_bad,
        "duplicate_ids": dups,
        "data_quality_policy": (policy.summary() if policy is not None else None),
        "tolerated_cases": {r["patient"]: r["tolerated_seg_labels"]
                            for r in results if r.get("tolerated_seg_labels")},
        "patients": results,
    }


def summarize(report: Dict[str, Any]) -> str:
    """Human-readable one-block summary of a validation report."""
    lines = [f"Dataset validation: {report['result']}",
             f"  root: {report.get('root')}"]
    if report.get("fatal"):
        lines.append(f"  FATAL: {report['fatal']}")
        return "\n".join(lines)
    lines += [
        f"  cases: {report['patient_count']} "
        f"(pass {report['passed_patients']}, fail {report['failed_patients']})",
    ]
    if "top_level_cases" in report:
        lines.append(f"  cohorts: top-level {report['top_level_cases']}, "
                     f"nested {report['nested_cases']}")
    if report.get("duplicate_ids"):
        lines.append(f"  duplicate IDs: {report['duplicate_ids']}")
    for r in report["patients"]:
        if not r["ok"]:
            lines.append(f"  FAIL {r['patient']}: {'; '.join(r['errors'])}")
    return "\n".join(lines)
