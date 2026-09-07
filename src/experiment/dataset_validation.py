r"""Professional dataset validation for BraTS-style data.

Checks every patient BEFORE expensive training: presence and readability of all
four modalities + segmentation, dimension consistency, NaN/Inf, unexpected
segmentation labels, duplicates and empty/corrupt files. Returns a PASS/FAIL
report and never silently skips a corrupted patient.
"""
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
                     modalities: List[str]) -> Dict[str, Any]:
    """Validate a single patient folder. Returns a dict with ok/errors/info."""
    errors: List[str] = []
    shapes: Dict[str, Tuple[int, ...]] = {}

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
            bad = uniq - ALLOWED_SEG_LABELS
            if bad:
                errors.append(f"unexpected seg labels: {sorted(bad)}")
        else:
            if np.isnan(vol).any():
                errors.append(f"NaN values in {mod}")
            if np.isinf(vol).any():
                errors.append(f"Inf values in {mod}")

    # Dimension consistency across all present volumes.
    if len(set(shapes.values())) > 1:
        errors.append(f"inconsistent dimensions: {shapes}")

    return {"patient": patient, "ok": not errors,
            "errors": errors, "shapes": shapes}


def validate_dataset(root: str,
                     modalities: Optional[List[str]] = None,
                     limit: Optional[int] = None) -> Dict[str, Any]:
    """Validate every patient under ``root``. Returns a full report dict with a
    top-level ``result`` of "PASS" or "FAIL"."""
    modalities = modalities or DEFAULT_MODALITIES
    if not root or not os.path.isdir(root):
        return {"result": "FAIL", "root": root,
                "fatal": f"dataset root not found: {root}", "patients": []}

    folders = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    if limit:
        folders = folders[:limit]

    # Duplicate patient IDs (case-insensitive) -- a real risk on merged datasets.
    lowered = [f.lower() for f in folders]
    dups = sorted({f for f in lowered if lowered.count(f) > 1})

    results = [validate_patient(os.path.join(root, f), f, modalities)
               for f in folders]
    n_ok = sum(r["ok"] for r in results)
    n_bad = len(results) - n_ok

    passed = (n_bad == 0 and not dups and len(results) > 0)
    return {
        "result": "PASS" if passed else "FAIL",
        "root": root,
        "modalities": modalities,
        "patient_count": len(results),
        "passed_patients": n_ok,
        "failed_patients": n_bad,
        "duplicate_ids": dups,
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
        f"  patients: {report['patient_count']} "
        f"(pass {report['passed_patients']}, fail {report['failed_patients']})",
    ]
    if report.get("duplicate_ids"):
        lines.append(f"  duplicate IDs: {report['duplicate_ids']}")
    for r in report["patients"]:
        if not r["ok"]:
            lines.append(f"  FAIL {r['patient']}: {'; '.join(r['errors'])}")
    return "\n".join(lines)
