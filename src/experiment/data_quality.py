r"""Operational data-quality policy (canonical split preserved).

Two training cases carry stray labels outside {0,1,2,3,4}:

    BraTS-MET-01094-002  label 6, 129 voxels
    BraTS-MET-01184-002  label 8,  28 voxels

Label conversion maps only {1,2,3,4} to WT/TC/ET, so these voxels become
background and the targets stay valid. The cases are kept and tolerated
explicitly rather than excluded or edited.

Rules: tolerance is per case and per label, nothing is auto-discovered, and
a missing or malformed policy fails loudly.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

# Raw BraTS label values accepted everywhere, for every case.
BASE_ALLOWED_SEG_LABELS: Set[int] = {0, 1, 2, 3, 4}

DEFAULT_POLICY_PATH = os.path.join("split", "data_quality_policy.json")


class DataQualityPolicyError(RuntimeError):
    """Raised when the policy file is missing, malformed or inconsistent."""


class DataQualityPolicy:
    """An explicit, per-case data-quality policy. Fail-closed by construction."""

    def __init__(self, raw: Dict[str, Any], path: str):
        self.path = path
        self.raw = raw
        self.version = str(raw.get("policy_version") or "")
        if not self.version:
            raise DataQualityPolicyError(
                f"{path}: 'policy_version' is required.")

        # case_id -> set of stray labels tolerated for THAT case only
        self._tolerated: Dict[str, Set[int]] = {}
        for i, entry in enumerate(raw.get("tolerated_seg_labels", []) or []):
            cid = entry.get("case_id")
            labels = entry.get("stray_labels")
            if not cid or not isinstance(labels, list) or not labels:
                raise DataQualityPolicyError(
                    f"{path}: tolerated_seg_labels[{i}] needs 'case_id' and a "
                    "non-empty 'stray_labels' list.")
            if not entry.get("reason"):
                raise DataQualityPolicyError(
                    f"{path}: tolerated_seg_labels[{i}] ({cid}) has no 'reason'. "
                    "Every tolerance must be justified in writing.")
            try:
                vals = {int(v) for v in labels}
            except (TypeError, ValueError) as exc:
                raise DataQualityPolicyError(
                    f"{path}: tolerated_seg_labels[{i}] ({cid}) has a "
                    f"non-integer label: {exc}") from exc
            overlap = vals & BASE_ALLOWED_SEG_LABELS
            if overlap:
                raise DataQualityPolicyError(
                    f"{path}: {cid} tolerates {sorted(overlap)}, which is already "
                    "universally allowed -- the entry is meaningless. Remove it.")
            self._tolerated[str(cid)] = vals

        self.excluded: List[str] = [str(c) for c in (raw.get("excluded_cases") or [])]
        if set(self.excluded) & set(self._tolerated):
            raise DataQualityPolicyError(
                f"{path}: a case cannot be both tolerated and excluded.")

    # Query.
    def allowed_labels_for(self, case_id: str) -> Set[int]:
        """Labels accepted for this case: the universal set plus its listed stray labels."""
        return BASE_ALLOWED_SEG_LABELS | self._tolerated.get(case_id, set())

    def is_excluded(self, case_id: str) -> bool:
        return case_id in self.excluded

    def summary(self) -> Dict[str, Any]:
        return {
            "policy_version": self.version,
            "policy_file": self.path,
            "tolerated_cases": {k: sorted(v) for k, v in self._tolerated.items()},
            "excluded_cases": list(self.excluded),
        }

    def describe(self) -> str:
        if not self._tolerated and not self.excluded:
            return f"data-quality policy {self.version}: no tolerances, no exclusions"
        parts = [f"data-quality policy {self.version}"]
        for cid, labels in sorted(self._tolerated.items()):
            parts.append(f"tolerate {cid} labels {sorted(labels)}")
        if self.excluded:
            parts.append(f"exclude {self.excluded}")
        return "; ".join(parts)

    # Integrity binding.
    def assert_known_cases(self, known: Set[str]) -> None:
        """Every policy id must exist in the split/dataset, so a stale id cannot silently do nothing."""
        named = set(self._tolerated) | set(self.excluded)
        unknown = sorted(named - known)
        if unknown:
            raise DataQualityPolicyError(
                f"{self.path}: policy names case ids that are not in the "
                f"dataset/split: {unknown}. Refusing to proceed with a stale "
                "or mistyped policy.")


def load_policy(path: Optional[str] = None, *, required: bool = False
                ) -> Optional[DataQualityPolicy]:
    """Load the policy; an absent file returns None (no tolerances), a malformed one always raises."""
    p = path or DEFAULT_POLICY_PATH
    if not os.path.isabs(p):
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        cand = p if os.path.exists(p) else os.path.join(root, p)
    else:
        cand = p
    if not os.path.exists(cand):
        if required:
            raise DataQualityPolicyError(f"data-quality policy not found: {cand}")
        return None
    try:
        with open(cand, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        raise DataQualityPolicyError(
            f"{cand}: policy file is present but unreadable/malformed: {exc}"
        ) from exc
    return DataQualityPolicy(raw, cand)
