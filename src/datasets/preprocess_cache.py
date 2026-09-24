r"""Deterministic preprocessing cache for the BraTS pipeline.

Caches only the deterministic head of Nii_Gz_Dataset_3D.__getitem__:
NIfTI load -> foreground crop -> resample to the 128³ working volume -> WT/TC/ET labels.
Patchify, augmentation, per-(epoch, case) seeding and sampling still run every epoch.
Lookups consume no random numbers, so cached and uncached runs draw identical sequences.

Format: torch.save of image float32 + label uint8 (lossless, restored to float32);
.pt reads far faster than npz or NIfTI.

Safety: identity-keyed (root, case, modality order, size, flags, label and format
versions); every read is validated and any mismatch is a miss; writes use a unique
temp file + os.replace, so concurrent workers cannot expose a partial entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Bump when the deterministic preprocessing changes (invalidates old entries).
CACHE_FORMAT_VERSION = "glonca-precache-v1"
LABEL_CONVERSION_VERSION = "wt-tc-et-v1"   # _labels_to_regions semantics


class PreprocessCache:
    """On-disk cache of deterministically preprocessed (image, label) pairs."""

    def __init__(self, directory: str, *, dataset_root: str,
                 modalities, size, crop_fg: bool, rescale: bool,
                 enabled: bool = True):
        self.enabled = bool(enabled) and bool(directory)
        self.directory = directory
        self.dataset_root = os.path.abspath(dataset_root) if dataset_root else ""
        self.modalities = list(modalities)
        self.size = tuple(int(s) for s in size)
        self.crop_fg = bool(crop_fg)
        self.rescale = bool(rescale)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.invalid = 0
        if self.enabled:
            os.makedirs(self.directory, exist_ok=True)

    # ------------------------------------------------------------- identity
    def identity(self) -> Dict[str, Any]:
        """Everything that must match for a cached entry to be reusable."""
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "label_conversion_version": LABEL_CONVERSION_VERSION,
            "dataset_root": self.dataset_root,
            "modalities": self.modalities,
            "size": list(self.size),
            "foreground_crop": self.crop_fg,
            "rescale": self.rescale,
        }

    def _fingerprint(self) -> str:
        blob = json.dumps(self.identity(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def path_for(self, case_id: str) -> str:
        safe = case_id.replace(os.sep, "_").replace("/", "_")
        return os.path.join(self.directory, f"{safe}.{self._fingerprint()}.pt")

    # ----------------------------------------------------------------- read
    def get(self, case_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (img, label) or None; any corruption or mismatch is a miss. Consumes no random numbers."""
        if not self.enabled:
            return None
        p = self.path_for(case_id)
        if not os.path.exists(p):
            self.misses += 1
            return None
        try:
            import torch
            blob = torch.load(p, map_location="cpu", weights_only=True)
            img = blob["img"].numpy()
            lab = blob["lab"].numpy()
            meta = json.loads(blob["meta"]) if isinstance(blob.get("meta"), str) \
                else blob.get("meta")
        except Exception:
            # truncated / corrupt / unreadable -> rebuild
            self.invalid += 1
            self.misses += 1
            return None

        if not self._validate(img, lab, meta):
            self.invalid += 1
            self.misses += 1
            return None

        self.hits += 1
        # Labels stored as uint8 (binary, lossless); restore float32.
        return img, lab.astype(np.float32, copy=False)

    def _validate(self, img, lab, meta) -> bool:
        if meta != self.identity():
            return False                      # different dataset/preprocessing
        if img.ndim != 4 or lab.ndim != 4:
            return False
        if tuple(img.shape[:3]) != self.size or tuple(lab.shape[:3]) != self.size:
            return False
        if img.shape[-1] != len(self.modalities) or lab.shape[-1] != 3:
            return False
        if img.dtype != np.float32:
            return False
        if not np.isfinite(img).all():
            return False
        u = np.unique(lab)
        if not np.isin(u, (0, 1)).all():      # labels must stay strictly binary
            return False
        return True

    # ---------------------------------------------------------------- write
    def put(self, case_id: str, img: np.ndarray, lab: np.ndarray) -> None:
        """Atomically store an entry. Never raises into the training path."""
        if not self.enabled:
            return
        try:
            import torch
            final = self.path_for(case_id)
            # Unique temp file per writer; the atomic rename means readers never see a partial file.
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
            os.close(fd)
            torch.save({"img": torch.from_numpy(np.ascontiguousarray(img)),
                        # Binary labels -> uint8: lossless, about 4x smaller.
                        "lab": torch.from_numpy(
                            np.ascontiguousarray(lab).astype(np.uint8)),
                        "meta": json.dumps(self.identity(), sort_keys=True)},
                       tmp)
            os.replace(tmp, final)            # atomic publish
            self.writes += 1
        except Exception:
            # A cache failure must never break training: fall back to recompute.
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    # -------------------------------------------------------------- reporting
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {"enabled": self.enabled, "directory": self.directory,
                "fingerprint": self._fingerprint() if self.enabled else None,
                "hits": self.hits, "misses": self.misses,
                "writes": self.writes, "invalid_entries": self.invalid,
                "hit_rate": round(self.hits / total, 4) if total else None}

    # Cache objects are sent to DataLoader workers; keep them picklable (plain attributes only).
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, st):
        self.__dict__.update(st)
