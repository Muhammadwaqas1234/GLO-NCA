r"""Deterministic preprocessing cache for the BraTS pipeline.

WHAT IS CACHED (and why it is safe)
-----------------------------------
Only the **deterministic** head of ``Nii_Gz_Dataset_3D.__getitem__``:

    NIfTI load -> foreground crop -> resample -> label conversion (WT/TC/ET)

Every one of those steps is a pure function of (case files, configured size,
crop/rescale flags). Verified empirically: running that stage twice under
*different* RNG seeds produces byte-identical arrays.

WHAT IS **NOT** CACHED (deliberately)
-------------------------------------
Everything stochastic stays at runtime, recomputed every epoch:

    patchify (random position, ET-aware retries), augmentation, per-(epoch,case)
    RNG seeding, sampler order, threshold decisions, model outputs.

Caching any of those would freeze randomness across epochs and silently change
the science. The cache is inserted at exactly the point the in-memory
``Data_Container`` already used, so the runtime stochastic path is untouched.

RNG NEUTRALITY
--------------
Cache lookup, read and write consume **no** ``random`` / ``numpy`` / ``torch``
random numbers. A cache hit and a cache miss leave the RNG in the same state, so
a cached run and an uncached run draw the identical stochastic sequence.

FORMAT
------
``torch.save`` of two tensors, chosen by measurement at production volume size
(128^3, 4 modalities + 3 regions):

    format              write        read      size
    npz (uncompressed)  168.6 ms   188.7 ms   56.0 MB
    npz (compressed)   6885.3 ms   815.9 ms   29.7 MB
    torch .pt           166.7 ms    27.9 ms   56.0 MB   <- chosen
    npz uint8 label     118.7 ms   115.8 ms   38.0 MB

``.pt`` reads ~6.8x faster than npz and ~54x faster than the measured 1517 ms
NIfTI materialisation. Compressed npz is disqualified by its 6.9 s write.
Labels are stored as uint8 (they are strictly binary, so this is lossless) and
restored to float32 on load, preserving the exact dtype the pipeline expects.

SAFETY
------
* identity-aware: a key derived from dataset root, case id, modality order,
  target size, crop/rescale flags, label-conversion version and cache format
  version. Any mismatch is a **miss**, never a silent reuse.
* fail-closed validation: shape, dtype, channel count and binary-label
  invariants are checked on every read; anything unexpected is treated as a
  miss and the entry is rebuilt from source.
* atomic writes: unique temp file + ``os.replace``, so a crashed or concurrent
  write can never leave a half-written entry visible.
* multiprocessing-safe: each DataLoader worker writes its own temp file; the
  final rename is atomic, so concurrent workers cannot corrupt an entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Bump when the deterministic preprocessing contract changes in a way that makes
# previously written entries invalid (e.g. a different resample or label rule).
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
        """Return (img, label) or ``None``. Consumes no random numbers.

        Any corruption, truncation, identity mismatch or invariant violation is
        reported as a MISS (never as bad data), and the caller rebuilds."""
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
        # labels are stored uint8 (lossless: strictly binary) -> restore dtype
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
            # Unique temp file per writer: several DataLoader workers may build
            # the same case concurrently. Each writes its own temp file and the
            # rename is atomic, so a reader never observes a partial file and
            # the losing writer merely overwrites with identical bytes.
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
            os.close(fd)
            torch.save({"img": torch.from_numpy(np.ascontiguousarray(img)),
                        # binary labels -> uint8 is lossless and ~4x smaller
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

    # Cache objects travel to DataLoader workers; keep them trivially picklable
    # (plain attributes only -- no file handles, locks or torch module refs).
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, st):
        self.__dict__.update(st)
