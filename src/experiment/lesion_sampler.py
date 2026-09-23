"""Case-level SMALL-LESION-aware sampling (EXPERIMENTAL -- never the baseline).

WHAT THIS ACTUALLY DOES -- read this before citing the mechanism.

  It oversamples cases whose ENHANCING-TUMOUR VOLUME IS SMALL. It is NOT
  "ET-positive oversampling", and must not be described as such.

  MEASURED on the BraTS-METS training split (extra/scripts/analyze_et_voxels.py):
  every measured case contains enhancing tumour, so the ET-positive share is
  already 1.0 under uniform sampling and cannot be increased. ET presence is
  definitional in a metastases cohort, unlike BraTS-GLIOMA where ET is often
  absent. What varies -- across roughly three orders of magnitude -- is ET
  VOLUME, and that is the only thing this sampler can and does act on.

  ET volume is therefore the weighting SIGNAL; small-lesion emphasis is the
  PURPOSE. The class name and the ``et_*`` fields refer to the signal.

The production baseline is uniform case-level sampling via ``_EpochSampler``.
This module is the single alternative sampler. Per the audit brief,
foreground-aware sampling, small-lesion oversampling and class-balanced
sampling are ONE mechanism here, not three: all are expressed as case weights
on the same sampler.

SCOPE -- this is CASE-level, not patch-level. Patchify remains OFF; nothing in
this module crops, and the working volume is unchanged.

WEIGHTING
  weight(case) = 1.0                        if the case has no ET voxels
               = 1.0 + (boost - 1.0) * s    if the case has ET voxels

  where s = 1.0 for a case whose ET volume is at or below
  ``small_lesion_voxels`` and decays to 0.0 as ET volume grows, so the
  smallest-lesion cases receive the largest boost and the largest lesions
  approach weight 1.0. ``boost`` defaults to 2.0. On a cohort where every case
  has ET, the effect is purely a re-weighting BY SIZE. This is the simplest
  defensible scheme, NOT a tuned optimum -- see LIMITATIONS.

ET-VOLUME MEASUREMENT
  From the ground-truth label of the TRAINING split only, counting voxels in
  the ET channel (REGIONS index 2). Statistics are computed once, from the
  deterministic preprocessing head, and cached.

SAMPLING
  With replacement, drawing exactly ``len(dataset)`` indices per epoch, so
  epoch length and optimiser-step count are IDENTICAL to the baseline and the
  cosine schedule is unaffected. Weights are normalised to a probability
  distribution. The generator is seeded from (base_seed, epoch) exactly like
  ``_EpochSampler``, so the draw is reproducible and differs per epoch.

ISOLATION
  Constructed from the training split only. Validation and test keep uniform,
  unshuffled, full-pass evaluation; this sampler is never applied to them.

LIMITATIONS
  * ET volume is counted in resampled 128^3 voxels (per-case scale factor, no
    spacing carried through), so "small" is approximate.
  * Sampling with replacement means some cases are unseen in a given epoch
    while others repeat; over many epochs this evens out, but a single epoch
    is no longer a full pass over the training set.
  * boost and small_lesion_voxels are NOT tuned. Tuning them on validation
    would be legitimate but has not been done; tuning on test is forbidden.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

ET_INDEX = 2


class LesionAwareSampler(torch.utils.data.Sampler):
    """Weighted case sampler yielding (epoch, index) like ``_EpochSampler``."""

    def __init__(self, et_voxels: Sequence[int], base_seed: int,
                 boost: float = 2.0, small_lesion_voxels: int = 100):
        self.n = len(et_voxels)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.boost = float(boost)
        self.small_lesion_voxels = int(small_lesion_voxels)
        self.et_voxels = np.asarray(et_voxels, dtype=np.float64)
        self.weights = self._compute_weights()
        self.probs = self.weights / self.weights.sum()

    def _compute_weights(self) -> np.ndarray:
        w = np.ones(self.n, dtype=np.float64)
        positive = self.et_voxels > 0
        if not positive.any():
            return w
        # Smallest ET lesions get the full boost; it decays toward 1.0 as the
        # lesion grows past the small-lesion threshold.
        scale = np.zeros(self.n, dtype=np.float64)
        thr = max(1.0, float(self.small_lesion_voxels))
        scale[positive] = np.clip(thr / np.maximum(self.et_voxels[positive], 1.0),
                                  0.0, 1.0)
        w = 1.0 + (self.boost - 1.0) * scale
        w[~positive] = 1.0
        return np.maximum(w, 1e-8)

    def set_epoch(self, ep: int) -> None:
        self.epoch = int(ep)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng((self.base_seed, self.epoch))
        idx = rng.choice(self.n, size=self.n, replace=True, p=self.probs)
        for i in idx:
            yield (self.epoch, int(i))

    def describe(self) -> dict:
        positive = self.et_voxels > 0
        return {
            "sampler": "LesionAwareSampler",
            "cases": int(self.n),
            "et_positive_cases": int(positive.sum()),
            "et_negative_cases": int((~positive).sum()),
            "boost": self.boost,
            "small_lesion_voxels": self.small_lesion_voxels,
            "weight_min": float(self.weights.min()),
            "weight_max": float(self.weights.max()),
            "samples_per_epoch": int(self.n),
            "with_replacement": True,
        }


def et_voxel_counts(dataset, indices: Sequence[int]) -> List[int]:
    """ET voxel count per training case, from ground truth only."""
    counts: List[int] = []
    for i in indices:
        _, _, label = dataset[i][:3] if isinstance(dataset[i], tuple) else (None, None, None)
        counts.append(int((np.asarray(label)[..., ET_INDEX] > 0.5).sum()))
    return counts
