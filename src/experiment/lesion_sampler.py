"""Case-level small-lesion sampling (production).

Oversamples cases with small enhancing-tumour (ET) volume. Every measured
BraTS-METS case contains ET, so ET volume is the weighting signal and
small-lesion emphasis is the purpose. Case-level only; nothing is cropped.

  weight(case) = 1.0                        if the case has no ET voxels
               = 1.0 + (boost - 1.0) * s    otherwise

where s = 1.0 at or below ``small_lesion_voxels`` and decays towards 0.0 as
ET volume grows. Draws ``len(dataset)`` indices per epoch with replacement,
seeded from (base_seed, epoch), so epoch length and the LR schedule are
unchanged. Built from the training split only; validation and test stay
uniform.

Limitations: ET volume is in resampled 128³ voxels (approximate), and
``boost`` / ``small_lesion_voxels`` are not tuned.
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
        # Smallest lesions get the full boost; it decays towards 1.0 above the threshold.
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
