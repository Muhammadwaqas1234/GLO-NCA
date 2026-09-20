r"""
================================================================================
GLO-NCA — GLOBAL CONTEXT + EFFICIENT MULTI-LEVEL FUSION
================================================================================
A SUBCLASS of ``GLO_NCA_V3_MultiLevel``. The frozen reference implementation in
``Model_GLO_NCA_V3.py`` is NOT modified; only ``forward`` is overridden.

ARCHITECTURE
------------
    FULL VOLUME
        -> GLOBAL CONTEXT LEVEL (L1, 64^3)  -- sees the WHOLE working volume
        -> GLO-NCA + SE + spatial global context
        -> GLOBAL CONTEXT FEATURES
        -> HIGH-RESOLUTION SEGMENTATION LEVEL (L2)
        -> GLO-NCA + SE + spatial global context
        -> LEARNABLE MULTI-LEVEL PROJECTIONS
        -> LEARNABLE MULTI-LEVEL FUSION
        -> WT / TC / ET

GLO-NCA uses a lightweight multi-level architecture in which a GLOBAL contextual
representation supplies whole-volume information while the high-resolution level
captures local segmentation detail, followed by learnable multi-level fusion.

THE EFFICIENCY MECHANISM (internal only)
----------------------------------------
In the reference forward every level receives the whole volume:

    mod_i = self._resize_cl(modalities_cl, res, mode="trilinear")   # full extent

so the expensive high-resolution level processes the entire field. Here the
GLOBAL level still processes the whole volume, while the high-resolution level
computes over a region of interest, which cuts its spatial workload (measured:
884,736 -> 262,144 voxels at ``roi_fraction = 2/3``).

The ROI is an INTERNAL COMPUTATIONAL OPTIMIZATION. It does not remove the global
pathway: L1 is computed over the entire working volume, and its state is carried
into the high-resolution level through the existing learnable projection, so
whole-volume context still reaches the segmentation pathway.

ROI coordinates are NORMALISED (fractions of extent), which is what makes the
coarse and fine grids describe the SAME anatomy despite different grid sizes.
``last_roi_box()`` exposes those fractions so the training target can be cropped
with EXACTLY the same coordinates -- never cropped independently.

WHAT IS PRESERVED
-----------------
SE channel attention, the spatial global-context block, learnable inter-level
projections, learnable multi-level fusion, the WT/TC/ET head, parameter count,
channels, hidden size, fire rate, dropout and NCA step counts.

SCIENTIFIC CONSEQUENCE (Category C)
-----------------------------------
With ``roi_fraction < 1`` the high-resolution level's SE / spatial-GC pool over
the ROI rather than the whole field. The GLOBAL level still pools over the entire
volume, which is what preserves the global-context contribution. This is a
methodology change and must be approved, not assumed.

``roi_fraction = 1.0`` (the default) is mathematically identical to the reference
forward and is used as the benchmark baseline and for evaluation.
================================================================================
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.models.Model_GLO_NCA_V3 import (GLO_NCA_V3_MultiLevel, LevelSpec,
                                         _to_cf, _to_cl)

# A normalised ROI box: ((x0,x1), (y0,y1), (z0,z1)) as fractions of extent.
RoiBox = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


class GLO_NCA_GlobalContext(GLO_NCA_V3_MultiLevel):
    r"""GLO-NCA with a whole-volume global-context level and an efficient
    high-resolution level.

    #Args (in addition to the base class):
        roi_fraction: side fraction of the volume the high-resolution level
            covers, in (0, 1]. 1.0 reproduces the reference forward exactly.
        global_levels: how many leading levels stay whole-volume (default 1).
    """

    def __init__(self, *args, roi_fraction: float = 1.0,
                 global_levels: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0.0 < roi_fraction <= 1.0):
            raise ValueError(f"roi_fraction must be in (0, 1], got {roi_fraction}")
        if not (1 <= global_levels <= len(self.levels)):
            raise ValueError(
                f"global_levels must be in [1, {len(self.levels)}], got {global_levels}")
        self.roi_fraction = float(roi_fraction)
        self.global_levels = int(global_levels)
        self._roi_origin: Optional[Tuple[float, float, float]] = None
        # Per-sample normalised boxes from the most recent forward, so the
        # training target can be cropped with identical coordinates.
        self._last_roi: List[RoiBox] = []

    # ------------------------------------------------------------------ ROI
    def set_roi_origin(self, origin: Optional[Tuple[float, float, float]]) -> None:
        """Pin the ROI origin in NORMALISED coords, or ``None`` to restore the
        default (random while training, centred while evaluating)."""
        self._roi_origin = origin

    def last_roi_box(self) -> List[RoiBox]:
        """Normalised ROI boxes used by the most recent ``forward``, one per
        batch sample. Empty when ``roi_fraction == 1`` (no crop was applied).

        The training target MUST be cropped with these exact fractions.
        """
        return list(self._last_roi)

    def _origin(self) -> Tuple[float, float, float]:
        span = 1.0 - self.roi_fraction
        if self._roi_origin is not None:
            return self._roi_origin
        if span <= 0.0:
            return (0.0, 0.0, 0.0)
        if self.training:
            r = torch.rand(3)
            return (float(r[0]) * span, float(r[1]) * span, float(r[2]) * span)
        return (span / 2.0, span / 2.0, span / 2.0)      # deterministic centre

    @staticmethod
    def crop_normalised(x_cl: torch.Tensor, box: RoiBox) -> torch.Tensor:
        """Crop a channels-last volume to a NORMALISED box.

        Shared by the model and by the target-cropping helper, so image and label
        can never diverge: both call this with the same ``box``.

        The crop SIZE is derived from the box WIDTH alone and the start is then
        clamped, so every sample in a batch yields the SAME spatial dimensions
        even though each has its own origin. Deriving start and stop
        independently would round differently per sample (e.g. 10 vs 11 voxels)
        and make the batch impossible to stack.
        """
        b, X, Y, Z, c = x_cl.shape
        idx = []
        for n, (lo, hi) in zip((X, Y, Z), box):
            size = max(1, min(n, int(round(n * (hi - lo)))))   # width-derived
            start = int(round(n * lo))
            start = max(0, min(start, n - size))               # clamp, keep size
            idx.append((start, start + size))
        (x0, x1), (y0, y1), (z0, z1) = idx
        return x_cl[:, x0:x1, y0:y1, z0:z1, :]

    def _box_from_origin(self, origin: Tuple[float, float, float]) -> RoiBox:
        f = self.roi_fraction
        return tuple((o, o + f) for o in origin)  # type: ignore[return-value]

    # -------------------------------------------------------------- forward
    def forward(self, modalities_cl: torch.Tensor) -> torch.Tensor:
        from src.profiling import get_profiler
        prof = get_profiler()

        # FULL-VOLUME EVALUATION GUARD. The ROI is a TRAINING-ONLY efficiency
        # mechanism. Whenever the module is not in training mode -- which is how
        # `metrics_eval.collect_probs` runs validation and test -- the whole
        # volume is processed regardless of `roi_fraction`, so evaluation can
        # never score a sub-region and the frozen-test protocol is preserved.
        # This makes the guarantee structural rather than dependent on every
        # caller remembering to pass `force_full_volume=True`.
        frac = self.roi_fraction if self.training else 1.0
        batch = modalities_cl.shape[0]
        if frac >= 1.0:
            self._last_roi = []
            boxes: List[Optional[RoiBox]] = [None] * batch
        else:
            # One ROI per sample, so different samples see different anatomy.
            boxes = [self._box_from_origin(self._origin()) for _ in range(batch)]
            self._last_roi = list(boxes)  # type: ignore[arg-type]

        def crop_batch(x_cl: torch.Tensor) -> torch.Tensor:
            """Crop each sample with its own box; shapes stay uniform because
            every box has the same fractional size."""
            if frac >= 1.0:
                return x_cl
            parts = [self.crop_normalised(x_cl[i:i + 1], boxes[i]) for i in range(batch)]
            return torch.cat(parts, dim=0)

        prev_state: Optional[torch.Tensor] = None
        level_states: List[torch.Tensor] = []

        for i, (lv, nca) in enumerate(zip(self.levels, self.ncas)):
            is_global = i < self.global_levels
            with prof.section(f"model/level{i + 1}", cuda=True):
                res = lv.resolution
                if is_global:
                    # GLOBAL CONTEXT: the whole working volume, as in the reference.
                    mod_i = self._resize_cl(modalities_cl, res, mode="trilinear")
                else:
                    # HIGH-RESOLUTION: crop first, then resize only the ROI, so
                    # fewer voxels are processed at this level.
                    mod_i = self._resize_cl(crop_batch(modalities_cl),
                                            max(1, int(round(res * frac))),
                                            mode="trilinear")
                seed = self._seed(mod_i, lv.channels)

                if prev_state is not None:
                    # Carry the previous level's state forward. Crossing from a
                    # global level into the high-resolution level, crop the
                    # global state to the SAME ROI -- this is how whole-volume
                    # context reaches the segmentation pathway.
                    src = prev_state
                    if (not is_global) and (i - 1) < self.global_levels:
                        src = crop_batch(src)
                    proj = self.projections[i - 1](src)
                    proj = self._resize_cl(proj, mod_i.shape[1], mode="nearest")
                    seed = seed.clone()
                    seed[..., self.input_channels:] = seed[..., self.input_channels:] + proj

                out = nca(seed, steps=lv.nca_steps, fire_rate=self.fire_rate)
                prev_state = out
                level_states.append(out)

        # ---- LEARNABLE MULTI-LEVEL FUSION (unchanged in kind) ----
        with prof.section("model/fusion", cuda=True):
            fine_res = level_states[-1].shape[1]
            fused = []
            for i, (state, to_fine) in enumerate(zip(level_states, self.level_to_fine)):
                s = state
                # Align every contribution to the same anatomy before fusing.
                if i < self.global_levels and self.global_levels < len(self.levels):
                    s = crop_batch(s)
                s = to_fine(s)
                s = self._resize_cl(s, fine_res, mode="nearest")
                fused.append(_to_cf(s))
            if self.fusion_type == "concat":
                fused_cf = self.fuse(torch.cat(fused, dim=1))
            else:
                fused_cf = torch.stack(fused, dim=0).sum(dim=0)
            logits = self.seg_head(fused_cf)
        return logits


def crop_target_to_roi(targets_cl: torch.Tensor, boxes: List[RoiBox],
                       out_size: int) -> torch.Tensor:
    r"""Crop a channels-last target with the model's OWN ROI boxes.

    This is the label-alignment contract: the caller passes the boxes returned by
    ``model.last_roi_box()``, so the target is never cropped independently of the
    image. Nearest-neighbour resizing keeps WT/TC/ET strictly binary and
    preserves ET subset TC subset WT.

    #Args
        targets_cl: (B, X, Y, Z, R) ground truth, channels-last.
        boxes: per-sample normalised boxes from ``model.last_roi_box()``.
        out_size: spatial size of the model's prediction (cube).
    #Returns
        (B, out_size, out_size, out_size, R)
    """
    if not boxes:                       # roi_fraction == 1 -> no crop happened
        if targets_cl.shape[1] == out_size:
            return targets_cl
        return GLO_NCA_GlobalContext._resize_cl(targets_cl, out_size, mode="nearest")
    if len(boxes) != targets_cl.shape[0]:
        raise ValueError(f"got {len(boxes)} ROI boxes for batch {targets_cl.shape[0]}")
    parts = []
    for i, box in enumerate(boxes):
        roi = GLO_NCA_GlobalContext.crop_normalised(targets_cl[i:i + 1], box)
        parts.append(GLO_NCA_GlobalContext._resize_cl(roi, out_size, mode="nearest"))
    return torch.cat(parts, dim=0)


def build_glo_nca_global_context(cfg, input_channels=4, output_channels=3,
                                 device=None, force_full_volume: bool = False):
    """Build a GLO_NCA_GlobalContext from a loaded Config.

    Reads the same ``model:`` block as the reference builder, plus::

        model:
          global_context:
            roi_fraction: 1.0     # 1.0 == reference behaviour
            global_levels: 1

    ``force_full_volume=True`` pins ``roi_fraction`` to 1.0 regardless of the
    config. Evaluation MUST use it so validation and test always score complete
    volumes and the training ROI can never leak into reported metrics.
    """
    m = cfg.raw.get("model", {}) or {}

    def lvl(key, default_res):
        s = m.get(key, {}) or {}
        return LevelSpec(enabled=bool(s.get("enabled", True)),
                         resolution=int(s.get("resolution", default_res)),
                         channels=int(s.get("channels", 24)),
                         nca_steps=int(s.get("nca_steps", 10)),
                         kernel_size=int(s.get("kernel_size", 3)))

    gc_cfg = m.get("global_context", {}) or {}
    roi = 1.0 if force_full_volume else float(gc_cfg.get("roi_fraction", 1.0))
    return GLO_NCA_GlobalContext(
        input_channels=input_channels, output_channels=output_channels,
        levels=[lvl("level1", 32), lvl("level2", 96), lvl("level3", 128)],
        fire_rate=float(m.get("fire_rate", 0.6)),
        use_attention=bool(m.get("use_attention", True)),
        use_spatial=bool(m.get("use_spatial", True)),
        dropout=float(m.get("dropout", 0.0)),
        fusion=str((m.get("feature_fusion", {}) or {}).get("type", "concat")),
        hidden_size=int(m.get("hidden", 128)), device=device,
        gradient_checkpointing=bool((cfg.raw.get("memory", {}) or {})
                                    .get("gradient_checkpointing", False)),
        roi_fraction=roi,
        global_levels=int(gc_cfg.get("global_levels", 1)))
