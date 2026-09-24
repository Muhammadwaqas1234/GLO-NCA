r"""Two-level GLO-NCA with global context and learned fusion (production model).

Subclass of ``GLO_NCA_V3_MultiLevel``; only ``forward`` is overridden.

    128³ working volume
      -> L1 (48³): GLO-NCA + SE + spatial global context, full volume
      -> L2 (64³): GLO-NCA + SE + spatial global context
      -> learned projections + Conv3d(48 -> 24) fusion
      -> WT / TC / ET

L1 always sees the whole working volume. ``roi_fraction < 1`` optionally
limits L2 to a region of interest during training only (production uses 1.0,
identical to the full-volume forward). ROI boxes are normalised, so image and
target are cropped with the same coordinates.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.models.Model_GLO_NCA_V3 import (GLO_NCA_V3_MultiLevel, LevelSpec,
                                         _to_cf, _to_cl)

# Normalised ROI box: ((x0,x1), (y0,y1), (z0,z1)) as fractions of extent.
RoiBox = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


class GLO_NCA_GlobalContext(GLO_NCA_V3_MultiLevel):
    r"""Two-level GLO-NCA with a full-volume global-context level (L1).

    #Args (in addition to the base class):
        roi_fraction: side fraction L2 covers, in (0, 1]; 1.0 is full volume.
        global_levels: leading levels kept full-volume (default 1).
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
        # Boxes from the last forward, for cropping the target identically.
        self._last_roi: List[RoiBox] = []

    # ROI.
    def set_roi_origin(self, origin: Optional[Tuple[float, float, float]]) -> None:
        """Pin the ROI origin (normalised), or None for random in training / centred in eval."""
        self._roi_origin = origin

    def last_roi_box(self) -> List[RoiBox]:
        """Normalised ROI boxes from the last forward (empty when ``roi_fraction == 1``)."""
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
        """Crop a channels-last volume to a normalised box; size comes from box width so batches stack."""
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

    # Forward.
    def forward(self, modalities_cl: torch.Tensor) -> torch.Tensor:
        from src.profiling import get_profiler
        prof = get_profiler()

        # Evaluation is always full-volume: the ROI applies in training mode only.
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
            """Crop each sample with its own box; all boxes share one size."""
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
                    # Global context (L1): the whole working volume.
                    mod_i = self._resize_cl(modalities_cl, res, mode="trilinear")
                else:
                    # L2: crop to the ROI first, then resize.
                    mod_i = self._resize_cl(crop_batch(modalities_cl),
                                            max(1, int(round(res * frac))),
                                            mode="trilinear")
                seed = self._seed(mod_i, lv.channels)

                if prev_state is not None:
                    # Carry the previous level's state forward, cropped to the same ROI.
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

        # Learned multi-level fusion.
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
            if self.deep_supervision and self.training and self.aux_heads:
                aux = []
                for i, head in enumerate(self.aux_heads):
                    s_i = level_states[i]
                    if i < self.global_levels and self.global_levels < len(self.levels):
                        s_i = crop_batch(s_i)
                    s_i = self.level_to_fine[i](s_i)
                    s_i = self._resize_cl(s_i, fine_res, mode="nearest")
                    aux.append(head(_to_cf(s_i)))
                return logits, aux
        return logits


def crop_target_to_roi(targets_cl: torch.Tensor, boxes: List[RoiBox],
                       out_size: int) -> torch.Tensor:
    r"""Crop a channels-last target with the model's own ROI boxes (nearest-neighbour, stays binary).

    #Args
        targets_cl: (B, X, Y, Z, R) ground truth.
        boxes: boxes from ``model.last_roi_box()``.
        out_size: prediction cube size.
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
    """Build a GLO_NCA_GlobalContext from a Config (``model.global_context``).

    ``force_full_volume=True`` pins ``roi_fraction`` to 1.0 so evaluation always
    scores complete volumes.
    """
    m = cfg.raw.get("model", {}) or {}

    # An unstated level3 is disabled: it can never switch on by omission.
    def lvl(key, default_res, default_enabled=True):
        s = m.get(key, {}) or {}
        return LevelSpec(enabled=bool(s.get("enabled", default_enabled)),
                         resolution=int(s.get("resolution", default_res)),
                         channels=int(s.get("channels", 24)),
                         nca_steps=int(s.get("nca_steps", 10)),
                         kernel_size=int(s.get("kernel_size", 3)))

    gc_cfg = m.get("global_context", {}) or {}
    roi = 1.0 if force_full_volume else float(gc_cfg.get("roi_fraction", 1.0))
    return GLO_NCA_GlobalContext(
        input_channels=input_channels, output_channels=output_channels,
        levels=[lvl("level1", 48), lvl("level2", 64), lvl("level3", 128, default_enabled=False)],
        fire_rate=float(m.get("fire_rate", 0.6)),
        use_attention=bool(m.get("use_attention", True)),
        use_spatial=bool(m.get("use_spatial", True)),
        dropout=float(m.get("dropout", 0.0)),
        fusion=str((m.get("feature_fusion", {}) or {}).get("type", "concat")),
        hidden_size=int(m.get("hidden", 128)), device=device,
        gradient_checkpointing=bool((cfg.raw.get("memory", {}) or {})
                                    .get("gradient_checkpointing", False)),
        # Spatial global-context kernel from config (production 7).
        spatial_kernel_size=int(m.get("spatial_kernel_size", 7)),
        deep_supervision=bool((m.get("deep_supervision", {}) or {})
                              .get("enabled", False)),
        roi_fraction=roi,
        global_levels=int(gc_cfg.get("global_levels", 1)))
