r"""
================================================================================
GLO-NCA V3 -- Multi-Level GLO-NCA (NEW architecture; V2 is untouched baseline).
================================================================================
A single UNIFIED model with three GLO-NCA levels connected by LEARNABLE feature
fusion, in strict information flow:

    Level 1 (global, low-res)  ->  Level 2 (high-res)  ->  optional Level 3
    ->  learnable multi-level fusion  ->  WT/TC/ET logits.

    GLO-NCA production runs two levels: L1 48^3 and L2 64^3, from a 128^3
    working volume, with level 3 disabled.

Design principles (all verified against the V2 code):
  * Each level is the EXISTING ``BasicNCA3D`` (SE + spatial GC, channels-last
    (B,X,Y,Z,C) convention), reused unchanged -- so the per-cell NCA rule,
    SE block and spatial-GC block are identical to V2. V3 adds only the
    cross-level projection + fusion, which is the new contribution.
  * Higher levels are NESTED, higher-resolution views of the SAME region: each
    level's learned state is upsampled and injected into the next level's seed,
    exactly like the V2 coarse-to-fine cascade (permute->Upsample->permute).
  * Output is 3-channel multi-label sigmoid (WT/TC/ET), same as V2 -- NOT
    softmax.
  * NOT an ensemble: there is one forward path and one prediction head; levels
    feed forward, they are not averaged.

Memory note: 3D activation memory (not parameters) dominates at 96^3/128^3.
Level channel widths and NCA steps are configurable so the high-res level can
use a narrower width without changing the architecture's identity.
================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.Model_BasicNCA3D import BasicNCA3D


@dataclass
class LevelSpec:
    """Configuration for one GLO-NCA level."""
    enabled: bool
    resolution: int          # cubic side length (e.g. 32, 96, 128)
    channels: int            # NCA state channels at this level
    nca_steps: int           # NCA update steps at this level
    kernel_size: int = 3


def _to_cf(x: torch.Tensor) -> torch.Tensor:
    """channels-last (B,X,Y,Z,C) -> channels-first (B,C,X,Y,Z)."""
    return x.permute(0, 4, 1, 2, 3).contiguous()


def _to_cl(x: torch.Tensor) -> torch.Tensor:
    """channels-first (B,C,X,Y,Z) -> channels-last (B,X,Y,Z,C)."""
    return x.permute(0, 2, 3, 4, 1).contiguous()


class FeatureProjection(nn.Module):
    """Learnable 1x1x1 conv projection between level channel widths, operating on
    channels-last tensors (projects the state channels only)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x_cl: torch.Tensor) -> torch.Tensor:
        return _to_cl(self.proj(_to_cf(x_cl)))


class GLO_NCA_V3_MultiLevel(nn.Module):
    r"""Unified multi-level GLO-NCA.

    #Args
        input_channels: number of MRI modalities (4 for BraTS).
        output_channels: number of output regions (3: WT/TC/ET).
        levels: list of LevelSpec (2 or 3 levels; level i feeds level i+1).
        fire_rate: NCA stochastic fire rate (V2 default 0.6).
        use_attention / use_spatial: enable SE / spatial-GC blocks in each level.
        dropout: NCA MLP dropout.
        fusion: 'concat' (learnable concat+conv, default) or 'add'.
        device: torch device.
    """

    def __init__(self, input_channels: int, output_channels: int,
                 levels: List[LevelSpec], fire_rate: float = 0.6,
                 use_attention: bool = True, use_spatial: bool = True,
                 dropout: float = 0.0, fusion: str = "concat",
                 hidden_size: int = 128, device=None,
                 gradient_checkpointing: bool = False,
                 spatial_kernel_size: int = 7,
                 deep_supervision: bool = False):
        super().__init__()
        assert len(levels) >= 2, "V3 needs at least 2 levels"
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.levels = [lv for lv in levels if lv.enabled]
        self.fire_rate = fire_rate
        self.fusion_type = fusion
        self.device = device or torch.device("cpu")
        # Memory-only optimization (default OFF). When True, each level's NCA
        # unroll uses gradient checkpointing: identical layers/steps/outputs, only
        # activation memory is traded for recompute. Architecture is unchanged.
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # One BasicNCA3D per active level. Each sees `input_channels` modalities
        # placed into the first channels of its state (V2 seed convention).
        self.ncas = nn.ModuleList([
            BasicNCA3D(channel_n=lv.channels, fire_rate=fire_rate,
                       device=self.device, hidden_size=hidden_size,
                       input_channels=input_channels, kernel_size=lv.kernel_size,
                       use_attention=use_attention, use_spatial=use_spatial,
                       dropout=dropout,
                       spatial_kernel_size=spatial_kernel_size)
            for lv in self.levels
        ])
        # Propagate the memory-only flag to each level's NCA (default False keeps
        # the exact original behaviour).
        for nca in self.ncas:
            nca.use_checkpoint = self.gradient_checkpointing

        # Learnable projections that carry the previous level's FULL state (input
        # modalities + learned channels) into the next level's state-channel
        # width, so cross-level information is fused, not just concatenated raw.
        # Output width = next level's state channels (everything after the input
        # modalities), because it is added into the next seed's state region.
        self.projections = nn.ModuleList()
        for i in range(len(self.levels) - 1):
            prev_full = self.levels[i].channels
            nxt_state = self.levels[i + 1].channels - input_channels
            self.projections.append(FeatureProjection(prev_full, nxt_state))

        # Learnable multi-level fusion: gather each level's final state (projected
        # to the finest level's width and resolution), concatenate, and fuse with
        # a small conv -> then the segmentation head. This is the V3 contribution.
        fine_ch = self.levels[-1].channels
        n_levels = len(self.levels)
        if fusion == "concat":
            self.level_to_fine = nn.ModuleList([
                FeatureProjection(lv.channels, fine_ch) for lv in self.levels
            ])
            self.fuse = nn.Conv3d(fine_ch * n_levels, fine_ch, kernel_size=1)
        else:  # 'add'
            self.level_to_fine = nn.ModuleList([
                FeatureProjection(lv.channels, fine_ch) for lv in self.levels
            ])
            self.fuse = nn.Identity()
        self.seg_head = nn.Conv3d(fine_ch, output_channels, kernel_size=1)

        # Deep supervision: one 1x1 auxiliary head per non-final level, used
        # during TRAINING only. forward() returns them alongside the primary
        # logits while self.training is set; eval() returns the primary logits
        # alone, so every reported metric comes from seg_head.
        self.deep_supervision = bool(deep_supervision)
        self.aux_heads = (nn.ModuleList([
            nn.Conv3d(fine_ch, output_channels, kernel_size=1)
            for _ in range(n_levels - 1)]) if self.deep_supervision else None)

        self.to(self.device)

    # ------------------------------------------------------------------ helpers
    def _seed(self, modalities_cl: torch.Tensor, channels: int) -> torch.Tensor:
        """Build a channels-last NCA seed: modalities in the first channels, the
        rest zero learnable state. modalities_cl: (B,X,Y,Z,input_channels)."""
        b, x, y, z, c = modalities_cl.shape
        seed = torch.zeros((b, x, y, z, channels), dtype=modalities_cl.dtype,
                           device=modalities_cl.device)
        seed[..., :c] = modalities_cl
        return seed

    @staticmethod
    def _resize_cl(x_cl: torch.Tensor, size: int, mode: str) -> torch.Tensor:
        """Resize a channels-last volume to (size,size,size)."""
        xcf = _to_cf(x_cl)
        xcf = F.interpolate(xcf, size=(size, size, size), mode=mode,
                            align_corners=False if mode == "trilinear" else None)
        return _to_cl(xcf)

    # ---------------------------------------------------------------- forward
    def forward(self, modalities_cl: torch.Tensor) -> torch.Tensor:
        r"""#Args
                modalities_cl: (B, X, Y, Z, input_channels) at the FINEST
                resolution (the level-3 size). Lower levels are produced by
                downsampling, matching the nested multi-scale design.
            #Returns
                logits: (B, output_channels, X, Y, Z) channels-first (matches the
                repository's [B,3,D,H,W] convention for the loss/metrics).
        """
        # Phase 2 profiling: OBSERVATIONAL ONLY. `prof` is a NullProfiler unless
        # profiling is explicitly enabled, in which case `section()` returns a
        # shared no-op context -- no CUDA sync, no behaviour change. The maths,
        # ordering, resolutions and NCA step counts below are untouched.
        from src.profiling import get_profiler
        prof = get_profiler()

        prev_state: Optional[torch.Tensor] = None
        level_states: List[torch.Tensor] = []

        for i, (lv, nca) in enumerate(zip(self.levels, self.ncas)):
            with prof.section(f"model/level{i + 1}", cuda=True):
                res = lv.resolution
                # modalities at this level's resolution (down/upsampled view)
                mod_i = self._resize_cl(modalities_cl, res, mode="trilinear")
                seed = self._seed(mod_i, lv.channels)

                # inject the previous level's (projected, upsampled) state into this
                # level's state channels (everything after the input modalities).
                if prev_state is not None:
                    proj = self.projections[i - 1](prev_state)         # -> nxt_state width
                    proj = self._resize_cl(proj, res, mode="nearest")  # -> this resolution
                    seed = seed.clone()
                    seed[..., self.input_channels:] = seed[..., self.input_channels:] + proj

                out = nca(seed, steps=lv.nca_steps, fire_rate=self.fire_rate)
                prev_state = out
                level_states.append(out)

        # ---- learnable multi-level fusion at the finest resolution ----
        with prof.section("model/fusion", cuda=True):
            fine_res = self.levels[-1].resolution
            fused = []
            for state, to_fine in zip(level_states, self.level_to_fine):
                s = to_fine(state)                                   # width -> fine_ch
                s = self._resize_cl(s, fine_res, mode="nearest")     # res -> fine
                fused.append(_to_cf(s))
            if self.fusion_type == "concat":
                fused_cf = self.fuse(torch.cat(fused, dim=1))
            else:
                fused_cf = torch.stack(fused, dim=0).sum(dim=0)
            logits = self.seg_head(fused_cf)                         # (B,3,X,Y,Z)
            if self.deep_supervision and self.training and self.aux_heads:
                aux = []
                for i, head in enumerate(self.aux_heads):
                    s_i = self.level_to_fine[i](level_states[i])
                    s_i = self._resize_cl(s_i, fine_res, mode="nearest")
                    aux.append(head(_to_cf(s_i)))
                return logits, aux
        return logits

    # ------------------------------------------------------------ introspection
    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        by_level = {f"level{i+1}": sum(p.numel() for p in nca.parameters())
                    for i, nca in enumerate(self.ncas)}
        by_level["projections"] = sum(p.numel() for m in self.projections for p in m.parameters())
        by_level["fusion+head"] = (sum(p.numel() for m in self.level_to_fine for p in m.parameters())
                                   + sum(p.numel() for p in self.fuse.parameters())
                                   + sum(p.numel() for p in self.seg_head.parameters()))
        aux = (sum(p.numel() for p in self.aux_heads.parameters())
               if self.aux_heads is not None else 0)
        by_level["deep_supervision_aux"] = aux
        return {"total_parameters": total, "trainable_parameters": trainable,
                "non_trainable_parameters": total - trainable,
                "auxiliary_parameters": aux,
                "inference_parameters": total - aux, "by_level": by_level}


def build_v3_from_config(cfg, input_channels=4, output_channels=3, device=None):
    """Construct a V3 model from a loaded Config (configs/v3_*.yaml)."""
    def lvl(key, default_res):
        s = cfg.raw.get("model", {}).get(key, {}) or {}
        return LevelSpec(
            enabled=bool(s.get("enabled", True)),
            resolution=int(s.get("resolution", default_res)),
            channels=int(s.get("channels", 24)),
            nca_steps=int(s.get("nca_steps", 10)),
            kernel_size=int(s.get("kernel_size", 3)),
        )
    m = cfg.raw.get("model", {})
    # Fallbacks match the GLO-NCA production geometry. The production config
    # states every value explicitly, so these apply only to a config that
    # omits a level.
    levels = [lvl("level1", 48), lvl("level2", 64), lvl("level3", 128)]
    # Memory-only opt-in flag; default OFF. Read from `memory.gradient_checkpointing`
    # (top-level) so the production config stays unchanged unless it opts in.
    gc = bool((cfg.raw.get("memory", {}) or {}).get("gradient_checkpointing", False))
    return GLO_NCA_V3_MultiLevel(
        input_channels=input_channels, output_channels=output_channels,
        levels=levels, fire_rate=float(m.get("fire_rate", 0.6)),
        use_attention=bool(m.get("use_attention", True)),
        use_spatial=bool(m.get("use_spatial", True)),
        dropout=float(m.get("dropout", 0.0)),
        fusion=str((m.get("feature_fusion", {}) or {}).get("type", "concat")),
        hidden_size=int(m.get("hidden", 128)), device=device,
        gradient_checkpointing=gc,
        # Receptive field of the spatial global-context block. Default 7
        # reproduces the original hardcoded value for any config that does
        # not state it.
        spatial_kernel_size=int(m.get("spatial_kernel_size", 7)),
        deep_supervision=bool(
            (m.get("deep_supervision", {}) or {}).get("enabled", False)))
