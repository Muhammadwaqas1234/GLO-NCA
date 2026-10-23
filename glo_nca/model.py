r"""GLO-NCA cell: learned depthwise perception, SE, spatial gate, optional global block."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SEBlock3D(nn.Module):
    r"""Squeeze-and-excitation block: per-channel gates from global average pooling over the volume."""
    def __init__(self, channel_n, reduction=4):
        r"""Build the block's layers."""
        super(SEBlock3D, self).__init__()
        reduced = max(1, channel_n // reduction)
        self.fc_reduce = nn.Linear(channel_n, reduced)
        self.fc_expand = nn.Linear(reduced, channel_n)

    def forward(self, x):
        r"""Rescale each channel of (B, C, X, Y, Z) features by its learned gate."""
        b, c = x.shape[0], x.shape[1]
        # Squeeze: global average pool over all spatial dimensions to (B, C).
        squeezed = x.mean(dim=(2, 3, 4))
        # Excite: bottleneck MLP to a per-channel gate in [0, 1].
        gate = F.relu(self.fc_reduce(squeezed))
        gate = torch.sigmoid(self.fc_expand(gate))
        # Scale: broadcast the gate back over the spatial dimensions.
        gate = gate.view(b, c, 1, 1, 1)
        return x * gate


class SpatialGate3D(nn.Module):
    r"""Spatial gate (CBAM-style): per-voxel attention from channel mean and max via a local convolution."""
    def __init__(self, kernel_size=7):
        r"""Build the block's layers."""
        super(SpatialGate3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        r"""Gate every voxel of (B, C, X, Y, Z) features by a sigmoid map from channel mean and maximum."""
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class GlobalContextBlock3D(nn.Module):
    r"""Global-context block (GCNet form): attention pooling over the whole volume to one context vector.

    g = sum_i softmax(w^T x)_i x_i, transformed by a bottleneck to ``out_channels``. The cell adds the
    result to every voxel of its hidden layer after normalisation; added before normalisation, a
    constant shift would be removed by the per-volume centring. The last layer starts at zero, so the
    block starts with no effect.
    """
    def __init__(self, channel_n, out_channels=None, reduction=4):
        r"""Build the attention key and the bottleneck transform."""
        super(GlobalContextBlock3D, self).__init__()
        out_channels = out_channels or channel_n
        # At least 4 channels: LayerNorm over fewer values collapses the context to near-constant signs.
        reduced = max(4, channel_n // reduction)
        self.key = nn.Conv3d(channel_n, 1, kernel_size=1)
        self.transform = nn.Sequential(nn.Linear(channel_n, reduced), nn.LayerNorm(reduced),
                                       nn.ReLU(inplace=True), nn.Linear(reduced, out_channels))
        with torch.no_grad():
            self.transform[-1].weight.zero_(); self.transform[-1].bias.zero_()

    def context(self, x):
        r"""Context vector per volume, (B, out_channels), from channels-first features (B, C, X, Y, Z)."""
        b, c = x.shape[0], x.shape[1]
        attn = torch.softmax(self.key(x).view(b, 1, -1), dim=-1)
        return self.transform((x.view(b, c, -1) * attn).sum(dim=-1))

    def forward(self, x):
        r"""Residual form for standalone use: x plus the context broadcast over the volume."""
        return x + self.context(x).view(x.shape[0], -1, 1, 1, 1)


# Layer names before the renaming; checkpoints written with them are translated on load.
LEGACY_LAYER_NAMES = {"se.fc1.": "se.fc_reduce.", "se.fc2.": "se.fc_expand.", "p0.": "perceive_conv.", "fc0.": "fc_in.", "fc1.": "fc_out.", "bn.": "batch_norm.",
                      "ln.": "layer_norm.", "gc.": "spatial_gate.", "glob.": "global_context."}


class GLONCACell(nn.Module):
    r"""GLO-NCA cell: perception, SE and global-context blocks, and a stochastic update."""
    def __init__(self, channel_n, skip_prob, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False, use_se=False, se_reduction=4, use_spatial=False, dropout=0.0, use_global=False, norm="batch"):
        r"""GLO-NCA cell: perception, SE and spatial global context, and a residual update MLP."""
        super(GLONCACell, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.use_se = use_se

        self.fc_in = nn.Linear(channel_n*2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.perceive_conv = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        # "batch": BatchNorm over the volume (spatial statistics); "layer": per-voxel, strictly local.
        if norm not in ("batch", "layer"):
            raise ValueError(f"unknown norm: {norm}")
        self.batch_norm = torch.nn.BatchNorm3d(hidden_size, track_running_stats=False) if norm == "batch" else None
        self.layer_norm = nn.LayerNorm(hidden_size) if norm == "layer" else None
        # Light dropout on the hidden update (regularisation; 0.0 = off).
        self.dropout_p = dropout
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None

        # Global-context blocks: SE (channel) and spatial.
        self.use_spatial = use_spatial
        self.se = SEBlock3D(channel_n, reduction=se_reduction) if use_se else None
        self.spatial_gate = SpatialGate3D(kernel_size=7) if use_spatial else None
        self.global_context = GlobalContextBlock3D(channel_n, out_channels=hidden_size, reduction=se_reduction) \
            if use_global else None

        with torch.no_grad():
            self.fc_out.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform_(self.fc_in.weight)
            torch.nn.init.xavier_uniform_(self.fc_out.weight)

        self.skip_prob = skip_prob
        self.grad_checkpoint = False   # recompute each step in backward: less memory, same result
        self.to(self.device)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        r"""Load a state dict, translating layer names of checkpoints written before the renaming."""
        for key in [k for k in state_dict if k.startswith(prefix)]:
            rest = key[len(prefix):]
            for old, new in LEGACY_LAYER_NAMES.items():
                if rest.startswith(old):
                    state_dict[prefix + new + rest[len(old):]] = state_dict.pop(key)
                    break
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def perceive(self, x):
        r"""Perception: learned convolutions plus the cell identity, gated by SE and the spatial gate."""
        y1 = self.perceive_conv(x)
        if self.se is not None:
            y1 = self.se(y1)          # channel recalibration (SE): global pooling over the volume per channel
        if self.spatial_gate is not None:
            y1 = self.spatial_gate(y1)          # spatial gate: local 7x7x7 convolution over channel mean and max
        self._perceived = y1          # input of the global-context block
        y = torch.cat((x,y1),1)
        return y

    def update(self, x_in, skip_prob):
        r"""One update of every cell, with stochastic fire-rate masking."""
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = dx.transpose(1,4)
        dx = self.fc_in(dx)
        if self.batch_norm is not None:
            dx = dx.transpose(1,4)
            dx = self.batch_norm(dx)
            dx = dx.transpose(1,4)
        else:
            dx = self.layer_norm(dx)
        if self.global_context is not None:
            # Global context after normalisation, so the volume-wide centring cannot remove it.
            g = self.global_context.context(self._perceived)
            dx = dx + g.view(g.shape[0], 1, 1, 1, -1)
        dx = F.relu(dx)
        if self.drop is not None:
            dx = self.drop(dx)
        dx = self.fc_out(dx)

        if skip_prob is None:
            skip_prob=self.skip_prob
        # skip_prob is the probability that a cell does not update: it updates when rand > skip_prob.
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1], device=dx.device)>skip_prob
        stochastic = stochastic.float()
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x

    def forward(self, x, steps=10, skip_prob=None):
        r"""Run the update for the given number of steps, keeping the input channels fixed.

        skip_prob is the probability that a cell does not update; None uses the configured value.
        """
        for step in range(steps):
            if self.grad_checkpoint and torch.is_grad_enabled():
                # preserve_rng_state (default) replays the same stochastic update mask on recompute.
                x2 = checkpoint(self.update, x, skip_prob, use_reentrant=False).clone()
            else:
                x2 = self.update(x, skip_prob).clone()
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
        return x
