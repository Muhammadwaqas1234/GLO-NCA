import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    r"""Squeeze-and-excitation block: channel global context."""
    def __init__(self, channel_n, reduction=4):
        r"""Build the block's layers."""
        super(SEBlock3D, self).__init__()
        reduced = max(1, channel_n // reduction)
        self.fc1 = nn.Linear(channel_n, reduced)
        self.fc2 = nn.Linear(reduced, channel_n)

    def forward(self, x):
        r"""Helper."""
        b, c = x.shape[0], x.shape[1]
        # Squeeze: global average pool over all spatial dimensions to (B, C).
        squeezed = x.mean(dim=(2, 3, 4))
        # Excite: bottleneck MLP to a per-channel gate in [0, 1].
        gate = F.relu(self.fc1(squeezed))
        gate = torch.sigmoid(self.fc2(gate))
        # Scale: broadcast the gate back over the spatial dimensions.
        gate = gate.view(b, c, 1, 1, 1)
        return x * gate


class GCSpatialBlock3D(nn.Module):
    r"""Spatial gate (CBAM-style): per-voxel attention from channel mean and max via a local convolution."""
    def __init__(self, kernel_size=7):
        r"""Build the block's layers."""
        super(GCSpatialBlock3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        r"""Helper."""
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class GlobalContextBlock3D(nn.Module):
    r"""Global-context block (GCNet): attention-pooled context vector added back to every voxel."""
    def __init__(self, channel_n, reduction=4):
        r"""Build the block's layers; the last layer starts at zero, so the block starts as identity."""
        super(GlobalContextBlock3D, self).__init__()
        # At least 4 channels: LayerNorm over fewer values collapses the context to near-constant signs.
        reduced = max(4, channel_n // reduction)
        self.key = nn.Conv3d(channel_n, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv3d(channel_n, reduced, kernel_size=1), nn.LayerNorm([reduced, 1, 1, 1]),
            nn.ReLU(inplace=True), nn.Conv3d(reduced, channel_n, kernel_size=1))
        with torch.no_grad():
            self.transform[-1].weight.zero_(); self.transform[-1].bias.zero_()

    def forward(self, x):
        r"""Softmax attention over all voxels pools one context vector, broadcast back as a residual."""
        b, c = x.shape[0], x.shape[1]
        attn = torch.softmax(self.key(x).view(b, 1, -1), dim=-1)
        context = (x.view(b, c, -1) * attn).sum(dim=-1).view(b, c, 1, 1, 1)
        return x + self.transform(context)


class GLO_NCA_Cell(nn.Module):
    r"""GLO-NCA cell: perception, SE and global-context blocks, and a stochastic update."""
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False, use_attention=False, se_reduction=4, use_spatial=False, dropout=0.0, use_global=False, norm="batch"):
        r"""GLO-NCA cell: perception, SE and spatial global context, and a residual update MLP."""
        super(GLO_NCA_Cell, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.use_attention = use_attention

        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        # "batch": BatchNorm over the volume (spatial statistics); "layer": per-voxel, strictly local.
        if norm not in ("batch", "layer"):
            raise ValueError(f"unknown norm: {norm}")
        self.bn = torch.nn.BatchNorm3d(hidden_size, track_running_stats=False) if norm == "batch" else None
        self.ln = nn.LayerNorm(hidden_size) if norm == "layer" else None
        # Light dropout on the hidden update (regularisation; 0.0 = off).
        self.dropout_p = dropout
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None

        # Global-context blocks: SE (channel) and spatial.
        self.use_spatial = use_spatial
        self.se = SEBlock3D(channel_n, reduction=se_reduction) if use_attention else None
        self.gc = GCSpatialBlock3D(kernel_size=7) if use_spatial else None
        self.glob = GlobalContextBlock3D(channel_n, reduction=se_reduction) if use_global else None

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform_(self.fc0.weight)
            torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        r"""Perception: learned convolutions plus the cell identity, gated by the global-context blocks."""
        y1 = self.p0(x)
        if self.se is not None:
            y1 = self.se(y1)          # channel global-context (SE)
        if self.gc is not None:
            y1 = self.gc(y1)          # spatial gate: local 7x7x7 convolution over channel mean and max
        if self.glob is not None:
            y1 = self.glob(y1)        # global context: attention pooling over the whole volume
        y = torch.cat((x,y1),1)
        return y

    def update(self, x_in, fire_rate):
        r"""One update of every cell, with stochastic fire-rate masking."""
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        if self.bn is not None:
            dx = dx.transpose(1,4)
            dx = self.bn(dx)
            dx = dx.transpose(1,4)
        else:
            dx = self.ln(dx)
        dx = F.relu(dx)
        if self.drop is not None:
            dx = self.drop(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        # fire_rate is the skip probability: a cell updates when rand > fire_rate.
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1], device=dx.device)>fire_rate
        stochastic = stochastic.float()
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x

    def forward(self, x, steps=10, fire_rate=0.5):
        r"""Run the update for the given number of steps, keeping the input channels fixed."""
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
        return x
