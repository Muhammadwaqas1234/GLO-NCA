"""GLO-NCA cell rule: the per-voxel update applied at every NCA step.

Formerly Model_BasicNCA3D.py / BasicNCA3D (renamed only; behaviour and parameters unchanged).

  ChannelsLastBatchNorm  channels-last BatchNorm wrapper
  SEBlock3D              squeeze-and-excitation channel attention
  GCSpatialBlock3D       spatial global context
  GLO_NCA_Cell           the NCA update rule composing the above

Two-level GLO-NCA uses one cell per level (L1 48³, L2 64³).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # optional gradient checkpointing (memory only)


class ChannelsLastBatchNorm(nn.Module):
    r"""BatchNorm over the channel dimension of a channels-last tensor.

    Same function as BatchNorm3d(hidden, track_running_stats=False): batch statistics in
    train and eval, via F.batch_norm on the (N, C) view, avoiding two 128-channel transposes
    per NCA step. State-dict keys (...bn.weight / ...bn.bias) match BatchNorm3d, so existing
    checkpoints load unchanged.
    """

    def __init__(self, num_features, eps=1e-5):
        super(ChannelsLastBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Normalise per channel over (B, X, Y, Z).
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = F.batch_norm(flat, None, None, self.weight, self.bias,
                           True, 0.0, self.eps)
        return out.reshape(shape)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Keys normally match; remap a nested <prefix>bn.* layout and drop (and report)
        # running-stat buffers, which this module never uses.
        for suffix in ("weight", "bias"):
            legacy = prefix + "bn." + suffix
            target = prefix + suffix
            if legacy in state_dict and target not in state_dict:
                state_dict[target] = state_dict.pop(legacy)
        dropped = [k for k in list(state_dict.keys())
                   if k.startswith(prefix) and k[len(prefix):] in
                   ("running_mean", "running_var", "num_batches_tracked")]
        for k in dropped:
            state_dict.pop(k)
        super(ChannelsLastBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)
        if dropped:
            error_msgs.append(
                f"ChannelsLastBatchNorm at '{prefix}': ignored running-stat "
                f"buffers {dropped}; this module uses batch statistics only "
                f"(equivalent to track_running_stats=False).")


class SEBlock3D(nn.Module):
    r"""Squeeze-and-excitation channel attention: global pool -> bottleneck MLP -> per-channel gate.

    Gives every cell whole-volume context for about 2*C^2/r extra parameters.
    """
    def __init__(self, channel_n, reduction=4):
        super(SEBlock3D, self).__init__()
        reduced = max(1, channel_n // reduction)
        self.fc1 = nn.Linear(channel_n, reduced)
        self.fc2 = nn.Linear(reduced, channel_n)

    def forward(self, x):
        r"""x: channels-first (B, C, D, H, W)."""
        b, c = x.shape[0], x.shape[1]
        # Squeeze: global average pool -> (B, C).
        squeezed = x.mean(dim=(2, 3, 4))
        # Excite: bottleneck MLP -> gate in [0, 1].
        gate = F.relu(self.fc1(squeezed))
        gate = torch.sigmoid(self.fc2(gate))
        # Scale: broadcast the gate over space.
        gate = gate.view(b, c, 1, 1, 1)
        return x * gate


class GCSpatialBlock3D(nn.Module):
    r"""Spatial global context: channel mean + max -> 3D conv -> per-voxel gate in [0, 1].

    Complements SE (which channels matter) with where in the volume matters.
    """
    def __init__(self, kernel_size=7):
        super(GCSpatialBlock3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        r"""x: channels-first (B, C, D, H, W)."""
        avg_map = x.mean(dim=1, keepdim=True)               # (B,1,D,H,W)
        max_map = x.max(dim=1, keepdim=True)[0]             # (B,1,D,H,W)
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class GLO_NCA_Cell(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False, use_attention=False, se_reduction=4, use_spatial=False, dropout=0.0, spatial_kernel_size=7):
        r"""#Args:
            channel_n: channels per cell
            fire_rate: probability a cell updates
            device: torch device
            hidden_size: hidden width of the update MLP
            input_channels: input modalities (4 for BraTS)
            init_method: weight initialisation function
            kernel_size: perception conv kernel size
            groups: whether input channels are interconnected
            use_attention: enable the SE block
            se_reduction: SE bottleneck reduction ratio
        """
        super(GLO_NCA_Cell, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.use_attention = use_attention
        # Optional gradient checkpointing of the NCA steps (memory only; RNG preserved,
        # so the fire-rate mask and outputs are identical). Default off.
        self.use_checkpoint = False

        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        # Channels-last equivalent of BatchNorm3d(hidden_size, track_running_stats=False).
        self.bn = ChannelsLastBatchNorm(hidden_size)
        # Dropout on the hidden update (0.0 = off).
        self.dropout_p = dropout
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None

        # Global-context blocks, built only when enabled: se (channel), gc (spatial).
        self.use_spatial = use_spatial
        self.se = SEBlock3D(channel_n, reduction=se_reduction) if use_attention else None
        # Spatial global-context kernel from config (default and production 7).
        self.spatial_kernel_size = spatial_kernel_size
        self.gc = (GCSpatialBlock3D(kernel_size=spatial_kernel_size)
                   if use_spatial else None)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform_(self.fc0.weight)
            torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        r"""Perception: learned conv features plus the cell identity, gated by SE / spatial GC when enabled.

        x: channels-first (B, C, D, H, W).
        """
        y1 = self.p0(x)
        if self.se is not None:
            y1 = self.se(y1)  # channel attention (SE)
        if self.gc is not None:
            y1 = self.gc(y1)  # spatial global context
        y = torch.cat((x,y1),1)
        return y

    def update(self, x_in, fire_rate):
        r"""One NCA update on every cell, with stochastic fire-rate masking.

        x_in: state; fire_rate: probability a cell updates.
        """
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        # self.bn works channels-last, so no transpose is needed.
        dx = self.bn(dx)
        dx = F.relu(dx)
        if self.drop is not None:
            dx = self.drop(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1], device=dx.device)>fire_rate
        stochastic = stochastic.float()
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x

    def forward(self, x, steps=10, fire_rate=0.5):
        r"""Run the update ``steps`` times, leaving the input channels unchanged.

        x: state; steps: NCA steps; fire_rate: probability a cell updates.
        """
        # Checkpoint whenever gradients are enabled; off under no_grad (memory only).
        use_ckpt = getattr(self, "use_checkpoint", False) and torch.is_grad_enabled()
        for step in range(steps):
            if use_ckpt:
                # Recompute in backward; preserve_rng_state keeps the fire-rate mask identical.
                x2 = torch.utils.checkpoint.checkpoint(
                    self.update, x, fire_rate,
                    use_reentrant=False, preserve_rng_state=True).clone()
            else:
                x2 = self.update(x, fire_rate).clone()
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
        return x
