import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    r"""Squeeze-and-Excitation block providing lightweight GLOBAL context.

    This is the global-context mechanism for the thesis. A plain NCA only
    perceives a local neighbourhood (the 3D conv kernel). The SE block adds a
    cheap global signal:

        1. Squeeze  - global average pool over the *whole* volume -> one value
                      per channel. This summarises the entire image, so the
                      information is global rather than local.
        2. Excite   - a tiny 2-layer bottleneck MLP turns that global summary
                      into a per-channel gate in [0, 1].
        3. Scale    - the perceived features are re-weighted by the gate, so
                      every cell's update is modulated by whole-volume context.

    Cost: only ~2 * C^2 / r extra parameters (a few hundred) and a single
    global pool, so it is well within a 6 GB RTX 3060 budget and barely affects
    training speed - exactly the "global context, low params, fast" trade-off
    requested for this work.
    """
    def __init__(self, channel_n, reduction=4):
        super(SEBlock3D, self).__init__()
        reduced = max(1, channel_n // reduction)
        self.fc1 = nn.Linear(channel_n, reduced)
        self.fc2 = nn.Linear(reduced, channel_n)

    def forward(self, x):
        r"""#Args: x in channels-first layout (B, C, D, H, W)."""
        b, c = x.shape[0], x.shape[1]
        # Squeeze: global average pool over all spatial dims -> (B, C)
        squeezed = x.mean(dim=(2, 3, 4))
        # Excite: bottleneck MLP -> per-channel gate in [0, 1]
        gate = F.relu(self.fc1(squeezed))
        gate = torch.sigmoid(self.fc2(gate))
        # Scale: broadcast gate back over the spatial dims
        gate = gate.view(b, c, 1, 1, 1)
        return x * gate


class BasicNCA3D(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False, use_attention=False, se_reduction=4):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels (e.g. 4 for BraTS
                    T1/T1ce/T2/FLAIR multi-modal input)
                init_method: Weight initialisation function
                kernel_size: defines kernel input size
                groups: if channels in input should be interconnected
                use_attention: enable the global-context SE block (thesis novelty).
                    Leave False for the plain-NCA baseline, True for the
                    global-context-aware model.
                se_reduction: bottleneck reduction ratio for the SE block
        """
        super(BasicNCA3D, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.use_attention = use_attention

        # One Input
        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        self.bn = torch.nn.BatchNorm3d(hidden_size, track_running_stats=False)

        # Global-context block (thesis novelty). Only built when enabled so the
        # baseline keeps the original parameter count exactly.
        self.se = SEBlock3D(channel_n, reduction=se_reduction) if use_attention else None

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform_(self.fc0.weight)
            torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell.

        When attention is enabled the locally-perceived features are modulated
        by the global SE gate, so each update sees both local neighbourhood
        (the conv) and whole-volume context (the SE block).
            #Args:
                x: image in channels-first layout (B, C, D, H, W)
        """
        y1 = self.p0(x)
        if self.se is not None:
            y1 = self.se(y1)
        y = torch.cat((x,y1),1)
        return y

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
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
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
        return x
