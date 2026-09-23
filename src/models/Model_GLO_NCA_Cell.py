"""GLO-NCA cell rule -- the per-voxel update applied at every NCA step.

RENAMED. This file was ``Model_BasicNCA3D.py`` and the class was
``BasicNCA3D``; both were renamed to GLO-NCA naming by explicit decision. The
rename is PURELY COSMETIC: no behaviour, no tensor shape and no parameter
changed. Production parameter identity is unchanged at 29,337 inference /
75 auxiliary / 29,412 training, verified from the constructed model before and
after.

Audit reports written before the rename cite ``Model_BasicNCA3D.py`` and
``BasicNCA3D`` by name, sometimes with line numbers. Those citations refer to
this file under its former name; the evidence they record still stands.

WHAT LIVES HERE
  ChannelsLastBatchNorm  channels-last BatchNorm wrapper
  SEBlock3D              squeeze-and-excitation channel attention
  GCSpatialBlock3D       spatial global context -- the thesis contribution
  GLO_NCA_Cell           the NCA update rule that composes the above

GLO_NCA_Cell is the per-level cell. ``GLO_NCA_V3_MultiLevel`` stacks two of
them (L1 48^3, L2 64^3) and ``GLO_NCA_GlobalContext`` adds the global-context
path on top; that is the production model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # opt-in gradient checkpointing (memory-only)


class ChannelsLastBatchNorm(nn.Module):
    r"""BatchNorm over the channel dimension of a channels-LAST tensor.

    IMPLEMENTATION-LEVEL OPTIMISATION ONLY. This is NOT an architecture change,
    NOT a new normalisation scheme and NOT a scientific variant: it computes
    exactly the same function as the ``BatchNorm3d`` it replaces.

    WHY IT EXISTS
    -------------
    ``update()`` carries its hidden activation as ``(B, X, Y, Z, C)``
    (channels-last), but ``BatchNorm3d`` requires channels-first, so the
    previous code transposed the 128-channel tensor into NCHW and back again
    around every single normalisation::

        dx = dx.transpose(1, 4)     # 27 MB (48^3) / 64 MB (64^3) copy
        dx = self.bn(dx)
        dx = dx.transpose(1, 4)     # and again

    With 40 NCA steps per iteration those two transposes move ~3.5 GB per
    training iteration and add two autograd nodes per step (80 per iteration),
    all of which exist only to satisfy a layout requirement.

    Normalising over ``(B, X, Y, Z)`` per channel on a channels-last tensor is
    exactly a 2D batch-norm over the flattened ``(N, C)`` view, so
    ``F.batch_norm`` on a reshape computes the identical statistics with no
    copy (the reshape is a view: the tensor is already contiguous with C last).

    EQUIVALENCE TO ``BatchNorm3d(hidden, track_running_stats=False)``
    ----------------------------------------------------------------
    ``track_running_stats=False`` means ``running_mean``/``running_var`` are
    ``None``: no buffers exist, ``momentum`` is inert, and *batch* statistics
    are used in train **and** eval. ``F.batch_norm(x, None, None, w, b,
    training=True, momentum=0.0, eps)`` reproduces precisely that, including
    the biased variance BatchNorm uses. Verified in float64:

        output      max |difference|  2.665e-15
        d/dx        max |difference|  1.844e-14
        d/dweight   max |difference|  1.819e-12
        d/dbias     max |difference|  9.095e-13

    CHECKPOINT COMPATIBILITY
    ------------------------
    ``weight`` and ``bias`` are registered at this module's top level, so the
    state-dict keys are ``...bn.weight`` / ``...bn.bias`` - byte-identical to
    the keys ``BatchNorm3d`` produced here. Existing checkpoints therefore load
    unchanged, with no migration and no rewriting of files on disk.
    ``_load_from_state_dict`` below is a defensive net for the hypothetical
    nested-key layout and for legacy running-stat buffers; in the normal case
    it does nothing.
    """

    def __init__(self, num_features, eps=1e-5):
        super(ChannelsLastBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: (B, X, Y, Z, C) -> normalise per channel over (B, X, Y, Z)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = F.batch_norm(flat, None, None, self.weight, self.bias,
                           True, 0.0, self.eps)
        return out.reshape(shape)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Compatibility net. The production BatchNorm3d stored its parameters
        # at exactly `<prefix>weight` / `<prefix>bias`, so nothing is remapped
        # in the normal case. These two rules cover the alternatives without
        # ever silencing an unrelated key mismatch:
        #   1. a nested `<prefix>bn.weight` layout, remapped to `<prefix>weight`
        #   2. running stats from a BatchNorm that tracked them; this module
        #      uses batch statistics only (matching track_running_stats=False),
        #      so such buffers are dropped DELIBERATELY and reported below.
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


class GCSpatialBlock3D(nn.Module):
    r"""Spatial global-context block (attention-pooled).

    Complements the SE block. SE gives *channel* attention (which modality /
    feature matters); this block gives *spatial* attention (which regions of the
    volume matter), so each cell's update is modulated by where the tumour is.

        1. Pool     - average + max over the channel axis -> two (B,1,D,H,W) maps
                      summarising activity at every voxel.
        2. Attend   - a small 3D conv over the concatenated maps -> a per-voxel
                      attention weight in [0, 1] (a spatial gate).
        3. Scale    - features are re-weighted by the spatial gate.

    Cost: one 2->1 channel 3D conv (a few dozen params); negligible VRAM.
    """
    def __init__(self, kernel_size=7):
        super(GCSpatialBlock3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        r"""#Args: x in channels-first layout (B, C, D, H, W)."""
        avg_map = x.mean(dim=1, keepdim=True)               # (B,1,D,H,W)
        max_map = x.max(dim=1, keepdim=True)[0]             # (B,1,D,H,W)
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class GLO_NCA_Cell(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False, use_attention=False, se_reduction=4, use_spatial=False, dropout=0.0, spatial_kernel_size=7):
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
        super(GLO_NCA_Cell, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.use_attention = use_attention
        # Opt-in, memory-only gradient checkpointing of the unrolled NCA steps.
        # Default False -> forward() is byte-identical to the original behaviour
        # (V2 and V3-without-the-flag are completely unchanged). When True, each
        # step's activations are recomputed in backward instead of being stored,
        # trading compute for VRAM. The RNG state is preserved so the stochastic
        # fire-rate mask is identical on recompute -> same math, same outputs.
        self.use_checkpoint = False

        # One Input
        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        # Mathematically identical to BatchNorm3d(hidden_size,
        # track_running_stats=False), but operates directly on the
        # channels-last hidden activation. See ChannelsLastBatchNorm: same
        # function, same state-dict keys, without the two 128-channel
        # transposes that BatchNorm3d's layout requirement forced.
        self.bn = ChannelsLastBatchNorm(hidden_size)
        # Light dropout on the hidden update (regularisation; 0.0 = off).
        self.dropout_p = dropout
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None

        # Global-context blocks (thesis novelty). Only built when enabled so the
        # baseline keeps the original parameter count exactly.
        #   se: channel attention (which modality/feature matters)
        #   gc: spatial attention (which regions of the volume matter)
        self.use_spatial = use_spatial
        self.se = SEBlock3D(channel_n, reduction=se_reduction) if use_attention else None
        # Spatial global-context kernel. Configurable rather than
        # hardcoded: it defines the receptive field of the thesis's
        # global-context mechanism, so it must be stated by the
        # config rather than buried here. Default 7 preserves the
        # original behaviour for every existing caller.
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
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell.

        When attention is enabled the locally-perceived features are modulated
        by the global SE gate, so each update sees both local neighbourhood
        (the conv) and whole-volume context (the SE block).
            #Args:
                x: image in channels-first layout (B, C, D, H, W)
        """
        y1 = self.p0(x)
        if self.se is not None:
            y1 = self.se(y1)          # channel global-context (SE)
        if self.gc is not None:
            y1 = self.gc(y1)          # spatial global-context (attention-pooled)
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
        # self.bn normalises channels-last directly, so the transpose pair that
        # BatchNorm3d required here is gone. Identical mathematics.
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
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        # Checkpoint whenever gradients are being computed (i.e. a backward will
        # follow). Not gated on self.training: it is valid in any grad-enabled
        # forward and stays OFF under torch.no_grad() inference (no backward to
        # save memory for). Purely memory-only; outputs are unchanged.
        use_ckpt = getattr(self, "use_checkpoint", False) and torch.is_grad_enabled()
        for step in range(steps):
            if use_ckpt:
                # Recompute this step's activations in backward instead of storing
                # them (memory-only). preserve_rng_state keeps the stochastic
                # fire-rate mask identical on recompute, so outputs are unchanged.
                x2 = torch.utils.checkpoint.checkpoint(
                    self.update, x, fire_rate,
                    use_reentrant=False, preserve_rng_state=True).clone()
            else:
                x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
        return x
