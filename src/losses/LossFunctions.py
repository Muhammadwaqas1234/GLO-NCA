import torch

class DiceLoss(torch.nn.Module):
    r"""Dice Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceLoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceCELoss(torch.nn.Module):
    r"""Dice + Cross-Entropy loss for multi-class / multi-label segmentation.

    Designed for BraTS, where the tumor regions (ET / TC / WT) are *nested*
    and therefore treated as independent binary channels (multi-label) rather
    than mutually exclusive softmax classes. Each channel gets a sigmoid, a
    Dice term and a binary-cross-entropy term; the per-channel losses are
    averaged. This matches the per-region Dice the Agent loop already computes.

    #Args:
        useSigmoid: apply sigmoid to logits (set False if inputs are already
            probabilities).
        dice_weight / ce_weight: relative weighting of the two terms.
    """
    def __init__(self, useSigmoid=True, dice_weight=1.0, ce_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.useSigmoid = useSigmoid
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, input, target, smooth=1):
        r"""Forward function.

        Accepts either a single channel (input shape == target shape) or a
        stack of channels, where the last dim indexes the region. The loss is
        averaged over channels so it is comparable across region counts.
            #Args:
                input: raw logits (or probabilities if useSigmoid=False)
                target: binary ground-truth, same shape as input
                smooth: Dice smoothing value
        """
        if self.useSigmoid:
            input = torch.sigmoid(input)

        # Per-channel BCE keeps the regions independent (multi-label).
        bce = torch.nn.functional.binary_cross_entropy(
            input.clamp(1e-6, 1. - 1e-6), target, reduction='mean')

        input_flat = torch.flatten(input)
        target_flat = torch.flatten(target)
        intersection = (input_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice

        return self.dice_weight * dice_loss + self.ce_weight * bce


class TverskyCELoss(torch.nn.Module):
    r"""Tversky + BCE loss - tuned for small, imbalanced regions (ET / TC).

    Tversky generalises Dice with separate penalties for false positives
    (alpha) and false negatives (beta). For tiny structures like the enhancing
    tumour, setting beta > alpha (e.g. 0.7 / 0.3) penalises MISSED tumour voxels
    harder than false alarms, which raises recall and typically lifts ET/TC Dice
    compared with plain Dice. A small BCE term keeps gradients stable.

    #Args:
        alpha: weight on false positives.
        beta:  weight on false negatives (use beta > alpha for small regions).
        ce_weight: weight of the auxiliary BCE term.
    """
    def __init__(self, alpha=0.3, beta=0.7, ce_weight=0.5, useSigmoid=True):
        super(TverskyCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_weight = ce_weight
        self.useSigmoid = useSigmoid

    def forward(self, input, target, smooth=1):
        prob = torch.sigmoid(input) if self.useSigmoid else input
        bce = torch.nn.functional.binary_cross_entropy(
            prob.clamp(1e-6, 1. - 1e-6), target, reduction='mean')
        p = torch.flatten(prob)
        t = torch.flatten(target)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        return (1 - tversky) + self.ce_weight * bce


class FocalTverskyCELoss(torch.nn.Module):
    r"""Focal Tversky + BCE - focuses learning on the hard, small regions (ET).

    Focal Tversky raises the Tversky loss to a power gamma > 1, which
    down-weights easy (already well-segmented) voxels and concentrates gradient
    on the hard cases. Combined with beta > alpha (recall focus), this is the
    standard SOTA choice for the tiny enhancing-tumour region on BraTS.

    #Args:
        alpha / beta: false-positive / false-negative weights (beta > alpha).
        gamma:  focal exponent (1.0 = plain Tversky; ~1.33 is common).
        ce_weight: weight of the auxiliary BCE term.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, ce_weight=0.5, useSigmoid=True):
        super(FocalTverskyCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_weight = ce_weight
        self.useSigmoid = useSigmoid

    def forward(self, input, target, smooth=1):
        prob = torch.sigmoid(input) if self.useSigmoid else input
        bce = torch.nn.functional.binary_cross_entropy(
            prob.clamp(1e-6, 1. - 1e-6), target, reduction='mean')
        p = torch.flatten(prob)
        t = torch.flatten(target)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        focal_tversky = torch.pow(1 - tversky, self.gamma)
        return focal_tversky + self.ce_weight * bce