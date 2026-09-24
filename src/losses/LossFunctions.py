import torch

class DiceLoss(torch.nn.Module):
    r"""Dice loss."""
    def __init__(self, useSigmoid = True):
        r"""useSigmoid: apply sigmoid to the input."""
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""input, target: arrays of the same shape; smooth: Dice smoothing."""
        if self.useSigmoid:
            input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceCELoss(torch.nn.Module):
    r"""Dice + BCE over independent sigmoid channels (nested WT/TC/ET are multi-label).

    useSigmoid: apply sigmoid to logits; dice_weight / ce_weight: term weights.
    """
    def __init__(self, useSigmoid=True, dice_weight=1.0, ce_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.useSigmoid = useSigmoid
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, input, target, smooth=1):
        r"""Loss averaged over channels (last dim = region); input shape == target shape."""
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
    r"""Tversky + BCE: alpha weights false positives, beta false negatives (beta > alpha favours recall)."""
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
    r"""Focal Tversky + BCE (production loss): Tversky^gamma focuses on hard, small regions such as ET.

    alpha / beta: false-positive / false-negative weights (production derives alpha = 1 - beta).
    gamma: focal exponent (1.0 = plain Tversky). ce_weight: auxiliary BCE weight.
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