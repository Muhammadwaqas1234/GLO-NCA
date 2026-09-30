import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    r"""Dice loss."""
    def __init__(self, useSigmoid = True):
        r"""useSigmoid: apply a sigmoid to the input."""
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function"""
        if self.useSigmoid:
            input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceLoss_mask(torch.nn.Module):
    r"""Dice loss computed on masked voxels only."""
    def __init__(self, useSigmoid = True):
        r"""useSigmoid: apply a sigmoid to the input."""
        self.useSigmoid = useSigmoid
        super(DiceLoss_mask, self).__init__()

    def forward(self, input, target, mask = None, smooth=1):
        r"""Forward function"""
        if self.useSigmoid:
            input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)
        mask = torch.flatten(mask)

        input = input[~mask]
        target = target[~mask]
        intersection = (input * target).sum()
        dice = (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)

        return 1 - dice

class DiceBCELoss(torch.nn.Module):
    r"""Dice + BCE loss."""
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceBCELoss"""
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function"""
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)
        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class BCELoss(torch.nn.Module):
    r"""BCE loss."""
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceBCELoss"""
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function"""
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        return BCE

class FocalLoss(torch.nn.Module):
    r"""Focal loss."""
    def __init__(self, gamma=2, eps=1e-7):
        r"""Initialisation method of DiceBCELoss"""
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        r"""Forward function"""
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        loss = loss_bce * (1 - logit) ** self.gamma  # focal term
        loss = loss.mean()
        return loss


class DiceFocalLoss(FocalLoss):
    r"""Dice + focal loss."""
    def __init__(self, gamma=2, eps=1e-7):
        r"""Initialisation method of DiceBCELoss"""
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        r"""Forward function"""
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + 1.)/(input.sum() + target.sum() + 1.)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        focal = loss_bce * (1 - logit) ** self.gamma  # focal term
        dice_focal = focal.mean() + dice_loss
        return dice_focal


class DiceCELoss(torch.nn.Module):
    r"""Dice + Cross-Entropy loss for multi-class / multi-label segmentation."""
    def __init__(self, useSigmoid=True, dice_weight=1.0, ce_weight=1.0):
        r"""Store the loss weights."""
        super(DiceCELoss, self).__init__()
        self.useSigmoid = useSigmoid
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, input, target, smooth=1):
        r"""Loss averaged over the region channels."""
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
    r"""Tversky + BCE loss - tuned for small, imbalanced regions (ET / TC)."""
    def __init__(self, alpha=0.3, beta=0.7, ce_weight=0.5, useSigmoid=True):
        r"""Store the loss weights."""
        super(TverskyCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_weight = ce_weight
        self.useSigmoid = useSigmoid

    def forward(self, input, target, smooth=1):
        r"""Compute the loss for one region."""
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
    r"""Focal Tversky + BCE - focuses learning on the hard, small regions (ET)."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, ce_weight=0.5, useSigmoid=True):
        r"""Store the loss weights."""
        super(FocalTverskyCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_weight = ce_weight
        self.useSigmoid = useSigmoid

    def forward(self, input, target, smooth=1):
        r"""Compute the loss for one region."""
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