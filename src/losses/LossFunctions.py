import torch
import torch.nn.functional as F

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

class DiceLoss_mask(torch.nn.Module):
    r"""Dice Loss mask, that only calculates on masked values
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceLoss mask
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceLoss_mask, self).__init__()

    def forward(self, input, target, mask = None, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
                mask: The mask which defines which values to consider
        """
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
    r"""Dice BCE Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)
        
        intersection = (input * target).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(input.sum() + target.sum() + smooth)  
        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class BCELoss(torch.nn.Module):
    r"""BCE Loss
    """
    def __init__(self, useSigmoid = True):
        r"""Initialisation method of DiceBCELoss
            #Args:
                useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
            #Args:
                input: input array
                target: target array
                smooth: Smoothing value
        """
        input = torch.sigmoid(input)       
        input = torch.flatten(input) 
        target = torch.flatten(target)

        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        return BCE

class FocalLoss(torch.nn.Module):
    r"""Focal Loss
    """
    def __init__(self, gamma=2, eps=1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        loss = loss_bce * (1 - logit) ** self.gamma  # focal loss
        loss = loss.mean()
        return loss


class DiceFocalLoss(FocalLoss):
    r"""Dice Focal Loss
    """
    def __init__(self, gamma=2, eps=1e-7):
        r"""Initialisation method of DiceBCELoss
            #Args:
                gamma
                eps
        """
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        r"""Forward function
            #Args:
                input: input array
                target: target array
        """
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + 1.)/(input.sum() + target.sum() + 1.)

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss_bce = torch.nn.functional.binary_cross_entropy(input, target, reduction='mean')
        focal = loss_bce * (1 - logit) ** self.gamma  # focal loss
        dice_focal = focal.mean() + dice_loss
        return dice_focal


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