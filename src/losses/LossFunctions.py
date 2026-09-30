import torch


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
