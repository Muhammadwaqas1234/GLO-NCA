"""Linear LR warmup preceding the production cosine schedule (EXPERIMENTAL).

The production baseline uses CosineAnnealingLR stepped ONCE PER OPTIMISER STEP
over ``total_steps`` with ``eta_min = optimizer.minimum_learning_rate``. This
wrapper prepends a linear warmup without changing that:

  step < warmup_steps :  lr = base_lr * (step + 1) / warmup_steps
  step >= warmup_steps:  lr = eta_min + 0.5 * (base_lr - eta_min) *
                              (1 + cos(pi * (step - warmup_steps) /
                                       (total_steps - warmup_steps)))

PRESERVED FROM THE BASELINE
  * peak LR is exactly ``base_lr`` (reached at the end of warmup)
  * final LR is exactly ``eta_min``
  * total number of steps is unchanged, so the training budget is identical
  * optimizer, weight decay and seed are untouched

The cosine tail is re-parameterised over the REMAINING steps so it still
completes at eta_min at the final step. This is a deliberate, documented change
of cosine semantics: the decay is compressed into (total - warmup) steps rather
than shifted. The alternative -- keeping the original cosine curve and letting
warmup cut into it -- would not reach the peak LR and was rejected.

Determinism: the LR depends only on the step index, so resume reproduces the
schedule exactly provided ``last_epoch`` is restored.

NOT MEASURED: no claim is made that warmup improves Dice.
"""
from __future__ import annotations

import math

import torch


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine annealing, stepped per optimiser step."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int,
                 eta_min: float = 0.0, last_epoch: int = -1):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must exceed warmup_steps "
                f"({warmup_steps})")
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        out = []
        for base_lr in self.base_lrs:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                lr = base_lr * float(step + 1) / float(self.warmup_steps)
            else:
                progress = ((step - self.warmup_steps) /
                            max(1, self.total_steps - self.warmup_steps))
                progress = min(1.0, max(0.0, progress))
                lr = (self.eta_min + 0.5 * (base_lr - self.eta_min) *
                      (1.0 + math.cos(math.pi * progress)))
            out.append(lr)
        return out


def warmup_steps_from_config(cfg, steps_per_epoch: int) -> int:
    """Resolve configured warmup epochs into optimiser steps."""
    epochs = float(cfg.get("optimizer", "warmup_epochs", 0) or 0)
    return int(round(epochs * int(steps_per_epoch)))
