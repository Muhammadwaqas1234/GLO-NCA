"""Linear LR warmup followed by cosine annealing (production schedule).

Stepped once per optimiser step:

  step < warmup_steps :  lr = base_lr * (step + 1) / warmup_steps
  step >= warmup_steps:  lr = eta_min + 0.5 * (base_lr - eta_min) *
                              (1 + cos(pi * (step - warmup_steps) /
                                       (total_steps - warmup_steps)))

Peak LR is exactly ``base_lr``, the final LR is exactly ``eta_min`` and the
total step count is unchanged. Past ``total_steps`` the LR holds at
``eta_min``. The LR depends only on the step index, so resume is exact.
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
