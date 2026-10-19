r"""Cascade agent: one NCA per level, coarse-to-fine inference and joint training (after M3D-NCA, Kalkhof et al.)."""
from __future__ import annotations

import contextlib
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim


class Agent_GLO_NCA:
    r"""GLO-NCA cascade agent: holds the NCA levels, their optimisers, and runs training and inference."""

    amp = "off"                       # "bf16": autocast the training forward pass to bfloat16
    case_seed: Optional[int] = None   # base seed for reproducible stochastic inference passes

    def __init__(self, model: List[torch.nn.Module]):
        r"""Keep the levels (coarse to fine); the experiment is attached later with set_exp."""
        self.model = model

    # Setup

    def set_exp(self, exp) -> None:
        r"""Attach the experiment and build optimisers from its config."""
        self.exp = exp
        self.initialize()

    def initialize(self) -> None:
        r"""One optimiser and scheduler per level, channel counts, and the optional loss settings."""
        self.device = torch.device(self.exp.get_from_config('device'))
        self.optimizer = [self._make_optimizer(m.parameters()) for m in self.model]
        # Placeholder schedulers; the trainer replaces them with per-step cosine annealing.
        self.scheduler = [optim.lr_scheduler.ExponentialLR(opt, self.exp.get_from_config('lr_gamma'))
                          for opt in self.optimizer]
        self.input_channels = self.exp.get_from_config('input_channels')
        self.output_channels = self.exp.get_from_config('output_channels')
        self.region_weights: Optional[List[float]] = None
        self.all_regions = False      # True: every region is scored, also when its target is empty
        self.crop_stats = [0, 0]      # ET-positive crops, all crops (ET-biased cropping only)

    def _make_optimizer(self, params) -> optim.Optimizer:
        r"""The optimiser from the config (AdamW by default)."""
        name = (self.exp.get_from_config('optimizer') or 'adamw').lower()
        lr = self.exp.get_from_config('lr')
        betas = self.exp.get_from_config('betas')
        weight_decay = self.exp.get_from_config('weight_decay')
        if weight_decay is None:
            weight_decay = 1e-4 if name == 'adamw' else 0.0
        if name == 'adam':
            return optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def inference_steps(self):
        r"""NCA steps per level from the config."""
        return self.exp.get_from_config('inference_steps')

    # Data

    def make_seed(self, img: torch.Tensor) -> torch.Tensor:
        r"""Initial cell state (B, X, Y, Z, channel_n) with the inputs in the first channels."""
        channel_n = self.exp.get_from_config('channel_n')
        seed = torch.zeros((*img.shape[:4], channel_n), dtype=torch.float32, device=self.device)
        if img.dim() == 4:
            seed[..., 0] = img                    # single-channel volume (B, X, Y, Z)
        else:
            seed[..., 0:img.shape[-1]] = img      # multi-modal volume (B, X, Y, Z, C)
        return seed

    def prepare_data(self, data, eval: bool = False):
        r"""Move a batch to the device, build the seed and repeat it for batch duplication."""
        ids, inputs, targets = data
        inputs = inputs.type(torch.FloatTensor).to(self.device)
        targets = targets.type(torch.FloatTensor).to(self.device)
        inputs = self.make_seed(inputs)
        if not eval:
            dup = self.exp.get_from_config('batch_duplication')
            inputs = torch.Tensor.repeat_interleave(inputs, dup, dim=0)
            targets = torch.Tensor.repeat_interleave(targets, dup, dim=0)
        return ids, inputs, targets

    # Training

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, loss_f) -> Tuple[torch.Tensor, Dict[int, float]]:
        r"""Summed region loss and the loss per region; legacy mode skips regions absent from the crop."""
        loss, per_region = 0, {}
        for m in range(outputs.shape[-1]):
            if self.all_regions:
                # Per-patient loss, then the batch mean, so the batch size does not change its meaning.
                loss_loc = torch.stack([loss_f(outputs[b, ..., m], targets[b, ..., m])
                                        for b in range(outputs.shape[0])]).mean()
            elif 1 in targets[..., m]:
                loss_loc = loss_f(outputs[..., m], targets[..., m])
            else:
                continue
            if self.region_weights is not None:
                loss_loc = self.region_weights[m] * loss_loc
            loss = loss + loss_loc
            per_region[m] = loss_loc.item()
        return loss, per_region

    def optimizer_step(self, loss) -> None:
        r"""Backpropagate and step every level's optimiser and scheduler (no-op for a zero loss)."""
        if loss != 0:
            loss.backward()
            for opt, sched in zip(self.optimizer, self.scheduler):
                opt.step()
                sched.step()

    def batch_step(self, data, loss_f) -> Dict[int, float]:
        r"""One optimisation step over all levels; returns the loss per region."""
        data = self.prepare_data(data)
        if self.amp == "bf16":
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs, targets = self.get_outputs(data)
            outputs = outputs.float()     # the loss is computed in float32
        else:
            outputs, targets = self.get_outputs(data)
        for opt in self.optimizer:
            opt.zero_grad()
        loss, per_region = self.compute_loss(outputs, targets, loss_f)
        self.optimizer_step(loss)
        return per_region

    # Cascade

    @staticmethod
    def _downscale(x: torch.Tensor, times: int, pool) -> torch.Tensor:
        r"""Max-pool a channels-last volume ``times`` times."""
        for _ in range(times):
            x = pool(x.transpose(1, 4)).transpose(1, 4)
        return x

    def get_outputs(self, data, full_img: bool = False, **kwargs):
        r"""Run the cascade coarse to fine; returns (output logits, targets at the output resolution).

        Training (full_img False) crops a random patch for every finer level; evaluation runs every
        level on the whole volume without gradients.
        """
        _, inputs, targets = data
        if targets.dim() < 5:
            targets = torch.unsqueeze(targets, 4)
        scale_fac = self.exp.get_from_config('scale_factor') or 2
        max_pool = torch.nn.MaxPool3d(2, 2, 0)
        up = torch.nn.Upsample(scale_factor=scale_fac, mode='nearest')
        last = self.exp.get_from_config('train_model')
        steps = self.inference_steps()
        fire_rate = self.exp.get_from_config('cell_fire_rate')
        n_in = self.exp.get_from_config('input_channels')

        # The coarsest level sees the input max-pooled once per finer level.
        full_res, full_res_gt = inputs, targets
        inputs_loc = self._downscale(inputs, last * int(math.log2(scale_fac)), max_pool)
        targets_loc = self._downscale(targets, last * int(math.log2(scale_fac)), max_pool)

        # Whole-volume inference runs without gradients unless keep_grad is set (receptive-field analysis).
        no_grad = full_img and not kwargs.get("keep_grad", False)
        with torch.no_grad() if no_grad else contextlib.nullcontext():
            for m in range(last + 1):
                stp = steps[m] if isinstance(steps, list) else steps
                outputs = self.model[m](inputs_loc, steps=stp, fire_rate=fire_rate)
                if m == last:
                    break
                # Upscale this level's state and pair its hidden channels with the finer image.
                outputs = torch.permute(up(torch.permute(outputs, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))
                next_res = self._downscale(full_res, last - (m + 1), max_pool)
                inputs_loc = torch.concat((next_res[..., :n_in], outputs[..., n_in:]), 4)
                if full_img:
                    targets_loc = targets
                else:
                    next_res_gt = self._downscale(full_res_gt, last - (m + 1), max_pool)
                    inputs_loc, targets_loc, full_res, full_res_gt = self._random_patch(
                        inputs_loc, next_res_gt, full_res, full_res_gt, m, scale_fac)

        return outputs[..., self.input_channels:self.input_channels + self.output_channels], targets_loc

    @staticmethod
    def _et_crop(target: torch.Tensor, size) -> Optional[Tuple[int, int, int]]:
        r"""Crop origin centred on a random ET voxel (shifted inside the volume), or None without ET."""
        et = torch.nonzero(target[..., -1] > 0)
        if len(et) == 0:
            return None
        c = et[random.randrange(len(et))].tolist()
        return tuple(min(max(c[d] - size[d] // 2, 0), target.shape[d] - size[d]) for d in range(3))

    def _random_patch(self, inputs_full, targets_full, full_res, full_res_gt, m, scale_fac):
        r"""Crop one patch per batch element for the next level, consistently at every resolution."""
        size = self.exp.get_from_config('input_size')[0]
        device = self.exp.get_from_config('device')
        n = inputs_full.shape[0]
        inputs_loc = torch.zeros((n, *size, inputs_full.shape[4])).to(device)
        targets_loc = torch.zeros((n, *size, targets_full.shape[4])).to(device)
        size_full = [int(full_res.shape[d] / scale_fac) for d in (1, 2, 3)]
        full_res_new = torch.zeros((n, *size_full, full_res.shape[4])).to(device)
        full_res_gt_new = torch.zeros((n, *size_full, full_res_gt.shape[4])).to(device)

        factor_pow = math.pow(2, self.exp.get_from_config('train_model') - m - 1)
        et_prob = self.exp.get_from_config('crop_et_prob') or 0.0
        for b in range(n):
            pos = self._et_crop(targets_full[b], size) if et_prob > 0 and random.random() < et_prob else None
            if pos is None:
                pos = tuple(random.randint(0, inputs_full.shape[d + 1] - size[d]) for d in range(3))
            x, y, z = pos
            if et_prob > 0:
                self.crop_stats[0] += int(targets_full[b, x:x+size[0], y:y+size[1], z:z+size[2], -1].any())
                self.crop_stats[1] += 1
            fx, fy, fz = int(x * factor_pow), int(y * factor_pow), int(z * factor_pow)
            inputs_loc[b] = inputs_full[b, x:x+size[0], y:y+size[1], z:z+size[2], :]
            targets_loc[b] = targets_full[b, x:x+size[0], y:y+size[1], z:z+size[2], :]
            full_res_new[b] = full_res[b, fx:fx+size_full[0], fy:fy+size_full[1], fz:fz+size_full[2], :]
            full_res_gt_new[b] = full_res_gt[b, fx:fx+size_full[0], fy:fy+size_full[1], fz:fz+size_full[2], :]
        return inputs_loc, targets_loc, full_res_new, full_res_gt_new
