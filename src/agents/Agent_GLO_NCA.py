import math
import random

import torch

from src.agents.Agent import BaseAgent


class Agent_GLO_NCA(BaseAgent):
    r"""GLO-NCA cascade agent: one NCA per level, coarse-to-fine inference and joint training."""
    def initialize(self):
        r"""Read the channel counts; optional per-region loss weights default to none."""
        super().initialize()
        self.input_channels = self.exp.get_from_config('input_channels')
        self.output_channels = self.exp.get_from_config('output_channels')
        self.region_weights = None

    def inference_steps(self):
        r"""NCA steps per level from the config."""
        return self.exp.get_from_config('inference_steps')

    def make_seed(self, img):
        r"""Initial cell state (B, X, Y, Z, channel_n) with the inputs in the first channels."""
        channel_n = self.exp.get_from_config('channel_n')
        seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3], channel_n),
                           dtype=torch.float32, device=self.device)
        if len(img.shape) == 4:
            # Single-channel volume (B, X, Y, Z): place in channel 0.
            seed[..., 0] = img
        else:
            # Multi-modal volume (B, X, Y, Z, C): inputs go in the first C channels.
            seed[..., 0:img.shape[-1]] = img
        return seed

    def prepare_data(self, data, eval=False):
        r"""Move a batch to the device, build the seed and repeat it for batch duplication."""
        id, inputs, targets = data
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.make_seed(inputs)
        if not eval:
            dup = self.exp.get_from_config('batch_duplication')
            inputs = torch.Tensor.repeat_interleave(inputs, dup, dim=0)
            targets = torch.Tensor.repeat_interleave(targets, dup, dim=0)
        return id, inputs, targets

    def batch_step(self, data, loss_f):
        r"""One optimisation step over all levels; returns the loss per region."""
        data = self.prepare_data(data)
        outputs, targets = self.get_outputs(data)
        for m in range(self.exp.get_from_config('train_model')+1):
            self.optimizer[m].zero_grad()
        loss = 0
        loss_ret = {}
        for m in range(outputs.shape[-1]):
            if 1 in targets[..., m]:
                loss_loc = loss_f(outputs[..., m], targets[..., m])
                if self.region_weights is not None:
                    loss_loc = self.region_weights[m] * loss_loc
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()

        if loss != 0:
            loss.backward()
            for m in range(self.exp.get_from_config('train_model')+1):
                self.optimizer[m].step()
                self.scheduler[m].step()
        return loss_ret

    def _downscale(self, x, times, pool):
        r"""Max-pool a channels-last volume ``times`` times."""
        for _ in range(times):
            x = pool(x.transpose(1, 4)).transpose(1, 4)
        return x

    def get_outputs(self, data, full_img=False, **kwargs):
        r"""Run the cascade and return (outputs, targets); full_img runs on the whole volume."""
        id, inputs, targets = data

        if len(targets.shape) < 5:
            targets = torch.unsqueeze(targets, 4)

        # Scaling factor between levels.
        scale_fac = 2
        if self.exp.get_from_config('scale_factor') is not None:
            scale_fac = self.exp.get_from_config('scale_factor')

        max_pool = torch.nn.MaxPool3d(2, 2, 0)
        up = torch.nn.Upsample(scale_factor=scale_fac, mode='nearest')
        last = self.exp.get_from_config('train_model')
        steps = self.inference_steps()
        fire_rate = self.exp.get_from_config('cell_fire_rate')
        input_channel = self.exp.get_from_config('input_channels')

        # Downscale the input and targets once per level for the first level.
        full_res = inputs
        full_res_gt = targets
        inputs_loc = self._downscale(inputs, last*int(math.log2(scale_fac)), max_pool)
        targets_loc = self._downscale(targets, last*int(math.log2(scale_fac)), max_pool)

        # Evaluation: inference on the full image, from the coarse level to the fine level.
        if full_img == True:
            with torch.no_grad():
                for m in range(last+1):
                    if m == last:
                        stp = steps[m] if type(steps) is list else steps
                        outputs = self.model[m](inputs_loc, steps=stp, fire_rate=fire_rate)
                    else:
                        outputs = self.model[m](inputs_loc, steps=steps[m], fire_rate=fire_rate)
                        # Upscale low-resolution features to the next level.
                        outputs = torch.permute(up(torch.permute(outputs, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))
                        next_res = self._downscale(full_res, last - (m + 1), max_pool)
                        # Concatenate low-resolution features with the higher-resolution image.
                        inputs_loc = torch.concat((next_res[..., :input_channel], outputs[..., input_channel:]), 4)
                        targets_loc = targets
        # Training: inference on patches.
        else:
            for m in range(last+1):
                if m == last:
                    stp = steps[m] if type(steps) is list else steps
                    outputs = self.model[m](inputs_loc, steps=stp, fire_rate=fire_rate)
                else:
                    next_res = self._downscale(full_res, last - (m + 1), max_pool)
                    next_res_gt = self._downscale(full_res_gt, last - (m + 1), max_pool)

                    outputs = self.model[m](inputs_loc, steps=steps[m], fire_rate=fire_rate)

                    # Upscale low-resolution features to the next level.
                    outputs = torch.permute(up(torch.permute(outputs, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))
                    # Concatenate low-resolution features with the higher-resolution image.
                    inputs_loc = torch.concat((next_res[..., :input_channel], outputs[..., input_channel:]), 4)
                    inputs_loc, targets_loc, full_res, full_res_gt = self._random_patch(
                        inputs_loc, next_res_gt, full_res, full_res_gt, m, scale_fac)

        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc

    def _random_patch(self, inputs_full, targets_full, full_res, full_res_gt, m, scale_fac):
        r"""Crop one random patch per batch element for the next level, at every resolution."""
        size = self.exp.get_from_config('input_size')[0]
        device = self.exp.get_from_config('device')
        inputs_loc = torch.zeros((inputs_full.shape[0], size[0], size[1], size[2], inputs_full.shape[4])).to(device)
        targets_loc = torch.zeros((targets_full.shape[0], size[0], size[1], size[2], targets_full.shape[4])).to(device)
        size_full = [int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac)]
        full_res_new = torch.zeros((full_res.shape[0], *size_full, full_res.shape[4])).to(device)
        full_res_gt_new = torch.zeros((full_res.shape[0], *size_full, full_res_gt.shape[4])).to(device)

        factor_pow = math.pow(2, self.exp.get_from_config('train_model') - m - 1)
        for b in range(inputs_loc.shape[0]):
            pos_x = random.randint(0, inputs_full.shape[1] - size[0])
            pos_y = random.randint(0, inputs_full.shape[2] - size[1])
            pos_z = random.randint(0, inputs_full.shape[3] - size[2])
            fx, fy, fz = int(pos_x * factor_pow), int(pos_y * factor_pow), int(pos_z * factor_pow)

            inputs_loc[b] = inputs_full[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
            targets_loc[b] = targets_full[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
            full_res_new[b] = full_res[b, fx:fx+size_full[0], fy:fy+size_full[1], fz:fz+size_full[2], :]
            full_res_gt_new[b] = full_res_gt[b, fx:fx+size_full[0], fy:fy+size_full[1], fz:fz+size_full[2], :]
        return inputs_loc, targets_loc, full_res_new, full_res_gt_new
