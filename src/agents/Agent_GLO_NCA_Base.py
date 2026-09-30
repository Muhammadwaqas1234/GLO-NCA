import torch
import numpy as np
from src.utils.helper import dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent import BaseAgent
import os

class Agent_GLO_NCA_Base(BaseAgent):
    r"""Base agent for GLO-NCA models: seeding, data preparation and output handling."""
    def initialize(self):
        r"""Read channel counts and create the output pool."""
        super().initialize()
        self.input_channels = self.exp.get_from_config('input_channels')
        self.output_channels = self.exp.get_from_config('output_channels')
        self.pool = Pool()

    def loss_no_oscillation(self, x, target, freeChange=True):
        r"""Penalty on large cell updates to damp oscillation."""
        if freeChange:
            x[x <= 1] = 0
            loss = x.sum() / torch.numel(x)
        else:
            xin_sum = torch.sum(x) + 1
            x = torch.square(target-x)
            loss = torch.sum(x) / xin_sum
        return loss

    def save_state(self, model_path):
        r"""Save the model state."""
        super().save_state(model_path)
        if self.pool.__len__() != 0 and self.exp.get_from_config('save_pool'):
            dump_compressed_pickle_file(self.pool, os.path.join(model_path, 'pool.pbz2'))

    def load_state(self, model_path):
        r"""Load a saved model state."""
        super().load_state(model_path)
        if os.path.exists(os.path.join(model_path, 'pool.pbz2')):
            self.pool = load_compressed_pickle_file(os.path.join(model_path, 'pool.pbz2'))

    def pad_target_f(self, target, padding):
        r"""Pad a tensor on all sides."""
        target = np.pad(target, [(padding, padding), (padding, padding), (0, 0)])
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target.astype(np.float32)).to(self.device)
        return target

    def make_seed(self, img):
        r"""Create the initial cell state with the input channels in the first channels."""
        # 2D data.
        if( self.exp.dataset.slice != None):
            if len(img.shape) == 3:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)
                seed[..., :img.shape[3]] = img
            else:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)
                seed[..., 0:img.shape[-1]] = img

        # 3D data.
        else:
            channel_n = self.exp.get_from_config('channel_n')
            seed = torch.zeros(
                (img.shape[0], img.shape[1], img.shape[2], img.shape[3], channel_n),
                dtype=torch.float32, device=self.device)
            if len(img.shape) == 4:
                # Single-channel volume (B, X, Y, Z) -> place in channel 0.
                seed[..., 0] = img
            else:
                # Multi-modal volume (B, X, Y, Z, C): inputs go in the first C channels.
                seed[..., 0:img.shape[-1]] = img

        return seed

    def repeatBatch(self, seed, target, batch_duplication):
        r"""Repeat the batch (seed and target) batch_duplication times."""
        return torch.Tensor.repeat_interleave(seed, batch_duplication, dim=0), torch.Tensor.repeat_interleave(target, batch_duplication, dim=0)

    def getInferenceSteps(self):
        r"""Number of inference steps from the config."""
        if type(self.exp.get_from_config('inference_steps')) is list:
            steps = self.exp.get_from_config('inference_steps')
        else:
            steps = self.exp.get_from_config('inference_steps')
        return steps

    def prepare_data(self, data, eval=False):
        r"""Prepare a batch: move to the device and build the seed."""
        id, inputs, targets = data
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.make_seed(inputs)
        if not eval:
            if self.exp.get_from_config('Persistence'):
                inputs = self.pool.getFromPool(inputs, id, self.device)
            inputs, targets = self.repeatBatch(inputs, targets, self.exp.get_from_config('batch_duplication'))
        return id, inputs, targets

    def get_outputs(self, data, full_img=False, **kwargs):
        r"""Run the model and return (outputs, targets)."""
        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets

    def initialize_epoch(self):
        r"""Create the sample pool for the current epoch."""
        if self.exp.get_from_config('Persistence'):
            self.epoch_pool = Pool()

    def conclude_epoch(self):
        r"""Make the epoch pool the active pool."""
        if self.exp.get_from_config('Persistence'):
            self.pool = self.epoch_pool
            print("Pool_size: " + str(len(self.pool)))
        return

    def prepare_image_for_display(self, image):
        r"""First three channels of an image for display."""
        return image[...,0:3]

class Pool():
    r"""Pool of previous model outputs."""
    def __init__(self):
        r"""Create an empty pool."""
        self.pool = {}

    def __len__(self):
        r"""Number of stored outputs."""
        return len(self.pool)

    def addToPool(self, outputs, ids):
        r"""Add an output to the pool."""
        for i, key in enumerate(ids):
            self.pool[key] = outputs[i]

    def getFromPool(self, inputs, ids, device):
        r"""Get outputs from the pool."""
        for i, key in enumerate(ids):
            if key in self.pool.keys():
                inputs[i] = self.pool[key].to(device)
        return inputs

