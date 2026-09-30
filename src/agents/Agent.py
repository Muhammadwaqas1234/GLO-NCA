import torch
import torch.optim as optim


class BaseAgent():
    r"""Base class for all agents: holds the models and builds their optimizers."""
    def __init__(self, model):
        r"""Keep the models; the experiment is attached later."""
        self.model = model

    def set_exp(self, exp):
        r"""Set experiment of agent and initialize."""
        self.exp = exp
        self.initialize()

    def initialize(self):
        r"""Build optimizers and schedulers for the agent's models."""
        self.device = torch.device(self.exp.get_from_config('device'))
        # One optimizer and scheduler per model when several models are stacked.
        if isinstance(self.model, list):
            self.optimizer = []
            self.scheduler = []
            for m in range(len(self.model)):
                self.optimizer.append(self._make_optimizer(self.model[m].parameters()))
                self.scheduler.append(optim.lr_scheduler.ExponentialLR(self.optimizer[m], self.exp.get_from_config('lr_gamma')))
        else:
            self.optimizer = self._make_optimizer(self.model.parameters())
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))

    def _make_optimizer(self, params):
        r"""Build the optimizer from the config (AdamW by default)."""
        name = (self.exp.get_from_config('optimizer') or 'adamw').lower()
        lr = self.exp.get_from_config('lr')
        betas = self.exp.get_from_config('betas')
        weight_decay = self.exp.get_from_config('weight_decay')
        if weight_decay is None:
            weight_decay = 1e-4 if name == 'adamw' else 0.0
        if name == 'adam':
            return optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
