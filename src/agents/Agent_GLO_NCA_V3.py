r"""
================================================================================
Agent_GLO_NCA_V3 -- production-runner adapter for the V3 multi-level model.
================================================================================
V3 is a single UNIFIED ``nn.Module`` (``GLO_NCA_V3_MultiLevel``) whose forward
takes channels-last modalities (B,X,Y,Z,4) and returns channels-first logits
(B,3,X,Y,Z). The existing experiment runner, however, drives the model ONLY
through the agent interface used by V2:

    prepare_data(data, eval=...)  -> (id, inputs, targets)   [channels-last]
    get_outputs(data, full_img=?) -> (outputs, targets)      [channels-last,
                                                               WT/TC/ET last dim]
    agent.model      : list[nn.Module]     (grad-clip + param count)
    agent.optimizer  : list[Optimizer]     (zero_grad/step in the training step)
    agent.scheduler  : list[_LRScheduler]  (step in the training step)

This adapter presents EXACTLY that interface around the one V3 model, so the
runner's training step, loss handling, EMA, checkpoint/resume, evaluation,
threshold tuning, manifest and graphs are all reused UNCHANGED. Nothing about
V2 is touched: V2 keeps using ``Agent_GLO_NCA`` with two ``BasicNCA3D`` models.

Design notes:
  * ``self.model`` is a single-element list ``[v3_model]`` so the base
    ``Agent_NCA.initialize`` (which branches on ``isinstance(self.model, list)``)
    builds ONE optimizer + ONE scheduler -- matching V3's single set of params.
  * ``prepare_data`` does NOT build a V2 seed: V3 seeds each level internally, so
    it needs the raw modalities channels-last. Targets are kept channels-last
    with WT/TC/ET in the last dim (as the dataset yields them).
  * ``get_outputs`` runs the V3 forward and permutes the channels-first logits
    back to channels-last so the runner's per-region loss/eval code is identical.
    ``full_img`` is accepted (eval calls it) but V3 processes the whole volume in
    one pass anyway, so there is no separate patch/full-image code path.
================================================================================
"""
import torch

from src.agents.Agent_NCA import Agent_NCA


class Agent_GLO_NCA_V3(Agent_NCA):
    """Adapter agent that drives the unified V3 multi-level model through the
    same interface the runner uses for V2."""

    def initialize(self):
        # Base Agent_NCA.initialize -> BaseAgent.initialize builds one optimizer +
        # one ExponentialLR scheduler because self.model is a length-1 list. The
        # runner then REPLACES the scheduler with a CosineAnnealingLR over the
        # whole run (identical to V2), so the placeholder here is harmless.
        super().initialize()

    def prepare_data(self, data, eval=False):
        r"""Return (id, modalities_cl, targets_cl) on the device.

        Unlike the V2 agent we do NOT call ``make_seed``: the V3 model builds its
        own per-level seeds. ``inputs`` stay as raw modalities (B,X,Y,Z,4) and
        ``targets`` stay channels-last (B,X,Y,Z,3), exactly as the dataset emits
        them, so the runner's per-region loss/metrics indexing is unchanged.
        """
        id, inputs, targets = data
        # non_blocking pairs with the loader's pin_memory=True so the host->device
        # copy can overlap compute. Numerically identical; PyTorch inserts the
        # needed stream synchronisation before the tensors are used.
        inputs = inputs.type(torch.FloatTensor).to(self.device, non_blocking=True)
        targets = targets.type(torch.FloatTensor).to(self.device, non_blocking=True)
        if targets.dim() < 5:  # (B,X,Y,Z) -> (B,X,Y,Z,1); normally already 5-D
            targets = targets.unsqueeze(-1)
        return id, inputs, targets

    def get_outputs(self, data, full_img=False, tag="", **kwargs):
        r"""Run the unified V3 forward and return channels-last (outputs, targets).

            #Args
                data: (id, modalities_cl, targets_cl) from ``prepare_data``.
                full_img: accepted for API parity with the V2 agent (evaluation
                    passes it). V3 always processes the whole input volume in a
                    single forward, so there is no separate patch code path.
            #Returns
                outputs_cl: (B, X, Y, Z, 3) logits (WT/TC/ET last), channels-last.
                targets_cl: (B, X, Y, Z, 3) ground truth, channels-last.
        """
        id, inputs, targets = data
        model = self.model[0]
        logits_cf = model(inputs)                      # (B, 3, X, Y, Z)
        outputs_cl = logits_cf.permute(0, 2, 3, 4, 1).contiguous()  # -> channels-last
        # Align targets to the model's output resolution (V3's finest level fixes
        # the output size; the dataset volume may differ). Trilinear on the GT is
        # only needed when sizes differ; for the standard config they match.
        if targets.shape[1:4] != outputs_cl.shape[1:4]:
            t_cf = targets.permute(0, 4, 1, 2, 3).contiguous()
            t_cf = torch.nn.functional.interpolate(
                t_cf, size=tuple(outputs_cl.shape[1:4]), mode="nearest")
            targets = t_cf.permute(0, 2, 3, 4, 1).contiguous()
        return outputs_cl, targets
