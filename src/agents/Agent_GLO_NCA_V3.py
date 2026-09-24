r"""Agent adapter that drives the two-level GLO-NCA model through the runner's agent interface.

The model takes channels-last modalities (B,X,Y,Z,4) and returns channels-first logits
(B,3,X,Y,Z). The runner expects:

    prepare_data(data, eval=...)  -> (id, inputs, targets)   channels-last
    get_outputs(data, full_img=?) -> (outputs, targets)      channels-last, WT/TC/ET last
    agent.model / optimizer / scheduler : one-element lists

self.model is [model], so the base initialize builds one optimizer and one scheduler.
No seed is built here (the model seeds each level), and the whole volume is processed
in one pass, so full_img is accepted only for API parity.
"""
import torch

from src.agents.Agent_NCA import Agent_NCA


class Agent_GLO_NCA_V3(Agent_NCA):
    """Single-model agent for two-level GLO-NCA."""

    def initialize(self):
        # The base initialize builds a placeholder ExponentialLR; the runner replaces it
        # (WarmupCosineLR in production).
        super().initialize()

    def prepare_data(self, data, eval=False):
        r"""Return (id, modalities_cl, targets_cl) on the device.

        No make_seed: the model builds its own per-level seeds. Inputs (B,X,Y,Z,4) and
        targets (B,X,Y,Z,3) stay channels-last as the dataset emits them.
        """
        id, inputs, targets = data
        # non_blocking pairs with pin_memory=True to overlap the copy with compute.
        inputs = inputs.type(torch.FloatTensor).to(self.device, non_blocking=True)
        targets = targets.type(torch.FloatTensor).to(self.device, non_blocking=True)
        if targets.dim() < 5:  # (B,X,Y,Z) -> (B,X,Y,Z,1); normally already 5-D
            targets = targets.unsqueeze(-1)
        return id, inputs, targets

    def get_outputs(self, data, full_img=False, tag="", **kwargs):
        r"""Run the forward and return channels-last (outputs, targets).

        #Args
            data: (id, modalities_cl, targets_cl) from prepare_data.
            full_img: accepted for API parity; the whole volume is always one forward.
        #Returns
            outputs_cl: (B, X, Y, Z, 3) logits, WT/TC/ET last.
            targets_cl: (B, X, Y, Z, 3) ground truth.
        """
        id, inputs, targets = data
        model = self.model[0]
        logits_cf = model(inputs)                      # (B, 3, X, Y, Z)
        # Deep supervision: in training the model returns (logits, [aux...]); aux logits
        # are stashed for the training step, the only consumer.
        self.last_aux_logits = []
        if isinstance(logits_cf, tuple):
            logits_cf, self.last_aux_logits = logits_cf
        outputs_cl = logits_cf.permute(0, 2, 3, 4, 1).contiguous()  # -> channels-last

        # ROI alignment: if L2 ran on a region of interest, crop the target with the
        # model's own ROI box. last_roi_box() is [] when roi_fraction == 1 (production),
        # so the resize path below handles the normal case.
        boxes = model.last_roi_box() if hasattr(model, "last_roi_box") else []
        if boxes:
            if len(boxes) != outputs_cl.shape[0]:
                raise RuntimeError(
                    f"ROI/target mismatch: model reported {len(boxes)} ROI boxes "
                    f"for a batch of {outputs_cl.shape[0]}. Refusing to train on "
                    "misaligned targets.")
            from src.models.Model_GLO_NCA_GlobalContext import crop_target_to_roi
            targets = crop_target_to_roi(targets, boxes, outputs_cl.shape[1])
        elif targets.shape[1:4] != outputs_cl.shape[1:4]:
            # No ROI: resize the target to the output resolution (L2, 64³ in production).
            # Nearest keeps masks binary, preserving ET within TC within WT.
            t_cf = targets.permute(0, 4, 1, 2, 3).contiguous()
            t_cf = torch.nn.functional.interpolate(
                t_cf, size=tuple(outputs_cl.shape[1:4]), mode="nearest")
            targets = t_cf.permute(0, 2, 3, 4, 1).contiguous()

        if targets.shape[1:4] != outputs_cl.shape[1:4]:
            raise RuntimeError(
                f"prediction/target geometry mismatch: prediction "
                f"{tuple(outputs_cl.shape[1:4])} vs target {tuple(targets.shape[1:4])}")
        return outputs_cl, targets
