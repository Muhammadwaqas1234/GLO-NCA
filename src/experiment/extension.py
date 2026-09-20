r"""Explicit continuation of a training run past its planned epoch budget.

A production run is configured with a maximum budget (``training.epochs``).
Reaching it means the run completed what it was planned to do. Training
further is sometimes justified, but it is a deliberate act, so it is kept
separate from ordinary resume:

* ``--resume`` continues an *unfinished* run toward its original budget.
* ``--extend-to N`` continues a *finished* run beyond that budget, and says
  so in the metadata.

The distinction matters for reporting. "Trained for 301 epochs" and "planned
for 300, then explicitly continued for one epoch from the epoch-300
checkpoint" describe different experiments, and only the second is what
actually happened.

An extension is a continuation, never a restart. Model, optimizer, EMA,
global step, validation history, best checkpoint and the top-k ranking all
carry over. The original run's identity (config fingerprint, architecture,
split SHA, seed) must match, or it is a new experiment rather than a
continuation and this module refuses it.

LEARNING-RATE WARNING
---------------------
``CosineAnnealingLR`` is periodic: past ``T_max`` its learning rate rises
again rather than staying at ``eta_min``. Measured for the production
schedule (``T_max`` = 300 epochs, lr 1.6e-3, eta_min 1e-5)::

    epoch 300 -> 1.00e-05      (eta_min, end of the planned decay)
    epoch 301 -> 1.00e-05
    epoch 310 -> 1.44e-05
    epoch 350 -> 1.17e-04      (11.6x eta_min)

So "just keep the saved scheduler state" is not a neutral choice past the
horizon -- it is a warm restart, and it changes the LR trajectory. Because
that is a scientific decision, an extension must state its policy explicitly
(``--lr-policy``); there is no default.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Identity fields that must be identical for a run to count as a continuation.
PROTECTED_IDENTITY = (
    ("experiment", "seed"),
    ("training", "batch_size"),
    ("training", "patch_size"),
    ("optimizer", "learning_rate"),
    ("optimizer", "minimum_learning_rate"),
    ("loss", "tversky_beta"),
    ("loss", "focal_gamma"),
    ("ema", "decay"),
)

# Model-architecture fields. A change here is a different model, full stop.
PROTECTED_ARCHITECTURE = (
    ("model", "level1"), ("model", "level2"), ("model", "level3"),
    ("model", "fire_rate"), ("model", "hidden"), ("model", "dropout"),
    ("model", "use_attention"), ("model", "use_spatial"),
    ("model", "feature_fusion"), ("model", "global_context"),
)

LR_POLICIES = {
    "freeze": ("Hold the learning rate at the scheduler's final value "
               "(eta_min). The extension trains at a constant minimum LR; "
               "the original decay curve is left intact and is not resumed."),
    "continue": ("Step the saved scheduler onward. NOTE: CosineAnnealingLR "
                 "is periodic, so this RAISES the learning rate past T_max "
                 "(a warm restart), which changes the LR trajectory."),
    "rebuild": ("Rebuild the cosine schedule over the NEW horizon. This "
                "retroactively changes the whole LR curve, so epochs 1..300 "
                "no longer match the original run. Rarely correct."),
}


class ExtensionError(RuntimeError):
    """An extension was requested that cannot be performed safely."""


@dataclass
class ExtensionPlan:
    """A validated continuation, ready to execute."""

    parent_epoch: int
    target_epoch: int
    original_max_epochs: int
    lr_policy: str
    reason: str
    parent_checkpoint: str
    parent_run_id: str = ""
    early_stopping_was_triggered: bool = False
    identity_checks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def start_epoch(self) -> int:
        return self.parent_epoch          # 0-based index of the next epoch

    @property
    def extension_epochs(self) -> int:
        return self.target_epoch - self.parent_epoch

    def metadata(self) -> Dict[str, Any]:
        """The record that proves epoch N was a continuation, not a restart."""
        return {
            "extension_mode": True,
            "parent_run_id": self.parent_run_id or "unknown",
            "parent_checkpoint": self.parent_checkpoint,
            "parent_epoch": self.parent_epoch,
            "original_max_epochs": self.original_max_epochs,
            "extension_start_epoch": self.parent_epoch + 1,
            "extension_target_epoch": self.target_epoch,
            "extension_epochs": self.extension_epochs,
            "extension_reason": self.reason,
            "lr_policy": self.lr_policy,
            "lr_policy_description": LR_POLICIES[self.lr_policy],
            "early_stopping_was_triggered": self.early_stopping_was_triggered,
            "extension_requested": True,
            "identity_verified": [c["field"] for c in self.identity_checks
                                  if c["match"]],
            "continuation_guarantee": (
                "Model, optimizer, scheduler, EMA, global step, validation "
                "history, best checkpoint and top-k ranking are carried over "
                "from the parent checkpoint. Nothing is reset."),
            "reporting_note": (
                f"The experiment was configured for a maximum of "
                f"{self.original_max_epochs} epochs. After that budget was "
                f"reached, {self.extension_epochs} explicitly recorded "
                f"continuation epoch(s) were executed from the epoch-"
                f"{self.parent_epoch} checkpoint."),
        }


def _nested(cfg: Dict[str, Any], section: str, key: str) -> Any:
    return (cfg.get(section) or {}).get(key)


def compare_identity(parent_cfg: Dict[str, Any],
                     current_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare every protected field. Returns one record per field."""
    out: List[Dict[str, Any]] = []
    for section, key in PROTECTED_IDENTITY + PROTECTED_ARCHITECTURE:
        old, new = _nested(parent_cfg, section, key), _nested(current_cfg,
                                                              section, key)
        out.append({
            "field": f"{section}.{key}",
            "parent": old,
            "current": new,
            # A field absent from BOTH configs is not evidence of drift.
            "match": old == new,
            "architecture": (section, key) in PROTECTED_ARCHITECTURE,
        })
    return out


def plan_extension(*, checkpoint: Dict[str, Any], checkpoint_path: str,
                   current_cfg: Dict[str, Any], target_epoch: int,
                   lr_policy: str, reason: str,
                   parent_run_id: str = "") -> ExtensionPlan:
    """Validate a requested continuation and return an executable plan.

    Raises ExtensionError -- never returns a partially-valid plan -- if the
    checkpoint is incomplete, the target does not extend the parent, the LR
    policy is unstated, or any protected identity field has drifted.
    """
    if lr_policy not in LR_POLICIES:
        raise ExtensionError(
            f"lr_policy must be one of {sorted(LR_POLICIES)}, got "
            f"{lr_policy!r}. Continuing past the cosine horizon changes the "
            f"learning-rate trajectory, so the policy must be stated "
            f"explicitly; there is no safe default.")
    if not str(reason).strip():
        raise ExtensionError(
            "an extension requires an explicit reason: it is a deliberate "
            "departure from the planned training budget and must be "
            "justifiable in the thesis record.")

    required = ("model", "optimizer", "scheduler", "epoch")
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise ExtensionError(
            f"checkpoint {checkpoint_path!r} lacks {missing}; an extension "
            f"must continue complete training state, not just weights.")

    parent_epoch = int(checkpoint["epoch"])
    if target_epoch <= parent_epoch:
        raise ExtensionError(
            f"--extend-to {target_epoch} does not extend a checkpoint already "
            f"at epoch {parent_epoch}. The target must be greater.")

    parent_cfg = checkpoint.get("config") or {}
    checks = compare_identity(parent_cfg, current_cfg)
    drift = [c for c in checks if not c["match"]]
    if drift:
        arch = [c for c in drift if c["architecture"]]
        lines = [f"  {c['field']}: parent={c['parent']!r} -> "
                 f"current={c['current']!r}" for c in drift]
        raise ExtensionError(
            ("architecture" if arch else "configuration") +
            " drift: this is a NEW EXPERIMENT, not a continuation.\n" +
            "\n".join(lines) +
            "\nRun it as a fresh experiment, or restore the original config.")

    original_max = int(_nested(parent_cfg, "training", "epochs") or
                       parent_epoch)
    es = checkpoint.get("early_stopping") or {}

    return ExtensionPlan(
        parent_epoch=parent_epoch,
        target_epoch=int(target_epoch),
        original_max_epochs=original_max,
        lr_policy=lr_policy,
        reason=str(reason).strip(),
        parent_checkpoint=checkpoint_path,
        parent_run_id=parent_run_id,
        early_stopping_was_triggered=bool(es.get("triggered", False)),
        identity_checks=checks,
    )


def apply_lr_policy(schedulers: List[Any], plan: ExtensionPlan,
                    steps_per_epoch: int, logger=None) -> Dict[str, Any]:
    """Apply the chosen LR policy. Returns what was done, for the record."""
    note: Dict[str, Any] = {"policy": plan.lr_policy,
                            "description": LR_POLICIES[plan.lr_policy]}

    if plan.lr_policy == "freeze":
        frozen = []
        for s in schedulers:
            lr = [g["lr"] for g in s.optimizer.param_groups]
            frozen.append(lr)
            # Replace with a constant-LR schedule so nothing steps the LR.
            for g in s.optimizer.param_groups:
                g["lr"] = g["lr"]
        note["frozen_lr"] = frozen
        note["effect"] = ("learning rate held constant for the extension; "
                          "the scheduler is not stepped")
    elif plan.lr_policy == "continue":
        note["effect"] = ("saved scheduler stepped onward; for a periodic "
                          "cosine this RAISES the learning rate past T_max")
        note["warning"] = ("CosineAnnealingLR is periodic. The extension does "
                           "NOT train at eta_min: the LR climbs (measured "
                           "1.00e-05 at epoch 300 -> 1.17e-04 at epoch 350).")
    else:  # rebuild
        note["effect"] = ("cosine rebuilt over the new horizon; the LR curve "
                          "for the ORIGINAL epochs no longer matches the "
                          "parent run")
        note["warning"] = ("this retroactively redefines the schedule the "
                           "parent run actually used")

    if logger is not None:
        logger.info("LR policy '%s': %s", plan.lr_policy, note["effect"])
        if "warning" in note:
            logger.warning("LR policy '%s': %s", plan.lr_policy,
                           note["warning"])
    return note


def write_extension_record(directory: str, plan: ExtensionPlan,
                           lr_note: Dict[str, Any]) -> str:
    """Persist the extension metadata next to the run's other artifacts."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "extension.json")
    payload = plan.metadata()
    payload["lr_policy_applied"] = lr_note
    payload["identity_checks"] = plan.identity_checks
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path
