r"""Early stopping on a validation metric.

Training stops once the monitored validation metric has failed to improve by
more than ``min_delta`` for ``patience`` consecutive validation evaluations.
The best checkpoint written during the run is the one that should be used
afterwards; stopping early never overwrites it.

Two rules this module exists to enforce:

* **The monitor is a validation metric, never training loss.** A falling
  training loss is exactly what overfitting looks like, so using it to decide
  when to stop would defeat the purpose.
* **The test split is never consulted.** Early stopping is model selection,
  and model selection on test data leaks it. The test set stays frozen until
  the configuration is final.

The counter advances per *validation evaluation*, not per epoch. Those are the
same thing when validation runs every epoch (the production default), but the
distinction matters if validation is ever made less frequent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EarlyStopping:
    """Patience-based stopping on a monitored validation metric.

    Args:
        patience: consecutive non-improving evaluations tolerated before
            stopping. ``0`` disables stopping while still tracking the best
            value.
        min_delta: how much better a value must be to count as an
            improvement. Guards against stopping being deferred forever by
            noise-level gains.
        mode: ``"max"`` for metrics where higher is better (Dice, IoU),
            ``"min"`` for metrics where lower is better (loss, HD95).
        monitor: name of the monitored metric, recorded for the run report.
        enabled: ``False`` tracks the best value but never signals a stop.
    """

    patience: int = 15
    min_delta: float = 0.001
    mode: str = "max"
    monitor: str = "validation mean foreground Dice"
    enabled: bool = True

    best: Optional[float] = None
    best_epoch: int = 0
    counter: int = 0
    stopped_epoch: int = 0
    should_stop: bool = False
    history: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")

    # ------------------------------------------------------------------ core
    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float, epoch: int) -> bool:
        """Record one validation evaluation. Returns True if training should stop.

        ``value`` must come from the validation split. A non-finite value is
        treated as a non-improvement rather than raising, so a single bad
        evaluation cannot abort a long run.
        """
        import math

        finite = isinstance(value, (int, float)) and math.isfinite(value)
        improved = finite and self._is_improvement(float(value))

        if improved:
            self.best = float(value)
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.enabled and self.patience and self.counter >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch

        self.history.append({
            "epoch": epoch,
            "value": float(value) if finite else float("nan"),
            "best": self.best if self.best is not None else float("nan"),
            "counter": self.counter,
            "improved": improved,
        })
        return self.should_stop

    # --------------------------------------------------------------- reporting
    def status(self) -> str:
        """One-line status for the training log."""
        if self.best is None:
            return f"early-stop: no validation value yet (monitor={self.monitor})"
        pat = f"{self.counter}/{self.patience}" if self.patience else "disabled"
        return (f"early-stop: best {self.best:.4f} @ epoch {self.best_epoch} | "
                f"patience {pat}")

    def report(self) -> Dict[str, object]:
        """Machine-readable summary for the run artifacts."""
        return {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best,
            "best_epoch": self.best_epoch,
            "final_counter": self.counter,
            "triggered": self.should_stop,
            "stopped_epoch": self.stopped_epoch or None,
            "evaluations": len(self.history),
            "note": ("Monitored on the VALIDATION split only. The test split is "
                     "never used for stopping or model selection."),
        }

    # ------------------------------------------------------------------ resume
    def state_dict(self) -> Dict[str, object]:
        return {"best": self.best, "best_epoch": self.best_epoch,
                "counter": self.counter, "should_stop": self.should_stop,
                "stopped_epoch": self.stopped_epoch, "history": self.history}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore after a preemption so patience is not silently reset."""
        self.best = state.get("best")
        self.best_epoch = int(state.get("best_epoch", 0) or 0)
        self.counter = int(state.get("counter", 0) or 0)
        self.should_stop = bool(state.get("should_stop", False))
        self.stopped_epoch = int(state.get("stopped_epoch", 0) or 0)
        self.history = list(state.get("history", []) or [])


def from_config(cfg_section: Optional[Dict[str, object]]) -> EarlyStopping:
    """Build from the ``training.early_stopping`` config block.

    A missing block yields the documented defaults with stopping DISABLED, so
    an older config can never acquire new stopping behaviour implicitly.
    """
    section = dict(cfg_section or {})
    return EarlyStopping(
        enabled=bool(section.get("enabled", False)),
        patience=int(section.get("patience", 15)),
        min_delta=float(section.get("min_delta", 0.001)),
        mode=str(section.get("mode", "max")),
        monitor=str(section.get("monitor",
                                "validation mean foreground Dice")),
    )
