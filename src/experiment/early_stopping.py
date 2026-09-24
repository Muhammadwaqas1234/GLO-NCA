r"""Early stopping on a validation metric.

Stops after ``patience`` evaluations without a gain above ``min_delta``. The
monitor is always a validation metric (never training loss) and the test split
is never consulted. Stopping never overwrites the best checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EarlyStopping:
    """Patience-based stopping on a monitored validation metric.

    Args:
        patience: non-improving evaluations tolerated (0 disables stopping).
        min_delta: minimum gain that counts as an improvement.
        mode: "max" (Dice, IoU) or "min" (loss, HD95).
        monitor: metric name, recorded in the run report.
        enabled: False tracks the best value but never stops.
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

    # Core.
    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float, epoch: int) -> bool:
        """Record one validation evaluation; return True to stop. Non-finite values count as no improvement."""
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

    # Reporting.
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

    # Resume.
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
    """Build from ``training.early_stopping``; a missing block disables stopping."""
    section = dict(cfg_section or {})
    return EarlyStopping(
        enabled=bool(section.get("enabled", False)),
        patience=int(section.get("patience", 15)),
        min_delta=float(section.get("min_delta", 0.001)),
        mode=str(section.get("mode", "max")),
        monitor=str(section.get("monitor",
                                "validation mean foreground Dice")),
    )
