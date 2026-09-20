r"""Explicit, persisted training lifecycle.

A long run gets interrupted: a Spot VM is reclaimed, a process is killed, a
laptop sleeps. Afterwards someone has to answer, from the directory alone,
what actually happened -- did it finish its budget, stop early, get extended,
or die? A free-text status string cannot answer that reliably, so the
lifecycle is a state machine whose transitions are enumerated and whose state
is written to disk after every change.

States
------
``CREATED``              workspace exists, nothing has run
``PREFLIGHT``            identity and data gates being checked
``RUNNING``              training an epoch
``VALIDATING``           scoring the validation split
``BEST_UPDATED``         a new best checkpoint was written
``CHECKPOINTED``         full state persisted
``EARLY_STOPPED``        validation plateaued; stopped before the budget
``COMPLETED_300``        the planned epoch budget was reached
``EXTENDED``             continuing past the budget by explicit request
``COMPLETED_EXTENSION``  an extension finished
``FAILED``               unrecoverable error

Two distinctions the file format exists to protect:

* a resumed run must not look like a new experiment, and
* an extended run must not look like the original budget.

Both are recorded as history, not inferred afterwards. ``FAILED`` is
reachable from anywhere, because a crash can happen anywhere.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

CREATED = "CREATED"
PREFLIGHT = "PREFLIGHT"
RUNNING = "RUNNING"
VALIDATING = "VALIDATING"
BEST_UPDATED = "BEST_UPDATED"
CHECKPOINTED = "CHECKPOINTED"
EARLY_STOPPED = "EARLY_STOPPED"
COMPLETED_300 = "COMPLETED_300"
EXTENDED = "EXTENDED"
COMPLETED_EXTENSION = "COMPLETED_EXTENSION"
FAILED = "FAILED"

STATES = (CREATED, PREFLIGHT, RUNNING, VALIDATING, BEST_UPDATED,
          CHECKPOINTED, EARLY_STOPPED, COMPLETED_300, EXTENDED,
          COMPLETED_EXTENSION, FAILED)

# Terminal for this process. A run may LEAVE these only by an explicit new
# action: resuming a stopped run, or extending a completed one.
TERMINAL = (EARLY_STOPPED, COMPLETED_300, COMPLETED_EXTENSION, FAILED)

#: Allowed transitions. FAILED is appended to every source below.
_ALLOWED: Dict[str, tuple] = {
    CREATED: (PREFLIGHT, RUNNING),
    PREFLIGHT: (RUNNING,),
    # A resumed run re-enters RUNNING from a terminal-ish state; that is why
    # EARLY_STOPPED and COMPLETED_300 can lead back to RUNNING/EXTENDED.
    RUNNING: (VALIDATING, CHECKPOINTED, EARLY_STOPPED, COMPLETED_300,
              COMPLETED_EXTENSION, RUNNING),
    VALIDATING: (BEST_UPDATED, CHECKPOINTED, RUNNING, EARLY_STOPPED,
                 COMPLETED_300, COMPLETED_EXTENSION),
    BEST_UPDATED: (CHECKPOINTED, RUNNING, VALIDATING, EARLY_STOPPED,
                   COMPLETED_300, COMPLETED_EXTENSION),
    CHECKPOINTED: (RUNNING, VALIDATING, EARLY_STOPPED, COMPLETED_300,
                   COMPLETED_EXTENSION),
    EARLY_STOPPED: (RUNNING, EXTENDED),          # resume / explicit extension
    COMPLETED_300: (EXTENDED, RUNNING),          # extension of a finished run
    EXTENDED: (RUNNING, VALIDATING, CHECKPOINTED, COMPLETED_EXTENSION),
    COMPLETED_EXTENSION: (EXTENDED, RUNNING),    # a further continuation
    FAILED: (PREFLIGHT, RUNNING),                # retry after a fix
}


class InvalidTransition(RuntimeError):
    """A transition that the lifecycle does not permit."""


class TrainingState:
    """The run's lifecycle, persisted to ``state.json`` on every change."""

    FILENAME = "state.json"

    def __init__(self, directory: str, *, run_id: str = "",
                 autosave: bool = True):
        self.directory = directory
        self.run_id = run_id
        self.autosave = bool(autosave)
        self.state: str = CREATED
        self.history: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self._record(CREATED, "workspace created")

    # ------------------------------------------------------------- internals
    @property
    def path(self) -> str:
        return os.path.join(self.directory, self.FILENAME)

    def _record(self, state: str, reason: str, **extra: Any) -> None:
        entry = {
            "state": state,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
        }
        entry.update(extra)
        self.history.append(entry)
        self.state = state
        if self.autosave:
            self.save()

    # --------------------------------------------------------------- control
    def can_transition(self, to: str) -> bool:
        if to not in STATES:
            return False
        if to == FAILED:
            return True                      # a crash can happen anywhere
        return to in _ALLOWED.get(self.state, ())

    def transition(self, to: str, reason: str = "", **extra: Any) -> str:
        """Move to `to`. Raises InvalidTransition if the lifecycle forbids it.

        Refusing is the point: a run that silently jumps from CREATED to
        COMPLETED_300 would produce a directory that lies about what ran.
        """
        if to not in STATES:
            raise InvalidTransition(
                f"unknown state {to!r}; expected one of {STATES}")
        if not self.can_transition(to):
            raise InvalidTransition(
                f"{self.state} -> {to} is not a valid transition. "
                f"Allowed from {self.state}: "
                f"{sorted(_ALLOWED.get(self.state, ())) + [FAILED]}")
        self._record(to, reason or f"transition to {to}", **extra)
        return self.state

    def fail(self, reason: str, **extra: Any) -> str:
        """Enter FAILED from any state."""
        self._record(FAILED, reason or "unrecoverable error", **extra)
        return self.state

    def set_metadata(self, **kw: Any) -> None:
        self.metadata.update(kw)
        if self.autosave:
            self.save()

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "is_terminal": self.state in TERMINAL,
            "history": self.history,
            "metadata": self.metadata,
            "states_defined": list(STATES),
            "note": ("A resumed run must not look like a new experiment, and "
                     "an extended run must not look like the original budget. "
                     "`history` is the evidence for both."),
        }

    def save(self) -> str:
        os.makedirs(self.directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        os.replace(tmp, self.path)           # atomic: a crash mid-write cannot
        return self.path                     # leave a truncated state file

    @classmethod
    def load(cls, directory: str, *, run_id: str = "") -> "TrainingState":
        """Restore from disk, or start a fresh CREATED state if absent."""
        obj = cls(directory, run_id=run_id, autosave=False)
        path = os.path.join(directory, cls.FILENAME)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                obj.state = data.get("state", CREATED)
                obj.history = data.get("history", [])
                obj.metadata = data.get("metadata", {})
                obj.run_id = data.get("run_id", run_id)
            except (OSError, ValueError):
                # A corrupt state file must not block recovery; the run
                # restarts its lifecycle rather than dying on JSON.
                obj.history.append({
                    "state": CREATED,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime()),
                    "reason": "state.json unreadable; lifecycle restarted"})
        obj.autosave = True
        return obj

    # --------------------------------------------------------------- queries
    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for h in self.history:
            counts[h["state"]] = counts.get(h["state"], 0) + 1
        return {
            "run_id": self.run_id,
            "current_state": self.state,
            "is_terminal": self.state in TERMINAL,
            "transitions": len(self.history),
            "state_counts": counts,
            "was_resumed": counts.get(RUNNING, 0) > 1,
            "was_extended": EXTENDED in counts,
            "early_stopped": EARLY_STOPPED in counts,
            "completed_budget": COMPLETED_300 in counts,
            "failed": FAILED in counts,
            "first_transition": self.history[0] if self.history else None,
            "last_transition": self.history[-1] if self.history else None,
        }
