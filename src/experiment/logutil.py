r"""Logging: a single logger that writes both a readable console stream and a
detailed ``logs/training.log`` file, plus per-metric CSV writers and an optional
TensorBoard wrapper.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List, Optional


def get_logger(ws) -> logging.Logger:
    """Create a logger writing to logs/training.log and the console."""
    logger = logging.getLogger(f"glo_nca.{ws.experiment_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(ws.path("logs", "training.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))  # clean console
    logger.addHandler(ch)
    return logger


class CSVLogger:
    """Append rows to a CSV with a fixed header. Creates the file with the
    header on first use; appends thereafter (survives resume)."""

    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.fieldnames,
                           extrasaction="ignore").writerow(row)


class TensorBoard:
    """Thin wrapper around SummaryWriter that degrades gracefully if TensorBoard
    is unavailable, so a missing dependency never crashes training."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except Exception:
            self.writer = None

    def add_scalars(self, epoch: int, scalars: Dict[str, float]) -> None:
        if self.writer is None:
            return
        for tag, value in scalars.items():
            if value is None:
                continue
            try:
                self.writer.add_scalar(tag, float(value), epoch)
            except Exception:
                pass

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
