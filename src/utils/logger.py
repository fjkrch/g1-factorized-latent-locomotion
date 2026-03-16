"""
Logging utility.

Wraps TensorBoard logging and console logging.
Also writes CSV metric logs for easy post-hoc analysis.

Usage:
    from src.utils.logger import Logger
    logger = Logger(run_dir="outputs/flat/dynamite/seed_42/20260316_120000")
    logger.log_scalar("reward/mean", 100.5, step=1000)
    logger.log_dict({"reward/mean": 100.5, "loss/policy": 0.02}, step=1000)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np


def setup_console_logger(name: str = "dynamite", level: int = logging.INFO) -> logging.Logger:
    """Create a console logger with timestamp formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class Logger:
    """
    Multi-backend logger for training metrics.

    Writes to:
    - TensorBoard (if available)
    - CSV file (always)
    - Console (always)
    """

    def __init__(self, run_dir: str | Path, use_tb: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.console = setup_console_logger()

        # CSV logger
        self.csv_path = self.run_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields: list[str] = []

        # TensorBoard
        self.tb_writer = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except ImportError:
                self.console.warning("TensorBoard not available. Logging to CSV only.")

        # In-memory buffer for current epoch
        self._buffer: dict[str, list[float]] = {}

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar metric."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        """Log a dictionary of metrics at a given step."""
        row = {"step": step, **metrics}

        # TensorBoard
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

        # CSV
        self._write_csv_row(row)

        # Console (summary only)
        summary = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in list(metrics.items())[:6])
        self.console.info(f"[step {step}] {summary}")

    def _write_csv_row(self, row: dict[str, Any]) -> None:
        """Append a row to the CSV log file."""
        fields = list(row.keys())
        if not self._csv_fields:
            self._csv_fields = fields
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()
        else:
            # Handle new fields dynamically
            new_fields = [f for f in fields if f not in self._csv_fields]
            if new_fields:
                self._csv_fields.extend(new_fields)
                # Rewrite header by reopening (simple approach for research code)
                self._csv_file.close()
                self._csv_file = open(self.csv_path, "a", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_config(self, cfg: dict) -> None:
        """Save config as JSON in the run directory."""
        with open(self.run_dir / "config_logged.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    def close(self) -> None:
        """Flush and close all writers."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self._csv_file is not None:
            self._csv_file.close()

    def __del__(self):
        self.close()
