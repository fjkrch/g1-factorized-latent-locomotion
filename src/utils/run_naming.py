"""
Experiment run naming utilities.

Provides deterministic, parseable run IDs and path construction.

Naming convention:
    run_id   = {YYYYMMDD}_{HHMMSS}_{project}_{task}_{model}_{variant}_seed{N}
    group_id = {project}_{task}_{model}_{variant}

Usage:
    from src.utils.run_naming import make_run_id, make_group_id, parse_run_id
    rid = make_run_id(cfg)                      # "20260316_143022_dynamite_randomized_dynamite_full_seed42"
    gid = make_group_id(cfg)                    # "dynamite_randomized_dynamite_full"
    parts = parse_run_id(rid)                   # dict
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_id(
    cfg: dict,
    timestamp: str | None = None,
    variant: str = "full",
) -> str:
    """
    Build a deterministic run_id string.

    Format: {YYYYMMDD}_{HHMMSS}_{project}_{task}_{model}_{variant}_seed{N}

    Args:
        cfg: Merged config dict.
        timestamp: Override timestamp (for testing); defaults to now.
        variant: Ablation or experiment variant tag (default "full").

    Returns:
        run_id string.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    project = cfg.get("project", {}).get("name", "dynamite")
    task = cfg["task"]["name"]
    model = cfg["model"]["name"]
    seed = cfg["seed"]
    return f"{ts}_{project}_{task}_{model}_{variant}_seed{seed}"


def make_group_id(
    cfg: dict,
    variant: str = "full",
) -> str:
    """
    Build a group_id that identifies an experiment group (all seeds).

    Format: {project}_{task}_{model}_{variant}
    """
    project = cfg.get("project", {}).get("name", "dynamite")
    task = cfg["task"]["name"]
    model = cfg["model"]["name"]
    return f"{project}_{task}_{model}_{variant}"


def parse_run_id(run_id: str) -> dict[str, str]:
    """
    Parse a run_id string into its components.

    Returns dict with keys: timestamp, date, time, project, task, model, variant, seed.
    """
    pattern = r"^(\d{8})_(\d{6})_([a-zA-Z0-9]+)_([a-zA-Z0-9_]+?)_([a-zA-Z0-9]+)_([a-zA-Z0-9_]+?)_seed(\d+)$"
    m = re.match(pattern, run_id)
    if not m:
        return {"raw": run_id, "parse_error": True}
    return {
        "date": m.group(1),
        "time": m.group(2),
        "timestamp": f"{m.group(1)}_{m.group(2)}",
        "project": m.group(3),
        "task": m.group(4),
        "model": m.group(5),
        "variant": m.group(6),
        "seed": int(m.group(7)),
    }


def make_run_dir(
    cfg: dict,
    base_dir: str = "outputs",
    variant: str = "full",
) -> Path:
    """
    Build the canonical run output directory path.

    Pattern: {base_dir}/{task}/{model}_{variant}/seed_{seed}/{timestamp}/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task = cfg["task"]["name"]
    model = cfg["model"]["name"]
    seed = cfg["seed"]
    run_dir = Path(base_dir) / task / f"{model}_{variant}" / f"seed_{seed}" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def checkpoint_name(step: int) -> str:
    """Canonical checkpoint filename for a given step."""
    return f"ckpt_{step:010d}.pt"


def eval_result_name(task: str, step: int | str = "best") -> str:
    """Canonical eval result filename."""
    return f"eval_{task}_step{step}.json"


def sweep_result_name(sweep_name: str) -> str:
    """Canonical sweep result filename."""
    return f"sweep_{sweep_name}.json"


def figure_name(plot_type: str, fmt: str = "png") -> str:
    """
    Canonical figure filename.

    plot_type examples: learning_curves, eval_bars, ablation_bars,
                        sweep_push_magnitude, latent_tsne_friction
    """
    return f"fig_{plot_type}.{fmt}"


def table_name(table_type: str, fmt: str = "md") -> str:
    """
    Canonical table filename.

    table_type: main_comparison, ablation, efficiency, runtime
    fmt: md | tex | csv
    """
    return f"table_{table_type}.{fmt}"
