"""
Git information collector for reproducibility manifests.

Records commit hash, branch, dirty state, remote URL, and recent log.

Usage:
    from src.utils.git_info import collect_git_info
    info = collect_git_info()
"""

from __future__ import annotations

import subprocess
from typing import Any


def _run_git(args: list[str], timeout: int = 5) -> str:
    """Run a git command and return stdout, or '' on failure."""
    try:
        r = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def collect_git_info() -> dict[str, Any]:
    """
    Collect git repository metadata.

    Returns dict with:
        commit, branch, dirty, remote_url, diff_stat, last_5_commits
    """
    commit = _run_git(["rev-parse", "HEAD"]) or "unknown"
    short = _run_git(["rev-parse", "--short", "HEAD"]) or "unknown"
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    diff_stat = _run_git(["diff", "--stat"])
    dirty = bool(diff_stat)

    remote_url = _run_git(["remote", "get-url", "origin"]) or "none"

    log_lines = _run_git(["log", "--oneline", "-5"]).split("\n") if _run_git(["log", "--oneline", "-5"]) else []

    return {
        "commit": commit,
        "commit_short": short,
        "branch": branch,
        "dirty": dirty,
        "diff_stat": diff_stat,
        "remote_url": remote_url,
        "last_5_commits": log_lines,
    }
