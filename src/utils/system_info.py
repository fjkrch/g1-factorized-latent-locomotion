"""
System information collector for reproducibility manifests.

Records OS, CPU, GPU, CUDA, Python, PyTorch, Isaac Lab/Sim versions,
and hardware details into a single dict for manifest embedding.

Usage:
    from src.utils.system_info import collect_system_info
    info = collect_system_info()
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

import torch


def _run_cmd(cmd: list[str], timeout: int = 5) -> str:
    """Run a shell command and return stripped stdout, or '' on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def get_nvidia_driver_version() -> str:
    out = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return out.split("\n")[0].strip() if out else "unknown"


def get_isaac_lab_version() -> str:
    try:
        import omni.isaac.lab  # type: ignore
        return getattr(omni.isaac.lab, "__version__", "installed-unknown")
    except ImportError:
        return "not_installed"


def get_isaac_sim_version() -> str:
    try:
        import omni.isaac.core  # type: ignore
        return getattr(omni.isaac.core, "__version__", "installed-unknown")
    except ImportError:
        return "not_installed"


def collect_system_info() -> dict[str, Any]:
    """
    Collect comprehensive system information for reproducibility.

    Returns dict with:
        hostname, os, python, torch, cuda, cudnn, gpu, driver,
        isaac_lab, isaac_sim, cpu, ram_gb, env vars
    """
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "os_release": platform.release(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_mb"] = props.total_memory // (1024 * 1024)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["num_gpus"] = torch.cuda.device_count()
        info["nvidia_driver"] = get_nvidia_driver_version()
    else:
        info["cuda_version"] = "N/A"
        info["gpu_name"] = "N/A"

    info["isaac_lab_version"] = get_isaac_lab_version()
    info["isaac_sim_version"] = get_isaac_sim_version()

    # CPU & RAM
    info["cpu_count"] = os.cpu_count()
    info["cpu_model"] = platform.processor() or _run_cmd(["lscpu"]).split("\n")[0] if platform.processor() else "unknown"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / (1024 * 1024), 1)
                    break
    except Exception:
        info["ram_gb"] = "unknown"

    # Relevant env vars
    info["env_vars"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not_set"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not_set"),
    }

    return info
