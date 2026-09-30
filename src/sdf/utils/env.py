"""Environment capture, recorded in every report so comparisons are auditable."""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any

_PACKAGES = ("torch", "transformers", "datasets", "numpy", "openpyxl", "optuna", "pymoo")


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in _PACKAGES:
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda"] = torch.version.cuda
        else:
            info["gpu"] = "none (CPU)"
    except ImportError:
        pass
    return info


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
