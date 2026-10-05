"""Environment capture, recorded in every report so comparisons are auditable."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from importlib import metadata
from typing import Any

_PACKAGES = ("torch", "transformers", "datasets", "numpy", "openpyxl", "optuna", "pymoo")


def _git() -> dict[str, Any]:
    """Code version of this package's checkout (not the run folder): commit and whether uncommitted changes exist."""
    here = Path(__file__).resolve().parent
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=here, capture_output=True, text=True, timeout=10,  # noqa: E731
                                        check=True).stdout
        return {"git_commit": run("rev-parse", "--short", "HEAD").strip(), "git_dirty": bool(run("status", "--porcelain").strip())}
    except (OSError, subprocess.SubprocessError):
        return {}


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), **_git()}
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
