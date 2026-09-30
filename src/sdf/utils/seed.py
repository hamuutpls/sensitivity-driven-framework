"""Seeding: one call fixes every RNG the pipeline touches."""

from __future__ import annotations

import os
import random

from sdf.utils.logging import get_logger

log = get_logger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Needed by cuBLAS for deterministic matmuls; must be set before the first CUDA call to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only: some GPTQ/attention kernels have no deterministic variant; log instead of crashing.
        torch.use_deterministic_algorithms(True, warn_only=True)
    log.info("seed=%d deterministic=%s", seed, deterministic)
