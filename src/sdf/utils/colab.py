"""Where the Colab notebook and scripts keep results, and the one archive the host PC collects them from."""

from __future__ import annotations

import os
import tarfile

LOCAL_ROOT = "D:/Thesis/thesis_compression"  # a local runtime on the PC: what main.py writes when run from D:\Thesis
COLAB_ROOT = "/content/thesis_compression"  # Colab's own disk, deleted when the session ends
ARCHIVE = "/content/thesis_results.tgz"  # fixed name, so the host PC session always knows what to copy to D:\Thesis


def results_root(local: str = LOCAL_ROOT, colab: str = COLAB_ROOT) -> str:
    """D:\\Thesis when the notebook runs on the PC (a local runtime), else Colab's disk."""
    return local if os.path.isdir(os.path.dirname(local)) else colab


def pack_results(root: str, archive: str = ARCHIVE, local: str = LOCAL_ROOT) -> str | None:
    """Pack root/results into one .tgz; None when root is already on D:\\Thesis (nothing to bring back).

    Written under a temporary name and renamed, so a copy taken while it is being written is never half a file.
    """
    if os.path.abspath(root) == os.path.abspath(local):
        return None
    tmp = archive + ".part"
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(os.path.join(root, "results"), arcname="results")
    os.replace(tmp, archive)
    return archive
