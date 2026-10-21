"""Run a serving-backend worker script (vllm_worker.py, trtllm_worker.py) in the Python that has the backend
installed, and read its JSON result."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from sdf.utils.logging import get_logger

log = get_logger(__name__)

# shortcut: one fixed limit for a whole worker run (build, perplexity, timing), not configurable; make it an
# eval setting if a bigger model needs longer
TIMEOUT_S = 4 * 3600


def run_killing_group(cmd: list[str], timeout: float, **kw) -> subprocess.CompletedProcess:
    """subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw), but a timeout or an interrupt
    (Ctrl-C, Colab's stop button) kills the program's whole process group, not only the program: vLLM's engine
    process or TensorRT-LLM's MPI workers would otherwise keep holding graphics-card memory (on Windows, which has no
    process groups, only the program is killed)."""
    group = hasattr(os, "killpg")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=group,
                          **kw) as p:
        try:
            out, err = p.communicate(timeout=timeout)
        except BaseException:
            if group:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGKILL)
            else:
                p.kill()
            p.communicate()
            raise
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def run_worker(python: str, worker: str, args: dict[str, Any], tmp: Path, error: type[Exception]) -> dict[str, Any]:
    """Write `args` to tmp/args.json, run `python <worker next to this file> args.json`, and return the JSON dict
    on the worker's last output line. A failed worker raises `error` naming the last line that mentions an error."""
    (tmp / "args.json").write_text(json.dumps(args))
    cmd = [python, str(Path(__file__).with_name(worker)), str(tmp / "args.json")]
    log.info("%s: %s", worker, " ".join(cmd))
    # the backends compile kernels with tools installed next to their Python (ninja); a venv's bin/ is not on PATH
    # unless the venv is activated, so put it there
    env = {**os.environ, "PATH": os.pathsep.join([str(Path(python).parent), os.environ.get("PATH", "")])}
    try:
        r = run_killing_group(cmd, TIMEOUT_S, encoding="utf-8", errors="replace", env=env)
    except subprocess.TimeoutExpired:  # a hung backend must not stall the run; the row keeps its HF numbers
        raise error(f"{worker.removesuffix('.py')} timed out after {TIMEOUT_S} s") from None
    if r.returncode:
        text = r.stderr or r.stdout
        log.warning("%s failed (%d):\n%s", worker, r.returncode, text[-4000:])
        # the cause is the last line naming an error (vLLM's "Unsupported quantization config: ..." etc.)
        cause = next((ln.strip() for ln in reversed(text.splitlines()) if "rror" in ln and "http" not in ln),
                     text.strip()[-300:])
        raise error(f"{worker.removesuffix('.py')} failed ({r.returncode}): {cause}")
    return json.loads(r.stdout.strip().splitlines()[-1])
