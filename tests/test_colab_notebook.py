import json
from pathlib import Path
from types import SimpleNamespace

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "colab_run.ipynb"


def test_reload_cell_drops_the_loaded_sdf_modules_first():
    # else main.main() runs the sdf code loaded before a fresh pull (audit 2026-10-19, finding 8)
    (cell,) = [c for c in json.loads(NOTEBOOK.read_text())["cells"] if "importlib.reload(main)" in "".join(c["source"])]
    lines = cell["source"]
    loop = next(i for i, line in enumerate(lines) if line.startswith("for name in") and "sdf" in line)
    assert loop < next(i for i, line in enumerate(lines) if line.startswith("importlib.reload(main)"))
    modules = dict.fromkeys(["sdf", "sdf.stages.methods", "sdfx", "main", "torch"])
    exec("".join(lines[loop:loop + 2]), {"sys": SimpleNamespace(modules=modules)})
    assert set(modules) == {"sdfx", "main", "torch"}
