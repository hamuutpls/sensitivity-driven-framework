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


def _code(text):
    """Code lines without comments or blank lines."""
    return [line.split("  #")[0].rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_the_notebook_makes_the_vllm_venv_installs_the_extras_and_matches_the_readme():
    # the README's "On Google Colab" steps must be what the notebook does (2026-10-29: "the code must be
    # reproduceable by the next user")
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]]
    readme = (NOTEBOOK.parent.parent / "README.md").read_text(encoding="utf-8")
    install = next(c for c in cells if "pip install -q -e" in c)
    assert '[dev,search,downstream]" bitsandbytes' in install and '[dev,search,downstream]" bitsandbytes' in readme
    venv = next(c for c in cells if c.startswith("# 2b."))
    for cmd in ("uv venv /content/vllm-venv --python 3.12", "vllm==0.31.0", "pip install -q uv"):
        assert cmd in venv and cmd in readme
    assert "-m pytest -q -rs tests" in next(c for c in cells if c.startswith("# 3.")) and "-m pytest -q -rs tests" in readme
    step4 = _code(next(c for c in cells if c.startswith("# 4.")))
    block = readme.split("4. Sets the run:", 1)[1].split("```python", 1)[1].split("```", 1)[0]
    shown = _code(block)
    assert shown and all(line.strip() in [s.strip() for s in step4] for line in shown), shown
    assert {line.split("=")[0].strip() for line in step4 if line.startswith("main.")} == \
        {line.split("=")[0].strip() for line in shown}
