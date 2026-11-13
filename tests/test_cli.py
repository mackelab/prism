import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from prism import cli


def test_cli_prints_help_without_command(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prism"])
    cli.main()
    assert "prism <command>" in capsys.readouterr().out


def test_cli_rejects_unknown_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prism", "unknown"])
    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_cli_sets_native_log_policy_before_jax_imports():
    env = os.environ.copy()
    env.pop("PRISM_SHOW_NATIVE_LOGS", None)
    env.pop("TF_CPP_MIN_LOG_LEVEL", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; import prism.cli; "
                "print(os.environ['TF_CPP_MIN_LOG_LEVEL']); "
                "print('jax' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.splitlines() == ["3", "False"]


def test_native_logs_have_an_explicit_escape_hatch():
    env = os.environ.copy()
    env["PRISM_SHOW_NATIVE_LOGS"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import prism.cli; print(os.environ['TF_CPP_MIN_LOG_LEVEL'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "1"


SCRIPT_MODULES = (
    "prism.dmri.predict",
    "prism.dmri.eval.eval_script",
    "prism.train.train_script",
)


@pytest.mark.parametrize("module", SCRIPT_MODULES)
def test_script_modules_run_under_dash_m(module):
    """`python -m <module>` must do something, not exit 0 in silence.

    Without an `if __name__ == "__main__"` guard these modules import cleanly,
    run nothing, and return 0 -- so a mistyped invocation looks like a
    successful run that produced no output. Checked in the source rather than by
    launching a subprocess per module, which would import JAX three times.
    """
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    assert '__name__ == "__main__"' in source, (
        f"{module} has no __main__ guard, so `python -m {module}` is a silent no-op"
    )


def test_prism_is_executable_as_a_module():
    """`python -m prism` needs a __main__.py; the package alone cannot run."""
    result = subprocess.run(
        [sys.executable, "-m", "prism", "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr
    assert "prism <command>" in result.stdout, (
        f"no usage text; stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def _hydra_config_paths():
    package = Path(cli.__file__).resolve().parent
    pattern = re.compile(r"hydra\.main\(\s*config_path=\"([^\"]+)\"")
    for path in sorted(package.rglob("*.py")):
        for config_path in pattern.findall(path.read_text()):
            yield path, config_path


@pytest.mark.parametrize(
    ("module", "config_path"),
    list(_hydra_config_paths()),
    ids=lambda value: getattr(value, "name", value),
)
def test_hydra_entry_points_find_the_packaged_config_tree(module, config_path):
    # Moving prism/eval to prism/dmri/eval left config_path one level short,
    # so `prism eval` and `prism predict` looked for a missing prism.conf.
    conf = Path(cli.__file__).resolve().parents[1] / "conf"
    assert (conf / "__init__.py").is_file()
    assert (module.parent / config_path).resolve() == conf
