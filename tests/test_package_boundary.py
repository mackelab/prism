"""The method (core ``prism``) never depends on an application package.

Applications (``prism.dmri``, ``prism.symbolic``) build on the core and on
nothing from each other. Saved configs may still name application classes as
strings; those resolve at run time and are not imports.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "prism"
APPLICATIONS = ("prism.dmri", "prism.symbolic")


def _module_name(path):
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path):
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                yield ".".join([*base, node.module] if node.module else base)
            else:
                yield node.module


def _owner(module):
    return next(
        (app for app in APPLICATIONS if module == app or module.startswith(app + ".")),
        None,
    )


def _violations():
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        owner = _owner(_module_name(path))
        for imported in _imports(path):
            target = _owner(imported)
            if target is not None and target != owner:
                found.append(f"{path.relative_to(PACKAGE.parent)} imports {imported}")
    return found


def test_core_and_applications_import_only_allowed_packages():
    assert _violations() == []


@pytest.mark.parametrize("application", APPLICATIONS)
def test_application_package_exists(application):
    assert (PACKAGE.parent / application.replace(".", "/")).is_dir()


def test_importing_the_core_loads_no_application():
    core_modules = sorted(
        _module_name(path)
        for path in PACKAGE.rglob("*.py")
        if _owner(_module_name(path)) is None
    )
    script = (
        "import importlib, sys\n"
        f"for name in {core_modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(sorted(m for m in sys.modules if m.startswith({APPLICATIONS!r})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={"JAX_PLATFORMS": "cpu", "PATH": ""},
    )
    assert result.stdout.strip() == "[]"


def test_every_module_imports_first_in_a_fresh_package():
    # Import cycles only show when a module is the first prism import, as when
    # prism.simulators.config (via prism.config) failed before prism.config.
    modules = sorted(_module_name(path) for path in PACKAGE.rglob("*.py"))
    script = (
        "import importlib, sys, traceback\n"
        "failed = []\n"
        f"for name in {modules!r}:\n"
        "    for loaded in [m for m in sys.modules if m.split('.')[0] == 'prism']:\n"
        "        del sys.modules[loaded]\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception:\n"
        "        failed.append(name + ': ' + traceback.format_exc(limit=0).strip())\n"
        "print('\\n'.join(failed))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={"JAX_PLATFORMS": "cpu", "PATH": ""},
    )
    assert result.stdout.strip() == ""
