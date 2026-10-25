"""Tests for the package layout: each subpackage imports only what its
layer allows, so the core stays free of I/O and no package needs a cloud
account.

Imports are read from the source with ``ast``, not by importing, so a
forbidden import is caught even in a branch no other test runs.

Adapted from collar-telemetry @ v0.1.0 (tests/test_layout.py).
"""

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "vessels"
CLOUD = PACKAGE.parent / "cloud"
# Top-level names of cloud SDKs. Google's clients import as
# ``google.cloud``, kept to one part (``google``) by ``imports``.
CLOUD_SDKS = {"boto3", "botocore", "aiobotocore", "moto", "awslambdaric", "google"}
PIPELINE = {"polars", "numpy", "shapely", "scipy", "h3"}

# What each layer may import beyond the standard library: third-party
# top-level names, and the vessels subpackages it may use. Widen an entry
# only with a reason; the core never gains file or network libraries.
ALLOWED = {
    "core": (PIPELINE, {"core"}),
    "io": ({"polars", "pyarrow", "shapely"}, {"core", "io"}),
    # shapely places planted towing runs inside closure polygons.
    "sim": ({"polars", "numpy", "shapely"}, {"core", "sim"}),
    "stream": ({"polars"}, {"core", "stream"}),
    "evaluation": (
        PIPELINE | {"duckdb", "geopandas", "matplotlib", "movingpandas"},
        {"core", "io", "sim", "stream", "evaluation"},
    ),
}

# The service runs the core and the stream wrapper, nothing above them,
# and no cloud SDK: it runs in a container on any machine.
CLOUD_ALLOWED = (
    PIPELINE
    | {"vessels.core", "vessels.stream"}
    | {f"cloud.{p.stem}" for p in CLOUD.glob("*.py")}
)


def imports(path):
    """List the top-level names a module imports.

    Args:
        path: Path to a Python source file.

    Returns:
        A set of imported module names, with ``vessels.<sub>`` and
        ``cloud.<module>`` kept to two parts and everything else to one.
    """
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules = [node.module]
        else:
            continue
        for module in modules:
            parts = module.split(".")
            keep = 2 if parts[0] in ("vessels", "cloud") else 1
            names.add(".".join(parts[:keep]))
    return names


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_imports_only_what_it_may(layer):
    """No module imports a layer it may not use or an unexpected library."""
    third_party, internal = ALLOWED[layer]
    allowed = third_party | {f"vessels.{sub}" for sub in internal}
    modules = sorted((PACKAGE / layer).rglob("*.py"))
    assert modules
    for module in modules:
        extra = {
            name
            for name in imports(module) - allowed
            if name not in sys.stdlib_module_names
        }
        assert not extra, f"{module.relative_to(PACKAGE)} imports {sorted(extra)}"


def test_every_module_belongs_to_a_layer():
    """A module dropped at the top of the package would escape the layer checks."""
    loose = [p.name for p in PACKAGE.glob("*.py") if p.name != "__init__.py"]
    assert not loose
    assert {
        p.name for p in PACKAGE.iterdir() if p.is_dir() and p.name != "__pycache__"
    } == set(ALLOWED)


def test_vessels_package_imports_no_cloud_sdk():
    """Every result reproduces without a cloud account."""
    for module in sorted(PACKAGE.rglob("*.py")):
        found = {name.split(".")[0] for name in imports(module)} & CLOUD_SDKS
        assert not found, f"{module.relative_to(PACKAGE)} imports {sorted(found)}"


@pytest.mark.parametrize("module", sorted(p.stem for p in CLOUD.glob("*.py")))
def test_cloud_modules_import_only_what_they_may(module):
    """The service carries only the core and stream wrapper, not evaluation code."""
    extra = {
        name
        for name in imports(CLOUD / f"{module}.py") - CLOUD_ALLOWED
        if name.split(".")[0] not in sys.stdlib_module_names
    }
    assert not extra, f"cloud/{module}.py imports {sorted(extra)}"
