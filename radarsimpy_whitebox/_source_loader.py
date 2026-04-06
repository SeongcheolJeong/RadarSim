"""Helpers for loading readable source modules from the origin snapshot.

The packaged `radarsimpy_origin` directory cannot be imported as a normal Python
package on this macOS host because its package root eagerly imports Windows-only
binary extensions. For phase 1 we load the readable source files under a private
package namespace instead.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys
import types

_SOURCE_PACKAGE = "_radarsimpy_whitebox_origin"
_SUPPORTED_MODULES = frozenset({"processing", "tools"})


def _origin_root() -> Path:
    env_override = os.environ.get("RADARSIMPY_ORIGIN_DIR")
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
    else:
        candidate = Path(__file__).resolve().parents[1] / "radarsimpy_origin"

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Readable origin source directory not found: {candidate}"
        )

    return candidate


def _ensure_source_package() -> str:
    if _SOURCE_PACKAGE not in sys.modules:
        package = types.ModuleType(_SOURCE_PACKAGE)
        package.__path__ = [str(_origin_root())]
        sys.modules[_SOURCE_PACKAGE] = package

    return _SOURCE_PACKAGE


@lru_cache(maxsize=None)
def load_origin_module(module_name: str):
    """Load a readable origin source module under a private namespace."""

    if module_name not in _SUPPORTED_MODULES:
        raise ValueError(
            f"Unsupported origin source module '{module_name}'. "
            f"Supported modules: {', '.join(sorted(_SUPPORTED_MODULES))}"
        )

    package_name = _ensure_source_package()

    # `processing.py` uses a relative import from `.tools`, so load tools first.
    if module_name == "processing":
        load_origin_module("tools")

    qualified_name = f"{package_name}.{module_name}"
    if qualified_name in sys.modules:
        return sys.modules[qualified_name]

    module_path = _origin_root() / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module
