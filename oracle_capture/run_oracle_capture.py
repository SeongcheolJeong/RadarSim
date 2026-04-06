"""CLI wrapper for the RadarSimPy oracle capture suite."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

from oracle_capture.capture_suite import run_capture


DEFAULT_MODULE_NAMES = [
    "radarsimpy",
    "radarsimpy_macos_arm",
    "radarsimpy_macos",
    "radarsimpy_origin",
]


def _import_module(module_names: list[str]):
    last_error = None
    for module_name in module_names:
        try:
            return module_name, importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - import behavior depends on host
            last_error = exc
    raise last_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture RadarSimPy oracle artifacts on a supported host."
    )
    parser.add_argument(
        "--module-name",
        action="append",
        dest="module_names",
        default=[],
        help="Module import name to try. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="oracle_capture_output",
        help="Directory where manifest.json and artifacts will be written.",
    )
    args = parser.parse_args(argv)

    module_names = args.module_names or DEFAULT_MODULE_NAMES
    module_name, module = _import_module(module_names)
    manifest = run_capture(module, Path(args.output_dir), module_name=module_name)
    captured_scenarios = manifest.get("scenario_index", {})

    print(f"Capture complete for {module_name}")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print(f"Captured version: {manifest['metadata']['version']}")
    print(f"Captured scenarios: {len(captured_scenarios)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv[1:]))
