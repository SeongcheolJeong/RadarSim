"""CLI wrapper for comparing a candidate module against a captured oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from oracle_capture.compare_suite import compare_against_oracle
from oracle_capture.run_oracle_capture import _import_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a RadarSimPy-like module against a captured oracle bundle."
    )
    parser.add_argument(
        "--module-name",
        action="append",
        dest="module_names",
        default=[],
        help="Candidate module import name to try. May be passed multiple times.",
    )
    parser.add_argument(
        "--oracle-dir",
        default="oracle_capture_output/macos_arm_py311",
        help="Directory containing a previously captured oracle manifest and artifacts.",
    )
    parser.add_argument(
        "--output-path",
        default="oracle_capture_output/macos_arm_py311/whitebox_compare_report.json",
        help="Path where the comparison report JSON will be written.",
    )
    args = parser.parse_args(argv)

    module_names = args.module_names or ["radarsimpy_whitebox"]
    module_name, module = _import_module(module_names)
    report = compare_against_oracle(
        module,
        Path(args.oracle_dir),
        candidate_module_name=module_name,
        output_path=Path(args.output_path),
    )

    print(f"Oracle module: {report['oracle_module_name']}")
    print(f"Candidate module: {report['candidate_module_name']}")
    print(f"Overall match: {report['overall']}")
    for section_name, section in report["comparisons"].items():
        print(f"{section_name}: {section['overall']}")
    matched_scenarios = sum(
        1 for scenario in report.get("scenario_results", {}).values() if scenario.get("status") == "matched"
    )
    print(f"Matched scenarios: {matched_scenarios}/{len(report.get('scenario_results', {}))}")
    print(f"Report path: {Path(args.output_path).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv[1:]))
