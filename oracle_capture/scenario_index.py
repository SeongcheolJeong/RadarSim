"""Scenario-index helpers for oracle capture and compare workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any


CAPTURED_SCENARIOS: dict[str, dict[str, Any]] = {
    "SIM-PT-001": {
        "family": "sim_radar",
        "title": "Minimal point-target sim_radar contract",
        "capture_ref": "captures.sim_radar.point_target",
        "compare_ref": "comparisons.sim_radar.point_target.overall",
        "artifact_paths": ["artifacts/sim_radar_point_target.npz"],
    },
    "SIM-PT-002": {
        "family": "sim_radar",
        "title": "MIMO ordering for point-target sim_radar",
        "capture_ref": "captures.sim_radar.point_mimo",
        "compare_ref": "comparisons.sim_radar.point_mimo.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo.npz"],
    },
    "SIM-PT-003": {
        "family": "sim_radar",
        "title": "Seed and noise behavior for sim_radar",
        "capture_ref": "captures.sim_radar.seed_repeat",
        "compare_ref": "comparisons.sim_radar.seed_repeat.overall",
        "artifact_paths": [],
    },
    "SIM-PT-004": {
        "family": "sim_radar",
        "title": "Two-point-target range-profile separation",
        "capture_ref": "captures.sim_radar.point_multi",
        "compare_ref": "comparisons.sim_radar.point_multi.overall",
        "artifact_paths": ["artifacts/sim_radar_point_multi.npz"],
    },
    "SIM-PT-005": {
        "family": "sim_radar",
        "title": "MIMO multi-frame flattening for point-target sim_radar",
        "capture_ref": "captures.sim_radar.point_mimo_multiframe",
        "compare_ref": "comparisons.sim_radar.point_mimo_multiframe.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_multiframe.npz"],
    },
    "SIM-PT-006": {
        "family": "sim_radar",
        "title": "Point-target phase rotation for sim_radar",
        "capture_ref": "captures.sim_radar.point_phase",
        "compare_ref": "comparisons.sim_radar.point_phase.overall",
        "artifact_paths": ["artifacts/sim_radar_point_phase.npz"],
    },
    "SIM-PT-007": {
        "family": "sim_radar",
        "title": "Single-frame moving point-target Doppler response",
        "capture_ref": "captures.sim_radar.point_moving_doppler",
        "compare_ref": "comparisons.sim_radar.point_moving_doppler.overall",
        "artifact_paths": ["artifacts/sim_radar_point_moving_doppler.npz"],
    },
    "SIM-PT-008": {
        "family": "sim_radar",
        "title": "Off-axis MIMO point-target spatial phase",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_offaxis.npz"],
    },
    "SIM-PT-009": {
        "family": "sim_radar",
        "title": "Off-axis moving MIMO point-target Doppler and spatial phase",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_moving",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_moving.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_offaxis_moving.npz"],
    },
    "SIM-PT-010": {
        "family": "sim_radar",
        "title": "Off-axis moving MIMO multi-frame point-target contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_moving_multiframe",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_moving_multiframe.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_offaxis_moving_multiframe.npz"],
    },
    "SIM-PT-011": {
        "family": "sim_radar",
        "title": "Negative-radial-speed point-target Doppler direction",
        "capture_ref": "captures.sim_radar.point_moving_doppler_negative",
        "compare_ref": "comparisons.sim_radar.point_moving_doppler_negative.overall",
        "artifact_paths": ["artifacts/sim_radar_point_moving_doppler_negative.npz"],
    },
    "SIM-PT-012": {
        "family": "sim_radar",
        "title": "Off-axis negative-speed MIMO point-target Doppler and spatial phase",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_moving_negative",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_moving_negative.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_offaxis_moving_negative.npz"],
    },
    "SIM-PT-013": {
        "family": "sim_radar",
        "title": "Off-axis negative-speed MIMO multi-frame point-target contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_moving_negative_multiframe",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_moving_negative_multiframe.overall",
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_moving_negative_multiframe.npz"
        ],
    },
    "SIM-PT-014": {
        "family": "sim_radar",
        "title": "Static-plus-moving point-target range-Doppler separation",
        "capture_ref": "captures.sim_radar.point_static_moving",
        "compare_ref": "comparisons.sim_radar.point_static_moving.overall",
        "artifact_paths": ["artifacts/sim_radar_point_static_moving.npz"],
    },
    "SIM-PT-015": {
        "family": "sim_radar",
        "title": "Static-plus-negative-moving point-target range-Doppler separation",
        "capture_ref": "captures.sim_radar.point_static_moving_negative",
        "compare_ref": "comparisons.sim_radar.point_static_moving_negative.overall",
        "artifact_paths": ["artifacts/sim_radar_point_static_moving_negative.npz"],
    },
    "SIM-PT-016": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-moving point-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_moving",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_moving.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_offaxis_static_moving.npz"],
    },
    "SIM-PT-017": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-negative-moving point-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_moving_negative",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_moving_negative.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_offaxis_static_moving_negative.npz"],
    },
    "SIM-PT-018": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-moving multi-frame point-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_moving_multiframe",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_moving_multiframe.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_offaxis_static_moving_multiframe.npz"],
    },
    "SIM-PT-019": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-negative-moving multi-frame point-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_moving_negative_multiframe",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_moving_negative_multiframe.overall",
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_moving_negative_multiframe.npz"
        ],
    },
    "SIM-PT-020": {
        "family": "sim_radar",
        "title": "Static-plus-static-plus-moving point-target range-Doppler separation",
        "capture_ref": "captures.sim_radar.point_static_static_moving",
        "compare_ref": "comparisons.sim_radar.point_static_static_moving.overall",
        "artifact_paths": ["artifacts/sim_radar_point_static_static_moving.npz"],
    },
    "SIM-PT-021": {
        "family": "sim_radar",
        "title": "Static-plus-static-plus-negative-moving point-target range-Doppler separation",
        "capture_ref": "captures.sim_radar.point_static_static_moving_negative",
        "compare_ref": "comparisons.sim_radar.point_static_static_moving_negative.overall",
        "artifact_paths": ["artifacts/sim_radar_point_static_static_moving_negative.npz"],
    },
    "SIM-PT-022": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-moving point-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_static_moving",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_static_moving.overall",
        "artifact_paths": ["artifacts/sim_radar_point_mimo_offaxis_static_static_moving.npz"],
    },
    "SIM-PT-023": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-negative-moving point-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_static_moving_negative",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_static_moving_negative.overall",
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_static_moving_negative.npz"
        ],
    },
    "SIM-PT-024": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-moving multi-frame point-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_static_moving_multiframe",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_static_moving_multiframe.overall",
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_static_moving_multiframe.npz"
        ],
    },
    "SIM-PT-025": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-negative-moving multi-frame point-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_static_moving_negative_multiframe",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_static_moving_negative_multiframe.overall",
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_static_moving_negative_multiframe.npz"
        ],
    },
    "SIM-PT-026": {
        "family": "sim_radar",
        "title": "Static-plus-static-plus-moving-plus-moving four-target separation",
        "capture_ref": "captures.sim_radar.point_static_static_moving_moving",
        "compare_ref": "comparisons.sim_radar.point_static_static_moving_moving.overall",
        "artifact_paths": ["artifacts/sim_radar_point_static_static_moving_moving.npz"],
    },
    "SIM-PT-027": {
        "family": "sim_radar",
        "title": "Static-plus-static-plus-negative-moving-plus-negative-moving four-target separation",
        "capture_ref": "captures.sim_radar.point_static_static_moving_moving_negative",
        "compare_ref": "comparisons.sim_radar.point_static_static_moving_moving_negative.overall",
        "artifact_paths": [
            "artifacts/sim_radar_point_static_static_moving_moving_negative.npz"
        ],
    },
    "SIM-PT-028": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-moving-plus-moving four-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_static_moving_moving",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_static_moving_moving.overall",
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_static_moving_moving.npz"
        ],
    },
    "SIM-PT-029": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-moving-plus-moving multi-frame four-target joint contract",
        "capture_ref": (
            "captures.sim_radar.point_mimo_offaxis_static_static_moving_moving_multiframe"
        ),
        "compare_ref": (
            "comparisons.sim_radar.point_mimo_offaxis_static_static_moving_moving_multiframe.overall"
        ),
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_static_moving_moving_multiframe.npz"
        ],
    },
    "SIM-PT-030": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-mixed-sign-moving four-target joint contract",
        "capture_ref": "captures.sim_radar.point_mimo_offaxis_static_static_mixed_sign",
        "compare_ref": "comparisons.sim_radar.point_mimo_offaxis_static_static_mixed_sign.overall",
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_static_mixed_sign.npz"
        ],
    },
    "SIM-PT-031": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-mixed-sign-moving multi-frame four-target joint contract",
        "capture_ref": (
            "captures.sim_radar.point_mimo_offaxis_static_static_mixed_sign_multiframe"
        ),
        "compare_ref": (
            "comparisons.sim_radar.point_mimo_offaxis_static_static_mixed_sign_multiframe.overall"
        ),
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_static_mixed_sign_multiframe.npz"
        ],
    },
    "SIM-PT-032": {
        "family": "sim_radar",
        "title": "Off-axis MIMO static-plus-static-plus-negative-moving-plus-negative-moving multi-frame four-target accepted-band contract",
        "capture_ref": (
            "captures.sim_radar.point_mimo_offaxis_static_static_moving_moving_negative_multiframe"
        ),
        "compare_ref": (
            "comparisons.sim_radar.point_mimo_offaxis_static_static_moving_moving_negative_multiframe.overall"
        ),
        "artifact_paths": [
            "artifacts/sim_radar_point_mimo_offaxis_static_static_moving_moving_negative_multiframe.npz"
        ],
    },
    "SIM-MESH-001": {
        "family": "sim_radar",
        "title": "Static mesh-target sim_radar contract",
        "capture_ref": "captures.sim_radar.mesh_target",
        "compare_ref": "comparisons.sim_radar.mesh_target.overall",
        "artifact_paths": ["artifacts/sim_radar_mesh_target.npz"],
    },
    "SIM-MESH-002": {
        "family": "sim_radar",
        "title": "Mesh aspect-sensitivity trend in sim_radar",
        "capture_ref": "captures.sim_radar.mesh_aspect",
        "compare_ref": "comparisons.sim_radar.mesh_aspect.overall",
        "artifact_paths": [
            "artifacts/sim_radar_mesh_broadside.npz",
            "artifacts/sim_radar_mesh_edge_on.npz",
        ],
    },
    "LIDAR-001": {
        "family": "sim_lidar",
        "title": "Baseline LiDAR point-cloud contract",
        "capture_ref": "captures.sim_lidar.baseline",
        "compare_ref": "comparisons.sim_lidar.baseline.overall",
        "artifact_paths": ["artifacts/sim_lidar_hits.npy"],
    },
    "LIDAR-002": {
        "family": "sim_lidar",
        "title": "Frame-time LiDAR motion behavior",
        "capture_ref": "captures.sim_lidar.moving",
        "compare_ref": "comparisons.sim_lidar.moving.overall",
        "artifact_paths": ["artifacts/sim_lidar_moving_hits.npy"],
    },
    "LIDAR-003": {
        "family": "sim_lidar",
        "title": "Multi-hit LiDAR ordering and reflected directions",
        "capture_ref": "captures.sim_lidar.multi_hit",
        "compare_ref": "comparisons.sim_lidar.multi_hit.overall",
        "artifact_paths": ["artifacts/sim_lidar_multi_hits.npy"],
    },
    "RCS-001": {
        "family": "sim_rcs",
        "title": "Single-object RCS baseline at broadside",
        "capture_ref": "captures.sim_rcs.broadside",
        "compare_ref": "comparisons.sim_rcs.cases.broadside",
        "artifact_paths": [],
    },
    "RCS-002": {
        "family": "sim_rcs",
        "title": "Angle-sweep RCS main-lobe shape",
        "capture_ref": "captures.sim_rcs.obs_phi_sweep",
        "compare_ref": "comparisons.sim_rcs.sweep.overall",
        "artifact_paths": ["artifacts/sim_rcs_obs_phi_sweep.npz"],
    },
    "RCS-003": {
        "family": "sim_rcs",
        "title": "Polarization discrimination at broadside",
        "capture_ref": "captures.sim_rcs.polarization_cases",
        "compare_ref": "comparisons.sim_rcs.polarization.overall",
        "artifact_paths": ["artifacts/sim_rcs_polarization_cases.npz"],
    },
}


def _get_path(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def build_capture_scenario_index(
    manifest: dict[str, Any],
    output_dir: Path | str,
) -> dict[str, dict[str, Any]]:
    output_dir = Path(output_dir)
    index: dict[str, dict[str, Any]] = {}
    for scenario_id, spec in CAPTURED_SCENARIOS.items():
        capture_entry = _get_path(manifest, spec["capture_ref"])
        artifact_paths = list(spec["artifact_paths"])
        artifact_presence = {
            relative_path: (output_dir / relative_path).exists() for relative_path in artifact_paths
        }
        if isinstance(capture_entry, dict) and "ok" in capture_entry:
            capture_ok = bool(capture_entry["ok"])
        else:
            capture_ok = capture_entry is not None
        index[scenario_id] = {
            "family": spec["family"],
            "title": spec["title"],
            "capture_ref": spec["capture_ref"],
            "artifact_paths": artifact_paths,
            "artifacts_present": artifact_presence,
            "capture_ok": capture_ok,
            "status": "captured" if capture_ok else "capture_failed",
        }
    return index


def build_compare_scenario_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for scenario_id, spec in CAPTURED_SCENARIOS.items():
        compare_ref = spec["compare_ref"]
        comparison_entry = _get_path(report, compare_ref)
        parent_ref = compare_ref.rsplit(".", 1)[0] if "." in compare_ref else compare_ref
        comparison_parent = _get_path(report, parent_ref)

        if isinstance(comparison_parent, dict) and "available" in comparison_parent:
            available = bool(comparison_parent["available"])
            overall = comparison_parent.get("overall") if available else None
        elif isinstance(comparison_entry, dict):
            available = True
            overall = comparison_entry.get("overall")
        else:
            available = comparison_entry is not None
            overall = comparison_entry
        results[scenario_id] = {
            "family": spec["family"],
            "title": spec["title"],
            "compare_ref": compare_ref,
            "available": available,
            "overall": bool(overall) if available else None,
            "status": (
                "matched"
                if available and bool(overall)
                else "mismatch"
                if available
                else "not_compared"
            ),
        }
    return results
