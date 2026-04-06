"""Oracle capture helpers for Windows-compatible RadarSimPy runtimes.

The same harness can also be exercised locally against `radarsimpy_whitebox`
to validate artifact structure before a real Windows oracle run is available.
"""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from .scenario_index import build_capture_scenario_index


PLATE_STL = """solid plate
facet normal 1 0 0
  outer loop
    vertex 0 -0.5 -0.5
    vertex 0 0.5 -0.5
    vertex 0 0.5 0.5
  endloop
endfacet
facet normal 1 0 0
  outer loop
    vertex 0 -0.5 -0.5
    vertex 0 0.5 0.5
    vertex 0 -0.5 0.5
  endloop
endfacet
endsolid plate
"""


def _safe_signature(value) -> dict[str, Any]:
    try:
        return {"available": True, "signature": str(inspect.signature(value))}
    except Exception as exc:  # pragma: no cover - defensive for binary callables
        return {
            "available": True,
            "signature": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _summarize_array(value: np.ndarray) -> dict[str, Any]:
    summary = {
        "kind": "ndarray",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    if value.dtype.names is not None:
        summary["fields"] = list(value.dtype.names)
    if np.iscomplexobj(value):
        summary["complex"] = True
    return summary


def _summarize_value(value) -> Any:
    if isinstance(value, np.ndarray):
        return _summarize_array(value)
    if isinstance(value, dict):
        return {key: _summarize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "repr": repr(value)}


def _capture_call(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        value = fn(*args, **kwargs)
        return {
            "ok": True,
            "summary": _summarize_value(value),
            "value": value,
        }
    except Exception as exc:  # pragma: no cover - exercised in fallback/error paths
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "value": None,
        }


def write_plate_stl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PLATE_STL, encoding="utf-8")
    return path


def _build_minimal_radar(rs_module, *, seed: int = 2026):
    tx = rs_module.Transmitter(
        f=[76.0e9, 76.2e9],
        t=[0.0, 40e-6],
        tx_power=13,
        pulses=4,
        prp=60e-6,
        channels=[{"location": [0.0, 0.0, 0.0]}],
    )
    rx = rs_module.Receiver(
        fs=4e6,
        noise_figure=11,
        rf_gain=0,
        load_resistor=500,
        baseband_gain=0,
        bb_type="complex",
        channels=[{"location": [0.0, 0.0, 0.0]}],
    )
    return rs_module.Radar(tx, rx, seed=seed)


def _build_mimo_radar(rs_module, *, seed: int = 2026):
    tx = rs_module.Transmitter(
        f=[76.0e9, 76.2e9],
        t=[0.0, 40e-6],
        tx_power=13,
        pulses=4,
        prp=60e-6,
        channels=[
            {"location": [0.0, 0.0, 0.0]},
            {"location": [0.0, 0.05, 0.0]},
        ],
    )
    rx = rs_module.Receiver(
        fs=4e6,
        noise_figure=11,
        rf_gain=0,
        load_resistor=500,
        baseband_gain=0,
        bb_type="complex",
        channels=[
            {"location": [0.0, 0.0, 0.0]},
            {"location": [0.0, 0.03, 0.0]},
        ],
    )
    return rs_module.Radar(tx, rx, seed=seed)


def _build_mimo_multiframe_radar(rs_module, *, seed: int = 2026):
    tx = rs_module.Transmitter(
        f=[76.0e9, 76.2e9],
        t=[0.0, 40e-6],
        tx_power=13,
        pulses=4,
        prp=60e-6,
        channels=[
            {"location": [0.0, 0.0, 0.0]},
            {"location": [0.0, 0.05, 0.0]},
        ],
    )
    rx = rs_module.Receiver(
        fs=4e6,
        noise_figure=11,
        rf_gain=0,
        load_resistor=500,
        baseband_gain=0,
        bb_type="complex",
        channels=[
            {"location": [0.0, 0.0, 0.0]},
            {"location": [0.0, 0.03, 0.0]},
        ],
    )
    return rs_module.Radar(tx, rx, frame_time=[0.0, 1e-3], seed=seed)


def _build_point_target() -> dict[str, Any]:
    return {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]}


def _build_multi_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
    ]


def _build_static_moving_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
    ]


def _build_static_moving_negative_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
    ]


def _build_static_static_moving_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
    ]


def _build_static_static_moving_negative_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
    ]


def _build_static_static_moving_moving_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [10.0, 0.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
        {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
    ]


def _build_static_static_moving_moving_negative_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [10.0, 0.0, 0.0], "rcs": 7.0, "speed": [-8.0, 0.0, 0.0]},
        {"location": [50.0, 0.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
    ]


def _build_mimo_offaxis_static_moving_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
    ]


def _build_mimo_offaxis_static_moving_negative_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
    ]


def _build_mimo_offaxis_static_static_moving_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
    ]


def _build_mimo_offaxis_static_static_moving_negative_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
    ]


def _build_mimo_offaxis_static_static_moving_moving_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
        {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [8.0, 0.0, 0.0]},
    ]


def _build_mimo_offaxis_static_static_mixed_sign_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [8.0, 0.0, 0.0]},
        {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
    ]


def _build_mimo_offaxis_static_static_moving_moving_negative_point_targets() -> list[dict[str, Any]]:
    return [
        {"location": [20.0, 0.0, 0.0], "rcs": 10.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [35.0, 0.0, 0.0], "rcs": 8.0, "speed": [0.0, 0.0, 0.0]},
        {"location": [10.0, 1.0, 0.0], "rcs": 7.0, "speed": [-8.0, 0.0, 0.0]},
        {"location": [50.0, 1.0, 0.0], "rcs": 10.0, "speed": [-8.0, 0.0, 0.0]},
    ]


def _build_lidar() -> dict[str, Any]:
    return {
        "position": [0.0, 0.1, 0.1],
        "phi": np.array([0.0]),
        "theta": np.array([90.0]),
    }


def _build_lidar_multi() -> dict[str, Any]:
    return {
        "position": [0.0, 0.0, 0.0],
        "phi": np.degrees(np.arctan2(np.array([-1.0, 0.0, 1.0]), 10.0)),
        "theta": np.array([90.0]),
    }


def _build_lidar_multi_targets(plate_path: Path) -> list[dict[str, Any]]:
    return [
        {"model": str(plate_path), "location": [10.0, -1.0, 0.0]},
        {"model": str(plate_path), "location": [10.0, 0.0, 0.0]},
        {"model": str(plate_path), "location": [10.0, 1.0, 0.0]},
    ]


def _first_spatial_phase_sample_index(
    baseband: np.ndarray,
    *,
    pulse_index: int = 0,
    epsilon: float = 1e-12,
) -> int:
    valid = np.all(np.abs(baseband[:, pulse_index, :]) > epsilon, axis=0)
    if np.any(valid):
        return int(np.argmax(valid))
    return int(np.argmax(np.abs(baseband[0, pulse_index])))


def _top_level_inventory(rs_module) -> dict[str, Any]:
    names = [
        "Radar",
        "Transmitter",
        "Receiver",
        "sim_radar",
        "sim_lidar",
        "sim_rcs",
        "set_license",
        "is_licensed",
        "get_license_info",
        "get_version",
        "get_info",
        "print_info",
        "check_installation",
        "hello",
    ]
    inventory = {}
    for name in names:
        value = getattr(rs_module, name, None)
        if value is None:
            inventory[name] = {"available": False}
            continue
        if callable(value):
            inventory[name] = _safe_signature(value)
        else:
            inventory[name] = {"available": True, "summary": _summarize_value(value)}
    return inventory


def _capture_license(rs_module) -> dict[str, Any]:
    results = {}
    for name in ("set_license", "is_licensed", "get_license_info"):
        value = getattr(rs_module, name, None)
        if value is None:
            results[name] = {"available": False}
            continue
        results[name] = _capture_call(value)
        results[name].pop("value", None)
    return results


def _capture_sim_radar(rs_module, artifacts_dir: Path, plate_path: Path) -> dict[str, Any]:
    radar = _build_minimal_radar(rs_module, seed=2026)
    target = _build_point_target()
    multi_targets = _build_multi_point_targets()
    static_moving_targets = _build_static_moving_point_targets()
    static_moving_negative_targets = _build_static_moving_negative_point_targets()
    static_static_moving_targets = _build_static_static_moving_point_targets()
    static_static_moving_negative_targets = _build_static_static_moving_negative_point_targets()
    static_static_moving_moving_targets = _build_static_static_moving_moving_point_targets()
    static_static_moving_moving_negative_targets = (
        _build_static_static_moving_moving_negative_point_targets()
    )
    mimo_offaxis_static_moving_targets = _build_mimo_offaxis_static_moving_point_targets()
    mimo_offaxis_static_moving_negative_targets = (
        _build_mimo_offaxis_static_moving_negative_point_targets()
    )
    mimo_offaxis_static_static_moving_targets = (
        _build_mimo_offaxis_static_static_moving_point_targets()
    )
    mimo_offaxis_static_static_moving_negative_targets = (
        _build_mimo_offaxis_static_static_moving_negative_point_targets()
    )
    mimo_offaxis_static_static_moving_moving_targets = (
        _build_mimo_offaxis_static_static_moving_moving_point_targets()
    )
    mimo_offaxis_static_static_mixed_sign_targets = (
        _build_mimo_offaxis_static_static_mixed_sign_point_targets()
    )
    mimo_offaxis_static_static_moving_moving_negative_targets = (
        _build_mimo_offaxis_static_static_moving_moving_negative_point_targets()
    )
    mesh_target = {"model": str(plate_path), "location": [50.0, 0.0, 0.0]}

    point_run = _capture_call(rs_module.sim_radar, radar, [target])
    if point_run["ok"]:
        np.savez(artifacts_dir / "sim_radar_point_target.npz", **point_run["value"])

    point_mimo = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        [target],
    )
    if point_mimo["ok"]:
        np.savez(artifacts_dir / "sim_radar_point_mimo.npz", **point_mimo["value"])

    point_mimo_offaxis = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        [{**target, "location": [50.0, 1.0, 0.0]}],
    )
    point_mimo_offaxis_summary = {
        k: v for k, v in point_mimo_offaxis.items() if k != "value"
    }
    if point_mimo_offaxis["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis.npz",
            **point_mimo_offaxis["value"],
        )
        peak_sample_index = int(np.argmax(np.abs(point_mimo_offaxis["value"]["baseband"][0, 0])))
        ref = point_mimo_offaxis["value"]["baseband"][0, 0, peak_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(point_mimo_offaxis["value"]["baseband"][:, 0, peak_sample_index] / ref)
        )
        point_mimo_offaxis_summary["peak_sample_index"] = peak_sample_index
        point_mimo_offaxis_summary["channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]

    point_mimo_offaxis_moving = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        [
            {
                **target,
                "location": [50.0, 1.0, 0.0],
                "speed": [8.0, 0.0, 0.0],
            }
        ],
    )
    point_mimo_offaxis_moving_summary = {
        k: v for k, v in point_mimo_offaxis_moving.items() if k != "value"
    }
    if point_mimo_offaxis_moving["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_moving.npz",
            **point_mimo_offaxis_moving["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_mimo_offaxis_moving["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0]
        valid_range_bins = point_mimo_offaxis_moving["value"]["baseband"].shape[-1] // 2
        doppler_peak, range_peak = np.unravel_index(
            int(np.argmax(range_doppler[:, :valid_range_bins])),
            range_doppler[:, :valid_range_bins].shape,
        )
        phase_sample_index = 2
        ref = point_mimo_offaxis_moving["value"]["baseband"][0, 0, phase_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_moving["value"]["baseband"][:, 0, phase_sample_index] / ref
            )
        )
        point_mimo_offaxis_moving_summary["doppler_peak"] = int(doppler_peak)
        point_mimo_offaxis_moving_summary["range_peak"] = int(range_peak)
        point_mimo_offaxis_moving_summary["phase_sample_index"] = phase_sample_index
        point_mimo_offaxis_moving_summary["channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]

    point_mimo_offaxis_moving_negative = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        [
            {
                **target,
                "location": [50.0, 1.0, 0.0],
                "speed": [-8.0, 0.0, 0.0],
            }
        ],
    )
    point_mimo_offaxis_moving_negative_summary = {
        k: v for k, v in point_mimo_offaxis_moving_negative.items() if k != "value"
    }
    if point_mimo_offaxis_moving_negative["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_moving_negative.npz",
            **point_mimo_offaxis_moving_negative["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_mimo_offaxis_moving_negative["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0]
        valid_range_bins = point_mimo_offaxis_moving_negative["value"]["baseband"].shape[-1] // 2
        doppler_peak, range_peak = np.unravel_index(
            int(np.argmax(range_doppler[:, :valid_range_bins])),
            range_doppler[:, :valid_range_bins].shape,
        )
        phase_sample_index = 2
        ref = point_mimo_offaxis_moving_negative["value"]["baseband"][0, 0, phase_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_moving_negative["value"]["baseband"][:, 0, phase_sample_index]
                / ref
            )
        )
        point_mimo_offaxis_moving_negative_summary["doppler_peak"] = int(doppler_peak)
        point_mimo_offaxis_moving_negative_summary["range_peak"] = int(range_peak)
        point_mimo_offaxis_moving_negative_summary["phase_sample_index"] = phase_sample_index
        point_mimo_offaxis_moving_negative_summary["channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]

    point_mimo_offaxis_moving_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        [
            {
                **target,
                "location": [50.0, 1.0, 0.0],
                "speed": [8.0, 0.0, 0.0],
            }
        ],
    )
    point_mimo_offaxis_moving_multiframe_summary = {
        k: v for k, v in point_mimo_offaxis_moving_multiframe.items() if k != "value"
    }
    if point_mimo_offaxis_moving_multiframe["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_moving_multiframe.npz",
            **point_mimo_offaxis_moving_multiframe["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_mimo_offaxis_moving_multiframe["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0]
        valid_range_bins = point_mimo_offaxis_moving_multiframe["value"]["baseband"].shape[-1] // 2
        doppler_peak, range_peak = np.unravel_index(
            int(np.argmax(range_doppler[:, :valid_range_bins])),
            range_doppler[:, :valid_range_bins].shape,
        )
        phase_sample_index = 2
        ref = point_mimo_offaxis_moving_multiframe["value"]["baseband"][0, 0, phase_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_moving_multiframe["value"]["baseband"][:4, 0, phase_sample_index]
                / ref
            )
        )
        point_mimo_offaxis_moving_multiframe_summary["doppler_peak"] = int(doppler_peak)
        point_mimo_offaxis_moving_multiframe_summary["range_peak"] = int(range_peak)
        point_mimo_offaxis_moving_multiframe_summary["phase_sample_index"] = phase_sample_index
        point_mimo_offaxis_moving_multiframe_summary["frame_block_size"] = 4
        point_mimo_offaxis_moving_multiframe_summary["frame0_channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]

    point_mimo_offaxis_moving_negative_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        [
            {
                **target,
                "location": [50.0, 1.0, 0.0],
                "speed": [-8.0, 0.0, 0.0],
            }
        ],
    )
    point_mimo_offaxis_moving_negative_multiframe_summary = {
        k: v for k, v in point_mimo_offaxis_moving_negative_multiframe.items() if k != "value"
    }
    if point_mimo_offaxis_moving_negative_multiframe["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_moving_negative_multiframe.npz",
            **point_mimo_offaxis_moving_negative_multiframe["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(
                    point_mimo_offaxis_moving_negative_multiframe["value"]["baseband"], axis=-1
                ),
                axis=1,
            )
        )[0]
        valid_range_bins = (
            point_mimo_offaxis_moving_negative_multiframe["value"]["baseband"].shape[-1] // 2
        )
        doppler_peak, range_peak = np.unravel_index(
            int(np.argmax(range_doppler[:, :valid_range_bins])),
            range_doppler[:, :valid_range_bins].shape,
        )
        phase_sample_index = 2
        ref = point_mimo_offaxis_moving_negative_multiframe["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_moving_negative_multiframe["value"]["baseband"][
                    :4, 0, phase_sample_index
                ]
                / ref
            )
        )
        point_mimo_offaxis_moving_negative_multiframe_summary["doppler_peak"] = int(
            doppler_peak
        )
        point_mimo_offaxis_moving_negative_multiframe_summary["range_peak"] = int(
            range_peak
        )
        point_mimo_offaxis_moving_negative_multiframe_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_moving_negative_multiframe_summary["frame_block_size"] = 4
        point_mimo_offaxis_moving_negative_multiframe_summary["frame0_channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]

    point_mimo_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        [
            {
                "location": [50.0, 0.0, 0.0],
                "rcs": 10.0,
                "speed": [1500.0, 0.0, 0.0],
            }
        ],
    )
    if point_mimo_multiframe["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_multiframe.npz",
            **point_mimo_multiframe["value"],
        )

    point_phase = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        [{**target, "phase": 90.0}],
    )
    if point_phase["ok"]:
        np.savez(artifacts_dir / "sim_radar_point_phase.npz", **point_phase["value"])

    point_moving_doppler = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        [{**target, "speed": [8.0, 0.0, 0.0]}],
    )
    point_moving_summary = {k: v for k, v in point_moving_doppler.items() if k != "value"}
    if point_moving_doppler["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_moving_doppler.npz",
            **point_moving_doppler["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_moving_doppler["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0]
        valid_range_bins = point_moving_doppler["value"]["baseband"].shape[-1] // 2
        doppler_peak, range_peak = np.unravel_index(
            int(np.argmax(range_doppler[:, :valid_range_bins])),
            range_doppler[:, :valid_range_bins].shape,
        )
        point_moving_summary["doppler_peak"] = int(doppler_peak)
        point_moving_summary["range_peak"] = int(range_peak)

    point_moving_doppler_negative = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        [{**target, "speed": [-8.0, 0.0, 0.0]}],
    )
    point_moving_negative_summary = {
        k: v for k, v in point_moving_doppler_negative.items() if k != "value"
    }
    if point_moving_doppler_negative["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_moving_doppler_negative.npz",
            **point_moving_doppler_negative["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_moving_doppler_negative["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0]
        valid_range_bins = point_moving_doppler_negative["value"]["baseband"].shape[-1] // 2
        doppler_peak, range_peak = np.unravel_index(
            int(np.argmax(range_doppler[:, :valid_range_bins])),
            range_doppler[:, :valid_range_bins].shape,
        )
        point_moving_negative_summary["doppler_peak"] = int(doppler_peak)
        point_moving_negative_summary["range_peak"] = int(range_peak)

    point_multi = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        multi_targets,
    )
    if point_multi["ok"]:
        np.savez(artifacts_dir / "sim_radar_point_multi.npz", **point_multi["value"])

    point_static_moving = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        static_moving_targets,
    )
    point_static_moving_summary = {
        k: v for k, v in point_static_moving.items() if k != "value"
    }
    if point_static_moving["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_static_moving.npz",
            **point_static_moving["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_static_moving["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0, :, : point_static_moving["value"]["baseband"].shape[-1] // 2]
        static_window = range_doppler[0:2, 23:32]
        moving_window = range_doppler[0:4, 63:72]
        static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
        static_peak = (int(static_peak[0]), int(static_peak[1] + 23))
        moving_peak = (int(moving_peak[0]), int(moving_peak[1] + 63))
        profile = np.abs(np.fft.fft(point_static_moving["value"]["baseband"], axis=-1))[0, 0, :80]
        point_static_moving_summary["static_peak"] = [static_peak[0], static_peak[1]]
        point_static_moving_summary["moving_peak"] = [moving_peak[0], moving_peak[1]]
        point_static_moving_summary["profile_norm"] = float(np.linalg.norm(profile))

    point_static_moving_negative = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        static_moving_negative_targets,
    )
    point_static_moving_negative_summary = {
        k: v for k, v in point_static_moving_negative.items() if k != "value"
    }
    if point_static_moving_negative["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_static_moving_negative.npz",
            **point_static_moving_negative["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_static_moving_negative["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0, :, : point_static_moving_negative["value"]["baseband"].shape[-1] // 2]
        static_window = range_doppler[0:2, 23:32]
        moving_window = range_doppler[2:4, 63:72]
        static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
        static_peak = (int(static_peak[0]), int(static_peak[1] + 23))
        moving_peak = (int(moving_peak[0] + 2), int(moving_peak[1] + 63))
        profile = np.abs(
            np.fft.fft(point_static_moving_negative["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_static_moving_negative_summary["static_peak"] = [static_peak[0], static_peak[1]]
        point_static_moving_negative_summary["moving_peak"] = [moving_peak[0], moving_peak[1]]
        point_static_moving_negative_summary["profile_norm"] = float(np.linalg.norm(profile))

    point_static_static_moving = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        static_static_moving_targets,
    )
    point_static_static_moving_summary = {
        k: v for k, v in point_static_static_moving.items() if k != "value"
    }
    if point_static_static_moving["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_static_static_moving.npz",
            **point_static_static_moving["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_static_static_moving["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0, :, : point_static_static_moving["value"]["baseband"].shape[-1] // 2]
        static_a_window = range_doppler[0:2, 23:32]
        static_b_window = range_doppler[0:2, 43:52]
        moving_window = range_doppler[0:4, 63:72]
        static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
        static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
        static_a_peak = (int(static_a_peak[0]), int(static_a_peak[1] + 23))
        static_b_peak = (int(static_b_peak[0]), int(static_b_peak[1] + 43))
        moving_peak = (int(moving_peak[0]), int(moving_peak[1] + 63))
        profile = np.abs(
            np.fft.fft(point_static_static_moving["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_static_static_moving_summary["static_a_peak"] = [
            static_a_peak[0],
            static_a_peak[1],
        ]
        point_static_static_moving_summary["static_b_peak"] = [
            static_b_peak[0],
            static_b_peak[1],
        ]
        point_static_static_moving_summary["moving_peak"] = [
            moving_peak[0],
            moving_peak[1],
        ]
        point_static_static_moving_summary["profile_norm"] = float(np.linalg.norm(profile))

    point_static_static_moving_negative = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        static_static_moving_negative_targets,
    )
    point_static_static_moving_negative_summary = {
        k: v for k, v in point_static_static_moving_negative.items() if k != "value"
    }
    if point_static_static_moving_negative["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_static_static_moving_negative.npz",
            **point_static_static_moving_negative["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_static_static_moving_negative["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0, :, : point_static_static_moving_negative["value"]["baseband"].shape[-1] // 2]
        static_a_window = range_doppler[0:2, 23:32]
        static_b_window = range_doppler[0:2, 43:52]
        moving_window = range_doppler[2:4, 63:72]
        static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
        static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
        moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
        static_a_peak = (int(static_a_peak[0]), int(static_a_peak[1] + 23))
        static_b_peak = (int(static_b_peak[0]), int(static_b_peak[1] + 43))
        moving_peak = (int(moving_peak[0] + 2), int(moving_peak[1] + 63))
        profile = np.abs(
            np.fft.fft(point_static_static_moving_negative["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_static_static_moving_negative_summary["static_a_peak"] = [
            static_a_peak[0],
            static_a_peak[1],
        ]
        point_static_static_moving_negative_summary["static_b_peak"] = [
            static_b_peak[0],
            static_b_peak[1],
        ]
        point_static_static_moving_negative_summary["moving_peak"] = [
            moving_peak[0],
            moving_peak[1],
        ]
        point_static_static_moving_negative_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_static_static_moving_moving = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        static_static_moving_moving_targets,
    )
    point_static_static_moving_moving_summary = {
        k: v for k, v in point_static_static_moving_moving.items() if k != "value"
    }
    if point_static_static_moving_moving["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_static_static_moving_moving.npz",
            **point_static_static_moving_moving["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(point_static_static_moving_moving["value"]["baseband"], axis=-1),
                axis=1,
            )
        )[0, :, : point_static_static_moving_moving["value"]["baseband"].shape[-1] // 2]
        static_a_window = range_doppler[0:2, 23:32]
        static_b_window = range_doppler[0:2, 43:52]
        moving_a_window = range_doppler[0:4, 8:18]
        moving_b_window = range_doppler[0:4, 63:72]
        static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
        static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
        moving_a_peak = np.unravel_index(int(np.argmax(moving_a_window)), moving_a_window.shape)
        moving_b_peak = np.unravel_index(int(np.argmax(moving_b_window)), moving_b_window.shape)
        static_a_peak = (int(static_a_peak[0]), int(static_a_peak[1] + 23))
        static_b_peak = (int(static_b_peak[0]), int(static_b_peak[1] + 43))
        moving_a_peak = (int(moving_a_peak[0]), int(moving_a_peak[1] + 8))
        moving_b_peak = (int(moving_b_peak[0]), int(moving_b_peak[1] + 63))
        profile = np.abs(
            np.fft.fft(point_static_static_moving_moving["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_static_static_moving_moving_summary["static_a_peak"] = [
            static_a_peak[0],
            static_a_peak[1],
        ]
        point_static_static_moving_moving_summary["static_b_peak"] = [
            static_b_peak[0],
            static_b_peak[1],
        ]
        point_static_static_moving_moving_summary["moving_a_peak"] = [
            moving_a_peak[0],
            moving_a_peak[1],
        ]
        point_static_static_moving_moving_summary["moving_b_peak"] = [
            moving_b_peak[0],
            moving_b_peak[1],
        ]
        point_static_static_moving_moving_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_static_static_moving_moving_negative = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        static_static_moving_moving_negative_targets,
    )
    point_static_static_moving_moving_negative_summary = {
        k: v for k, v in point_static_static_moving_moving_negative.items() if k != "value"
    }
    if point_static_static_moving_moving_negative["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_static_static_moving_moving_negative.npz",
            **point_static_static_moving_moving_negative["value"],
        )
        range_doppler = np.abs(
            np.fft.fft(
                np.fft.fft(
                    point_static_static_moving_moving_negative["value"]["baseband"], axis=-1
                ),
                axis=1,
            )
        )[0, :, : point_static_static_moving_moving_negative["value"]["baseband"].shape[-1] // 2]
        static_a_window = range_doppler[0:2, 23:32]
        static_b_window = range_doppler[0:2, 43:52]
        moving_a_window = range_doppler[2:4, 8:18]
        moving_b_window = range_doppler[2:4, 63:72]
        static_a_peak = np.unravel_index(int(np.argmax(static_a_window)), static_a_window.shape)
        static_b_peak = np.unravel_index(int(np.argmax(static_b_window)), static_b_window.shape)
        moving_a_peak = np.unravel_index(int(np.argmax(moving_a_window)), moving_a_window.shape)
        moving_b_peak = np.unravel_index(int(np.argmax(moving_b_window)), moving_b_window.shape)
        static_a_peak = (int(static_a_peak[0]), int(static_a_peak[1] + 23))
        static_b_peak = (int(static_b_peak[0]), int(static_b_peak[1] + 43))
        moving_a_peak = (int(moving_a_peak[0] + 2), int(moving_a_peak[1] + 8))
        moving_b_peak = (int(moving_b_peak[0] + 2), int(moving_b_peak[1] + 63))
        profile = np.abs(
            np.fft.fft(point_static_static_moving_moving_negative["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_static_static_moving_moving_negative_summary["static_a_peak"] = [
            static_a_peak[0],
            static_a_peak[1],
        ]
        point_static_static_moving_moving_negative_summary["static_b_peak"] = [
            static_b_peak[0],
            static_b_peak[1],
        ]
        point_static_static_moving_moving_negative_summary["moving_a_peak"] = [
            moving_a_peak[0],
            moving_a_peak[1],
        ]
        point_static_static_moving_moving_negative_summary["moving_b_peak"] = [
            moving_b_peak[0],
            moving_b_peak[1],
        ]
        point_static_static_moving_moving_negative_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_mimo_offaxis_static_moving = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        mimo_offaxis_static_moving_targets,
    )
    point_mimo_offaxis_static_moving_summary = {
        k: v for k, v in point_mimo_offaxis_static_moving.items() if k != "value"
    }
    if point_mimo_offaxis_static_moving["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_moving.npz",
            **point_mimo_offaxis_static_moving["value"],
        )
        valid_range_bins = point_mimo_offaxis_static_moving["value"]["baseband"].shape[-1] // 2
        static_peaks = []
        moving_peaks = []
        for channel_index in range(point_mimo_offaxis_static_moving["value"]["baseband"].shape[0]):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_moving["value"]["baseband"][
                            channel_index : channel_index + 1
                        ],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_window = range_doppler[0:2, 23:32]
            moving_window = range_doppler[0:4, 63:72]
            static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_peaks.append([int(static_peak[0]), int(static_peak[1] + 23)])
            moving_peaks.append([int(moving_peak[0]), int(moving_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_moving["value"]["baseband"][0, 0, phase_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_moving["value"]["baseband"][:, 0, phase_sample_index]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(point_mimo_offaxis_static_moving["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_mimo_offaxis_static_moving_summary["static_peaks"] = static_peaks
        point_mimo_offaxis_static_moving_summary["moving_peaks"] = moving_peaks
        point_mimo_offaxis_static_moving_summary["phase_sample_index"] = phase_sample_index
        point_mimo_offaxis_static_moving_summary["channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]
        point_mimo_offaxis_static_moving_summary["profile_norm"] = float(np.linalg.norm(profile))

    point_mimo_offaxis_static_moving_negative = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        mimo_offaxis_static_moving_negative_targets,
    )
    point_mimo_offaxis_static_moving_negative_summary = {
        k: v for k, v in point_mimo_offaxis_static_moving_negative.items() if k != "value"
    }
    if point_mimo_offaxis_static_moving_negative["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_moving_negative.npz",
            **point_mimo_offaxis_static_moving_negative["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_moving_negative["value"]["baseband"].shape[-1] // 2
        )
        static_peaks = []
        moving_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_moving_negative["value"]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_moving_negative["value"]["baseband"][
                            channel_index : channel_index + 1
                        ],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_window = range_doppler[0:2, 23:32]
            moving_window = range_doppler[2:4, 63:72]
            static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_peaks.append([int(static_peak[0]), int(static_peak[1] + 23)])
            moving_peaks.append([int(moving_peak[0] + 2), int(moving_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_moving_negative["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_moving_negative["value"]["baseband"][
                    :, 0, phase_sample_index
                ]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(point_mimo_offaxis_static_moving_negative["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_mimo_offaxis_static_moving_negative_summary["static_peaks"] = static_peaks
        point_mimo_offaxis_static_moving_negative_summary["moving_peaks"] = moving_peaks
        point_mimo_offaxis_static_moving_negative_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_static_moving_negative_summary["channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]
        point_mimo_offaxis_static_moving_negative_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_mimo_offaxis_static_static_moving = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_moving_targets,
    )
    point_mimo_offaxis_static_static_moving_summary = {
        k: v for k, v in point_mimo_offaxis_static_static_moving.items() if k != "value"
    }
    if point_mimo_offaxis_static_static_moving["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving.npz",
            **point_mimo_offaxis_static_static_moving["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_moving["value"]["baseband"].shape[-1] // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_moving["value"]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_moving["value"]["baseband"][
                            channel_index : channel_index + 1
                        ],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_window = range_doppler[0:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_peaks.append([int(moving_peak[0]), int(moving_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_moving["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_moving["value"]["baseband"][
                    :, 0, phase_sample_index
                ]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(point_mimo_offaxis_static_static_moving["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_mimo_offaxis_static_static_moving_summary["static_a_peaks"] = static_a_peaks
        point_mimo_offaxis_static_static_moving_summary["static_b_peaks"] = static_b_peaks
        point_mimo_offaxis_static_static_moving_summary["moving_peaks"] = moving_peaks
        point_mimo_offaxis_static_static_moving_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_static_static_moving_summary["channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]
        point_mimo_offaxis_static_static_moving_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_mimo_offaxis_static_static_moving_negative = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_moving_negative_targets,
    )
    point_mimo_offaxis_static_static_moving_negative_summary = {
        k: v
        for k, v in point_mimo_offaxis_static_static_moving_negative.items()
        if k != "value"
    }
    if point_mimo_offaxis_static_static_moving_negative["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving_negative.npz",
            **point_mimo_offaxis_static_static_moving_negative["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_moving_negative["value"]["baseband"].shape[-1]
            // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_moving_negative["value"]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_moving_negative["value"][
                            "baseband"
                        ][channel_index : channel_index + 1],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_peaks.append([int(moving_peak[0] + 2), int(moving_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_moving_negative["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_moving_negative["value"]["baseband"][
                    :, 0, phase_sample_index
                ]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_static_moving_negative["value"]["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_static_moving_negative_summary["static_a_peaks"] = (
            static_a_peaks
        )
        point_mimo_offaxis_static_static_moving_negative_summary["static_b_peaks"] = (
            static_b_peaks
        )
        point_mimo_offaxis_static_static_moving_negative_summary["moving_peaks"] = (
            moving_peaks
        )
        point_mimo_offaxis_static_static_moving_negative_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_static_static_moving_negative_summary[
            "channel_phase_degrees"
        ] = [float(value) for value in rel_phase_deg]
        point_mimo_offaxis_static_static_moving_negative_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_mimo_offaxis_static_static_moving_moving = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_moving_moving_targets,
    )
    point_mimo_offaxis_static_static_moving_moving_summary = {
        k: v
        for k, v in point_mimo_offaxis_static_static_moving_moving.items()
        if k != "value"
    }
    if point_mimo_offaxis_static_static_moving_moving["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving_moving.npz",
            **point_mimo_offaxis_static_static_moving_moving["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_moving_moving["value"]["baseband"].shape[-1]
            // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_a_peaks = []
        moving_b_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_moving_moving["value"]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_moving_moving["value"]["baseband"][
                            channel_index : channel_index + 1
                        ],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_a_window = range_doppler[0:4, 8:18]
            moving_b_window = range_doppler[0:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_a_peak = np.unravel_index(
                int(np.argmax(moving_a_window)), moving_a_window.shape
            )
            moving_b_peak = np.unravel_index(
                int(np.argmax(moving_b_window)), moving_b_window.shape
            )
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_a_peaks.append([int(moving_a_peak[0]), int(moving_a_peak[1] + 8)])
            moving_b_peaks.append([int(moving_b_peak[0]), int(moving_b_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_moving_moving["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_moving_moving["value"]["baseband"][
                    :, 0, phase_sample_index
                ]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_static_moving_moving["value"]["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_static_moving_moving_summary["static_a_peaks"] = (
            static_a_peaks
        )
        point_mimo_offaxis_static_static_moving_moving_summary["static_b_peaks"] = (
            static_b_peaks
        )
        point_mimo_offaxis_static_static_moving_moving_summary["moving_a_peaks"] = (
            moving_a_peaks
        )
        point_mimo_offaxis_static_static_moving_moving_summary["moving_b_peaks"] = (
            moving_b_peaks
        )
        point_mimo_offaxis_static_static_moving_moving_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_static_static_moving_moving_summary["channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]
        point_mimo_offaxis_static_static_moving_moving_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_mimo_offaxis_static_static_mixed_sign = _capture_call(
        rs_module.sim_radar,
        _build_mimo_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_mixed_sign_targets,
    )
    point_mimo_offaxis_static_static_mixed_sign_summary = {
        k: v for k, v in point_mimo_offaxis_static_static_mixed_sign.items() if k != "value"
    }
    if point_mimo_offaxis_static_static_mixed_sign["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_mixed_sign.npz",
            **point_mimo_offaxis_static_static_mixed_sign["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_mixed_sign["value"]["baseband"].shape[-1] // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_pos_peaks = []
        moving_neg_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_mixed_sign["value"]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_mixed_sign["value"]["baseband"][
                            channel_index : channel_index + 1
                        ],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_pos_window = range_doppler[0:4, 8:18]
            moving_neg_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_pos_peak = np.unravel_index(
                int(np.argmax(moving_pos_window)), moving_pos_window.shape
            )
            moving_neg_peak = np.unravel_index(
                int(np.argmax(moving_neg_window)), moving_neg_window.shape
            )
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_pos_peaks.append([int(moving_pos_peak[0]), int(moving_pos_peak[1] + 8)])
            moving_neg_peaks.append([int(moving_neg_peak[0] + 2), int(moving_neg_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_mixed_sign["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_mixed_sign["value"]["baseband"][
                    :, 0, phase_sample_index
                ]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_static_mixed_sign["value"]["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_static_mixed_sign_summary["static_a_peaks"] = (
            static_a_peaks
        )
        point_mimo_offaxis_static_static_mixed_sign_summary["static_b_peaks"] = (
            static_b_peaks
        )
        point_mimo_offaxis_static_static_mixed_sign_summary["moving_pos_peaks"] = (
            moving_pos_peaks
        )
        point_mimo_offaxis_static_static_mixed_sign_summary["moving_neg_peaks"] = (
            moving_neg_peaks
        )
        point_mimo_offaxis_static_static_mixed_sign_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_static_static_mixed_sign_summary["channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]
        point_mimo_offaxis_static_static_mixed_sign_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_mimo_offaxis_static_static_mixed_sign_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_mixed_sign_targets,
    )
    point_mimo_offaxis_static_static_mixed_sign_multiframe_summary = {
        k: v
        for k, v in point_mimo_offaxis_static_static_mixed_sign_multiframe.items()
        if k != "value"
    }
    if point_mimo_offaxis_static_static_mixed_sign_multiframe["ok"]:
        np.savez(
            artifacts_dir
            / "sim_radar_point_mimo_offaxis_static_static_mixed_sign_multiframe.npz",
            **point_mimo_offaxis_static_static_mixed_sign_multiframe["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_mixed_sign_multiframe["value"][
                "baseband"
            ].shape[-1]
            // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_pos_peaks = []
        moving_neg_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_mixed_sign_multiframe["value"][
                "baseband"
            ].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "value"
                        ]["baseband"][channel_index : channel_index + 1],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_pos_window = range_doppler[0:4, 8:18]
            moving_neg_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_pos_peak = np.unravel_index(
                int(np.argmax(moving_pos_window)), moving_pos_window.shape
            )
            moving_neg_peak = np.unravel_index(
                int(np.argmax(moving_neg_window)), moving_neg_window.shape
            )
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_pos_peaks.append([int(moving_pos_peak[0]), int(moving_pos_peak[1] + 8)])
            moving_neg_peaks.append([int(moving_neg_peak[0] + 2), int(moving_neg_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_mixed_sign_multiframe["value"][
            "baseband"
        ][0, 0, phase_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_mixed_sign_multiframe["value"][
                    "baseband"
                ][:4, 0, phase_sample_index]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_static_mixed_sign_multiframe["value"][
                    "baseband"
                ],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_static_mixed_sign_multiframe_summary[
            "static_a_peaks"
        ] = static_a_peaks
        point_mimo_offaxis_static_static_mixed_sign_multiframe_summary[
            "static_b_peaks"
        ] = static_b_peaks
        point_mimo_offaxis_static_static_mixed_sign_multiframe_summary[
            "moving_pos_peaks"
        ] = moving_pos_peaks
        point_mimo_offaxis_static_static_mixed_sign_multiframe_summary[
            "moving_neg_peaks"
        ] = moving_neg_peaks
        point_mimo_offaxis_static_static_mixed_sign_multiframe_summary[
            "phase_sample_index"
        ] = phase_sample_index
        point_mimo_offaxis_static_static_mixed_sign_multiframe_summary[
            "frame_block_size"
        ] = 4
        point_mimo_offaxis_static_static_mixed_sign_multiframe_summary[
            "frame0_channel_phase_degrees"
        ] = [float(value) for value in rel_phase_deg]
        point_mimo_offaxis_static_static_mixed_sign_multiframe_summary[
            "profile_norm"
        ] = float(np.linalg.norm(profile))

    point_mimo_offaxis_static_static_moving_moving_negative_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_moving_moving_negative_targets,
    )
    point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary = {
        k: v
        for k, v in point_mimo_offaxis_static_static_moving_moving_negative_multiframe.items()
        if k != "value"
    }
    if point_mimo_offaxis_static_static_moving_moving_negative_multiframe["ok"]:
        np.savez(
            artifacts_dir
            / "sim_radar_point_mimo_offaxis_static_static_moving_moving_negative_multiframe.npz",
            **point_mimo_offaxis_static_static_moving_moving_negative_multiframe["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_moving_moving_negative_multiframe["value"][
                "baseband"
            ].shape[-1]
            // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_a_peaks = []
        moving_b_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "value"
            ]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "value"
                        ]["baseband"][channel_index : channel_index + 1],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_a_window = range_doppler[2:4, 8:18]
            moving_b_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_a_peak = np.unravel_index(
                int(np.argmax(moving_a_window)), moving_a_window.shape
            )
            moving_b_peak = np.unravel_index(
                int(np.argmax(moving_b_window)), moving_b_window.shape
            )
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_a_peaks.append([int(moving_a_peak[0] + 2), int(moving_a_peak[1] + 8)])
            moving_b_peaks.append([int(moving_b_peak[0] + 2), int(moving_b_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_moving_moving_negative_multiframe[
            "value"
        ]["baseband"][0, 0, phase_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "value"
                ]["baseband"][:4, 0, phase_sample_index]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "value"
                ]["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary[
            "static_a_peaks"
        ] = static_a_peaks
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary[
            "static_b_peaks"
        ] = static_b_peaks
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary[
            "moving_a_peaks"
        ] = moving_a_peaks
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary[
            "moving_b_peaks"
        ] = moving_b_peaks
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary[
            "phase_sample_index"
        ] = phase_sample_index
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary[
            "frame_block_size"
        ] = 4
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary[
            "frame0_channel_phase_degrees"
        ] = [float(value) for value in rel_phase_deg]
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary[
            "profile_norm"
        ] = float(np.linalg.norm(profile))

    point_mimo_offaxis_static_static_moving_moving_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_moving_moving_targets,
    )
    point_mimo_offaxis_static_static_moving_moving_multiframe_summary = {
        k: v
        for k, v in point_mimo_offaxis_static_static_moving_moving_multiframe.items()
        if k != "value"
    }
    if point_mimo_offaxis_static_static_moving_moving_multiframe["ok"]:
        np.savez(
            artifacts_dir
            / "sim_radar_point_mimo_offaxis_static_static_moving_moving_multiframe.npz",
            **point_mimo_offaxis_static_static_moving_moving_multiframe["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_moving_moving_multiframe["value"][
                "baseband"
            ].shape[-1]
            // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_a_peaks = []
        moving_b_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_moving_moving_multiframe["value"][
                "baseband"
            ].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_moving_moving_multiframe[
                            "value"
                        ]["baseband"][channel_index : channel_index + 1],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_a_window = range_doppler[0:4, 8:18]
            moving_b_window = range_doppler[0:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_a_peak = np.unravel_index(
                int(np.argmax(moving_a_window)), moving_a_window.shape
            )
            moving_b_peak = np.unravel_index(
                int(np.argmax(moving_b_window)), moving_b_window.shape
            )
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_a_peaks.append([int(moving_a_peak[0]), int(moving_a_peak[1] + 8)])
            moving_b_peaks.append([int(moving_b_peak[0]), int(moving_b_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_moving_moving_multiframe["value"][
            "baseband"
        ][0, 0, phase_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_moving_moving_multiframe["value"][
                    "baseband"
                ][:4, 0, phase_sample_index]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_static_moving_moving_multiframe["value"][
                    "baseband"
                ],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_static_moving_moving_multiframe_summary[
            "static_a_peaks"
        ] = static_a_peaks
        point_mimo_offaxis_static_static_moving_moving_multiframe_summary[
            "static_b_peaks"
        ] = static_b_peaks
        point_mimo_offaxis_static_static_moving_moving_multiframe_summary[
            "moving_a_peaks"
        ] = moving_a_peaks
        point_mimo_offaxis_static_static_moving_moving_multiframe_summary[
            "moving_b_peaks"
        ] = moving_b_peaks
        point_mimo_offaxis_static_static_moving_moving_multiframe_summary[
            "phase_sample_index"
        ] = phase_sample_index
        point_mimo_offaxis_static_static_moving_moving_multiframe_summary[
            "frame_block_size"
        ] = 4
        point_mimo_offaxis_static_static_moving_moving_multiframe_summary[
            "frame0_channel_phase_degrees"
        ] = [float(value) for value in rel_phase_deg]
        point_mimo_offaxis_static_static_moving_moving_multiframe_summary[
            "profile_norm"
        ] = float(np.linalg.norm(profile))

    point_mimo_offaxis_static_static_moving_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_moving_targets,
    )
    point_mimo_offaxis_static_static_moving_multiframe_summary = {
        k: v
        for k, v in point_mimo_offaxis_static_static_moving_multiframe.items()
        if k != "value"
    }
    if point_mimo_offaxis_static_static_moving_multiframe["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_static_moving_multiframe.npz",
            **point_mimo_offaxis_static_static_moving_multiframe["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_moving_multiframe["value"]["baseband"].shape[-1]
            // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_moving_multiframe["value"]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_moving_multiframe["value"][
                            "baseband"
                        ][channel_index : channel_index + 1],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_window = range_doppler[0:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_peaks.append([int(moving_peak[0]), int(moving_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_moving_multiframe["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_moving_multiframe["value"]["baseband"][
                    :4, 0, phase_sample_index
                ]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_static_moving_multiframe["value"]["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_static_moving_multiframe_summary["static_a_peaks"] = (
            static_a_peaks
        )
        point_mimo_offaxis_static_static_moving_multiframe_summary["static_b_peaks"] = (
            static_b_peaks
        )
        point_mimo_offaxis_static_static_moving_multiframe_summary["moving_peaks"] = (
            moving_peaks
        )
        point_mimo_offaxis_static_static_moving_multiframe_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_static_static_moving_multiframe_summary["frame_block_size"] = 4
        point_mimo_offaxis_static_static_moving_multiframe_summary[
            "frame0_channel_phase_degrees"
        ] = [float(value) for value in rel_phase_deg]
        point_mimo_offaxis_static_static_moving_multiframe_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_mimo_offaxis_static_static_moving_negative_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        mimo_offaxis_static_static_moving_negative_targets,
    )
    point_mimo_offaxis_static_static_moving_negative_multiframe_summary = {
        k: v
        for k, v in point_mimo_offaxis_static_static_moving_negative_multiframe.items()
        if k != "value"
    }
    if point_mimo_offaxis_static_static_moving_negative_multiframe["ok"]:
        np.savez(
            artifacts_dir
            / "sim_radar_point_mimo_offaxis_static_static_moving_negative_multiframe.npz",
            **point_mimo_offaxis_static_static_moving_negative_multiframe["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_static_moving_negative_multiframe["value"][
                "baseband"
            ].shape[-1]
            // 2
        )
        static_a_peaks = []
        static_b_peaks = []
        moving_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_static_moving_negative_multiframe["value"][
                "baseband"
            ].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_static_moving_negative_multiframe["value"][
                            "baseband"
                        ][channel_index : channel_index + 1],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_a_window = range_doppler[0:2, 23:32]
            static_b_window = range_doppler[0:2, 43:52]
            moving_window = range_doppler[2:4, 63:72]
            static_a_peak = np.unravel_index(
                int(np.argmax(static_a_window)), static_a_window.shape
            )
            static_b_peak = np.unravel_index(
                int(np.argmax(static_b_window)), static_b_window.shape
            )
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_a_peaks.append([int(static_a_peak[0]), int(static_a_peak[1] + 23)])
            static_b_peaks.append([int(static_b_peak[0]), int(static_b_peak[1] + 43)])
            moving_peaks.append([int(moving_peak[0] + 2), int(moving_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_static_moving_negative_multiframe["value"][
            "baseband"
        ][0, 0, phase_sample_index]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_static_moving_negative_multiframe["value"][
                    "baseband"
                ][:4, 0, phase_sample_index]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_static_moving_negative_multiframe["value"][
                    "baseband"
                ],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_static_moving_negative_multiframe_summary[
            "static_a_peaks"
        ] = static_a_peaks
        point_mimo_offaxis_static_static_moving_negative_multiframe_summary[
            "static_b_peaks"
        ] = static_b_peaks
        point_mimo_offaxis_static_static_moving_negative_multiframe_summary[
            "moving_peaks"
        ] = moving_peaks
        point_mimo_offaxis_static_static_moving_negative_multiframe_summary[
            "phase_sample_index"
        ] = phase_sample_index
        point_mimo_offaxis_static_static_moving_negative_multiframe_summary[
            "frame_block_size"
        ] = 4
        point_mimo_offaxis_static_static_moving_negative_multiframe_summary[
            "frame0_channel_phase_degrees"
        ] = [float(value) for value in rel_phase_deg]
        point_mimo_offaxis_static_static_moving_negative_multiframe_summary[
            "profile_norm"
        ] = float(np.linalg.norm(profile))

    point_mimo_offaxis_static_moving_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        mimo_offaxis_static_moving_targets,
    )
    point_mimo_offaxis_static_moving_multiframe_summary = {
        k: v for k, v in point_mimo_offaxis_static_moving_multiframe.items() if k != "value"
    }
    if point_mimo_offaxis_static_moving_multiframe["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_moving_multiframe.npz",
            **point_mimo_offaxis_static_moving_multiframe["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_moving_multiframe["value"]["baseband"].shape[-1] // 2
        )
        static_peaks = []
        moving_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_moving_multiframe["value"]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_moving_multiframe["value"]["baseband"][
                            channel_index : channel_index + 1
                        ],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_window = range_doppler[0:2, 23:32]
            moving_window = range_doppler[0:4, 63:72]
            static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_peaks.append([int(static_peak[0]), int(static_peak[1] + 23)])
            moving_peaks.append([int(moving_peak[0]), int(moving_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_moving_multiframe["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_moving_multiframe["value"]["baseband"][
                    :4, 0, phase_sample_index
                ]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(point_mimo_offaxis_static_moving_multiframe["value"]["baseband"], axis=-1)
        )[0, 0, :80]
        point_mimo_offaxis_static_moving_multiframe_summary["static_peaks"] = static_peaks
        point_mimo_offaxis_static_moving_multiframe_summary["moving_peaks"] = moving_peaks
        point_mimo_offaxis_static_moving_multiframe_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_static_moving_multiframe_summary["frame_block_size"] = 4
        point_mimo_offaxis_static_moving_multiframe_summary["frame0_channel_phase_degrees"] = [
            float(value) for value in rel_phase_deg
        ]
        point_mimo_offaxis_static_moving_multiframe_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    point_mimo_offaxis_static_moving_negative_multiframe = _capture_call(
        rs_module.sim_radar,
        _build_mimo_multiframe_radar(rs_module, seed=2026),
        mimo_offaxis_static_moving_negative_targets,
    )
    point_mimo_offaxis_static_moving_negative_multiframe_summary = {
        k: v
        for k, v in point_mimo_offaxis_static_moving_negative_multiframe.items()
        if k != "value"
    }
    if point_mimo_offaxis_static_moving_negative_multiframe["ok"]:
        np.savez(
            artifacts_dir / "sim_radar_point_mimo_offaxis_static_moving_negative_multiframe.npz",
            **point_mimo_offaxis_static_moving_negative_multiframe["value"],
        )
        valid_range_bins = (
            point_mimo_offaxis_static_moving_negative_multiframe["value"]["baseband"].shape[
                -1
            ]
            // 2
        )
        static_peaks = []
        moving_peaks = []
        for channel_index in range(
            point_mimo_offaxis_static_moving_negative_multiframe["value"]["baseband"].shape[0]
        ):
            range_doppler = np.abs(
                np.fft.fft(
                    np.fft.fft(
                        point_mimo_offaxis_static_moving_negative_multiframe["value"][
                            "baseband"
                        ][channel_index : channel_index + 1],
                        axis=-1,
                    ),
                    axis=1,
                )
            )[0, :, :valid_range_bins]
            static_window = range_doppler[0:2, 23:32]
            moving_window = range_doppler[2:4, 63:72]
            static_peak = np.unravel_index(int(np.argmax(static_window)), static_window.shape)
            moving_peak = np.unravel_index(int(np.argmax(moving_window)), moving_window.shape)
            static_peaks.append([int(static_peak[0]), int(static_peak[1] + 23)])
            moving_peaks.append([int(moving_peak[0] + 2), int(moving_peak[1] + 63)])
        phase_sample_index = 2
        ref = point_mimo_offaxis_static_moving_negative_multiframe["value"]["baseband"][
            0, 0, phase_sample_index
        ]
        rel_phase_deg = np.degrees(
            np.angle(
                point_mimo_offaxis_static_moving_negative_multiframe["value"]["baseband"][
                    :4, 0, phase_sample_index
                ]
                / ref
            )
        )
        profile = np.abs(
            np.fft.fft(
                point_mimo_offaxis_static_moving_negative_multiframe["value"]["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        point_mimo_offaxis_static_moving_negative_multiframe_summary["static_peaks"] = (
            static_peaks
        )
        point_mimo_offaxis_static_moving_negative_multiframe_summary["moving_peaks"] = (
            moving_peaks
        )
        point_mimo_offaxis_static_moving_negative_multiframe_summary["phase_sample_index"] = (
            phase_sample_index
        )
        point_mimo_offaxis_static_moving_negative_multiframe_summary["frame_block_size"] = 4
        point_mimo_offaxis_static_moving_negative_multiframe_summary[
            "frame0_channel_phase_degrees"
        ] = [float(value) for value in rel_phase_deg]
        point_mimo_offaxis_static_moving_negative_multiframe_summary["profile_norm"] = float(
            np.linalg.norm(profile)
        )

    dry_run = _capture_call(rs_module.sim_radar, radar, [target], dry_run=True)
    if dry_run["ok"]:
        np.savez(artifacts_dir / "sim_radar_dry_run.npz", **dry_run["value"])

    repeat_a = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        [target],
    )
    repeat_b = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        [target],
    )

    reproducibility = {"ok": False}
    if repeat_a["ok"] and repeat_b["ok"]:
        reproducibility = {
            "ok": True,
            "baseband_allclose": bool(
                np.allclose(repeat_a["value"]["baseband"], repeat_b["value"]["baseband"])
            ),
            "noise_allclose": bool(
                np.allclose(repeat_a["value"]["noise"], repeat_b["value"]["noise"])
            ),
        }

    mesh_run = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        [mesh_target],
        density=1.0,
    )
    if mesh_run["ok"]:
        np.savez(artifacts_dir / "sim_radar_mesh_target.npz", **mesh_run["value"])

    mesh_broadside = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        [{**mesh_target, "rotation": [0.0, 0.0, 0.0]}],
        density=1.0,
    )
    if mesh_broadside["ok"]:
        np.savez(artifacts_dir / "sim_radar_mesh_broadside.npz", **mesh_broadside["value"])

    mesh_edge_on = _capture_call(
        rs_module.sim_radar,
        _build_minimal_radar(rs_module, seed=2026),
        [{**mesh_target, "rotation": [0.0, 90.0, 0.0]}],
        density=1.0,
    )
    if mesh_edge_on["ok"]:
        np.savez(artifacts_dir / "sim_radar_mesh_edge_on.npz", **mesh_edge_on["value"])

    mesh_aspect = {"ok": False}
    if mesh_broadside["ok"] and mesh_edge_on["ok"]:
        broadside_max_abs = float(np.max(np.abs(mesh_broadside["value"]["baseband"])))
        edge_on_max_abs = float(np.max(np.abs(mesh_edge_on["value"]["baseband"])))
        mesh_aspect = {
            "ok": True,
            "broadside_max_abs": broadside_max_abs,
            "edge_on_max_abs": edge_on_max_abs,
            "broadside_gt_edge_on": bool(broadside_max_abs > edge_on_max_abs),
        }

    return {
        "point_target": {k: v for k, v in point_run.items() if k != "value"},
        "point_mimo": {k: v for k, v in point_mimo.items() if k != "value"},
        "point_mimo_offaxis": point_mimo_offaxis_summary,
        "point_mimo_offaxis_moving": point_mimo_offaxis_moving_summary,
        "point_mimo_offaxis_moving_negative": point_mimo_offaxis_moving_negative_summary,
        "point_mimo_offaxis_moving_multiframe": point_mimo_offaxis_moving_multiframe_summary,
        "point_mimo_offaxis_moving_negative_multiframe": (
            point_mimo_offaxis_moving_negative_multiframe_summary
        ),
        "point_mimo_multiframe": {
            k: v for k, v in point_mimo_multiframe.items() if k != "value"
        },
        "point_phase": {k: v for k, v in point_phase.items() if k != "value"},
        "point_moving_doppler": point_moving_summary,
        "point_moving_doppler_negative": point_moving_negative_summary,
        "point_multi": {k: v for k, v in point_multi.items() if k != "value"},
        "point_static_moving": point_static_moving_summary,
        "point_static_moving_negative": point_static_moving_negative_summary,
        "point_static_static_moving": point_static_static_moving_summary,
        "point_static_static_moving_negative": point_static_static_moving_negative_summary,
        "point_static_static_moving_moving": point_static_static_moving_moving_summary,
        "point_static_static_moving_moving_negative": (
            point_static_static_moving_moving_negative_summary
        ),
        "point_mimo_offaxis_static_moving": point_mimo_offaxis_static_moving_summary,
        "point_mimo_offaxis_static_moving_negative": (
            point_mimo_offaxis_static_moving_negative_summary
        ),
        "point_mimo_offaxis_static_static_moving": (
            point_mimo_offaxis_static_static_moving_summary
        ),
        "point_mimo_offaxis_static_static_moving_negative": (
            point_mimo_offaxis_static_static_moving_negative_summary
        ),
        "point_mimo_offaxis_static_static_moving_moving": (
            point_mimo_offaxis_static_static_moving_moving_summary
        ),
        "point_mimo_offaxis_static_static_mixed_sign": (
            point_mimo_offaxis_static_static_mixed_sign_summary
        ),
        "point_mimo_offaxis_static_static_mixed_sign_multiframe": (
            point_mimo_offaxis_static_static_mixed_sign_multiframe_summary
        ),
        "point_mimo_offaxis_static_static_moving_moving_negative_multiframe": (
            point_mimo_offaxis_static_static_moving_moving_negative_multiframe_summary
        ),
        "point_mimo_offaxis_static_static_moving_moving_multiframe": (
            point_mimo_offaxis_static_static_moving_moving_multiframe_summary
        ),
        "point_mimo_offaxis_static_static_moving_multiframe": (
            point_mimo_offaxis_static_static_moving_multiframe_summary
        ),
        "point_mimo_offaxis_static_static_moving_negative_multiframe": (
            point_mimo_offaxis_static_static_moving_negative_multiframe_summary
        ),
        "point_mimo_offaxis_static_moving_multiframe": (
            point_mimo_offaxis_static_moving_multiframe_summary
        ),
        "point_mimo_offaxis_static_moving_negative_multiframe": (
            point_mimo_offaxis_static_moving_negative_multiframe_summary
        ),
        "dry_run": {k: v for k, v in dry_run.items() if k != "value"},
        "seed_repeat": reproducibility,
        "mesh_target": {k: v for k, v in mesh_run.items() if k != "value"},
        "mesh_aspect": mesh_aspect,
    }


def _capture_sim_lidar(rs_module, artifacts_dir: Path, plate_path: Path) -> dict[str, Any]:
    lidar = _build_lidar()
    target = {"model": str(plate_path), "location": [10.0, 0.0, 0.0]}

    baseline = _capture_call(rs_module.sim_lidar, lidar, [target], frame_time=0.0)
    if baseline["ok"]:
        np.save(artifacts_dir / "sim_lidar_hits.npy", baseline["value"], allow_pickle=False)

    moving = _capture_call(
        rs_module.sim_lidar,
        lidar,
        [{**target, "speed": [2.0, 0.0, 0.0]}],
        frame_time=1.5,
    )
    if moving["ok"]:
        np.save(
            artifacts_dir / "sim_lidar_moving_hits.npy",
            moving["value"],
            allow_pickle=False,
        )

    multi_hit = _capture_call(
        rs_module.sim_lidar,
        _build_lidar_multi(),
        _build_lidar_multi_targets(plate_path),
        frame_time=0.0,
    )
    if multi_hit["ok"]:
        np.save(
            artifacts_dir / "sim_lidar_multi_hits.npy",
            multi_hit["value"],
            allow_pickle=False,
        )

    return {
        "baseline": {k: v for k, v in baseline.items() if k != "value"},
        "moving": {k: v for k, v in moving.items() if k != "value"},
        "multi_hit": {k: v for k, v in multi_hit.items() if k != "value"},
    }


def _capture_sim_rcs(rs_module, artifacts_dir: Path, plate_path: Path) -> dict[str, Any]:
    normal_incidence = _capture_call(
        rs_module.sim_rcs,
        [{"model": str(plate_path)}],
        f=77e9,
        inc_phi=0.0,
        inc_theta=0.0,
    )
    broadside = _capture_call(
        rs_module.sim_rcs,
        [{"model": str(plate_path)}],
        f=77e9,
        inc_phi=0.0,
        inc_theta=90.0,
    )
    edge_on = _capture_call(
        rs_module.sim_rcs,
        [{"model": str(plate_path)}],
        f=77e9,
        inc_phi=90.0,
        inc_theta=90.0,
    )
    obs_backscatter = _capture_call(
        rs_module.sim_rcs,
        [{"model": str(plate_path)}],
        f=77e9,
        inc_phi=0.0,
        inc_theta=90.0,
        obs_phi=0.0,
        obs_theta=90.0,
    )
    obs_forwardscatter = _capture_call(
        rs_module.sim_rcs,
        [{"model": str(plate_path)}],
        f=77e9,
        inc_phi=0.0,
        inc_theta=90.0,
        obs_phi=180.0,
        obs_theta=90.0,
    )
    sweep_obs_phi = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
    obs_phi_sweep = _capture_call(
        rs_module.sim_rcs,
        [{"model": str(plate_path)}],
        f=77e9,
        inc_phi=np.zeros_like(sweep_obs_phi),
        inc_theta=np.full_like(sweep_obs_phi, 90.0),
        obs_phi=sweep_obs_phi,
        obs_theta=np.full_like(sweep_obs_phi, 90.0),
    )
    if obs_phi_sweep["ok"]:
        np.savez(
            artifacts_dir / "sim_rcs_obs_phi_sweep.npz",
            obs_phi=sweep_obs_phi,
            rcs=np.asarray(obs_phi_sweep["value"], dtype=float),
        )

    polarization_specs = [
        ("co_pol", [0.0, 0.0, 1.0]),
        ("cross_y", [0.0, 1.0, 0.0]),
        ("cross_x", [1.0, 0.0, 0.0]),
        ("cross_xy", [1.0, 1.0, 0.0]),
    ]
    polarization_results: list[tuple[str, float]] = []
    polarization_ok = True
    for case_name, obs_pol in polarization_specs:
        capture = _capture_call(
            rs_module.sim_rcs,
            [{"model": str(plate_path)}],
            f=77e9,
            inc_phi=0.0,
            inc_theta=90.0,
            inc_pol=[0.0, 0.0, 1.0],
            obs_phi=0.0,
            obs_theta=90.0,
            obs_pol=obs_pol,
        )
        polarization_ok = polarization_ok and capture["ok"]
        if capture["ok"]:
            polarization_results.append((case_name, float(capture["value"])))

    polarization_cases = {"ok": False}
    if polarization_ok and len(polarization_results) == len(polarization_specs):
        case_names = np.asarray([name for name, _ in polarization_results], dtype="<U16")
        rcs_values = np.asarray([value for _, value in polarization_results], dtype=float)
        np.savez(
            artifacts_dir / "sim_rcs_polarization_cases.npz",
            case_names=case_names,
            rcs=rcs_values,
        )
        co_pol = float(rcs_values[0])
        max_cross = float(np.max(rcs_values[1:]))
        polarization_cases = {
            "ok": True,
            "case_names": case_names.tolist(),
            "co_pol": co_pol,
            "max_cross_pol": max_cross,
            "co_pol_gt_cross": bool(co_pol > max_cross),
        }

    trend = {"ok": False}
    if broadside["ok"] and edge_on["ok"]:
        trend = {
            "ok": True,
            "broadside_gt_edge_on": bool(broadside["value"] > edge_on["value"]),
            "broadside": float(broadside["value"]),
            "edge_on": float(edge_on["value"]),
        }

    observation_trend = {"ok": False}
    if obs_backscatter["ok"] and obs_forwardscatter["ok"]:
        observation_trend = {
            "ok": True,
            "backscatter_gt_forwardscatter": bool(
                obs_backscatter["value"] > obs_forwardscatter["value"]
            ),
            "backscatter": float(obs_backscatter["value"]),
            "forwardscatter": float(obs_forwardscatter["value"]),
        }

    sweep_analysis = {"ok": False}
    if obs_phi_sweep["ok"]:
        sweep_values = np.asarray(obs_phi_sweep["value"], dtype=float)
        peak_index = int(np.argmax(sweep_values))
        peak_value = float(sweep_values[peak_index])
        tail_max = float(np.max(sweep_values[3:]))
        tail_to_peak_ratio = float(
            tail_max / peak_value if peak_value > np.finfo(float).eps else np.inf
        )
        sweep_analysis = {
            "ok": True,
            "peak_index": peak_index,
            "peak_obs_phi": float(sweep_obs_phi[peak_index]),
            "first_segment_monotonic_decrease": bool(np.all(np.diff(sweep_values[:3]) < 0.0)),
            "tail_quiet": bool(tail_to_peak_ratio <= 1e-6),
            "tail_to_peak_ratio": tail_to_peak_ratio,
        }

    return {
        "normal_incidence": {k: v for k, v in normal_incidence.items() if k != "value"},
        "broadside": {k: v for k, v in broadside.items() if k != "value"},
        "edge_on": {k: v for k, v in edge_on.items() if k != "value"},
        "obs_backscatter": {k: v for k, v in obs_backscatter.items() if k != "value"},
        "obs_forwardscatter": {k: v for k, v in obs_forwardscatter.items() if k != "value"},
        "obs_phi_sweep": {k: v for k, v in obs_phi_sweep.items() if k != "value"},
        "polarization_cases": polarization_cases,
        "trend": trend,
        "observation_trend": observation_trend,
        "sweep_analysis": sweep_analysis,
    }


def run_capture(rs_module, output_dir: Path | str, *, module_name: str) -> dict[str, Any]:
    """Run the oracle capture suite against a RadarSimPy-like module."""

    output_dir = Path(output_dir)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plate_path = write_plate_stl(artifacts_dir / "plate.stl")

    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "module_name": module_name,
        "module_file": getattr(rs_module, "__file__", None),
        "metadata": {
            "version": getattr(rs_module, "__version__", None),
            "author": getattr(rs_module, "__author__", None),
            "email": getattr(rs_module, "__email__", None),
            "url": getattr(rs_module, "__url__", None),
        },
        "top_level_inventory": _top_level_inventory(rs_module),
        "license": _capture_license(rs_module),
        "captures": {
            "sim_radar": _capture_sim_radar(rs_module, artifacts_dir, plate_path),
            "sim_lidar": _capture_sim_lidar(rs_module, artifacts_dir, plate_path),
            "sim_rcs": _capture_sim_rcs(rs_module, artifacts_dir, plate_path),
        },
    }
    manifest["scenario_index"] = build_capture_scenario_index(manifest, output_dir)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
