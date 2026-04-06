"""Helpers for comparing a whitebox module against a captured oracle bundle."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from .capture_suite import run_capture
from .scenario_index import build_compare_scenario_results


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_dict(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def _shared_field_parity(vendor: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    vendor_fields = list(vendor.dtype.names or [])
    candidate_fields = list(candidate.dtype.names or [])
    shared_fields = [field for field in vendor_fields if field in candidate_fields]

    field_checks = {}
    overall = True
    for field in shared_fields:
        matched = bool(np.allclose(vendor[field], candidate[field], atol=1e-6))
        field_checks[field] = matched
        overall = overall and matched

    return {
        "vendor_fields": vendor_fields,
        "candidate_fields": candidate_fields,
        "shared_fields": shared_fields,
        "shared_field_allclose": field_checks,
        "overall": overall and len(vendor) == len(candidate),
    }


def _normalized_correlation_abs(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.ravel(a)
    b_flat = np.ravel(b)
    a_norm = float(np.linalg.norm(a_flat))
    b_norm = float(np.linalg.norm(b_flat))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    correlation = np.vdot(a_flat, b_flat) / (a_norm * b_norm)
    return float(np.abs(correlation))


def _range_profile_top_bins(
    baseband: np.ndarray,
    *,
    valid_bins: int | None = None,
    top_n: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    profile = np.abs(np.fft.fft(baseband, axis=-1)[0, 0])
    if valid_bins is not None:
        profile = profile[:valid_bins]
    top_bins = np.argsort(profile)[-top_n:][::-1]
    return profile, top_bins


def _range_doppler_peak_indices(
    baseband: np.ndarray,
    *,
    valid_bins: int | None = None,
) -> tuple[int, int]:
    range_doppler = np.abs(np.fft.fft(np.fft.fft(baseband, axis=-1), axis=1))[0]
    if valid_bins is not None:
        range_doppler = range_doppler[:, :valid_bins]
    doppler_peak, range_peak = np.unravel_index(
        int(np.argmax(range_doppler)),
        range_doppler.shape,
    )
    return int(doppler_peak), int(range_peak)


def _local_range_doppler_peak_indices(
    baseband: np.ndarray,
    *,
    doppler_slice: slice,
    range_slice: slice,
    valid_bins: int | None = None,
) -> tuple[int, int]:
    range_doppler = np.abs(np.fft.fft(np.fft.fft(baseband, axis=-1), axis=1))[0]
    if valid_bins is not None:
        range_doppler = range_doppler[:, :valid_bins]
    window = range_doppler[doppler_slice, range_slice]
    local_peak = np.unravel_index(int(np.argmax(window)), window.shape)
    return (
        int(local_peak[0] + doppler_slice.start),
        int(local_peak[1] + range_slice.start),
    )


def _relative_or_absolute_close(
    oracle_value: float,
    candidate_value: float,
    *,
    rtol: float = 0.01,
    atol: float = 1e-9,
) -> bool:
    return bool(np.isclose(oracle_value, candidate_value, rtol=rtol, atol=atol))


def _quadrature_phase_report(
    reference: np.ndarray,
    shifted: np.ndarray,
    *,
    expected: complex = 1j,
    atol: float = 1e-6,
) -> dict[str, Any]:
    reference_flat = np.ravel(reference)
    shifted_flat = np.ravel(shifted)
    valid_mask = np.abs(reference_flat) > np.finfo(float).eps
    sample_count = int(np.count_nonzero(valid_mask))
    if sample_count == 0:
        return {
            "sample_count": 0,
            "mean_ratio_real": None,
            "mean_ratio_imag": None,
            "max_abs_error_to_expected": None,
            "expected_close": False,
        }

    ratios = shifted_flat[valid_mask] / reference_flat[valid_mask]
    mean_ratio = np.mean(ratios)
    max_abs_error = float(np.max(np.abs(ratios - expected)))
    return {
        "sample_count": sample_count,
        "mean_ratio_real": float(np.real(mean_ratio)),
        "mean_ratio_imag": float(np.imag(mean_ratio)),
        "max_abs_error_to_expected": max_abs_error,
        "expected_close": bool(max_abs_error <= atol),
    }


def _channel_relative_phase_degrees(
    baseband: np.ndarray,
    *,
    pulse_index: int = 0,
) -> tuple[int, np.ndarray]:
    peak_sample_index = int(np.argmax(np.abs(baseband[0, pulse_index])))
    ref = baseband[0, pulse_index, peak_sample_index]
    rel_phase_deg = np.degrees(np.angle(baseband[:, pulse_index, peak_sample_index] / ref))
    return peak_sample_index, np.asarray(rel_phase_deg, dtype=float)


def _compare_license(
    oracle_manifest: dict[str, Any], candidate_manifest: dict[str, Any]
) -> dict[str, Any]:
    oracle_license = oracle_manifest["license"]
    candidate_license = candidate_manifest["license"]

    set_license_match = (
        oracle_license["set_license"].get("ok") is True
        and candidate_license["set_license"].get("ok") is True
        and oracle_license["set_license"].get("summary") is None
        and candidate_license["set_license"].get("summary") is None
    )
    is_licensed_match = (
        oracle_license["is_licensed"].get("summary")
        == candidate_license["is_licensed"].get("summary")
    )
    candidate_info = candidate_license["get_license_info"].get("summary")
    get_license_info_is_str = isinstance(candidate_info, str) and bool(candidate_info.strip())

    overall = set_license_match and is_licensed_match and get_license_info_is_str
    return {
        "set_license_none_match": set_license_match,
        "is_licensed_match": is_licensed_match,
        "get_license_info_is_str": get_license_info_is_str,
        "overall": overall,
    }


def _compare_sim_radar(oracle_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    oracle = _load_npz_dict(oracle_dir / "artifacts" / "sim_radar_point_target.npz")
    candidate = _load_npz_dict(candidate_dir / "artifacts" / "sim_radar_point_target.npz")

    oracle_dry = _load_npz_dict(oracle_dir / "artifacts" / "sim_radar_dry_run.npz")
    candidate_dry = _load_npz_dict(candidate_dir / "artifacts" / "sim_radar_dry_run.npz")

    timestamp_exact = bool(np.array_equal(oracle["timestamp"], candidate["timestamp"]))
    baseband_dtype_match = oracle["baseband"].dtype == candidate["baseband"].dtype
    baseband_shape_match = oracle["baseband"].shape == candidate["baseband"].shape
    baseband_allclose = bool(np.allclose(oracle["baseband"], candidate["baseband"], atol=1e-6))
    baseband_max_abs = float(np.max(np.abs(oracle["baseband"] - candidate["baseband"])))
    noise_dtype_match = oracle["noise"].dtype == candidate["noise"].dtype
    noise_shape_match = oracle["noise"].shape == candidate["noise"].shape

    oracle_manifest = _load_manifest(oracle_dir / "manifest.json")
    candidate_manifest = _load_manifest(candidate_dir / "manifest.json")
    seed_repeat_match = (
        oracle_manifest["captures"]["sim_radar"]["seed_repeat"]
        == candidate_manifest["captures"]["sim_radar"]["seed_repeat"]
    )

    oracle_interference_none = (
        oracle_manifest["captures"]["sim_radar"]["point_target"]["summary"]["interference"] is None
        and oracle_manifest["captures"]["sim_radar"]["dry_run"]["summary"]["interference"] is None
    )
    candidate_interference_none = (
        candidate_manifest["captures"]["sim_radar"]["point_target"]["summary"]["interference"]
        is None
        and candidate_manifest["captures"]["sim_radar"]["dry_run"]["summary"]["interference"]
        is None
    )

    dry_run_zero_match = bool(
        np.all(candidate_dry["baseband"] == 0) and np.all(candidate_dry["noise"] == 0)
    )

    point_target_overall = all(
        [
            timestamp_exact,
            baseband_dtype_match,
            baseband_shape_match,
            baseband_allclose,
            noise_dtype_match,
            noise_shape_match,
            seed_repeat_match,
            oracle_interference_none,
            candidate_interference_none,
            dry_run_zero_match,
        ]
    )
    point_target_report = {
        "timestamp_exact": timestamp_exact,
        "baseband_dtype_match": baseband_dtype_match,
        "baseband_shape_match": baseband_shape_match,
        "baseband_allclose": baseband_allclose,
        "baseband_max_abs": baseband_max_abs,
        "noise_dtype_match": noise_dtype_match,
        "noise_shape_match": noise_shape_match,
        "oracle_interference_none": oracle_interference_none,
        "candidate_interference_none": candidate_interference_none,
        "overall": point_target_overall,
    }

    point_mimo_report = {"available": False, "overall": True}
    oracle_mimo_path = oracle_dir / "artifacts" / "sim_radar_point_mimo.npz"
    candidate_mimo_path = candidate_dir / "artifacts" / "sim_radar_point_mimo.npz"
    if oracle_mimo_path.exists() and candidate_mimo_path.exists():
        oracle_mimo = _load_npz_dict(oracle_mimo_path)
        candidate_mimo = _load_npz_dict(candidate_mimo_path)
        mimo_timestamp_exact = bool(
            np.array_equal(oracle_mimo["timestamp"], candidate_mimo["timestamp"])
        )
        mimo_baseband_dtype_match = (
            oracle_mimo["baseband"].dtype == candidate_mimo["baseband"].dtype
        )
        mimo_baseband_shape_match = (
            oracle_mimo["baseband"].shape == candidate_mimo["baseband"].shape
        )
        mimo_baseband_allclose = bool(
            np.allclose(oracle_mimo["baseband"], candidate_mimo["baseband"], atol=1e-6)
        )
        mimo_noise_dtype_match = oracle_mimo["noise"].dtype == candidate_mimo["noise"].dtype
        mimo_noise_shape_match = oracle_mimo["noise"].shape == candidate_mimo["noise"].shape
        oracle_mimo_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        point_mimo_report = {
            "available": True,
            "timestamp_exact": mimo_timestamp_exact,
            "baseband_dtype_match": mimo_baseband_dtype_match,
            "baseband_shape_match": mimo_baseband_shape_match,
            "baseband_allclose": mimo_baseband_allclose,
            "noise_dtype_match": mimo_noise_dtype_match,
            "noise_shape_match": mimo_noise_shape_match,
            "oracle_interference_none": oracle_mimo_interference_none,
            "candidate_interference_none": candidate_mimo_interference_none,
            "overall": all(
                [
                    mimo_timestamp_exact,
                    mimo_baseband_dtype_match,
                    mimo_baseband_shape_match,
                    mimo_baseband_allclose,
                    mimo_noise_dtype_match,
                    mimo_noise_shape_match,
                    oracle_mimo_interference_none,
                    candidate_mimo_interference_none,
                ]
            ),
        }

    point_mimo_offaxis_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_path = oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis.npz"
    candidate_mimo_offaxis_path = candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis.npz"
    if oracle_mimo_offaxis_path.exists() and candidate_mimo_offaxis_path.exists():
        oracle_mimo_offaxis = _load_npz_dict(oracle_mimo_offaxis_path)
        candidate_mimo_offaxis = _load_npz_dict(candidate_mimo_offaxis_path)
        mimo_offaxis_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis["timestamp"],
                candidate_mimo_offaxis["timestamp"],
            )
        )
        mimo_offaxis_baseband_dtype_match = (
            oracle_mimo_offaxis["baseband"].dtype == candidate_mimo_offaxis["baseband"].dtype
        )
        mimo_offaxis_baseband_shape_match = (
            oracle_mimo_offaxis["baseband"].shape == candidate_mimo_offaxis["baseband"].shape
        )
        mimo_offaxis_baseband_allclose = bool(
            np.allclose(
                oracle_mimo_offaxis["baseband"],
                candidate_mimo_offaxis["baseband"],
                atol=1e-6,
            )
        )
        mimo_offaxis_noise_dtype_match = (
            oracle_mimo_offaxis["noise"].dtype == candidate_mimo_offaxis["noise"].dtype
        )
        mimo_offaxis_noise_shape_match = (
            oracle_mimo_offaxis["noise"].shape == candidate_mimo_offaxis["noise"].shape
        )
        oracle_mimo_offaxis_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_peak_sample_index, oracle_rel_phase_deg = _channel_relative_phase_degrees(
            oracle_mimo_offaxis["baseband"]
        )
        candidate_peak_sample_index, candidate_rel_phase_deg = _channel_relative_phase_degrees(
            candidate_mimo_offaxis["baseband"]
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        point_mimo_offaxis_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_timestamp_exact,
            "baseband_dtype_match": mimo_offaxis_baseband_dtype_match,
            "baseband_shape_match": mimo_offaxis_baseband_shape_match,
            "baseband_allclose": mimo_offaxis_baseband_allclose,
            "noise_dtype_match": mimo_offaxis_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_noise_shape_match,
            "oracle_interference_none": oracle_mimo_offaxis_interference_none,
            "candidate_interference_none": candidate_mimo_offaxis_interference_none,
            "oracle_peak_sample_index": oracle_peak_sample_index,
            "candidate_peak_sample_index": candidate_peak_sample_index,
            "peak_sample_index_exact": oracle_peak_sample_index == candidate_peak_sample_index,
            "oracle_channel_phase_degrees": oracle_rel_phase_deg.tolist(),
            "candidate_channel_phase_degrees": candidate_rel_phase_deg.tolist(),
            "channel_phase_allclose": channel_phase_allclose,
            "overall": all(
                [
                    mimo_offaxis_timestamp_exact,
                    mimo_offaxis_baseband_dtype_match,
                    mimo_offaxis_baseband_shape_match,
                    mimo_offaxis_baseband_allclose,
                    mimo_offaxis_noise_dtype_match,
                    mimo_offaxis_noise_shape_match,
                    oracle_mimo_offaxis_interference_none,
                    candidate_mimo_offaxis_interference_none,
                    oracle_peak_sample_index == candidate_peak_sample_index,
                    channel_phase_allclose,
                ]
            ),
        }

    point_mimo_offaxis_moving_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_moving_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving.npz"
    )
    candidate_mimo_offaxis_moving_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving.npz"
    )
    if oracle_mimo_offaxis_moving_path.exists() and candidate_mimo_offaxis_moving_path.exists():
        oracle_mimo_offaxis_moving = _load_npz_dict(oracle_mimo_offaxis_moving_path)
        candidate_mimo_offaxis_moving = _load_npz_dict(candidate_mimo_offaxis_moving_path)
        moving_offaxis_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_moving["timestamp"],
                candidate_mimo_offaxis_moving["timestamp"],
            )
        )
        moving_offaxis_baseband_dtype_match = (
            oracle_mimo_offaxis_moving["baseband"].dtype
            == candidate_mimo_offaxis_moving["baseband"].dtype
        )
        moving_offaxis_baseband_shape_match = (
            oracle_mimo_offaxis_moving["baseband"].shape
            == candidate_mimo_offaxis_moving["baseband"].shape
        )
        moving_offaxis_baseband_allclose = bool(
            np.allclose(
                oracle_mimo_offaxis_moving["baseband"],
                candidate_mimo_offaxis_moving["baseband"],
                atol=1e-6,
            )
        )
        moving_offaxis_noise_dtype_match = (
            oracle_mimo_offaxis_moving["noise"].dtype
            == candidate_mimo_offaxis_moving["noise"].dtype
        )
        moving_offaxis_noise_shape_match = (
            oracle_mimo_offaxis_moving["noise"].shape
            == candidate_mimo_offaxis_moving["noise"].shape
        )
        oracle_mimo_offaxis_moving_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_moving_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_mimo_offaxis_moving["baseband"].shape[-1] // 2
        oracle_doppler_peak, oracle_range_peak = _range_doppler_peak_indices(
            oracle_mimo_offaxis_moving["baseband"],
            valid_bins=valid_bins,
        )
        candidate_doppler_peak, candidate_range_peak = _range_doppler_peak_indices(
            candidate_mimo_offaxis_moving["baseband"],
            valid_bins=valid_bins,
        )
        oracle_phase_sample_index = int(
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving", {})
            .get("phase_sample_index", 2)
        )
        candidate_phase_sample_index = oracle_phase_sample_index
        oracle_ref = oracle_mimo_offaxis_moving["baseband"][0, 0, oracle_phase_sample_index]
        candidate_ref = candidate_mimo_offaxis_moving["baseband"][0, 0, candidate_phase_sample_index]
        oracle_phase_deg = np.degrees(
            np.angle(oracle_mimo_offaxis_moving["baseband"][:, 0, oracle_phase_sample_index] / oracle_ref)
        )
        candidate_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_moving["baseband"][:, 0, candidate_phase_sample_index]
                / candidate_ref
            )
        )
        point_mimo_offaxis_moving_report = {
            "available": True,
            "timestamp_exact": moving_offaxis_timestamp_exact,
            "baseband_dtype_match": moving_offaxis_baseband_dtype_match,
            "baseband_shape_match": moving_offaxis_baseband_shape_match,
            "baseband_allclose": moving_offaxis_baseband_allclose,
            "noise_dtype_match": moving_offaxis_noise_dtype_match,
            "noise_shape_match": moving_offaxis_noise_shape_match,
            "oracle_interference_none": oracle_mimo_offaxis_moving_interference_none,
            "candidate_interference_none": candidate_mimo_offaxis_moving_interference_none,
            "oracle_doppler_peak": oracle_doppler_peak,
            "oracle_range_peak": oracle_range_peak,
            "candidate_doppler_peak": candidate_doppler_peak,
            "candidate_range_peak": candidate_range_peak,
            "doppler_peak_exact": oracle_doppler_peak == candidate_doppler_peak,
            "range_peak_exact": oracle_range_peak == candidate_range_peak,
            "oracle_phase_sample_index": oracle_phase_sample_index,
            "candidate_phase_sample_index": candidate_phase_sample_index,
            "oracle_channel_phase_degrees": oracle_phase_deg.tolist(),
            "candidate_channel_phase_degrees": candidate_phase_deg.tolist(),
            "channel_phase_allclose": bool(np.allclose(oracle_phase_deg, candidate_phase_deg, atol=1e-6)),
            "overall": all(
                [
                    moving_offaxis_timestamp_exact,
                    moving_offaxis_baseband_dtype_match,
                    moving_offaxis_baseband_shape_match,
                    moving_offaxis_baseband_allclose,
                    moving_offaxis_noise_dtype_match,
                    moving_offaxis_noise_shape_match,
                    oracle_mimo_offaxis_moving_interference_none,
                    candidate_mimo_offaxis_moving_interference_none,
                    oracle_doppler_peak == candidate_doppler_peak,
                    oracle_range_peak == candidate_range_peak,
                    bool(np.allclose(oracle_phase_deg, candidate_phase_deg, atol=1e-6)),
                ]
            ),
        }

    point_mimo_offaxis_moving_negative_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_moving_negative_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_negative.npz"
    )
    candidate_mimo_offaxis_moving_negative_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_negative.npz"
    )
    if (
        oracle_mimo_offaxis_moving_negative_path.exists()
        and candidate_mimo_offaxis_moving_negative_path.exists()
    ):
        oracle_mimo_offaxis_moving_negative = _load_npz_dict(
            oracle_mimo_offaxis_moving_negative_path
        )
        candidate_mimo_offaxis_moving_negative = _load_npz_dict(
            candidate_mimo_offaxis_moving_negative_path
        )
        moving_offaxis_negative_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_moving_negative["timestamp"],
                candidate_mimo_offaxis_moving_negative["timestamp"],
            )
        )
        moving_offaxis_negative_baseband_dtype_match = (
            oracle_mimo_offaxis_moving_negative["baseband"].dtype
            == candidate_mimo_offaxis_moving_negative["baseband"].dtype
        )
        moving_offaxis_negative_baseband_shape_match = (
            oracle_mimo_offaxis_moving_negative["baseband"].shape
            == candidate_mimo_offaxis_moving_negative["baseband"].shape
        )
        moving_offaxis_negative_baseband_allclose = bool(
            np.allclose(
                oracle_mimo_offaxis_moving_negative["baseband"],
                candidate_mimo_offaxis_moving_negative["baseband"],
                atol=1e-6,
            )
        )
        moving_offaxis_negative_noise_dtype_match = (
            oracle_mimo_offaxis_moving_negative["noise"].dtype
            == candidate_mimo_offaxis_moving_negative["noise"].dtype
        )
        moving_offaxis_negative_noise_shape_match = (
            oracle_mimo_offaxis_moving_negative["noise"].shape
            == candidate_mimo_offaxis_moving_negative["noise"].shape
        )
        oracle_mimo_offaxis_moving_negative_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_moving_negative_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_mimo_offaxis_moving_negative["baseband"].shape[-1] // 2
        oracle_doppler_peak, oracle_range_peak = _range_doppler_peak_indices(
            oracle_mimo_offaxis_moving_negative["baseband"],
            valid_bins=valid_bins,
        )
        candidate_doppler_peak, candidate_range_peak = _range_doppler_peak_indices(
            candidate_mimo_offaxis_moving_negative["baseband"],
            valid_bins=valid_bins,
        )
        oracle_phase_sample_index = int(
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_negative", {})
            .get("phase_sample_index", 2)
        )
        candidate_phase_sample_index = oracle_phase_sample_index
        oracle_ref = oracle_mimo_offaxis_moving_negative["baseband"][
            0, 0, oracle_phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_moving_negative["baseband"][
            0, 0, candidate_phase_sample_index
        ]
        oracle_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_moving_negative["baseband"][:, 0, oracle_phase_sample_index]
                / oracle_ref
            )
        )
        candidate_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_moving_negative["baseband"][
                    :, 0, candidate_phase_sample_index
                ]
                / candidate_ref
            )
        )
        point_mimo_offaxis_moving_negative_report = {
            "available": True,
            "timestamp_exact": moving_offaxis_negative_timestamp_exact,
            "baseband_dtype_match": moving_offaxis_negative_baseband_dtype_match,
            "baseband_shape_match": moving_offaxis_negative_baseband_shape_match,
            "baseband_allclose": moving_offaxis_negative_baseband_allclose,
            "noise_dtype_match": moving_offaxis_negative_noise_dtype_match,
            "noise_shape_match": moving_offaxis_negative_noise_shape_match,
            "oracle_interference_none": oracle_mimo_offaxis_moving_negative_interference_none,
            "candidate_interference_none": candidate_mimo_offaxis_moving_negative_interference_none,
            "oracle_doppler_peak": oracle_doppler_peak,
            "oracle_range_peak": oracle_range_peak,
            "candidate_doppler_peak": candidate_doppler_peak,
            "candidate_range_peak": candidate_range_peak,
            "doppler_peak_exact": oracle_doppler_peak == candidate_doppler_peak,
            "range_peak_exact": oracle_range_peak == candidate_range_peak,
            "oracle_phase_sample_index": oracle_phase_sample_index,
            "candidate_phase_sample_index": candidate_phase_sample_index,
            "oracle_channel_phase_degrees": oracle_phase_deg.tolist(),
            "candidate_channel_phase_degrees": candidate_phase_deg.tolist(),
            "channel_phase_allclose": bool(
                np.allclose(oracle_phase_deg, candidate_phase_deg, atol=1e-6)
            ),
            "overall": all(
                [
                    moving_offaxis_negative_timestamp_exact,
                    moving_offaxis_negative_baseband_dtype_match,
                    moving_offaxis_negative_baseband_shape_match,
                    moving_offaxis_negative_baseband_allclose,
                    moving_offaxis_negative_noise_dtype_match,
                    moving_offaxis_negative_noise_shape_match,
                    oracle_mimo_offaxis_moving_negative_interference_none,
                    candidate_mimo_offaxis_moving_negative_interference_none,
                    oracle_doppler_peak == candidate_doppler_peak,
                    oracle_range_peak == candidate_range_peak,
                    bool(np.allclose(oracle_phase_deg, candidate_phase_deg, atol=1e-6)),
                ]
            ),
        }

    point_mimo_offaxis_moving_multiframe_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_moving_multiframe_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_multiframe.npz"
    )
    candidate_mimo_offaxis_moving_multiframe_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_moving_multiframe_path.exists()
        and candidate_mimo_offaxis_moving_multiframe_path.exists()
    ):
        oracle_mm_offaxis = _load_npz_dict(oracle_mimo_offaxis_moving_multiframe_path)
        candidate_mm_offaxis = _load_npz_dict(candidate_mimo_offaxis_moving_multiframe_path)
        mm_offaxis_timestamp_exact = bool(
            np.array_equal(oracle_mm_offaxis["timestamp"], candidate_mm_offaxis["timestamp"])
        )
        mm_offaxis_baseband_dtype_match = (
            oracle_mm_offaxis["baseband"].dtype == candidate_mm_offaxis["baseband"].dtype
        )
        mm_offaxis_baseband_shape_match = (
            oracle_mm_offaxis["baseband"].shape == candidate_mm_offaxis["baseband"].shape
        )
        mm_offaxis_baseband_allclose = bool(
            np.allclose(oracle_mm_offaxis["baseband"], candidate_mm_offaxis["baseband"], atol=1e-6)
        )
        mm_offaxis_noise_dtype_match = (
            oracle_mm_offaxis["noise"].dtype == candidate_mm_offaxis["noise"].dtype
        )
        mm_offaxis_noise_shape_match = (
            oracle_mm_offaxis["noise"].shape == candidate_mm_offaxis["noise"].shape
        )
        mm_offaxis_frame_block_size = oracle_mm_offaxis["baseband"].shape[0] // 2
        oracle_mm_offaxis_frame_offset_exact = bool(
            np.array_equal(
                oracle_mm_offaxis["timestamp"][:mm_offaxis_frame_block_size] + 0.001,
                oracle_mm_offaxis["timestamp"][mm_offaxis_frame_block_size:],
            )
        )
        candidate_mm_offaxis_frame_offset_exact = bool(
            np.array_equal(
                candidate_mm_offaxis["timestamp"][:mm_offaxis_frame_block_size] + 0.001,
                candidate_mm_offaxis["timestamp"][mm_offaxis_frame_block_size:],
            )
        )
        oracle_mm_offaxis_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mm_offaxis_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_mm_offaxis["baseband"].shape[-1] // 2
        oracle_doppler_peak, oracle_range_peak = _range_doppler_peak_indices(
            oracle_mm_offaxis["baseband"],
            valid_bins=valid_bins,
        )
        candidate_doppler_peak, candidate_range_peak = _range_doppler_peak_indices(
            candidate_mm_offaxis["baseband"],
            valid_bins=valid_bins,
        )
        phase_sample_index = int(
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_multiframe", {})
            .get("phase_sample_index", 2)
        )
        oracle_ref = oracle_mm_offaxis["baseband"][0, 0, phase_sample_index]
        candidate_ref = candidate_mm_offaxis["baseband"][0, 0, phase_sample_index]
        oracle_phase_deg = np.degrees(
            np.angle(oracle_mm_offaxis["baseband"][:4, 0, phase_sample_index] / oracle_ref)
        )
        candidate_phase_deg = np.degrees(
            np.angle(candidate_mm_offaxis["baseband"][:4, 0, phase_sample_index] / candidate_ref)
        )
        point_mimo_offaxis_moving_multiframe_report = {
            "available": True,
            "timestamp_exact": mm_offaxis_timestamp_exact,
            "baseband_dtype_match": mm_offaxis_baseband_dtype_match,
            "baseband_shape_match": mm_offaxis_baseband_shape_match,
            "baseband_allclose": mm_offaxis_baseband_allclose,
            "noise_dtype_match": mm_offaxis_noise_dtype_match,
            "noise_shape_match": mm_offaxis_noise_shape_match,
            "frame_block_size": mm_offaxis_frame_block_size,
            "oracle_frame_offset_exact": oracle_mm_offaxis_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_mm_offaxis_frame_offset_exact,
            "oracle_interference_none": oracle_mm_offaxis_interference_none,
            "candidate_interference_none": candidate_mm_offaxis_interference_none,
            "oracle_doppler_peak": oracle_doppler_peak,
            "oracle_range_peak": oracle_range_peak,
            "candidate_doppler_peak": candidate_doppler_peak,
            "candidate_range_peak": candidate_range_peak,
            "doppler_peak_exact": oracle_doppler_peak == candidate_doppler_peak,
            "range_peak_exact": oracle_range_peak == candidate_range_peak,
            "phase_sample_index": phase_sample_index,
            "oracle_frame0_channel_phase_degrees": oracle_phase_deg.tolist(),
            "candidate_frame0_channel_phase_degrees": candidate_phase_deg.tolist(),
            "channel_phase_allclose": bool(np.allclose(oracle_phase_deg, candidate_phase_deg, atol=1e-6)),
            "overall": all(
                [
                    mm_offaxis_timestamp_exact,
                    mm_offaxis_baseband_dtype_match,
                    mm_offaxis_baseband_shape_match,
                    mm_offaxis_baseband_allclose,
                    mm_offaxis_noise_dtype_match,
                    mm_offaxis_noise_shape_match,
                    oracle_mm_offaxis_frame_offset_exact,
                    candidate_mm_offaxis_frame_offset_exact,
                    oracle_mm_offaxis_interference_none,
                    candidate_mm_offaxis_interference_none,
                    oracle_doppler_peak == candidate_doppler_peak,
                    oracle_range_peak == candidate_range_peak,
                    bool(np.allclose(oracle_phase_deg, candidate_phase_deg, atol=1e-6)),
                ]
            ),
        }

    point_mimo_offaxis_moving_negative_multiframe_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_moving_negative_multiframe_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_negative_multiframe.npz"
    )
    candidate_mimo_offaxis_moving_negative_multiframe_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis_moving_negative_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_moving_negative_multiframe_path.exists()
        and candidate_mimo_offaxis_moving_negative_multiframe_path.exists()
    ):
        oracle_mm_offaxis_negative = _load_npz_dict(
            oracle_mimo_offaxis_moving_negative_multiframe_path
        )
        candidate_mm_offaxis_negative = _load_npz_dict(
            candidate_mimo_offaxis_moving_negative_multiframe_path
        )
        mm_offaxis_negative_timestamp_exact = bool(
            np.array_equal(
                oracle_mm_offaxis_negative["timestamp"],
                candidate_mm_offaxis_negative["timestamp"],
            )
        )
        mm_offaxis_negative_baseband_dtype_match = (
            oracle_mm_offaxis_negative["baseband"].dtype
            == candidate_mm_offaxis_negative["baseband"].dtype
        )
        mm_offaxis_negative_baseband_shape_match = (
            oracle_mm_offaxis_negative["baseband"].shape
            == candidate_mm_offaxis_negative["baseband"].shape
        )
        mm_offaxis_negative_baseband_allclose = bool(
            np.allclose(
                oracle_mm_offaxis_negative["baseband"],
                candidate_mm_offaxis_negative["baseband"],
                atol=1e-6,
            )
        )
        mm_offaxis_negative_noise_dtype_match = (
            oracle_mm_offaxis_negative["noise"].dtype
            == candidate_mm_offaxis_negative["noise"].dtype
        )
        mm_offaxis_negative_noise_shape_match = (
            oracle_mm_offaxis_negative["noise"].shape
            == candidate_mm_offaxis_negative["noise"].shape
        )
        mm_offaxis_negative_frame_block_size = oracle_mm_offaxis_negative["baseband"].shape[0] // 2
        oracle_mm_offaxis_negative_frame_offset_exact = bool(
            np.array_equal(
                oracle_mm_offaxis_negative["timestamp"][:mm_offaxis_negative_frame_block_size]
                + 0.001,
                oracle_mm_offaxis_negative["timestamp"][mm_offaxis_negative_frame_block_size:],
            )
        )
        candidate_mm_offaxis_negative_frame_offset_exact = bool(
            np.array_equal(
                candidate_mm_offaxis_negative["timestamp"][:mm_offaxis_negative_frame_block_size]
                + 0.001,
                candidate_mm_offaxis_negative["timestamp"][
                    mm_offaxis_negative_frame_block_size:
                ],
            )
        )
        oracle_mm_offaxis_negative_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_negative_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mm_offaxis_negative_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_negative_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_mm_offaxis_negative["baseband"].shape[-1] // 2
        oracle_doppler_peak, oracle_range_peak = _range_doppler_peak_indices(
            oracle_mm_offaxis_negative["baseband"],
            valid_bins=valid_bins,
        )
        candidate_doppler_peak, candidate_range_peak = _range_doppler_peak_indices(
            candidate_mm_offaxis_negative["baseband"],
            valid_bins=valid_bins,
        )
        phase_sample_index = int(
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_moving_negative_multiframe", {})
            .get("phase_sample_index", 2)
        )
        oracle_ref = oracle_mm_offaxis_negative["baseband"][0, 0, phase_sample_index]
        candidate_ref = candidate_mm_offaxis_negative["baseband"][0, 0, phase_sample_index]
        oracle_phase_deg = np.degrees(
            np.angle(
                oracle_mm_offaxis_negative["baseband"][:4, 0, phase_sample_index] / oracle_ref
            )
        )
        candidate_phase_deg = np.degrees(
            np.angle(
                candidate_mm_offaxis_negative["baseband"][:4, 0, phase_sample_index]
                / candidate_ref
            )
        )
        point_mimo_offaxis_moving_negative_multiframe_report = {
            "available": True,
            "timestamp_exact": mm_offaxis_negative_timestamp_exact,
            "baseband_dtype_match": mm_offaxis_negative_baseband_dtype_match,
            "baseband_shape_match": mm_offaxis_negative_baseband_shape_match,
            "baseband_allclose": mm_offaxis_negative_baseband_allclose,
            "noise_dtype_match": mm_offaxis_negative_noise_dtype_match,
            "noise_shape_match": mm_offaxis_negative_noise_shape_match,
            "frame_block_size": mm_offaxis_negative_frame_block_size,
            "oracle_frame_offset_exact": oracle_mm_offaxis_negative_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_mm_offaxis_negative_frame_offset_exact,
            "oracle_interference_none": oracle_mm_offaxis_negative_interference_none,
            "candidate_interference_none": candidate_mm_offaxis_negative_interference_none,
            "oracle_doppler_peak": oracle_doppler_peak,
            "oracle_range_peak": oracle_range_peak,
            "candidate_doppler_peak": candidate_doppler_peak,
            "candidate_range_peak": candidate_range_peak,
            "doppler_peak_exact": oracle_doppler_peak == candidate_doppler_peak,
            "range_peak_exact": oracle_range_peak == candidate_range_peak,
            "phase_sample_index": phase_sample_index,
            "oracle_frame0_channel_phase_degrees": oracle_phase_deg.tolist(),
            "candidate_frame0_channel_phase_degrees": candidate_phase_deg.tolist(),
            "channel_phase_allclose": bool(
                np.allclose(oracle_phase_deg, candidate_phase_deg, atol=1e-6)
            ),
            "overall": all(
                [
                    mm_offaxis_negative_timestamp_exact,
                    mm_offaxis_negative_baseband_dtype_match,
                    mm_offaxis_negative_baseband_shape_match,
                    mm_offaxis_negative_baseband_allclose,
                    mm_offaxis_negative_noise_dtype_match,
                    mm_offaxis_negative_noise_shape_match,
                    oracle_mm_offaxis_negative_frame_offset_exact,
                    candidate_mm_offaxis_negative_frame_offset_exact,
                    oracle_mm_offaxis_negative_interference_none,
                    candidate_mm_offaxis_negative_interference_none,
                    oracle_doppler_peak == candidate_doppler_peak,
                    oracle_range_peak == candidate_range_peak,
                    bool(np.allclose(oracle_phase_deg, candidate_phase_deg, atol=1e-6)),
                ]
            ),
        }

    point_mimo_multiframe_report = {"available": False, "overall": True}
    oracle_mimo_multiframe_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_multiframe.npz"
    )
    candidate_mimo_multiframe_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_multiframe.npz"
    )
    if oracle_mimo_multiframe_path.exists() and candidate_mimo_multiframe_path.exists():
        oracle_mm = _load_npz_dict(oracle_mimo_multiframe_path)
        candidate_mm = _load_npz_dict(candidate_mimo_multiframe_path)
        mm_timestamp_exact = bool(np.array_equal(oracle_mm["timestamp"], candidate_mm["timestamp"]))
        mm_baseband_dtype_match = oracle_mm["baseband"].dtype == candidate_mm["baseband"].dtype
        mm_baseband_shape_match = oracle_mm["baseband"].shape == candidate_mm["baseband"].shape
        mm_baseband_allclose = bool(
            np.allclose(oracle_mm["baseband"], candidate_mm["baseband"], atol=1e-6)
        )
        mm_noise_dtype_match = oracle_mm["noise"].dtype == candidate_mm["noise"].dtype
        mm_noise_shape_match = oracle_mm["noise"].shape == candidate_mm["noise"].shape
        mm_frame_block_size = oracle_mm["baseband"].shape[0] // 2
        oracle_frame_offset_exact = bool(
            np.array_equal(
                oracle_mm["timestamp"][:mm_frame_block_size] + 0.001,
                oracle_mm["timestamp"][mm_frame_block_size:],
            )
        )
        candidate_frame_offset_exact = bool(
            np.array_equal(
                candidate_mm["timestamp"][:mm_frame_block_size] + 0.001,
                candidate_mm["timestamp"][mm_frame_block_size:],
            )
        )
        oracle_mm_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mm_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        point_mimo_multiframe_report = {
            "available": True,
            "timestamp_exact": mm_timestamp_exact,
            "baseband_dtype_match": mm_baseband_dtype_match,
            "baseband_shape_match": mm_baseband_shape_match,
            "baseband_allclose": mm_baseband_allclose,
            "noise_dtype_match": mm_noise_dtype_match,
            "noise_shape_match": mm_noise_shape_match,
            "frame_block_size": mm_frame_block_size,
            "oracle_frame_offset_exact": oracle_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_frame_offset_exact,
            "oracle_interference_none": oracle_mm_interference_none,
            "candidate_interference_none": candidate_mm_interference_none,
            "overall": all(
                [
                    mm_timestamp_exact,
                    mm_baseband_dtype_match,
                    mm_baseband_shape_match,
                    mm_baseband_allclose,
                    mm_noise_dtype_match,
                    mm_noise_shape_match,
                    oracle_frame_offset_exact,
                    candidate_frame_offset_exact,
                    oracle_mm_interference_none,
                    candidate_mm_interference_none,
                ]
            ),
        }

    point_phase_report = {"available": False, "overall": True}
    oracle_phase_path = oracle_dir / "artifacts" / "sim_radar_point_phase.npz"
    candidate_phase_path = candidate_dir / "artifacts" / "sim_radar_point_phase.npz"
    if oracle_phase_path.exists() and candidate_phase_path.exists():
        oracle_phase = _load_npz_dict(oracle_phase_path)
        candidate_phase = _load_npz_dict(candidate_phase_path)
        phase_timestamp_exact = bool(
            np.array_equal(oracle_phase["timestamp"], candidate_phase["timestamp"])
        )
        phase_baseband_dtype_match = (
            oracle_phase["baseband"].dtype == candidate_phase["baseband"].dtype
        )
        phase_baseband_shape_match = (
            oracle_phase["baseband"].shape == candidate_phase["baseband"].shape
        )
        phase_baseband_allclose = bool(
            np.allclose(oracle_phase["baseband"], candidate_phase["baseband"], atol=1e-6)
        )
        phase_noise_dtype_match = oracle_phase["noise"].dtype == candidate_phase["noise"].dtype
        phase_noise_shape_match = oracle_phase["noise"].shape == candidate_phase["noise"].shape
        oracle_phase_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_phase", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_phase_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_phase", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_quadrature = _quadrature_phase_report(
            oracle["baseband"],
            oracle_phase["baseband"],
        )
        candidate_quadrature = _quadrature_phase_report(
            candidate["baseband"],
            candidate_phase["baseband"],
        )
        point_phase_report = {
            "available": True,
            "timestamp_exact": phase_timestamp_exact,
            "baseband_dtype_match": phase_baseband_dtype_match,
            "baseband_shape_match": phase_baseband_shape_match,
            "baseband_allclose": phase_baseband_allclose,
            "noise_dtype_match": phase_noise_dtype_match,
            "noise_shape_match": phase_noise_shape_match,
            "oracle_interference_none": oracle_phase_interference_none,
            "candidate_interference_none": candidate_phase_interference_none,
            "oracle_quadrature": oracle_quadrature,
            "candidate_quadrature": candidate_quadrature,
            "overall": all(
                [
                    phase_timestamp_exact,
                    phase_baseband_dtype_match,
                    phase_baseband_shape_match,
                    phase_baseband_allclose,
                    phase_noise_dtype_match,
                    phase_noise_shape_match,
                    oracle_phase_interference_none,
                    candidate_phase_interference_none,
                    oracle_quadrature["expected_close"],
                    candidate_quadrature["expected_close"],
                ]
            ),
        }

    point_moving_report = {"available": False, "overall": True}
    oracle_moving_path = oracle_dir / "artifacts" / "sim_radar_point_moving_doppler.npz"
    candidate_moving_path = candidate_dir / "artifacts" / "sim_radar_point_moving_doppler.npz"
    if oracle_moving_path.exists() and candidate_moving_path.exists():
        oracle_moving = _load_npz_dict(oracle_moving_path)
        candidate_moving = _load_npz_dict(candidate_moving_path)
        moving_timestamp_exact = bool(
            np.array_equal(oracle_moving["timestamp"], candidate_moving["timestamp"])
        )
        moving_baseband_dtype_match = (
            oracle_moving["baseband"].dtype == candidate_moving["baseband"].dtype
        )
        moving_baseband_shape_match = (
            oracle_moving["baseband"].shape == candidate_moving["baseband"].shape
        )
        moving_baseband_allclose = bool(
            np.allclose(oracle_moving["baseband"], candidate_moving["baseband"], atol=1e-6)
        )
        moving_noise_dtype_match = (
            oracle_moving["noise"].dtype == candidate_moving["noise"].dtype
        )
        moving_noise_shape_match = (
            oracle_moving["noise"].shape == candidate_moving["noise"].shape
        )
        oracle_moving_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_moving_doppler", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_moving_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_moving_doppler", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_moving["baseband"].shape[-1] // 2
        oracle_doppler_peak, oracle_range_peak = _range_doppler_peak_indices(
            oracle_moving["baseband"],
            valid_bins=valid_bins,
        )
        candidate_doppler_peak, candidate_range_peak = _range_doppler_peak_indices(
            candidate_moving["baseband"],
            valid_bins=valid_bins,
        )
        point_moving_report = {
            "available": True,
            "timestamp_exact": moving_timestamp_exact,
            "baseband_dtype_match": moving_baseband_dtype_match,
            "baseband_shape_match": moving_baseband_shape_match,
            "baseband_allclose": moving_baseband_allclose,
            "noise_dtype_match": moving_noise_dtype_match,
            "noise_shape_match": moving_noise_shape_match,
            "oracle_interference_none": oracle_moving_interference_none,
            "candidate_interference_none": candidate_moving_interference_none,
            "oracle_doppler_peak": oracle_doppler_peak,
            "oracle_range_peak": oracle_range_peak,
            "candidate_doppler_peak": candidate_doppler_peak,
            "candidate_range_peak": candidate_range_peak,
            "doppler_peak_exact": oracle_doppler_peak == candidate_doppler_peak,
            "range_peak_exact": oracle_range_peak == candidate_range_peak,
            "overall": all(
                [
                    moving_timestamp_exact,
                    moving_baseband_dtype_match,
                    moving_baseband_shape_match,
                    moving_baseband_allclose,
                    moving_noise_dtype_match,
                    moving_noise_shape_match,
                    oracle_moving_interference_none,
                    candidate_moving_interference_none,
                    oracle_doppler_peak == candidate_doppler_peak,
                    oracle_range_peak == candidate_range_peak,
                ]
            ),
        }

    point_moving_negative_report = {"available": False, "overall": True}
    oracle_moving_negative_path = (
        oracle_dir / "artifacts" / "sim_radar_point_moving_doppler_negative.npz"
    )
    candidate_moving_negative_path = (
        candidate_dir / "artifacts" / "sim_radar_point_moving_doppler_negative.npz"
    )
    if oracle_moving_negative_path.exists() and candidate_moving_negative_path.exists():
        oracle_moving_negative = _load_npz_dict(oracle_moving_negative_path)
        candidate_moving_negative = _load_npz_dict(candidate_moving_negative_path)
        moving_negative_timestamp_exact = bool(
            np.array_equal(
                oracle_moving_negative["timestamp"],
                candidate_moving_negative["timestamp"],
            )
        )
        moving_negative_baseband_dtype_match = (
            oracle_moving_negative["baseband"].dtype
            == candidate_moving_negative["baseband"].dtype
        )
        moving_negative_baseband_shape_match = (
            oracle_moving_negative["baseband"].shape
            == candidate_moving_negative["baseband"].shape
        )
        moving_negative_baseband_allclose = bool(
            np.allclose(
                oracle_moving_negative["baseband"],
                candidate_moving_negative["baseband"],
                atol=1e-6,
            )
        )
        moving_negative_noise_dtype_match = (
            oracle_moving_negative["noise"].dtype == candidate_moving_negative["noise"].dtype
        )
        moving_negative_noise_shape_match = (
            oracle_moving_negative["noise"].shape == candidate_moving_negative["noise"].shape
        )
        oracle_moving_negative_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_moving_doppler_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_moving_negative_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_moving_doppler_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_moving_negative["baseband"].shape[-1] // 2
        oracle_doppler_peak, oracle_range_peak = _range_doppler_peak_indices(
            oracle_moving_negative["baseband"],
            valid_bins=valid_bins,
        )
        candidate_doppler_peak, candidate_range_peak = _range_doppler_peak_indices(
            candidate_moving_negative["baseband"],
            valid_bins=valid_bins,
        )
        point_moving_negative_report = {
            "available": True,
            "timestamp_exact": moving_negative_timestamp_exact,
            "baseband_dtype_match": moving_negative_baseband_dtype_match,
            "baseband_shape_match": moving_negative_baseband_shape_match,
            "baseband_allclose": moving_negative_baseband_allclose,
            "noise_dtype_match": moving_negative_noise_dtype_match,
            "noise_shape_match": moving_negative_noise_shape_match,
            "oracle_interference_none": oracle_moving_negative_interference_none,
            "candidate_interference_none": candidate_moving_negative_interference_none,
            "oracle_doppler_peak": oracle_doppler_peak,
            "oracle_range_peak": oracle_range_peak,
            "candidate_doppler_peak": candidate_doppler_peak,
            "candidate_range_peak": candidate_range_peak,
            "doppler_peak_exact": oracle_doppler_peak == candidate_doppler_peak,
            "range_peak_exact": oracle_range_peak == candidate_range_peak,
            "overall": all(
                [
                    moving_negative_timestamp_exact,
                    moving_negative_baseband_dtype_match,
                    moving_negative_baseband_shape_match,
                    moving_negative_baseband_allclose,
                    moving_negative_noise_dtype_match,
                    moving_negative_noise_shape_match,
                    oracle_moving_negative_interference_none,
                    candidate_moving_negative_interference_none,
                    oracle_doppler_peak == candidate_doppler_peak,
                    oracle_range_peak == candidate_range_peak,
                ]
            ),
        }

    point_multi_report = {"available": False, "overall": True}
    oracle_multi_path = oracle_dir / "artifacts" / "sim_radar_point_multi.npz"
    candidate_multi_path = candidate_dir / "artifacts" / "sim_radar_point_multi.npz"
    if oracle_multi_path.exists() and candidate_multi_path.exists():
        oracle_multi = _load_npz_dict(oracle_multi_path)
        candidate_multi = _load_npz_dict(candidate_multi_path)
        multi_timestamp_exact = bool(
            np.array_equal(oracle_multi["timestamp"], candidate_multi["timestamp"])
        )
        multi_baseband_dtype_match = (
            oracle_multi["baseband"].dtype == candidate_multi["baseband"].dtype
        )
        multi_baseband_shape_match = (
            oracle_multi["baseband"].shape == candidate_multi["baseband"].shape
        )
        multi_noise_dtype_match = (
            oracle_multi["noise"].dtype == candidate_multi["noise"].dtype
        )
        multi_noise_shape_match = (
            oracle_multi["noise"].shape == candidate_multi["noise"].shape
        )
        oracle_multi_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_multi", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_multi_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_multi", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_profile, oracle_top_bins = _range_profile_top_bins(
            oracle_multi["baseband"],
            valid_bins=oracle_multi["baseband"].shape[-1] // 2,
        )
        candidate_profile, candidate_top_bins = _range_profile_top_bins(
            candidate_multi["baseband"],
            valid_bins=candidate_multi["baseband"].shape[-1] // 2,
        )
        top_bins_exact = bool(np.array_equal(oracle_top_bins, candidate_top_bins))
        profile_correlation_abs = _normalized_correlation_abs(oracle_profile, candidate_profile)
        profile_correlation_pass = profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_multi["baseband"],
            candidate_multi["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_multi_report = {
            "available": True,
            "timestamp_exact": multi_timestamp_exact,
            "baseband_dtype_match": multi_baseband_dtype_match,
            "baseband_shape_match": multi_baseband_shape_match,
            "noise_dtype_match": multi_noise_dtype_match,
            "noise_shape_match": multi_noise_shape_match,
            "oracle_interference_none": oracle_multi_interference_none,
            "candidate_interference_none": candidate_multi_interference_none,
            "oracle_top_bins": oracle_top_bins.tolist(),
            "candidate_top_bins": candidate_top_bins.tolist(),
            "top_bins_exact": top_bins_exact,
            "profile_correlation_abs": profile_correlation_abs,
            "profile_correlation_pass": profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    multi_timestamp_exact,
                    multi_baseband_dtype_match,
                    multi_baseband_shape_match,
                    multi_noise_dtype_match,
                    multi_noise_shape_match,
                    oracle_multi_interference_none,
                    candidate_multi_interference_none,
                    top_bins_exact,
                    profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_static_moving_report = {"available": False, "overall": True}
    oracle_static_moving_path = oracle_dir / "artifacts" / "sim_radar_point_static_moving.npz"
    candidate_static_moving_path = candidate_dir / "artifacts" / "sim_radar_point_static_moving.npz"
    if oracle_static_moving_path.exists() and candidate_static_moving_path.exists():
        oracle_static_moving = _load_npz_dict(oracle_static_moving_path)
        candidate_static_moving = _load_npz_dict(candidate_static_moving_path)
        static_moving_timestamp_exact = bool(
            np.array_equal(oracle_static_moving["timestamp"], candidate_static_moving["timestamp"])
        )
        static_moving_baseband_dtype_match = (
            oracle_static_moving["baseband"].dtype == candidate_static_moving["baseband"].dtype
        )
        static_moving_baseband_shape_match = (
            oracle_static_moving["baseband"].shape == candidate_static_moving["baseband"].shape
        )
        static_moving_noise_dtype_match = (
            oracle_static_moving["noise"].dtype == candidate_static_moving["noise"].dtype
        )
        static_moving_noise_shape_match = (
            oracle_static_moving["noise"].shape == candidate_static_moving["noise"].shape
        )
        oracle_static_moving_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_static_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_static_moving_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_static_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_static_moving["baseband"].shape[-1] // 2
        oracle_static_peak = _local_range_doppler_peak_indices(
            oracle_static_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        candidate_static_peak = _local_range_doppler_peak_indices(
            candidate_static_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        oracle_moving_peak = _local_range_doppler_peak_indices(
            oracle_static_moving["baseband"],
            doppler_slice=slice(0, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        candidate_moving_peak = _local_range_doppler_peak_indices(
            candidate_static_moving["baseband"],
            doppler_slice=slice(0, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        oracle_profile = np.abs(np.fft.fft(oracle_static_moving["baseband"], axis=-1))[0, 0, :80]
        candidate_profile = np.abs(np.fft.fft(candidate_static_moving["baseband"], axis=-1))[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_static_moving["baseband"],
            candidate_static_moving["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_static_moving_report = {
            "available": True,
            "timestamp_exact": static_moving_timestamp_exact,
            "baseband_dtype_match": static_moving_baseband_dtype_match,
            "baseband_shape_match": static_moving_baseband_shape_match,
            "noise_dtype_match": static_moving_noise_dtype_match,
            "noise_shape_match": static_moving_noise_shape_match,
            "oracle_interference_none": oracle_static_moving_interference_none,
            "candidate_interference_none": candidate_static_moving_interference_none,
            "oracle_static_peak": list(oracle_static_peak),
            "candidate_static_peak": list(candidate_static_peak),
            "oracle_moving_peak": list(oracle_moving_peak),
            "candidate_moving_peak": list(candidate_moving_peak),
            "static_peak_exact": oracle_static_peak == candidate_static_peak,
            "moving_peak_exact": oracle_moving_peak == candidate_moving_peak,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    static_moving_timestamp_exact,
                    static_moving_baseband_dtype_match,
                    static_moving_baseband_shape_match,
                    static_moving_noise_dtype_match,
                    static_moving_noise_shape_match,
                    oracle_static_moving_interference_none,
                    candidate_static_moving_interference_none,
                    oracle_static_peak == candidate_static_peak,
                    oracle_moving_peak == candidate_moving_peak,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_static_moving_negative_report = {"available": False, "overall": True}
    oracle_static_moving_negative_path = (
        oracle_dir / "artifacts" / "sim_radar_point_static_moving_negative.npz"
    )
    candidate_static_moving_negative_path = (
        candidate_dir / "artifacts" / "sim_radar_point_static_moving_negative.npz"
    )
    if oracle_static_moving_negative_path.exists() and candidate_static_moving_negative_path.exists():
        oracle_static_moving_negative = _load_npz_dict(oracle_static_moving_negative_path)
        candidate_static_moving_negative = _load_npz_dict(candidate_static_moving_negative_path)
        static_moving_negative_timestamp_exact = bool(
            np.array_equal(
                oracle_static_moving_negative["timestamp"],
                candidate_static_moving_negative["timestamp"],
            )
        )
        static_moving_negative_baseband_dtype_match = (
            oracle_static_moving_negative["baseband"].dtype
            == candidate_static_moving_negative["baseband"].dtype
        )
        static_moving_negative_baseband_shape_match = (
            oracle_static_moving_negative["baseband"].shape
            == candidate_static_moving_negative["baseband"].shape
        )
        static_moving_negative_noise_dtype_match = (
            oracle_static_moving_negative["noise"].dtype
            == candidate_static_moving_negative["noise"].dtype
        )
        static_moving_negative_noise_shape_match = (
            oracle_static_moving_negative["noise"].shape
            == candidate_static_moving_negative["noise"].shape
        )
        oracle_static_moving_negative_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_static_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_static_moving_negative_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_static_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_static_moving_negative["baseband"].shape[-1] // 2
        oracle_static_peak = _local_range_doppler_peak_indices(
            oracle_static_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        candidate_static_peak = _local_range_doppler_peak_indices(
            candidate_static_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        oracle_moving_peak = _local_range_doppler_peak_indices(
            oracle_static_moving_negative["baseband"],
            doppler_slice=slice(2, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        candidate_moving_peak = _local_range_doppler_peak_indices(
            candidate_static_moving_negative["baseband"],
            doppler_slice=slice(2, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_static_moving_negative["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_static_moving_negative["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_static_moving_negative["baseband"],
            candidate_static_moving_negative["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_static_moving_negative_report = {
            "available": True,
            "timestamp_exact": static_moving_negative_timestamp_exact,
            "baseband_dtype_match": static_moving_negative_baseband_dtype_match,
            "baseband_shape_match": static_moving_negative_baseband_shape_match,
            "noise_dtype_match": static_moving_negative_noise_dtype_match,
            "noise_shape_match": static_moving_negative_noise_shape_match,
            "oracle_interference_none": oracle_static_moving_negative_interference_none,
            "candidate_interference_none": candidate_static_moving_negative_interference_none,
            "oracle_static_peak": list(oracle_static_peak),
            "candidate_static_peak": list(candidate_static_peak),
            "oracle_moving_peak": list(oracle_moving_peak),
            "candidate_moving_peak": list(candidate_moving_peak),
            "static_peak_exact": oracle_static_peak == candidate_static_peak,
            "moving_peak_exact": oracle_moving_peak == candidate_moving_peak,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    static_moving_negative_timestamp_exact,
                    static_moving_negative_baseband_dtype_match,
                    static_moving_negative_baseband_shape_match,
                    static_moving_negative_noise_dtype_match,
                    static_moving_negative_noise_shape_match,
                    oracle_static_moving_negative_interference_none,
                    candidate_static_moving_negative_interference_none,
                    oracle_static_peak == candidate_static_peak,
                    oracle_moving_peak == candidate_moving_peak,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_static_static_moving_report = {"available": False, "overall": True}
    oracle_static_static_moving_path = (
        oracle_dir / "artifacts" / "sim_radar_point_static_static_moving.npz"
    )
    candidate_static_static_moving_path = (
        candidate_dir / "artifacts" / "sim_radar_point_static_static_moving.npz"
    )
    if oracle_static_static_moving_path.exists() and candidate_static_static_moving_path.exists():
        oracle_static_static_moving = _load_npz_dict(oracle_static_static_moving_path)
        candidate_static_static_moving = _load_npz_dict(candidate_static_static_moving_path)
        static_static_moving_timestamp_exact = bool(
            np.array_equal(
                oracle_static_static_moving["timestamp"],
                candidate_static_static_moving["timestamp"],
            )
        )
        static_static_moving_baseband_dtype_match = (
            oracle_static_static_moving["baseband"].dtype
            == candidate_static_static_moving["baseband"].dtype
        )
        static_static_moving_baseband_shape_match = (
            oracle_static_static_moving["baseband"].shape
            == candidate_static_static_moving["baseband"].shape
        )
        static_static_moving_noise_dtype_match = (
            oracle_static_static_moving["noise"].dtype
            == candidate_static_static_moving["noise"].dtype
        )
        static_static_moving_noise_shape_match = (
            oracle_static_static_moving["noise"].shape
            == candidate_static_static_moving["noise"].shape
        )
        oracle_static_static_moving_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_static_static_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_static_static_moving_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_static_static_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_static_static_moving["baseband"].shape[-1] // 2
        oracle_static_a_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        candidate_static_a_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        oracle_static_b_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(43, 52),
            valid_bins=valid_bins,
        )
        candidate_static_b_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(43, 52),
            valid_bins=valid_bins,
        )
        oracle_moving_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving["baseband"],
            doppler_slice=slice(0, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        candidate_moving_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving["baseband"],
            doppler_slice=slice(0, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_static_static_moving["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_static_static_moving["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_static_static_moving["baseband"],
            candidate_static_static_moving["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_static_static_moving_report = {
            "available": True,
            "timestamp_exact": static_static_moving_timestamp_exact,
            "baseband_dtype_match": static_static_moving_baseband_dtype_match,
            "baseband_shape_match": static_static_moving_baseband_shape_match,
            "noise_dtype_match": static_static_moving_noise_dtype_match,
            "noise_shape_match": static_static_moving_noise_shape_match,
            "oracle_interference_none": oracle_static_static_moving_interference_none,
            "candidate_interference_none": candidate_static_static_moving_interference_none,
            "oracle_static_a_peak": list(oracle_static_a_peak),
            "candidate_static_a_peak": list(candidate_static_a_peak),
            "oracle_static_b_peak": list(oracle_static_b_peak),
            "candidate_static_b_peak": list(candidate_static_b_peak),
            "oracle_moving_peak": list(oracle_moving_peak),
            "candidate_moving_peak": list(candidate_moving_peak),
            "static_a_peak_exact": oracle_static_a_peak == candidate_static_a_peak,
            "static_b_peak_exact": oracle_static_b_peak == candidate_static_b_peak,
            "moving_peak_exact": oracle_moving_peak == candidate_moving_peak,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    static_static_moving_timestamp_exact,
                    static_static_moving_baseband_dtype_match,
                    static_static_moving_baseband_shape_match,
                    static_static_moving_noise_dtype_match,
                    static_static_moving_noise_shape_match,
                    oracle_static_static_moving_interference_none,
                    candidate_static_static_moving_interference_none,
                    oracle_static_a_peak == candidate_static_a_peak,
                    oracle_static_b_peak == candidate_static_b_peak,
                    oracle_moving_peak == candidate_moving_peak,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_static_static_moving_negative_report = {"available": False, "overall": True}
    oracle_static_static_moving_negative_path = (
        oracle_dir / "artifacts" / "sim_radar_point_static_static_moving_negative.npz"
    )
    candidate_static_static_moving_negative_path = (
        candidate_dir / "artifacts" / "sim_radar_point_static_static_moving_negative.npz"
    )
    if (
        oracle_static_static_moving_negative_path.exists()
        and candidate_static_static_moving_negative_path.exists()
    ):
        oracle_static_static_moving_negative = _load_npz_dict(
            oracle_static_static_moving_negative_path
        )
        candidate_static_static_moving_negative = _load_npz_dict(
            candidate_static_static_moving_negative_path
        )
        static_static_moving_negative_timestamp_exact = bool(
            np.array_equal(
                oracle_static_static_moving_negative["timestamp"],
                candidate_static_static_moving_negative["timestamp"],
            )
        )
        static_static_moving_negative_baseband_dtype_match = (
            oracle_static_static_moving_negative["baseband"].dtype
            == candidate_static_static_moving_negative["baseband"].dtype
        )
        static_static_moving_negative_baseband_shape_match = (
            oracle_static_static_moving_negative["baseband"].shape
            == candidate_static_static_moving_negative["baseband"].shape
        )
        static_static_moving_negative_noise_dtype_match = (
            oracle_static_static_moving_negative["noise"].dtype
            == candidate_static_static_moving_negative["noise"].dtype
        )
        static_static_moving_negative_noise_shape_match = (
            oracle_static_static_moving_negative["noise"].shape
            == candidate_static_static_moving_negative["noise"].shape
        )
        oracle_static_static_moving_negative_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_static_static_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_static_static_moving_negative_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_static_static_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_static_static_moving_negative["baseband"].shape[-1] // 2
        oracle_static_a_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        candidate_static_a_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        oracle_static_b_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(43, 52),
            valid_bins=valid_bins,
        )
        candidate_static_b_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(43, 52),
            valid_bins=valid_bins,
        )
        oracle_moving_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_negative["baseband"],
            doppler_slice=slice(2, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        candidate_moving_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_negative["baseband"],
            doppler_slice=slice(2, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_static_static_moving_negative["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_static_static_moving_negative["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_static_static_moving_negative["baseband"],
            candidate_static_static_moving_negative["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_static_static_moving_negative_report = {
            "available": True,
            "timestamp_exact": static_static_moving_negative_timestamp_exact,
            "baseband_dtype_match": static_static_moving_negative_baseband_dtype_match,
            "baseband_shape_match": static_static_moving_negative_baseband_shape_match,
            "noise_dtype_match": static_static_moving_negative_noise_dtype_match,
            "noise_shape_match": static_static_moving_negative_noise_shape_match,
            "oracle_interference_none": oracle_static_static_moving_negative_interference_none,
            "candidate_interference_none": candidate_static_static_moving_negative_interference_none,
            "oracle_static_a_peak": list(oracle_static_a_peak),
            "candidate_static_a_peak": list(candidate_static_a_peak),
            "oracle_static_b_peak": list(oracle_static_b_peak),
            "candidate_static_b_peak": list(candidate_static_b_peak),
            "oracle_moving_peak": list(oracle_moving_peak),
            "candidate_moving_peak": list(candidate_moving_peak),
            "static_a_peak_exact": oracle_static_a_peak == candidate_static_a_peak,
            "static_b_peak_exact": oracle_static_b_peak == candidate_static_b_peak,
            "moving_peak_exact": oracle_moving_peak == candidate_moving_peak,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    static_static_moving_negative_timestamp_exact,
                    static_static_moving_negative_baseband_dtype_match,
                    static_static_moving_negative_baseband_shape_match,
                    static_static_moving_negative_noise_dtype_match,
                    static_static_moving_negative_noise_shape_match,
                    oracle_static_static_moving_negative_interference_none,
                    candidate_static_static_moving_negative_interference_none,
                    oracle_static_a_peak == candidate_static_a_peak,
                    oracle_static_b_peak == candidate_static_b_peak,
                    oracle_moving_peak == candidate_moving_peak,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_static_static_moving_moving_report = {"available": False, "overall": True}
    oracle_static_static_moving_moving_path = (
        oracle_dir / "artifacts" / "sim_radar_point_static_static_moving_moving.npz"
    )
    candidate_static_static_moving_moving_path = (
        candidate_dir / "artifacts" / "sim_radar_point_static_static_moving_moving.npz"
    )
    if (
        oracle_static_static_moving_moving_path.exists()
        and candidate_static_static_moving_moving_path.exists()
    ):
        oracle_static_static_moving_moving = _load_npz_dict(
            oracle_static_static_moving_moving_path
        )
        candidate_static_static_moving_moving = _load_npz_dict(
            candidate_static_static_moving_moving_path
        )
        static_static_moving_moving_timestamp_exact = bool(
            np.array_equal(
                oracle_static_static_moving_moving["timestamp"],
                candidate_static_static_moving_moving["timestamp"],
            )
        )
        static_static_moving_moving_baseband_dtype_match = (
            oracle_static_static_moving_moving["baseband"].dtype
            == candidate_static_static_moving_moving["baseband"].dtype
        )
        static_static_moving_moving_baseband_shape_match = (
            oracle_static_static_moving_moving["baseband"].shape
            == candidate_static_static_moving_moving["baseband"].shape
        )
        static_static_moving_moving_noise_dtype_match = (
            oracle_static_static_moving_moving["noise"].dtype
            == candidate_static_static_moving_moving["noise"].dtype
        )
        static_static_moving_moving_noise_shape_match = (
            oracle_static_static_moving_moving["noise"].shape
            == candidate_static_static_moving_moving["noise"].shape
        )
        oracle_static_static_moving_moving_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_static_static_moving_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_static_static_moving_moving_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_static_static_moving_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_static_static_moving_moving["baseband"].shape[-1] // 2
        oracle_static_a_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        candidate_static_a_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        oracle_static_b_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(43, 52),
            valid_bins=valid_bins,
        )
        candidate_static_b_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_moving["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(43, 52),
            valid_bins=valid_bins,
        )
        oracle_moving_a_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_moving["baseband"],
            doppler_slice=slice(0, 4),
            range_slice=slice(8, 18),
            valid_bins=valid_bins,
        )
        candidate_moving_a_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_moving["baseband"],
            doppler_slice=slice(0, 4),
            range_slice=slice(8, 18),
            valid_bins=valid_bins,
        )
        oracle_moving_b_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_moving["baseband"],
            doppler_slice=slice(0, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        candidate_moving_b_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_moving["baseband"],
            doppler_slice=slice(0, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_static_static_moving_moving["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_static_static_moving_moving["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_static_static_moving_moving["baseband"],
            candidate_static_static_moving_moving["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_static_static_moving_moving_report = {
            "available": True,
            "timestamp_exact": static_static_moving_moving_timestamp_exact,
            "baseband_dtype_match": static_static_moving_moving_baseband_dtype_match,
            "baseband_shape_match": static_static_moving_moving_baseband_shape_match,
            "noise_dtype_match": static_static_moving_moving_noise_dtype_match,
            "noise_shape_match": static_static_moving_moving_noise_shape_match,
            "oracle_interference_none": oracle_static_static_moving_moving_interference_none,
            "candidate_interference_none": candidate_static_static_moving_moving_interference_none,
            "oracle_static_a_peak": list(oracle_static_a_peak),
            "candidate_static_a_peak": list(candidate_static_a_peak),
            "oracle_static_b_peak": list(oracle_static_b_peak),
            "candidate_static_b_peak": list(candidate_static_b_peak),
            "oracle_moving_a_peak": list(oracle_moving_a_peak),
            "candidate_moving_a_peak": list(candidate_moving_a_peak),
            "oracle_moving_b_peak": list(oracle_moving_b_peak),
            "candidate_moving_b_peak": list(candidate_moving_b_peak),
            "static_a_peak_exact": oracle_static_a_peak == candidate_static_a_peak,
            "static_b_peak_exact": oracle_static_b_peak == candidate_static_b_peak,
            "moving_a_peak_exact": oracle_moving_a_peak == candidate_moving_a_peak,
            "moving_b_peak_exact": oracle_moving_b_peak == candidate_moving_b_peak,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    static_static_moving_moving_timestamp_exact,
                    static_static_moving_moving_baseband_dtype_match,
                    static_static_moving_moving_baseband_shape_match,
                    static_static_moving_moving_noise_dtype_match,
                    static_static_moving_moving_noise_shape_match,
                    oracle_static_static_moving_moving_interference_none,
                    candidate_static_static_moving_moving_interference_none,
                    oracle_static_a_peak == candidate_static_a_peak,
                    oracle_static_b_peak == candidate_static_b_peak,
                    oracle_moving_a_peak == candidate_moving_a_peak,
                    oracle_moving_b_peak == candidate_moving_b_peak,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_static_static_moving_moving_negative_report = {
        "available": False,
        "overall": True,
    }
    oracle_static_static_moving_moving_negative_path = (
        oracle_dir / "artifacts" / "sim_radar_point_static_static_moving_moving_negative.npz"
    )
    candidate_static_static_moving_moving_negative_path = (
        candidate_dir / "artifacts" / "sim_radar_point_static_static_moving_moving_negative.npz"
    )
    if (
        oracle_static_static_moving_moving_negative_path.exists()
        and candidate_static_static_moving_moving_negative_path.exists()
    ):
        oracle_static_static_moving_moving_negative = _load_npz_dict(
            oracle_static_static_moving_moving_negative_path
        )
        candidate_static_static_moving_moving_negative = _load_npz_dict(
            candidate_static_static_moving_moving_negative_path
        )
        static_static_moving_moving_negative_timestamp_exact = bool(
            np.array_equal(
                oracle_static_static_moving_moving_negative["timestamp"],
                candidate_static_static_moving_moving_negative["timestamp"],
            )
        )
        static_static_moving_moving_negative_baseband_dtype_match = (
            oracle_static_static_moving_moving_negative["baseband"].dtype
            == candidate_static_static_moving_moving_negative["baseband"].dtype
        )
        static_static_moving_moving_negative_baseband_shape_match = (
            oracle_static_static_moving_moving_negative["baseband"].shape
            == candidate_static_static_moving_moving_negative["baseband"].shape
        )
        static_static_moving_moving_negative_noise_dtype_match = (
            oracle_static_static_moving_moving_negative["noise"].dtype
            == candidate_static_static_moving_moving_negative["noise"].dtype
        )
        static_static_moving_moving_negative_noise_shape_match = (
            oracle_static_static_moving_moving_negative["noise"].shape
            == candidate_static_static_moving_moving_negative["noise"].shape
        )
        oracle_static_static_moving_moving_negative_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_static_static_moving_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_static_static_moving_moving_negative_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_static_static_moving_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_static_static_moving_moving_negative["baseband"].shape[-1] // 2
        oracle_static_a_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        candidate_static_a_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(23, 32),
            valid_bins=valid_bins,
        )
        oracle_static_b_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(43, 52),
            valid_bins=valid_bins,
        )
        candidate_static_b_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_moving_negative["baseband"],
            doppler_slice=slice(0, 2),
            range_slice=slice(43, 52),
            valid_bins=valid_bins,
        )
        oracle_moving_a_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_moving_negative["baseband"],
            doppler_slice=slice(2, 4),
            range_slice=slice(8, 18),
            valid_bins=valid_bins,
        )
        candidate_moving_a_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_moving_negative["baseband"],
            doppler_slice=slice(2, 4),
            range_slice=slice(8, 18),
            valid_bins=valid_bins,
        )
        oracle_moving_b_peak = _local_range_doppler_peak_indices(
            oracle_static_static_moving_moving_negative["baseband"],
            doppler_slice=slice(2, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        candidate_moving_b_peak = _local_range_doppler_peak_indices(
            candidate_static_static_moving_moving_negative["baseband"],
            doppler_slice=slice(2, 4),
            range_slice=slice(63, 72),
            valid_bins=valid_bins,
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_static_static_moving_moving_negative["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_static_static_moving_moving_negative["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_static_static_moving_moving_negative["baseband"],
            candidate_static_static_moving_moving_negative["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_static_static_moving_moving_negative_report = {
            "available": True,
            "timestamp_exact": static_static_moving_moving_negative_timestamp_exact,
            "baseband_dtype_match": static_static_moving_moving_negative_baseband_dtype_match,
            "baseband_shape_match": static_static_moving_moving_negative_baseband_shape_match,
            "noise_dtype_match": static_static_moving_moving_negative_noise_dtype_match,
            "noise_shape_match": static_static_moving_moving_negative_noise_shape_match,
            "oracle_interference_none": (
                oracle_static_static_moving_moving_negative_interference_none
            ),
            "candidate_interference_none": (
                candidate_static_static_moving_moving_negative_interference_none
            ),
            "oracle_static_a_peak": list(oracle_static_a_peak),
            "candidate_static_a_peak": list(candidate_static_a_peak),
            "oracle_static_b_peak": list(oracle_static_b_peak),
            "candidate_static_b_peak": list(candidate_static_b_peak),
            "oracle_moving_a_peak": list(oracle_moving_a_peak),
            "candidate_moving_a_peak": list(candidate_moving_a_peak),
            "oracle_moving_b_peak": list(oracle_moving_b_peak),
            "candidate_moving_b_peak": list(candidate_moving_b_peak),
            "static_a_peak_exact": oracle_static_a_peak == candidate_static_a_peak,
            "static_b_peak_exact": oracle_static_b_peak == candidate_static_b_peak,
            "moving_a_peak_exact": oracle_moving_a_peak == candidate_moving_a_peak,
            "moving_b_peak_exact": oracle_moving_b_peak == candidate_moving_b_peak,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    static_static_moving_moving_negative_timestamp_exact,
                    static_static_moving_moving_negative_baseband_dtype_match,
                    static_static_moving_moving_negative_baseband_shape_match,
                    static_static_moving_moving_negative_noise_dtype_match,
                    static_static_moving_moving_negative_noise_shape_match,
                    oracle_static_static_moving_moving_negative_interference_none,
                    candidate_static_static_moving_moving_negative_interference_none,
                    oracle_static_a_peak == candidate_static_a_peak,
                    oracle_static_b_peak == candidate_static_b_peak,
                    oracle_moving_a_peak == candidate_moving_a_peak,
                    oracle_moving_b_peak == candidate_moving_b_peak,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_moving_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_static_moving_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving.npz"
    )
    candidate_mimo_offaxis_static_moving_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving.npz"
    )
    if (
        oracle_mimo_offaxis_static_moving_path.exists()
        and candidate_mimo_offaxis_static_moving_path.exists()
    ):
        oracle_mimo_offaxis_static_moving = _load_npz_dict(oracle_mimo_offaxis_static_moving_path)
        candidate_mimo_offaxis_static_moving = _load_npz_dict(
            candidate_mimo_offaxis_static_moving_path
        )
        mimo_offaxis_static_moving_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_moving["timestamp"],
                candidate_mimo_offaxis_static_moving["timestamp"],
            )
        )
        mimo_offaxis_static_moving_baseband_dtype_match = (
            oracle_mimo_offaxis_static_moving["baseband"].dtype
            == candidate_mimo_offaxis_static_moving["baseband"].dtype
        )
        mimo_offaxis_static_moving_baseband_shape_match = (
            oracle_mimo_offaxis_static_moving["baseband"].shape
            == candidate_mimo_offaxis_static_moving["baseband"].shape
        )
        mimo_offaxis_static_moving_noise_dtype_match = (
            oracle_mimo_offaxis_static_moving["noise"].dtype
            == candidate_mimo_offaxis_static_moving["noise"].dtype
        )
        mimo_offaxis_static_moving_noise_shape_match = (
            oracle_mimo_offaxis_static_moving["noise"].shape
            == candidate_mimo_offaxis_static_moving["noise"].shape
        )
        oracle_mimo_offaxis_static_moving_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_moving_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_mimo_offaxis_static_moving["baseband"].shape[-1] // 2
        oracle_static_peaks = []
        candidate_static_peaks = []
        oracle_moving_peaks = []
        candidate_moving_peaks = []
        for channel_index in range(oracle_mimo_offaxis_static_moving["baseband"].shape[0]):
            oracle_static_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_moving["baseband"][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_moving["baseband"][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_moving["baseband"][0, 0, phase_sample_index]
        candidate_ref = candidate_mimo_offaxis_static_moving["baseband"][0, 0, phase_sample_index]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_moving["baseband"][:, 0, phase_sample_index]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_moving["baseband"][:, 0, phase_sample_index]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_mimo_offaxis_static_moving["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_mimo_offaxis_static_moving["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_moving["baseband"],
            candidate_mimo_offaxis_static_moving["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_moving_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_moving_timestamp_exact,
            "baseband_dtype_match": mimo_offaxis_static_moving_baseband_dtype_match,
            "baseband_shape_match": mimo_offaxis_static_moving_baseband_shape_match,
            "noise_dtype_match": mimo_offaxis_static_moving_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_moving_noise_shape_match,
            "oracle_interference_none": oracle_mimo_offaxis_static_moving_interference_none,
            "candidate_interference_none": candidate_mimo_offaxis_static_moving_interference_none,
            "phase_sample_index": phase_sample_index,
            "oracle_static_peaks": oracle_static_peaks,
            "candidate_static_peaks": candidate_static_peaks,
            "oracle_moving_peaks": oracle_moving_peaks,
            "candidate_moving_peaks": candidate_moving_peaks,
            "static_peaks_exact": oracle_static_peaks == candidate_static_peaks,
            "moving_peaks_exact": oracle_moving_peaks == candidate_moving_peaks,
            "oracle_channel_phase_degrees": [float(value) for value in oracle_rel_phase_deg],
            "candidate_channel_phase_degrees": [float(value) for value in candidate_rel_phase_deg],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_moving_timestamp_exact,
                    mimo_offaxis_static_moving_baseband_dtype_match,
                    mimo_offaxis_static_moving_baseband_shape_match,
                    mimo_offaxis_static_moving_noise_dtype_match,
                    mimo_offaxis_static_moving_noise_shape_match,
                    oracle_mimo_offaxis_static_moving_interference_none,
                    candidate_mimo_offaxis_static_moving_interference_none,
                    oracle_static_peaks == candidate_static_peaks,
                    oracle_moving_peaks == candidate_moving_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_moving_negative_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_static_moving_negative_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_negative.npz"
    )
    candidate_mimo_offaxis_static_moving_negative_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_negative.npz"
    )
    if (
        oracle_mimo_offaxis_static_moving_negative_path.exists()
        and candidate_mimo_offaxis_static_moving_negative_path.exists()
    ):
        oracle_mimo_offaxis_static_moving_negative = _load_npz_dict(
            oracle_mimo_offaxis_static_moving_negative_path
        )
        candidate_mimo_offaxis_static_moving_negative = _load_npz_dict(
            candidate_mimo_offaxis_static_moving_negative_path
        )
        mimo_offaxis_static_moving_negative_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_moving_negative["timestamp"],
                candidate_mimo_offaxis_static_moving_negative["timestamp"],
            )
        )
        mimo_offaxis_static_moving_negative_baseband_dtype_match = (
            oracle_mimo_offaxis_static_moving_negative["baseband"].dtype
            == candidate_mimo_offaxis_static_moving_negative["baseband"].dtype
        )
        mimo_offaxis_static_moving_negative_baseband_shape_match = (
            oracle_mimo_offaxis_static_moving_negative["baseband"].shape
            == candidate_mimo_offaxis_static_moving_negative["baseband"].shape
        )
        mimo_offaxis_static_moving_negative_noise_dtype_match = (
            oracle_mimo_offaxis_static_moving_negative["noise"].dtype
            == candidate_mimo_offaxis_static_moving_negative["noise"].dtype
        )
        mimo_offaxis_static_moving_negative_noise_shape_match = (
            oracle_mimo_offaxis_static_moving_negative["noise"].shape
            == candidate_mimo_offaxis_static_moving_negative["noise"].shape
        )
        oracle_mimo_offaxis_static_moving_negative_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_moving_negative_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_mimo_offaxis_static_moving_negative["baseband"].shape[-1] // 2
        oracle_static_peaks = []
        candidate_static_peaks = []
        oracle_moving_peaks = []
        candidate_moving_peaks = []
        for channel_index in range(oracle_mimo_offaxis_static_moving_negative["baseband"].shape[0]):
            oracle_static_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_moving_negative["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_moving_negative["baseband"][
            0, 0, phase_sample_index
        ]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_moving_negative["baseband"][
                    :, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_moving_negative["baseband"][
                    :, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_mimo_offaxis_static_moving_negative["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_mimo_offaxis_static_moving_negative["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_moving_negative["baseband"],
            candidate_mimo_offaxis_static_moving_negative["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_moving_negative_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_moving_negative_timestamp_exact,
            "baseband_dtype_match": mimo_offaxis_static_moving_negative_baseband_dtype_match,
            "baseband_shape_match": mimo_offaxis_static_moving_negative_baseband_shape_match,
            "noise_dtype_match": mimo_offaxis_static_moving_negative_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_moving_negative_noise_shape_match,
            "oracle_interference_none": oracle_mimo_offaxis_static_moving_negative_interference_none,
            "candidate_interference_none": candidate_mimo_offaxis_static_moving_negative_interference_none,
            "phase_sample_index": phase_sample_index,
            "oracle_static_peaks": oracle_static_peaks,
            "candidate_static_peaks": candidate_static_peaks,
            "oracle_moving_peaks": oracle_moving_peaks,
            "candidate_moving_peaks": candidate_moving_peaks,
            "static_peaks_exact": oracle_static_peaks == candidate_static_peaks,
            "moving_peaks_exact": oracle_moving_peaks == candidate_moving_peaks,
            "oracle_channel_phase_degrees": [float(value) for value in oracle_rel_phase_deg],
            "candidate_channel_phase_degrees": [float(value) for value in candidate_rel_phase_deg],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_moving_negative_timestamp_exact,
                    mimo_offaxis_static_moving_negative_baseband_dtype_match,
                    mimo_offaxis_static_moving_negative_baseband_shape_match,
                    mimo_offaxis_static_moving_negative_noise_dtype_match,
                    mimo_offaxis_static_moving_negative_noise_shape_match,
                    oracle_mimo_offaxis_static_moving_negative_interference_none,
                    candidate_mimo_offaxis_static_moving_negative_interference_none,
                    oracle_static_peaks == candidate_static_peaks,
                    oracle_moving_peaks == candidate_moving_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_moving_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_static_static_moving_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving.npz"
    )
    candidate_mimo_offaxis_static_static_moving_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_moving_path.exists()
        and candidate_mimo_offaxis_static_static_moving_path.exists()
    ):
        oracle_mimo_offaxis_static_static_moving = _load_npz_dict(
            oracle_mimo_offaxis_static_static_moving_path
        )
        candidate_mimo_offaxis_static_static_moving = _load_npz_dict(
            candidate_mimo_offaxis_static_static_moving_path
        )
        mimo_offaxis_static_static_moving_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving["timestamp"],
                candidate_mimo_offaxis_static_static_moving["timestamp"],
            )
        )
        mimo_offaxis_static_static_moving_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_moving["baseband"].dtype
            == candidate_mimo_offaxis_static_static_moving["baseband"].dtype
        )
        mimo_offaxis_static_static_moving_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_moving["baseband"].shape
            == candidate_mimo_offaxis_static_static_moving["baseband"].shape
        )
        mimo_offaxis_static_static_moving_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_moving["noise"].dtype
            == candidate_mimo_offaxis_static_static_moving["noise"].dtype
        )
        mimo_offaxis_static_static_moving_noise_shape_match = (
            oracle_mimo_offaxis_static_static_moving["noise"].shape
            == candidate_mimo_offaxis_static_static_moving["noise"].shape
        )
        oracle_mimo_offaxis_static_static_moving_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_moving_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = oracle_mimo_offaxis_static_static_moving["baseband"].shape[-1] // 2
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_peaks = []
        candidate_moving_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_moving["baseband"].shape[0]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_moving["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_static_moving["baseband"][
            0, 0, phase_sample_index
        ]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_moving["baseband"][
                    :, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_moving["baseband"][
                    :, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_mimo_offaxis_static_static_moving["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_mimo_offaxis_static_static_moving["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_moving["baseband"],
            candidate_mimo_offaxis_static_static_moving["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_moving_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_static_moving_timestamp_exact,
            "baseband_dtype_match": mimo_offaxis_static_static_moving_baseband_dtype_match,
            "baseband_shape_match": mimo_offaxis_static_static_moving_baseband_shape_match,
            "noise_dtype_match": mimo_offaxis_static_static_moving_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_static_moving_noise_shape_match,
            "oracle_interference_none": oracle_mimo_offaxis_static_static_moving_interference_none,
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_moving_interference_none
            ),
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_peaks": oracle_moving_peaks,
            "candidate_moving_peaks": candidate_moving_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_peaks_exact": oracle_moving_peaks == candidate_moving_peaks,
            "oracle_channel_phase_degrees": [float(value) for value in oracle_rel_phase_deg],
            "candidate_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_moving_timestamp_exact,
                    mimo_offaxis_static_static_moving_baseband_dtype_match,
                    mimo_offaxis_static_static_moving_baseband_shape_match,
                    mimo_offaxis_static_static_moving_noise_dtype_match,
                    mimo_offaxis_static_static_moving_noise_shape_match,
                    oracle_mimo_offaxis_static_static_moving_interference_none,
                    candidate_mimo_offaxis_static_static_moving_interference_none,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_peaks == candidate_moving_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_moving_negative_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_static_moving_negative_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving_negative.npz"
    )
    candidate_mimo_offaxis_static_static_moving_negative_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_negative.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_moving_negative_path.exists()
        and candidate_mimo_offaxis_static_static_moving_negative_path.exists()
    ):
        oracle_mimo_offaxis_static_static_moving_negative = _load_npz_dict(
            oracle_mimo_offaxis_static_static_moving_negative_path
        )
        candidate_mimo_offaxis_static_static_moving_negative = _load_npz_dict(
            candidate_mimo_offaxis_static_static_moving_negative_path
        )
        mimo_offaxis_static_static_moving_negative_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_negative["timestamp"],
                candidate_mimo_offaxis_static_static_moving_negative["timestamp"],
            )
        )
        mimo_offaxis_static_static_moving_negative_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_negative["baseband"].dtype
            == candidate_mimo_offaxis_static_static_moving_negative["baseband"].dtype
        )
        mimo_offaxis_static_static_moving_negative_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_moving_negative["baseband"].shape
            == candidate_mimo_offaxis_static_static_moving_negative["baseband"].shape
        )
        mimo_offaxis_static_static_moving_negative_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_negative["noise"].dtype
            == candidate_mimo_offaxis_static_static_moving_negative["noise"].dtype
        )
        mimo_offaxis_static_static_moving_negative_noise_shape_match = (
            oracle_mimo_offaxis_static_static_moving_negative["noise"].shape
            == candidate_mimo_offaxis_static_static_moving_negative["noise"].shape
        )
        oracle_mimo_offaxis_static_static_moving_negative_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_moving_negative_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_negative", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = (
            oracle_mimo_offaxis_static_static_moving_negative["baseband"].shape[-1] // 2
        )
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_peaks = []
        candidate_moving_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_moving_negative["baseband"].shape[0]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_negative["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_moving_negative["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_static_moving_negative["baseband"][
            0, 0, phase_sample_index
        ]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_moving_negative["baseband"][
                    :, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_moving_negative["baseband"][
                    :, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_static_moving_negative["baseband"], axis=-1
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_static_moving_negative["baseband"], axis=-1
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_moving_negative["baseband"],
            candidate_mimo_offaxis_static_static_moving_negative["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_moving_negative_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_static_moving_negative_timestamp_exact,
            "baseband_dtype_match": (
                mimo_offaxis_static_static_moving_negative_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_static_moving_negative_baseband_shape_match
            ),
            "noise_dtype_match": mimo_offaxis_static_static_moving_negative_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_static_moving_negative_noise_shape_match,
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_static_moving_negative_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_moving_negative_interference_none
            ),
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_peaks": oracle_moving_peaks,
            "candidate_moving_peaks": candidate_moving_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_peaks_exact": oracle_moving_peaks == candidate_moving_peaks,
            "oracle_channel_phase_degrees": [float(value) for value in oracle_rel_phase_deg],
            "candidate_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_moving_negative_timestamp_exact,
                    mimo_offaxis_static_static_moving_negative_baseband_dtype_match,
                    mimo_offaxis_static_static_moving_negative_baseband_shape_match,
                    mimo_offaxis_static_static_moving_negative_noise_dtype_match,
                    mimo_offaxis_static_static_moving_negative_noise_shape_match,
                    oracle_mimo_offaxis_static_static_moving_negative_interference_none,
                    candidate_mimo_offaxis_static_static_moving_negative_interference_none,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_peaks == candidate_moving_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_moving_moving_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_static_moving_moving_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving_moving.npz"
    )
    candidate_mimo_offaxis_static_static_moving_moving_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_moving.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_moving_moving_path.exists()
        and candidate_mimo_offaxis_static_static_moving_moving_path.exists()
    ):
        oracle_mimo_offaxis_static_static_moving_moving = _load_npz_dict(
            oracle_mimo_offaxis_static_static_moving_moving_path
        )
        candidate_mimo_offaxis_static_static_moving_moving = _load_npz_dict(
            candidate_mimo_offaxis_static_static_moving_moving_path
        )
        mimo_offaxis_static_static_moving_moving_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_moving["timestamp"],
                candidate_mimo_offaxis_static_static_moving_moving["timestamp"],
            )
        )
        mimo_offaxis_static_static_moving_moving_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_moving["baseband"].dtype
            == candidate_mimo_offaxis_static_static_moving_moving["baseband"].dtype
        )
        mimo_offaxis_static_static_moving_moving_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_moving_moving["baseband"].shape
            == candidate_mimo_offaxis_static_static_moving_moving["baseband"].shape
        )
        mimo_offaxis_static_static_moving_moving_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_moving["noise"].dtype
            == candidate_mimo_offaxis_static_static_moving_moving["noise"].dtype
        )
        mimo_offaxis_static_static_moving_moving_noise_shape_match = (
            oracle_mimo_offaxis_static_static_moving_moving["noise"].shape
            == candidate_mimo_offaxis_static_static_moving_moving["noise"].shape
        )
        oracle_mimo_offaxis_static_static_moving_moving_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_moving_moving_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_moving", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = (
            oracle_mimo_offaxis_static_static_moving_moving["baseband"].shape[-1] // 2
        )
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_a_peaks = []
        candidate_moving_a_peaks = []
        oracle_moving_b_peaks = []
        candidate_moving_b_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_moving_moving["baseband"].shape[0]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_moving_moving["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_static_moving_moving["baseband"][
            0, 0, phase_sample_index
        ]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_moving_moving["baseband"][
                    :, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_moving_moving["baseband"][
                    :, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_static_moving_moving["baseband"], axis=-1
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_static_moving_moving["baseband"], axis=-1
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_moving_moving["baseband"],
            candidate_mimo_offaxis_static_static_moving_moving["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_moving_moving_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_static_moving_moving_timestamp_exact,
            "baseband_dtype_match": (
                mimo_offaxis_static_static_moving_moving_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_static_moving_moving_baseband_shape_match
            ),
            "noise_dtype_match": mimo_offaxis_static_static_moving_moving_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_static_moving_moving_noise_shape_match,
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_static_moving_moving_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_moving_moving_interference_none
            ),
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_a_peaks": oracle_moving_a_peaks,
            "candidate_moving_a_peaks": candidate_moving_a_peaks,
            "oracle_moving_b_peaks": oracle_moving_b_peaks,
            "candidate_moving_b_peaks": candidate_moving_b_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_a_peaks_exact": oracle_moving_a_peaks == candidate_moving_a_peaks,
            "moving_b_peaks_exact": oracle_moving_b_peaks == candidate_moving_b_peaks,
            "oracle_channel_phase_degrees": [float(value) for value in oracle_rel_phase_deg],
            "candidate_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_moving_moving_timestamp_exact,
                    mimo_offaxis_static_static_moving_moving_baseband_dtype_match,
                    mimo_offaxis_static_static_moving_moving_baseband_shape_match,
                    mimo_offaxis_static_static_moving_moving_noise_dtype_match,
                    mimo_offaxis_static_static_moving_moving_noise_shape_match,
                    oracle_mimo_offaxis_static_static_moving_moving_interference_none,
                    candidate_mimo_offaxis_static_static_moving_moving_interference_none,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_a_peaks == candidate_moving_a_peaks,
                    oracle_moving_b_peaks == candidate_moving_b_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_mixed_sign_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_static_mixed_sign_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_mixed_sign.npz"
    )
    candidate_mimo_offaxis_static_static_mixed_sign_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_mixed_sign.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_mixed_sign_path.exists()
        and candidate_mimo_offaxis_static_static_mixed_sign_path.exists()
    ):
        oracle_mimo_offaxis_static_static_mixed_sign = _load_npz_dict(
            oracle_mimo_offaxis_static_static_mixed_sign_path
        )
        candidate_mimo_offaxis_static_static_mixed_sign = _load_npz_dict(
            candidate_mimo_offaxis_static_static_mixed_sign_path
        )
        mimo_offaxis_static_static_mixed_sign_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_mixed_sign["timestamp"],
                candidate_mimo_offaxis_static_static_mixed_sign["timestamp"],
            )
        )
        mimo_offaxis_static_static_mixed_sign_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_mixed_sign["baseband"].dtype
            == candidate_mimo_offaxis_static_static_mixed_sign["baseband"].dtype
        )
        mimo_offaxis_static_static_mixed_sign_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_mixed_sign["baseband"].shape
            == candidate_mimo_offaxis_static_static_mixed_sign["baseband"].shape
        )
        mimo_offaxis_static_static_mixed_sign_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_mixed_sign["noise"].dtype
            == candidate_mimo_offaxis_static_static_mixed_sign["noise"].dtype
        )
        mimo_offaxis_static_static_mixed_sign_noise_shape_match = (
            oracle_mimo_offaxis_static_static_mixed_sign["noise"].shape
            == candidate_mimo_offaxis_static_static_mixed_sign["noise"].shape
        )
        oracle_mimo_offaxis_static_static_mixed_sign_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_mixed_sign", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_mixed_sign_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_mixed_sign", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        valid_bins = (
            oracle_mimo_offaxis_static_static_mixed_sign["baseband"].shape[-1] // 2
        )
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_pos_peaks = []
        candidate_moving_pos_peaks = []
        oracle_moving_neg_peaks = []
        candidate_moving_neg_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_mixed_sign["baseband"].shape[0]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_mixed_sign["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_mixed_sign["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_mixed_sign["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_mixed_sign["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_pos_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_mixed_sign["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_pos_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_mixed_sign["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_neg_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_mixed_sign["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_neg_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_mixed_sign["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        oracle_moving_neg_peaks = [
            [int(peak[0] + 2), int(peak[1])] for peak in oracle_moving_neg_peaks
        ]
        candidate_moving_neg_peaks = [
            [int(peak[0] + 2), int(peak[1])] for peak in candidate_moving_neg_peaks
        ]
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_mixed_sign["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_static_mixed_sign["baseband"][
            0, 0, phase_sample_index
        ]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_mixed_sign["baseband"][
                    :, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_mixed_sign["baseband"][
                    :, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_static_mixed_sign["baseband"], axis=-1
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_static_mixed_sign["baseband"], axis=-1
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_mixed_sign["baseband"],
            candidate_mimo_offaxis_static_static_mixed_sign["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_mixed_sign_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_static_mixed_sign_timestamp_exact,
            "baseband_dtype_match": (
                mimo_offaxis_static_static_mixed_sign_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_static_mixed_sign_baseband_shape_match
            ),
            "noise_dtype_match": mimo_offaxis_static_static_mixed_sign_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_static_mixed_sign_noise_shape_match,
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_static_mixed_sign_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_mixed_sign_interference_none
            ),
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_pos_peaks": oracle_moving_pos_peaks,
            "candidate_moving_pos_peaks": candidate_moving_pos_peaks,
            "oracle_moving_neg_peaks": oracle_moving_neg_peaks,
            "candidate_moving_neg_peaks": candidate_moving_neg_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_pos_peaks_exact": oracle_moving_pos_peaks == candidate_moving_pos_peaks,
            "moving_neg_peaks_exact": oracle_moving_neg_peaks == candidate_moving_neg_peaks,
            "oracle_channel_phase_degrees": [float(value) for value in oracle_rel_phase_deg],
            "candidate_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_mixed_sign_timestamp_exact,
                    mimo_offaxis_static_static_mixed_sign_baseband_dtype_match,
                    mimo_offaxis_static_static_mixed_sign_baseband_shape_match,
                    mimo_offaxis_static_static_mixed_sign_noise_dtype_match,
                    mimo_offaxis_static_static_mixed_sign_noise_shape_match,
                    oracle_mimo_offaxis_static_static_mixed_sign_interference_none,
                    candidate_mimo_offaxis_static_static_mixed_sign_interference_none,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_pos_peaks == candidate_moving_pos_peaks,
                    oracle_moving_neg_peaks == candidate_moving_neg_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_mixed_sign_multiframe_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_static_mixed_sign_multiframe_path = (
        oracle_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_mixed_sign_multiframe.npz"
    )
    candidate_mimo_offaxis_static_static_mixed_sign_multiframe_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_mixed_sign_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_mixed_sign_multiframe_path.exists()
        and candidate_mimo_offaxis_static_static_mixed_sign_multiframe_path.exists()
    ):
        oracle_mimo_offaxis_static_static_mixed_sign_multiframe = _load_npz_dict(
            oracle_mimo_offaxis_static_static_mixed_sign_multiframe_path
        )
        candidate_mimo_offaxis_static_static_mixed_sign_multiframe = _load_npz_dict(
            candidate_mimo_offaxis_static_static_mixed_sign_multiframe_path
        )
        mimo_offaxis_static_static_mixed_sign_multiframe_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_mixed_sign_multiframe["timestamp"],
                candidate_mimo_offaxis_static_static_mixed_sign_multiframe[
                    "timestamp"
                ],
            )
        )
        mimo_offaxis_static_static_mixed_sign_multiframe_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"].dtype
            == candidate_mimo_offaxis_static_static_mixed_sign_multiframe[
                "baseband"
            ].dtype
        )
        mimo_offaxis_static_static_mixed_sign_multiframe_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"].shape
            == candidate_mimo_offaxis_static_static_mixed_sign_multiframe[
                "baseband"
            ].shape
        )
        mimo_offaxis_static_static_mixed_sign_multiframe_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_mixed_sign_multiframe["noise"].dtype
            == candidate_mimo_offaxis_static_static_mixed_sign_multiframe["noise"].dtype
        )
        mimo_offaxis_static_static_mixed_sign_multiframe_noise_shape_match = (
            oracle_mimo_offaxis_static_static_mixed_sign_multiframe["noise"].shape
            == candidate_mimo_offaxis_static_static_mixed_sign_multiframe["noise"].shape
        )
        oracle_mimo_offaxis_static_static_mixed_sign_multiframe_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_mixed_sign_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_mixed_sign_multiframe_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_mixed_sign_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_frame_offset_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_mixed_sign_multiframe["timestamp"][:4]
                + 1e-3,
                oracle_mimo_offaxis_static_static_mixed_sign_multiframe["timestamp"][4:],
            )
        )
        candidate_frame_offset_exact = bool(
            np.array_equal(
                candidate_mimo_offaxis_static_static_mixed_sign_multiframe["timestamp"][
                    :4
                ]
                + 1e-3,
                candidate_mimo_offaxis_static_static_mixed_sign_multiframe["timestamp"][
                    4:
                ],
            )
        )
        valid_bins = (
            oracle_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"].shape[
                -1
            ]
            // 2
        )
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_pos_peaks = []
        candidate_moving_pos_peaks = []
        oracle_moving_neg_peaks = []
        candidate_moving_neg_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"].shape[
                0
            ]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_pos_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_pos_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_neg_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_neg_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_mixed_sign_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        oracle_moving_neg_peaks = [
            [int(peak[0] + 2), int(peak[1])] for peak in oracle_moving_neg_peaks
        ]
        candidate_moving_neg_peaks = [
            [int(peak[0] + 2), int(peak[1])] for peak in candidate_moving_neg_peaks
        ]
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_static_mixed_sign_multiframe[
            "baseband"
        ][0, 0, phase_sample_index]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"],
            candidate_mimo_offaxis_static_static_mixed_sign_multiframe["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_mixed_sign_multiframe_report = {
            "available": True,
            "timestamp_exact": (
                mimo_offaxis_static_static_mixed_sign_multiframe_timestamp_exact
            ),
            "baseband_dtype_match": (
                mimo_offaxis_static_static_mixed_sign_multiframe_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_static_mixed_sign_multiframe_baseband_shape_match
            ),
            "noise_dtype_match": (
                mimo_offaxis_static_static_mixed_sign_multiframe_noise_dtype_match
            ),
            "noise_shape_match": (
                mimo_offaxis_static_static_mixed_sign_multiframe_noise_shape_match
            ),
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_static_mixed_sign_multiframe_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_mixed_sign_multiframe_interference_none
            ),
            "oracle_frame_offset_exact": oracle_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_frame_offset_exact,
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_pos_peaks": oracle_moving_pos_peaks,
            "candidate_moving_pos_peaks": candidate_moving_pos_peaks,
            "oracle_moving_neg_peaks": oracle_moving_neg_peaks,
            "candidate_moving_neg_peaks": candidate_moving_neg_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_pos_peaks_exact": (
                oracle_moving_pos_peaks == candidate_moving_pos_peaks
            ),
            "moving_neg_peaks_exact": (
                oracle_moving_neg_peaks == candidate_moving_neg_peaks
            ),
            "oracle_frame0_channel_phase_degrees": [
                float(value) for value in oracle_rel_phase_deg
            ],
            "candidate_frame0_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_mixed_sign_multiframe_timestamp_exact,
                    mimo_offaxis_static_static_mixed_sign_multiframe_baseband_dtype_match,
                    mimo_offaxis_static_static_mixed_sign_multiframe_baseband_shape_match,
                    mimo_offaxis_static_static_mixed_sign_multiframe_noise_dtype_match,
                    mimo_offaxis_static_static_mixed_sign_multiframe_noise_shape_match,
                    oracle_mimo_offaxis_static_static_mixed_sign_multiframe_interference_none,
                    candidate_mimo_offaxis_static_static_mixed_sign_multiframe_interference_none,
                    oracle_frame_offset_exact,
                    candidate_frame_offset_exact,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_pos_peaks == candidate_moving_pos_peaks,
                    oracle_moving_neg_peaks == candidate_moving_neg_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_moving_moving_negative_multiframe_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe_path = (
        oracle_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_moving_negative_multiframe.npz"
    )
    candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_moving_negative_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe_path.exists()
        and candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe_path.exists()
    ):
        oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe = _load_npz_dict(
            oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe_path
        )
        candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe = _load_npz_dict(
            candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe_path
        )
        mimo_offaxis_static_static_moving_moving_negative_multiframe_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "timestamp"
                ],
                candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "timestamp"
                ],
            )
        )
        mimo_offaxis_static_static_moving_moving_negative_multiframe_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "baseband"
            ].dtype
            == candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "baseband"
            ].dtype
        )
        mimo_offaxis_static_static_moving_moving_negative_multiframe_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "baseband"
            ].shape
            == candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "baseband"
            ].shape
        )
        mimo_offaxis_static_static_moving_moving_negative_multiframe_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "noise"
            ].dtype
            == candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "noise"
            ].dtype
        )
        mimo_offaxis_static_static_moving_moving_negative_multiframe_noise_shape_match = (
            oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "noise"
            ].shape
            == candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "noise"
            ].shape
        )
        oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_moving_negative_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_moving_negative_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_frame_offset_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "timestamp"
                ][:4]
                + 1e-3,
                oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "timestamp"
                ][4:],
            )
        )
        candidate_frame_offset_exact = bool(
            np.array_equal(
                candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "timestamp"
                ][:4]
                + 1e-3,
                candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "timestamp"
                ][4:],
            )
        )
        valid_bins = (
            oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "baseband"
            ].shape[-1]
            // 2
        )
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_a_peaks = []
        candidate_moving_a_peaks = []
        oracle_moving_b_peaks = []
        candidate_moving_b_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "baseband"
            ].shape[0]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        oracle_moving_a_peaks = [[int(peak[0]), int(peak[1])] for peak in oracle_moving_a_peaks]
        candidate_moving_a_peaks = [
            [int(peak[0]), int(peak[1])] for peak in candidate_moving_a_peaks
        ]
        oracle_moving_b_peaks = [[int(peak[0]), int(peak[1])] for peak in oracle_moving_b_peaks]
        candidate_moving_b_peaks = [
            [int(peak[0]), int(peak[1])] for peak in candidate_moving_b_peaks
        ]
        accepted_moving_b_peaks = {(3, 66), (3, 67)}
        oracle_moving_b_peaks_in_accepted_band = all(
            tuple(peak) in accepted_moving_b_peaks for peak in oracle_moving_b_peaks
        )
        candidate_moving_b_peaks_in_accepted_band = all(
            tuple(peak) in accepted_moving_b_peaks for peak in candidate_moving_b_peaks
        )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
            "baseband"
        ][0, 0, phase_sample_index]
        candidate_ref = candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
            "baseband"
        ][0, 0, phase_sample_index]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "baseband"
                ][:4, 0, phase_sample_index]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "baseband"
                ][:4, 0, phase_sample_index]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "baseband"
                ],
                axis=-1,
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                    "baseband"
                ],
                axis=-1,
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "baseband"
            ],
            candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe[
                "baseband"
            ],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_moving_moving_negative_multiframe_report = {
            "available": True,
            "timestamp_exact": (
                mimo_offaxis_static_static_moving_moving_negative_multiframe_timestamp_exact
            ),
            "baseband_dtype_match": (
                mimo_offaxis_static_static_moving_moving_negative_multiframe_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_static_moving_moving_negative_multiframe_baseband_shape_match
            ),
            "noise_dtype_match": (
                mimo_offaxis_static_static_moving_moving_negative_multiframe_noise_dtype_match
            ),
            "noise_shape_match": (
                mimo_offaxis_static_static_moving_moving_negative_multiframe_noise_shape_match
            ),
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe_interference_none
            ),
            "oracle_frame_offset_exact": oracle_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_frame_offset_exact,
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_a_peaks": oracle_moving_a_peaks,
            "candidate_moving_a_peaks": candidate_moving_a_peaks,
            "oracle_moving_b_peaks": oracle_moving_b_peaks,
            "candidate_moving_b_peaks": candidate_moving_b_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_a_peaks_exact": oracle_moving_a_peaks == candidate_moving_a_peaks,
            "moving_b_peaks_exact": oracle_moving_b_peaks == candidate_moving_b_peaks,
            "oracle_moving_b_peaks_in_accepted_band": (
                oracle_moving_b_peaks_in_accepted_band
            ),
            "candidate_moving_b_peaks_in_accepted_band": (
                candidate_moving_b_peaks_in_accepted_band
            ),
            "oracle_frame0_channel_phase_degrees": [
                float(value) for value in oracle_rel_phase_deg
            ],
            "candidate_frame0_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_moving_moving_negative_multiframe_timestamp_exact,
                    mimo_offaxis_static_static_moving_moving_negative_multiframe_baseband_dtype_match,
                    mimo_offaxis_static_static_moving_moving_negative_multiframe_baseband_shape_match,
                    mimo_offaxis_static_static_moving_moving_negative_multiframe_noise_dtype_match,
                    mimo_offaxis_static_static_moving_moving_negative_multiframe_noise_shape_match,
                    oracle_mimo_offaxis_static_static_moving_moving_negative_multiframe_interference_none,
                    candidate_mimo_offaxis_static_static_moving_moving_negative_multiframe_interference_none,
                    oracle_frame_offset_exact,
                    candidate_frame_offset_exact,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_a_peaks == candidate_moving_a_peaks,
                    oracle_moving_b_peaks_in_accepted_band,
                    candidate_moving_b_peaks_in_accepted_band,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_moving_moving_multiframe_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_static_moving_moving_multiframe_path = (
        oracle_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_moving_multiframe.npz"
    )
    candidate_mimo_offaxis_static_static_moving_moving_multiframe_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_moving_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_moving_moving_multiframe_path.exists()
        and candidate_mimo_offaxis_static_static_moving_moving_multiframe_path.exists()
    ):
        oracle_mimo_offaxis_static_static_moving_moving_multiframe = _load_npz_dict(
            oracle_mimo_offaxis_static_static_moving_moving_multiframe_path
        )
        candidate_mimo_offaxis_static_static_moving_moving_multiframe = _load_npz_dict(
            candidate_mimo_offaxis_static_static_moving_moving_multiframe_path
        )
        mimo_offaxis_static_static_moving_moving_multiframe_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_moving_multiframe["timestamp"],
                candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                    "timestamp"
                ],
            )
        )
        mimo_offaxis_static_static_moving_moving_multiframe_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_moving_multiframe["baseband"].dtype
            == candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                "baseband"
            ].dtype
        )
        mimo_offaxis_static_static_moving_moving_multiframe_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_moving_moving_multiframe["baseband"].shape
            == candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                "baseband"
            ].shape
        )
        mimo_offaxis_static_static_moving_moving_multiframe_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_moving_multiframe["noise"].dtype
            == candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                "noise"
            ].dtype
        )
        mimo_offaxis_static_static_moving_moving_multiframe_noise_shape_match = (
            oracle_mimo_offaxis_static_static_moving_moving_multiframe["noise"].shape
            == candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                "noise"
            ].shape
        )
        oracle_mimo_offaxis_static_static_moving_moving_multiframe_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_moving_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_moving_moving_multiframe_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_moving_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_frame_offset_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_moving_multiframe["timestamp"][
                    :4
                ]
                + 1e-3,
                oracle_mimo_offaxis_static_static_moving_moving_multiframe["timestamp"][
                    4:
                ],
            )
        )
        candidate_frame_offset_exact = bool(
            np.array_equal(
                candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                    "timestamp"
                ][:4]
                + 1e-3,
                candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                    "timestamp"
                ][4:],
            )
        )
        valid_bins = (
            oracle_mimo_offaxis_static_static_moving_moving_multiframe["baseband"].shape[
                -1
            ]
            // 2
        )
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_a_peaks = []
        candidate_moving_a_peaks = []
        oracle_moving_b_peaks = []
        candidate_moving_b_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_moving_moving_multiframe["baseband"].shape[
                0
            ]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(8, 18),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_moving_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_moving_moving_multiframe[
            "baseband"
        ][0, 0, phase_sample_index]
        candidate_ref = candidate_mimo_offaxis_static_static_moving_moving_multiframe[
            "baseband"
        ][0, 0, phase_sample_index]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_moving_moving_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                    "baseband"
                ][:4, 0, phase_sample_index]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_static_moving_moving_multiframe["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_static_moving_moving_multiframe[
                    "baseband"
                ],
                axis=-1,
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_moving_moving_multiframe["baseband"],
            candidate_mimo_offaxis_static_static_moving_moving_multiframe["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_moving_moving_multiframe_report = {
            "available": True,
            "timestamp_exact": (
                mimo_offaxis_static_static_moving_moving_multiframe_timestamp_exact
            ),
            "baseband_dtype_match": (
                mimo_offaxis_static_static_moving_moving_multiframe_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_static_moving_moving_multiframe_baseband_shape_match
            ),
            "noise_dtype_match": (
                mimo_offaxis_static_static_moving_moving_multiframe_noise_dtype_match
            ),
            "noise_shape_match": (
                mimo_offaxis_static_static_moving_moving_multiframe_noise_shape_match
            ),
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_static_moving_moving_multiframe_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_moving_moving_multiframe_interference_none
            ),
            "oracle_frame_offset_exact": oracle_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_frame_offset_exact,
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_a_peaks": oracle_moving_a_peaks,
            "candidate_moving_a_peaks": candidate_moving_a_peaks,
            "oracle_moving_b_peaks": oracle_moving_b_peaks,
            "candidate_moving_b_peaks": candidate_moving_b_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_a_peaks_exact": oracle_moving_a_peaks == candidate_moving_a_peaks,
            "moving_b_peaks_exact": oracle_moving_b_peaks == candidate_moving_b_peaks,
            "oracle_frame0_channel_phase_degrees": [
                float(value) for value in oracle_rel_phase_deg
            ],
            "candidate_frame0_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_moving_moving_multiframe_timestamp_exact,
                    mimo_offaxis_static_static_moving_moving_multiframe_baseband_dtype_match,
                    mimo_offaxis_static_static_moving_moving_multiframe_baseband_shape_match,
                    mimo_offaxis_static_static_moving_moving_multiframe_noise_dtype_match,
                    mimo_offaxis_static_static_moving_moving_multiframe_noise_shape_match,
                    oracle_mimo_offaxis_static_static_moving_moving_multiframe_interference_none,
                    candidate_mimo_offaxis_static_static_moving_moving_multiframe_interference_none,
                    oracle_frame_offset_exact,
                    candidate_frame_offset_exact,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_a_peaks == candidate_moving_a_peaks,
                    oracle_moving_b_peaks == candidate_moving_b_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_moving_multiframe_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_static_moving_multiframe_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_static_moving_multiframe.npz"
    )
    candidate_mimo_offaxis_static_static_moving_multiframe_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_moving_multiframe_path.exists()
        and candidate_mimo_offaxis_static_static_moving_multiframe_path.exists()
    ):
        oracle_mimo_offaxis_static_static_moving_multiframe = _load_npz_dict(
            oracle_mimo_offaxis_static_static_moving_multiframe_path
        )
        candidate_mimo_offaxis_static_static_moving_multiframe = _load_npz_dict(
            candidate_mimo_offaxis_static_static_moving_multiframe_path
        )
        mimo_offaxis_static_static_moving_multiframe_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_multiframe["timestamp"],
                candidate_mimo_offaxis_static_static_moving_multiframe["timestamp"],
            )
        )
        mimo_offaxis_static_static_moving_multiframe_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_multiframe["baseband"].dtype
            == candidate_mimo_offaxis_static_static_moving_multiframe["baseband"].dtype
        )
        mimo_offaxis_static_static_moving_multiframe_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_moving_multiframe["baseband"].shape
            == candidate_mimo_offaxis_static_static_moving_multiframe["baseband"].shape
        )
        mimo_offaxis_static_static_moving_multiframe_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_multiframe["noise"].dtype
            == candidate_mimo_offaxis_static_static_moving_multiframe["noise"].dtype
        )
        mimo_offaxis_static_static_moving_multiframe_noise_shape_match = (
            oracle_mimo_offaxis_static_static_moving_multiframe["noise"].shape
            == candidate_mimo_offaxis_static_static_moving_multiframe["noise"].shape
        )
        oracle_mimo_offaxis_static_static_moving_multiframe_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_moving_multiframe_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_frame_offset_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_multiframe["timestamp"][:4] + 1e-3,
                oracle_mimo_offaxis_static_static_moving_multiframe["timestamp"][4:],
            )
        )
        candidate_frame_offset_exact = bool(
            np.array_equal(
                candidate_mimo_offaxis_static_static_moving_multiframe["timestamp"][:4] + 1e-3,
                candidate_mimo_offaxis_static_static_moving_multiframe["timestamp"][4:],
            )
        )
        valid_bins = (
            oracle_mimo_offaxis_static_static_moving_multiframe["baseband"].shape[-1] // 2
        )
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_peaks = []
        candidate_moving_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_moving_multiframe["baseband"].shape[0]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_moving_multiframe["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_static_moving_multiframe["baseband"][
            0, 0, phase_sample_index
        ]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_moving_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_moving_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_static_moving_multiframe["baseband"], axis=-1
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_static_moving_multiframe["baseband"], axis=-1
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_moving_multiframe["baseband"],
            candidate_mimo_offaxis_static_static_moving_multiframe["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_moving_multiframe_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_static_moving_multiframe_timestamp_exact,
            "baseband_dtype_match": (
                mimo_offaxis_static_static_moving_multiframe_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_static_moving_multiframe_baseband_shape_match
            ),
            "noise_dtype_match": mimo_offaxis_static_static_moving_multiframe_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_static_moving_multiframe_noise_shape_match,
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_static_moving_multiframe_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_moving_multiframe_interference_none
            ),
            "oracle_frame_offset_exact": oracle_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_frame_offset_exact,
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_peaks": oracle_moving_peaks,
            "candidate_moving_peaks": candidate_moving_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_peaks_exact": oracle_moving_peaks == candidate_moving_peaks,
            "oracle_frame0_channel_phase_degrees": [
                float(value) for value in oracle_rel_phase_deg
            ],
            "candidate_frame0_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_moving_multiframe_timestamp_exact,
                    mimo_offaxis_static_static_moving_multiframe_baseband_dtype_match,
                    mimo_offaxis_static_static_moving_multiframe_baseband_shape_match,
                    mimo_offaxis_static_static_moving_multiframe_noise_dtype_match,
                    mimo_offaxis_static_static_moving_multiframe_noise_shape_match,
                    oracle_mimo_offaxis_static_static_moving_multiframe_interference_none,
                    candidate_mimo_offaxis_static_static_moving_multiframe_interference_none,
                    oracle_frame_offset_exact,
                    candidate_frame_offset_exact,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_peaks == candidate_moving_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_static_moving_negative_multiframe_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_static_moving_negative_multiframe_path = (
        oracle_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_negative_multiframe.npz"
    )
    candidate_mimo_offaxis_static_static_moving_negative_multiframe_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_static_moving_negative_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_static_static_moving_negative_multiframe_path.exists()
        and candidate_mimo_offaxis_static_static_moving_negative_multiframe_path.exists()
    ):
        oracle_mimo_offaxis_static_static_moving_negative_multiframe = _load_npz_dict(
            oracle_mimo_offaxis_static_static_moving_negative_multiframe_path
        )
        candidate_mimo_offaxis_static_static_moving_negative_multiframe = _load_npz_dict(
            candidate_mimo_offaxis_static_static_moving_negative_multiframe_path
        )
        mimo_offaxis_static_static_moving_negative_multiframe_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_negative_multiframe["timestamp"],
                candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                    "timestamp"
                ],
            )
        )
        mimo_offaxis_static_static_moving_negative_multiframe_baseband_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_negative_multiframe["baseband"].dtype
            == candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                "baseband"
            ].dtype
        )
        mimo_offaxis_static_static_moving_negative_multiframe_baseband_shape_match = (
            oracle_mimo_offaxis_static_static_moving_negative_multiframe["baseband"].shape
            == candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                "baseband"
            ].shape
        )
        mimo_offaxis_static_static_moving_negative_multiframe_noise_dtype_match = (
            oracle_mimo_offaxis_static_static_moving_negative_multiframe["noise"].dtype
            == candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                "noise"
            ].dtype
        )
        mimo_offaxis_static_static_moving_negative_multiframe_noise_shape_match = (
            oracle_mimo_offaxis_static_static_moving_negative_multiframe["noise"].shape
            == candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                "noise"
            ].shape
        )
        oracle_mimo_offaxis_static_static_moving_negative_multiframe_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_negative_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_static_moving_negative_multiframe_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_static_moving_negative_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_frame_offset_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_static_moving_negative_multiframe[
                    "timestamp"
                ][:4]
                + 1e-3,
                oracle_mimo_offaxis_static_static_moving_negative_multiframe[
                    "timestamp"
                ][4:],
            )
        )
        candidate_frame_offset_exact = bool(
            np.array_equal(
                candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                    "timestamp"
                ][:4]
                + 1e-3,
                candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                    "timestamp"
                ][4:],
            )
        )
        valid_bins = (
            oracle_mimo_offaxis_static_static_moving_negative_multiframe["baseband"].shape[
                -1
            ]
            // 2
        )
        oracle_static_a_peaks = []
        candidate_static_a_peaks = []
        oracle_static_b_peaks = []
        candidate_static_b_peaks = []
        oracle_moving_peaks = []
        candidate_moving_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_static_moving_negative_multiframe["baseband"].shape[
                0
            ]
        ):
            oracle_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_a_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_b_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(43, 52),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_static_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        oracle_moving_peaks = [
            [int(peak[0] + 2), int(peak[1])] for peak in oracle_moving_peaks
        ]
        candidate_moving_peaks = [
            [int(peak[0] + 2), int(peak[1])] for peak in candidate_moving_peaks
        ]
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_static_moving_negative_multiframe[
            "baseband"
        ][0, 0, phase_sample_index]
        candidate_ref = candidate_mimo_offaxis_static_static_moving_negative_multiframe[
            "baseband"
        ][0, 0, phase_sample_index]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_static_moving_negative_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                    "baseband"
                ][:4, 0, phase_sample_index]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_static_moving_negative_multiframe["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_static_moving_negative_multiframe[
                    "baseband"
                ],
                axis=-1,
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_static_moving_negative_multiframe["baseband"],
            candidate_mimo_offaxis_static_static_moving_negative_multiframe["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_static_moving_negative_multiframe_report = {
            "available": True,
            "timestamp_exact": (
                mimo_offaxis_static_static_moving_negative_multiframe_timestamp_exact
            ),
            "baseband_dtype_match": (
                mimo_offaxis_static_static_moving_negative_multiframe_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_static_moving_negative_multiframe_baseband_shape_match
            ),
            "noise_dtype_match": (
                mimo_offaxis_static_static_moving_negative_multiframe_noise_dtype_match
            ),
            "noise_shape_match": (
                mimo_offaxis_static_static_moving_negative_multiframe_noise_shape_match
            ),
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_static_moving_negative_multiframe_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_static_moving_negative_multiframe_interference_none
            ),
            "oracle_frame_offset_exact": oracle_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_frame_offset_exact,
            "phase_sample_index": phase_sample_index,
            "oracle_static_a_peaks": oracle_static_a_peaks,
            "candidate_static_a_peaks": candidate_static_a_peaks,
            "oracle_static_b_peaks": oracle_static_b_peaks,
            "candidate_static_b_peaks": candidate_static_b_peaks,
            "oracle_moving_peaks": oracle_moving_peaks,
            "candidate_moving_peaks": candidate_moving_peaks,
            "static_a_peaks_exact": oracle_static_a_peaks == candidate_static_a_peaks,
            "static_b_peaks_exact": oracle_static_b_peaks == candidate_static_b_peaks,
            "moving_peaks_exact": oracle_moving_peaks == candidate_moving_peaks,
            "oracle_frame0_channel_phase_degrees": [
                float(value) for value in oracle_rel_phase_deg
            ],
            "candidate_frame0_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_static_moving_negative_multiframe_timestamp_exact,
                    mimo_offaxis_static_static_moving_negative_multiframe_baseband_dtype_match,
                    mimo_offaxis_static_static_moving_negative_multiframe_baseband_shape_match,
                    mimo_offaxis_static_static_moving_negative_multiframe_noise_dtype_match,
                    mimo_offaxis_static_static_moving_negative_multiframe_noise_shape_match,
                    oracle_mimo_offaxis_static_static_moving_negative_multiframe_interference_none,
                    candidate_mimo_offaxis_static_static_moving_negative_multiframe_interference_none,
                    oracle_frame_offset_exact,
                    candidate_frame_offset_exact,
                    oracle_static_a_peaks == candidate_static_a_peaks,
                    oracle_static_b_peaks == candidate_static_b_peaks,
                    oracle_moving_peaks == candidate_moving_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_moving_multiframe_report = {"available": False, "overall": True}
    oracle_mimo_offaxis_static_moving_multiframe_path = (
        oracle_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_multiframe.npz"
    )
    candidate_mimo_offaxis_static_moving_multiframe_path = (
        candidate_dir / "artifacts" / "sim_radar_point_mimo_offaxis_static_moving_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_static_moving_multiframe_path.exists()
        and candidate_mimo_offaxis_static_moving_multiframe_path.exists()
    ):
        oracle_mimo_offaxis_static_moving_multiframe = _load_npz_dict(
            oracle_mimo_offaxis_static_moving_multiframe_path
        )
        candidate_mimo_offaxis_static_moving_multiframe = _load_npz_dict(
            candidate_mimo_offaxis_static_moving_multiframe_path
        )
        mimo_offaxis_static_moving_multiframe_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_moving_multiframe["timestamp"],
                candidate_mimo_offaxis_static_moving_multiframe["timestamp"],
            )
        )
        mimo_offaxis_static_moving_multiframe_baseband_dtype_match = (
            oracle_mimo_offaxis_static_moving_multiframe["baseband"].dtype
            == candidate_mimo_offaxis_static_moving_multiframe["baseband"].dtype
        )
        mimo_offaxis_static_moving_multiframe_baseband_shape_match = (
            oracle_mimo_offaxis_static_moving_multiframe["baseband"].shape
            == candidate_mimo_offaxis_static_moving_multiframe["baseband"].shape
        )
        mimo_offaxis_static_moving_multiframe_noise_dtype_match = (
            oracle_mimo_offaxis_static_moving_multiframe["noise"].dtype
            == candidate_mimo_offaxis_static_moving_multiframe["noise"].dtype
        )
        mimo_offaxis_static_moving_multiframe_noise_shape_match = (
            oracle_mimo_offaxis_static_moving_multiframe["noise"].shape
            == candidate_mimo_offaxis_static_moving_multiframe["noise"].shape
        )
        oracle_mimo_offaxis_static_moving_multiframe_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_moving_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_moving_multiframe_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_moving_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_frame_offset_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_moving_multiframe["timestamp"][:4] + 1e-3,
                oracle_mimo_offaxis_static_moving_multiframe["timestamp"][4:],
            )
        )
        candidate_frame_offset_exact = bool(
            np.array_equal(
                candidate_mimo_offaxis_static_moving_multiframe["timestamp"][:4] + 1e-3,
                candidate_mimo_offaxis_static_moving_multiframe["timestamp"][4:],
            )
        )
        valid_bins = oracle_mimo_offaxis_static_moving_multiframe["baseband"].shape[-1] // 2
        oracle_static_peaks = []
        candidate_static_peaks = []
        oracle_moving_peaks = []
        candidate_moving_peaks = []
        for channel_index in range(oracle_mimo_offaxis_static_moving_multiframe["baseband"].shape[0]):
            oracle_static_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_moving_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_moving_multiframe["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_moving_multiframe["baseband"][
            0, 0, phase_sample_index
        ]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_moving_multiframe["baseband"][:4, 0, phase_sample_index]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_moving_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(oracle_mimo_offaxis_static_moving_multiframe["baseband"], axis=-1)
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(candidate_mimo_offaxis_static_moving_multiframe["baseband"], axis=-1)
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_moving_multiframe["baseband"],
            candidate_mimo_offaxis_static_moving_multiframe["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_moving_multiframe_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_moving_multiframe_timestamp_exact,
            "baseband_dtype_match": mimo_offaxis_static_moving_multiframe_baseband_dtype_match,
            "baseband_shape_match": mimo_offaxis_static_moving_multiframe_baseband_shape_match,
            "noise_dtype_match": mimo_offaxis_static_moving_multiframe_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_moving_multiframe_noise_shape_match,
            "oracle_interference_none": oracle_mimo_offaxis_static_moving_multiframe_interference_none,
            "candidate_interference_none": candidate_mimo_offaxis_static_moving_multiframe_interference_none,
            "oracle_frame_offset_exact": oracle_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_frame_offset_exact,
            "phase_sample_index": phase_sample_index,
            "oracle_static_peaks": oracle_static_peaks,
            "candidate_static_peaks": candidate_static_peaks,
            "oracle_moving_peaks": oracle_moving_peaks,
            "candidate_moving_peaks": candidate_moving_peaks,
            "static_peaks_exact": oracle_static_peaks == candidate_static_peaks,
            "moving_peaks_exact": oracle_moving_peaks == candidate_moving_peaks,
            "oracle_frame0_channel_phase_degrees": [float(value) for value in oracle_rel_phase_deg],
            "candidate_frame0_channel_phase_degrees": [float(value) for value in candidate_rel_phase_deg],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_moving_multiframe_timestamp_exact,
                    mimo_offaxis_static_moving_multiframe_baseband_dtype_match,
                    mimo_offaxis_static_moving_multiframe_baseband_shape_match,
                    mimo_offaxis_static_moving_multiframe_noise_dtype_match,
                    mimo_offaxis_static_moving_multiframe_noise_shape_match,
                    oracle_mimo_offaxis_static_moving_multiframe_interference_none,
                    candidate_mimo_offaxis_static_moving_multiframe_interference_none,
                    oracle_frame_offset_exact,
                    candidate_frame_offset_exact,
                    oracle_static_peaks == candidate_static_peaks,
                    oracle_moving_peaks == candidate_moving_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    point_mimo_offaxis_static_moving_negative_multiframe_report = {
        "available": False,
        "overall": True,
    }
    oracle_mimo_offaxis_static_moving_negative_multiframe_path = (
        oracle_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_moving_negative_multiframe.npz"
    )
    candidate_mimo_offaxis_static_moving_negative_multiframe_path = (
        candidate_dir
        / "artifacts"
        / "sim_radar_point_mimo_offaxis_static_moving_negative_multiframe.npz"
    )
    if (
        oracle_mimo_offaxis_static_moving_negative_multiframe_path.exists()
        and candidate_mimo_offaxis_static_moving_negative_multiframe_path.exists()
    ):
        oracle_mimo_offaxis_static_moving_negative_multiframe = _load_npz_dict(
            oracle_mimo_offaxis_static_moving_negative_multiframe_path
        )
        candidate_mimo_offaxis_static_moving_negative_multiframe = _load_npz_dict(
            candidate_mimo_offaxis_static_moving_negative_multiframe_path
        )
        mimo_offaxis_static_moving_negative_multiframe_timestamp_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_moving_negative_multiframe["timestamp"],
                candidate_mimo_offaxis_static_moving_negative_multiframe["timestamp"],
            )
        )
        mimo_offaxis_static_moving_negative_multiframe_baseband_dtype_match = (
            oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"].dtype
            == candidate_mimo_offaxis_static_moving_negative_multiframe["baseband"].dtype
        )
        mimo_offaxis_static_moving_negative_multiframe_baseband_shape_match = (
            oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"].shape
            == candidate_mimo_offaxis_static_moving_negative_multiframe["baseband"].shape
        )
        mimo_offaxis_static_moving_negative_multiframe_noise_dtype_match = (
            oracle_mimo_offaxis_static_moving_negative_multiframe["noise"].dtype
            == candidate_mimo_offaxis_static_moving_negative_multiframe["noise"].dtype
        )
        mimo_offaxis_static_moving_negative_multiframe_noise_shape_match = (
            oracle_mimo_offaxis_static_moving_negative_multiframe["noise"].shape
            == candidate_mimo_offaxis_static_moving_negative_multiframe["noise"].shape
        )
        oracle_mimo_offaxis_static_moving_negative_multiframe_interference_none = (
            oracle_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_moving_negative_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        candidate_mimo_offaxis_static_moving_negative_multiframe_interference_none = (
            candidate_manifest["captures"]["sim_radar"]
            .get("point_mimo_offaxis_static_moving_negative_multiframe", {})
            .get("summary", {})
            .get("interference")
            is None
        )
        oracle_frame_offset_exact = bool(
            np.array_equal(
                oracle_mimo_offaxis_static_moving_negative_multiframe["timestamp"][:4]
                + 1e-3,
                oracle_mimo_offaxis_static_moving_negative_multiframe["timestamp"][4:],
            )
        )
        candidate_frame_offset_exact = bool(
            np.array_equal(
                candidate_mimo_offaxis_static_moving_negative_multiframe["timestamp"][:4]
                + 1e-3,
                candidate_mimo_offaxis_static_moving_negative_multiframe["timestamp"][4:],
            )
        )
        valid_bins = (
            oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"].shape[-1] // 2
        )
        oracle_static_peaks = []
        candidate_static_peaks = []
        oracle_moving_peaks = []
        candidate_moving_peaks = []
        for channel_index in range(
            oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"].shape[0]
        ):
            oracle_static_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_static_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_moving_negative_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(0, 2),
                        range_slice=slice(23, 32),
                        valid_bins=valid_bins,
                    )
                )
            )
            oracle_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"][
                            channel_index : channel_index + 1
                        ],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
            candidate_moving_peaks.append(
                list(
                    _local_range_doppler_peak_indices(
                        candidate_mimo_offaxis_static_moving_negative_multiframe[
                            "baseband"
                        ][channel_index : channel_index + 1],
                        doppler_slice=slice(2, 4),
                        range_slice=slice(63, 72),
                        valid_bins=valid_bins,
                    )
                )
            )
        phase_sample_index = 2
        oracle_ref = oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"][
            0, 0, phase_sample_index
        ]
        candidate_ref = candidate_mimo_offaxis_static_moving_negative_multiframe["baseband"][
            0, 0, phase_sample_index
        ]
        oracle_rel_phase_deg = np.degrees(
            np.angle(
                oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / oracle_ref
            )
        )
        candidate_rel_phase_deg = np.degrees(
            np.angle(
                candidate_mimo_offaxis_static_moving_negative_multiframe["baseband"][
                    :4, 0, phase_sample_index
                ]
                / candidate_ref
            )
        )
        channel_phase_allclose = bool(
            np.allclose(oracle_rel_phase_deg, candidate_rel_phase_deg, atol=1e-6)
        )
        oracle_profile = np.abs(
            np.fft.fft(
                oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"], axis=-1
            )
        )[0, 0, :80]
        candidate_profile = np.abs(
            np.fft.fft(
                candidate_mimo_offaxis_static_moving_negative_multiframe["baseband"],
                axis=-1,
            )
        )[0, 0, :80]
        range_profile_correlation_abs = _normalized_correlation_abs(
            oracle_profile,
            candidate_profile,
        )
        range_profile_correlation_pass = range_profile_correlation_abs >= 0.999
        waveform_correlation_abs = _normalized_correlation_abs(
            oracle_mimo_offaxis_static_moving_negative_multiframe["baseband"],
            candidate_mimo_offaxis_static_moving_negative_multiframe["baseband"],
        )
        waveform_correlation_pass = waveform_correlation_abs >= 0.995
        point_mimo_offaxis_static_moving_negative_multiframe_report = {
            "available": True,
            "timestamp_exact": mimo_offaxis_static_moving_negative_multiframe_timestamp_exact,
            "baseband_dtype_match": (
                mimo_offaxis_static_moving_negative_multiframe_baseband_dtype_match
            ),
            "baseband_shape_match": (
                mimo_offaxis_static_moving_negative_multiframe_baseband_shape_match
            ),
            "noise_dtype_match": mimo_offaxis_static_moving_negative_multiframe_noise_dtype_match,
            "noise_shape_match": mimo_offaxis_static_moving_negative_multiframe_noise_shape_match,
            "oracle_interference_none": (
                oracle_mimo_offaxis_static_moving_negative_multiframe_interference_none
            ),
            "candidate_interference_none": (
                candidate_mimo_offaxis_static_moving_negative_multiframe_interference_none
            ),
            "oracle_frame_offset_exact": oracle_frame_offset_exact,
            "candidate_frame_offset_exact": candidate_frame_offset_exact,
            "phase_sample_index": phase_sample_index,
            "oracle_static_peaks": oracle_static_peaks,
            "candidate_static_peaks": candidate_static_peaks,
            "oracle_moving_peaks": oracle_moving_peaks,
            "candidate_moving_peaks": candidate_moving_peaks,
            "static_peaks_exact": oracle_static_peaks == candidate_static_peaks,
            "moving_peaks_exact": oracle_moving_peaks == candidate_moving_peaks,
            "oracle_frame0_channel_phase_degrees": [
                float(value) for value in oracle_rel_phase_deg
            ],
            "candidate_frame0_channel_phase_degrees": [
                float(value) for value in candidate_rel_phase_deg
            ],
            "channel_phase_allclose": channel_phase_allclose,
            "range_profile_correlation_abs": range_profile_correlation_abs,
            "range_profile_correlation_pass": range_profile_correlation_pass,
            "waveform_correlation_abs": waveform_correlation_abs,
            "waveform_correlation_pass": waveform_correlation_pass,
            "overall": all(
                [
                    mimo_offaxis_static_moving_negative_multiframe_timestamp_exact,
                    mimo_offaxis_static_moving_negative_multiframe_baseband_dtype_match,
                    mimo_offaxis_static_moving_negative_multiframe_baseband_shape_match,
                    mimo_offaxis_static_moving_negative_multiframe_noise_dtype_match,
                    mimo_offaxis_static_moving_negative_multiframe_noise_shape_match,
                    oracle_mimo_offaxis_static_moving_negative_multiframe_interference_none,
                    candidate_mimo_offaxis_static_moving_negative_multiframe_interference_none,
                    oracle_frame_offset_exact,
                    candidate_frame_offset_exact,
                    oracle_static_peaks == candidate_static_peaks,
                    oracle_moving_peaks == candidate_moving_peaks,
                    channel_phase_allclose,
                    range_profile_correlation_pass,
                    waveform_correlation_pass,
                ]
            ),
        }

    dry_run_report = {
        "oracle_interference_none": oracle_interference_none,
        "candidate_interference_none": candidate_interference_none,
        "candidate_zero_match": dry_run_zero_match,
        "overall": oracle_interference_none and candidate_interference_none and dry_run_zero_match,
    }
    seed_repeat_report = {
        "match": seed_repeat_match,
        "overall": seed_repeat_match,
    }

    mesh_report = {"available": False, "overall": True}
    oracle_mesh_path = oracle_dir / "artifacts" / "sim_radar_mesh_target.npz"
    candidate_mesh_path = candidate_dir / "artifacts" / "sim_radar_mesh_target.npz"
    if oracle_mesh_path.exists() and candidate_mesh_path.exists():
        oracle_mesh = _load_npz_dict(oracle_mesh_path)
        candidate_mesh = _load_npz_dict(candidate_mesh_path)

        mesh_timestamp_exact = bool(
            np.array_equal(oracle_mesh["timestamp"], candidate_mesh["timestamp"])
        )
        mesh_baseband_dtype_match = (
            oracle_mesh["baseband"].dtype == candidate_mesh["baseband"].dtype
        )
        mesh_baseband_shape_match = (
            oracle_mesh["baseband"].shape == candidate_mesh["baseband"].shape
        )
        mesh_noise_dtype_match = oracle_mesh["noise"].dtype == candidate_mesh["noise"].dtype
        mesh_noise_shape_match = oracle_mesh["noise"].shape == candidate_mesh["noise"].shape
        mesh_correlation_abs = _normalized_correlation_abs(
            oracle_mesh["baseband"],
            candidate_mesh["baseband"],
        )
        mesh_correlation_pass = mesh_correlation_abs >= 0.99
        oracle_mesh_power = float(np.mean(np.abs(oracle_mesh["baseband"]) ** 2))
        candidate_mesh_power = float(np.mean(np.abs(candidate_mesh["baseband"]) ** 2))
        mesh_power_ratio = float(
            candidate_mesh_power / oracle_mesh_power
            if oracle_mesh_power > 0.0
            else np.inf
        )

        oracle_mesh_manifest = oracle_manifest["captures"]["sim_radar"].get("mesh_target", {})
        candidate_mesh_manifest = candidate_manifest["captures"]["sim_radar"].get(
            "mesh_target",
            {},
        )
        oracle_mesh_interference_none = (
            oracle_mesh_manifest.get("summary", {}).get("interference") is None
        )
        candidate_mesh_interference_none = (
            candidate_mesh_manifest.get("summary", {}).get("interference") is None
        )
        oracle_mesh_nonzero = bool(np.max(np.abs(oracle_mesh["baseband"])) > 0.0)
        candidate_mesh_nonzero = bool(np.max(np.abs(candidate_mesh["baseband"])) > 0.0)

        mesh_report = {
            "available": True,
            "timestamp_exact": mesh_timestamp_exact,
            "baseband_dtype_match": mesh_baseband_dtype_match,
            "baseband_shape_match": mesh_baseband_shape_match,
            "noise_dtype_match": mesh_noise_dtype_match,
            "noise_shape_match": mesh_noise_shape_match,
            "waveform_correlation_abs": mesh_correlation_abs,
            "waveform_correlation_pass": mesh_correlation_pass,
            "oracle_interference_none": oracle_mesh_interference_none,
            "candidate_interference_none": candidate_mesh_interference_none,
            "oracle_nonzero": oracle_mesh_nonzero,
            "candidate_nonzero": candidate_mesh_nonzero,
            "power_ratio_candidate_over_oracle": mesh_power_ratio,
            "overall": all(
                [
                    mesh_timestamp_exact,
                    mesh_baseband_dtype_match,
                    mesh_baseband_shape_match,
                    mesh_noise_dtype_match,
                    mesh_noise_shape_match,
                    mesh_correlation_pass,
                    oracle_mesh_interference_none,
                    candidate_mesh_interference_none,
                    oracle_mesh_nonzero,
                    candidate_mesh_nonzero,
                ]
            ),
        }

    mesh_aspect_report = {"available": False, "overall": True}
    oracle_mesh_aspect = oracle_manifest["captures"]["sim_radar"].get("mesh_aspect")
    candidate_mesh_aspect = candidate_manifest["captures"]["sim_radar"].get("mesh_aspect")
    if oracle_mesh_aspect is not None and candidate_mesh_aspect is not None:
        trend_match = oracle_mesh_aspect.get("broadside_gt_edge_on") == candidate_mesh_aspect.get(
            "broadside_gt_edge_on"
        )
        oracle_broadside_max_abs = float(oracle_mesh_aspect.get("broadside_max_abs", 0.0))
        oracle_edge_on_max_abs = float(oracle_mesh_aspect.get("edge_on_max_abs", 0.0))
        candidate_broadside_max_abs = float(candidate_mesh_aspect.get("broadside_max_abs", 0.0))
        candidate_edge_on_max_abs = float(candidate_mesh_aspect.get("edge_on_max_abs", 0.0))
        mesh_aspect_report = {
            "available": True,
            "trend_match": trend_match,
            "oracle_broadside_max_abs": oracle_broadside_max_abs,
            "oracle_edge_on_max_abs": oracle_edge_on_max_abs,
            "candidate_broadside_max_abs": candidate_broadside_max_abs,
            "candidate_edge_on_max_abs": candidate_edge_on_max_abs,
            "overall": all(
                [
                    trend_match,
                    oracle_broadside_max_abs > oracle_edge_on_max_abs,
                    candidate_broadside_max_abs > candidate_edge_on_max_abs,
                ]
            ),
        }

    overall = all(
        [
            point_target_overall,
            point_mimo_report["overall"],
            point_mimo_offaxis_report["overall"],
            point_mimo_offaxis_moving_report["overall"],
            point_mimo_offaxis_moving_negative_report["overall"],
            point_mimo_offaxis_moving_multiframe_report["overall"],
            point_mimo_offaxis_moving_negative_multiframe_report["overall"],
            point_mimo_multiframe_report["overall"],
            point_phase_report["overall"],
            point_moving_report["overall"],
            point_moving_negative_report["overall"],
            point_multi_report["overall"],
            point_static_moving_report["overall"],
            point_static_moving_negative_report["overall"],
            point_static_static_moving_report["overall"],
            point_static_static_moving_negative_report["overall"],
            point_static_static_moving_moving_report["overall"],
            point_static_static_moving_moving_negative_report["overall"],
            point_mimo_offaxis_static_moving_report["overall"],
            point_mimo_offaxis_static_moving_negative_report["overall"],
            point_mimo_offaxis_static_static_moving_report["overall"],
            point_mimo_offaxis_static_static_moving_negative_report["overall"],
            point_mimo_offaxis_static_static_moving_moving_report["overall"],
            point_mimo_offaxis_static_static_mixed_sign_report["overall"],
            point_mimo_offaxis_static_static_mixed_sign_multiframe_report["overall"],
            point_mimo_offaxis_static_static_moving_moving_negative_multiframe_report[
                "overall"
            ],
            point_mimo_offaxis_static_static_moving_moving_multiframe_report["overall"],
            point_mimo_offaxis_static_static_moving_multiframe_report["overall"],
            point_mimo_offaxis_static_static_moving_negative_multiframe_report["overall"],
            point_mimo_offaxis_static_moving_multiframe_report["overall"],
            point_mimo_offaxis_static_moving_negative_multiframe_report["overall"],
            dry_run_report["overall"],
            seed_repeat_report["overall"],
            mesh_report["overall"],
            mesh_aspect_report["overall"],
        ]
    )
    return {
        "timestamp_exact": timestamp_exact,
        "baseband_dtype_match": baseband_dtype_match,
        "baseband_shape_match": baseband_shape_match,
        "baseband_allclose": baseband_allclose,
        "baseband_max_abs": baseband_max_abs,
        "noise_dtype_match": noise_dtype_match,
        "noise_shape_match": noise_shape_match,
        "seed_repeat_match": seed_repeat_match,
        "oracle_interference_none": oracle_interference_none,
        "candidate_interference_none": candidate_interference_none,
        "dry_run_zero_match": dry_run_zero_match,
        "point_target_overall": point_target_overall,
        "point_target": point_target_report,
        "point_mimo": point_mimo_report,
        "point_mimo_offaxis": point_mimo_offaxis_report,
        "point_mimo_offaxis_moving": point_mimo_offaxis_moving_report,
        "point_mimo_offaxis_moving_negative": point_mimo_offaxis_moving_negative_report,
        "point_mimo_offaxis_moving_multiframe": point_mimo_offaxis_moving_multiframe_report,
        "point_mimo_offaxis_moving_negative_multiframe": (
            point_mimo_offaxis_moving_negative_multiframe_report
        ),
        "point_mimo_multiframe": point_mimo_multiframe_report,
        "point_phase": point_phase_report,
        "point_moving_doppler": point_moving_report,
        "point_moving_doppler_negative": point_moving_negative_report,
        "point_multi": point_multi_report,
        "point_static_moving": point_static_moving_report,
        "point_static_moving_negative": point_static_moving_negative_report,
        "point_static_static_moving": point_static_static_moving_report,
        "point_static_static_moving_negative": point_static_static_moving_negative_report,
        "point_static_static_moving_moving": point_static_static_moving_moving_report,
        "point_static_static_moving_moving_negative": (
            point_static_static_moving_moving_negative_report
        ),
        "point_mimo_offaxis_static_moving": point_mimo_offaxis_static_moving_report,
        "point_mimo_offaxis_static_moving_negative": (
            point_mimo_offaxis_static_moving_negative_report
        ),
        "point_mimo_offaxis_static_static_moving": (
            point_mimo_offaxis_static_static_moving_report
        ),
        "point_mimo_offaxis_static_static_moving_negative": (
            point_mimo_offaxis_static_static_moving_negative_report
        ),
        "point_mimo_offaxis_static_static_moving_moving": (
            point_mimo_offaxis_static_static_moving_moving_report
        ),
        "point_mimo_offaxis_static_static_mixed_sign": (
            point_mimo_offaxis_static_static_mixed_sign_report
        ),
        "point_mimo_offaxis_static_static_mixed_sign_multiframe": (
            point_mimo_offaxis_static_static_mixed_sign_multiframe_report
        ),
        "point_mimo_offaxis_static_static_moving_moving_negative_multiframe": (
            point_mimo_offaxis_static_static_moving_moving_negative_multiframe_report
        ),
        "point_mimo_offaxis_static_static_moving_moving_multiframe": (
            point_mimo_offaxis_static_static_moving_moving_multiframe_report
        ),
        "point_mimo_offaxis_static_static_moving_multiframe": (
            point_mimo_offaxis_static_static_moving_multiframe_report
        ),
        "point_mimo_offaxis_static_static_moving_negative_multiframe": (
            point_mimo_offaxis_static_static_moving_negative_multiframe_report
        ),
        "point_mimo_offaxis_static_moving_multiframe": (
            point_mimo_offaxis_static_moving_multiframe_report
        ),
        "point_mimo_offaxis_static_moving_negative_multiframe": (
            point_mimo_offaxis_static_moving_negative_multiframe_report
        ),
        "dry_run": dry_run_report,
        "seed_repeat": seed_repeat_report,
        "mesh_target": mesh_report,
        "mesh_aspect": mesh_aspect_report,
        "overall": overall,
    }


def _compare_sim_lidar(oracle_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline_oracle = np.load(oracle_dir / "artifacts" / "sim_lidar_hits.npy", allow_pickle=False)
    baseline_candidate = np.load(
        candidate_dir / "artifacts" / "sim_lidar_hits.npy",
        allow_pickle=False,
    )

    baseline_report = {
        "length_match": len(baseline_oracle) == len(baseline_candidate),
        **_shared_field_parity(baseline_oracle, baseline_candidate),
    }

    moving_report = {"available": False, "overall": True}
    oracle_moving_path = oracle_dir / "artifacts" / "sim_lidar_moving_hits.npy"
    candidate_moving_path = candidate_dir / "artifacts" / "sim_lidar_moving_hits.npy"
    if oracle_moving_path.exists() and candidate_moving_path.exists():
        moving_oracle = np.load(oracle_moving_path, allow_pickle=False)
        moving_candidate = np.load(candidate_moving_path, allow_pickle=False)
        moving_report = {
            "available": True,
            "length_match": len(moving_oracle) == len(moving_candidate),
            **_shared_field_parity(moving_oracle, moving_candidate),
        }

    multi_hit_report = {"available": False, "overall": True}
    oracle_multi_path = oracle_dir / "artifacts" / "sim_lidar_multi_hits.npy"
    candidate_multi_path = candidate_dir / "artifacts" / "sim_lidar_multi_hits.npy"
    if oracle_multi_path.exists() and candidate_multi_path.exists():
        multi_oracle = np.load(oracle_multi_path, allow_pickle=False)
        multi_candidate = np.load(candidate_multi_path, allow_pickle=False)

        vendor_shared = _shared_field_parity(multi_oracle, multi_candidate)
        oracle_y_positions = np.asarray(multi_oracle["positions"][:, 1], dtype=float)
        candidate_y_positions = np.asarray(multi_candidate["positions"][:, 1], dtype=float)
        y_order_match = bool(np.allclose(oracle_y_positions, candidate_y_positions, atol=1e-6))

        oracle_distances = np.linalg.norm(
            np.asarray(multi_oracle["positions"], dtype=float),
            axis=1,
        )
        if multi_candidate.dtype.names and "distance" in multi_candidate.dtype.names:
            candidate_distances = np.asarray(multi_candidate["distance"], dtype=float)
        else:
            candidate_distances = np.linalg.norm(
                np.asarray(multi_candidate["positions"], dtype=float),
                axis=1,
            )
        derived_distance_match = bool(
            np.allclose(oracle_distances, candidate_distances, atol=1e-5)
        )

        multi_hit_report = {
            "available": True,
            "length_match": len(multi_oracle) == len(multi_candidate),
            **vendor_shared,
            "oracle_y_positions": oracle_y_positions.tolist(),
            "candidate_y_positions": candidate_y_positions.tolist(),
            "y_order_match": y_order_match,
            "oracle_distances": oracle_distances.tolist(),
            "candidate_distances": candidate_distances.tolist(),
            "derived_distance_match": derived_distance_match,
            "overall": vendor_shared["overall"] and y_order_match and derived_distance_match,
        }

    return {
        "baseline": baseline_report,
        "moving": moving_report,
        "multi_hit": multi_hit_report,
        "overall": (
            baseline_report["overall"]
            and moving_report["overall"]
            and multi_hit_report["overall"]
        ),
    }


def _compare_sim_rcs(
    oracle_dir: Path,
    candidate_dir: Path,
    oracle_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> dict[str, Any]:
    oracle_rcs = oracle_manifest["captures"]["sim_rcs"]
    candidate_rcs = candidate_manifest["captures"]["sim_rcs"]

    cases = [
        "normal_incidence",
        "broadside",
        "edge_on",
        "obs_backscatter",
        "obs_forwardscatter",
    ]
    case_reports = {}
    overall = True
    for case_name in cases:
        oracle_case = oracle_rcs.get(case_name)
        candidate_case = candidate_rcs.get(case_name)
        if oracle_case is None or candidate_case is None:
            case_reports[case_name] = {"available": False, "overall": True}
            continue

        oracle_value = float(oracle_case["summary"])
        candidate_value = float(candidate_case["summary"])
        matched = _relative_or_absolute_close(oracle_value, candidate_value)
        case_reports[case_name] = {
            "available": True,
            "oracle": oracle_value,
            "candidate": candidate_value,
            "match": matched,
            "relative_error": float(
                abs(candidate_value - oracle_value) / abs(oracle_value)
                if oracle_value != 0.0
                else abs(candidate_value - oracle_value)
            ),
            "overall": matched,
        }
        overall = overall and matched

    trend_match = oracle_rcs["trend"]["broadside_gt_edge_on"] == candidate_rcs["trend"][
        "broadside_gt_edge_on"
    ]
    observation_trend_match = True
    if "observation_trend" in oracle_rcs and "observation_trend" in candidate_rcs:
        observation_trend_match = (
            oracle_rcs["observation_trend"]["backscatter_gt_forwardscatter"]
            == candidate_rcs["observation_trend"]["backscatter_gt_forwardscatter"]
        )

    sweep_report = {"available": False, "overall": True}
    oracle_sweep_path = oracle_dir / "artifacts" / "sim_rcs_obs_phi_sweep.npz"
    candidate_sweep_path = candidate_dir / "artifacts" / "sim_rcs_obs_phi_sweep.npz"
    if oracle_sweep_path.exists() and candidate_sweep_path.exists():
        oracle_sweep = _load_npz_dict(oracle_sweep_path)
        candidate_sweep = _load_npz_dict(candidate_sweep_path)

        oracle_angles = oracle_sweep["obs_phi"]
        candidate_angles = candidate_sweep["obs_phi"]
        oracle_values = oracle_sweep["rcs"]
        candidate_values = candidate_sweep["rcs"]

        angle_exact = bool(np.array_equal(oracle_angles, candidate_angles))
        shape_match = oracle_values.shape == candidate_values.shape
        peak_index_match = int(np.argmax(oracle_values)) == int(np.argmax(candidate_values))
        correlation_abs = _normalized_correlation_abs(oracle_values, candidate_values)
        correlation_pass = correlation_abs >= 0.85

        oracle_first_segment_monotonic = bool(np.all(np.diff(oracle_values[:3]) < 0.0))
        candidate_first_segment_monotonic = bool(np.all(np.diff(candidate_values[:3]) < 0.0))

        oracle_peak = float(np.max(oracle_values))
        candidate_peak = float(np.max(candidate_values))
        oracle_tail_ratio = float(
            np.max(oracle_values[3:]) / oracle_peak
            if oracle_peak > np.finfo(float).eps
            else np.inf
        )
        candidate_tail_ratio = float(
            np.max(candidate_values[3:]) / candidate_peak
            if candidate_peak > np.finfo(float).eps
            else np.inf
        )
        oracle_tail_quiet = oracle_tail_ratio <= 1e-6
        candidate_tail_quiet = candidate_tail_ratio <= 1e-6

        sweep_report = {
            "available": True,
            "angle_exact": angle_exact,
            "shape_match": shape_match,
            "peak_index_match": peak_index_match,
            "waveform_correlation_abs": correlation_abs,
            "waveform_correlation_pass": correlation_pass,
            "oracle_first_segment_monotonic": oracle_first_segment_monotonic,
            "candidate_first_segment_monotonic": candidate_first_segment_monotonic,
            "oracle_tail_quiet": oracle_tail_quiet,
            "candidate_tail_quiet": candidate_tail_quiet,
            "oracle_tail_to_peak_ratio": oracle_tail_ratio,
            "candidate_tail_to_peak_ratio": candidate_tail_ratio,
            "overall": all(
                [
                    angle_exact,
                    shape_match,
                    peak_index_match,
                    correlation_pass,
                    oracle_first_segment_monotonic,
                    candidate_first_segment_monotonic,
                    oracle_tail_quiet,
                    candidate_tail_quiet,
                ]
            ),
        }

    polarization_report = {"available": False, "overall": True}
    oracle_pol_path = oracle_dir / "artifacts" / "sim_rcs_polarization_cases.npz"
    candidate_pol_path = candidate_dir / "artifacts" / "sim_rcs_polarization_cases.npz"
    if oracle_pol_path.exists() and candidate_pol_path.exists():
        oracle_pol = _load_npz_dict(oracle_pol_path)
        candidate_pol = _load_npz_dict(candidate_pol_path)

        oracle_case_names = oracle_pol["case_names"]
        candidate_case_names = candidate_pol["case_names"]
        oracle_values = np.asarray(oracle_pol["rcs"], dtype=float)
        candidate_values = np.asarray(candidate_pol["rcs"], dtype=float)

        case_names_exact = bool(np.array_equal(oracle_case_names, candidate_case_names))
        shape_match = oracle_values.shape == candidate_values.shape
        co_pol_close = _relative_or_absolute_close(
            float(oracle_values[0]),
            float(candidate_values[0]),
            rtol=0.01,
            atol=1e-9,
        )
        oracle_cross_ratio = float(
            np.max(oracle_values[1:]) / oracle_values[0]
            if oracle_values[0] > np.finfo(float).eps
            else np.inf
        )
        candidate_cross_ratio = float(
            np.max(candidate_values[1:]) / candidate_values[0]
            if candidate_values[0] > np.finfo(float).eps
            else np.inf
        )
        oracle_cross_quiet = oracle_cross_ratio <= 1e-20
        candidate_cross_quiet = candidate_cross_ratio <= 1e-20
        oracle_co_pol_gt_cross = bool(oracle_values[0] > np.max(oracle_values[1:]))
        candidate_co_pol_gt_cross = bool(candidate_values[0] > np.max(candidate_values[1:]))

        polarization_report = {
            "available": True,
            "case_names_exact": case_names_exact,
            "shape_match": shape_match,
            "co_pol_close": co_pol_close,
            "oracle_cross_ratio": oracle_cross_ratio,
            "candidate_cross_ratio": candidate_cross_ratio,
            "oracle_cross_quiet": oracle_cross_quiet,
            "candidate_cross_quiet": candidate_cross_quiet,
            "oracle_co_pol_gt_cross": oracle_co_pol_gt_cross,
            "candidate_co_pol_gt_cross": candidate_co_pol_gt_cross,
            "overall": all(
                [
                    case_names_exact,
                    shape_match,
                    co_pol_close,
                    oracle_cross_quiet,
                    candidate_cross_quiet,
                    oracle_co_pol_gt_cross,
                    candidate_co_pol_gt_cross,
                ]
            ),
        }

    overall = (
        overall
        and trend_match
        and observation_trend_match
        and sweep_report["overall"]
        and polarization_report["overall"]
    )
    return {
        "cases": case_reports,
        "trend_match": trend_match,
        "observation_trend_match": observation_trend_match,
        "sweep": sweep_report,
        "polarization": polarization_report,
        "overall": overall,
    }


def compare_against_oracle(
    rs_module,
    oracle_dir: Path | str,
    *,
    candidate_module_name: str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compare a module against a previously captured oracle bundle."""

    oracle_dir = Path(oracle_dir)
    oracle_manifest = _load_manifest(oracle_dir / "manifest.json")

    with TemporaryDirectory() as temp_dir:
        candidate_dir = Path(temp_dir)
        candidate_manifest = run_capture(
            rs_module,
            candidate_dir,
            module_name=candidate_module_name,
        )

        comparisons = {
            "license": _compare_license(oracle_manifest, candidate_manifest),
            "sim_radar": _compare_sim_radar(oracle_dir, candidate_dir),
            "sim_lidar": _compare_sim_lidar(oracle_dir, candidate_dir),
            "sim_rcs": _compare_sim_rcs(
                oracle_dir,
                candidate_dir,
                oracle_manifest,
                candidate_manifest,
            ),
        }

    overall = all(section["overall"] for section in comparisons.values())
    report = {
        "oracle_module_name": oracle_manifest["module_name"],
        "candidate_module_name": candidate_module_name,
        "oracle_version": oracle_manifest["metadata"]["version"],
        "candidate_version": candidate_manifest["metadata"]["version"],
        "comparisons": comparisons,
        "overall": overall,
    }
    report["scenario_results"] = build_compare_scenario_results(report)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return report
