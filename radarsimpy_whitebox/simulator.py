"""Whitebox simulator entry points.

Current scope:

- `sim_radar`: reference implementation for ideal point targets and static mesh targets
- `sim_lidar`: structured-array reference path for mesh point clouds
- `sim_rcs`: geometry-backed far-field reference path
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional
import warnings

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from . import mesh_kit
from .radar import Radar

_LIDAR_HIT_DTYPE = np.dtype(
    [
        ("positions", np.float64, (3,)),
        ("origins", np.float64, (3,)),
        ("directions", np.float64, (3,)),
        ("distance", np.float64),
        ("phi", np.float64),
        ("theta", np.float64),
        ("target_index", np.int64),
    ]
)


def _validate_sim_radar_args(
    radar: Radar,
    density: float,
    level,
    interf,
    ray_filter,
    back_propagating: bool,
    device: str,
    log_path,
):
    if not isinstance(radar, Radar):
        raise TypeError("radar must be a Radar instance")
    if density <= 0:
        raise ValueError("density must be positive")
    if level not in (None, "frame", "pulse", "sample"):
        raise ValueError("level must be one of None, 'frame', 'pulse', or 'sample'")
    if interf is not None:
        raise NotImplementedError(
            "Whitebox sim_radar currently does not support interference radar input"
        )
    if ray_filter is not None:
        warnings.warn(
            "ray_filter is ignored by the current whitebox simulator",
            RuntimeWarning,
            stacklevel=2,
        )
    if back_propagating:
        warnings.warn(
            "back_propagating is ignored by the current whitebox simulator",
            RuntimeWarning,
            stacklevel=2,
        )
    if device not in ("gpu", "cpu"):
        raise ValueError("device must be 'gpu' or 'cpu'")
    if log_path is not None:
        warnings.warn(
            "log_path is ignored by the current whitebox simulator",
            RuntimeWarning,
            stacklevel=2,
        )


def _rotation_matrix_zyx(rotation_rad: np.ndarray) -> np.ndarray:
    yaw = rotation_rad[..., 0]
    pitch = rotation_rad[..., 1]
    roll = rotation_rad[..., 2]

    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    matrix = np.empty(rotation_rad.shape[:-1] + (3, 3), dtype=float)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def _rotate_local_to_global(local_vector: np.ndarray, rotation_rad: np.ndarray) -> np.ndarray:
    matrix = _rotation_matrix_zyx(rotation_rad)
    local = np.broadcast_to(local_vector, rotation_rad.shape[:-1] + (3,))
    return np.einsum("...ij,...j->...i", matrix, local)


def _rotate_global_to_local(global_vector: np.ndarray, rotation_rad: np.ndarray) -> np.ndarray:
    matrix = _rotation_matrix_zyx(rotation_rad)
    return np.einsum("...ji,...j->...i", matrix, global_vector)


def _pattern_gain_db(angle_deg: np.ndarray, angle_grid: np.ndarray, pattern_db: np.ndarray) -> np.ndarray:
    return np.interp(
        angle_deg,
        angle_grid,
        pattern_db,
        left=pattern_db[0],
        right=pattern_db[-1],
    )


def _antenna_direction_gain_db(
    direction_global: np.ndarray,
    rotation_rad: np.ndarray,
    az_angles: np.ndarray,
    az_pattern: np.ndarray,
    el_angles: np.ndarray,
    el_pattern: np.ndarray,
    peak_gain_db: float,
) -> np.ndarray:
    local_direction = _rotate_global_to_local(direction_global, rotation_rad)
    local_norm = np.linalg.norm(local_direction, axis=-1, keepdims=True)
    safe_direction = np.divide(
        local_direction,
        np.maximum(local_norm, np.finfo(float).eps),
    )

    azimuth_deg = np.degrees(np.arctan2(safe_direction[..., 1], safe_direction[..., 0]))
    elevation_deg = np.degrees(
        np.arctan2(
            safe_direction[..., 2],
            np.sqrt(safe_direction[..., 0] ** 2 + safe_direction[..., 1] ** 2),
        )
    )

    az_gain = _pattern_gain_db(azimuth_deg, az_angles, az_pattern)
    el_gain = _pattern_gain_db(elevation_deg, el_angles, el_pattern)
    return peak_gain_db + az_gain + el_gain


def _build_waveform_phase_function(frequency: np.ndarray, time_grid: np.ndarray):
    frequency = np.asarray(frequency, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    dt = np.diff(time_grid)
    trapezoids = 0.5 * (frequency[:-1] + frequency[1:]) * dt
    cumulative = np.concatenate(([0.0], np.cumsum(trapezoids)))

    def evaluate(time_value):
        t = np.asarray(time_value, dtype=float)
        clipped = np.clip(t, time_grid[0], time_grid[-1])
        indices = np.searchsorted(time_grid, clipped, side="right") - 1
        indices = np.clip(indices, 0, len(time_grid) - 2)

        t0 = time_grid[indices]
        t1 = time_grid[indices + 1]
        f0 = frequency[indices]
        f1 = frequency[indices + 1]
        slope = np.divide(
            f1 - f0,
            t1 - t0,
            out=np.zeros_like(f0),
            where=(t1 - t0) != 0,
        )
        delta = clipped - t0
        integral = cumulative[indices] + f0 * delta + 0.5 * slope * delta**2
        return 2 * np.pi * integral

    return evaluate


def _evaluate_waveform_modulation(modulation: Dict, local_time: np.ndarray) -> np.ndarray:
    if not modulation["enabled"]:
        return np.ones_like(local_time, dtype=complex)

    mod_t = np.asarray(modulation["t"], dtype=float)
    mod_var = np.asarray(modulation["var"], dtype=complex)
    real_part = np.interp(
        local_time, mod_t, np.real(mod_var), left=np.real(mod_var[0]), right=np.real(mod_var[-1])
    )
    imag_part = np.interp(
        local_time, mod_t, np.imag(mod_var), left=np.imag(mod_var[0]), right=np.imag(mod_var[-1])
    )
    return real_part + 1j * imag_part


def _coerce_target_location_component(
    component, timestamp_shape, full_timestamp_shape, channel_index: int, pulse_index: int
):
    if np.size(component) == 1:
        return None, float(component)

    component_array = np.asarray(component, dtype=float)
    if component_array.shape == full_timestamp_shape:
        return component_array[channel_index, pulse_index], None
    if component_array.shape == timestamp_shape:
        return component_array, None
    raise ValueError(
        "Time-varying point target location components must match either "
        "radar.timestamp shape or the active timestamp slice shape"
    )


def _resolve_point_target_position(
    target: Dict,
    timestamp_slice: np.ndarray,
    full_timestamp_shape,
    channel_index: int,
    pulse_index: int,
) -> np.ndarray:
    location = target.get("location", [0.0, 0.0, 0.0])
    if len(location) != 3:
        raise ValueError("Point target location must have three components")
    speed = np.asarray(target.get("speed", [0.0, 0.0, 0.0]), dtype=float)
    if speed.shape != (3,):
        raise ValueError("Point target speed must be a length-3 vector")

    result = np.zeros(timestamp_slice.shape + (3,), dtype=float)
    for idx in range(3):
        component_array, component_scalar = _coerce_target_location_component(
            location[idx],
            timestamp_slice.shape,
            full_timestamp_shape,
            channel_index,
            pulse_index,
        )
        if component_array is not None:
            result[..., idx] = component_array
        else:
            result[..., idx] = component_scalar + speed[idx] * timestamp_slice
    return result


def _ensure_point_target(target: Dict) -> None:
    allowed_keys = {"location", "rcs", "speed", "phase"}
    unknown_keys = sorted(set(target) - allowed_keys)
    if unknown_keys:
        warnings.warn(
            "Ignoring unsupported point-target keys: " + ", ".join(unknown_keys),
            RuntimeWarning,
            stacklevel=2,
        )
    if "location" not in target:
        raise ValueError("Point target requires 'location'")
    if "rcs" not in target:
        raise ValueError("Point target requires 'rcs'")


def _ensure_mesh_target(target: Dict) -> None:
    allowed_keys = {
        "model",
        "location",
        "rotation",
        "origin",
        "unit",
        "permittivity",
        "phase",
        "rcs",
    }
    unknown_keys = sorted(set(target) - allowed_keys)
    if unknown_keys:
        warnings.warn(
            "Ignoring unsupported mesh-target keys: " + ", ".join(unknown_keys),
            RuntimeWarning,
            stacklevel=2,
        )

    if "model" not in target:
        raise ValueError("Mesh target requires 'model'")

    for vector_key in ("location", "rotation", "origin"):
        if vector_key not in target:
            continue
        vector = np.asarray(target[vector_key], dtype=float)
        if vector.shape != (3,):
            raise NotImplementedError(
                f"Mesh target '{vector_key}' must be a static length-3 vector"
            )

    if "speed" in target or "rotation_rate" in target:
        raise NotImplementedError(
            "Whitebox sim_radar mesh target motion is not implemented yet"
        )


def _prepare_sim_target(target: Dict) -> Dict:
    if "model" not in target:
        _ensure_point_target(target)
        return {"kind": "point", "target": target}

    _ensure_mesh_target(target)
    points, cells = _load_target_mesh(target)
    area, normals, centroids = _triangle_geometry(points, cells)
    valid_facets = area > 0
    return {
        "kind": "mesh",
        "areas": area[valid_facets],
        "normals": normals[valid_facets],
        "centroids": centroids[valid_facets],
        "reflectivity": _reflectivity_from_permittivity(target.get("permittivity")),
        "phase": np.exp(1j * np.deg2rad(float(target.get("phase", 0.0)))),
        "rcs_scale": 10 ** (float(target.get("rcs", 0.0)) / 10.0),
    }


def _channel_indices(radar: Radar, channel_index: int) -> tuple[int, int]:
    rx_size = radar.receiver.rxchannel_prop["size"]
    local_channel = channel_index % radar.num_channels
    tx_index = local_channel // rx_size
    rx_index = local_channel % rx_size
    return int(tx_index), int(rx_index)


def _channel_state(radar: Radar, channel_index: int):
    timestamp_slice = radar.time_prop["timestamp"][channel_index]
    pulses, samples = timestamp_slice.shape

    location = radar.radar_prop["location"]
    if np.ndim(location) == 1:
        radar_location = np.broadcast_to(np.asarray(location, dtype=float), (pulses, samples, 3))
    else:
        radar_location = np.asarray(location[channel_index], dtype=float)

    rotation = radar.radar_prop["rotation"]
    if np.ndim(rotation) == 1:
        radar_rotation = np.broadcast_to(np.asarray(rotation, dtype=float), (pulses, samples, 3))
    else:
        radar_rotation = np.asarray(rotation[channel_index], dtype=float)

    tx_index, rx_index = _channel_indices(radar, channel_index)
    tx_offset = np.asarray(radar.transmitter.txchannel_prop["locations"][tx_index], dtype=float)
    rx_offset = np.asarray(radar.receiver.rxchannel_prop["locations"][rx_index], dtype=float)

    tx_position = radar_location + _rotate_local_to_global(tx_offset, radar_rotation)
    rx_position = radar_location + _rotate_local_to_global(rx_offset, radar_rotation)

    return {
        "timestamp": timestamp_slice,
        "rotation": radar_rotation,
        "tx_position": tx_position,
        "rx_position": rx_position,
        "channel_index": channel_index,
        "local_channel_index": channel_index % radar.num_channels,
        "tx_index": tx_index,
        "rx_index": rx_index,
    }


def _scatterer_signal(
    radar: Radar,
    state: Dict,
    pulse_index: int,
    scatterer_position: np.ndarray,
    sigma_linear,
    scatterer_phase,
    local_sample_time: np.ndarray,
    pulse_length: float,
    waveform_phase,
    base_frequency: float,
) -> np.ndarray:
    tx = radar.transmitter
    rx = radar.receiver
    tx_index = state["tx_index"]
    rx_index = state["rx_index"]

    rotation = state["rotation"][pulse_index]
    tx_position = state["tx_position"][pulse_index]
    rx_position = state["rx_position"][pulse_index]

    vec_tx = scatterer_position - tx_position
    vec_rx = scatterer_position - rx_position
    range_tx = np.linalg.norm(vec_tx, axis=-1)
    range_rx = np.linalg.norm(vec_rx, axis=-1)
    total_delay = (range_tx + range_rx) / SPEED_OF_LIGHT

    delayed_local_time = local_sample_time - total_delay
    valid = (delayed_local_time >= 0.0) & (delayed_local_time <= pulse_length)
    if not np.any(valid):
        return np.zeros(state["timestamp"][pulse_index].shape, dtype=complex)

    tx_gain_db = _antenna_direction_gain_db(
        vec_tx,
        rotation,
        np.asarray(tx.txchannel_prop["az_angles"][tx_index], dtype=float),
        np.asarray(tx.txchannel_prop["az_patterns"][tx_index], dtype=float),
        np.asarray(tx.txchannel_prop["el_angles"][tx_index], dtype=float),
        np.asarray(tx.txchannel_prop["el_patterns"][tx_index], dtype=float),
        float(tx.txchannel_prop["antenna_gains"][tx_index]),
    )
    rx_gain_db = _antenna_direction_gain_db(
        -vec_rx,
        rotation,
        np.asarray(rx.rxchannel_prop["az_angles"][rx_index], dtype=float),
        np.asarray(rx.rxchannel_prop["az_patterns"][rx_index], dtype=float),
        np.asarray(rx.rxchannel_prop["el_angles"][rx_index], dtype=float),
        np.asarray(rx.rxchannel_prop["el_patterns"][rx_index], dtype=float),
        float(rx.rxchannel_prop["antenna_gains"][rx_index]),
    )

    lambda_val = SPEED_OF_LIGHT / (
        base_frequency + tx.waveform_prop["f_offset"][pulse_index]
    )
    tx_power_watts = 1e-3 * 10 ** (float(tx.rf_prop["tx_power"]) / 10)
    received_power_watts = (
        tx_power_watts
        * 10 ** (tx_gain_db / 10)
        * 10 ** (rx_gain_db / 10)
        * lambda_val**2
        * np.maximum(sigma_linear, 0.0)
        / (
            (4 * np.pi) ** 3
            * np.maximum(range_tx, 1e-12) ** 2
            * np.maximum(range_rx, 1e-12) ** 2
        )
    )
    signal_amplitude = np.sqrt(
        np.maximum(received_power_watts, 0.0) * rx.bb_prop["load_resistor"]
    )

    phase_difference = waveform_phase(local_sample_time) - waveform_phase(
        delayed_local_time
    )
    phase_difference += (
        2 * np.pi * tx.waveform_prop["f_offset"][pulse_index] * total_delay
    )

    waveform_mod = _evaluate_waveform_modulation(
        tx.txchannel_prop["waveform_mod"][tx_index], delayed_local_time
    )
    pulse_mod = tx.txchannel_prop["pulse_mod"][tx_index, pulse_index]
    signal = signal_amplitude * pulse_mod * waveform_mod * scatterer_phase * np.exp(
        1j * phase_difference
    )
    signal = np.where(valid, signal, 0.0)

    phase_noise = radar.sample_prop["phase_noise"]
    if phase_noise is not None:
        pn_indices = np.rint(
            radar.time_prop["origin_timestamp"][state["local_channel_index"], pulse_index]
            * rx.bb_prop["fs"]
        ).astype(int)
        pn_indices = np.clip(pn_indices, 0, len(phase_noise) - 1)
        signal = signal * phase_noise[pn_indices]

    return signal


def _mesh_sigma_linear(
    centroid: np.ndarray,
    normal: np.ndarray,
    area: float,
    reflectivity: float,
    rcs_scale: float,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    vec_tx = centroid - tx_position
    vec_rx = centroid - rx_position
    tx_dir = _normalize(vec_tx)
    rx_dir = _normalize(vec_rx)

    illumination = np.maximum(0.0, np.sum(tx_dir * normal, axis=-1))
    visibility = np.maximum(0.0, np.sum(rx_dir * normal, axis=-1))
    effective_area = reflectivity * area * illumination * visibility
    return (
        rcs_scale
        * 4.0
        * np.pi
        * effective_area**2
        / np.maximum(wavelength**2, np.finfo(float).eps)
    )


def sim_radar(
    radar: Radar,
    targets: Iterable[Dict],
    density: float = 1,
    level=None,
    interf=None,
    ray_filter=None,
    back_propagating: bool = False,
    device: str = "gpu",
    log_path=None,
    dry_run: bool = False,
):
    """Simulate a radar scene with point targets or static mesh targets.

    This is a whitebox reference path for ideal point targets and static mesh
    targets. Interference modeling and full ray-tracing behavior are still out
    of scope on the current host.
    """

    _validate_sim_radar_args(
        radar, density, level, interf, ray_filter, back_propagating, device, log_path
    )

    timestamp = np.asarray(radar.time_prop["timestamp"])
    output_shape = timestamp.shape
    bb_type = radar.receiver.bb_prop["bb_type"]
    signal_dtype = float if bb_type == "real" else complex
    baseband = np.zeros(output_shape, dtype=complex)

    if dry_run:
        zeros = np.zeros(output_shape, dtype=signal_dtype)
        return {
            "timestamp": timestamp.copy(),
            "baseband": zeros.copy(),
            "noise": zeros.copy(),
            "interference": None,
        }

    targets = [_prepare_sim_target(target) for target in targets]

    tx = radar.transmitter
    rx = radar.receiver
    samples = radar.samples_per_pulse
    local_sample_time = np.arange(samples, dtype=float) / rx.bb_prop["fs"]
    pulse_length = tx.waveform_prop["pulse_length"]
    waveform_phase = _build_waveform_phase_function(
        np.asarray(tx.waveform_prop["f"], dtype=float),
        np.asarray(tx.waveform_prop["t"], dtype=float),
    )
    base_frequency = float(np.mean(tx.waveform_prop["f"]))
    for channel_index in range(output_shape[0]):
        state = _channel_state(radar, channel_index)

        for pulse_index in range(output_shape[1]):
            wavelength = SPEED_OF_LIGHT / (
                base_frequency + tx.waveform_prop["f_offset"][pulse_index]
            )

            for target in targets:
                if target["kind"] == "point":
                    point_target = target["target"]
                    target_position = _resolve_point_target_position(
                        point_target,
                        state["timestamp"][pulse_index],
                        radar.time_prop["timestamp_shape"],
                        channel_index,
                        pulse_index,
                    )
                    sigma = 10 ** (float(point_target["rcs"]) / 10)
                    target_phase = np.exp(
                        1j * np.deg2rad(float(point_target.get("phase", 0.0)))
                    )
                    baseband[channel_index, pulse_index, :] += _scatterer_signal(
                        radar,
                        state,
                        pulse_index,
                        target_position,
                        sigma,
                        target_phase,
                        local_sample_time,
                        pulse_length,
                        waveform_phase,
                        base_frequency,
                    )
                    continue

                for facet_index in range(len(target["areas"])):
                    centroid = target["centroids"][facet_index]
                    sigma = _mesh_sigma_linear(
                        centroid,
                        target["normals"][facet_index],
                        float(target["areas"][facet_index]),
                        float(target["reflectivity"]),
                        float(target["rcs_scale"]),
                        state["tx_position"][pulse_index],
                        state["rx_position"][pulse_index],
                        wavelength,
                    )
                    if not np.any(sigma > 0.0):
                        continue

                    baseband[channel_index, pulse_index, :] += _scatterer_signal(
                        radar,
                        state,
                        pulse_index,
                        centroid,
                        sigma,
                        target["phase"],
                        local_sample_time,
                        pulse_length,
                        waveform_phase,
                        base_frequency,
                    )

    # The captured vendor runtime uses the seed for deterministic signal paths,
    # but thermal noise is not reproducible across repeated runs with the same seed.
    rng = np.random.default_rng()
    if bb_type == "complex":
        noise_std = float(radar.sample_prop["noise"])
        noise = (noise_std / np.sqrt(2.0)) * (
            rng.standard_normal(output_shape) + 1j * rng.standard_normal(output_shape)
        )
        baseband_out = baseband.astype(complex)
    else:
        noise_std = float(radar.sample_prop["noise"])
        noise = noise_std * rng.standard_normal(output_shape)
        baseband_out = np.real(baseband).astype(float)

    return {
        "timestamp": timestamp.copy(),
        "baseband": baseband_out,
        "noise": noise.astype(signal_dtype, copy=False),
        "interference": None,
    }

def _validate_lidar_config(lidar: Dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(lidar, dict):
        raise TypeError("lidar must be a dictionary")

    missing = sorted({"position", "phi", "theta"} - set(lidar))
    if missing:
        raise ValueError("Lidar configuration missing keys: " + ", ".join(missing))

    position = np.asarray(lidar["position"], dtype=float)
    if position.shape != (3,):
        raise ValueError("lidar['position'] must be a length-3 vector")

    phi = np.asarray(lidar["phi"], dtype=float).reshape(-1)
    theta = np.asarray(lidar["theta"], dtype=float).reshape(-1)
    return position, phi, theta


def _ensure_lidar_target(target: Dict) -> None:
    allowed_keys = {
        "model",
        "origin",
        "location",
        "speed",
        "rotation",
        "rotation_rate",
        "unit",
        "permittivity",
        "rcs",
        "phase",
        "skip_diffusion",
        "environment",
    }
    unknown_keys = sorted(set(target) - allowed_keys)
    if unknown_keys:
        warnings.warn(
            "Ignoring unsupported lidar-target keys: " + ", ".join(unknown_keys),
            RuntimeWarning,
            stacklevel=2,
        )

    if "model" not in target:
        raise ValueError("Lidar target requires 'model'")

    _target_vector(target, "origin", [0.0, 0.0, 0.0])
    _target_vector(target, "location", [0.0, 0.0, 0.0])
    _target_vector(target, "speed", [0.0, 0.0, 0.0])
    _target_vector(target, "rotation", [0.0, 0.0, 0.0])
    _target_vector(target, "rotation_rate", [0.0, 0.0, 0.0])


def _lidar_angle_to_direction(phi_deg, theta_deg) -> np.ndarray:
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    return np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=-1,
    )


def _ray_triangle_intersection(
    origin: np.ndarray, direction: np.ndarray, triangle: np.ndarray
) -> Optional[float]:
    eps = np.finfo(float).eps * 16
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    h_vec = np.cross(direction, edge2)
    det = float(np.dot(edge1, h_vec))
    if abs(det) <= eps:
        return None

    inv_det = 1.0 / det
    s_vec = origin - v0
    u = inv_det * float(np.dot(s_vec, h_vec))
    if u < 0.0 or u > 1.0:
        return None

    q_vec = np.cross(s_vec, edge1)
    v = inv_det * float(np.dot(direction, q_vec))
    if v < 0.0 or (u + v) > 1.0:
        return None

    distance = inv_det * float(np.dot(edge2, q_vec))
    if distance <= eps:
        return None

    return distance


def _reflect_direction(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    direction = _normalize(np.asarray(direction, dtype=float))
    normal = _normalize(np.asarray(normal, dtype=float))
    return direction - 2.0 * np.sum(direction * normal, axis=-1, keepdims=True) * normal


def _ray_scene_intersection(
    origin: np.ndarray, direction: np.ndarray, triangles_by_target
) -> Optional[tuple[float, np.ndarray, int, np.ndarray]]:
    best_distance = np.inf
    best_position = None
    best_target_index = -1
    best_normal = None

    for target_index, triangles in triangles_by_target:
        for triangle in triangles:
            distance = _ray_triangle_intersection(origin, direction, triangle)
            if distance is None or distance >= best_distance:
                continue
            best_distance = distance
            best_position = origin + distance * direction
            best_target_index = target_index
            best_normal = _normalize(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )

    if best_position is None:
        return None

    return best_distance, best_position, best_target_index, best_normal


def sim_lidar(lidar, targets, frame_time=0):
    """Simulate a lidar scene and return hit rays as a structured array.

    The local whitebox reference returns one structured-array row per hit.
    Rays that miss all targets are excluded from the output.
    """

    position, phi, theta = _validate_lidar_config(lidar)
    if np.ndim(frame_time) != 0:
        raise ValueError("frame_time must be a scalar")

    frame_time = float(frame_time)
    targets = list(targets)
    for target in targets:
        _ensure_lidar_target(target)

    triangles_by_target = []
    for target_index, target in enumerate(targets):
        points, cells = _load_target_mesh(target, frame_time=frame_time)
        triangles_by_target.append((target_index, points[cells]))

    if phi.size == 0 or theta.size == 0 or not triangles_by_target:
        return np.empty((0,), dtype=_LIDAR_HIT_DTYPE)

    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
    phi_flat = phi_grid.reshape(-1)
    theta_flat = theta_grid.reshape(-1)
    directions = _lidar_angle_to_direction(phi_flat, theta_flat)

    records = []
    for ray_index, direction in enumerate(directions):
        hit = _ray_scene_intersection(position, direction, triangles_by_target)
        if hit is None:
            continue

        distance, hit_position, target_index, hit_normal = hit
        reflected_direction = _reflect_direction(direction, hit_normal)
        records.append(
            (
                hit_position,
                position,
                reflected_direction,
                distance,
                float(phi_flat[ray_index]),
                float(theta_flat[ray_index]),
                int(target_index),
            )
        )

    if not records:
        return np.empty((0,), dtype=_LIDAR_HIT_DTYPE)

    result = np.empty((len(records),), dtype=_LIDAR_HIT_DTYPE)
    result["positions"] = np.asarray([record[0] for record in records], dtype=float)
    result["origins"] = np.asarray([record[1] for record in records], dtype=float)
    result["directions"] = np.asarray([record[2] for record in records], dtype=float)
    result["distance"] = np.asarray([record[3] for record in records], dtype=float)
    result["phi"] = np.asarray([record[4] for record in records], dtype=float)
    result["theta"] = np.asarray([record[5] for record in records], dtype=float)
    result["target_index"] = np.asarray([record[6] for record in records], dtype=np.int64)
    return result


def _normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return np.divide(vec, np.maximum(norm, np.finfo(float).eps))


def _angle_to_direction(phi_deg, theta_deg) -> np.ndarray:
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    return np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=-1,
    )


def _unit_scale_divisor(unit: str) -> float:
    mapping = {"m": 1.0, "cm": 100.0, "mm": 1000.0}
    if unit not in mapping:
        raise ValueError("unit must be one of 'mm', 'cm', or 'm'")
    return mapping[unit]


def _target_vector(target: Dict, key: str, default) -> np.ndarray:
    vector = np.asarray(target.get(key, default), dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"Target '{key}' must be a length-3 vector")
    return vector


def _target_pose_at_time(target: Dict, frame_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = _target_vector(target, "origin", [0.0, 0.0, 0.0])
    location = _target_vector(target, "location", [0.0, 0.0, 0.0])
    speed = _target_vector(target, "speed", [0.0, 0.0, 0.0])
    rotation_deg = _target_vector(target, "rotation", [0.0, 0.0, 0.0])
    rotation_rate = _target_vector(target, "rotation_rate", [0.0, 0.0, 0.0])
    return origin, location + speed * frame_time, rotation_deg + rotation_rate * frame_time


def _transform_points(points: np.ndarray, target: Dict, frame_time: float = 0.0) -> np.ndarray:
    origin, location, rotation_deg = _target_pose_at_time(target, frame_time)
    rotation_rad = np.deg2rad(rotation_deg)

    shifted = points - origin
    rotation_matrix = _rotation_matrix_zyx(rotation_rad)
    rotated = shifted @ rotation_matrix.T
    return rotated + origin + location


def _load_target_mesh(target: Dict, frame_time: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if "model" not in target:
        raise ValueError("Mesh target requires 'model'")

    unit = target.get("unit", "m")
    divisor = _unit_scale_divisor(unit)
    module = mesh_kit.import_mesh_module()
    mesh = mesh_kit.load_mesh(target["model"], divisor, module)

    points = _transform_points(np.asarray(mesh["points"], dtype=float), target, frame_time)
    cells = np.asarray(mesh["cells"], dtype=int)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(
            "Whitebox mesh-backed simulator paths currently support triangular meshes only"
        )
    return points, cells


def _triangle_geometry(points: np.ndarray, cells: np.ndarray):
    tri = points[cells]
    edge1 = tri[:, 1, :] - tri[:, 0, :]
    edge2 = tri[:, 2, :] - tri[:, 0, :]
    cross = np.cross(edge1, edge2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normals = _normalize(cross)
    centroids = np.mean(tri, axis=1)
    return area, normals, centroids


def _reflectivity_from_permittivity(permittivity) -> float:
    if permittivity is None:
        return 1.0

    eps = complex(permittivity)
    try:
        root = np.sqrt(eps)
    except Exception:
        return 1.0

    denom = root + 1
    if abs(denom) < np.finfo(float).eps:
        return 1.0

    gamma = (root - 1) / denom
    return float(abs(gamma))


def _polarization_gain(inc_pol, obs_pol) -> float:
    inc = _normalize(np.asarray(inc_pol, dtype=float))
    obs = _normalize(np.asarray(obs_pol, dtype=float))
    return float(abs(np.dot(inc, obs)))


def sim_rcs(
    targets,
    f,
    inc_phi,
    inc_theta,
    inc_pol=[0, 0, 1],
    obs_phi=None,
    obs_theta=None,
    obs_pol=None,
    density: float = 1.0,
):
    """Reference whitebox RCS calculation for triangular mesh targets.

    This is not an SBR implementation. It is a far-field coherent facet-sum
    reference path intended to unblock whitebox development until oracle-backed
    mesh/ray-tracing parity is available.
    """

    if density <= 0:
        raise ValueError("density must be positive")

    if obs_phi is None:
        obs_phi = inc_phi
    if obs_theta is None:
        obs_theta = inc_theta
    if obs_pol is None:
        obs_pol = inc_pol

    inc_phi_arr, inc_theta_arr, obs_phi_arr, obs_theta_arr = np.broadcast_arrays(
        np.asarray(inc_phi, dtype=float),
        np.asarray(inc_theta, dtype=float),
        np.asarray(obs_phi, dtype=float),
        np.asarray(obs_theta, dtype=float),
    )

    source_hat = _angle_to_direction(inc_phi_arr, inc_theta_arr)
    observer_hat = _angle_to_direction(obs_phi_arr, obs_theta_arr)

    wavelength = SPEED_OF_LIGHT / float(f)
    wavenumber = 2 * np.pi / wavelength
    pol_gain = _polarization_gain(inc_pol, obs_pol)

    fields = np.zeros(inc_phi_arr.shape, dtype=complex)
    for target in targets:
        points, cells = _load_target_mesh(target)
        area, normals, centroids = _triangle_geometry(points, cells)
        reflectivity = _reflectivity_from_permittivity(target.get("permittivity"))

        for facet_idx in range(len(area)):
            n = normals[facet_idx]
            a = area[facet_idx]
            centroid = centroids[facet_idx]

            illum = np.maximum(0.0, np.sum(source_hat * n, axis=-1))
            visible = np.maximum(0.0, np.sum(observer_hat * n, axis=-1))
            amplitude = reflectivity * pol_gain * a * illum * visible

            phase = np.exp(
                -1j
                * wavenumber
                * (
                    np.sum(observer_hat * centroid, axis=-1)
                    - np.sum(source_hat * centroid, axis=-1)
                )
            )
            fields += amplitude * phase

    rcs = 4 * np.pi * np.abs(fields) ** 2 / np.maximum(wavelength**2, np.finfo(float).eps)
    if rcs.shape == ():
        return float(rcs)
    return rcs
