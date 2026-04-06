"""Whitebox radar model for RadarSimPy."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .receiver import Receiver
    from .transmitter import Transmitter

BOLTZMANN_CONSTANT = 1.38064852e-23
SQRT_HALF = 0.5**0.5
MILLIWATTS_TO_WATTS = 1e-3


def _interpolate_phase_noise_power(
    freq: NDArray, power: NDArray, f_grid: NDArray, realmin: float
) -> NDArray:
    intrvl_num = len(freq)
    log_p = np.zeros(len(f_grid))

    for intrvl_index in range(intrvl_num):
        left_bound = freq[intrvl_index]
        t1 = power[intrvl_index]

        if intrvl_index == intrvl_num - 1:
            right_bound = f_grid[-1] * 2
            t2 = power[-1]
            inside = np.where(
                np.logical_and(f_grid >= left_bound, f_grid <= right_bound)
            )
        else:
            right_bound = freq[intrvl_index + 1]
            t2 = power[intrvl_index + 1]
            inside = np.where(
                np.logical_and(f_grid >= left_bound, f_grid < right_bound)
            )

        log_p[inside] = t1 + (
            np.log10(f_grid[inside] + realmin) - np.log10(left_bound + realmin)
        ) / (np.log10(right_bound + 2 * realmin) - np.log10(left_bound + realmin)) * (
            t2 - t1
        )

    return 10 ** (np.real(log_p) / 10)


def _generate_noise_spectrum(
    p_interp: NDArray,
    delta_f: NDArray,
    shape: Tuple[int, int],
    rng,
    validation: bool = False,
) -> NDArray:
    row, num_f_points = shape

    if validation:
        awgn_p1 = SQRT_HALF * (
            np.ones((row, num_f_points)) + 1j * np.ones((row, num_f_points))
        )
    else:
        awgn_p1 = SQRT_HALF * (
            rng.standard_normal((row, num_f_points))
            + 1j * rng.standard_normal((row, num_f_points))
        )

    return num_f_points * np.sqrt(delta_f * p_interp) * awgn_p1


def cal_phase_noise(
    signal: NDArray,
    fs: float,
    freq: NDArray,
    power: NDArray,
    seed: Optional[int] = None,
    validation: bool = False,
) -> NDArray:
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if len(freq) != len(power):
        raise ValueError(
            f"freq and power arrays must have same length: "
            f"freq={len(freq)}, power={len(power)}"
        )
    if np.any(freq < 0):
        raise ValueError("All frequency values must be non-negative")

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    signal = signal.astype(complex)

    sort_idx = np.argsort(freq)
    freq = freq[sort_idx]
    power = power[sort_idx]

    cut_idx = np.where(freq < fs / 2)
    freq = freq[cut_idx]
    power = power[cut_idx]

    if not np.any(np.isin(freq, 0)):
        freq = np.concatenate(([0], freq))
        power = np.concatenate(([0], power))

    row, num_samples = np.shape(signal)
    if np.remainder(num_samples, 2):
        num_f_points = int((num_samples + 1) / 2 + 1)
    else:
        num_f_points = int(num_samples / 2 + 1)

    f_grid = np.linspace(0, fs / 2, int(num_f_points))
    delta_f = np.concatenate((np.diff(f_grid), [f_grid[-1] - f_grid[-2]]))
    realmin = float(np.finfo(np.float64).tiny)
    p_interp = _interpolate_phase_noise_power(freq, power, f_grid, realmin)
    spec_noise = _generate_noise_spectrum(
        p_interp, delta_f, (row, num_f_points), rng, validation
    )

    tmp_spec_noise = np.zeros((row, int(num_f_points * 2 - 2)), dtype=complex)
    tmp_spec_noise[:, 0:num_f_points] = spec_noise
    tmp_spec_noise[:, num_f_points : (2 * num_f_points - 2)] = np.fliplr(
        np.conjugate(spec_noise[:, 1:-1])
    )

    spec_noise = tmp_spec_noise
    spec_noise[:, 0] = 0
    x_t = np.fft.ifft(spec_noise, axis=1)
    phase_noise = np.exp(-1j * np.real(x_t[:, 0:num_samples]))
    return signal * phase_noise


class Radar:
    """Radar system model."""

    def __init__(
        self,
        transmitter: "Transmitter",
        receiver: "Receiver",
        frame_time: Union[
            float, int, List[Union[float, int]], Tuple[Union[float, int], ...], NDArray
        ] = 0,
        location: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        speed: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        rotation: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        rotation_rate: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        seed: Optional[int] = None,
        **kwargs,
    ):
        self._seed = seed
        self.time_prop: dict[str, Any] = {}

        samples_per_pulse = int(
            transmitter.waveform_prop["pulse_length"] * receiver.bb_prop["fs"]
        )
        if samples_per_pulse <= 0:
            pulse_length = transmitter.waveform_prop["pulse_length"]
            fs = receiver.bb_prop["fs"]
            product = pulse_length * fs
            raise ValueError(
                f"samples_per_pulse must be greater than 0, got {samples_per_pulse}. "
                f"This occurs when pulse_length ({pulse_length}) * fs ({fs}) = "
                f"{product:.6f} < 1. Either increase the pulse_length or increase "
                "the sampling frequency."
            )

        self.sample_prop: dict[str, Union[int, float, NDArray, None]] = {
            "samples_per_pulse": samples_per_pulse
        }
        self.array_prop: dict[str, Union[int, NDArray]] = {
            "size": (
                transmitter.txchannel_prop["size"] * receiver.rxchannel_prop["size"]
            ),
            "virtual_array": np.repeat(
                transmitter.txchannel_prop["locations"],
                receiver.rxchannel_prop["size"],
                axis=0,
            )
            + np.tile(
                receiver.rxchannel_prop["locations"],
                (transmitter.txchannel_prop["size"], 1),
            ),
        }
        self.radar_prop: dict[str, Any] = {
            "transmitter": transmitter,
            "receiver": receiver,
        }

        self.time_prop["origin_timestamp"] = self._generate_origin_timestamp()
        self.time_prop["origin_timestamp_shape"] = np.shape(
            self.time_prop["origin_timestamp"]
        )
        self.time_prop["frame_start_time"] = np.array(frame_time, dtype=np.float64)
        self.time_prop["timestamp"] = self._generate_timestamp()
        self.time_prop["timestamp_shape"] = np.shape(self.time_prop["timestamp"])

        self.sample_prop["noise"] = self._calculate_noise_amp()

        if (
            transmitter.rf_prop["pn_f"] is not None
            and transmitter.rf_prop["pn_power"] is not None
        ):
            num_pn_samples = (
                np.ceil(
                    (
                        np.max(self.time_prop["origin_timestamp"])
                        - np.min(self.time_prop["origin_timestamp"])
                    )
                    * self.radar_prop["receiver"].bb_prop["fs"]
                ).astype(int)
                + 1
            )
            self.sample_prop["phase_noise"] = cal_phase_noise(
                np.ones((1, num_pn_samples)),
                receiver.bb_prop["fs"],
                transmitter.rf_prop["pn_f"],
                transmitter.rf_prop["pn_power"],
                seed=seed,
                validation=kwargs.get("validation", False),
            )
            self.sample_prop["phase_noise"] = self.sample_prop["phase_noise"].flatten()
        else:
            self.sample_prop["phase_noise"] = None

        self._process_radar_motion(
            list(location),
            list(speed),
            list(rotation),
            list(rotation_rate),
        )

    def set_motion(
        self,
        location: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        speed: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        rotation: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
        rotation_rate: Union[Tuple[float, float, float], List[float]] = (0, 0, 0),
    ) -> None:
        self._process_radar_motion(
            list(location),
            list(speed),
            list(rotation),
            list(rotation_rate),
        )

    def _generate_origin_timestamp(self) -> NDArray:
        channel_size = int(self.array_prop["size"])
        rx_channel_size = int(self.radar_prop["receiver"].rxchannel_prop["size"])
        samples_per_pulse = self.sample_prop["samples_per_pulse"]

        if not isinstance(samples_per_pulse, int) or samples_per_pulse <= 0:
            raise ValueError(
                f"samples_per_pulse must be a positive integer, got {samples_per_pulse}"
            )

        prp = self.radar_prop["transmitter"].waveform_prop["prp"]
        tx_delays = self.radar_prop["transmitter"].txchannel_prop["delay"]
        fs = self.radar_prop["receiver"].bb_prop["fs"]

        sample_times = np.arange(samples_per_pulse, dtype=np.float64) / fs
        pulse_start_times = np.cumsum(prp) - prp[0]
        tx_indices = np.arange(channel_size) // rx_channel_size
        channel_delays = tx_delays[tx_indices]

        return (
            channel_delays[:, np.newaxis, np.newaxis]
            + pulse_start_times[np.newaxis, :, np.newaxis]
            + sample_times[np.newaxis, np.newaxis, :]
        )

    def _generate_timestamp(self) -> NDArray:
        frame_start_time = self.time_prop["frame_start_time"]
        origin_timestamp = self.time_prop["origin_timestamp"]

        if np.size(frame_start_time) > 1:
            num_frames = np.size(frame_start_time)
            channels, pulses, samples = self.time_prop["origin_timestamp_shape"]
            time_offset = np.broadcast_to(
                frame_start_time.reshape(num_frames, 1, 1),
                (num_frames, pulses, samples),
            )
            time_offset = np.repeat(
                time_offset[:, np.newaxis, :, :], channels, axis=1
            ).reshape(num_frames * channels, pulses, samples)
            timestamp = np.tile(origin_timestamp, (num_frames, 1, 1)) + time_offset
        else:
            timestamp = origin_timestamp + frame_start_time

        return timestamp

    def _calculate_noise_amp(self, noise_temp: float = 290) -> float:
        input_noise_dbm = 10 * np.log10(BOLTZMANN_CONSTANT * noise_temp * 1000)
        receiver_noise_dbm = (
            input_noise_dbm
            + self.radar_prop["receiver"].rf_prop["rf_gain"]
            + self.radar_prop["receiver"].rf_prop["noise_figure"]
            + 10 * np.log10(self.radar_prop["receiver"].bb_prop["noise_bandwidth"])
            + self.radar_prop["receiver"].bb_prop["baseband_gain"]
        )
        receiver_noise_watts = MILLIWATTS_TO_WATTS * 10 ** (receiver_noise_dbm / 10)
        return np.sqrt(
            receiver_noise_watts * self.radar_prop["receiver"].bb_prop["load_resistor"]
        )

    def _validate_radar_motion(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
    ) -> None:
        if len(location) != 3:
            raise ValueError(f"location must have 3 elements, got {len(location)}")
        if len(speed) != 3:
            raise ValueError(f"speed must have 3 elements, got {len(speed)}")
        if len(rotation) != 3:
            raise ValueError(f"rotation must have 3 elements, got {len(rotation)}")
        if len(rotation_rate) != 3:
            raise ValueError(
                f"rotation_rate must have 3 elements, got {len(rotation_rate)}"
            )

        has_time_varying_location = any(np.size(location[idx]) > 1 for idx in range(3))
        has_time_varying_rotation = any(np.size(rotation[idx]) > 1 for idx in range(3))

        if has_time_varying_location or has_time_varying_rotation:
            speed_array = np.array(speed)
            if np.any(speed_array != 0):
                raise ValueError(
                    "When using time-varying location or rotation, speed must be [0, 0, 0]. "
                    f"Got speed={speed}. Time-varying motion should be specified directly "
                    "in the location/rotation arrays, not through constant velocities."
                )

            rotation_rate_array = np.array(rotation_rate)
            if np.any(rotation_rate_array != 0):
                raise ValueError(
                    "When using time-varying location or rotation, rotation_rate must be [0, 0, 0]. "
                    f"Got rotation_rate={rotation_rate}. Time-varying motion should be specified "
                    "directly in the location/rotation arrays, not through constant angular velocities."
                )

        for idx, coord in enumerate(["x", "y", "z"]):
            if np.size(location[idx]) > 1:
                if np.shape(location[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        f"location[{coord}] must be a scalar or have the same shape as timestamp. "
                        f"Got shape {np.shape(location[idx])}, expected {self.time_prop['timestamp_shape']}"
                    )
            if np.size(rotation[idx]) > 1:
                if np.shape(rotation[idx]) != self.time_prop["timestamp_shape"]:
                    raise ValueError(
                        f"rotation[{coord}] must be a scalar or have the same shape as timestamp. "
                        f"Got shape {np.shape(rotation[idx])}, expected {self.time_prop['timestamp_shape']}"
                    )
            if np.size(speed[idx]) > 1:
                raise ValueError(
                    f"speed[{coord}] must be a scalar. Time-varying speed arrays are not supported. "
                    "Use time-varying location arrays instead."
                )
            if np.size(rotation_rate[idx]) > 1:
                raise ValueError(
                    f"rotation_rate[{coord}] must be a scalar. Time-varying rotation_rate arrays are not supported. "
                    "Use time-varying rotation arrays instead."
                )

    def _process_radar_motion(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
    ) -> None:
        self._validate_radar_motion(location, speed, rotation, rotation_rate)
        shape = self.time_prop["timestamp_shape"]
        has_time_varying_motion = any(
            np.size(var) > 1 for var in list(location) + list(rotation)
        )

        if has_time_varying_motion:
            self._setup_time_varying_motion(
                location, speed, rotation, rotation_rate, shape
            )
        else:
            self._setup_static_motion(location, speed, rotation, rotation_rate)

    def _setup_time_varying_motion(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
        shape: Tuple[int, ...],
    ) -> None:
        self.radar_prop["location"] = np.zeros(shape + (3,))
        self.radar_prop["rotation"] = np.zeros(shape + (3,))
        self.radar_prop["speed"] = np.array(speed)
        self.radar_prop["rotation_rate"] = np.radians(np.array(rotation_rate))

        for idx in range(3):
            self._process_location_dimension(location, speed, idx)
            self._process_rotation_dimension(rotation, rotation_rate, idx)

    def _process_location_dimension(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        idx: int,
    ) -> None:
        if np.size(location[idx]) > 1:
            self.radar_prop["location"][:, :, :, idx] = location[idx]
        else:
            self.radar_prop["location"][:, :, :, idx] = (
                location[idx] + speed[idx] * self.time_prop["timestamp"]
            )

    def _process_rotation_dimension(
        self,
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
        idx: int,
    ) -> None:
        if np.size(rotation[idx]) > 1:
            self.radar_prop["rotation"][:, :, :, idx] = np.radians(rotation[idx])
        else:
            self.radar_prop["rotation"][:, :, :, idx] = (
                np.radians(rotation[idx])
                + np.radians(rotation_rate[idx]) * self.time_prop["timestamp"]
            )

    def _setup_static_motion(
        self,
        location: List[Union[float, NDArray]],
        speed: List[Union[float, NDArray]],
        rotation: List[Union[float, NDArray]],
        rotation_rate: List[Union[float, NDArray]],
    ) -> None:
        self.radar_prop["speed"] = np.array(speed)
        self.radar_prop["location"] = np.array(location)
        self.radar_prop["rotation"] = np.radians(np.array(rotation))
        self.radar_prop["rotation_rate"] = np.radians(np.array(rotation_rate))

    @property
    def num_channels(self) -> int:
        return int(self.array_prop["size"])

    @property
    def samples_per_pulse(self) -> int:
        samples = self.sample_prop["samples_per_pulse"]
        assert isinstance(samples, int)
        return samples

    @property
    def transmitter(self):
        return self.radar_prop["transmitter"]

    @property
    def receiver(self):
        return self.radar_prop["receiver"]

    @property
    def virtual_array_locations(self) -> NDArray:
        virtual_array = self.array_prop["virtual_array"]
        assert isinstance(virtual_array, np.ndarray)
        return virtual_array

    def __str__(self) -> str:
        return (
            f"Radar(channels={self.num_channels}, "
            f"samples_per_pulse={self.samples_per_pulse}, "
            f"fs={self.receiver.bb_prop['fs']/1e6:.1f} MHz)"
        )

    def __repr__(self) -> str:
        return (
            f"Radar(transmitter={self.transmitter.__class__.__name__}, "
            f"receiver={self.receiver.__class__.__name__}, "
            f"channels={self.num_channels}, "
            f"samples_per_pulse={self.samples_per_pulse})"
        )
