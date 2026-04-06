"""Whitebox receiver model for RadarSimPy."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

DEFAULT_POLARIZATION = [0, 0, 1]
DEFAULT_AZIMUTH_RANGE = [-90, 90]
DEFAULT_ELEVATION_RANGE = [-90, 90]
DEFAULT_PATTERN_DB = [0, 0]
VALID_BB_TYPES = {"complex", "real"}


class Receiver:
    """Radar receiver configuration model."""

    def __init__(
        self,
        fs: float,
        noise_figure: float = 10,
        rf_gain: float = 0,
        load_resistor: float = 500,
        baseband_gain: float = 0,
        bb_type: str = "complex",
        channels: Optional[List[Dict]] = None,
    ):
        if fs <= 0:
            raise ValueError("Sampling rate (fs) must be positive")
        if not isinstance(noise_figure, (int, float)):
            raise ValueError("noise_figure must be a number")
        if not isinstance(rf_gain, (int, float)):
            raise ValueError("rf_gain must be a number")
        if load_resistor <= 0:
            raise ValueError("load_resistor must be positive")
        if not isinstance(baseband_gain, (int, float)):
            raise ValueError("baseband_gain must be a number")
        if bb_type not in VALID_BB_TYPES:
            raise ValueError(
                f"Invalid baseband type '{bb_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_BB_TYPES))}"
            )

        self.rf_prop: Dict[str, float] = {}
        self.bb_prop: Dict[str, float | str] = {}
        self.rxchannel_prop: Dict[str, object] = {}

        self.rf_prop["rf_gain"] = rf_gain
        self.rf_prop["noise_figure"] = noise_figure

        self.bb_prop["fs"] = fs
        self.bb_prop["load_resistor"] = load_resistor
        self.bb_prop["baseband_gain"] = baseband_gain
        self.bb_prop["bb_type"] = bb_type
        if bb_type == "complex":
            self.bb_prop["noise_bandwidth"] = fs
        else:
            self.bb_prop["noise_bandwidth"] = fs / 2

        self.validate_bb_prop(self.bb_prop)

        if channels is None:
            channels = [{"location": (0, 0, 0)}]

        self.rxchannel_prop = self.process_rxchannel_prop(channels)

    @staticmethod
    def _validate_array_lengths(
        arr1: NDArray,
        arr2: NDArray,
        name1: str,
        name2: str,
        channel_idx: Optional[int] = None,
    ) -> None:
        if len(arr1) != len(arr2):
            channel_info = f" for channel {channel_idx}" if channel_idx is not None else ""
            raise ValueError(
                f"Length mismatch{channel_info}: {name1} ({len(arr1)}) "
                f"and {name2} ({len(arr2)}) must have same length"
            )

    def validate_bb_prop(self, bb_prop: Dict) -> None:
        if bb_prop["bb_type"] not in VALID_BB_TYPES:
            raise ValueError(
                f"Invalid baseband type '{bb_prop['bb_type']}'. "
                f"Must be one of: {', '.join(sorted(VALID_BB_TYPES))}"
            )
        if bb_prop["fs"] <= 0:
            raise ValueError("Sampling rate (fs) must be positive")
        if bb_prop["load_resistor"] <= 0:
            raise ValueError("Load resistor must be positive")

    def process_rxchannel_prop(self, channels: List[Dict]) -> Dict:
        rxch_prop: Dict[str, object] = {}
        rxch_prop["size"] = len(channels)
        rxch_prop["locations"] = np.zeros((rxch_prop["size"], 3))
        rxch_prop["polarization"] = np.zeros((rxch_prop["size"], 3))
        rxch_prop["az_patterns"] = []
        rxch_prop["az_angles"] = []
        rxch_prop["el_patterns"] = []
        rxch_prop["el_angles"] = []
        rxch_prop["antenna_gains"] = np.zeros(rxch_prop["size"])

        for rx_idx, rx_element in enumerate(channels):
            rxch_prop["locations"][rx_idx, :] = np.array(rx_element.get("location"))
            rxch_prop["polarization"][rx_idx, :] = np.array(
                rx_element.get("polarization", DEFAULT_POLARIZATION)
            )

            az_angle = np.array(rx_element.get("azimuth_angle", DEFAULT_AZIMUTH_RANGE))
            az_pattern = np.array(rx_element.get("azimuth_pattern", DEFAULT_PATTERN_DB))
            self._validate_array_lengths(
                az_angle, az_pattern, "azimuth_angle", "azimuth_pattern", rx_idx
            )
            rxch_prop["antenna_gains"][rx_idx] = np.max(az_pattern)
            az_pattern = az_pattern - rxch_prop["antenna_gains"][rx_idx]
            rxch_prop["az_angles"].append(az_angle)
            rxch_prop["az_patterns"].append(az_pattern)

            el_angle = np.array(
                rx_element.get("elevation_angle", DEFAULT_ELEVATION_RANGE)
            )
            el_pattern = np.array(
                rx_element.get("elevation_pattern", DEFAULT_PATTERN_DB)
            )
            self._validate_array_lengths(
                el_angle, el_pattern, "elevation_angle", "elevation_pattern", rx_idx
            )
            el_pattern = el_pattern - np.max(el_pattern)
            rxch_prop["el_angles"].append(el_angle)
            rxch_prop["el_patterns"].append(el_pattern)

        return rxch_prop

    @property
    def sampling_rate(self) -> float:
        return self.bb_prop["fs"]

    @property
    def noise_bandwidth(self) -> float:
        return self.bb_prop["noise_bandwidth"]

    @property
    def num_channels(self) -> int:
        return self.rxchannel_prop["size"]

    @property
    def channel_locations(self) -> NDArray:
        return self.rxchannel_prop["locations"]

    def get_channel_info(self, channel_idx: int) -> Dict[str, NDArray]:
        if not 0 <= channel_idx < self.num_channels:
            raise IndexError(
                f"Channel index {channel_idx} out of range [0, {self.num_channels-1}]"
            )

        return {
            "location": self.rxchannel_prop["locations"][channel_idx],
            "polarization": self.rxchannel_prop["polarization"][channel_idx],
            "antenna_gain": self.rxchannel_prop["antenna_gains"][channel_idx],
            "azimuth_angles": self.rxchannel_prop["az_angles"][channel_idx],
            "azimuth_pattern": self.rxchannel_prop["az_patterns"][channel_idx],
            "elevation_angles": self.rxchannel_prop["el_angles"][channel_idx],
            "elevation_pattern": self.rxchannel_prop["el_patterns"][channel_idx],
        }

    def __str__(self) -> str:
        return (
            f"Receiver(channels={self.num_channels}, "
            f"fs={self.sampling_rate/1e6:.1f} MHz, "
            f"noise_figure={self.rf_prop['noise_figure']} dB, "
            f"bb_type={self.bb_prop['bb_type']})"
        )

    def __repr__(self) -> str:
        return (
            f"Receiver(fs={self.sampling_rate}, "
            f"noise_figure={self.rf_prop['noise_figure']}, "
            f"rf_gain={self.rf_prop['rf_gain']}, "
            f"load_resistor={self.bb_prop['load_resistor']}, "
            f"baseband_gain={self.bb_prop['baseband_gain']}, "
            f"bb_type='{self.bb_prop['bb_type']}', "
            f"channels={self.num_channels})"
        )
