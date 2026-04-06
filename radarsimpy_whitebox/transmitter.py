"""Whitebox transmitter model for RadarSimPy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

DEFAULT_POLARIZATION = [0, 0, 1]
DEFAULT_AZIMUTH_RANGE = [-90, 90]
DEFAULT_ELEVATION_RANGE = [-90, 90]
DEFAULT_PATTERN_DB = [0, 0]
DEFAULT_GRID_SIZE = 1.0
DEGREES_TO_RADIANS = np.pi / 180


class Transmitter:
    """Radar transmitter configuration model."""

    def __init__(
        self,
        f: Union[float, List, NDArray],
        t: Union[float, List, NDArray],
        tx_power: float = 0,
        pulses: int = 1,
        prp: Optional[Union[float, List, NDArray]] = None,
        f_offset: Optional[Union[float, List, NDArray]] = None,
        pn_f: Optional[NDArray] = None,
        pn_power: Optional[NDArray] = None,
        channels: Optional[List[Dict]] = None,
    ):
        if pulses < 1:
            raise ValueError("Number of pulses must be at least 1")
        if not isinstance(tx_power, (int, float)):
            raise ValueError("tx_power must be a number")

        self.rf_prop: Dict[str, Any] = {}
        self.waveform_prop: Dict[str, Any] = {}
        self.txchannel_prop: Dict[str, Any] = {}

        self.rf_prop["tx_power"] = tx_power
        self.rf_prop["pn_f"] = pn_f
        self.rf_prop["pn_power"] = pn_power
        self.validate_rf_prop(self.rf_prop)

        f = self._ensure_array(f)

        if isinstance(t, (list, tuple, np.ndarray)):
            t_array = np.array(t)
            t_array = t_array - t_array[0]
        else:
            t_array = np.array([0, t])

        self.waveform_prop["f"] = f
        self.waveform_prop["t"] = t_array
        self.waveform_prop["bandwidth"] = np.max(f) - np.min(f)
        self.waveform_prop["pulse_length"] = t_array[-1]
        self.waveform_prop["pulses"] = pulses

        if f_offset is None:
            f_offset = np.zeros(pulses)
        else:
            if isinstance(f_offset, (list, tuple, np.ndarray)):
                f_offset = np.array(f_offset)
            else:
                f_offset = f_offset + np.zeros(pulses)

            if len(f_offset) != pulses:
                raise ValueError(
                    f"f_offset length ({len(f_offset)}) must match pulses ({pulses})"
                )
        self.waveform_prop["f_offset"] = f_offset

        if prp is None:
            prp_array = self.waveform_prop["pulse_length"] + np.zeros(pulses)
        else:
            if isinstance(prp, (list, tuple, np.ndarray)):
                prp_array = np.array(prp)
            else:
                prp_array = prp + np.zeros(pulses)
        self.waveform_prop["prp"] = prp_array
        self.waveform_prop["pulse_start_time"] = np.cumsum(prp_array) - prp_array[0]

        self.validate_waveform_prop(self.waveform_prop)

        if channels is None:
            channels = [{"location": (0, 0, 0)}]

        self.txchannel_prop = self.process_txchannel_prop(channels)

    @staticmethod
    def _validate_array_lengths(
        arr1: NDArray, arr2: NDArray, name1: str, name2: str
    ) -> None:
        if len(arr1) != len(arr2):
            raise ValueError(
                f"Lengths of `{name1}` ({len(arr1)}) and `{name2}` ({len(arr2)}) "
                "should be the same"
            )

    @staticmethod
    def _ensure_array(
        value: Union[float, List, NDArray], default_value: Optional[float] = None
    ) -> NDArray:
        if isinstance(value, (list, tuple, np.ndarray)):
            return np.array(value)
        if default_value is not None:
            return np.array([default_value, value])
        return np.array([value, value])

    def validate_rf_prop(self, rf_prop: Dict) -> None:
        pn_f_present = rf_prop["pn_f"] is not None
        pn_power_present = rf_prop["pn_power"] is not None

        if pn_f_present != pn_power_present:
            raise ValueError(
                "Both `pn_f` and `pn_power` must be provided together or both None"
            )

        if pn_f_present and pn_power_present:
            if len(rf_prop["pn_f"]) != len(rf_prop["pn_power"]):
                raise ValueError("Lengths of `pn_f` and `pn_power` should be the same")

    def validate_waveform_prop(self, waveform_prop: Dict) -> None:
        self._validate_array_lengths(waveform_prop["f"], waveform_prop["t"], "f", "t")

        if len(waveform_prop["f_offset"]) != waveform_prop["pulses"]:
            raise ValueError(
                f"f_offset length ({len(waveform_prop['f_offset'])}) must match "
                f"pulses ({waveform_prop['pulses']})"
            )

        if len(waveform_prop["prp"]) != waveform_prop["pulses"]:
            raise ValueError(
                f"prp length ({len(waveform_prop['prp'])}) must match "
                f"pulses ({waveform_prop['pulses']})"
            )

        if np.min(waveform_prop["prp"]) < waveform_prop["pulse_length"]:
            raise ValueError(
                f"All PRP values ({np.min(waveform_prop['prp']):.2e} s) must be >= "
                f"pulse_length ({waveform_prop['pulse_length']:.2e} s)"
            )

    def process_waveform_modulation(
        self, mod_t: Optional[NDArray], amp: Optional[NDArray], phs: Optional[NDArray]
    ) -> Dict:
        if phs is not None and amp is None:
            amp = np.ones_like(phs)
        elif phs is None and amp is not None:
            phs = np.zeros_like(amp)

        if mod_t is None or amp is None or phs is None:
            return {"enabled": False, "var": None, "t": None}

        amp = self._ensure_array(amp)
        phs = self._ensure_array(phs)
        mod_t = self._ensure_array(mod_t, 0.0)

        self._validate_array_lengths(amp, phs, "amp", "phs")
        self._validate_array_lengths(mod_t, amp, "mod_t", "amp")

        mod_var = amp * np.exp(1j * phs * DEGREES_TO_RADIANS)
        return {"enabled": True, "var": mod_var, "t": mod_t}

    def process_pulse_modulation(
        self, pulse_amp: NDArray, pulse_phs: NDArray
    ) -> NDArray:
        if len(pulse_amp) != self.waveform_prop["pulses"]:
            raise ValueError("Lengths of `pulse_amp` and `pulses` should be the same")
        if len(pulse_phs) != self.waveform_prop["pulses"]:
            raise ValueError("Length of `pulse_phs` and `pulses` should be the same")

        return np.array(pulse_amp) * np.exp(
            1j * (np.array(pulse_phs) * DEGREES_TO_RADIANS)
        )

    def process_txchannel_prop(self, channels: List[Dict]) -> Dict:
        txch_prop: Dict[str, Any] = {}
        txch_prop["size"] = len(channels)
        txch_prop["delay"] = np.zeros(txch_prop["size"])
        txch_prop["grid"] = np.zeros(txch_prop["size"])
        txch_prop["locations"] = np.zeros((txch_prop["size"], 3))
        txch_prop["polarization"] = np.zeros((txch_prop["size"], 3))
        txch_prop["waveform_mod"] = []
        txch_prop["pulse_mod"] = np.ones(
            (txch_prop["size"], self.waveform_prop["pulses"]), dtype=complex
        )
        txch_prop["az_patterns"] = []
        txch_prop["az_angles"] = []
        txch_prop["el_patterns"] = []
        txch_prop["el_angles"] = []
        txch_prop["antenna_gains"] = np.zeros(txch_prop["size"])

        for tx_idx, tx_element in enumerate(channels):
            txch_prop["delay"][tx_idx] = tx_element.get("delay", 0)
            txch_prop["grid"][tx_idx] = tx_element.get("grid", DEFAULT_GRID_SIZE)
            txch_prop["locations"][tx_idx, :] = np.array(tx_element.get("location"))
            txch_prop["polarization"][tx_idx, :] = np.array(
                tx_element.get("polarization", DEFAULT_POLARIZATION)
            )

            txch_prop["waveform_mod"].append(
                self.process_waveform_modulation(
                    tx_element.get("mod_t", None),
                    tx_element.get("amp", None),
                    tx_element.get("phs", None),
                )
            )

            txch_prop["pulse_mod"][tx_idx, :] = self.process_pulse_modulation(
                tx_element.get("pulse_amp", np.ones(self.waveform_prop["pulses"])),
                tx_element.get("pulse_phs", np.zeros(self.waveform_prop["pulses"])),
            )

            az_angle = np.array(tx_element.get("azimuth_angle", DEFAULT_AZIMUTH_RANGE))
            az_pattern = np.array(tx_element.get("azimuth_pattern", DEFAULT_PATTERN_DB))
            if len(az_angle) != len(az_pattern):
                raise ValueError(
                    f"Length mismatch for channel {tx_idx}: azimuth_angle "
                    f"({len(az_angle)}) and azimuth_pattern ({len(az_pattern)}) "
                    "must have same length"
                )

            txch_prop["antenna_gains"][tx_idx] = np.max(az_pattern)
            az_pattern = az_pattern - txch_prop["antenna_gains"][tx_idx]
            txch_prop["az_angles"].append(az_angle)
            txch_prop["az_patterns"].append(az_pattern)

            el_angle = np.array(
                tx_element.get("elevation_angle", DEFAULT_ELEVATION_RANGE)
            )
            el_pattern = np.array(
                tx_element.get("elevation_pattern", DEFAULT_PATTERN_DB)
            )
            if len(el_angle) != len(el_pattern):
                raise ValueError(
                    f"Length mismatch for channel {tx_idx}: elevation_angle "
                    f"({len(el_angle)}) and elevation_pattern ({len(el_pattern)}) "
                    "must have same length"
                )
            el_pattern = el_pattern - np.max(el_pattern)
            txch_prop["el_angles"].append(el_angle)
            txch_prop["el_patterns"].append(el_pattern)

        return txch_prop

    @property
    def frequency(self) -> NDArray:
        return self.waveform_prop["f"]

    @property
    def bandwidth(self) -> float:
        return self.waveform_prop["bandwidth"]

    @property
    def pulse_length(self) -> float:
        return self.waveform_prop["pulse_length"]

    @property
    def num_pulses(self) -> int:
        return self.waveform_prop["pulses"]

    @property
    def num_channels(self) -> int:
        return self.txchannel_prop["size"]

    @property
    def channel_locations(self) -> NDArray:
        return self.txchannel_prop["locations"]

    def get_channel_info(self, channel_idx: int) -> Dict[str, Any]:
        if not 0 <= channel_idx < self.num_channels:
            raise IndexError(
                f"Channel index {channel_idx} out of range [0, {self.num_channels-1}]"
            )

        return {
            "location": self.txchannel_prop["locations"][channel_idx],
            "polarization": self.txchannel_prop["polarization"][channel_idx],
            "delay": self.txchannel_prop["delay"][channel_idx],
            "grid": self.txchannel_prop["grid"][channel_idx],
            "antenna_gain": self.txchannel_prop["antenna_gains"][channel_idx],
            "azimuth_angles": self.txchannel_prop["az_angles"][channel_idx],
            "azimuth_pattern": self.txchannel_prop["az_patterns"][channel_idx],
            "elevation_angles": self.txchannel_prop["el_angles"][channel_idx],
            "elevation_pattern": self.txchannel_prop["el_patterns"][channel_idx],
        }

    def __str__(self) -> str:
        return (
            f"Transmitter(channels={self.num_channels}, "
            f"pulses={self.num_pulses}, "
            f"bandwidth={self.bandwidth/1e9:.3f} GHz, "
            f"pulse_length={self.pulse_length*1e6:.1f} μs)"
        )

    def __repr__(self) -> str:
        return (
            f"Transmitter(f={self.frequency}, "
            f"t={self.waveform_prop['t']}, "
            f"tx_power={self.rf_prop['tx_power']}, "
            f"pulses={self.num_pulses}, "
            f"channels={self.num_channels})"
        )
