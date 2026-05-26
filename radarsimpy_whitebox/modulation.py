"""TDM/BPM pulse modulation helpers for transmitter channel dictionaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _as_tx_values(value, tx_channels: int, name: str) -> NDArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(tx_channels, float(array), dtype=float)
    if array.shape == (tx_channels,):
        return array.astype(float)
    raise ValueError(f"{name} must be a scalar or a length-{tx_channels} array")


def _normalize_order(
    order: Optional[Sequence[int]], tx_channels: int
) -> NDArray[np.int_]:
    if order is None:
        return np.arange(tx_channels, dtype=int)

    order_array = np.asarray(order)
    if order_array.ndim != 1 or order_array.size == 0:
        raise ValueError("order must be a non-empty 1D sequence")
    if not np.issubdtype(order_array.dtype, np.integer):
        raise ValueError("order must contain integer transmitter indices")
    if np.any(order_array < 0) or np.any(order_array >= tx_channels):
        raise ValueError("order contains transmitter indices outside the channel range")
    return order_array.astype(int)


def _repeat_rows(matrix: NDArray, pulses: int) -> NDArray:
    repeats = int(np.ceil(pulses / matrix.shape[0]))
    return np.tile(matrix, (repeats, 1))[:pulses]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _hadamard(order: int) -> NDArray:
    if not _is_power_of_two(order):
        raise ValueError(
            "Default BPM code generation requires tx_channels to be a power of two; "
            "provide code_matrix for a custom binary code"
        )

    matrix = np.array([[1.0]])
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def tdm_code(
    tx_channels: int,
    pulses: int,
    order: Optional[Sequence[int]] = None,
    active_amp=1.0,
    active_phase=0.0,
) -> Dict[str, NDArray]:
    """Build a time-division multiplexing mask for `pulse_amp`/`pulse_phs`.

    Each pulse enables one transmitter from `order` and disables the rest by
    setting their pulse amplitude to zero.
    """

    tx_channels = _validate_positive_int(tx_channels, "tx_channels")
    pulses = _validate_positive_int(pulses, "pulses")
    order_array = _normalize_order(order, tx_channels)
    active_amp_values = _as_tx_values(active_amp, tx_channels, "active_amp")
    active_phase_values = _as_tx_values(active_phase, tx_channels, "active_phase")

    pulse_amp = np.zeros((tx_channels, pulses), dtype=float)
    pulse_phs = np.zeros((tx_channels, pulses), dtype=float)

    for pulse_index in range(pulses):
        tx_index = int(order_array[pulse_index % order_array.size])
        pulse_amp[tx_index, pulse_index] = active_amp_values[tx_index]
        pulse_phs[tx_index, pulse_index] = active_phase_values[tx_index]

    return {"pulse_amp": pulse_amp, "pulse_phs": pulse_phs}


def bpm_code(
    tx_channels: int,
    pulses: int,
    code_matrix: Optional[Iterable[Iterable[float]]] = None,
) -> Dict[str, NDArray]:
    """Build a binary phase modulation code for `pulse_amp`/`pulse_phs`.

    `code_matrix` rows are repeated across pulses and columns map to TX
    channels. Values must be `+1` or `-1`; they are converted to phase values
    of `0` and `180` degrees with all transmitters active.
    """

    tx_channels = _validate_positive_int(tx_channels, "tx_channels")
    pulses = _validate_positive_int(pulses, "pulses")

    if code_matrix is None:
        base_code = _hadamard(tx_channels)
    else:
        base_code = np.asarray(code_matrix, dtype=float)
        if base_code.ndim != 2:
            raise ValueError("code_matrix must be a 2D array")
        if base_code.shape[1] != tx_channels:
            raise ValueError("code_matrix columns must match tx_channels")
        if base_code.shape[0] < 1:
            raise ValueError("code_matrix must contain at least one code row")
        if not np.all(np.isin(base_code, [-1.0, 1.0])):
            raise ValueError("code_matrix values must be +1 or -1")

    repeated = _repeat_rows(base_code, pulses)
    pulse_phs = np.where(repeated.T < 0.0, 180.0, 0.0)
    pulse_amp = np.ones((tx_channels, pulses), dtype=float)
    return {"pulse_amp": pulse_amp, "pulse_phs": pulse_phs}


def apply_pulse_modulation(
    channels: Sequence[Dict],
    pulse_amp: NDArray,
    pulse_phs: NDArray,
    *,
    copy: bool = True,
) -> List[Dict]:
    """Attach per-channel pulse modulation arrays to TX channel dictionaries."""

    pulse_amp = np.asarray(pulse_amp, dtype=float)
    pulse_phs = np.asarray(pulse_phs, dtype=float)
    if pulse_amp.shape != pulse_phs.shape:
        raise ValueError("pulse_amp and pulse_phs must have the same shape")
    if pulse_amp.ndim != 2:
        raise ValueError("pulse_amp and pulse_phs must be shaped [tx_channels, pulses]")
    if len(channels) != pulse_amp.shape[0]:
        raise ValueError("channel count must match pulse modulation rows")

    output = deepcopy(list(channels)) if copy else list(channels)
    for tx_index, channel in enumerate(output):
        channel["pulse_amp"] = pulse_amp[tx_index].copy()
        channel["pulse_phs"] = pulse_phs[tx_index].copy()
    return output


def tdm_channels(
    channels: Sequence[Dict],
    pulses: int,
    order: Optional[Sequence[int]] = None,
    active_amp=1.0,
    active_phase=0.0,
    *,
    copy: bool = True,
) -> List[Dict]:
    """Return channel dictionaries with TDM pulse modulation attached."""

    code = tdm_code(
        len(channels),
        pulses,
        order=order,
        active_amp=active_amp,
        active_phase=active_phase,
    )
    return apply_pulse_modulation(channels, copy=copy, **code)


def bpm_channels(
    channels: Sequence[Dict],
    pulses: int,
    code_matrix: Optional[Iterable[Iterable[float]]] = None,
    *,
    copy: bool = True,
) -> List[Dict]:
    """Return channel dictionaries with BPM pulse modulation attached."""

    code = bpm_code(len(channels), pulses, code_matrix=code_matrix)
    return apply_pulse_modulation(channels, copy=copy, **code)


tdm = tdm_code
bpm = bpm_code


__all__ = [
    "apply_pulse_modulation",
    "bpm",
    "bpm_channels",
    "bpm_code",
    "tdm",
    "tdm_channels",
    "tdm_code",
]
