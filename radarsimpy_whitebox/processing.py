"""Independent whitebox implementations for RadarSimPy signal processing."""

from warnings import warn
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import fft, linalg
from scipy.signal import convolve, find_peaks

from .tools import log_factorial


def range_fft(data: NDArray, rwin: Optional[NDArray] = None, n: Optional[int] = None) -> NDArray:
    """Calculate the range profile matrix."""

    shape = np.shape(data)

    if rwin is None:
        rwin = 1
    else:
        rwin = np.tile(rwin[np.newaxis, np.newaxis, ...], (shape[0], shape[1], 1))

    return fft.fft(data * rwin, n=n, axis=2)


def doppler_fft(data: NDArray, dwin: Optional[NDArray] = None, n: Optional[int] = None) -> NDArray:
    """Calculate the range-Doppler matrix."""

    shape = np.shape(data)

    if dwin is None:
        dwin = 1
    else:
        dwin = np.tile(dwin[np.newaxis, ..., np.newaxis], (shape[0], 1, shape[2]))

    return fft.fft(data * dwin, n=n, axis=1)


def range_doppler_fft(
    data: NDArray,
    rwin: Optional[NDArray] = None,
    dwin: Optional[NDArray] = None,
    rn: Optional[int] = None,
    dn: Optional[int] = None
) -> NDArray:
    """Run chained range then Doppler FFT processing."""

    return doppler_fft(range_fft(data, rwin=rwin, n=rn), dwin=dwin, n=dn)


def cfar_ca_1d(
    data: NDArray,
    guard: int,
    trailing: int,
    pfa: float = 1e-5,
    axis: int = 0,
    detector: str = "squarelaw",
    offset: Optional[float] = None
) -> NDArray:
    """1-D Cell Averaging CFAR."""

    if np.iscomplexobj(data):
        raise ValueError("Input data should not be complex.")

    data_shape = np.shape(data)
    cfar = np.zeros_like(data)

    if offset is None:
        if detector == "squarelaw":
            a = trailing * 2 * (pfa ** (-1 / (trailing * 2)) - 1)
        elif detector == "linear":
            a = np.sqrt(trailing * 2 * (pfa ** (-1 / (trailing * 2)) - 1))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    cfar_win = np.ones((guard + trailing) * 2 + 1)
    cfar_win[trailing : (trailing + guard * 2 + 1)] = 0
    cfar_win = cfar_win / np.sum(cfar_win)

    if axis == 0:
        if data.ndim == 1:
            cfar = a * convolve(data, cfar_win, mode="same")
        elif data.ndim == 2:
            for idx in range(0, data_shape[1]):
                cfar[:, idx] = a * convolve(data[:, idx], cfar_win, mode="same")
    elif axis == 1:
        for idx in range(0, data_shape[0]):
            cfar[idx, :] = a * convolve(data[idx, :], cfar_win, mode="same")

    return cfar


def cfar_ca_2d(
    data: NDArray,
    guard: Union[int, List[int]],
    trailing: Union[int, List[int]],
    pfa: float = 1e-5,
    detector: str = "squarelaw",
    offset: Optional[float] = None
) -> NDArray:
    """2-D Cell Averaging CFAR."""

    if np.iscomplexobj(data):
        raise ValueError("Input data should not be complex.")

    guard = np.array(guard)
    if guard.size == 1:
        guard = np.tile(guard, 2)
    trailing = np.array(trailing)
    if trailing.size == 1:
        trailing = np.tile(trailing, 2)

    if offset is None:
        tg_sum = trailing + guard
        t_num = (2 * tg_sum[0] + 1) * (2 * tg_sum[1] + 1)
        g_num = (2 * guard[0] + 1) * (2 * guard[1] + 1)

        if t_num == g_num:
            raise ValueError("No trailing bins!")

        if detector == "squarelaw":
            a = (t_num - g_num) * (pfa ** (-1 / (t_num - g_num)) - 1)
        elif detector == "linear":
            a = np.sqrt((t_num - g_num) * (pfa ** (-1 / (t_num - g_num)) - 1))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    cfar_win = np.ones(((guard + trailing) * 2 + 1))
    cfar_win[
        trailing[0] : (trailing[0] + guard[0] * 2 + 1),
        trailing[1] : (trailing[1] + guard[1] * 2 + 1),
    ] = 0
    cfar_win = cfar_win / np.sum(cfar_win)

    return a * convolve(data, cfar_win, mode="same")


def os_cfar_threshold(k: int, n: int, pfa: float) -> float:
    """Calculate the OS-CFAR threshold with the secant method."""

    def fun(k, n, t_os, pfa):
        return (
            log_factorial(n)
            - log_factorial(n - k)
            - np.sum(np.log(np.arange(n, n - k, -1) + t_os))
            - np.log(pfa)
        )

    max_iter = 10000
    t_max = 1e32
    t_min = 1

    for _ in range(0, max_iter):
        m_n = t_max - fun(k, n, t_max, pfa) * (t_min - t_max) / (
            fun(k, n, t_min, pfa) - fun(k, n, t_max, pfa)
        )
        f_m_n = fun(k, n, m_n, pfa)
        if f_m_n == 0:
            return m_n
        if np.abs(f_m_n) < 0.0001:
            return m_n

        if fun(k, n, t_max, pfa) * f_m_n < 0:
            t_min = m_n
        elif fun(k, n, t_min, pfa) * f_m_n < 0:
            t_max = m_n
        else:
            break

    return None


def cfar_os_1d(
    data: NDArray,
    guard: int,
    trailing: int,
    k: int,
    pfa: float = 1e-5,
    axis: int = 0,
    detector: str = "squarelaw",
    offset: Optional[float] = None
) -> NDArray:
    """1-D Ordered Statistic CFAR."""

    if np.iscomplexobj(data):
        raise ValueError("Input data should not be complex.")

    data_shape = np.shape(data)
    cfar = np.zeros_like(data)
    leading = trailing

    if offset is None:
        if detector == "squarelaw":
            a = os_cfar_threshold(k, trailing * 2, pfa)
        elif detector == "linear":
            a = np.sqrt(os_cfar_threshold(k, trailing * 2, pfa))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    if k < trailing or k > trailing * 2:
        warn(
            "``k`` is usuall chosen to satisfy ``N/2 < k < N "
            "(N = " + str(trailing * 2) + ")``. "
            "Typically, ``k`` is on the order of ``0.75N``"
        )

    if axis == 0:
        for idx in range(0, data_shape[0]):
            win_idx = np.mod(
                np.concatenate(
                    [
                        np.arange(idx - leading - guard, idx - guard, 1),
                        np.arange(idx + 1 + guard, idx + 1 + trailing + guard, 1),
                    ]
                ),
                data_shape[0],
            )
            if data.ndim == 1:
                samples = np.sort(data[win_idx.astype(int)])
                cfar[idx] = a * samples[k]
            elif data.ndim == 2:
                samples = np.sort(data[win_idx.astype(int), :], axis=0)
                cfar[idx, :] = a * samples[k, :]
    elif axis == 1:
        for idx in range(0, data_shape[1]):
            win_idx = np.mod(
                np.concatenate(
                    [
                        np.arange(idx - leading - guard, idx - guard, 1),
                        np.arange(idx + 1 + guard, idx + 1 + trailing + guard, 1),
                    ]
                ),
                data_shape[1],
            )
            samples = np.sort(data[:, win_idx.astype(int)], axis=1)
            cfar[:, idx] = a * samples[:, k]

    return cfar


def cfar_os_2d(
    data: NDArray,
    guard: Union[int, List[int]],
    trailing: Union[int, List[int]],
    k: int,
    pfa: float = 1e-5,
    detector: str = "squarelaw",
    offset: Optional[float] = None
) -> NDArray:
    """2-D Ordered Statistic CFAR."""

    if np.iscomplexobj(data):
        raise ValueError("Input data should not be complex.")

    data_shape = np.shape(data)
    cfar = np.zeros_like(data)

    guard = np.array(guard)
    if guard.size == 1:
        guard = np.tile(guard, 2)
    trailing = np.array(trailing)
    if trailing.size == 1:
        trailing = np.tile(trailing, 2)

    tg_sum = trailing + guard
    if offset is None:
        t_num = (2 * tg_sum[0] + 1) * (2 * tg_sum[1] + 1)
        g_num = (2 * guard[0] + 1) * (2 * guard[1] + 1)

        if t_num == g_num:
            raise ValueError("No trailing bins!")

        if detector == "squarelaw":
            a = os_cfar_threshold(k, t_num - g_num, pfa)
        elif detector == "linear":
            a = np.sqrt(os_cfar_threshold(k, t_num - g_num, pfa))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    if k < (t_num - g_num) / 2 or k > t_num - g_num:
        warn(
            "``k`` is usuall chosen to satisfy ``N/2 < k < N "
            "(N = " + str(t_num - g_num) + ")``. "
            "Typically, ``k`` is on the order of ``0.75N``"
        )

    cfar_win = np.ones((tg_sum * 2 + 1), dtype=bool)
    cfar_win[
        trailing[0] : (trailing[0] + guard[0] * 2 + 1),
        trailing[1] : (trailing[1] + guard[1] * 2 + 1),
    ] = False

    for idx_0 in range(0, data_shape[0]):
        for idx_1 in range(0, data_shape[1]):
            win_idx_0 = np.mod(
                np.arange(idx_0 - tg_sum[0], idx_0 + 1 + tg_sum[0], 1), data_shape[0]
            )
            win_idx_1 = np.mod(
                np.arange(idx_1 - tg_sum[1], idx_1 + 1 + tg_sum[1], 1), data_shape[1]
            )

            x, y = np.meshgrid(win_idx_0, win_idx_1, indexing="ij")
            sample_cube = data[x, y]
            samples = np.sort(sample_cube[cfar_win].flatten())
            cfar[idx_0, idx_1] = a * samples[k]

    return cfar


def doa_music(
    covmat: NDArray,
    nsig: int,
    spacing: float = 0.5,
    scanangles: Union[range, List[int], NDArray] = range(-90, 91)
) -> tuple[list, list, NDArray]:
    """Estimate signal arrival directions using MUSIC for a ULA."""

    n_array = np.shape(covmat)[0]
    array = np.linspace(0, (n_array - 1) * spacing, n_array)
    scanangles = np.array(scanangles)

    _, eig_vects = linalg.eigh(covmat)
    noise_subspace = eig_vects[:, :-nsig]

    array_grid, angle_grid = np.meshgrid(array, np.radians(scanangles), indexing="ij")
    steering_vect = np.exp(1j * 2 * np.pi * array_grid * np.sin(angle_grid)) / np.sqrt(
        n_array
    )

    pseudo_spectrum = 1 / linalg.norm((noise_subspace.T.conj() @ steering_vect), axis=0)

    ps_db = 10 * np.log10(pseudo_spectrum / pseudo_spectrum.min())
    doa_idx, _ = find_peaks(ps_db)
    doa_idx = doa_idx[np.argsort(ps_db[doa_idx])[-nsig:]]

    return scanangles[doa_idx], doa_idx, ps_db


def doa_root_music(covmat: NDArray, nsig: int, spacing: float = 0.5) -> list:
    """Estimate arrival directions using root-MUSIC for a ULA."""

    n_covmat = np.shape(covmat)[0]

    _, eig_vects = linalg.eigh(covmat)
    noise_subspace = eig_vects[:, :-nsig]

    noise_mat = noise_subspace @ noise_subspace.T.conj()
    coeff = np.zeros((n_covmat - 1,), dtype=np.complex128)
    for i in range(1, n_covmat):
        coeff[i - 1] = np.trace(noise_mat, i)
    coeff = np.hstack((coeff[::-1], np.trace(noise_mat), coeff.conj()))

    roots = np.roots(coeff)
    mask = np.abs(roots) <= 1
    for _, i in enumerate(np.where(np.abs(roots) == 1)[0]):
        mask_idx = np.argsort(np.abs(roots - roots[i]))[1]
        mask[mask_idx] = False

    roots = roots[mask]
    sorted_indices = np.argsort(1.0 - np.abs(roots))
    sin_vals = np.angle(roots[sorted_indices[:nsig]]) / (2 * np.pi * spacing)

    return np.degrees(np.arcsin(sin_vals))


def doa_esprit(covmat: NDArray, nsig: int, spacing: float = 0.5) -> list:
    """Estimate arrival directions using ESPRIT for a ULA."""

    _, eig_vects = linalg.eigh(covmat)
    signal_subspace = eig_vects[:, -nsig:]

    phi = linalg.pinv(signal_subspace[0:-1]) @ signal_subspace[1:]
    eigs = linalg.eigvals(phi)
    return np.degrees(np.arcsin(np.angle(eigs) / np.pi / (spacing / 0.5)))


def doa_iaa(
    beam_vect: NDArray,
    steering_vect: NDArray,
    num_it: int = 15,
    p_init: Optional[NDArray] = None
) -> NDArray:
    """IAA-APES spectrum estimation."""

    num_grid = np.shape(steering_vect)[1]

    if p_init is None:
        spectrum_k = np.zeros(num_grid, dtype=complex)
        for ik in range(0, num_grid):
            a_vect = steering_vect[:, ik]
            a_vect = np.conj(a_vect[np.newaxis, :])
            spectrum_k[ik] = (
                1
                / ((a_vect @ a_vect.conj().T) ** 2)
                * np.mean(np.abs(a_vect @ beam_vect) ** 2)
            ).item()
    else:
        spectrum_k = p_init

    for _ in range(0, num_it - 1):
        p_diag = np.diag(spectrum_k.flatten())
        r_mat = steering_vect @ p_diag @ steering_vect.conj().T
        r_mat_inv = np.linalg.inv(r_mat)
        for ik in range(0, num_grid):
            a_vect = steering_vect[:, ik]
            a_vect = np.conj(a_vect[np.newaxis, :])
            spec = (
                a_vect @ r_mat_inv @ beam_vect / (a_vect @ r_mat_inv @ a_vect.conj().T)
            )
            spectrum_k[ik] = np.mean(np.abs(spec) ** 2)
    return 10 * np.log10(np.real(spectrum_k))


def doa_bartlett(
    covmat: NDArray,
    spacing: float = 0.5,
    scanangles: Union[range, List[int], NDArray] = range(-90, 91)
) -> NDArray:
    """Bartlett beamforming for a ULA."""

    n_array = np.shape(covmat)[0]
    array = np.linspace(0, (n_array - 1) * spacing, n_array)
    scanangles = np.array(scanangles)

    array_grid, angle_grid = np.meshgrid(array, np.radians(scanangles), indexing="ij")
    steering_vect = np.exp(1j * 2 * np.pi * array_grid * np.sin(angle_grid)) / np.sqrt(
        n_array
    )

    ps = np.sum(steering_vect.conj() * (covmat @ steering_vect), axis=0).real
    return 10 * np.log10(ps)


def doa_capon(
    covmat: NDArray,
    spacing: float = 0.5,
    scanangles: Union[range, List[int], NDArray] = range(-90, 91)
) -> NDArray:
    """Capon (MVDR) beamforming for a ULA."""

    n_array = np.shape(covmat)[0]
    array = np.linspace(0, (n_array - 1) * spacing, n_array)
    scanangles = np.array(scanangles)

    array_grid, angle_grid = np.meshgrid(array, np.radians(scanangles), indexing="ij")
    steering_vect = np.exp(1j * 2 * np.pi * array_grid * np.sin(angle_grid)) / np.sqrt(
        n_array
    )

    covmat = covmat + np.eye(n_array) * 0.000000001
    inv_covmat = linalg.pinv(covmat)

    ps = np.zeros(scanangles.shape)
    for idx, _ in enumerate(scanangles):
        s_vect = steering_vect[:, idx]
        weight = inv_covmat @ s_vect / (s_vect.T.conj() @ inv_covmat @ s_vect)
        ps[idx] = np.abs(weight.T.conj() @ covmat @ weight)

    return 10 * np.log10(ps)


__all__ = [
    "range_fft",
    "doppler_fft",
    "range_doppler_fft",
    "cfar_ca_1d",
    "cfar_ca_2d",
    "os_cfar_threshold",
    "cfar_os_1d",
    "cfar_os_2d",
    "doa_music",
    "doa_root_music",
    "doa_esprit",
    "doa_iaa",
    "doa_bartlett",
    "doa_capon",
]
