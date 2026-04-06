"""Independent whitebox implementations for RadarSimPy analysis utilities."""

from typing import Union
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, erfcinv, gammainc, gammaincinv, iv
from scipy.stats import distributions


def marcumq(a: float, x: float, m: int = 1) -> float:
    """Calculate the generalized Marcum Q function."""

    return 1 - distributions.ncx2.cdf(df=m * 2, nc=a**2, x=x**2)


def log_factorial(n: Union[int, NDArray]) -> Union[float, NDArray]:
    """Compute ``log(n!)`` directly to avoid factorial overflow."""

    if np.isscalar(n):
        return np.sum(np.log(np.arange(1, n + 1)))

    val = np.zeros_like(n, dtype=float)
    for idx, n_item in enumerate(n):
        val[idx] = np.sum(np.log(np.arange(1, n_item + 1)))

    return val


def threshold(pfa: float, npulses: int) -> float:
    """Return the non-coherent integration threshold ratio."""

    return gammaincinv(npulses, 1 - pfa)


def pd_swerling0(npulses: int, snr: Union[float, NDArray], thred: float) -> float:
    """Calculate probability of detection for the Swerling 0 model."""

    if npulses <= 50:
        if np.isscalar(snr):
            sum_array = np.arange(2, npulses + 1)

            warnings.filterwarnings("ignore", category=RuntimeWarning)
            var_1 = np.exp(-(thred + npulses * snr)) * np.sum(
                (thred / (npulses * snr)) ** ((sum_array - 1) / 2)
                * iv(sum_array - 1, 2 * np.sqrt(npulses * snr * thred))
            )
            warnings.filterwarnings("default", category=RuntimeWarning)

            if np.isnan(var_1):
                var_1 = 0
        else:
            snr_len = np.size(snr)
            sum_array = np.arange(2, npulses + 1)
            sum_array = np.repeat(sum_array[np.newaxis, :], snr_len, axis=0)
            snr_mat = np.repeat(snr[:, np.newaxis], np.shape(sum_array)[1], axis=1)

            warnings.filterwarnings("ignore", category=RuntimeWarning)
            var_1 = np.exp(-(thred + npulses * snr)) * np.sum(
                (thred / (npulses * snr_mat)) ** ((sum_array - 1) / 2)
                * iv(sum_array - 1, 2 * np.sqrt(npulses * snr_mat * thred)),
                axis=1,
            )
            warnings.filterwarnings("default", category=RuntimeWarning)

            var_1[np.isnan(var_1)] = 0

        return marcumq(np.sqrt(2 * npulses * snr), np.sqrt(2 * thred)) + var_1

    temp_1 = 2 * snr + 1
    omegabar = np.sqrt(npulses * temp_1)
    c3 = -(snr + 1 / 3) / (np.sqrt(npulses) * temp_1**1.5)
    c4 = (snr + 0.25) / (npulses * temp_1**2.0)
    c6 = c3 * c3 / 2
    v_var = (thred - npulses * (1 + snr)) / omegabar
    v_sqr = v_var**2
    val1 = np.exp(-v_sqr / 2) / np.sqrt(2 * np.pi)
    val2 = (
        c3 * (v_sqr - 1)
        + c4 * v_var * (3 - v_sqr)
        - c6 * v_var * (v_var**4 - 10 * v_sqr + 15)
    )
    q = 0.5 * erfc(v_var / np.sqrt(2))
    return q - val1 * val2


def pd_swerling1(npulses: int, snr: float, thred: float) -> float:
    """Calculate probability of detection for the Swerling 1 model."""

    if npulses == 1:
        return np.exp(-thred / (1 + snr))

    temp_sw1 = 1 + 1 / (npulses * snr)
    igf1 = gammainc(npulses - 1, thred)
    igf2 = gammainc(npulses - 1, thred / temp_sw1)
    return (
        1
        - igf1
        + (temp_sw1 ** (npulses - 1)) * igf2 * np.exp(-thred / (1 + npulses * snr))
    )


def pd_swerling2(npulses: int, snr: float, thred: float) -> float:
    """Calculate probability of detection for the Swerling 2 model."""

    return 1 - gammainc(npulses, (thred / (1 + snr)))


def pd_swerling3(npulses: int, snr: float, thred: float) -> float:
    """Calculate probability of detection for the Swerling 3 model."""

    temp_1 = thred / (1 + 0.5 * npulses * snr)
    ko = (
        np.exp(-temp_1)
        * (1 + 2 / (npulses * snr)) ** (npulses - 2)
        * (1 + temp_1 - 2 * (npulses - 2) / (npulses * snr))
    )
    if npulses <= 2:
        return ko

    var_1 = np.exp(
        (npulses - 1) * np.log(thred) - thred - log_factorial(npulses - 2.0)
    ) / (1 + 0.5 * npulses * snr)

    pd = (
        var_1
        + 1
        - gammainc(npulses - 1, thred)
        + ko * gammainc(npulses - 1, thred / (1 + 2 / (npulses * snr)))
    )

    return pd


def pd_swerling4(npulses: int, snr: float, thred: float) -> float:
    """Calculate probability of detection for the Swerling 4 model."""

    beta = 1 + snr / 2
    if npulses >= 50:
        omegabar = np.sqrt(npulses * (2 * beta**2 - 1))
        c3 = (2 * beta**3 - 1) / (3 * (2 * beta**2 - 1) * omegabar)
        c4 = (2 * beta**4 - 1) / (4 * npulses * (2 * beta**2 - 1) ** 2)
        c6 = c3**2 / 2
        v_var = (thred - npulses * (1 + snr)) / omegabar
        v_sqr = v_var**2
        val1 = np.exp(-v_sqr / 2) / np.sqrt(2 * np.pi)
        val2 = (
            c3 * (v_sqr - 1)
            + c4 * v_var * (3 - v_sqr)
            - c6 * v_var * (v_var**4 - 10 * v_sqr + 15)
        )
        return 0.5 * erfc(v_var / np.sqrt(2)) - val1 * val2

    gamma0 = gammainc(npulses, thred / beta)
    a1 = (thred / beta) ** npulses / (
        np.exp(log_factorial(npulses)) * np.exp(thred / beta)
    )
    sum_var = gamma0
    for idx_1 in range(1, npulses + 1, 1):
        if idx_1 == 1:
            ai = a1
        else:
            ai = (thred / beta) * a1 / (npulses + idx_1 - 1)
        a1 = ai
        gammai = gamma0 - ai
        gamma0 = gammai

        temp_sw4 = np.sum(np.log(npulses + 1 - np.arange(1, idx_1 + 1)))

        try:
            term = (snr / 2) ** idx_1 * gammai * np.exp(
                temp_sw4 - log_factorial(idx_1)
            )
        except OverflowError:
            term = 0

        sum_var = sum_var + term
    return 1 - sum_var / beta**npulses


def roc_pd(
    pfa: Union[float, NDArray],
    snr: Union[float, NDArray],
    npulses: int = 1,
    stype: str = "Coherent",
) -> Union[float, NDArray, None]:
    """Calculate probability of detection in receiver operating characteristic."""

    snr_db = snr
    snr = 10.0 ** (snr_db / 10.0)

    size_pfa = np.size(pfa)
    size_snr = np.size(snr)

    pd = np.zeros((size_pfa, size_snr))

    it_pfa = np.nditer(pfa, flags=["f_index"])
    while not it_pfa.finished:
        thred = threshold(it_pfa[0], npulses)

        if stype == "Swerling 1":
            pd[it_pfa.index, :] = pd_swerling1(npulses, snr, thred)
        elif stype == "Swerling 2":
            pd[it_pfa.index, :] = pd_swerling2(npulses, snr, thred)
        elif stype == "Swerling 3":
            pd[it_pfa.index, :] = pd_swerling3(npulses, snr, thred)
        elif stype == "Swerling 4":
            pd[it_pfa.index, :] = pd_swerling4(npulses, snr, thred)
        elif stype in ("Swerling 5", "Swerling 0"):
            pd[it_pfa.index, :] = pd_swerling0(npulses, snr, thred)
        elif stype == "Coherent":
            snr = snr * npulses
            pd[it_pfa.index, :] = erfc(erfcinv(2 * it_pfa[0]) - np.sqrt(snr)) / 2
        elif stype == "Real":
            snr = snr * npulses / 2
            pd[it_pfa.index, :] = erfc(erfcinv(2 * it_pfa[0]) - np.sqrt(snr)) / 2
        else:
            return None

        it_pfa.iternext()

    if size_pfa == 1 and size_snr == 1:
        return pd[0, 0]
    if size_pfa == 1 and size_snr > 1:
        return pd[0, :]
    if size_pfa > 1 and size_snr == 1:
        return pd[:, 0]
    return pd


def roc_snr(
    pfa: Union[float, NDArray],
    pd: Union[float, NDArray],
    npulses: int = 1,
    stype: str = "Coherent",
) -> Union[float, NDArray, None]:
    """Calculate the minimum SNR for a desired Pd/Pfa pair."""

    def fun(pfa, pd, snr):
        return roc_pd(pfa, snr, npulses, stype) - pd

    max_iter = 1000
    snra = 40
    snrb = -20

    if stype == "Coherent" or stype == "Real":
        snrb = -40

    size_pd = np.size(pd)
    size_pfa = np.size(pfa)
    snr = np.zeros((size_pfa, size_pd))

    it_pfa = np.nditer(pfa, flags=["f_index"])
    while not it_pfa.finished:
        it_pd = np.nditer(pd, flags=["f_index"])
        while not it_pd.finished:
            if fun(it_pfa[0], it_pd[0], snra) * fun(it_pfa[0], it_pd[0], snrb) >= 0:
                return None
            a_n = snra
            b_n = snrb
            for _ in range(1, max_iter + 1):
                m_n = a_n - fun(it_pfa[0], it_pd[0], a_n) * (b_n - a_n) / (
                    fun(it_pfa[0], it_pd[0], b_n) - fun(it_pfa[0], it_pd[0], a_n)
                )
                f_m_n = fun(it_pfa[0], it_pd[0], m_n)
                if f_m_n == 0:
                    snr[it_pfa.index, it_pd.index] = m_n
                    break
                if np.abs(f_m_n) < 0.00001:
                    snr[it_pfa.index, it_pd.index] = m_n
                    break
                if fun(it_pfa[0], it_pd[0], a_n) * f_m_n < 0:
                    b_n = m_n
                elif fun(it_pfa[0], it_pd[0], b_n) * f_m_n < 0:
                    a_n = m_n
                else:
                    snr[it_pfa.index, it_pd.index] = float("nan")
                    break

            snr[it_pfa.index, it_pd.index] = a_n - fun(
                it_pfa[0], it_pd[0], a_n
            ) * (b_n - a_n) / (fun(it_pfa[0], it_pd[0], b_n) - fun(it_pfa[0], it_pd[0], a_n))
            it_pd.iternext()
        it_pfa.iternext()

    if size_pfa == 1 and size_pd == 1:
        return snr[0, 0]
    if size_pfa == 1 and size_pd > 1:
        return snr[0, :]
    if size_pfa > 1 and size_pd == 1:
        return snr[:, 0]
    return snr


__all__ = [
    "marcumq",
    "log_factorial",
    "threshold",
    "pd_swerling0",
    "pd_swerling1",
    "pd_swerling2",
    "pd_swerling3",
    "pd_swerling4",
    "roc_pd",
    "roc_snr",
]
