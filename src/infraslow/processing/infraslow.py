"""Infraslow (~0.02 Hz) oscillations of sigma-band power.

Sleep-spindle (sigma, ~12-15 Hz) power is not steady across NREM: it waxes and
wanes with an **infraslow rhythm** around 0.02 Hz (a ~50 s period), and spindles
cluster on the rising phase / peaks of that rhythm (Lecci et al., 2017;
Watson, 2018). This module quantifies that rhythm from the continuous EEG --
band-pass to sigma, take the Hilbert power envelope, down-sample it, and read
off its low-frequency spectrum (:func:`power_envelope` -> :func:`infraslow_spectrum`).

Everything here is pure NumPy/SciPy (no matplotlib, no YASA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit

from .signal import ButterFilter

# ``np.trapz`` is deprecated in NumPy 2.0 in favour of ``np.trapezoid``; use
# whichever the installed NumPy provides.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Sigma (spindle) band the ISO sigma-power series is integrated over (Hz).
# Matches the reference ``get_iso`` (Liu et al.), which uses 11-16 Hz for the
# multitaper sigma-power estimate -- deliberately wider than the detector's
# ``detection.spindles_detect`` ``freq_sp`` (12-15 Hz), as this is a power
# integration, not a detection band.
DEFAULT_SIGMA_BAND: Tuple[float, float] = (11.0, 16.0)
# Infraslow band the sigma-power oscillation is expected to peak in (Hz).
# Widened to 0.01-0.1 Hz (~10-100 s) around the canonical ~0.02 Hz (50 s)
# sigma-power rhythm, so peak-finding and band power stay anchored to it while
# capturing slower and faster infraslow components.
DEFAULT_INFRASLOW_BAND: Tuple[float, float] = (0.01, 0.1)
# (Hz). 1 Hz is far above the infraslow Nyquist and keeps the series compact.
DEFAULT_SF_ENV: float = 1.0
# Welch segment length (seconds) for the infraslow spectrum -- long enough to
# resolve the 0.01 Hz lower edge of DEFAULT_INFRASLOW_BAND (0.01 Hz resolution
# at 100 s).
DEFAULT_WINDOW_SEC: float = 100.0
# Band (Hz) the bi-Gaussian ISFS fit (:func:`fit_isfs`) uses to estimate the
# noise floor for its detection threshold and chromatogram baseline.
DEFAULT_BASELINE_BAND: Tuple[float, float] = (0.06, 0.1)


@dataclass
class InfraslowSpectrum:
    """Low-frequency power spectrum of a slow time series, with band metrics.

    Attributes:
        freqs, psd: The full one-sided spectrum (``psd`` in units^2/Hz).
        sf_env: Sampling rate of the analysed series (Hz).
        infraslow_band: The infraslow band ``(lo, hi)`` the metrics summarise.
        peak_freq, peak_power: Frequency (Hz) and PSD of the largest peak *within*
            ``band`` (``nan`` if the spectrum has no bin in band).
        full_peak_freq, full_peak_power: Frequency (Hz) and PSD of the largest
            peak over the *whole* spectrum (excluding DC). This is the reference
            ``get_iso`` peak metric (``freq_iso[np.argmax(iso_power)]``): if it
            falls inside ``band`` the infraslow rhythm dominates the slow series.
        band_power: Integrated PSD across ``band`` (units^2).
        total_power: Integrated PSD across the whole spectrum.
        rel_band_power: ``band_power / total_power`` -- fraction of the slow-series
            variance carried by the infraslow band (a scale-free peak strength).
    """

    freqs: np.ndarray
    psd: np.ndarray
    sf_env: float
    infraslow_band: Tuple[float, float]
    peak_freq: float
    peak_power: float
    full_peak_freq: float
    full_peak_power: float
    band_power: float
    total_power: float
    rel_band_power: float


def _to_1d(x, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {np.shape(x)}.")
    return arr


def _to_db(power: np.ndarray) -> np.ndarray:
    """Convert linear power to dB (``10*log10``), like the reference ``get_iso``.

    Non-positive bins (which would give ``-inf``) are floored to the smallest
    positive value in the series so the log stays finite without inventing a
    magnitude; this virtually never triggers on a Hilbert power envelope.
    """
    power = np.asarray(power, dtype=float)
    positive = power[power > 0]
    floor = positive.min() if positive.size else 1.0
    return 10.0 * np.log10(np.where(power > 0, power, floor))


def power_envelope(
    data,
    sf: float,
    *,
    band: Tuple[float, float] = DEFAULT_SIGMA_BAND,
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: Optional[float] = None,
    order: int = 4,
    to_db: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sigma-band power envelope of one channel, down-sampled to ``sf_env``.

    Band-passes ``data`` to ``band`` (zero-phase Butterworth), squares the
    Hilbert analytic amplitude to get instantaneous sigma power, optionally
    smooths it, averages it into consecutive ``1/sf_env``-second bins, and (by
    default) converts to dB -- so the result matches the reference ``get_iso``,
    whose infraslow spectrum is taken on the **log** sigma-power series. Working
    in dB compresses heavy-tailed spindle transients that would otherwise smear
    variance across the whole slow spectrum and dilute the infraslow band.

    Args:
        data: 1-D EEG signal for one channel.
        sf: Sampling rate of ``data`` (Hz).
        band: Band-pass edges (Hz) for the sigma band.
        sf_env: Output rate (Hz) of the down-sampled envelope.
        smooth_sec: If given, moving-average the full-rate power over this many
            seconds before down-sampling (extra high-frequency smoothing).
        order: Butterworth order for the band-pass.
        to_db: If ``True`` (default, matches the reference), return the envelope
            in dB (``10*log10``); set ``False`` for the raw linear power.

    Returns:
        ``(t, power)`` -- bin-centre times (s) and per-bin sigma power (dB if
        ``to_db`` else linear), both sampled at ``sf_env``.
    """
    x = _to_1d(data, name="data")
    if sf <= 0 or sf_env <= 0:
        raise ValueError(f"sf and sf_env must be positive; got {sf}, {sf_env}.")

    filt = ButterFilter(sf, list(band), mode="band", order=order)(x)
    power = np.abs(signal.hilbert(filt)) ** 2

    if smooth_sec:
        from scipy.ndimage import uniform_filter1d  # noqa: PLC0415 - lazy

        win = max(1, int(round(smooth_sec * sf)))
        power = uniform_filter1d(power, size=win, mode="nearest")

    t, means = _bin_average(power, sf, sf_env)
    if to_db:
        means = _to_db(means)
    return t, means


def _bin_average(x: np.ndarray, sf: float, sf_env: float) -> Tuple[np.ndarray, np.ndarray]:
    """Average ``x`` (sampled at ``sf``) into non-overlapping ``1/sf_env`` s bins."""
    factor = sf / sf_env
    n_bins = int(np.floor(len(x) / factor))
    if n_bins < 1:
        raise ValueError(
            f"Signal too short ({len(x)} samples at {sf} Hz) for even one "
            f"{1 / sf_env:g}-s bin."
        )
    # Edges in samples; average each [edge_i, edge_{i+1}) block.
    edges = (np.arange(n_bins + 1) * factor).round().astype(int)
    means = np.array([x[edges[i] : edges[i + 1]].mean() for i in range(n_bins)])
    t = (np.arange(n_bins) + 0.5) / sf_env
    return t, means


def infraslow_spectrum(
    x,
    sf_env: float,
    *,
    infraslow_band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND,
    window_sec: float = DEFAULT_WINDOW_SEC,
    detrend: str = "linear",
) -> InfraslowSpectrum:
    """Welch power spectrum of a slow series, summarised over the infraslow band.

    Args:
        x: Slow time series (e.g. a sigma-power envelope.
        sf_env: Sampling rate of ``x`` (Hz).
        infraslow_band: Infraslow band ``(lo, hi)`` (Hz) to summarise (peak + band power).
        window_sec: Welch segment length (s). Clipped to the series length; a
            longer window sharpens low-frequency resolution.
        detrend: Passed to :func:`scipy.signal.welch`. Default ``"linear"`` matches
            the reference ``get_iso`` (``scipy.signal.detrend`` defaults to linear),
            removing each segment's slow drift so a trend can't leak into the
            lowest infraslow bins; ``"constant"`` removes only the mean.

    Returns:
        An :class:`InfraslowSpectrum`.
    """
    x = _to_1d(x, name="x")
    if sf_env <= 0:
        raise ValueError(f"sf_env must be positive; got {sf_env}.")
    nperseg = int(min(len(x), max(8, round(window_sec * sf_env))))
    freqs, psd = signal.welch(
        x,
        fs=sf_env,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=detrend,
        window="hann",
    )

    total_power = float(_trapz(psd, freqs))
    in_band = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
    if in_band.any():
        band_power = float(_trapz(psd[in_band], freqs[in_band]))
        loc = np.argmax(psd[in_band])
        peak_freq = float(freqs[in_band][loc])
        peak_power = float(psd[in_band][loc])
    else:
        band_power = peak_freq = peak_power = float("nan")
    rel = band_power / total_power if total_power else float("nan")

    # Reference ``get_iso`` peak: argmax over the whole spectrum (excluding DC),
    # not just within ``infraslow_band`` -- its landing in-band is the sanity check.
    nz = freqs > 0
    if nz.any():
        floc = int(np.argmax(psd[nz]))
        full_peak_freq = float(freqs[nz][floc])
        full_peak_power = float(psd[nz][floc])
    else:
        full_peak_freq = full_peak_power = float("nan")

    return InfraslowSpectrum(
        freqs=freqs,
        psd=psd,
        sf_env=float(sf_env),
        infraslow_band=infraslow_band,
        peak_freq=peak_freq,
        peak_power=peak_power,
        full_peak_freq=full_peak_freq,
        full_peak_power=full_peak_power,
        band_power=band_power,
        total_power=total_power,
        rel_band_power=rel,
    )


def bigaussian(f, amp, mu, sd_l, sd_r):
    """Two Gaussian halves sharing one peak but independent left/right widths --
    captures the asymmetric shape (steep rise, slow decay) real infraslow spectra
    show, which a symmetric Gaussian would pull away from the true peak to compromise on."""
    sd = np.where(f < mu, sd_l, sd_r)
    return amp * np.exp(-0.5 * ((f - mu) / sd) ** 2)


def fit_isfs(freqs, corrected, infraslow_band=DEFAULT_INFRASLOW_BAND, baseline_band=DEFAULT_BASELINE_BAND):
    """Bi-Gaussian ISFS fit (peak, bandwidth, AUC, detection) with `mu` fixed at the
    empirical argmax -- see plot_infraslow.ipynb's fit_isfs for why (too few points
    in the fit window to also let a 4th free parameter float)."""
    base_m = (freqs >= baseline_band[0]) & (freqs <= baseline_band[1])
    fit_m = (freqs >= infraslow_band[0]) & (freqs < baseline_band[0])
    ff, yy = freqs[fit_m], corrected[fit_m]
    mu = float(ff[np.argmax(yy)])

    def _bigaussian_fixed_mu(f, amp, sd_l, sd_r):
        return bigaussian(f, amp, mu, sd_l, sd_r)

    p0 = [max(yy.max(), 1e-9), 0.01, 0.01]
    (amp, sd_l, sd_r), _ = curve_fit(_bigaussian_fixed_mu, ff, yy, p0=p0,
                                     bounds=([0, 1e-3, 1e-3], [np.inf, 0.05, 0.05]),
                                     maxfev=10000)
    popt = (amp, mu, sd_l, sd_r)
    lo, hi = mu - sd_l, mu + sd_r
    bandwidth = hi - lo
    f_auc = np.linspace(lo, hi, 400)
    auc = float(_trapz(bigaussian(f_auc, *popt), f_auc))
    threshold = 1.5 * corrected[base_m].std()
    return dict(popt=popt, amp=amp, mu=mu, sd_l=sd_l, sd_r=sd_r, lo=lo, hi=hi,
                bandwidth=bandwidth, auc=auc, threshold=threshold,
                detected=bool(amp > threshold))


def _threshold_crossing(curve_freqs, curve, start_idx, threshold=0.0):
    """First frequency, at or after `start_idx`, where `curve` drops from >=
    `threshold` to < `threshold`, linearly interpolated between the two
    bracketing samples (see plot_infraslow.ipynb)."""
    seg = curve[start_idx:]
    crossings = np.flatnonzero((seg[:-1] >= threshold) & (seg[1:] < threshold))
    if crossings.size == 0:
        return float(curve_freqs[-1])
    i = start_idx + int(crossings[0])
    f_a, f_b = curve_freqs[i], curve_freqs[i + 1]
    y_a, y_b = curve[i], curve[i + 1]
    return float(f_a + (threshold - y_a) * (f_b - f_a) / (y_b - y_a))


def chromatogram_peak_area(curve_freqs, curve, threshold=0.0, infraslow_band=DEFAULT_INFRASLOW_BAND):
    """Chromatogram-style peak area: `curve` integrated above a sloped baseline
    from (infraslow_band[0], curve there) down to where `curve` drops to
    `threshold` (see plot_infraslow.ipynb)."""
    x0 = infraslow_band[0]
    y0 = float(np.interp(x0, curve_freqs, curve))
    peak_idx = int(np.argmax(curve))
    x1 = _threshold_crossing(curve_freqs, curve, peak_idx, threshold)

    peak_m = (curve_freqs >= x0) & (curve_freqs <= x1)
    xf, yf = curve_freqs[peak_m], curve[peak_m]
    incline = threshold + (y0 - threshold) * (x1 - xf) / (x1 - x0)
    above = np.clip(yf - incline, 0, None)
    area = float(_trapz(above, xf))
    return dict(area=area, freqs=xf, curve=yf, incline=incline, x0=x0, y0=y0, x1=x1,
                threshold=threshold)


__all__ = [
    "InfraslowSpectrum",
    "DEFAULT_SIGMA_BAND",
    "DEFAULT_INFRASLOW_BAND",
    "DEFAULT_SF_ENV",
    "DEFAULT_WINDOW_SEC",
    "DEFAULT_BASELINE_BAND",
    "power_envelope",
    "infraslow_spectrum",
    "bigaussian",
    "fit_isfs",
    "chromatogram_peak_area",
]
