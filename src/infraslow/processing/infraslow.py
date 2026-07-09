"""Infraslow (~0.02 Hz) oscillations of sigma-band power.

Sleep-spindle (sigma, ~12-15 Hz) power is not steady across NREM: it waxes and
wanes with an **infraslow rhythm** around 0.02 Hz (a ~50 s period), and spindles
cluster on the rising phase / peaks of that rhythm (Lecci et al., 2017;
Watson, 2018). This module quantifies that rhythm two complementary ways:

* **From the continuous EEG** -- band-pass to sigma, take the Hilbert power
  envelope, down-sample it, and read off its low-frequency spectrum
  (:func:`sigma_power_envelope` -> :func:`infraslow_spectrum`, wrapped by
  :func:`sigma_infraslow_oscillation`). Detector-independent; the reference.

* **From a spindle detector's events** -- turn the detected spindle times into a
  smooth spindle-rate time series and read off *its* infraslow spectrum
  (:func:`spindle_rate_series` -> :func:`infraslow_spectrum`, wrapped by
  :func:`spindle_infraslow_oscillation`). This is what lets YASA and Luna be
  compared: do the two detectors recover the same infraslow rhythm of spindle
  occurrence?

:func:`average_spindle_spectrum` gives the companion *high*-frequency view -- the
mean power spectrum of the detected spindle epochs, showing where in the sigma
band each detector's events concentrate.

Everything here is pure NumPy/SciPy (no matplotlib, no YASA): plotting lives in
:mod:`infraslow.viz.infraslow` and statistical comparison in
:mod:`infraslow.stats.infraslow`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.signal as signal

from .signal import ButterFilter

# ``np.trapz`` is deprecated in NumPy 2.0 in favour of ``np.trapezoid``; use
# whichever the installed NumPy provides.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Sigma (spindle) band the ISO sigma-power series is integrated over (Hz).
# Matches the reference ``get_iso`` (Liu et al.), which uses 11-15 Hz for the
# multitaper sigma-power estimate -- deliberately wider than the detector's
# ``detection.spindles_detect`` ``freq_sp`` (12-15 Hz), as this is a power
# integration, not a detection band.
DEFAULT_SIGMA_BAND: Tuple[float, float] = (11.0, 15.0)
# Infraslow band the sigma-power oscillation is expected to peak in (Hz).
# Widened to 0.01-0.1 Hz (~10-100 s) around the canonical ~0.02 Hz (50 s)
# sigma-power rhythm, so peak-finding and band power stay anchored to it while
# capturing slower and faster infraslow components.
DEFAULT_INFRASLOW_BAND: Tuple[float, float] = (0.01, 0.1)
# Rate the slow envelope / spindle-rate series is sampled at for spectral analysis
# (Hz). 1 Hz is far above the infraslow Nyquist and keeps the series compact.
DEFAULT_SF_ENV: float = 1.0
# Welch segment length (seconds) for the infraslow spectrum -- long enough to
# resolve the 0.01 Hz lower edge of DEFAULT_INFRASLOW_BAND (0.01 Hz resolution
# at 100 s).
DEFAULT_WINDOW_SEC: float = 100.0


@dataclass
class InfraslowSpectrum:
    """Low-frequency power spectrum of a slow time series, with band metrics.

    Attributes:
        freqs, psd: The full one-sided spectrum (``psd`` in units^2/Hz).
        sf_env: Sampling rate of the analysed series (Hz).
        band: The infraslow band ``(lo, hi)`` the metrics summarise.
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
    band: Tuple[float, float]
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


def sigma_power_envelope(
    data,
    sf: float,
    *,
    sigma_band: Tuple[float, float] = DEFAULT_SIGMA_BAND,
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: Optional[float] = None,
    order: int = 4,
    to_db: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sigma-band power envelope of one channel, down-sampled to ``sf_env``.

    Band-passes ``data`` to ``sigma_band`` (zero-phase Butterworth), squares the
    Hilbert analytic amplitude to get instantaneous sigma power, optionally
    smooths it, averages it into consecutive ``1/sf_env``-second bins, and (by
    default) converts to dB -- so the result matches the reference ``get_iso``,
    whose infraslow spectrum is taken on the **log** sigma-power series. Working
    in dB compresses heavy-tailed spindle transients that would otherwise smear
    variance across the whole slow spectrum and dilute the infraslow band.

    Args:
        data: 1-D EEG signal for one channel.
        sf: Sampling rate of ``data`` (Hz).
        sigma_band: Band-pass edges (Hz) for the sigma band.
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

    filt = ButterFilter(sf, list(sigma_band), mode="band", order=order)(x)
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


def spindle_rate_series(
    peaks_sec,
    *,
    duration_sec: float,
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous spindle-rate time series from detected spindle times.

    Bins the spindle event times into ``1/sf_env``-second bins over
    ``[0, duration_sec)`` and Gaussian-smooths the per-bin counts into a smooth
    "instantaneous spindle rate" (spindles/s) -- the signal whose slow spectrum
    reveals the infraslow clustering of spindles.

    Args:
        peaks_sec: Spindle event times (seconds from recording start), e.g.
            ``spindles.summary()["Peak"]``.
        duration_sec: Length of the recording (s); sets the time base so two
            detectors' series are directly comparable.
        sf_env: Output rate (Hz) of the rate series.
        smooth_sec: Gaussian smoothing width (s). ``0`` disables smoothing.

    Returns:
        ``(t, rate)`` -- bin-centre times (s) and spindle rate (spindles/s), at
        ``sf_env``.
    """
    peaks = _to_1d(peaks_sec, name="peaks_sec") if np.size(peaks_sec) else np.empty(0)
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive; got {duration_sec}.")
    n_bins = max(1, int(round(duration_sec * sf_env)))
    counts, _ = np.histogram(peaks, bins=n_bins, range=(0.0, duration_sec))
    rate = counts.astype(float) * sf_env  # per-bin count -> spindles per second

    if smooth_sec:
        from scipy.ndimage import gaussian_filter1d  # noqa: PLC0415 - lazy

        rate = gaussian_filter1d(rate, sigma=max(1e-6, smooth_sec * sf_env), mode="nearest")

    t = (np.arange(n_bins) + 0.5) / sf_env
    return t, rate


def infraslow_spectrum(
    x,
    sf_env: float,
    *,
    band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND,
    window_sec: float = DEFAULT_WINDOW_SEC,
    detrend: str = "linear",
) -> InfraslowSpectrum:
    """Welch power spectrum of a slow series, summarised over the infraslow band.

    Args:
        x: Slow time series (e.g. a sigma-power envelope or a spindle-rate series).
        sf_env: Sampling rate of ``x`` (Hz).
        band: Infraslow band ``(lo, hi)`` (Hz) to summarise (peak + band power).
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
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if in_band.any():
        band_power = float(_trapz(psd[in_band], freqs[in_band]))
        loc = np.argmax(psd[in_band])
        peak_freq = float(freqs[in_band][loc])
        peak_power = float(psd[in_band][loc])
    else:
        band_power = peak_freq = peak_power = float("nan")
    rel = band_power / total_power if total_power else float("nan")

    # Reference ``get_iso`` peak: argmax over the whole spectrum (excluding DC),
    # not just within ``band`` -- its landing in-band is the sanity check.
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
        band=band,
        peak_freq=peak_freq,
        peak_power=peak_power,
        full_peak_freq=full_peak_freq,
        full_peak_power=full_peak_power,
        band_power=band_power,
        total_power=total_power,
        rel_band_power=rel,
    )


def sigma_infraslow_oscillation(
    data,
    sf: float,
    *,
    sigma_band: Tuple[float, float] = DEFAULT_SIGMA_BAND,
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: Optional[float] = None,
    band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND,
    window_sec: float = DEFAULT_WINDOW_SEC,
    order: int = 4,
    to_db: bool = True,
) -> Tuple[InfraslowSpectrum, np.ndarray, np.ndarray]:
    """Continuous-EEG infraslow oscillation of sigma power (envelope + spectrum).

    Convenience wrapper: :func:`sigma_power_envelope` followed by
    :func:`infraslow_spectrum`. By default the sigma-power envelope is taken in
    dB (``to_db=True``), matching the reference ``get_iso``.

    Returns:
        ``(spectrum, t, power)`` -- the :class:`InfraslowSpectrum` plus the
        down-sampled sigma-power envelope it was computed from (dB if ``to_db``).
    """
    t, power = sigma_power_envelope(
        data, sf, sigma_band=sigma_band, sf_env=sf_env, smooth_sec=smooth_sec,
        order=order, to_db=to_db,
    )
    spec = infraslow_spectrum(power, sf_env, band=band, window_sec=window_sec)
    return spec, t, power


def spindle_infraslow_oscillation(
    peaks_sec,
    *,
    duration_sec: float,
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: float = 10.0,
    band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND,
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> Tuple[InfraslowSpectrum, np.ndarray, np.ndarray]:
    """Detector-derived infraslow oscillation of spindle occurrence.

    Convenience wrapper: :func:`spindle_rate_series` followed by
    :func:`infraslow_spectrum`.

    Returns:
        ``(spectrum, t, rate)`` -- the :class:`InfraslowSpectrum` plus the
        spindle-rate series it was computed from.
    """
    t, rate = spindle_rate_series(
        peaks_sec, duration_sec=duration_sec, sf_env=sf_env, smooth_sec=smooth_sec
    )
    spec = infraslow_spectrum(rate, sf_env, band=band, window_sec=window_sec)
    return spec, t, rate


def average_spindle_spectrum(
    data,
    sf: float,
    peaks_sec,
    *,
    halfwidth_sec: float = 1.0,
    fmax: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Mean power spectrum of the detected spindle epochs (the sigma-band view).

    Extracts a ``±halfwidth_sec`` window of the raw signal around each spindle
    peak, computes a Hann-tapered periodogram per window, and averages them --
    showing where in the sigma band a detector's spindles concentrate. Windows
    that run off a recording edge are skipped.

    Args:
        data: 1-D EEG signal for one channel.
        sf: Sampling rate of ``data`` (Hz).
        peaks_sec: Spindle peak times (seconds), e.g. ``summary()["Peak"]``.
        halfwidth_sec: Half-window (s) around each peak.
        fmax: Upper frequency (Hz) to return.

    Returns:
        ``(f, psd, n_used)`` -- frequencies (<= ``fmax``), the mean periodogram,
        and how many spindle epochs contributed.

    Raises:
        ValueError: if no spindle window fits inside the recording.
    """
    x = _to_1d(data, name="data")
    peaks = _to_1d(peaks_sec, name="peaks_sec") if np.size(peaks_sec) else np.empty(0)
    h = int(round(halfwidth_sec * sf))
    if h < 1:
        raise ValueError("halfwidth_sec too small for this sampling rate.")
    win = signal.windows.hann(2 * h)
    scale = 1.0 / (sf * (win**2).sum())  # PSD normalisation

    acc = None
    n_used = 0
    for p in peaks:
        i = int(round(float(p) * sf))
        lo, hi = i - h, i + h
        if lo < 0 or hi > len(x):
            continue
        seg = x[lo:hi]
        seg = (seg - seg.mean()) * win
        p_seg = (np.abs(np.fft.rfft(seg)) ** 2) * scale
        acc = p_seg if acc is None else acc + p_seg
        n_used += 1
    if n_used == 0:
        raise ValueError("No spindle window fit inside the recording; check peaks/sf.")

    f = np.fft.rfftfreq(2 * h, d=1.0 / sf)
    psd = acc / n_used
    keep = f <= fmax
    return f[keep], psd[keep], n_used


__all__ = [
    "InfraslowSpectrum",
    "DEFAULT_SIGMA_BAND",
    "DEFAULT_INFRASLOW_BAND",
    "DEFAULT_SF_ENV",
    "DEFAULT_WINDOW_SEC",
    "sigma_power_envelope",
    "spindle_rate_series",
    "infraslow_spectrum",
    "sigma_infraslow_oscillation",
    "spindle_infraslow_oscillation",
    "average_spindle_spectrum",
]
