"""Infraslow (~0.02 Hz) oscillations of sigma-band power.

Sleep-spindle (sigma, ~12-15 Hz) power is not steady across NREM: it waxes and
wanes with an **infraslow rhythm** around 0.02 Hz (a ~50 s period), and spindles
cluster on the rising phase / peaks of that rhythm (Lecci et al., 2017;
Watson, 2018). This module quantifies that rhythm from the continuous EEG --
band-pass to sigma, take the Hilbert power envelope, down-sample it, and read
off its low-frequency spectrum (:func:`eeg_envelope` -> :func:`infraslow_spectrum`).

Everything here is pure NumPy/SciPy plus MNE for band-pass filtering (no
matplotlib, no YASA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mne
import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit

from .signal import ButterFilter
from ..constants import (
    DEFAULT_BASELINE_BAND,
    DEFAULT_DELTA_BAND,
    DEFAULT_INFRASLOW_BAND,
    DEFAULT_ISFS_FILTER_ORDER,
    DEFAULT_ISFS_LOWPASS_HZ,
    DEFAULT_ISFS_MIN_EVENTS,
    DEFAULT_ISFS_PERIOD,
    DEFAULT_ISFS_TUKEY_ALPHA,
    DEFAULT_SF_ENV,
    DEFAULT_SIGMA_BAND,
    DEFAULT_WINDOW_SEC,
)

# ``np.trapz`` is deprecated in NumPy 2.0 in favour of ``np.trapezoid``; use
# whichever the installed NumPy provides.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# How much finer than window_sec's natural spacing infraslow_spectrum's frequency grid
# is -- unlike an FFT-based method (Welch, multitaper), a Morlet wavelet transform gets
# a real (not interpolated) value at any frequency we ask for, so raising this is just
# "compute more wavelets", not a statistical-vs-cosmetic-resolution tradeoff; it gives
# fit_isfs more points to localize its peak against, at the cost of more compute.
_GRID_OVERSAMPLE: int = 4


def _wavelet_freqs(sf: float, n_samples: int, window_sec: float, lo: float, hi: float) -> np.ndarray:
    """Frequency grid for a Morlet transform: ``~1/window_sec`` Hz spacing,
    :data:`_GRID_OVERSAMPLE` finer, fixed regardless of ``n_samples`` (so spectra
    from series of different lengths share one grid and can be averaged), spanning
    ``[lo, hi]``. Shared by every :func:`mne.time_frequency.tfr_array_morlet` call
    in this module so their grids are built identically (see :func:`infraslow_spectrum`).
    """
    nseg = int(min(n_samples, max(8, round(window_sec * sf))))
    df = sf / nseg / _GRID_OVERSAMPLE
    freqs = np.arange(max(df, lo), hi + 1e-9, df)
    if freqs.size == 0:
        freqs = np.array([(lo + hi) / 2.0])
    return freqs


def _wavelet_n_cycles(freqs: np.ndarray, n_samples: int, sf: float) -> np.ndarray:
    """Cycle count for a Morlet transform: fixed 3-cycle target, capped per
    frequency so every wavelet's support stays inside the series. Full Morlet
    support is ~(5/pi)*n_cycles/freq seconds, not n_cycles/freq (see
    ``mne.time_frequency.morlet``'s ``sigma_t = n_cycles/(2*pi*freq)``, support =
    5*``sigma_t`` each side); the 0.85 factor leaves a safety margin against
    ``arange``'s sample-count rounding. Shared by every
    :func:`mne.time_frequency.tfr_array_morlet` call in this module (see
    :func:`infraslow_spectrum`).
    """
    duration_sec = n_samples / sf
    max_cycles = freqs * duration_sec * (np.pi / 5.0) * 0.85
    return np.minimum(3.0, max_cycles)


@dataclass
class InfraslowSpectrum:
    """Low-frequency power spectrum of a slow time series.

    Attributes:
        freqs, psd: The full one-sided spectrum (``psd`` in units^2/Hz).
    """

    freqs: np.ndarray
    psd: np.ndarray


def _to_1d(x, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {np.shape(x)}.")
    return arr


def _to_db(power: np.ndarray, *, factor: float = 10.0) -> np.ndarray:
    """Convert a linear power (or, with ``factor=20``, amplitude) series to dB,
    like the reference ``get_iso``.

    Non-positive bins (which would give ``-inf``) are floored to the smallest
    positive value in the series so the log stays finite without inventing a
    magnitude; this virtually never triggers on a Hilbert envelope.
    """
    power = np.asarray(power, dtype=float)
    positive = power[power > 0]
    floor = positive.min() if positive.size else 1.0
    return factor * np.log10(np.where(power > 0, power, floor))


def eeg_envelope(
    data,
    sf: float,
    *,
    band: Tuple[float, float] = DEFAULT_SIGMA_BAND,
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: Optional[float] = None,
    to_db: bool = True,
    kind: str = "power",
    wavelet: bool = True,
    wavelet_window_sec: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sigma-band power or amplitude envelope of one channel, down-sampled to ``sf_env``.

    By default, band-passes ``data`` to ``band`` (MNE's zero-phase FIR filter,
    ``mne.filter.filter_data`` with ``method="fir"``) and takes the Hilbert
    analytic amplitude -- a practical stand-in for the paper's wavelet-derived
    sigma power. Set ``wavelet=True`` to get the analytic amplitude directly from
    a Morlet wavelet transform instead (magnitude averaged across a grid of
    frequencies spanning ``band``), matching the reference more closely at the
    cost of a slower, denser decomposition. The grid and per-frequency cycle
    count are built the same way as :func:`infraslow_spectrum` (see
    ``wavelet_window_sec``), just parameterised by ``band``/``sf`` instead of
    the infraslow band/``sf_env``.

    Either way, with ``kind="power"`` (default) the analytic amplitude is squared
    to get instantaneous sigma power, matching the reference ``get_iso``, whose
    infraslow spectrum is taken on the **log** sigma-power series -- working in
    dB compresses heavy-tailed spindle transients that would otherwise smear
    variance across the whole slow spectrum and dilute the infraslow band. With
    ``kind="amplitude"`` the raw (unsquared) analytic amplitude is kept instead,
    and ``to_db`` then uses the amplitude-ratio convention (``20*log10``) rather
    than power's (``10*log10``), so the two ``kind``s stay on a comparable dB scale.

    The full-rate series is optionally smoothed and averaged into consecutive
    ``1/sf_env``-second bins.

    Args:
        data: 1-D EEG signal for one channel.
        sf: Sampling rate of ``data`` (Hz).
        band: Band-pass edges (Hz) for the sigma band (or, with ``wavelet``, the
            frequency range the Morlet wavelets span).
        sf_env: Output rate (Hz) of the down-sampled envelope.
        smooth_sec: If given, moving-average the full-rate series over this many
            seconds before down-sampling (extra high-frequency smoothing).
        to_db: If ``True`` (default, matches the reference), return the envelope
            in dB (``10*log10`` for ``kind="power"``, ``20*log10`` for
            ``kind="amplitude"``); set ``False`` for the raw linear value.
        kind: ``"power"`` (default, matches the reference) or ``"amplitude"``.
        wavelet: If ``True``, use a Morlet wavelet transform (grid of frequencies
            spanning ``band``, magnitude averaged across frequency) instead of
            the default band-pass + Hilbert envelope.
        wavelet_window_sec: Only used when ``wavelet=True``; sets the
            (pre-oversampling) frequency grid spacing (``~1/wavelet_window_sec``
            Hz), the same role ``window_sec`` plays in :func:`infraslow_spectrum`,
            scaled down for the much higher sigma-band frequencies.

    Returns:
        ``(t, envelope)`` -- bin-centre times (s) and per-bin sigma power/amplitude
        (dB if ``to_db`` else linear), both sampled at ``sf_env``.
    """
    if kind not in ("power", "amplitude"):
        raise ValueError(f'kind must be "power" or "amplitude"; got {kind!r}.')

    x = _to_1d(data, name="data")
    if sf <= 0 or sf_env <= 0:
        raise ValueError(f"sf and sf_env must be positive; got {sf}, {sf_env}.")

    if wavelet:
        freqs = _wavelet_freqs(sf, x.size, wavelet_window_sec, band[0], band[1])
        n_cycles = _wavelet_n_cycles(freqs, x.size, sf)
        tfr = mne.time_frequency.tfr_array_morlet(
            x[np.newaxis, np.newaxis, :], sfreq=sf, freqs=freqs, n_cycles=n_cycles,
            output="power", zero_mean=True, verbose=False,
        )
        analytic_amp = tfr[0, 0].mean(axis=0)
    else:
        filt = mne.filter.filter_data(x, sf, l_freq=band[0], h_freq=band[1], method="fir", verbose=False)
        analytic_amp = np.abs(signal.hilbert(filt))

    envelope = analytic_amp ** 2 if kind == "power" else analytic_amp

    if smooth_sec:
        from scipy.ndimage import uniform_filter1d  # noqa: PLC0415 - lazy

        win = max(1, int(round(smooth_sec * sf)))
        envelope = uniform_filter1d(envelope, size=win, mode="nearest")

    t, means = _bin_average(envelope, sf, sf_env)
    if to_db:
        means = _to_db(means, factor=10.0 if kind == "power" else 20.0)
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
    window_sec: float = DEFAULT_WINDOW_SEC,
    detrend: str = "linear",
) -> InfraslowSpectrum:
    """Morlet-wavelet power spectrum of a slow series (0..0.1 Hz).

    Continuous Morlet-wavelet transform (:func:`mne.time_frequency.tfr_array_morlet`)
    of the whole series, evaluated directly at a grid of frequencies spanning DC..
    Nyquist. Unlike an FFT-based method (Welch, multitaper), each frequency gets its
    own dedicated wavelet rather than a shared DFT bin, so the grid's density and
    shape are entirely under our control -- fixed by `window_sec`/`_GRID_OVERSAMPLE`,
    independent of `len(x)` -- instead of being tied to the series length (which
    otherwise means spectra from series of different lengths, e.g. one call per bout,
    come back on different grids and can't be stacked/averaged together, as happened
    when this used `psd_array_multitaper`) or needing interpolation/zero-padding
    afterward (which for an FFT-based method risks silently cropping instead of
    padding if the target grid is denser than what the series length can natively
    support). Power is averaged over time at each frequency to give one spectral
    estimate per bin.

    Each wavelet's cycle count targets 3 (well below the TFR default of 7, which the
    lowest infraslow frequencies -- ~0.01 Hz, ~100 s per cycle -- couldn't support
    within a typical bout) but is additionally capped per frequency so its support
    never exceeds the series itself; the cap can push the lowest frequencies' cycle
    count well under 3, trading resolution there for just not raising.

    Band/peak metrics (relative power, baseline correction) are *not* computed
    here -- see :func:`relative_power` and :func:`baseline_correct`, which take
    this function's ``freqs``/``psd`` directly and compose with averaging across
    bouts (peak/band metrics on a single bout's own noisy spectrum are rarely
    what's wanted; see ``demo_infraslow_yasa_recheck.ipynb``'s Figure 4/5).

    Args:
        x: Slow time series (e.g. a sigma-power envelope.
        sf_env: Sampling rate of ``x`` (Hz).
        window_sec: Sets the (pre-oversampling) frequency grid spacing
            (``~1/window_sec`` Hz), the same role it played setting Welch's
            ``nperseg`` originally -- independent of series length, so spectra from
            series of different lengths (e.g. one call per bout) always share one
            ``freqs`` array and can be averaged together. A longer window sharpens
            resolution; see :data:`_GRID_OVERSAMPLE` for further (real, per-frequency)
            oversampling on top of this.
        detrend: ``"linear"`` (default, matches the reference ``get_iso``) or
            ``"constant"``, passed to :func:`scipy.signal.detrend` on the whole series
            before the wavelet transform, so a slow drift can't leak into the lowest
            infraslow bins; ``"constant"`` removes only the mean. ``None``/``False``
            skips detrending.

    Returns:
        An :class:`InfraslowSpectrum` (``freqs``, ``psd``).
    """
    x = _to_1d(x, name="x")
    if sf_env <= 0:
        raise ValueError(f"sf_env must be positive; got {sf_env}.")
    if detrend:
        x = signal.detrend(x, type=detrend)

    freqs = _wavelet_freqs(sf_env, len(x), window_sec, 0.0, 0.1)
    n_cycles = _wavelet_n_cycles(freqs, len(x), sf_env)
    tfr = mne.time_frequency.tfr_array_morlet(
        x[np.newaxis, np.newaxis, :], sfreq=sf_env, freqs=freqs, n_cycles=n_cycles,
        output="power", zero_mean=True, verbose=False,
    )
    psd = tfr[0, 0].mean(axis=1)

    return InfraslowSpectrum(freqs=freqs, psd=psd)


def relative_power(psd, freqs, band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND):
    """Normalize ``psd`` to unit area within ``band`` (Hz).

    ``psd`` may be a single spectrum or a stack of them (last axis = frequency,
    matching ``freqs``) -- e.g. every bout's own :func:`infraslow_spectrum` psd,
    normalized by the *same* denominator (typically their mean's) so they stay
    directly comparable/averageable (see ``demo_infraslow_yasa_recheck.ipynb``'s
    Figure 4A individual-bout lines and mean +/- SD band).
    """
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    band_m = (freqs >= band[0]) & (freqs <= band[1])
    denom = _trapz(psd[..., band_m], freqs[band_m], axis=-1)
    return psd / denom[..., np.newaxis] if psd.ndim > 1 else psd / denom


def baseline_correct(rel, freqs, baseline_band: Tuple[float, float] = DEFAULT_BASELINE_BAND):
    """Subtract the mean of ``rel`` over ``baseline_band`` (Hz) from ``rel``.

    ``rel`` is typically a :func:`relative_power` output; the noise floor over
    ``baseline_band`` (high infraslow frequencies with no expected ISFS peak) is
    treated as the zero level the true ISFS peak sits above.
    """
    rel = np.asarray(rel, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    base_m = (freqs >= baseline_band[0]) & (freqs <= baseline_band[1])
    baseline = rel[..., base_m].mean(axis=-1)
    return rel - (baseline[..., np.newaxis] if rel.ndim > 1 else baseline)


def band_power_spectrum(
    x,
    sf: float,
    *,
    freq_range: Tuple[float, float] = (0.1, 45.0),
    window_sec: float = 4.0,
    detrend: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """Morlet-wavelet power spectrum of a fast series (e.g. raw EEG) over ``freq_range``.

    Same fixed, length-independent frequency grid and cycle-count formula as
    :func:`infraslow_spectrum` (:func:`_wavelet_freqs` / :func:`_wavelet_n_cycles`,
    so spectra from segments of different lengths -- e.g. one call per bout --
    share one ``freqs`` array and can be averaged together), aimed at the
    standard EEG bands (delta..gamma) on raw EEG instead of the infraslow band
    on a ~1 Hz sigma-power envelope.

    Args:
        x: 1-D fast time series, e.g. one bout's raw EEG channel.
        sf: Sampling rate (Hz) of ``x``.
        freq_range: ``(low_hz, high_hz)`` span to cover.
        window_sec: Sets the frequency grid spacing (``~1/window_sec`` Hz),
            independent of ``len(x)``.
        detrend: ``"linear"``/``"constant"``/falsy, passed to
            :func:`scipy.signal.detrend` before the wavelet transform.

    Returns:
        ``(freqs, tfr)`` -- frequency grid (Hz, length ``n_freqs``) and the
        full time-resolved power, shape ``(n_freqs, len(x))`` -- **not**
        averaged over time (unlike :func:`infraslow_spectrum`'s ``psd``).
    """
    x = _to_1d(x, name="x")
    if sf <= 0:
        raise ValueError(f"sf must be positive; got {sf}.")
    lo, hi = freq_range
    if detrend:
        x = signal.detrend(x, type=detrend)

    freqs = _wavelet_freqs(sf, len(x), window_sec, lo, hi)
    n_cycles = _wavelet_n_cycles(freqs, len(x), sf)
    tfr = mne.time_frequency.tfr_array_morlet(
        x[np.newaxis, np.newaxis, :], sfreq=sf, freqs=freqs, n_cycles=n_cycles,
        output="power", zero_mean=True, verbose=False,
    )
    return freqs, tfr[0, 0]


def isfs_lowpass(
    x,
    sf_env: float,
    *,
    cutoff_hz: float = DEFAULT_ISFS_LOWPASS_HZ,
    order: int = DEFAULT_ISFS_FILTER_ORDER,
    tukey_alpha: float = DEFAULT_ISFS_TUKEY_ALPHA,
) -> np.ndarray:
    """Zero-phase low-pass of a sigma-power course for time-domain ISFS detection.

    Matches the reference ``get_iso``: ``x`` is tapered with a Tukey window
    (``tukey_alpha`` cosine fraction -- rolls the segment's edges down to zero
    before filtering so ``filtfilt``'s edge padding doesn't ring) and then
    zero-phase low-pass filtered (:class:`~infraslow.processing.signal.
    ButterFilter`, ``order``-th order Butterworth, -3 dB at ``cutoff_hz``,
    ``filtfilt``).

    Unlike reconstructing from a single Morlet wavelet frequency (which forces
    the output toward a pure sinusoid at whatever frequency it's evaluated at,
    and -- if that frequency is a fitted spectral peak -- makes any downstream
    cycle check circular), a low-pass keeps the whole passband's natural,
    possibly non-sinusoidal shape below ``cutoff_hz``. The peak/trough/
    zero-crossing scan this feeds is meant to be an independent, time-domain
    confirmation of the ISFS rhythm :func:`fit_isfs` finds in the frequency
    domain -- it needs that real shape, not one manufactured from the same
    peak estimate it's supposed to be checking.

    Args:
        x: 1-D slow series (e.g. a sigma-power envelope, or one bout's slice of it).
        sf_env: Sampling rate of ``x`` (Hz).
        cutoff_hz: -3 dB low-pass cutoff (Hz).
        order: Butterworth filter order.
        tukey_alpha: Tukey window cosine-taper fraction applied to ``x`` before filtering.

    Returns:
        The filtered series, same length as ``x``.
    """
    x = _to_1d(x, name="x")
    tapered = x * signal.windows.tukey(x.size, alpha=tukey_alpha)
    return ButterFilter(sfreq=sf_env, cutoff=cutoff_hz, mode="low", order=order)(tapered)


def isfs_phase_bins(
    t,
    x,
    *,
    isfs_period: Tuple[float, float] = DEFAULT_ISFS_PERIOD,
) -> Tuple[np.ndarray, list]:
    """Split a filtered, zero-mean sigma-power course into 8 ISFS phase bins per cycle.

    Matches the reference ``get_iso``. ``x`` is expected to already be the
    output of :func:`isfs_lowpass`, mean-centered (e.g. z-scored) so its own
    zero-line is "the mean of the signal" -- this function only finds zero-
    crossings, peaks and troughs and bins around them, it does no filtering or
    standardizing itself.

    A candidate ISFS cycle runs from one descending (+ -> -, "negative")
    zero-crossing to the next; it's only valid if that span's duration falls
    within ``isfs_period`` (default 25-100 s) *and* an ascending zero-crossing
    (the cycle's midpoint) falls strictly between them -- always true for a
    real sign change, but guards degenerate/edge segments. Everything outside
    a valid cycle (including a dangling first/last half-cycle at a bout's
    start/end that never completes both a peak and a trough) is "Not ISFS".

    Within a valid cycle, ``start`` (first crossing) to ``mid`` (ascending
    crossing) is the negative half-wave; ``mid`` to ``end`` (second crossing)
    is the positive half-wave. ``trough`` is the sample of minimum ``x``
    within the negative half-wave, ``peak`` the sample of maximum ``x`` within
    the positive half-wave (the literal minimum/maximum, not a
    :func:`scipy.signal.find_peaks` local-extremum call, so a trough/peak is
    always found once the half-wave itself is valid). Each half-wave is then
    quartered around its own extremum: bins 1/2 split at the mean **sample
    index** of ``start``/``trough``, bins 2/3 split at ``trough`` itself, bins
    3/4 split at the mean sample index of ``trough``/``mid``; bins 5/6/7/8
    mirror this around ``peak`` for the positive half-wave.

    Args:
        t: 1-D time (s), monotonically increasing, same sampling as ``x``.
        x: 1-D filtered, zero-mean sigma-power course (:func:`isfs_lowpass` output).
        isfs_period: ``(lo, hi)`` valid ISFS cycle period (s).

    Returns:
        ``(bins, cycles)`` -- ``bins`` is an ``int`` array the same length as
        ``x``: ``0`` ("Not ISFS") or ``1``-``8`` (phase bin) per sample.
        ``cycles`` is a list of dicts, one per valid cycle in time order, each
        with integer sample indices ``start``, ``trough``, ``mid``, ``peak``,
        ``end``, the 9-element ``edges`` array (bin boundaries, ``start`` to
        ``end``) used to assign that cycle's bins, and its ``period`` (s).
    """
    t = _to_1d(t, name="t")
    x = _to_1d(x, name="x")
    if t.shape != x.shape:
        raise ValueError(f"t and x must be the same shape; got {t.shape} vs {x.shape}.")

    zc_i = np.where(np.diff(np.signbit(x)))[0]  # index just before each zero-crossing
    desc = zc_i[x[zc_i] > 0]                    # descending (+ -> -), "negative" crossings
    asc = zc_i[x[zc_i] < 0]                     # ascending  (- -> +), "positive" crossings

    bins = np.zeros(x.size, dtype=int)
    cycles = []
    for c0, c1 in zip(desc[:-1].tolist(), desc[1:].tolist()):
        period = float(t[c1] - t[c0])
        if not (isfs_period[0] <= period <= isfs_period[1]):
            continue
        mid_candidates = asc[(asc > c0) & (asc < c1)]
        if mid_candidates.size == 0:
            continue
        mid = int(mid_candidates[0])

        trough = c0 + int(np.argmin(x[c0 : mid + 1]))
        peak = mid + int(np.argmax(x[mid : c1 + 1]))

        edges = np.array([
            c0,
            int(round((c0 + trough) / 2)),
            trough,
            int(round((trough + mid) / 2)),
            mid,
            int(round((mid + peak) / 2)),
            peak,
            int(round((peak + c1) / 2)),
            c1,
        ])
        for k in range(8):
            bins[edges[k] : edges[k + 1]] = k + 1
        bins[c1] = 8  # closed on the right: the final crossing belongs to bin 8

        cycles.append(dict(start=c0, trough=trough, mid=mid, peak=peak, end=c1,
                            edges=edges, period=period))

    return bins, cycles


def _nearest_sample_index(t: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index into `t` of the sample nearest each of `times` (searchsorted's
    insertion index vs. its left neighbour, whichever is actually closer).

    Shared by :func:`isfs_event_phase_distribution` and
    `pipeline.phase.compute_subject_phase_features`, which both need to map
    event onset times onto the nearest ISFS-phase-bin sample. Safe to call
    with an empty `t` or `times` (both call sites already guard the zero
    case before calling, but the helper itself does not assume it)."""
    if t.size == 0 or times.size == 0:
        return np.empty(0, dtype=int)
    idx = np.clip(np.searchsorted(t, times), 0, t.size - 1)
    left = np.clip(idx - 1, 0, t.size - 1)
    return np.where(np.abs(t[left] - times) <= np.abs(t[idx] - times), left, idx)


def isfs_event_phase_distribution(
    bouts_data, *, min_events: int = DEFAULT_ISFS_MIN_EVENTS, denominator: str = "in_cycle",
):
    """Percentage distribution of event onsets across the 8 ISFS phase bins.

    Based on the reference ``get_iso``'s microarousal/spindle/slow-wave/slow-wave-
    spindle-coupling phase-bin analysis: usable for any one of those event types,
    one call per type. Each event is assigned to the ISFS phase bin (see
    :func:`isfs_phase_bins`) of its nearest sample -- ``0`` ("Not ISFS") if it
    falls outside a valid 25-100 s cycle. Counts are tallied across every bout
    passed in, then each bin's count is expressed as a percentage of either just
    the events landing inside a valid ISFS cycle (``denominator="in_cycle"``, the
    default -- "Not ISFS" events are excluded from the calculation entirely, so
    the 8 percentages sum to 100) or the **total** number of events detected
    (``denominator="total"``, the reference's own denominator, so the 8
    percentages do not necessarily sum to 100).

    Per the reference, a participant with fewer than ``min_events`` events total
    is excluded rather than reported on a near-empty denominator.

    Args:
        bouts_data: Iterable of ``(t, phase_bins, event_times)``, one entry per
            bout, all three on that bout's own time base (e.g. bout-relative
            seconds): ``t`` and ``phase_bins`` as passed to/returned by
            :func:`isfs_phase_bins` for that bout, ``event_times`` this bout's
            slice of event onset/peak times on the same time base.
        min_events: Minimum total event count required to return a result.
        denominator: ``"in_cycle"`` (default) or ``"total"`` (matches the
            reference) -- which count ``pct`` is expressed against.

    Returns:
        ``None`` if fewer than ``min_events`` events total (participant
        excluded), else a dict with ``counts`` (``(8,)`` int, per phase bin 1-8),
        ``pct`` (``(8,)`` float, ``100 * counts / n_total`` or
        ``100 * counts / n_in_isfs`` per ``denominator``, all-NaN if that
        denominator is zero), ``n_in_isfs`` (events landing in a valid cycle,
        i.e. ``counts.sum()``), and ``n_total`` (every event across every bout
        passed in).
    """
    if denominator not in ("total", "in_cycle"):
        raise ValueError(f"denominator must be 'total' or 'in_cycle', got {denominator!r}")

    counts = np.zeros(8, dtype=int)
    n_total = 0
    for t, phase_bins, event_times in bouts_data:
        t = _to_1d(t, name="t")
        phase_bins = np.asarray(phase_bins).ravel()
        event_times = np.asarray(event_times, dtype=float).ravel()
        n_total += event_times.size
        if event_times.size == 0 or t.size == 0:
            continue

        idx = _nearest_sample_index(t, event_times)
        bins_for_events = phase_bins[idx]
        for b in range(1, 9):
            counts[b - 1] += int((bins_for_events == b).sum())

    if n_total < min_events:
        return None

    n_in_isfs = int(counts.sum())
    denom = n_total if denominator == "total" else n_in_isfs
    pct = 100.0 * counts / denom if denom > 0 else np.full(8, np.nan)

    return dict(counts=counts, pct=pct, n_in_isfs=n_in_isfs, n_total=n_total)


def isfs_phase_curve(bouts_data):
    """Canonical ISFS phase curve: mean standardized amplitude in each of the 8
    phase bins, across every valid ISFS cycle passed in.

    A companion to :func:`isfs_event_phase_distribution` -- overlaying this on
    that function's phase-bin bar chart gives the event distribution a shape to
    read against (e.g. "bin 2 sits near the trough, bin 6 near the peak"),
    matching how the reference ``get_iso`` presents its phase-bin figures.

    Args:
        bouts_data: Iterable of ``(phase_bins, std)``, one entry per bout, both
            arrays the same length: ``phase_bins`` as returned by
            :func:`isfs_phase_bins` (``0`` = Not ISFS, ``1``-``8`` = phase bin) and
            ``std``, the same standardized (z-scored), :func:`isfs_lowpass`-
            filtered course :func:`isfs_phase_bins` was computed from.

    Returns:
        ``(8,)`` float array: the mean ``std`` value of samples in each phase bin
        1-8, pooled across every bout passed in (``nan`` for a bin with no samples
        at all, e.g. no valid cycle was found anywhere).
    """
    sums = np.zeros(8)
    counts = np.zeros(8, dtype=int)
    for phase_bins, std in bouts_data:
        phase_bins = np.asarray(phase_bins).ravel()
        std = np.asarray(std, dtype=float).ravel()
        for b in range(1, 9):
            mask = phase_bins == b
            if mask.any():
                sums[b - 1] += std[mask].sum()
                counts[b - 1] += int(mask.sum())

    return np.divide(sums, counts, out=np.full(8, np.nan), where=counts > 0)


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
    "DEFAULT_ISFS_LOWPASS_HZ",
    "DEFAULT_ISFS_FILTER_ORDER",
    "DEFAULT_ISFS_TUKEY_ALPHA",
    "DEFAULT_ISFS_PERIOD",
    "DEFAULT_ISFS_MIN_EVENTS",
    "eeg_envelope",
    "infraslow_spectrum",
    "relative_power",
    "baseline_correct",
    "band_power_spectrum",
    "bigaussian",
    "fit_isfs",
    "isfs_lowpass",
    "isfs_phase_bins",
    "isfs_event_phase_distribution",
    "isfs_phase_curve",
    "chromatogram_peak_area",
]
