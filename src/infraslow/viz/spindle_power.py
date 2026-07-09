"""Sigma-band relative-power plots time-locked to detected spindles.

Reproduces YASA's STFT relative-power view (a spectrogram on top, the sigma-band
relative-power trace below) -- but instead of one continuous stretch of signal it
**averages the time-frequency map across spindle events**:

* :func:`plot_spindle_sigma_power` -- one subject: epoch the STFT map around every
  detected spindle and average those epochs.
* :func:`plot_spindles_sigma_power_grand_average` -- many subjects: average within
  each subject first, then average those per-subject maps across subjects so every
  subject is weighted equally (mirrors :func:`plot_spindles_grand_average`).

The relative power is computed as in the YASA tutorial, after band-passing the
signal to the broadband range with :class:`~infraslow.processing.signal.ButterFilter`::

    data_broad = ButterFilter(sf, [1, 30], mode="band")(data)
    f, t, Sxx = yasa.stft_power(data_broad, sf, window=2, step=.2, band=freq_broad,
                                norm=True, interp=True)
    rel_pow = Sxx[(f >= 11) & (f <= 16)].sum(0)

Kept in the ``viz`` layer so the processing/io code carries no matplotlib
dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .helpers import spindle_event_times
from .utils import BAND_ALPHA, DEFAULT_FIGSIZE, PRIMARY, error_band, seaborn_theme, style_axes

logger = logging.getLogger(__name__)

DEFAULT_FREQ_BROAD: Tuple[float, float] = (1.0, 30.0)
DEFAULT_SIGMA_BAND: Tuple[float, float] = (12.0, 15.0)


def _resolve_signal_and_peaks(
    loader: Any, spindles: Any, channel: Optional[str], mark: str
):
    """Pull ``(data, sf, peaks_sec, channel)`` for one subject, or ``None``.

    ``peaks_sec`` are the spindle event times (seconds from recording start) read
    from ``spindles.summary()``; ``data`` is the matching channel's signal off the
    loader. Returns ``None`` when the subject has no spindles.
    """
    if spindles is None:
        return None
    summary = spindles.summary() if hasattr(spindles, "summary") else spindles
    if summary is None or len(summary) == 0:
        return None

    has_chan = hasattr(summary, "columns") and "Channel" in summary.columns
    # Default to the (first) channel the spindles were detected on, then pull that
    # channel's event times via the shared helper.
    if channel is None and has_chan:
        channel = str(summary["Channel"].iloc[0])
    peaks = spindle_event_times(summary, mark=mark, channel=channel)
    if peaks.size == 0:
        return None

    if hasattr(loader, "is_loaded") and not loader.is_loaded:
        loader.load()
    if channel is None:
        names = list(getattr(loader, "channel_names", []) or [])
        if not names:
            raise ValueError(
                "Cannot determine which channel to use; pass `channel` explicitly."
            )
        channel = names[0]
    data = np.asarray(loader.get_channel(channel), dtype=float)
    sf = getattr(loader, "sf", None)
    if sf is None:
        raise ValueError("loader has no sampling rate 'sf'; load the recording.")
    return data, float(sf), peaks, channel


def _event_locked_sigma_power(
    data: np.ndarray,
    sf: float,
    peaks: np.ndarray,
    *,
    time_before: float,
    time_after: float,
    window: float,
    step: float,
    freq_broad: Tuple[float, float],
    sigma_band: Tuple[float, float],
):
    """STFT the recording once, epoch the map around each peak, average over events.

    Returns ``(f, rel_t, mean_Sxx, relpow_stack, n_used)`` -- the frequency vector,
    the relative-time axis centred on the spindle, the event-averaged spectrogram
    ``(n_freq, n_relt)``, the per-event sigma-band relative-power traces
    ``(n_events, n_relt)`` (un-averaged, for an error band), and how many spindles
    contributed -- or ``None`` if no event fit a full window.
    """
    import yasa  # noqa: PLC0415 - lazy, optional dependency

    from ..processing.signal import ButterFilter  # noqa: PLC0415 - lazy, avoids cycle

    # Band-pass to the broadband range first (the YASA tutorial's ``data_broad``):
    # a zero-phase Butterworth before the STFT.
    data_broad = ButterFilter(sf, list(freq_broad), mode="band")(data)
    f, t, Sxx = yasa.stft_power(
        data_broad, sf, window=window, step=step, band=freq_broad, norm=True, interp=True
    )
    f = np.asarray(f)
    t = np.asarray(t)
    if t.size < 2:
        return None

    dt = float(np.median(np.diff(t)))
    n_before = int(round(time_before / dt))
    n_after = int(round(time_after / dt))
    rel_t = np.arange(-n_before, n_after + 1) * dt
    idx_sigma = np.logical_and(f >= sigma_band[0], f <= sigma_band[1])

    # Epoch the precomputed time-frequency map around each spindle peak. Bins are
    # uniformly spaced, so the nearest bin to a peak is a direct index.
    n_bins = Sxx.shape[1]
    t0 = float(t[0])
    patches = []
    for p in peaks:
        i = int(round((float(p) - t0) / dt))
        lo, hi = i - n_before, i + n_after + 1
        if lo < 0 or hi > n_bins:
            continue  # event window runs off a recording edge -> skip
        patches.append(Sxx[:, lo:hi])
    if not patches:
        return None

    patches_arr = np.asarray(patches)                       # (n_events, n_f, n_relt)
    mean_Sxx = patches_arr.mean(axis=0)                     # (n_freq, n_relt)
    # Per-event sigma-band relative-power trace, kept un-averaged so the caller
    # can compute a spread (error band) across events.
    relpow_stack = patches_arr[:, idx_sigma, :].sum(axis=1)  # (n_events, n_relt)
    return f, rel_t, mean_Sxx, relpow_stack, len(patches)


def _make_axes(axes, plt, figsize):
    if axes is not None:
        ax1, ax2 = axes
        return ax1.figure, ax1, ax2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.25)
    return fig, ax1, ax2


def _draw(
    ax1,
    ax2,
    f,
    rel_t,
    Sxx,
    relpow_mean,
    relpow_band,
    *,
    sigma_band,
    vmax,
    threshold,
    cmap,
    errorbar,
    title,
):
    """Render the two-panel figure: spectrogram on top, sigma rel-power below.

    The colorbar is drawn in an inset anchored just outside the right edge of the
    top axes, so the spectrogram keeps the exact width of the relative-power panel
    below it (a normal colorbar would shrink only the top axes and break the
    shared-x alignment).
    """
    mesh = ax1.pcolormesh(rel_t, f, Sxx, cmap=cmap, vmax=vmax, shading="auto", rasterized=True)
    # Bracket the sigma band on the spectrogram (white reads on any colormap).
    for y in sigma_band:
        ax1.axhline(y, color="w", ls=":", lw=1.4, alpha=0.85)
    ax1.axvline(0, color="w", ls="--", lw=1.4, alpha=0.9)
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(title, weight="bold")
    ax1.grid(False)
    # Slim colorbar in the right margin -- placed via inset so ax1 keeps ax2's width.
    cax = ax1.inset_axes([1.015, 0.0, 0.018, 1.0])
    cbar = ax1.figure.colorbar(mesh, cax=cax)
    cbar.set_label("Relative power", fontsize="small")
    cbar.ax.tick_params(labelsize="x-small")

    ax2.plot(rel_t, relpow_mean, color=PRIMARY, lw=2.5, label="Mean")
    ax2.fill_between(
        rel_t,
        relpow_mean - relpow_band,
        relpow_mean + relpow_band,
        color=PRIMARY,
        alpha=BAND_ALPHA,
        lw=0,
        label=errorbar.upper(),
    )
    ax2.axvline(0, color="0.35", ls="--", lw=1.2)
    ax2.set_ylabel("Relative power (% $uV^2$)")
    ax2.set_xlim(rel_t[0], rel_t[-1])
    ax2.set_xlabel("Time relative to spindle (s)")
    if threshold is not None:
        ax2.axhline(
            threshold, ls=":", lw=2, color="indianred", label=f"Threshold ({threshold})"
        )
    ax2.set_title("Relative power in the sigma band", weight="bold")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    style_axes(ax2)


def plot_spindle_sigma_power(
    loader: Any,
    spindles: Any,
    *,
    channel: Optional[str] = None,
    mark: str = "Peak",
    time_before: float = 1.5,
    time_after: float = 1.5,
    window: float = 2.0,
    step: float = 0.2,
    freq_broad: Tuple[float, float] = DEFAULT_FREQ_BROAD,
    sigma_band: Tuple[float, float] = DEFAULT_SIGMA_BAND,
    errorbar: str = "sem",
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    vmax: Optional[float] = None,
    threshold: Optional[float] = None,
    cmap: str = "Spectral_r",
    axes: Optional[Tuple[Any, Any]] = None,
):
    """Spindle-locked sigma relative power, averaged over **one subject's** spindles.

    Computes YASA's STFT relative-power map across the channel, then averages the
    map over a ``[-time_before, +time_after]`` window centred on every detected
    spindle (instead of showing a single epoch). The top panel is the averaged
    spectrogram; the bottom panel is the averaged sigma-band relative power with a
    shaded error band (``errorbar``) computed *across that subject's spindles*.

    Args:
        loader: A loaded (or loadable)
            :class:`~infraslow.io.psg_loader.BioserenityPSGLoader` providing the
            channel signal and ``sf``.
        spindles: The matching YASA ``SpindlesResults`` (or its ``summary()``
            DataFrame) for this recording -- the source of spindle event times.
        channel: Channel to analyse. Defaults to the channel the spindles were
            detected on (first ``Channel`` in the summary).
        mark: Spindle summary column to centre on (``"Peak"``/``"Start"``/``"End"``).
        time_before, time_after: Half-window (seconds) averaged around each spindle.
        window, step: STFT window/step (seconds) passed to ``yasa.stft_power``.
        freq_broad: Broadband range the relative power is normalised within.
        sigma_band: Band summed for the relative-power trace (default 11-16 Hz).
        errorbar: Shaded band on the rel-power panel -- ``"sem"`` or ``"std"``
            across the subject's spindles.
        figsize: Size of a newly created figure (ignored when ``axes`` is passed).
            Defaults to the shared :data:`~infraslow.viz.utils.DEFAULT_FIGSIZE`.
        vmax: Spectrogram colour ceiling (``None`` autoscales -- averaged maps peak
            far below the single-epoch ``0.2`` of the YASA example).
        threshold: If given, draw a dashed reference line on the rel-power panel.
        cmap: Spectrogram colormap.
        axes: Optional ``(ax_top, ax_bottom)`` to draw on; a new 2-panel figure is
            created when ``None``.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ImportError: if YASA is not installed.
        ValueError: if ``errorbar`` is invalid, the subject has no spindles, or
            none fit a full window.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - lazy, optional dependency

    if errorbar not in {"sem", "std"}:
        raise ValueError("errorbar must be 'sem' or 'std'.")

    resolved = _resolve_signal_and_peaks(loader, spindles, channel, mark)
    if resolved is None:
        raise ValueError("No spindles to average for this subject.")
    data, sf, peaks, chan = resolved

    out = _event_locked_sigma_power(
        data,
        sf,
        peaks,
        time_before=time_before,
        time_after=time_after,
        window=window,
        step=step,
        freq_broad=freq_broad,
        sigma_band=sigma_band,
    )
    if out is None:
        raise ValueError(
            "No spindle fit a full window; shrink time_before/time_after."
        )
    f, rel_t, mean_Sxx, relpow_stack, n_used = out
    relpow_mean = relpow_stack.mean(axis=0)
    relpow_band = error_band(relpow_stack, errorbar)

    with seaborn_theme():
        fig, ax1, ax2 = _make_axes(axes, plt, figsize)
        _draw(
            ax1,
            ax2,
            f,
            rel_t,
            mean_Sxx,
            relpow_mean,
            relpow_band,
            sigma_band=sigma_band,
            vmax=vmax,
            threshold=threshold,
            cmap=cmap,
            errorbar=errorbar,
            title=f"Spindle-locked sigma power -- {chan}, n={n_used} spindles",
        )
    return fig


def plot_spindles_sigma_power_grand_average(
    subjects: Any,
    *,
    channel: Optional[str] = None,
    mark: str = "Peak",
    time_before: float = 1.5,
    time_after: float = 1.5,
    window: float = 2.0,
    step: float = 0.2,
    freq_broad: Tuple[float, float] = DEFAULT_FREQ_BROAD,
    sigma_band: Tuple[float, float] = DEFAULT_SIGMA_BAND,
    errorbar: str = "sem",
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    vmax: Optional[float] = None,
    threshold: Optional[float] = None,
    cmap: str = "mako",
    axes: Optional[Tuple[Any, Any]] = None,
):
    """Spindle-locked sigma relative power, **grand-averaged across subjects**.

    Each subject's spindle epochs are averaged into one per-subject map (as in
    :func:`plot_spindle_sigma_power`); those per-subject maps are then averaged
    across subjects -- every subject weighted equally regardless of how many
    spindles it had. The rel-power panel carries a shaded error band (``errorbar``)
    computed *across subjects*. Subjects with no spindles are skipped.

    Args:
        subjects: Either an iterable of ``(loader, spindles)`` pairs or a mapping
            ``{subject_id: (loader, spindles)}``. With
            :func:`~infraslow.processing.detection.detect_subjects_spindles`
            output, pair the loaders with the results, e.g.
            ``zip(loaders, results.values())``.
        errorbar: Shaded band on the rel-power panel -- ``"sem"`` or ``"std"``
            across subjects.
        channel, mark, time_before, time_after, window, step, freq_broad,
        sigma_band, figsize, vmax, threshold, cmap, axes: As in
            :func:`plot_spindle_sigma_power`.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ImportError: if YASA is not installed.
        ValueError: if ``errorbar`` is invalid, or no subject had spindles.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - lazy, optional dependency

    if errorbar not in {"sem", "std"}:
        raise ValueError("errorbar must be 'sem' or 'std'.")

    pairs = subjects.values() if isinstance(subjects, Mapping) else subjects

    per_subj_Sxx: list = []
    per_subj_relpow: list = []
    ref_f = ref_t = None
    for item in pairs:
        loader, spindles = item
        resolved = _resolve_signal_and_peaks(loader, spindles, channel, mark)
        if resolved is None:
            continue
        data, sf, peaks, _chan = resolved
        out = _event_locked_sigma_power(
            data,
            sf,
            peaks,
            time_before=time_before,
            time_after=time_after,
            window=window,
            step=step,
            freq_broad=freq_broad,
            sigma_band=sigma_band,
        )
        if out is None:
            continue
        f, rel_t, mean_Sxx, relpow_stack, _n = out
        if ref_f is None:
            ref_f, ref_t = f, rel_t
        elif mean_Sxx.shape != per_subj_Sxx[0].shape:
            # Different sf/STFT grid -> cannot average; skip with a note.
            logger.warning(
                "Skipping a subject whose time-frequency grid %s differs from the "
                "reference %s (check all subjects share one sampling rate).",
                mean_Sxx.shape,
                per_subj_Sxx[0].shape,
            )
            continue
        per_subj_Sxx.append(mean_Sxx)
        # One trace per subject: that subject's mean over its own spindles.
        per_subj_relpow.append(relpow_stack.mean(axis=0))

    if not per_subj_Sxx:
        raise ValueError("No subject had any spindles to average.")

    grand_Sxx = np.mean(per_subj_Sxx, axis=0)
    relpow_arr = np.asarray(per_subj_relpow)        # (n_subjects, n_relt)
    grand_relpow = relpow_arr.mean(axis=0)
    grand_band = error_band(relpow_arr, errorbar)

    with seaborn_theme():
        fig, ax1, ax2 = _make_axes(axes, plt, figsize)
        _draw(
            ax1,
            ax2,
            ref_f,
            ref_t,
            grand_Sxx,
            grand_relpow,
            grand_band,
            sigma_band=sigma_band,
            vmax=vmax,
            threshold=threshold,
            cmap=cmap,
            errorbar=errorbar,
            title=f"Spindle-locked sigma power -- grand average (n={len(per_subj_Sxx)} subjects)",
        )
    return fig


__all__ = [
    "plot_spindle_sigma_power",
    "plot_spindles_sigma_power_grand_average",
]
