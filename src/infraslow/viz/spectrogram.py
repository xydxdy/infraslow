"""Full-night spectrogram plotting for loaded PSG recordings.

A thin adapter over :func:`yasa.plot_spectrogram` that fits this repo's
:class:`~infraslow.io.psg_loader.BioserenityPSGLoader`: it pulls one channel's
signal (and, when available, the subject's hypnogram) straight off the loader and
hands them to YASA. Kept in the ``viz`` layer so the processing/io code carries no
matplotlib dependency.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

from ..processing.detection import (
    DEFAULT_EPOCH_SEC,
    DEFAULT_STAGE_MAP,
    _build_sample_hypno,
)
from .utils import (
    BAND_ALPHA,
    DEFAULT_FIGSIZE,
    PRIMARY,
    error_band,
    seaborn_theme,
    style_axes,
)


def plot_spectrogram(
    loader: Any,
    *,
    channel: Optional[str] = None,
    sf: Optional[float] = None,
    hypno: bool = True,
    epoch_sec: float = DEFAULT_EPOCH_SEC,
    stage_map: Mapping[str, int] = DEFAULT_STAGE_MAP,
    stage_column: str = "stage",
    win_sec: float = 30.0,
    fmin: float = 0.5,
    fmax: float = 25.0,
    trimperc: float = 2.5,
    cmap: str = "Spectral_r",
    **kwargs: Any,
):
    """Plot a full-night spectrogram for one channel of a loaded recording.

    Thin adapter over :func:`yasa.plot_spectrogram`: it reads one EEG channel and
    the sampling rate off ``loader``, optionally overlays the subject's hypnogram,
    and returns YASA's figure. Mirrors the loader-driven workflow of
    :func:`~infraslow.processing.detection.spindles_detect`.

    Args:
        loader: A loaded (or loadable)
            :class:`~infraslow.io.psg_loader.BioserenityPSGLoader` -- anything
            exposing ``get_channel``/``channel_names``, ``sf``, ``annotations``
            and ``is_loaded``/``load()``. Loaded in place if not already.
        channel: Canonical channel to plot. Defaults to the loader's first
            resolved channel.
        sf: Sampling frequency (Hz). Defaults to ``loader.sf``.
        hypno: Overlay the subject's hypnogram (from ``loader.annotations``) when
            available. Set ``False`` to plot the spectrogram alone.
        epoch_sec, stage_map, stage_column: Control how the per-epoch hypnogram is
            upsampled to a sample-resolution stage array for the overlay (same
            contract as :func:`~infraslow.processing.detection.spindles_detect`).
        win_sec, fmin, fmax, trimperc, cmap, **kwargs: Passed straight through to
            :func:`yasa.plot_spectrogram`.

    Returns:
        The :class:`matplotlib.figure.Figure` produced by YASA.

    Raises:
        ImportError: if YASA is not installed.
        ValueError: if the loader exposes no channels, or ``sf`` is unavailable.
    """
    try:
        import yasa  # noqa: PLC0415 - lazy, optional dependency
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "plot_spectrogram requires the 'yasa' package. Install it with "
            "`pip install yasa` (in a Slurm job or interactive node, not the "
            "login node)."
        ) from exc

    # Load on demand so a freshly-built loader can be passed straight in.
    if hasattr(loader, "is_loaded") and not loader.is_loaded:
        loader.load()

    if channel is None:
        names = list(getattr(loader, "channel_names", []) or [])
        if not names:
            raise ValueError(
                "loader exposes no resolved channels; pass an explicit `channel`."
            )
        channel = names[0]

    data = loader.get_channel(channel)

    if sf is None:
        sf = getattr(loader, "sf", None)
    if sf is None:
        raise ValueError("plot_spectrogram requires a sampling frequency 'sf' (Hz).")

    # Build the sample-resolution hypnogram overlay only when asked and available.
    annotations = getattr(loader, "annotations", None)
    sample_hypno = None
    if hypno and annotations is not None:
        sample_hypno = _build_sample_hypno(
            annotations,
            data=data,
            sf=sf,
            epoch_sec=epoch_sec,
            stage_map=stage_map,
            stage_column=stage_column,
        )

    # Draw under the shared seaborn theme (without a grid over the spectrogram)
    # so fonts/context match the rest of the viz layer.
    with seaborn_theme(style="ticks"):
        return yasa.plot_spectrogram(
            data,
            sf,
            hypno=sample_hypno,
            win_sec=win_sec,
            fmin=fmin,
            fmax=fmax,
            trimperc=trimperc,
            cmap=cmap,
            **kwargs,
        )


def _resolve_channel_signal(
    loader: Any, channel: Optional[str], sf: Optional[float]
) -> Tuple[np.ndarray, float, str]:
    """Pull ``(data, sf, channel)`` for one subject off a (loadable) loader."""
    if hasattr(loader, "is_loaded") and not loader.is_loaded:
        loader.load()
    if channel is None:
        names = list(getattr(loader, "channel_names", []) or [])
        if not names:
            raise ValueError(
                "loader exposes no resolved channels; pass an explicit `channel`."
            )
        channel = names[0]
    data = np.asarray(loader.get_channel(channel), dtype=float)
    if sf is None:
        sf = getattr(loader, "sf", None)
    if sf is None:
        raise ValueError(
            "plot_spectrogram_grand_average requires a sampling frequency 'sf' (Hz)."
        )
    return data, float(sf), channel


def _subject_spectrogram_db(
    data: np.ndarray, sf: float, *, win_sec: float, fmin: float, fmax: float
):
    """One subject's full-recording spectrogram in dB, cropped to ``[fmin, fmax]``.

    Uses :func:`scipy.signal.spectrogram` (Hann window, ``win_sec`` segments, 50%
    overlap) and ``10*log10`` -- matching the presentation of the single-subject
    :func:`plot_spectrogram`, but returning the raw array so it can be averaged.
    Returns ``(f, t, Sxx_db)`` or ``None`` if the recording is too short.
    """
    from scipy import signal as sp_signal  # noqa: PLC0415 - lazy, optional dependency

    nperseg = max(1, int(round(win_sec * sf)))
    f, t, Sxx = sp_signal.spectrogram(
        data, fs=sf, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend=False
    )
    keep = (f >= fmin) & (f <= fmax)
    f, Sxx = f[keep], Sxx[keep]
    if f.size < 2 or t.size < 2:
        return None
    Sxx_db = 10.0 * np.log10(Sxx + np.finfo(float).tiny)
    return f, t, Sxx_db


def plot_spectrogram_grand_average(
    subjects: Union[Mapping[str, Any], Iterable[Any]],
    *,
    channel: Optional[str] = None,
    sf: Optional[float] = None,
    win_sec: float = 30.0,
    fmin: float = 0.5,
    fmax: float = 25.0,
    n_freq: int = 200,
    n_time: int = 200,
    errorbar: str = "sem",
    trimperc: float = 2.5,
    cmap: str = "Spectral_r",
    sigma_band: Tuple[float, float] = (11.0, 15.0),
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    axes: Optional[Tuple[Any, Any]] = None,
):
    """Grand-average full-night spectrogram across subjects (two panels).

    For every subject the recording's spectrogram is computed and averaged the way
    the other grand averages are -- **over epochs first, then across subjects**:

    * **Top (2D):** each subject's spectrogram is put on one frequency grid and its
      time axis resampled to a common *relative* time axis (0-1 = start-to-end of
      the recording, so different night lengths line up), then the maps are averaged
      across subjects into a grand-average spectrogram.
    * **Bottom (1D):** each subject's spectrogram is averaged over its own epochs
      (time bins) into one mean power spectrum; those per-subject spectra are then
      averaged across subjects, with a shaded ``errorbar`` band computed *across
      subjects*.

    Every subject is weighted equally regardless of recording length. Power is in dB
    throughout.

    Args:
        subjects: A ``{subject_id: loader}`` mapping or an iterable of loaders --
            each a loaded (or loadable)
            :class:`~infraslow.io.psg_loader.BioserenityPSGLoader`.
        channel: Channel to analyse (defaults to each loader's first channel).
        sf: Sampling rate (Hz); defaults to each loader's ``sf``.
        win_sec: Spectrogram window length (s).
        fmin, fmax: Frequency range to keep and grid over (Hz).
        n_freq, n_time: Common grid resolution (frequency bins / relative-time bins).
        errorbar: Across-subjects band on the 1D panel -- ``"sem"`` or ``"std"``.
        trimperc: Percentile trim for the spectrogram colour limits (as in
            :func:`plot_spectrogram`).
        cmap: Spectrogram colormap.
        sigma_band: Band bracketed on both panels (Hz).
        figsize: Size of a newly created figure (ignored when ``axes`` is passed).
        axes: Optional ``(ax_top, ax_bottom)`` to draw on; a new 2-panel figure is
            created when ``None``.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: if ``errorbar`` is invalid or no subject produced a spectrogram.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - lazy, optional dependency

    if errorbar not in {"sem", "std"}:
        raise ValueError("errorbar must be 'sem' or 'std'.")

    items = subjects.values() if isinstance(subjects, Mapping) else subjects
    fg = np.linspace(fmin, fmax, n_freq)
    tg = np.linspace(0.0, 1.0, n_time)

    maps_2d: list = []      # per subject: (n_freq, n_time), time-warped to 0-1
    specs_1d: list = []     # per subject: (n_freq,), epoch-averaged spectrum
    for loader in items:
        data, s, _chan = _resolve_channel_signal(loader, channel, sf)
        out = _subject_spectrogram_db(data, s, win_sec=win_sec, fmin=fmin, fmax=fmax)
        if out is None:
            continue
        f_i, _t_i, Sxx_db = out
        # Interpolate onto the common frequency grid (one column per time bin).
        Sxx_fg = np.column_stack([np.interp(fg, f_i, Sxx_db[:, j]) for j in range(Sxx_db.shape[1])])
        # 1D: average over this subject's epochs (time bins) -> mean spectrum.
        specs_1d.append(Sxx_fg.mean(axis=1))
        # 2D: resample the time axis to the common relative-time grid.
        rel_i = np.linspace(0.0, 1.0, Sxx_fg.shape[1])
        maps_2d.append(np.vstack([np.interp(tg, rel_i, Sxx_fg[r]) for r in range(fg.size)]))

    if not maps_2d:
        raise ValueError("No subject produced a spectrogram to average.")

    grand_2d = np.mean(maps_2d, axis=0)
    stack_1d = np.vstack(specs_1d)
    grand_1d = stack_1d.mean(axis=0)
    band_1d = error_band(stack_1d, errorbar)
    n_subj = len(maps_2d)

    with seaborn_theme(style="ticks"):
        if axes is not None:
            ax1, ax2 = axes
            fig = ax1.figure
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            fig.subplots_adjust(hspace=0.32)

        vmin, vmax = np.nanpercentile(grand_2d, [trimperc, 100.0 - trimperc])
        mesh = ax1.pcolormesh(
            tg, fg, grand_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True
        )
        for y in sigma_band:
            ax1.axhline(y, color="w", ls=":", lw=1.4, alpha=0.85)
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_xlabel("Relative time (fraction of recording)")
        ax1.set_title(f"Grand-average spectrogram (n={n_subj} subjects)", weight="bold")
        ax1.grid(False)
        cax = ax1.inset_axes([1.015, 0.0, 0.018, 1.0])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Power (dB)", fontsize="small")
        cbar.ax.tick_params(labelsize="x-small")

        ax2.fill_between(
            fg, grand_1d - band_1d, grand_1d + band_1d, color=PRIMARY, alpha=BAND_ALPHA, lw=0
        )
        ax2.plot(
            fg, grand_1d, color=PRIMARY, lw=2.5,
            label=f"Grand average (n={n_subj}) ± {errorbar.upper()}",
        )
        ax2.axvspan(
            sigma_band[0], sigma_band[1], color="0.6", alpha=0.15, zorder=0,
            label=f"Sigma band ({sigma_band[0]:g}-{sigma_band[1]:g} Hz)",
        )
        ax2.set_xlim(fmin, fmax)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Power (dB)")
        ax2.set_title("Mean power spectrum across subjects", weight="bold")
        ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
        style_axes(ax2)
    return fig


__all__ = ["plot_spectrogram", "plot_spectrogram_grand_average"]
