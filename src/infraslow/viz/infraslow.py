"""Power-frequency plots for infraslow / spindle-sigma spectra.

Renders the spectra produced by :mod:`infraslow.processing.infraslow` so two
detectors (YASA vs. Luna) can be overlaid:

* :func:`plot_spectra` -- the general overlay: several labelled ``(freq, psd)``
  curves on one axis, with optional band shading and per-curve peak markers.
* :func:`plot_infraslow_spectra` -- thin wrapper for the low-frequency
  (~0.02 Hz) sigma-power / spindle-rate spectra (shades the infraslow band,
  marks each curve's in-band peak).
* :func:`plot_spindle_sigma_spectra` -- thin wrapper for the average
  spindle-epoch spectra (shades the sigma band).

Uses the shared seaborn theme from :mod:`infraslow.viz.utils`, so these match the
rest of the viz layer. Kept dependency-lazy (matplotlib/seaborn imported inside).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .utils import (
    BAND_ALPHA,
    DEFAULT_FIGSIZE,
    SUBJECT_GREY,
    error_band,
    seaborn_theme,
    style_axes,
)

# A curve is a labelled spectrum: an ``(freqs, psd)`` pair or anything exposing
# ``.freqs``/``.psd`` (e.g. an ``InfraslowSpectrum``).
Curve = Union[Tuple[Any, Any], Any]
Curves = Union[Mapping[str, Curve], Sequence[Tuple[str, Curve]]]


def _as_fp(curve: Curve) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce a curve to ``(freqs, psd)`` arrays."""
    if hasattr(curve, "freqs") and hasattr(curve, "psd"):
        f, p = curve.freqs, curve.psd
    else:
        f, p = curve
    return np.asarray(f, dtype=float), np.asarray(p, dtype=float)


def _iter_curves(curves: Curves):
    """Yield ``(label, freqs, psd)`` from a mapping or a sequence of pairs."""
    items = curves.items() if isinstance(curves, Mapping) else curves
    for label, curve in items:
        f, p = _as_fp(curve)
        yield str(label), f, p


def _peak_in(f: np.ndarray, p: np.ndarray, band: Optional[Tuple[float, float]]):
    """``(peak_freq, peak_power)`` of the largest PSD value, optionally within ``band``."""
    mask = np.ones_like(f, dtype=bool) if band is None else (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return None
    idx = np.argmax(np.where(mask, p, -np.inf))
    return float(f[idx]), float(p[idx])


def plot_spectra(
    curves: Curves,
    *,
    band: Optional[Tuple[float, float]] = None,
    band_label: Optional[str] = None,
    mark_peaks: bool = False,
    peak_in_band: bool = True,
    logy: bool = True,
    logx: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Power spectral density",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    ax: Optional[Any] = None,
):
    """Overlay several labelled power spectra on one axis.

    Args:
        curves: The spectra to draw -- a ``{label: curve}`` mapping or a sequence
            of ``(label, curve)`` pairs, where each ``curve`` is an
            ``(freqs, psd)`` pair or an object exposing ``.freqs``/``.psd``
            (e.g. an :class:`~infraslow.processing.infraslow.InfraslowSpectrum`).
        band: If given, shade ``(lo, hi)`` Hz to highlight the band of interest.
        band_label: Legend label for the shaded band.
        mark_peaks: Mark each curve's peak with a dot and a dropline.
        peak_in_band: When marking peaks, restrict the peak search to ``band``
            (ignored if ``band`` is ``None``).
        logy, logx: Log-scale the respective axes (``logy`` on by default -- PSDs
            span orders of magnitude).
        xlim: x-axis limits (Hz).
        xlabel, ylabel, title: Axis labels and title.
        figsize: Size of a newly created figure (ignored when ``ax`` is passed).
        ax: Existing Axes to draw on; a new one is created when ``None``.

    Returns:
        The matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - lazy, optional dependency
    import seaborn as sns  # noqa: PLC0415 - lazy, optional dependency

    rows = list(_iter_curves(curves))
    if not rows:
        raise ValueError("plot_spectra received no curves.")
    colors = sns.color_palette(n_colors=len(rows))

    with seaborn_theme():
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if band is not None:
            ax.axvspan(
                band[0],
                band[1],
                color="0.6",
                alpha=0.15,
                zorder=0,
                label=band_label or f"{band[0]:g}-{band[1]:g} Hz",
            )

        peak_band = band if peak_in_band else None
        for (label, f, p), color in zip(rows, colors):
            ax.plot(f, p, color=color, lw=2.2, label=label, zorder=3)
            if mark_peaks:
                peak = _peak_in(f, p, peak_band)
                if peak is not None:
                    pf, pp = peak
                    ax.plot([pf], [pp], marker="o", ms=7, color=color, zorder=4)
                    ax.annotate(
                        f"{pf:.3g} Hz",
                        xy=(pf, pp),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize="small",
                        color=color,
                    )

        if logy:
            ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, weight="bold")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        style_axes(ax)
    return ax


def plot_infraslow_spectra(
    spectra: Curves,
    *,
    # Mirrors processing.infraslow.DEFAULT_INFRASLOW_BAND (kept a literal so
    # importing viz doesn't pull in scipy); 0.001-0.1 Hz around the ~0.02 Hz
    # (50 s) rhythm.
    band: Tuple[float, float] = (0.001, 0.1),
    xlim: Tuple[float, float] = (0.0, 0.12),
    mark_peaks: bool = True,
    title: Optional[str] = "Infraslow oscillation of sigma power",
    ax: Optional[Any] = None,
    **kwargs: Any,
):
    """Overlay infraslow (~0.02 Hz) spectra, shading the infraslow band.

    Convenience wrapper over :func:`plot_spectra` for the low-frequency view;
    ``spectra`` is typically ``{"YASA": spec_yasa, "Luna": spec_luna,
    "Sigma power": spec_continuous}`` of
    :class:`~infraslow.processing.infraslow.InfraslowSpectrum` objects.
    """
    return plot_spectra(
        spectra,
        band=band,
        band_label=f"Infraslow band ({band[0]:g}-{band[1]:g} Hz)",
        mark_peaks=mark_peaks,
        xlim=xlim,
        xlabel="Frequency (Hz)",
        ylabel="Power spectral density (a.u.$^2$/Hz)",
        title=title,
        ax=ax,
        **kwargs,
    )


def plot_spindle_sigma_spectra(
    spectra: Curves,
    *,
    sigma_band: Tuple[float, float] = (12.0, 15.0),
    xlim: Tuple[float, float] = (0.0, 20.0),
    logy: bool = False,
    title: Optional[str] = "Spindle-epoch power spectrum",
    ax: Optional[Any] = None,
    **kwargs: Any,
):
    """Overlay average spindle-epoch spectra, shading the sigma band.

    Convenience wrapper over :func:`plot_spectra` for the sigma-band view;
    ``spectra`` is typically ``{"YASA": (f, psd), "Luna": (f, psd)}`` from
    :func:`~infraslow.processing.infraslow.average_spindle_spectrum`.
    """
    return plot_spectra(
        spectra,
        band=sigma_band,
        band_label=f"Sigma band ({sigma_band[0]:g}-{sigma_band[1]:g} Hz)",
        mark_peaks=True,
        logy=logy,
        xlim=xlim,
        xlabel="Frequency (Hz)",
        ylabel="Power spectral density (µV$^2$/Hz)",
        title=title,
        ax=ax,
        **kwargs,
    )


def _subject_fps(subject_curves: Curves):
    """List of ``(freqs, psd)`` from a ``{subject: curve}`` mapping or curve iterable."""
    items = (
        subject_curves.values() if isinstance(subject_curves, Mapping) else subject_curves
    )
    fps = [_as_fp(c) for c in items]
    if not fps:
        raise ValueError("A subject group has no spectra to average.")
    return fps


def _stack_on_grid(fps):
    """Interpolate per-subject ``(f, p)`` spectra onto one common frequency grid.

    The grid is the finest input frequency axis clipped to the range every subject
    covers, so each subject contributes a value at every grid point (no NaNs).
    Returns ``(grid, stack)`` with ``stack`` shaped ``(n_subjects, n_grid)``.
    """
    grid = max((f for f, _ in fps), key=len)
    lo = max(float(f.min()) for f, _ in fps)
    hi = min(float(f.max()) for f, _ in fps)
    grid = grid[(grid >= lo) & (grid <= hi)]
    if grid.size == 0:
        raise ValueError("Subjects share no common frequency range to average over.")
    stack = np.vstack([np.interp(grid, f, p) for f, p in fps])
    return grid, stack


def plot_spectra_grand_average(
    groups: Union[Mapping[str, Curves], Sequence[Tuple[str, Curves]]],
    *,
    band: Optional[Tuple[float, float]] = None,
    band_label: Optional[str] = None,
    errorbar: str = "sem",
    show_subjects: bool = False,
    mark_peaks: bool = False,
    peak_in_band: bool = True,
    logy: bool = True,
    logx: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Power spectral density",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    ax: Optional[Any] = None,
):
    """Overlay grand-average spectra: within each group, average subjects' spectra.

    Every *group* is a set of per-subject spectra, each already averaged over that
    subject's epochs (e.g. one
    :class:`~infraslow.processing.infraslow.InfraslowSpectrum` per subject -- its
    Welch PSD *is* the epoch average). The subjects in a group are interpolated
    onto one frequency grid and averaged into a grand-mean curve with a shaded
    ``errorbar`` band computed *across subjects*, so each subject is weighted
    equally regardless of recording length. Group-level counterpart of
    :func:`plot_spectra` (which overlays single curves).

    Args:
        groups: ``{group_label: subject_curves}`` (or a sequence of
            ``(group_label, subject_curves)`` pairs), where ``subject_curves`` is a
            ``{subject_id: curve}`` mapping or an iterable of curves -- each an
            ``(freqs, psd)`` pair or an object exposing ``.freqs``/``.psd``.
            Typically one group per detector, e.g.
            ``{"YASA": yasa_specs, "Luna": luna_specs}``.
        errorbar: Across-subjects band -- ``"sem"`` or ``"std"``.
        show_subjects: Overlay each subject's spectrum faintly in grey.
        band, band_label, mark_peaks, peak_in_band, logy, logx, xlim, xlabel,
        ylabel, title, figsize, ax: As in :func:`plot_spectra`.

    Returns:
        The matplotlib ``Axes``.

    Raises:
        ValueError: if ``groups`` is empty, ``errorbar`` is invalid, or a group's
            subjects share no common frequency range.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - lazy, optional dependency
    import seaborn as sns  # noqa: PLC0415 - lazy, optional dependency

    if errorbar not in {"sem", "std"}:
        raise ValueError("errorbar must be 'sem' or 'std'.")
    rows = list(groups.items() if isinstance(groups, Mapping) else groups)
    if not rows:
        raise ValueError("plot_spectra_grand_average received no groups.")
    colors = sns.color_palette(n_colors=len(rows))
    peak_band = band if peak_in_band else None
    tiny = np.finfo(float).tiny

    with seaborn_theme():
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if band is not None:
            ax.axvspan(
                band[0],
                band[1],
                color="0.6",
                alpha=0.15,
                zorder=0,
                label=band_label or f"{band[0]:g}-{band[1]:g} Hz",
            )

        for (label, subject_curves), color in zip(rows, colors):
            grid, stack = _stack_on_grid(_subject_fps(subject_curves))
            n_subj = stack.shape[0]
            mean = stack.mean(axis=0)
            half = error_band(stack, errorbar)

            if show_subjects:
                for row in stack:
                    ax.plot(grid, row, color=SUBJECT_GREY, lw=0.8, alpha=0.6, zorder=1)

            lower = mean - half
            if logy:  # keep the band off the non-positive part of a log axis
                lower = np.clip(lower, tiny, None)
            ax.fill_between(grid, lower, mean + half, color=color, alpha=BAND_ALPHA, lw=0, zorder=2)
            ax.plot(
                grid,
                mean,
                color=color,
                lw=2.4,
                zorder=3,
                label=f"{label} (n={n_subj}) ± {errorbar.upper()}",
            )
            if mark_peaks:
                peak = _peak_in(grid, mean, peak_band)
                if peak is not None:
                    pf, pp = peak
                    ax.plot([pf], [pp], marker="o", ms=7, color=color, zorder=4)
                    ax.annotate(
                        f"{pf:.3g} Hz",
                        xy=(pf, pp),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize="small",
                        color=color,
                    )

        if logy:
            ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, weight="bold")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        style_axes(ax)
    return ax


def plot_infraslow_spectra_grand_average(
    groups: Union[Mapping[str, Curves], Sequence[Tuple[str, Curves]]],
    *,
    band: Tuple[float, float] = (0.005, 0.03),
    xlim: Tuple[float, float] = (0.0, 0.1),
    mark_peaks: bool = True,
    errorbar: str = "sem",
    title: Optional[str] = "Infraslow oscillation of sigma power (grand average)",
    ax: Optional[Any] = None,
    **kwargs: Any,
):
    """Grand-average infraslow (~0.02 Hz) spectra across subjects, shading the band.

    Grand-average wrapper over :func:`plot_spectra_grand_average` for the
    low-frequency view; ``groups`` is typically
    ``{"YASA": {sid: spec, ...}, "Luna": {sid: spec, ...}}`` of per-subject
    :class:`~infraslow.processing.infraslow.InfraslowSpectrum` objects.
    """
    return plot_spectra_grand_average(
        groups,
        band=band,
        band_label=f"Infraslow band ({band[0]:g}-{band[1]:g} Hz)",
        mark_peaks=mark_peaks,
        errorbar=errorbar,
        xlim=xlim,
        xlabel="Frequency (Hz)",
        ylabel="Power spectral density (a.u.$^2$/Hz)",
        title=title,
        ax=ax,
        **kwargs,
    )


def plot_spindle_sigma_spectra_grand_average(
    groups: Union[Mapping[str, Curves], Sequence[Tuple[str, Curves]]],
    *,
    sigma_band: Tuple[float, float] = (12.0, 15.0),
    xlim: Tuple[float, float] = (0.0, 20.0),
    logy: bool = False,
    errorbar: str = "sem",
    title: Optional[str] = "Spindle-epoch power spectrum (grand average)",
    ax: Optional[Any] = None,
    **kwargs: Any,
):
    """Grand-average spindle-epoch spectra across subjects, shading the sigma band.

    Grand-average wrapper over :func:`plot_spectra_grand_average` for the
    sigma-band view; ``groups`` is typically ``{"YASA": {sid: (f, psd), ...},
    "Luna": {...}}`` from
    :func:`~infraslow.processing.infraslow.average_spindle_spectrum`.
    """
    return plot_spectra_grand_average(
        groups,
        band=sigma_band,
        band_label=f"Sigma band ({sigma_band[0]:g}-{sigma_band[1]:g} Hz)",
        mark_peaks=True,
        logy=logy,
        errorbar=errorbar,
        xlim=xlim,
        xlabel="Frequency (Hz)",
        ylabel="Power spectral density (µV$^2$/Hz)",
        title=title,
        ax=ax,
        **kwargs,
    )


__all__ = [
    "plot_spectra",
    "plot_infraslow_spectra",
    "plot_spindle_sigma_spectra",
    "plot_spectra_grand_average",
    "plot_infraslow_spectra_grand_average",
    "plot_spindle_sigma_spectra_grand_average",
]
