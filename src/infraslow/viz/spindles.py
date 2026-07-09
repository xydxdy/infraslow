"""Plotting for spindle detection results.

Visualisations built on top of the YASA detection results produced by
:mod:`infraslow.processing.detection` -- kept in the ``viz`` layer so the
processing code carries no matplotlib dependency.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import pandas as pd

from ..processing.detection import SpindlesResult
from .helpers import normalize_waveform
from .utils import BAND_ALPHA, DEFAULT_FIGSIZE, PRIMARY, SUBJECT_GREY, seaborn_theme, style_axes

# seaborn spells the analytic error band as "se"/"sd"; we expose "sem"/"std".
_ERRORBAR = {"sem": "se", "std": "sd"}

_YLABEL = {
    None: "Amplitude (µV)",
    "zscore": "Amplitude (z-scored)",
    "peak": "Amplitude (peak-normalized)",
}


def plot_spindles_grand_average(
    results: Mapping[str, "SpindlesResult"],
    *,
    center: str = "Peak",
    time_before: float = 1.5,
    time_after: float = 1.5,
    errorbar: str = "sem",
    normalize: Union[bool, str] = False,
    show_subjects: bool = True,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    ax: Optional[Any] = None,
):
    """Plot the grand-average spindle waveform across subjects.

    Each subject's detected events are time-locked (via YASA's
    :meth:`get_sync_events`) and averaged into one mean waveform; those
    per-subject waveforms are then averaged into the grand mean, with a shaded
    band of ``errorbar`` (``"sem"`` or ``"std"``) computed *across subjects*. This
    mirrors YASA's ``plot_average`` but weights every subject equally rather than
    every event. Subjects with no spindles (``None``) are skipped.

    Args:
        center, time_before, time_after: Passed to ``get_sync_events`` to define
            the window locked to each spindle ``center`` (default ``"Peak"``).
        errorbar: Shaded band across subjects -- ``"sem"`` or ``"std"``.
        normalize: Normalize each subject's mean waveform *before* averaging, so
            absolute-amplitude differences between subjects don't dominate the
            grand average (compare shape, not µV). ``False`` (default) keeps raw
            µV; ``True``/``"zscore"`` z-scores each waveform; ``"peak"`` scales
            each so its peak magnitude is 1.
        show_subjects: Overlay each subject's (normalized) mean waveform in grey.
        figsize: Size of a newly created figure (ignored when ``ax`` is passed).
            Defaults to the shared :data:`~infraslow.viz.utils.DEFAULT_FIGSIZE`,
            matching the relative-power plots.
        ax: Existing Axes to draw on; a new one is created when ``None``.

    Returns:
        The matplotlib ``Axes`` (so callers can further style/save it).

    Raises:
        ValueError: if no subject had spindles, or ``errorbar``/``normalize`` is
            invalid.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - lazy, optional dependency
    import seaborn as sns  # noqa: PLC0415 - lazy, optional dependency

    if errorbar not in {"sem", "std"}:
        raise ValueError("errorbar must be 'sem' or 'std'.")
    method = {False: None, True: "zscore"}.get(normalize, normalize)

    waveforms: "dict[str, pd.Series]" = {}
    for sid, res in results.items():
        if res is None:
            continue
        df = res.get_sync_events(
            center=center, time_before=time_before, time_after=time_after
        )
        for col in ("Time", "Amplitude"):
            if col not in df.columns:
                raise KeyError(
                    f"get_sync_events() returned no '{col}' column; got "
                    f"{list(df.columns)}."
                )
        # Per-subject mean waveform: average amplitude across that subject's events
        # at each time offset relative to the spindle center.
        w = df.groupby("Time")["Amplitude"].mean()
        if method is not None:
            w = normalize_waveform(w, method)
        waveforms[sid] = w
    if not waveforms:
        raise ValueError("No subject had any detected spindles to plot.")

    wf = pd.DataFrame(waveforms)  # index = Time offset (s), columns = subjects
    wf.index.name = "Time"
    n_subj = wf.shape[1]
    # Long form so seaborn can compute the mean and the across-subjects error band
    # (its "se"/"sd" bands are analytic, matching the previous sem()/std()).
    long = wf.reset_index().melt(id_vars="Time", var_name="Subject", value_name="Amplitude")

    with seaborn_theme():
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if show_subjects:
            sns.lineplot(
                data=long,
                x="Time",
                y="Amplitude",
                units="Subject",
                estimator=None,
                color=SUBJECT_GREY,
                lw=0.9,
                alpha=0.7,
                ax=ax,
                zorder=1,
                legend=False,
            )
        sns.lineplot(
            data=long,
            x="Time",
            y="Amplitude",
            estimator="mean",
            errorbar=_ERRORBAR[errorbar],
            color=PRIMARY,
            lw=2.5,
            ax=ax,
            zorder=3,
            label=f"Grand average (n={n_subj}) ± {errorbar.upper()}",
            err_kws={"alpha": BAND_ALPHA, "zorder": 2},
        )
        ax.axvline(0, color="0.35", ls="--", lw=1.2, zorder=0)
        ax.set_xlabel(f"Time relative to spindle {center.lower()} (s)")
        ax.set_ylabel(_YLABEL[method])
        ax.set_title("Spindle grand average across subjects", weight="bold")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        style_axes(ax)
    return ax


def plot_spindles(
    spindles: "SpindlesResult",
    *,
    center: str = "Peak",
    time_before: float = 1.5,
    time_after: float = 1.5,
    errorbar: str = "sem",
    normalize: Union[bool, str] = False,
    show_events: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    ax: Optional[Any] = None,
):
    """Plot **one subject's** average spindle waveform (mean over its spindle epochs).

    Time-locks every detected spindle (via YASA's :meth:`get_sync_events`) and
    averages them into a single mean waveform, with a shaded band of ``errorbar``
    (``"sem"`` or ``"std"``) computed *across that subject's spindles*. This is the
    single-subject counterpart of :func:`plot_spindles_grand_average` (which then
    averages such per-subject waveforms across subjects).

    Args:
        spindles: A YASA ``SpindlesResults`` for one recording (e.g. from
            :func:`~infraslow.processing.detection.spindles_detect`).
        center, time_before, time_after: Passed to ``get_sync_events`` to define
            the window locked to each spindle ``center`` (default ``"Peak"``).
        errorbar: Shaded band across spindles -- ``"sem"`` or ``"std"``.
        normalize: Normalize the mean waveform for display. ``False`` (default)
            keeps raw µV; ``True``/``"zscore"`` z-scores it; ``"peak"`` scales it so
            its peak magnitude is 1. The band (and any event overlays) are scaled
            by the same factor.
        show_events: Overlay each individual spindle's waveform in light grey.
        figsize: Size of a newly created figure (ignored when ``ax`` is passed).
        ax: Existing Axes to draw on; a new one is created when ``None``.

    Returns:
        The matplotlib ``Axes``.

    Raises:
        ValueError: if ``spindles`` is ``None``/has no events, or
            ``errorbar``/``normalize`` is invalid.
        KeyError: if ``get_sync_events`` lacks the expected columns.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 - lazy, optional dependency
    import seaborn as sns  # noqa: PLC0415 - lazy, optional dependency

    if errorbar not in {"sem", "std"}:
        raise ValueError("errorbar must be 'sem' or 'std'.")
    if spindles is None:
        raise ValueError("plot_spindles received no spindles (None).")
    method = {False: None, True: "zscore"}.get(normalize, normalize)

    df = spindles.get_sync_events(
        center=center, time_before=time_before, time_after=time_after
    )
    if len(df) == 0:
        raise ValueError("spindles has no events to plot.")
    for col in ("Time", "Amplitude"):
        if col not in df.columns:
            raise KeyError(
                f"get_sync_events() returned no '{col}' column; got "
                f"{list(df.columns)}."
            )

    # Mean spindle across the subject's individual spindles, at each time offset
    # relative to the spindle center -- used only to derive the normalization
    # constants; seaborn re-derives the plotted mean and its error band below.
    mean = df.groupby("Time")["Amplitude"].mean()
    n_events = int(df["Event"].nunique()) if "Event" in df.columns else 0

    # Normalize *every raw sample* by the same (linear) centre/divisor derived
    # from the mean waveform. Because the transform is linear, seaborn's mean of
    # the normalized samples equals the normalized mean, and its error band is
    # scaled by the same divisor -- matching the old explicit-band approach.
    df = df.copy()
    if method is not None:
        center_val = mean.mean() if method == "zscore" else 0.0
        divisor = mean.std() if method == "zscore" else mean.abs().max()
        normalize_waveform(mean, method)  # validates `method` (raises on bad input)
        if divisor:
            df["Amplitude"] = (df["Amplitude"] - center_val) / divisor

    with seaborn_theme():
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if show_events and "Event" in df.columns:
            sns.lineplot(
                data=df,
                x="Time",
                y="Amplitude",
                units="Event",
                estimator=None,
                color="0.8",
                lw=0.4,
                alpha=0.5,
                ax=ax,
                zorder=1,
                legend=False,
            )
        sns.lineplot(
            data=df,
            x="Time",
            y="Amplitude",
            estimator="mean",
            errorbar=_ERRORBAR[errorbar],
            color=PRIMARY,
            lw=2.5,
            ax=ax,
            zorder=3,
            label=(
                f"Mean (n={n_events} spindles) ± {errorbar.upper()}"
                if n_events
                else f"Mean ± {errorbar.upper()}"
            ),
            err_kws={"alpha": BAND_ALPHA, "zorder": 2},
        )
        ax.axvline(0, color="0.35", ls="--", lw=1.2, zorder=0)
        ax.set_xlabel(f"Time relative to spindle {center.lower()} (s)")
        ax.set_ylabel(_YLABEL[method])
        ax.set_title("Average spindle", weight="bold")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        style_axes(ax)
    return ax


__all__ = ["plot_spindles_grand_average", "plot_spindles"]
