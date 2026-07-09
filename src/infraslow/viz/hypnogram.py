"""Hypnogram plotting for loaded PSG recordings, with optional spindle marks.

A thin adapter over :func:`yasa.plot_hypnogram` that fits this repo's
:class:`~infraslow.io.psg_loader.BioserenityPSGLoader`: it turns the loader's
per-epoch ``(timestamp, stage)`` annotations into the integer hypnogram YASA
expects, and -- optionally -- overlays the times of detected spindles on top.
Kept in the ``viz`` layer so the processing/io code carries no matplotlib
dependency.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..processing.detection import (
    DEFAULT_EPOCH_SEC,
    DEFAULT_STAGE_MAP,
    _extract_epoch_stages,
    _stages_to_int,
)
from .helpers import spindle_event_times
from .utils import seaborn_theme, style_axes


def plot_hypnogram(
    loader: Any,
    *,
    spindles: Any = None,
    epoch_sec: float = DEFAULT_EPOCH_SEC,
    stage_map: Mapping[str, int] = DEFAULT_STAGE_MAP,
    stage_column: str = "stage",
    mark: str = "Peak",
    mark_color: str = "red",
    mark_alpha: float = 0.4,
    ax: Optional[Any] = None,
    **kwargs: Any,
):
    """Plot a subject's hypnogram, optionally marking detected spindles.

    Thin adapter over :func:`yasa.plot_hypnogram`: it reads the loader's
    ``annotations`` (the per-epoch ``(timestamp, stage)`` hypnogram), maps the
    stage labels to YASA's integer codes, and plots them. When ``spindles`` is
    given, each event's time is overlaid as a thin vertical line so spindle
    activity can be read against sleep stage.

    Args:
        loader: A loaded (or loadable)
            :class:`~infraslow.io.psg_loader.BioserenityPSGLoader` -- anything
            exposing ``annotations`` and ``is_loaded``/``load()``. Loaded in
            place if not already.
        spindles: Optional spindle detection result for the *same* recording --
            a YASA ``SpindlesResults`` (from
            :func:`~infraslow.processing.detection.spindles_detect`), its
            ``summary()`` DataFrame, or a 1-D array of event times in seconds.
            ``None`` (default) plots the hypnogram alone.
        epoch_sec: Seconds per scored epoch (sets ``sf_hypno = 1/epoch_sec``).
        stage_map, stage_column: How to map the per-epoch stage labels to YASA
            integer codes (same contract as
            :func:`~infraslow.processing.detection.spindles_detect`).
        mark: Spindle summary column to mark (``"Peak"``, ``"Start"``, ``"End"``).
        mark_color, mark_alpha: Colour/opacity of the spindle marker lines.
        ax: Existing Axes to draw on; YASA creates one when ``None``.
        **kwargs: Passed straight through to :func:`yasa.plot_hypnogram`.

    Returns:
        The matplotlib ``Axes`` the hypnogram was drawn on.

    Raises:
        ImportError: if YASA is not installed.
        ValueError: if the loader exposes no annotations to plot.
    """
    try:
        import yasa  # noqa: PLC0415 - lazy, optional dependency
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "plot_hypnogram requires the 'yasa' package. Install it with "
            "`pip install yasa` (in a Slurm job or interactive node, not the "
            "login node)."
        ) from exc

    # Load on demand so a freshly-built loader can be passed straight in.
    if hasattr(loader, "is_loaded") and not loader.is_loaded:
        loader.load()

    annotations = getattr(loader, "annotations", None)
    if annotations is None:
        raise ValueError(
            "loader has no annotations to plot; construct it with an "
            "annotation_loader (the default attaches the hypnodensity hypnogram)."
        )

    # Per-epoch integer hypnogram at one value every ``epoch_sec`` seconds.
    stages = _extract_epoch_stages(annotations, stage_column=stage_column)
    hypno_int = _stages_to_int(stages, stage_map)
    sf_hypno = 1.0 / float(epoch_sec)

    with seaborn_theme():
        result = yasa.plot_hypnogram(hypno_int, sf_hypno=sf_hypno, ax=ax, **kwargs)
        # YASA returns an Axes; tolerate a Figure-returning build by taking its first.
        out_ax = result
        if not hasattr(result, "axvline"):
            axes = getattr(result, "axes", None)
            out_ax = axes[0] if axes else result

        # Overlay spindle marks. YASA's functional hypnogram x-axis is in hours, so
        # convert the event times (seconds from recording start) to hours.
        times = spindle_event_times(spindles, mark=mark)
        if times.size:
            times_hr = times / 3600.0
            for i, x in enumerate(times_hr):
                out_ax.axvline(
                    x,
                    color=mark_color,
                    lw=0.8,
                    alpha=mark_alpha,
                    zorder=5,
                    label=f"Spindle ({mark}, n={times.size})" if i == 0 else None,
                )
            out_ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        style_axes(out_ax, grid=False)

    return out_ax


__all__ = ["plot_hypnogram"]
