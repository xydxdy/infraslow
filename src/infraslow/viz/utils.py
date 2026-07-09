"""Generic utilities for the :mod:`infraslow.viz` plotting layer.

Small, reusable plotting/stat building blocks. Companion to
:mod:`infraslow.viz.helpers`.

This module also owns the *shared visual identity* for every figure in
:mod:`infraslow.viz` -- a single seaborn theme applied through
:func:`seaborn_theme` -- so the hand-drawn waveform/rel-power plots and the
YASA-backed spectrogram/hypnogram all look like one deck.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

# Shared default figure size so the multi-panel relative-power plots and the
# single-panel waveform plot stay visually consistent (same width).
DEFAULT_FIGSIZE = (14.0, 8.0)

# --- Shared visual identity -------------------------------------------------
# One seaborn context/style/palette for the whole viz layer. ``PRIMARY`` matches
# the green accent of the progress-update slide deck (slides/build_slides.py);
# ``BAND_ALPHA`` is the opacity of every shaded error band.
SEABORN_CONTEXT = "talk"
SEABORN_STYLE = "whitegrid"
SEABORN_PALETTE = "deep"
PRIMARY = "#2F6F3E"
SUBJECT_GREY = "0.75"
BAND_ALPHA = 0.25


@contextmanager
def seaborn_theme(
    context: str = SEABORN_CONTEXT,
    style: str = SEABORN_STYLE,
    palette: str = SEABORN_PALETTE,
):
    """Temporarily apply infraslow's shared seaborn look for one figure.

    Wrap a plotting body in ``with seaborn_theme():`` so every plot -- including
    the YASA-backed spectrogram/hypnogram -- shares one context, grid style and
    palette. All rcParams touched here (including the colour cycle) are restored
    on exit, so the caller's global matplotlib state is left untouched.

    seaborn is imported lazily so merely importing :mod:`infraslow.viz` does not
    pull in seaborn/matplotlib until a plot is actually drawn.
    """
    import matplotlib as mpl  # noqa: PLC0415 - lazy, optional dependency
    import seaborn as sns  # noqa: PLC0415 - lazy, optional dependency

    prev_cycle = mpl.rcParams["axes.prop_cycle"]
    with sns.plotting_context(context), sns.axes_style(style):
        sns.set_palette(palette)
        try:
            yield
        finally:
            mpl.rcParams["axes.prop_cycle"] = prev_cycle


def style_axes(ax, *, grid: bool = True):
    """Apply the shared finishing touches to one Axes: despine + subtle grid.

    Kept tiny and dependency-lazy so both the hand-drawn plots and the YASA
    wrappers can share the exact same axis polish.
    """
    import seaborn as sns  # noqa: PLC0415 - lazy, optional dependency

    sns.despine(ax=ax)
    if grid:
        ax.grid(True, alpha=0.3, lw=0.6)
    else:
        ax.grid(False)
    return ax


def error_band(stack: "np.ndarray", errorbar: str = "sem") -> "np.ndarray":
    """Half-height of an error band over the first axis (units) of ``stack``.

    ``stack`` is ``(n_units, n_points)`` -- e.g. one row per spindle, or one row
    per subject. ``"sem"`` returns the standard error of the mean, ``"std"`` the
    standard deviation. Returns zeros when there is a single unit (no spread).
    """
    if errorbar not in {"sem", "std"}:
        raise ValueError("errorbar must be 'sem' or 'std'.")
    n = stack.shape[0]
    if n < 2:
        return np.zeros(stack.shape[1])
    sd = stack.std(axis=0, ddof=1)
    return sd if errorbar == "std" else sd / np.sqrt(n)


__all__ = [
    "error_band",
    "DEFAULT_FIGSIZE",
    "SEABORN_CONTEXT",
    "SEABORN_STYLE",
    "SEABORN_PALETTE",
    "PRIMARY",
    "SUBJECT_GREY",
    "BAND_ALPHA",
    "seaborn_theme",
    "style_axes",
]
