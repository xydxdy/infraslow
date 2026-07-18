"""Generic utilities for the :mod:`infraslow.viz` plotting layer.

This module also owns the *shared visual identity* for every figure in
:mod:`infraslow.viz` -- a single seaborn theme applied through
:func:`seaborn_theme` -- so every plot looks like one deck.
"""

from __future__ import annotations

from contextlib import contextmanager

# --- Shared visual identity -------------------------------------------------
# One seaborn context/style/palette for the whole viz layer.
SEABORN_CONTEXT = "talk"
SEABORN_STYLE = "whitegrid"
SEABORN_PALETTE = "deep"


@contextmanager
def seaborn_theme(
    context: str = SEABORN_CONTEXT,
    style: str = SEABORN_STYLE,
    palette: str = SEABORN_PALETTE,
):
    """Temporarily apply infraslow's shared seaborn look for one figure.

    Wrap a plotting body in ``with seaborn_theme():`` so every plot -- including
    the YASA-backed hypnogram -- shares one context, grid style and palette. All
    rcParams touched here (including the colour cycle) are restored on exit, so
    the caller's global matplotlib state is left untouched.

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
    """Apply the shared finishing touches to one Axes: despine + subtle grid."""
    import seaborn as sns  # noqa: PLC0415 - lazy, optional dependency

    sns.despine(ax=ax)
    if grid:
        ax.grid(True, alpha=0.3, lw=0.6)
    else:
        ax.grid(False)
    return ax


__all__ = [
    "SEABORN_CONTEXT",
    "SEABORN_STYLE",
    "SEABORN_PALETTE",
    "seaborn_theme",
    "style_axes",
]
