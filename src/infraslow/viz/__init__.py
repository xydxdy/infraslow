"""Visualisation layer: plot the results produced by the processing layer.

Kept separate from :mod:`infraslow.processing` so the analysis code carries no
matplotlib dependency -- matplotlib is imported lazily inside the plot functions.

* :mod:`~infraslow.viz.hypnogram` -- the
  :func:`~infraslow.viz.hypnogram.plot_hypnogram` hypnogram plot, optionally
  marking detected spindles.
* :mod:`~infraslow.viz.helpers` -- small numeric helpers shared by the plots.
* :mod:`~infraslow.viz.utils` -- generic plotting utilities.
"""

from __future__ import annotations

from .hypnogram import plot_hypnogram

__all__ = [
    "plot_hypnogram",
]
