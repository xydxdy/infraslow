"""Visualisation layer: plot the results produced by the processing layer.

Kept separate from :mod:`infraslow.processing` so the analysis code carries no
matplotlib dependency -- matplotlib is imported lazily inside the plot functions.

* :mod:`~infraslow.viz.spindles` -- average spindle waveforms:
  :func:`~infraslow.viz.spindles.plot_spindles` (one subject) and
  :func:`~infraslow.viz.spindles.plot_spindles_grand_average` (across subjects).
* :mod:`~infraslow.viz.spectrogram` -- the
  :func:`~infraslow.viz.spectrogram.plot_spectrogram` full-night spectrogram for
  a loaded recording.
* :mod:`~infraslow.viz.hypnogram` -- the
  :func:`~infraslow.viz.hypnogram.plot_hypnogram` hypnogram plot, optionally
  marking detected spindles.
* :mod:`~infraslow.viz.spindle_power` -- spindle-locked sigma-band relative power
  (:func:`~infraslow.viz.spindle_power.plot_spindle_sigma_power` for one subject,
  :func:`~infraslow.viz.spindle_power.plot_spindles_sigma_power_grand_average`
  across subjects).
* :mod:`~infraslow.viz.helpers` -- small numeric helpers shared by the plots.
* :mod:`~infraslow.viz.utils` -- generic plotting utilities.
"""

from __future__ import annotations

from .hypnogram import plot_hypnogram
from .infraslow import (
    plot_infraslow_spectra,
    plot_infraslow_spectra_grand_average,
    plot_spectra,
    plot_spectra_grand_average,
    plot_spindle_sigma_spectra,
    plot_spindle_sigma_spectra_grand_average,
)
from .spectrogram import plot_spectrogram, plot_spectrogram_grand_average
from .spindle_power import (
    plot_spindle_sigma_power,
    plot_spindles_sigma_power_grand_average,
)
from .spindles import plot_spindles, plot_spindles_grand_average

__all__ = [
    "plot_spindles",
    "plot_spindles_grand_average",
    "plot_spectrogram",
    "plot_spectrogram_grand_average",
    "plot_hypnogram",
    "plot_spindle_sigma_power",
    "plot_spindles_sigma_power_grand_average",
    "plot_spectra",
    "plot_spectra_grand_average",
    "plot_infraslow_spectra",
    "plot_infraslow_spectra_grand_average",
    "plot_spindle_sigma_spectra",
    "plot_spindle_sigma_spectra_grand_average",
]
