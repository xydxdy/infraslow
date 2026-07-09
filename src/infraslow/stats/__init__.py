"""Statistical comparison layer for :mod:`infraslow`.

Small, dependency-light helpers for *comparing* results produced by the
processing layer -- e.g. two spindle detectors (YASA vs. Luna) run on the same
recording. The heavy lifting (t-tests, rank tests, multiple-comparison
correction, Poisson rate tests) is delegated to :mod:`statsmodels`, imported
lazily inside the functions so importing :mod:`infraslow.stats` stays cheap.

* :func:`~infraslow.stats.spindle.compare_spindle_features` -- per-feature
  two-sample comparison (Welch t-test + Brunner-Munzel rank test + effect sizes)
  of two detectors' per-event summary tables.
* :func:`~infraslow.stats.spindle.compare_spindle_counts` -- compare the two
  detectors' event *counts* as Poisson rates (optionally per unit of staged
  time).
* :func:`~infraslow.stats.infraslow.compare_spindle_infraslow` /
  :func:`~infraslow.stats.infraslow.associate_spindle_rate` -- compare the two
  detectors' **infraslow** (~0.02 Hz) sigma-power/spindle-rate oscillations.
"""

from __future__ import annotations

from .infraslow import (
    associate_spindle_rate,
    compare_spindle_infraslow,
    spindle_infraslow_stats,
)
from .spindle import (
    SPINDLE_FEATURES,
    compare_spindle_counts,
    compare_spindle_features,
)

__all__ = [
    "SPINDLE_FEATURES",
    "compare_spindle_features",
    "compare_spindle_counts",
    "spindle_infraslow_stats",
    "compare_spindle_infraslow",
    "associate_spindle_rate",
]
