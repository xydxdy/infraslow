"""Reusable statistics for group-level comparisons (see ``src/group_analysis.py``).

* :mod:`~infraslow.stats.group_assignment` -- log1p mean +/- std cutoff
  assignment of subjects into low/mid/high spindle-rate groups.
* :mod:`~infraslow.stats.group_comparison` -- descriptive statistics, test
  selection (Welch vs. Mann-Whitney), and BH-FDR correction across parameters.
* :mod:`~infraslow.stats.effect_sizes` -- Hedges' g and rank-biserial / Cliff's
  delta, both oriented ``high - low``.
"""

from __future__ import annotations

from .effect_sizes import hedges_g, rank_biserial_cliffs_delta
from .group_assignment import HIGH_LABEL, LOW_LABEL, MID_LABEL, assign_spindle_rate_groups
from .group_comparison import compare_parameter, compare_parameters, descriptive_stats

__all__ = [
    "hedges_g",
    "rank_biserial_cliffs_delta",
    "HIGH_LABEL",
    "LOW_LABEL",
    "MID_LABEL",
    "assign_spindle_rate_groups",
    "compare_parameter",
    "compare_parameters",
    "descriptive_stats",
]
