"""infraslow — PSG/EDF loading and sleep-study CSV processing for the Bioserenity dataset.

Two cohesive concerns, each in its own subpackage:

* :mod:`infraslow.io` — reading inputs: the LunaAPI-backed
  :class:`~infraslow.io.psg_loader.BioserenityPSGLoader` for EDF recordings and
  the per-file CSV tools used before merging.
* :mod:`infraslow.processing` — turning inputs into results: the reusable
  :func:`~infraslow.processing.merge.merge_csv_columns` merge and the
  sleep-parameter calculations / subject grouping.

The most commonly used entry points are re-exported here for convenience.
"""

from __future__ import annotations

from .io.psg_loader import BIOSERENITY_ALIAS_MAP, BioserenityPSGLoader
from .processing.merge import merge_csv_columns
from .processing.sleep_params import (
    compute_ahi,
    selected_groups_by_ahi,
    selected_groups_by_se,
)

__all__ = [
    "BIOSERENITY_ALIAS_MAP",
    "BioserenityPSGLoader",
    "merge_csv_columns",
    "compute_ahi",
    "selected_groups_by_ahi",
    "selected_groups_by_se",
]
