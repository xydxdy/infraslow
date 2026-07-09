"""Input layer: load raw data from disk into usable in-memory objects.

* :mod:`~infraslow.io.psg_loader` — deterministic, alias-aware EDF/PSG loading
  via LunaAPI (:class:`~infraslow.io.psg_loader.BioserenityPSGLoader`).
* :mod:`~infraslow.io.csv_tools` — per-file CSV reading, column selection, and
  duplicate handling used by the merge step.
* :mod:`~infraslow.io.hypnodensity` — reduce a per-epoch hypnodensity CSV to a
  ``(timestamp, stage)`` hypnogram via argmax.
* :mod:`~infraslow.io.availability` — drop selected subjects that lack an EDF or
  hypnodensity file on disk.
"""

from __future__ import annotations

from .availability import filter_groups_by_available_files
from .csv_tools import (
    handle_duplicates,
    read_csv_strict,
    select_columns,
    validate_columns,
)
from .hypnodensity import (
    hypnodensity_to_annotations,
    make_hypnodensity_annotation_loader,
)
from .psg_loader import BIOSERENITY_ALIAS_MAP, BioserenityPSGLoader

__all__ = [
    "BIOSERENITY_ALIAS_MAP",
    "BioserenityPSGLoader",
    "filter_groups_by_available_files",
    "handle_duplicates",
    "hypnodensity_to_annotations",
    "make_hypnodensity_annotation_loader",
    "read_csv_strict",
    "select_columns",
    "validate_columns",
]
