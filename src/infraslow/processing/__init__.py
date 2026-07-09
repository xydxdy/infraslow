"""Processing layer: turn loaded inputs into merged tables and derived metrics.

* :mod:`~infraslow.processing.merge` — the reusable
  :func:`~infraslow.processing.merge.merge_csv_columns` function that merges the
  stages, metadata, and apnea CSVs on a shared ID column.
* :mod:`~infraslow.processing.sleep_params` — sleep-metric calculations
  (e.g. :func:`~infraslow.processing.sleep_params.compute_ahi`) and two-threshold
  subject grouping.
* :mod:`~infraslow.processing.signal` — signal transforms (resampling, filters)
  and a resampling ``signal_reader`` for the PSG loader.
* :mod:`~infraslow.processing.detection` — sleep-event detection
  (e.g. :func:`~infraslow.processing.detection.spindles_detect`) via YASA.
* :mod:`~infraslow.processing.detection_luna` — the same spindle detection
  (:func:`~infraslow.processing.detection_luna.spindles_detect_luna`) via Luna's
  wavelet ``SPINDLES`` command (``lunapi``).
"""

from __future__ import annotations

from .detection import spindles_detect
from .detection_luna import spindles_detect_luna
from .merge import merge_csv_columns
from .signal import (
    DEFAULT_TARGET_SFREQ,
    make_resampling_signal_reader,
    resample_channels,
)
from .sleep_params import (
    compute_ahi,
    selected_groups_by_ahi,
    selected_groups_by_se,
)

__all__ = [
    "spindles_detect",
    "spindles_detect_luna",
    "merge_csv_columns",
    "compute_ahi",
    "selected_groups_by_ahi",
    "selected_groups_by_se",
    "DEFAULT_TARGET_SFREQ",
    "make_resampling_signal_reader",
    "resample_channels",
]
