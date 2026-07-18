"""Processing layer: turn loaded inputs into merged tables and derived metrics.

* :mod:`~infraslow.processing.signal` — signal transforms (resampling, filters)
  and a resampling ``signal_reader`` for the PSG loader.
* :mod:`~infraslow.processing.detection` — sleep-event detection
  (e.g. :func:`~infraslow.processing.detection.spindles_detect`) via YASA.
* :mod:`~infraslow.processing.infraslow` — infraslow (~0.02 Hz) sigma-power
  oscillation analysis.
* :mod:`~infraslow.processing.subject_pipeline` — per-subject sleep + infraslow
  metrics pipeline (see ``src/run_all_metrics.py``).
"""

from __future__ import annotations
