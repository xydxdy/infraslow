"""Processing layer: turn loaded inputs into merged tables and derived metrics.

* :mod:`~infraslow.processing.signal` — signal transforms (resampling, filters)
  and a resampling ``signal_reader`` for the PSG loader.
* :mod:`~infraslow.processing.spindle` — sleep-event detection
  (e.g. :func:`~infraslow.processing.spindle.spindles_detect`) via YASA.
* :mod:`~infraslow.processing.infraslow` — infraslow (~0.02 Hz) sigma-power
  oscillation analysis.
* :mod:`~infraslow.processing.utils` — small, generic building blocks (e.g.
  :func:`~infraslow.processing.utils.find_stage_bouts`).

Per-subject preprocessing itself lives in ``src/scripts/preprocessing.py``, not here.
"""

from __future__ import annotations
