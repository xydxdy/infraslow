"""Input layer: load raw data from disk into usable in-memory objects.

* :mod:`~infraslow.io.psg_loader` — deterministic, alias-aware EDF/PSG loading
  via LunaAPI (:class:`~infraslow.io.psg_loader.BioserenityPSGLoader`).
* :mod:`~infraslow.io.hypnodensity` — reduce a per-epoch hypnodensity CSV to a
  ``(timestamp, stage)`` hypnogram via argmax.
"""

from __future__ import annotations
