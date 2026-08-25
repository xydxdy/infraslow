"""Input layer: load raw data from disk into usable in-memory objects.

* :mod:`~infraslow.io.psg_loader` — deterministic, alias-aware EDF/PSG loading
  via LunaAPI (:class:`~infraslow.io.psg_loader.BioserenityPSGLoader`).
* :mod:`~infraslow.io.hypnodensity` — reduce a per-epoch hypnodensity CSV to a
  ``(timestamp, stage)`` hypnogram via argmax.
* :mod:`~infraslow.io.usleep_hypnodensity` — load/channel-resolve 1-s U-Sleep
  hypnodensities and reduce them to a coarser epoch/hypnogram.
* :mod:`~infraslow.io.metadata` — load/combine Bioserenity subject metadata
  CSVs and discover the cohort with usable EDF + hypnodensity data.
"""

from __future__ import annotations
