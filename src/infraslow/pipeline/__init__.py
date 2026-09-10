"""Infraslow spectrum and phase analysis pipeline for N2/N3 sleep-EEG data.

* :mod:`~infraslow.pipeline.io` — read preprocessed subject trees
  and organize them for analysis.
* :mod:`~infraslow.pipeline.spectrum` — compute infraslow power spectral
  density and extract ISFS (infraslow frequency) features from bouts.
* :mod:`~infraslow.pipeline.phase` — infraslow phase analysis and
  phase-locked coupling estimates.
* :mod:`~infraslow.pipeline.features` — aggregate features across
  subjects and sleep stages for statistical analysis.

Preprocessed data (envelopes, temporal_ISFS, stage bouts, spindle/SW
detections) lives under each subject's directory tree created by
``src/scripts/preprocessing.py``.
"""

from __future__ import annotations
