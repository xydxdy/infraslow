"""Generic utilities for the :mod:`infraslow.processing` layer.

Small, reusable building blocks specific to turning inputs into derived results.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from .spindle import DEFAULT_EPOCH_SEC


def is_nan(value: Any) -> bool:
    """Return ``True`` when ``value`` is NaN, tolerating non-numeric inputs.

    Coerces to ``float`` first; anything that cannot be coerced (``None``,
    strings, arbitrary objects) is treated as *not* NaN rather than raising.
    Handy for filtering values read from heterogeneous sources (e.g. lunapi
    header tables) where a cell may be blank, text, or a real number.
    """
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def find_stage_bouts(
    hypnogram: np.ndarray,
    stage_codes: Sequence[int],
    *,
    epoch_sec: float = DEFAULT_EPOCH_SEC,
    min_dur: float = 200.0,
) -> List[Tuple[float, float]]:
    """``(start, stop)`` times (s) of consecutive-stage runs of at least ``min_dur``.

    A run is a maximal block of epochs whose code is in ``stage_codes``.
    """
    codes = np.asarray(hypnogram)
    wanted = set(int(c) for c in stage_codes)
    in_stage = np.fromiter((int(c) in wanted for c in codes), dtype=bool, count=codes.size)

    bouts: List[Tuple[float, float]] = []
    i, n = 0, codes.size
    while i < n:
        if in_stage[i]:
            j = i
            while j < n and in_stage[j]:
                j += 1
            if (j - i) * epoch_sec >= min_dur:
                bouts.append((i * epoch_sec, j * epoch_sec))
            i = j
        else:
            i += 1
    return bouts


__all__ = ["is_nan", "find_stage_bouts"]
