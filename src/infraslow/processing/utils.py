"""Generic utilities for the :mod:`infraslow.processing` layer.

Small, reusable building blocks specific to turning inputs into derived results.
Companion to :mod:`infraslow.processing.helpers`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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


__all__ = ["is_nan"]
