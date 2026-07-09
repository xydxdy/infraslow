"""Small numeric helpers shared by the :mod:`infraslow.viz` plotting functions.

Kept separate from the plotting code so the waveform math can be unit-tested and
reused without pulling in matplotlib. Companion to :mod:`infraslow.viz.utils`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def spindle_event_times(
    spindles: Any, *, mark: str = "Peak", channel: Optional[str] = None
) -> "np.ndarray":
    """Per-spindle event times (seconds from recording start).

    Accepts a YASA ``SpindlesResults`` (anything with ``.summary()``), the summary
    :class:`~pandas.DataFrame` itself, or a bare 1-D array of times already in
    seconds. ``mark`` selects the event column (``"Peak"``/``"Start"``/``"End"``),
    falling back to ``"Start"`` when the requested one is absent. When ``channel``
    is given and the summary has a ``Channel`` column, only that channel's events
    are returned. Empty input yields an empty array.
    """
    if spindles is None:
        return np.empty(0, dtype=float)
    summary = spindles.summary() if hasattr(spindles, "summary") else spindles
    if summary is None or len(summary) == 0:
        return np.empty(0, dtype=float)
    if hasattr(summary, "columns"):
        if channel is not None and "Channel" in summary.columns:
            summary = summary[summary["Channel"].astype(str) == str(channel)]
        col = (
            mark
            if mark in summary.columns
            else ("Start" if "Start" in summary.columns else None)
        )
        if col is None:
            raise KeyError(
                f"Spindle summary has no '{mark}' (or 'Start') column; got "
                f"{list(summary.columns)}."
            )
        return np.asarray(summary[col], dtype=float)
    # Otherwise assume an array-like of times already in seconds.
    return np.asarray(summary, dtype=float).ravel()


def normalize_waveform(w: "pd.Series", method: str) -> "pd.Series":
    """Normalize one per-subject mean waveform before it is averaged.

    ``"zscore"`` centres and scales to unit standard deviation over the window;
    ``"peak"`` divides by the maximum absolute amplitude (peak magnitude -> 1).
    A degenerate (constant) waveform is returned centred but unscaled rather than
    producing NaNs/inf.
    """
    if method == "zscore":
        std = w.std()
        return (w - w.mean()) / std if std else w - w.mean()
    if method == "peak":
        peak = w.abs().max()
        return w / peak if peak else w
    raise ValueError("normalize must be False, True, 'zscore', or 'peak'.")


__all__ = ["normalize_waveform", "spindle_event_times"]
