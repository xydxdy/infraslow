"""Subject/state/channel-level spindle, slow-wave, and band-power features.

Pure aggregation of `preprocessing.py`'s already-saved artifacts -- no
detection is re-run. `preprocessing.py` itself runs spindle/SW detection over
every epoch of a sleep stage across the *whole night* (not scoped to the
>= min_bout_sec consecutive-stage bouts `pipeline.py` selects features from),
so the raw `spindle_yasa.csv`/`sw_yasa.csv` summaries can contain events from
short stage fragments outside any bout. `compute_spindle_features` and
`compute_slow_wave_features` first filter `summary` down to only the events
whose onset timestamp (`Peak`/`NegPeak`) falls inside one of `bouts` (via
`_filter_events_in_bouts`), so every count/density/mean-* value they compute
shares the same bout-scoped denominator as `phase.py`'s `event_count` and the
spectrum/band-power features -- not inflated by how fragmented a subject's
sleep happened to be.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _bouts_total_minutes(bouts: np.ndarray) -> float:
    if bouts.size == 0:
        return 0.0
    return float((bouts[:, 1] - bouts[:, 0]).sum() / 60.0)


def _filter_events_in_bouts(summary: pd.DataFrame, bouts: np.ndarray, time_col: str) -> pd.DataFrame:
    """Rows of `summary` whose `time_col` timestamp falls inside any of `bouts`
    (start-inclusive, stop-exclusive) -- scopes night-wide YASA detections down
    to the bouts pipeline.py actually selected features from, so `*_count`/
    `*_density_per_min` share the same denominator scope as `bouts`."""
    if bouts.size == 0 or len(summary) == 0 or time_col not in summary.columns:
        return summary.iloc[0:0]
    times = summary[time_col].to_numpy()
    mask = np.zeros(len(summary), dtype=bool)
    for a, b in bouts:
        mask |= (times >= a) & (times < b)
    return summary[mask]


def _mean_or_nan(summary: pd.DataFrame, column: str) -> float:
    if column not in summary.columns or len(summary) == 0:
        return float("nan")
    return float(summary[column].mean())


def compute_spindle_features(summary: pd.DataFrame, bouts: np.ndarray) -> Dict[str, float]:
    """`spindle_count, spindle_density_per_min, spindle_mean_amplitude,
    spindle_mean_frequency, spindle_mean_duration_s` from one subject/state/
    channel's `spindle_yasa.csv` (filtered to events whose `Peak` falls inside
    `bouts`) + that stage's bout array."""
    summary = _filter_events_in_bouts(summary, bouts, "Peak")
    count = int(len(summary))
    total_min = _bouts_total_minutes(bouts)
    density = (count / total_min) if total_min > 0 else float("nan")
    duration_s = (
        float((summary["End"] - summary["Start"]).mean())
        if count > 0 and {"Start", "End"}.issubset(summary.columns)
        else float("nan")
    )
    return dict(
        spindle_count=count,
        spindle_density_per_min=density,
        spindle_mean_amplitude=_mean_or_nan(summary, "Amplitude"),
        spindle_mean_frequency=_mean_or_nan(summary, "Frequency"),
        spindle_mean_duration_s=duration_s,
    )


def compute_slow_wave_features(summary: pd.DataFrame, bouts: np.ndarray) -> Dict[str, float]:
    """`slow_wave_count, slow_wave_density_per_min, slow_wave_mean_ptp,
    slow_wave_mean_frequency, slow_wave_mean_duration_s` from one subject/
    state/channel's `sw_yasa.csv` (filtered to events whose `NegPeak` falls
    inside `bouts`) + that stage's bout array."""
    summary = _filter_events_in_bouts(summary, bouts, "NegPeak")
    count = int(len(summary))
    total_min = _bouts_total_minutes(bouts)
    density = (count / total_min) if total_min > 0 else float("nan")
    duration_s = (
        float((summary["End"] - summary["Start"]).mean())
        if count > 0 and {"Start", "End"}.issubset(summary.columns)
        else float("nan")
    )
    return dict(
        slow_wave_count=count,
        slow_wave_density_per_min=density,
        slow_wave_mean_ptp=_mean_or_nan(summary, "PTP"),
        slow_wave_mean_frequency=_mean_or_nan(summary, "Frequency"),
        slow_wave_mean_duration_s=duration_s,
    )


def compute_band_power_feature(
    t_env: np.ndarray, power: np.ndarray, bouts: np.ndarray, *, name: str,
) -> Dict[str, float]:
    """`{f"{name}_power_db": mean power over every sample whose t_env falls
    inside any of `bouts`}` -- `nan` if `bouts` is empty. `power` is already
    in dB (see `preprocessing.py`'s `eeg_envelope(..., to_db=True)`)."""
    if bouts.size == 0:
        return {f"{name}_power_db": float("nan")}
    mask = np.zeros(t_env.shape, dtype=bool)
    for a, b in bouts:
        mask |= (t_env >= a) & (t_env < b)
    value = float(power[mask].mean()) if mask.any() else float("nan")
    return {f"{name}_power_db": value}


__all__ = [
    "compute_spindle_features",
    "compute_slow_wave_features",
    "compute_band_power_feature",
]
