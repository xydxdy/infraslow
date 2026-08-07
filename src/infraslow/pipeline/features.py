"""Subject/state/channel-level spindle, slow-wave, and band-power features.

Pure aggregation of `preprocessing.py`'s already-saved artifacts -- no
detection is re-run. Duration/count come straight from the YASA summary
CSVs (`spindel_yasa.csv`/`sw_yasa.csv`); density denominators use the
stage's *every*-bout duration (`bouts["all"]`/`sw_bouts["all"]`), matching
the scope `preprocessing.py` itself ran spindle/SW detection over (every
consecutive-stage run >= min_bout_sec, not just spindle-containing ones).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _bouts_total_minutes(bouts: np.ndarray) -> float:
    if bouts.size == 0:
        return 0.0
    return float((bouts[:, 1] - bouts[:, 0]).sum() / 60.0)


def _mean_or_nan(summary: pd.DataFrame, column: str) -> float:
    if column not in summary.columns or len(summary) == 0:
        return float("nan")
    return float(summary[column].mean())


def compute_spindle_features(summary: pd.DataFrame, bouts: np.ndarray) -> Dict[str, float]:
    """`spindle_count, spindle_density_per_min, spindle_mean_amplitude,
    spindle_mean_frequency, spindle_mean_duration_s` from one subject/state/
    channel's `spindel_yasa.csv` + that stage's every-bout array."""
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
    state/channel's `sw_yasa.csv` + that stage's every-bout array."""
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
