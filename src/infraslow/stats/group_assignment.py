"""Log1p mean +/- std cutoff assignment of subjects into low/high spindle-rate groups.

Implements md/group_analysis.md Step 4: take ``log1p`` of every valid
subject's N2-C3 spindle rate, compute that log1p-scale distribution's mean
and (sample) standard deviation, then split subjects by two fixed cutoffs::

    low_spindle_rate  : log1p(rate) <  mean - std
    high_spindle_rate : log1p(rate) >  mean + std
    mid_spindle_rate  : mean - std <= log1p(rate) <= mean + std

The middle band is returned (never dropped) but is excluded from the
low-vs-high comparison downstream, by construction: callers keep only
``spindle_group in {low_spindle_rate, high_spindle_rate}`` rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "LOW_LABEL",
    "HIGH_LABEL",
    "MID_LABEL",
    "MIN_SUBJECTS_FOR_CUTOFF",
    "CutoffResult",
    "assign_spindle_rate_groups",
]

LOW_LABEL = "low_spindle_rate"
HIGH_LABEL = "high_spindle_rate"
MID_LABEL = "mid_spindle_rate"
MIN_SUBJECTS_FOR_CUTOFF = 2


@dataclass
class CutoffResult:
    """The log1p-scale mean/std and the low/high cutoff thresholds actually used."""

    log_mean: float
    log_std: float
    low_threshold: float  # log1p scale: mean - std
    high_threshold: float  # log1p scale: mean + std
    low_threshold_original_scale: float  # expm1(low_threshold), i.e. spindle-rate units
    high_threshold_original_scale: float  # expm1(high_threshold)


def assign_spindle_rate_groups(
    subject_ids: pd.Series,
    spindle_per_min: pd.Series,
    spindle_per_min_sem: pd.Series,
    *,
    rate_col: str = "spindle_per_min",
    rate_sem_col: str = "spindle_per_min_SEM",
) -> Tuple[pd.DataFrame, CutoffResult]:
    """One row per subject with a valid spindle rate: group by a log1p mean +/- std cutoff.

    Args:
        subject_ids, spindle_per_min, spindle_per_min_sem: Aligned per-subject
            Series (same length/order); callers should already have filtered
            to validated N2-C3 subjects. Unit-agnostic -- these may be
            expressed in spindles/min, spindles/hr, etc.
        rate_col, rate_sem_col: Column names for the rate/SEM in the returned
            ``assignments`` (defaults match ``spindle_per_min``'s native unit;
            pass e.g. ``"spindle_per_hr"``/``"spindle_per_hr_SEM"`` when the
            input Series are in spindles/hr instead).

    Returns:
        ``(assignments, cutoff)`` where ``assignments`` has columns
        ``subject_id, {rate_col}, {rate_sem_col}, spindle_group`` -- one row
        per subject with a finite rate value (non-finite subjects are the
        validation step's responsibility, not this function's, and are
        simply excluded here). ``spindle_group`` is one of :data:`LOW_LABEL`,
        :data:`HIGH_LABEL`, or :data:`MID_LABEL`.

    Raises:
        ValueError: if fewer than :data:`MIN_SUBJECTS_FOR_CUTOFF` valid values are given.
    """
    subject_ids = pd.Series(subject_ids).reset_index(drop=True)
    values = pd.Series(spindle_per_min).reset_index(drop=True).to_numpy(dtype=float)
    sems = pd.Series(spindle_per_min_sem).reset_index(drop=True)

    valid_mask = np.isfinite(values)
    n_excluded = int((~valid_mask).sum())
    if n_excluded:
        logger.warning(
            "%d subject(s) excluded from grouping (non-finite spindle rate); "
            "validation should already have flagged these", n_excluded,
        )

    valid_values = values[valid_mask]
    if valid_values.size < MIN_SUBJECTS_FOR_CUTOFF:
        raise ValueError(
            f"Need at least {MIN_SUBJECTS_FOR_CUTOFF} subjects with a valid spindle "
            f"rate to compute a log1p mean/std cutoff (got {valid_values.size})."
        )

    log_values = np.log1p(valid_values)
    log_mean = float(log_values.mean())
    log_std = float(log_values.std(ddof=1)) if log_values.size > 1 else 0.0
    low_threshold = log_mean
    high_threshold = log_mean
    logger.info(
        "log1p spindle-rate cutoff: mean=%.4f std=%.4f -> low<%.4f, high>%.4f (log1p scale)",
        log_mean, log_std, low_threshold, high_threshold,
    )

    spindle_group = np.full(log_values.shape, MID_LABEL, dtype=object)
    spindle_group[log_values < low_threshold] = LOW_LABEL
    spindle_group[log_values > high_threshold] = HIGH_LABEL

    cutoff = CutoffResult(
        log_mean=log_mean, log_std=log_std,
        low_threshold=low_threshold, high_threshold=high_threshold,
        low_threshold_original_scale=float(np.expm1(low_threshold)),
        high_threshold_original_scale=float(np.expm1(high_threshold)),
    )

    assignments = pd.DataFrame({
        "subject_id": subject_ids[valid_mask].to_numpy(),
        rate_col: valid_values,
        rate_sem_col: sems[valid_mask].to_numpy(),
        "spindle_group": spindle_group,
    })
    return assignments, cutoff
