"""Two-component GMM assignment of subjects into low/high spindle-rate groups.

Implements md/group_analysis.md Step 4: fit a 2-component Gaussian mixture on
one valid N2-C3 ``spindle_per_min`` value per subject, once on the raw scale and
once on ``log1p(spindle_per_min)``, keep whichever representation has the lower
BIC, then order the two components by their center on the *original*
spindle-rate scale so the lower-center component is always
``low_spindle_rate`` and the higher-center component is always
``high_spindle_rate``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

__all__ = [
    "LOW_LABEL",
    "HIGH_LABEL",
    "RANDOM_STATE_DEFAULT",
    "GMMFitResult",
    "fit_two_component_gmm",
    "assign_spindle_rate_groups",
]

RANDOM_STATE_DEFAULT = 42
LOW_LABEL = "low_spindle_rate"
HIGH_LABEL = "high_spindle_rate"
#: sklearn GaussianMixture restarts per representation, for a stable fit given
#: n_init random initialisations under one fixed random_state.
N_INIT = 10
MIN_SUBJECTS_FOR_GMM = 4


@dataclass
class GMMFitResult:
    """The selected 2-component GMM fit, plus both candidate representations' BIC."""

    scale: str  # "raw" or "log1p" -- the representation actually used
    bic_raw: float
    bic_log1p: float
    model: GaussianMixture
    labels: np.ndarray  # raw (unordered) component index per subject, fit-scale order
    probs: np.ndarray  # (n, 2) posterior probabilities, columns in raw component order
    centers_original_scale: np.ndarray  # each raw component's center, mapped back to spindles/min


def _fit_gmm(values_2d: np.ndarray, *, random_state: int) -> GaussianMixture:
    gmm = GaussianMixture(n_components=2, random_state=random_state, n_init=N_INIT)
    gmm.fit(values_2d)
    return gmm


def fit_two_component_gmm(
    spindle_per_min: np.ndarray, *, random_state: int = RANDOM_STATE_DEFAULT
) -> GMMFitResult:
    """Fit raw- and log1p-scale 2-component GMMs and return the lower-BIC one.

    Args:
        spindle_per_min: One valid (finite, non-negative), already-filtered
            N2-C3 ``spindle_per_min`` value per subject.
        random_state: Fixed seed for both candidate fits (reproducible groups).

    Returns:
        The selected :class:`GMMFitResult`; ``centers_original_scale`` maps
        component centers back to spindles/min even when ``scale == "log1p"``
        (via ``expm1``), so callers can always order groups on the natural scale.

    Raises:
        ValueError: if fewer than :data:`MIN_SUBJECTS_FOR_GMM` values are given.
    """
    values = np.asarray(spindle_per_min, dtype=float)
    if values.ndim != 1 or values.size < MIN_SUBJECTS_FOR_GMM:
        raise ValueError(
            f"Need at least {MIN_SUBJECTS_FOR_GMM} subjects with a valid "
            f"spindle_per_min value to fit a GMM (got {values.size})."
        )

    raw = values.reshape(-1, 1)
    log1p = np.log1p(values).reshape(-1, 1)

    gmm_raw = _fit_gmm(raw, random_state=random_state)
    gmm_log1p = _fit_gmm(log1p, random_state=random_state)
    bic_raw = float(gmm_raw.bic(raw))
    bic_log1p = float(gmm_log1p.bic(log1p))
    logger.info("GMM BIC: raw-scale=%.2f log1p-scale=%.2f", bic_raw, bic_log1p)

    if bic_raw <= bic_log1p:
        scale, gmm, fit_values = "raw", gmm_raw, raw
        centers = gmm.means_.ravel()
    else:
        scale, gmm, fit_values = "log1p", gmm_log1p, log1p
        centers = np.expm1(gmm.means_.ravel())
    logger.info("GMM representation selected: %s-scale", scale)

    labels = gmm.predict(fit_values)
    probs = gmm.predict_proba(fit_values)
    return GMMFitResult(
        scale=scale, bic_raw=bic_raw, bic_log1p=bic_log1p, model=gmm,
        labels=labels, probs=probs, centers_original_scale=centers,
    )


def assign_spindle_rate_groups(
    subject_ids: pd.Series,
    spindle_per_min: pd.Series,
    spindle_per_min_sem: pd.Series,
    *,
    random_state: int = RANDOM_STATE_DEFAULT,
    probability_threshold: float = 0.70,
) -> Tuple[pd.DataFrame, GMMFitResult]:
    """One row per subject with a valid ``spindle_per_min``: group + posterior probability.

    Args:
        subject_ids, spindle_per_min, spindle_per_min_sem: Aligned per-subject
            Series (same length/order); callers should already have filtered
            to validated N2-C3 subjects.
        random_state: Fixed seed, forwarded to :func:`fit_two_component_gmm`.
        probability_threshold: A subject's ``uncertain_assignment`` is ``True``
            when its posterior ``group_probability`` (max of the two component
            probabilities) falls below this. Uncertain subjects are still
            returned/labeled -- callers must not drop them from downstream
            analysis (md/group_analysis.md Step 4).

    Returns:
        ``(assignments, fit_result)`` where ``assignments`` has columns
        ``subject_id, spindle_per_min, spindle_per_min_SEM, spindle_group,
        group_probability, uncertain_assignment, gmm_input_scale,
        raw_scale_bic, log1p_scale_bic`` -- one row per subject with a finite
        ``spindle_per_min`` (non-finite subjects are the validation step's
        responsibility, not this function's, and are simply not fit here).
    """
    subject_ids = pd.Series(subject_ids).reset_index(drop=True)
    values = pd.Series(spindle_per_min).reset_index(drop=True).to_numpy(dtype=float)
    sems = pd.Series(spindle_per_min_sem).reset_index(drop=True)

    valid_mask = np.isfinite(values)
    n_excluded = int((~valid_mask).sum())
    if n_excluded:
        logger.warning(
            "%d subject(s) excluded from GMM fitting (non-finite spindle_per_min); "
            "validation should already have flagged these", n_excluded,
        )

    fit_result = fit_two_component_gmm(values[valid_mask], random_state=random_state)

    # Order raw component indices by their original-scale center (ascending):
    # `order[0]` is the low-rate component, `order[1]` the high-rate component.
    order = np.argsort(fit_result.centers_original_scale)
    remap = {int(order[0]): 0, int(order[1]): 1}

    ordered_labels = np.array([remap[int(label)] for label in fit_result.labels])
    ordered_probs = fit_result.probs[:, order]  # column 0 = P(low), column 1 = P(high)
    group_probability = ordered_probs.max(axis=1)
    spindle_group = np.where(ordered_labels == 0, LOW_LABEL, HIGH_LABEL)

    assignments = pd.DataFrame({
        "subject_id": subject_ids[valid_mask].to_numpy(),
        "spindle_per_min": values[valid_mask],
        "spindle_per_min_SEM": sems[valid_mask].to_numpy(),
        "spindle_group": spindle_group,
        "group_probability": group_probability,
        "uncertain_assignment": group_probability < probability_threshold,
        "gmm_input_scale": fit_result.scale,
        "raw_scale_bic": fit_result.bic_raw,
        "log1p_scale_bic": fit_result.bic_log1p,
    })
    return assignments, fit_result
