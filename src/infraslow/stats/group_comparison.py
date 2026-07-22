"""Descriptive statistics, test selection, and BH-FDR correction for the
N2-C3 infraslow-parameter comparison between spindle-rate groups
(md/group_analysis.md Step 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .effect_sizes import hedges_g, rank_biserial_cliffs_delta

logger = logging.getLogger(__name__)

__all__ = [
    "SKEW_THRESHOLD",
    "OUTLIER_IQR_MULTIPLIER",
    "MIN_N_FOR_DISTRIBUTION_CHECKS",
    "ParameterComparison",
    "descriptive_stats",
    "choose_test",
    "compare_parameter",
    "compare_parameters",
]

#: |skewness| at/above this is "strongly skewed" -> prefer Mann-Whitney U.
SKEW_THRESHOLD = 1.0
#: a value farther than this many IQRs from the nearest quartile is an "extreme outlier".
OUTLIER_IQR_MULTIPLIER = 3.0
#: below this many finite values per group, skewness/outlier diagnostics are unreliable
#: -> fall back to the distribution-free Mann-Whitney U rather than trusting them.
MIN_N_FOR_DISTRIBUTION_CHECKS = 8


def descriptive_stats(values: np.ndarray) -> Dict[str, float]:
    """``n, missing_n, mean, standard_deviation, median, q1, q3, minimum, maximum``.

    ``values`` may contain NaN; ``missing_n`` counts them (and +/-inf) and every
    other statistic is computed on the finite subset only.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    missing_n = int(values.size - finite.size)
    if finite.size == 0:
        return dict(n=0, missing_n=missing_n, mean=np.nan, standard_deviation=np.nan,
                    median=np.nan, q1=np.nan, q3=np.nan, minimum=np.nan, maximum=np.nan)
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    return dict(
        n=int(finite.size), missing_n=missing_n,
        mean=float(finite.mean()),
        standard_deviation=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        median=float(median), q1=float(q1), q3=float(q3),
        minimum=float(finite.min()), maximum=float(finite.max()),
    )


def _n_extreme_outliers(values: np.ndarray) -> int:
    """Count of values farther than :data:`OUTLIER_IQR_MULTIPLIER` * IQR from the nearest quartile."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
    return int(((values < lo) | (values > hi)).sum())


def choose_test(low: np.ndarray, high: np.ndarray) -> str:
    """``"welch"`` or ``"mann_whitney"`` from several distributional signals together.

    md/group_analysis.md Step 5 explicitly forbids choosing the test from a
    Shapiro-Wilk p-value alone, so this instead falls back to Mann-Whitney U
    whenever *any* of the following holds: either group is smaller than
    :data:`MIN_N_FOR_DISTRIBUTION_CHECKS` (skew/outlier estimates are unreliable
    below that), either group is strongly skewed
    (``|skew| >= SKEW_THRESHOLD``), or either group has an extreme outlier
    (:func:`_n_extreme_outliers`). Otherwise Welch's t-test is used.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if low.size < MIN_N_FOR_DISTRIBUTION_CHECKS or high.size < MIN_N_FOR_DISTRIBUTION_CHECKS:
        return "mann_whitney"
    if abs(stats.skew(low)) >= SKEW_THRESHOLD or abs(stats.skew(high)) >= SKEW_THRESHOLD:
        return "mann_whitney"
    if _n_extreme_outliers(low) > 0 or _n_extreme_outliers(high) > 0:
        return "mann_whitney"
    return "welch"


@dataclass
class ParameterComparison:
    parameter: str
    low: Dict[str, float]
    high: Dict[str, float]
    test: str
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float


def compare_parameter(parameter: str, low: np.ndarray, high: np.ndarray) -> ParameterComparison:
    """Descriptive stats + the chosen test + its effect size for one parameter.

    Effect-size and test direction is always ``high - low``
    (md/group_analysis.md Step 5).
    """
    low_finite = np.asarray(low, dtype=float)
    low_finite = low_finite[np.isfinite(low_finite)]
    high_finite = np.asarray(high, dtype=float)
    high_finite = high_finite[np.isfinite(high_finite)]

    low_desc = descriptive_stats(low)
    high_desc = descriptive_stats(high)

    if low_finite.size < 2 or high_finite.size < 2:
        logger.warning(
            "parameter %s: fewer than 2 finite values in a group (low n=%d, high n=%d); test skipped",
            parameter, low_finite.size, high_finite.size,
        )
        return ParameterComparison(parameter, low_desc, high_desc, "none", np.nan, np.nan, "none", np.nan)

    test = choose_test(low_finite, high_finite)
    if test == "welch":
        statistic, p_value = stats.ttest_ind(high_finite, low_finite, equal_var=False)
        effect_name = "hedges_g"
        effect = hedges_g(low_finite, high_finite)
    else:
        statistic, p_value = stats.mannwhitneyu(high_finite, low_finite, alternative="two-sided")
        effect_name = "rank_biserial_cliffs_delta"
        effect = rank_biserial_cliffs_delta(low_finite, high_finite)

    return ParameterComparison(
        parameter, low_desc, high_desc, test, float(statistic), float(p_value), effect_name, float(effect),
    )


def compare_parameters(
    df: pd.DataFrame,
    group_col: str,
    parameters: Sequence[str],
    *,
    low_label: str = "low_spindle_rate",
    high_label: str = "high_spindle_rate",
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per parameter: descriptive stats, chosen test, effect size, and BH-FDR q-value.

    Args:
        df: One row per subject, with a numeric column per entry in
            ``parameters`` and a ``group_col`` column containing exactly
            ``low_label``/``high_label``.
        parameters: Parameter (column) names to test, in output order.
        fdr_alpha: Benjamini-Hochberg FDR level; a row's ``significant_fdr`` is
            ``True`` only if its ``q_value`` is finite and below this.

    Returns:
        DataFrame with columns: ``parameter, low_group_n, high_group_n,
        low_mean, low_sd, low_median, low_q1, low_q3, high_mean, high_sd,
        high_median, high_q1, high_q3, test, statistic, p_value, q_value,
        effect_size_name, effect_size, significant_fdr`` -- one row per
        parameter, in the same order as ``parameters``.
    """
    low_mask = df[group_col] == low_label
    high_mask = df[group_col] == high_label

    comparisons = [
        compare_parameter(p, df.loc[low_mask, p].to_numpy(), df.loc[high_mask, p].to_numpy())
        for p in parameters
    ]

    p_values = np.array([c.p_value for c in comparisons], dtype=float)
    valid = np.isfinite(p_values)
    q_values = np.full_like(p_values, np.nan)
    if valid.any():
        _, q_valid, _, _ = multipletests(p_values[valid], alpha=fdr_alpha, method="fdr_bh")
        q_values[valid] = q_valid

    rows: List[Dict[str, object]] = []
    for comparison, q_value in zip(comparisons, q_values):
        low, high = comparison.low, comparison.high
        rows.append({
            "parameter": comparison.parameter,
            "low_group_n": low["n"], "high_group_n": high["n"],
            "low_mean": low["mean"], "low_sd": low["standard_deviation"],
            "low_median": low["median"], "low_q1": low["q1"], "low_q3": low["q3"],
            "high_mean": high["mean"], "high_sd": high["standard_deviation"],
            "high_median": high["median"], "high_q1": high["q1"], "high_q3": high["q3"],
            "test": comparison.test, "statistic": comparison.statistic,
            "p_value": comparison.p_value, "q_value": q_value,
            "effect_size_name": comparison.effect_size_name, "effect_size": comparison.effect_size,
            "significant_fdr": bool(np.isfinite(q_value) and q_value < fdr_alpha),
        })
    return pd.DataFrame(rows)
