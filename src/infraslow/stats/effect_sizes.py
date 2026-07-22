"""Effect-size helpers for the N2-C3 spindle-rate group comparison.

Both effect sizes are oriented ``high_spindle_rate - low_spindle_rate``
(md/group_analysis.md Step 5): a positive value always means "higher in the
high-spindle-rate group".
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["hedges_g", "rank_biserial_cliffs_delta"]


def hedges_g(low: np.ndarray, high: np.ndarray) -> float:
    """Hedges' g (bias-corrected Cohen's d) for ``high - low``.

    Uses the pooled standard deviation of both groups and the small-sample
    correction factor ``J = 1 - 3 / (4*(n1+n2) - 9)`` (Hedges, 1981). Returns
    ``NaN`` if either group has fewer than 2 finite values or both groups have
    zero pooled variance (effect size undefined).
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    n1, n2 = low.size, high.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = low.std(ddof=1), high.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return float("nan")
    d = (high.mean() - low.mean()) / pooled_sd
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(d * j)


def rank_biserial_cliffs_delta(low: np.ndarray, high: np.ndarray) -> float:
    """Cliff's delta / rank-biserial correlation for ``high - low``.

    ``delta = P(high > low) - P(high < low)``, derived from the Mann-Whitney U
    statistic (ties count as 0.5 per pair) so this stays O(n log n) via
    :func:`scipy.stats.mannwhitneyu`'s rank-based implementation rather than an
    explicit O(n1*n2) pairwise-comparison matrix.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    n1, n2 = high.size, low.size
    if n1 == 0 or n2 == 0:
        return float("nan")
    u_high, _ = stats.mannwhitneyu(high, low, alternative="two-sided")
    return float(2.0 * u_high / (n1 * n2) - 1.0)
