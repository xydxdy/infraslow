"""Paired N2 vs N3 statistical comparisons over `infraslow.pipeline`'s
per-subject/state records.

Every function here operates on already-computed values (subject-level ISFS
spectrum/phase features from `infraslow.pipeline.pipeline.find_paired_subject_data`,
or arrays built from them) -- no signal processing, spindle/slow-wave detection, or
spectral fitting happens here; that is entirely `infraslow.pipeline`'s and
`infraslow.processing.infraslow`'s job. Because N2 and N3 measurements come from the
same subjects, every comparison is a *paired* test (`scipy.stats.ttest_rel`), never
an independent-samples test, and NaNs are dropped pairwise (both members of a pair
must be valid) rather than column-wise, so `n_pairs` always reflects the actual
number of subjects contributing to that specific comparison.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def mean_sem(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """`(mean, sem)` of `x` along `axis` (SEM uses `ddof=1`) -- the project's standard
    "group average" convention (see `demo_infraslow_yasa_recheck.ipynb`'s Figure 5)."""
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=axis)
    n = x.shape[axis]
    sem = x.std(axis=axis, ddof=1) / np.sqrt(n)
    return mean, sem


def describe(x: np.ndarray) -> Dict[str, float]:
    """`mean`, `sd` (`ddof=1`), `median`, `iqr` (Q3-Q1) of `x`, ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return dict(mean=float("nan"), sd=float("nan"), median=float("nan"), iqr=float("nan"))
    q75, q25 = np.percentile(x, [75, 25])
    sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return dict(mean=float(x.mean()), sd=sd, median=float(np.median(x)), iqr=float(q75 - q25))


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Paired t-test between same-length, same-subject-order `x` and `y`.

    Pairs where either value is NaN are dropped before testing. Returns `n_pairs`
    (the actual number of valid pairs used), `x_mean`/`x_sd`, `y_mean`/`y_sd`
    (`ddof=1`), `mean_diff` (`mean(x - y)`), `t_stat`, `p_value`. `t_stat`/`p_value`
    are NaN when fewer than 2 valid pairs remain (a t-test is undefined) or the
    paired differences have zero variance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = int(x.size)
    if n < 2:
        return dict(
            n_pairs=n,
            x_mean=float(x.mean()) if n else float("nan"), x_sd=float("nan"),
            y_mean=float(y.mean()) if n else float("nan"), y_sd=float("nan"),
            mean_diff=float((x - y).mean()) if n else float("nan"),
            t_stat=float("nan"), p_value=float("nan"),
        )
    if (x - y).std(ddof=1) == 0:
        return dict(
            n_pairs=n,
            x_mean=float(x.mean()), x_sd=float(x.std(ddof=1)),
            y_mean=float(y.mean()), y_sd=float(y.std(ddof=1)),
            mean_diff=float((x - y).mean()),
            t_stat=float("nan"), p_value=float("nan"),
        )
    t_stat, p_value = stats.ttest_rel(x, y)
    return dict(
        n_pairs=n, x_mean=float(x.mean()), x_sd=float(x.std(ddof=1)),
        y_mean=float(y.mean()), y_sd=float(y.std(ddof=1)),
        mean_diff=float((x - y).mean()), t_stat=float(t_stat), p_value=float(p_value),
    )


def paired_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's dz for a paired comparison: `mean(x - y) / std(x - y, ddof=1)`.

    Same NaN-pair-dropping rule as `paired_ttest`. NaN if fewer than 2 valid pairs
    remain or the paired differences have zero variance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    diff = x[mask] - y[mask]
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diff.mean() / sd)


def fdr_correct(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values for `p_values`.

    NaN entries (e.g. from an underpowered `paired_ttest`) are excluded from the
    correction and returned as NaN at their original position, so a caller can zip
    `q` back against the same-length input it came from.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    if valid.any():
        _, q_valid, _, _ = multipletests(p[valid], method="fdr_bh")
        q[valid] = q_valid
    return q


def compare_phase_bins(
    n2_rates: np.ndarray, n3_rates: np.ndarray, bin_centers: np.ndarray,
) -> pd.DataFrame:
    """Paired t-test per ISFS phase bin, across subjects, BH-FDR corrected across bins.

    `n2_rates`/`n3_rates` are `(n_subjects, n_bins)` -- one subject's `phase_bin_rates`
    per row, same subject order in both (as produced by
    `infraslow.pipeline.pipeline.find_paired_subject_data`). `bin_centers` (length
    `n_bins`) is each bin's phase angle, e.g.
    `infraslow.pipeline.phase.bin_center_angle(np.arange(1, n_bins + 1))`.

    Returns one row per bin: `phase_bin` (1-indexed), `phase_center`, `N2_mean`,
    `N3_mean`, `mean_difference`, `n_pairs`, `t_stat`, `p_value`, `q_value`,
    `significant_FDR` (`q_value < 0.05`).
    """
    n2_rates = np.asarray(n2_rates, dtype=float)
    n3_rates = np.asarray(n3_rates, dtype=float)
    if n2_rates.shape != n3_rates.shape:
        raise ValueError(
            f"n2_rates and n3_rates must have the same shape, got {n2_rates.shape} vs {n3_rates.shape}"
        )
    n_bins = n2_rates.shape[1]
    bin_centers = np.asarray(bin_centers, dtype=float)
    if bin_centers.shape != (n_bins,):
        raise ValueError(f"bin_centers must have shape ({n_bins},), got {bin_centers.shape}")

    per_bin = [paired_ttest(n2_rates[:, k], n3_rates[:, k]) for k in range(n_bins)]
    q_values = fdr_correct([res["p_value"] for res in per_bin])

    rows = []
    for k, (res, q) in enumerate(zip(per_bin, q_values)):
        rows.append(dict(
            phase_bin=k + 1, phase_center=float(bin_centers[k]),
            N2_mean=res["x_mean"], N3_mean=res["y_mean"], mean_difference=res["mean_diff"],
            n_pairs=res["n_pairs"], t_stat=res["t_stat"], p_value=res["p_value"], q_value=float(q),
            significant_FDR=bool(not np.isnan(q) and q < 0.05),
        ))
    return pd.DataFrame(rows)


def compare_isfs_metrics(
    n2_df: pd.DataFrame, n3_df: pd.DataFrame, metrics: Mapping[str, str],
) -> pd.DataFrame:
    """Paired N2 vs N3 t-test for each metric in `metrics`, BH-FDR corrected across metrics.

    `metrics` maps a column name (present in both `n2_df` and `n3_df`, one row per
    subject, same subject order in both) to the display name used in the returned
    `Metric` column, e.g. `{"bi_gaussian_amp": "amplitude"}`. A column missing from
    either frame is silently skipped (not every ISFS metric is guaranteed to exist
    for every run).

    Returns columns: `Metric`, `N2 Mean`, `N2 SD`, `N3 Mean`, `N3 SD`,
    `Mean Difference`, `t`, `p`, `n_pairs`, `Cohen's dz`, `q`.
    """
    rows = []
    p_values = []
    for column, display_name in metrics.items():
        if column not in n2_df.columns or column not in n3_df.columns:
            continue
        x = n2_df[column].to_numpy(dtype=float)
        y = n3_df[column].to_numpy(dtype=float)
        res = paired_ttest(x, y)
        dz = paired_effect_size(x, y)
        rows.append({
            "Metric": display_name, "N2 Mean": res["x_mean"], "N2 SD": res["x_sd"],
            "N3 Mean": res["y_mean"], "N3 SD": res["y_sd"], "Mean Difference": res["mean_diff"],
            "t": res["t_stat"], "p": res["p_value"], "n_pairs": res["n_pairs"],
            "Cohen's dz": dz,
        })
        p_values.append(res["p_value"])

    q_values = fdr_correct(p_values)
    for row, q in zip(rows, q_values):
        row["q"] = float(q)

    columns = [
        "Metric", "N2 Mean", "N2 SD", "N3 Mean", "N3 SD", "Mean Difference",
        "t", "p", "n_pairs", "Cohen's dz", "q",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def circular_mean_and_resultant(angles: np.ndarray) -> Tuple[float, float]:
    """Circular mean angle (radians, in `(-pi, pi]`) and mean resultant length
    (`0`-`1`) of `angles`, ignoring NaNs -- the same mean-resultant-vector formula
    `infraslow.pipeline.phase.compute_subject_phase_features` and
    `infraslow.pipeline.pipeline.run_pipeline`'s cohort circular mean use,
    generalized to any array of angles (e.g. one subject-level `mean_phase` per
    subject, as a *supplemental* preferred-phase comparison -- distinct from
    `compare_phase_bins`'s ordinary paired t-tests on bin rates). `(nan, nan)` if no
    valid angle remains.
    """
    angles = np.asarray(angles, dtype=float)
    angles = angles[~np.isnan(angles)]
    if angles.size == 0:
        return float("nan"), float("nan")
    c, s = np.cos(angles).mean(), np.sin(angles).mean()
    return float(np.arctan2(s, c)), float(np.hypot(c, s))


__all__ = [
    "mean_sem",
    "describe",
    "paired_ttest",
    "paired_effect_size",
    "fdr_correct",
    "compare_phase_bins",
    "compare_isfs_metrics",
    "circular_mean_and_resultant",
]
