"""Two-sample statistical comparison of spindle-detection results.

Compares the per-event feature distributions of two spindle detectors -- e.g.
YASA vs. Luna, as produced in ``demo_compared_spindle.ipynb`` -- treating each
detector's detected events as an **independent sample**. The detectors run
different algorithms and find different (non-paired) events, so the natural
question is whether the *distribution* of each morphological feature (duration,
amplitude, frequency, ...) differs between them.

Every test comes from :mod:`statsmodels`:

* **Welch's t-test** (unequal variances, the default) on the feature means, with
  a confidence interval on the mean difference --
  :class:`statsmodels.stats.weightstats.CompareMeans`.
* **Brunner-Munzel rank test**, a distribution-free companion robust to the
  heavy-tailed, non-normal feature distributions spindle metrics usually have,
  plus its common-language effect size --
  :func:`statsmodels.stats.nonparametric.rank_compare_2indep`.
* **Multiple-comparison correction** across the tested features
  (:func:`statsmodels.stats.multitest.multipletests`).
* **Poisson two-sample rate test** for the event counts
  (:func:`statsmodels.stats.rates.test_poisson_2indep`).

statsmodels is imported lazily inside the functions, so importing this module
does not pull it in until a comparison is actually run.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# A feature to compare: either a single column name present in *both* summaries,
# or a ``(label, col_a, col_b)`` triple when the two detectors store the same
# quantity under different columns (e.g. filtered amplitude is ``AmpFiltered`` in
# YASA but ``Amplitude`` in Luna).
FeatureSpec = Union[str, Tuple[str, str, str]]

# Per-event spindle features shared by YASA's and Luna's ``.summary()`` tables:
# ``LunaSpindlesResult.summary()`` renames Luna's columns to these YASA names
# (see ``infraslow.processing.detection_luna``), so both detectors expose the
# same set. Ordered as usually reported.
SPINDLE_FEATURES: Tuple[str, ...] = (
    "Duration",
    "Amplitude",
    "Frequency",
    "Oscillations",
    "Symmetry",
)


def _to_summary(obj: Any, *, what: str) -> "pd.DataFrame":
    """Coerce a detector result (or its summary) to a per-event DataFrame.

    Accepts anything exposing ``.summary()`` (a YASA ``SpindlesResults`` or a
    :class:`~infraslow.processing.detection_luna.LunaSpindlesResult`) or a
    summary :class:`~pandas.DataFrame` directly.
    """
    if obj is None:
        raise ValueError(f"{what} is None -- no spindles to compare.")
    summary = obj.summary() if hasattr(obj, "summary") else obj
    if summary is None:
        raise ValueError(f"{what}.summary() is None -- no spindles to compare.")
    if not hasattr(summary, "columns"):
        raise TypeError(
            f"{what} must be a detector result with .summary() or a DataFrame; "
            f"got {type(obj).__name__}."
        )
    return summary


def _normalize_spec(spec: FeatureSpec) -> Tuple[str, str, str]:
    """Normalize a feature spec to ``(label, col_a, col_b)``."""
    if isinstance(spec, str):
        return spec, spec, spec
    if len(spec) == 3:
        return tuple(spec)  # type: ignore[return-value]
    raise ValueError(
        "Each feature must be a column name (str) or a (label, col_a, col_b) "
        f"triple; got {spec!r}."
    )


def _numeric(series: "pd.Series") -> "np.ndarray":
    """Feature column -> finite 1-D float array (non-numeric/NaN dropped)."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _cohen_d(x: "np.ndarray", y: "np.ndarray") -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    sd = float(np.sqrt(pooled))
    return float((x.mean() - y.mean()) / sd) if sd else float("nan")


def _stars(p: float) -> str:
    """Conventional significance marker for a (corrected) p-value."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _two_sample(
    x: "np.ndarray", y: "np.ndarray", *, equal_var: bool, alpha: float
) -> dict:
    """All two-sample statistics for one feature: Welch t + Brunner-Munzel."""
    from statsmodels.stats.nonparametric import (  # noqa: PLC0415 - lazy dep
        rank_compare_2indep,
    )
    from statsmodels.stats.weightstats import (  # noqa: PLC0415 - lazy dep
        CompareMeans,
        DescrStatsW,
    )

    usevar = "pooled" if equal_var else "unequal"
    cm = CompareMeans(DescrStatsW(x), DescrStatsW(y))
    t, p_t, dof = cm.ttest_ind(usevar=usevar)
    lo, hi = cm.tconfint_diff(alpha=alpha, usevar=usevar)

    # Distribution-free companion: robust to the non-normal feature distributions.
    # ``prob1`` is the common-language effect size P(a random group-A event >
    # a random group-B event), ties counted as 0.5.
    p_bm, prob_a_gt_b = float("nan"), float("nan")
    try:
        bm = rank_compare_2indep(x, y)
        p_bm, prob_a_gt_b = float(bm.pvalue), float(bm.prob1)
    except Exception:  # zero variance / all-tied / too few points
        pass

    return {
        "mean_diff": float(x.mean() - y.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "cohen_d": _cohen_d(x, y),
        "t": float(t),
        "dof": float(dof),
        "p_welch": float(p_t),
        "p_brunnermunzel": p_bm,
        "prob_a_gt_b": prob_a_gt_b,
    }


def compare_spindle_features(
    a: Any,
    b: Any,
    *,
    features: Optional[Sequence[FeatureSpec]] = None,
    labels: Tuple[str, str] = ("YASA", "Luna"),
    equal_var: bool = False,
    alpha: float = 0.05,
    correction: Optional[str] = "holm",
) -> "pd.DataFrame":
    """Compare two detectors' per-event spindle features, one row per feature.

    For every feature present (and numeric) in **both** summaries, runs a Welch
    two-sample t-test on the means and a Brunner-Munzel rank test on the
    distributions, reports descriptive stats and effect sizes, and (optionally)
    corrects the p-values for testing several features at once.

    Args:
        a, b: The two detector results to compare -- each a YASA
            ``SpindlesResults``, an
            :class:`~infraslow.processing.detection_luna.LunaSpindlesResult`, or a
            ``.summary()`` :class:`~pandas.DataFrame`. ``a`` is group 1 (``labels``
            names them for the output columns; positive ``mean_diff`` means
            ``a`` > ``b``).
        features: Features to test. Each is either a column name present in
            **both** summaries, or a ``(label, col_a, col_b)`` triple when the
            detectors store the same quantity under different columns (e.g.
            ``("Amplitude", "AmpFiltered", "Amplitude")`` to compare YASA's
            filtered amplitude against Luna's). Defaults to
            :data:`SPINDLE_FEATURES`, filtered to those present and numeric in
            both summaries.
        labels: ``(name_a, name_b)`` used to label the descriptive columns.
        equal_var: ``False`` (default) uses Welch's unequal-variance t-test;
            ``True`` uses the pooled-variance Student t-test.
        alpha: Significance level for the mean-difference confidence interval and
            the significance stars.
        correction: Multiple-comparison method passed to
            :func:`statsmodels.stats.multitest.multipletests` (e.g. ``"holm"``,
            ``"fdr_bh"``, ``"bonferroni"``). ``None`` skips correction (no
            ``p_welch_corr`` column, and stars use the raw p-value).

    Returns:
        A :class:`~pandas.DataFrame` indexed by feature with descriptive columns
        (``n_<a>``/``n_<b>``, ``mean_<a>``/``mean_<b>``, ``std_<a>``/``std_<b>``),
        the mean difference and its CI, ``cohen_d``, the Welch t-test
        (``t``/``dof``/``p_welch``), the corrected p-value (``p_welch_corr`` when
        ``correction`` is set), the Brunner-Munzel test
        (``p_brunnermunzel``/``prob_a_gt_b``) and a ``sig`` marker.

    Raises:
        ValueError: if either input has no spindles, or no shared numeric feature
            is available to compare.
    """
    la, lb = labels
    sa = _to_summary(a, what=f"'{la}'")
    sb = _to_summary(b, what=f"'{lb}'")

    if features is None:
        features = SPINDLE_FEATURES
    specs = [_normalize_spec(f) for f in features]
    usable = [
        (label, ca, cb)
        for (label, ca, cb) in specs
        if ca in sa.columns
        and cb in sb.columns
        and _numeric(sa[ca]).size >= 2
        and _numeric(sb[cb]).size >= 2
    ]
    if not usable:
        raise ValueError(
            "No shared numeric spindle feature to compare. Wanted "
            f"{[s[0] for s in specs]}; '{la}' has {list(sa.columns)}, '{lb}' has "
            f"{list(sb.columns)} (each feature needs >=2 finite values per side)."
        )

    rows = []
    for label, ca, cb in usable:
        x, y = _numeric(sa[ca]), _numeric(sb[cb])
        stats = _two_sample(x, y, equal_var=equal_var, alpha=alpha)
        rows.append(
            {
                "feature": label,
                f"n_{la}": x.size,
                f"n_{lb}": y.size,
                f"mean_{la}": float(x.mean()),
                f"mean_{lb}": float(y.mean()),
                f"std_{la}": float(x.std(ddof=1)),
                f"std_{lb}": float(y.std(ddof=1)),
                **stats,
            }
        )
    out = pd.DataFrame(rows).set_index("feature")

    p_for_stars = out["p_welch"]
    if correction is not None:
        from statsmodels.stats.multitest import (  # noqa: PLC0415 - lazy dep
            multipletests,
        )

        p_corr = multipletests(out["p_welch"].to_numpy(), alpha=alpha, method=correction)[1]
        out["p_welch_corr"] = p_corr
        p_for_stars = out["p_welch_corr"]
    out["sig"] = [_stars(p) for p in p_for_stars]
    return out


def compare_spindle_counts(
    a: Any,
    b: Any,
    *,
    exposure: Optional[Tuple[float, float]] = None,
    labels: Tuple[str, str] = ("YASA", "Luna"),
    alpha: float = 0.05,
    method: str = "score",
) -> "pd.Series":
    """Compare the two detectors' spindle *counts* as Poisson rates.

    Wraps :func:`statsmodels.stats.rates.test_poisson_2indep`. With no
    ``exposure`` the two raw counts are compared directly (equal exposure of 1,
    i.e. a test of the count ratio); pass ``exposure=(t_a, t_b)`` -- e.g. minutes
    of the included sleep stage seen by each detector -- to compare event
    *densities* instead, which is the fair comparison when the two ran over
    different amounts of usable signal.

    Args:
        a, b: The two detector results (or their ``.summary()`` DataFrames).
        exposure: ``(exposure_a, exposure_b)`` denominators for the rates. When
            ``None``, both default to 1 (compare raw counts).
        labels: ``(name_a, name_b)`` for the returned index.
        alpha: Significance level for the rate-ratio confidence interval.
        method: Test method forwarded to ``test_poisson_2indep`` (e.g. ``"score"``,
            ``"wald"``, ``"etest"``).

    Returns:
        A :class:`~pandas.Series` with the two counts, their exposures and rates,
        the rate ratio and its CI, the test statistic and p-value, and a ``sig``
        marker.
    """
    from statsmodels.stats.rates import (  # noqa: PLC0415 - lazy dep
        test_poisson_2indep,
    )

    la, lb = labels
    ca = len(_to_summary(a, what=f"'{la}'"))
    cb = len(_to_summary(b, what=f"'{lb}'"))
    ea, eb = (1.0, 1.0) if exposure is None else (float(exposure[0]), float(exposure[1]))
    if ea <= 0 or eb <= 0:
        raise ValueError(f"exposures must be positive; got {(ea, eb)}.")

    res = test_poisson_2indep(ca, ea, cb, eb, method=method, alternative="two-sided")
    rate_a, rate_b = ca / ea, cb / eb
    return pd.Series(
        {
            f"count_{la}": ca,
            f"count_{lb}": cb,
            f"exposure_{la}": ea,
            f"exposure_{lb}": eb,
            f"rate_{la}": rate_a,
            f"rate_{lb}": rate_b,
            "rate_ratio": (rate_a / rate_b) if rate_b else float("inf"),
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "sig": _stars(float(res.pvalue)),
        },
        name=f"{la}_vs_{lb}",
    )


__all__ = [
    "SPINDLE_FEATURES",
    "compare_spindle_features",
    "compare_spindle_counts",
]
