"""Statistical comparison of infraslow sigma-power oscillations.

Companion to :mod:`infraslow.processing.infraslow`, for the YASA-vs-Luna
comparison in ``demo_compared_infraslow.ipynb``. A single recording yields one
infraslow spectrum per detector, so the questions here are within-subject:

* **Is a detector's infraslow rhythm real?** -- :func:`spindle_infraslow_stats`
  tests the infraslow-band power of a detector's spindle-rate spectrum against an
  **inter-spindle-interval shuffle** null: reshuffling the gaps between spindles
  keeps the count and ISI distribution but destroys the ~50 s clustering, so a
  band power the shuffles rarely reach means the clustering is real.
  :func:`compare_spindle_infraslow` runs it for both detectors side by side.

* **Do the two detectors fluctuate together?** --
  :func:`associate_spindle_rate` regresses one detector's spindle-rate series on
  the other's with :mod:`statsmodels` OLS, reporting the correlation, slope and
  its p-value.

statsmodels is imported lazily; the permutation test is pure NumPy.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ..processing.infraslow import (
    DEFAULT_INFRASLOW_BAND,
    DEFAULT_SF_ENV,
    DEFAULT_WINDOW_SEC,
    infraslow_spectrum,
    spindle_rate_series,
)
from .spindle import _stars


def _event_times(obj: Any, *, mark: str, what: str) -> np.ndarray:
    """Spindle event times (s) from a detector result, its summary, or an array."""
    if obj is None:
        raise ValueError(f"{what} is None -- no spindles.")
    if hasattr(obj, "summary") or hasattr(obj, "columns"):
        summary = obj.summary() if hasattr(obj, "summary") else obj
        col = mark if mark in summary.columns else ("Peak" if "Peak" in summary.columns else "Start")
        if col not in summary.columns:
            raise KeyError(
                f"{what} summary has no '{mark}'/'Peak'/'Start' column; got "
                f"{list(summary.columns)}."
            )
        arr = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
    else:
        arr = np.asarray(obj, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def spindle_infraslow_stats(
    spindles: Any,
    *,
    duration_sec: float,
    mark: str = "Peak",
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: float = 10.0,
    band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND,
    window_sec: float = DEFAULT_WINDOW_SEC,
    n_perm: int = 1000,
    random_state: Optional[int] = 0,
) -> "pd.Series":
    """Test the infraslow oscillation of one detector's spindle train.

    Builds the detector's spindle-rate series, measures its infraslow-band power,
    and compares that to a null built by **shuffling the inter-spindle intervals**
    ``n_perm`` times (same count and ISI distribution, clustering destroyed).

    Args:
        spindles: A detector result (``.summary()``), a summary DataFrame, or a
            1-D array of event times (s).
        duration_sec: Recording length (s) for the rate-series time base.
        mark: Event column to use when ``spindles`` is a result/summary.
        sf_env, smooth_sec, band, window_sec: Forwarded to
            :func:`~infraslow.processing.infraslow.spindle_rate_series` /
            :func:`~infraslow.processing.infraslow.infraslow_spectrum`.
        n_perm: Number of ISI-shuffle permutations.
        random_state: Seed for the shuffles (reproducible).

    Returns:
        A :class:`~pandas.Series`: ``n_spindles``, ``peak_freq``, ``peak_power``,
        ``full_peak_freq`` (the reference ``get_iso`` whole-spectrum peak),
        ``band_power``, ``rel_band_power``, the permutation ``p_value`` and a
        ``sig`` marker.
    """
    peaks = np.sort(_event_times(spindles, mark=mark, what="spindles"))
    n = peaks.size

    def _band_power(pk: np.ndarray) -> Tuple[float, "Any"]:
        _, rate = spindle_rate_series(
            pk, duration_sec=duration_sec, sf_env=sf_env, smooth_sec=smooth_sec
        )
        spec = infraslow_spectrum(rate, sf_env, band=band, window_sec=window_sec)
        return spec.band_power, spec

    obs_bp, obs = _band_power(peaks)

    p_value = float("nan")
    if n >= 3 and np.isfinite(obs_bp):
        rng = np.random.default_rng(random_state)
        isi = np.diff(peaks)
        start = float(peaks[0])
        ge = 0
        for _ in range(n_perm):
            shuffled = start + np.concatenate([[0.0], np.cumsum(rng.permutation(isi))])
            bp, _ = _band_power(shuffled)
            if bp >= obs_bp:
                ge += 1
        p_value = (1.0 + ge) / (n_perm + 1.0)

    return pd.Series(
        {
            "n_spindles": n,
            "peak_freq": obs.peak_freq,
            "peak_power": obs.peak_power,
            "full_peak_freq": obs.full_peak_freq,
            "band_power": obs.band_power,
            "rel_band_power": obs.rel_band_power,
            "p_value": p_value,
            "sig": _stars(p_value),
        },
        name="infraslow",
    )


def compare_spindle_infraslow(
    a: Any,
    b: Any,
    *,
    duration_sec: float,
    labels: Tuple[str, str] = ("YASA", "Luna"),
    mark: str = "Peak",
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: float = 10.0,
    band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND,
    window_sec: float = DEFAULT_WINDOW_SEC,
    n_perm: int = 1000,
    random_state: Optional[int] = 0,
) -> "pd.DataFrame":
    """Per-detector infraslow statistics for two detectors, one row each.

    Runs :func:`spindle_infraslow_stats` on ``a`` and ``b`` and stacks the
    results, so YASA and Luna can be compared on infraslow peak frequency,
    band power and its significance. See :func:`associate_spindle_rate` for a
    direct test of whether the two rate series co-fluctuate.
    """
    rows = {}
    for label, obj in zip(labels, (a, b)):
        rows[label] = spindle_infraslow_stats(
            obj,
            duration_sec=duration_sec,
            mark=mark,
            sf_env=sf_env,
            smooth_sec=smooth_sec,
            band=band,
            window_sec=window_sec,
            n_perm=n_perm,
            random_state=random_state,
        )
    out = pd.DataFrame(rows).T
    out.index.name = "detector"
    # Keep integer/count column tidy.
    out["n_spindles"] = out["n_spindles"].astype(int)
    return out


def associate_spindle_rate(
    a: Any,
    b: Any,
    *,
    duration_sec: float,
    labels: Tuple[str, str] = ("YASA", "Luna"),
    mark: str = "Peak",
    sf_env: float = DEFAULT_SF_ENV,
    smooth_sec: float = 10.0,
) -> "pd.Series":
    """Do the two detectors' spindle-rate series co-fluctuate? (statsmodels OLS).

    Builds both detectors' spindle-rate series on a common time base and
    regresses ``b`` on ``a`` (``rate_b ~ const + rate_a``) with
    :class:`statsmodels.regression.linear_model.OLS`, reporting the Pearson
    correlation, R^2, slope and the slope's p-value -- a high positive
    correlation means the two detectors agree on *when* spindles cluster.

    Returns:
        A :class:`~pandas.Series` with ``n``, ``pearson_r``, ``r_squared``,
        ``slope``, ``p_value`` and a ``sig`` marker.
    """
    import statsmodels.api as sm  # noqa: PLC0415 - lazy, optional dependency

    la, lb = labels
    _, ra = spindle_rate_series(
        _event_times(a, mark=mark, what=f"'{la}'"),
        duration_sec=duration_sec,
        sf_env=sf_env,
        smooth_sec=smooth_sec,
    )
    _, rb = spindle_rate_series(
        _event_times(b, mark=mark, what=f"'{lb}'"),
        duration_sec=duration_sec,
        sf_env=sf_env,
        smooth_sec=smooth_sec,
    )
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]

    model = sm.OLS(rb, sm.add_constant(ra)).fit()
    slope = float(model.params[-1])
    r2 = float(model.rsquared)
    r = float(np.sign(slope) * np.sqrt(max(r2, 0.0)))
    p = float(model.pvalues[-1])
    return pd.Series(
        {
            "n": int(n),
            "pearson_r": r,
            "r_squared": r2,
            "slope": slope,
            "p_value": p,
            "sig": _stars(p),
        },
        name=f"{la}~{lb}",
    )


__all__ = [
    "spindle_infraslow_stats",
    "compare_spindle_infraslow",
    "associate_spindle_rate",
]
