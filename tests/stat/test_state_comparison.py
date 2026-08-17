from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from infraslow.stat import state_comparison as stc


def test_mean_sem_known_values():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean, sem = stc.mean_sem(x)
    assert np.allclose(mean, [3.0, 4.0])
    assert np.allclose(sem, x.std(axis=0, ddof=1) / np.sqrt(3))


def test_describe_reports_mean_sd_median_iqr():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = stc.describe(x)
    assert d["mean"] == pytest.approx(3.0)
    assert d["median"] == pytest.approx(3.0)
    assert d["sd"] == pytest.approx(x.std(ddof=1))
    assert d["iqr"] == pytest.approx(np.percentile(x, 75) - np.percentile(x, 25))


def test_describe_ignores_nan():
    x = np.array([1.0, np.nan, 3.0, 5.0])
    d = stc.describe(x)
    assert d["mean"] == pytest.approx(3.0)


def test_paired_ttest_known_values_match_scipy():
    x = np.array([10.0, 12.0, 9.0, 15.0, 11.0])
    y = np.array([8.0, 10.0, 9.5, 13.0, 9.0])
    result = stc.paired_ttest(x, y)
    t_expected, p_expected = scipy_stats.ttest_rel(x, y)
    assert result["n_pairs"] == 5
    assert result["x_mean"] == pytest.approx(11.4)
    assert result["y_mean"] == pytest.approx(9.9)
    assert result["mean_diff"] == pytest.approx(1.5)
    assert result["t_stat"] == pytest.approx(t_expected)
    assert result["p_value"] == pytest.approx(p_expected)


def test_paired_ttest_drops_nan_pairs_and_reports_n_pairs():
    x = np.array([1.0, np.nan, 3.0, 4.0])
    y = np.array([1.5, 2.0, np.nan, 3.5])
    result = stc.paired_ttest(x, y)
    assert result["n_pairs"] == 2  # only indices 0 and 3 are valid in both


def test_paired_ttest_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        stc.paired_ttest(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_paired_ttest_fewer_than_two_pairs_returns_nan_stats():
    result = stc.paired_ttest(np.array([1.0]), np.array([2.0]))
    assert result["n_pairs"] == 1
    assert np.isnan(result["t_stat"])
    assert np.isnan(result["p_value"])


def test_paired_effect_size_matches_manual_cohens_dz():
    x = np.array([10.0, 12.0, 9.0, 15.0, 11.0])
    y = np.array([8.0, 10.0, 9.5, 13.0, 9.0])
    diff = x - y
    expected = diff.mean() / diff.std(ddof=1)
    assert stc.paired_effect_size(x, y) == pytest.approx(expected)


def test_paired_effect_size_nan_when_zero_variance():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0])  # constant difference (1.0) -> zero variance
    assert np.isnan(stc.paired_effect_size(x, y))


def test_fdr_correct_matches_statsmodels():
    p = [0.001, 0.01, 0.03, 0.04, 0.5]
    q = stc.fdr_correct(p)
    _, q_expected, _, _ = multipletests(p, method="fdr_bh")
    assert np.allclose(q, q_expected)


def test_fdr_correct_preserves_nan_positions():
    p = [0.01, np.nan, 0.2]
    q = stc.fdr_correct(p)
    assert np.isnan(q[1])
    assert not np.isnan(q[0])
    assert not np.isnan(q[2])


def test_compare_phase_bins_shape_and_columns():
    rng = np.random.default_rng(0)
    n2_rates = rng.normal(12.5, 2.0, size=(6, 8))
    n3_rates = rng.normal(12.5, 2.0, size=(6, 8))
    bin_centers = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    result = stc.compare_phase_bins(n2_rates, n3_rates, bin_centers)
    assert len(result) == 8
    assert list(result["phase_bin"]) == list(range(1, 9))
    assert {"phase_center", "N2_mean", "N3_mean", "mean_difference", "t_stat",
            "p_value", "q_value", "significant_FDR"}.issubset(result.columns)


def test_compare_phase_bins_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        stc.compare_phase_bins(np.zeros((5, 8)), np.zeros((5, 7)), np.zeros(8))


def test_compare_phase_bins_rejects_wrong_bin_centers_length():
    with pytest.raises(ValueError):
        stc.compare_phase_bins(np.zeros((5, 8)), np.zeros((5, 8)), np.zeros(7))


def test_compare_isfs_metrics_reports_expected_rows_and_dz():
    n2_df = pd.DataFrame({"peak_freq_hz": [0.02, 0.021, 0.019], "auc": [1.0, 1.1, 0.9]})
    n3_df = pd.DataFrame({"peak_freq_hz": [0.018, 0.02, 0.017], "auc": [0.8, 0.9, 0.7]})
    metrics = {"peak_freq_hz": "peak_frequency_hz", "auc": "auc"}
    result = stc.compare_isfs_metrics(n2_df, n3_df, metrics)
    assert list(result["Metric"]) == ["peak_frequency_hz", "auc"]
    assert "Cohen's dz" in result.columns
    assert "q" in result.columns
    assert (result["n_pairs"] == 3).all()


def test_compare_isfs_metrics_skips_missing_columns():
    n2_df = pd.DataFrame({"auc": [1.0, 1.1]})
    n3_df = pd.DataFrame({"auc": [0.8, 0.9]})
    metrics = {"auc": "auc", "not_a_column": "missing"}
    result = stc.compare_isfs_metrics(n2_df, n3_df, metrics)
    assert list(result["Metric"]) == ["auc"]


def test_circular_mean_and_resultant_uniform_gives_near_zero_resultant():
    angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    _mean_angle, resultant = stc.circular_mean_and_resultant(angles)
    assert resultant == pytest.approx(0.0, abs=1e-9)


def test_circular_mean_and_resultant_concentrated_angles():
    angles = np.array([0.1, 0.0, -0.1, 0.05])
    mean_angle, resultant = stc.circular_mean_and_resultant(angles)
    assert mean_angle == pytest.approx(0.0125, abs=1e-6)
    assert resultant > 0.99


def test_circular_mean_and_resultant_ignores_nan():
    angles = np.array([0.0, np.nan, 0.0])
    mean_angle, _resultant = stc.circular_mean_and_resultant(angles)
    assert mean_angle == pytest.approx(0.0)


def test_circular_mean_and_resultant_all_nan_returns_nan():
    mean_angle, resultant = stc.circular_mean_and_resultant(np.array([np.nan, np.nan]))
    assert np.isnan(mean_angle)
    assert np.isnan(resultant)
