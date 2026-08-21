# tests/pipeline/test_phase.py
from __future__ import annotations

import numpy as np
import pytest

from infraslow.pipeline import io as pio
from infraslow.pipeline import phase as pph


def test_bin_center_angle_spans_full_circle_evenly():
    angles = pph.bin_center_angle(np.arange(1, 9))
    assert angles.shape == (8,)
    assert np.all(angles > -np.pi) and np.all(angles <= np.pi)
    # Evenly spaced by 2*pi/8
    diffs = np.diff(np.sort(angles))
    assert np.allclose(diffs, 2 * np.pi / 8, atol=1e-9)


def test_build_bout_phase_data_shapes(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    t_env, filtered = pio.load_temporal_isfs(subject_dir, "C3", "sigma")
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N2")
    spindles = pio.load_spindle_summary(subject_dir, "C3", "N2")
    event_times = spindles["Peak"].to_numpy()

    bouts_data = pph.build_bout_phase_data(t_env, filtered, bouts["spindle"], event_times)
    assert len(bouts_data) == 3
    for t, bins, events in bouts_data:
        assert t.shape == bins.shape
        assert np.all(t >= 0)  # bout-relative
        assert events.size <= 1  # this fixture has exactly one spindle peak per bout


def test_compute_subject_phase_features_below_min_events_returns_none():
    bouts_data = [(np.array([0.0, 1.0]), np.array([1, 2]), np.array([]))]
    assert pph.compute_subject_phase_features(bouts_data, min_events=5) is None


def test_compute_subject_phase_features_real_data(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    t_env, filtered = pio.load_temporal_isfs(subject_dir, "C3", "sigma")
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N2")
    spindles = pio.load_spindle_summary(subject_dir, "C3", "N2")
    event_times = spindles["Peak"].to_numpy()
    bouts_data = pph.build_bout_phase_data(t_env, filtered, bouts["spindle"], event_times)

    feats = pph.compute_subject_phase_features(bouts_data, min_events=1)
    assert feats is not None
    assert feats["event_count"] == 3
    assert len(feats["phase_bin_counts"]) == 8
    assert len(feats["phase_bin_rates"]) == 8
    assert -np.pi <= feats["mean_phase"] <= np.pi
    assert feats["preferred_phase"] == feats["mean_phase"]
    assert 0.0 <= feats["resultant_length"] <= 1.0


def test_pool_phase_distributions_sums_counts():
    d1 = dict(event_count=3, n_in_isfs=3, phase_bin_counts=[1, 0, 0, 0, 1, 0, 0, 1],
              phase_bin_rates=[33.3, 0, 0, 0, 33.3, 0, 0, 33.3], preferred_phase=0.1,
              mean_phase=0.1, resultant_length=0.5)
    d2 = dict(event_count=2, n_in_isfs=2, phase_bin_counts=[0, 1, 0, 0, 0, 0, 0, 1],
              phase_bin_rates=[0, 50, 0, 0, 0, 0, 0, 50], preferred_phase=-0.1,
              mean_phase=-0.1, resultant_length=0.4)
    pooled = pph.pool_phase_distributions([d1, d2])
    assert pooled["event_count"] == 5
    assert pooled["phase_bin_counts"] == [1, 1, 0, 0, 1, 0, 0, 2]
    assert pytest.approx(sum(pooled["phase_bin_rates"]), abs=1e-6) == 100.0 * pooled["n_in_isfs"] / pooled["event_count"]


def test_build_subject_phase_timeseries_absolute_time_and_shapes(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    t_env, filtered = pio.load_temporal_isfs(subject_dir, "C3", "sigma")
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N2")["all"]

    series = pph.build_subject_phase_timeseries(t_env, filtered, bouts)
    assert series["t"].shape == series["phase_bin"].shape == series["phase_angle"].shape
    assert series["t"].size > 0
    # Absolute time, not bout-relative: every sample must fall inside one of the bouts.
    for t in series["t"]:
        assert any(a <= t < b for a, b in bouts)
    assert np.all(np.diff(series["t"]) >= 0)  # sorted
    in_cycle = series["phase_bin"] > 0
    assert np.all(np.isnan(series["phase_angle"][~in_cycle]))
    assert not np.any(np.isnan(series["phase_angle"][in_cycle]))
    assert np.all((series["phase_angle"][in_cycle] > -np.pi) & (series["phase_angle"][in_cycle] <= np.pi))


def test_build_subject_phase_timeseries_empty_bouts_returns_empty_arrays(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    t_env, filtered = pio.load_temporal_isfs(subject_dir, "C3", "sigma")
    series = pph.build_subject_phase_timeseries(t_env, filtered, np.empty((0, 2)))
    assert series["t"].size == 0
    assert series["phase_bin"].size == 0
    assert series["phase_angle"].size == 0
