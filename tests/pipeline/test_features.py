from __future__ import annotations

import numpy as np
import pytest

from infraslow.pipeline import features as pft
from infraslow.pipeline import io as pio


def test_compute_spindle_features_real_data(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    summary = pio.load_spindle_summary(subject_dir, "C3", "N2")
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N2")
    feats = pft.compute_spindle_features(summary, bouts["all"])
    assert feats["spindle_count"] == 3
    total_min = (bouts["all"][:, 1] - bouts["all"][:, 0]).sum() / 60.0
    assert feats["spindle_density_per_min"] == pytest.approx(3 / total_min)
    assert feats["spindle_mean_duration_s"] == pytest.approx(2.0)
    assert not np.isnan(feats["spindle_mean_amplitude"])
    assert not np.isnan(feats["spindle_mean_frequency"])


def test_compute_spindle_features_empty_summary():
    import pandas as pd
    summary = pd.DataFrame(columns=["Start", "Peak", "End"])
    bouts = np.array([[0.0, 250.0]])
    feats = pft.compute_spindle_features(summary, bouts)
    assert feats["spindle_count"] == 0
    assert feats["spindle_density_per_min"] == 0.0
    assert np.isnan(feats["spindle_mean_amplitude"])


def test_compute_spindle_features_no_bouts_density_is_nan():
    import pandas as pd
    summary = pd.DataFrame(columns=["Start", "Peak", "End"])
    bouts = np.empty((0, 2))
    feats = pft.compute_spindle_features(summary, bouts)
    assert np.isnan(feats["spindle_density_per_min"])


def test_compute_slow_wave_features_real_data(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    summary = pio.load_sw_summary(subject_dir, "C3", "N2")
    bouts = pio.load_stage_sw_bouts(subject_dir, "C3", "N2")
    feats = pft.compute_slow_wave_features(summary, bouts["all"])
    assert feats["slow_wave_count"] == 3
    assert not np.isnan(feats["slow_wave_mean_ptp"])
    assert not np.isnan(feats["slow_wave_mean_frequency"])


def test_compute_band_power_feature_real_data(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    t_env, power = pio.load_envelope(subject_dir, "C3", "sigma")
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N2")
    feats = pft.compute_band_power_feature(t_env, power, bouts["all"], name="sigma")
    assert "sigma_power_db" in feats
    assert not np.isnan(feats["sigma_power_db"])


def test_compute_band_power_feature_no_bouts_is_nan(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    t_env, power = pio.load_envelope(subject_dir, "C3", "delta")
    feats = pft.compute_band_power_feature(t_env, power, np.empty((0, 2)), name="delta")
    assert np.isnan(feats["delta_power_db"])
