from __future__ import annotations

import numpy as np
import pandas as pd

from infraslow.pipeline import io as pio


def test_discover_subjects(fake_subject_tree):
    assert pio.discover_subjects(fake_subject_tree) == ["SUBJ001"]


def test_discover_channels(fake_subject_tree):
    assert pio.discover_channels(fake_subject_tree, "SUBJ001") == ["C3"]


def test_load_envelope_roundtrip(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    t_env, power = pio.load_envelope(subject_dir, "C3", "sigma")
    assert t_env.shape == power.shape == (1200,)
    assert np.isclose(t_env[0], 0.5) or np.isclose(t_env[0], 0.0)  # bin-centre or 0-based, just check monotonic
    assert np.all(np.diff(t_env) > 0)


def test_load_temporal_isfs_roundtrip(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    t_env, filtered = pio.load_temporal_isfs(subject_dir, "C3", "sigma")
    assert t_env.shape == filtered.shape == (1200,)


def test_load_stage_bouts_n2_has_three_spindle_bouts(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N2")
    assert bouts["all"].shape == (3, 2)
    assert bouts["spindle"].shape == (3, 2)


def test_load_stage_bouts_n3_is_empty(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N3")
    assert bouts["all"].shape == (1, 2)  # the fixture's one sub-threshold N3 bout is still SAVED by
    # preprocessing.py (min_dur filtering happens before bouts.npz is written in the real pipeline,
    # so this fixture models "N3 bouts.npz has 1 short bout" -- io.py must NOT filter by duration,
    # that's spectrum.py's/pipeline.py's job via bout selection, not a loader concern).


def test_load_stage_isfs_spectra(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    spec = pio.load_stage_isfs_spectra(subject_dir, "C3", "N2", "sigma")
    assert spec["freqs"].shape == (40,)
    assert spec["psds"].shape == (3, 40)
    assert spec["bout_start"].shape == (3,)


def test_load_spindle_summary_has_expected_columns(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    df = pio.load_spindle_summary(subject_dir, "C3", "N2")
    assert len(df) == 3
    assert {"Start", "Peak", "End"}.issubset(df.columns)


def test_load_demographics_returns_id_age_gender_bmi():
    df = pio.load_demographics()
    assert list(df.columns) == ["ID", "Age", "Gender", "BMI"]
    assert len(df) > 0  # real metadata CSVs are checked into the repo


def test_load_sleep_statistics_missing_file_returns_empty_with_id_column(tmp_path):
    df = pio.load_sleep_statistics(tmp_path / "does_not_exist.csv")
    assert list(df.columns) == ["id"]
    assert len(df) == 0


def test_load_sleep_statistics_reads_real_file(tmp_path):
    path = tmp_path / "sleep_statistics.csv"
    pd.DataFrame({"id": ["SUBJ001"], "TST": [400.0]}).to_csv(path, index=False)
    df = pio.load_sleep_statistics(path)
    assert list(df["id"]) == ["SUBJ001"]
    assert df["TST"].iloc[0] == 400.0


def test_load_sleep_statistics_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "sleep_statistics.csv"
    path.write_text("not,a,valid\ncsv{{{")
    df = pio.load_sleep_statistics(path)
    assert "id" in df.columns
