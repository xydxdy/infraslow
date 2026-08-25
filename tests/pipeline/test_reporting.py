from __future__ import annotations

import numpy as np
import pandas as pd

from infraslow.pipeline import reporting as prep


def test_build_bout_features_df_columns():
    records = [
        dict(subject_id="S1", sleep_state="N2", channel="C3", bout_id=0,
             bout_start=0.0, bout_stop=250.0, bout_duration=250.0, peak_freq_hz=0.02),
    ]
    df = prep.build_bout_features_df(records)
    assert list(df.columns) == list(records[0].keys())
    assert len(df) == 1


def test_build_subject_state_features_df():
    records = [
        dict(subject_id="S1", sleep_state="N2", channel="C3", peak_freq_hz=0.02),
        dict(subject_id="S1", sleep_state="N3", channel="C3", peak_freq_hz=np.nan),
    ]
    df = prep.build_subject_state_features_df(records)
    assert len(df) == 2
    assert set(df["sleep_state"]) == {"N2", "N3"}


def test_join_demographics_and_sleep_stats_left_join_preserves_rows():
    subject_state_df = pd.DataFrame({
        "subject_id": ["S1", "S1", "S2"], "sleep_state": ["N2", "N3", "N2"],
    })
    demographics = pd.DataFrame({"ID": ["S1"], "Age": [40.0], "Gender": ["F"], "BMI": [22.0]})
    sleep_stats = pd.DataFrame({"id": ["S1"], "TST": [400.0]})
    out = prep.join_demographics_and_sleep_stats(subject_state_df, demographics, sleep_stats)
    assert len(out) == 3  # no row dropped, incl. S2 which is in neither side table
    assert out.loc[out["subject_id"] == "S1", "Age"].iloc[0] == 40.0
    assert np.isnan(out.loc[out["subject_id"] == "S2", "Age"].iloc[0])
    assert "TST" in out.columns


def test_join_demographics_and_sleep_stats_tolerates_empty_sleep_stats():
    subject_state_df = pd.DataFrame({"subject_id": ["S1"], "sleep_state": ["N2"]})
    demographics = pd.DataFrame({"ID": ["S1"], "Age": [40.0], "Gender": ["F"], "BMI": [22.0]})
    sleep_stats = pd.DataFrame({"id": pd.Series(dtype=str)})
    out = prep.join_demographics_and_sleep_stats(subject_state_df, demographics, sleep_stats)
    assert len(out) == 1


def test_cohort_summary_long_form():
    df = pd.DataFrame({
        "sleep_state": ["N2", "N2", "N2", "N3"],
        "peak_freq_hz": [0.02, 0.021, np.nan, 0.019],
        "auc": [0.4, 0.5, 0.6, 0.3],
    })
    out = prep.cohort_summary(df, value_cols=["peak_freq_hz", "auc"])
    row = out[(out["sleep_state"] == "N2") & (out["variable"] == "peak_freq_hz")].iloc[0]
    assert row["n"] == 2  # NaN excluded
    assert np.isclose(row["mean"], np.mean([0.02, 0.021]))
    for col in ("n", "mean", "std", "sem", "median", "q25", "q75", "min", "max"):
        assert col in out.columns


def test_cohort_summary_group_col_list_keeps_channels_separate():
    # subject_state_df has one row per (subject_id, sleep_state, channel) -- grouping
    # by sleep_state alone (the old default) pools every channel's rows together as if
    # they were independent observations of the same thing, inflating n and averaging
    # unrelated measurements (e.g. C3 vs O2) together. group_col=["sleep_state",
    # "channel"] must keep each channel's rows separate.
    df = pd.DataFrame({
        "sleep_state": ["N2", "N2", "N2", "N2"],
        "channel": ["C3", "C3", "O2", "O2"],
        "peak_freq_hz": [0.02, 0.022, 0.05, 0.052],
    })
    out = prep.cohort_summary(df, group_col=["sleep_state", "channel"], value_cols=["peak_freq_hz"])
    assert len(out) == 2  # one row per (sleep_state, channel) pair, not pooled into one
    assert set(out["channel"]) == {"C3", "O2"}

    c3 = out[(out["sleep_state"] == "N2") & (out["channel"] == "C3")].iloc[0]
    o2 = out[(out["sleep_state"] == "N2") & (out["channel"] == "O2")].iloc[0]
    assert c3["n"] == 2
    assert o2["n"] == 2
    assert np.isclose(c3["mean"], np.mean([0.02, 0.022]))
    assert np.isclose(o2["mean"], np.mean([0.05, 0.052]))
    assert not np.isclose(c3["mean"], o2["mean"])


def test_write_csv_creates_parent_dirs(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "nested" / "dir" / "out.csv"
    prep.write_csv(df, path)
    assert path.is_file()
    assert len(pd.read_csv(path)) == 2
