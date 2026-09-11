from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import plot_hypnodensity_isfs_phase as phi


# --------------------------------------------------------------------------- #
# _resolve_data_dir -- derives preprocessing.py's <output-dir>/<N>s/data layout
# --------------------------------------------------------------------------- #
def test_resolve_data_dir_derives_from_preprocess_dir_and_epoch():
    preprocess_dir = Path("/scratch/users/chaisaen/usleep_processed_data")
    assert phi._resolve_data_dir(preprocess_dir, None, 3.0) == preprocess_dir / "3s" / "data"


def test_resolve_data_dir_non_integer_epoch():
    preprocess_dir = Path("/scratch/users/chaisaen/usleep_processed_data")
    assert phi._resolve_data_dir(preprocess_dir, None, 2.5) == preprocess_dir / "2.5s" / "data"


def test_resolve_data_dir_explicit_override_wins():
    # An explicit --data-dir always wins over --preprocess-dir/--hypno-epoch-sec,
    # for pointing directly at a non-standard data/ folder.
    preprocess_dir = Path("/scratch/users/chaisaen/usleep_processed_data")
    override = Path("/scratch/users/chaisaen/some_other_data_dir")
    assert phi._resolve_data_dir(preprocess_dir, override, 3.0) == override


# --------------------------------------------------------------------------- #
# _events_csv_path -- matches preprocessing.py's events_<channel>.csv naming
# --------------------------------------------------------------------------- #
def test_events_csv_path_derives_from_preprocess_dir_channel_and_epoch():
    preprocess_dir = Path("/scratch/users/chaisaen/usleep_processed_data")
    assert phi._events_csv_path(preprocess_dir, "C3", 3.0) == (
        preprocess_dir / "3s" / "events_C3.csv"
    )


# --------------------------------------------------------------------------- #
# _filter_subjects_with_events -- pre-filter candidates using preprocessing.py's
# already-computed events_<channel>.csv counts, before submitting them to the
# worker pool
# --------------------------------------------------------------------------- #
def _events_df(rows):
    return pd.DataFrame(rows)


def test_filter_keeps_subject_with_events_in_any_requested_state():
    events_df = _events_df([
        {"id": "A", "N2_Spindle": 0, "N2_SW": 0, "N3_Spindle": 2, "N3_SW": 0},
        {"id": "B", "N2_Spindle": 0, "N2_SW": 0, "N3_Spindle": 0, "N3_SW": 0},
    ])
    kept = phi._filter_subjects_with_events(["A", "B"], events_df, ["N2", "N3"], "spindle")
    assert kept == ["A"]


def test_filter_drops_subject_missing_from_events_csv():
    events_df = _events_df([{"id": "A", "N2_Spindle": 1, "N2_SW": 0}])
    kept = phi._filter_subjects_with_events(["A", "B"], events_df, ["N2"], "spindle")
    assert kept == ["A"]


def test_filter_drops_subject_with_blank_cell_not_a_guessed_zero():
    # Blank (NaN) means that channel raised outright during preprocessing --
    # never treated as a usable event, but distinct from a confirmed 0.
    events_df = _events_df([{"id": "A", "N2_Spindle": float("nan"), "N2_SW": 0}])
    kept = phi._filter_subjects_with_events(["A"], events_df, ["N2"], "spindle")
    assert kept == []


def test_filter_uses_sw_column_for_sw_event():
    events_df = _events_df([{"id": "A", "N2_Spindle": 0, "N2_SW": 3}])
    assert phi._filter_subjects_with_events(["A"], events_df, ["N2"], "sw") == ["A"]
    assert phi._filter_subjects_with_events(["A"], events_df, ["N2"], "spindle") == []


def test_filter_preserves_candidate_order():
    events_df = _events_df([
        {"id": "A", "N2_Spindle": 1},
        {"id": "B", "N2_Spindle": 1},
        {"id": "C", "N2_Spindle": 1},
    ])
    assert phi._filter_subjects_with_events(["C", "A", "B"], events_df, ["N2"], "spindle") == ["C", "A", "B"]


# --------------------------------------------------------------------------- #
# _parse_state_groups -- "+"-joined --states tokens pool stages into one group
# --------------------------------------------------------------------------- #
def test_parse_state_groups_bare_stages_are_singleton_groups():
    assert phi._parse_state_groups(["N2", "N3"]) == [("N2", ("N2",)), ("N3", ("N3",))]


def test_parse_state_groups_plus_joined_token_is_one_merged_group():
    assert phi._parse_state_groups(["N2+N3"]) == [("N2+N3", ("N2", "N3"))]


def test_parse_state_groups_mixes_bare_and_merged():
    assert phi._parse_state_groups(["N2", "N3", "N2+N3"]) == [
        ("N2", ("N2",)), ("N3", ("N3",)), ("N2+N3", ("N2", "N3")),
    ]


def test_parse_state_groups_rejects_duplicate_component_within_a_group():
    with pytest.raises(ValueError, match="duplicate"):
        phi._parse_state_groups(["N2+N2"])


def test_parse_state_groups_rejects_duplicate_group_label():
    with pytest.raises(ValueError, match="duplicate"):
        phi._parse_state_groups(["N2", "N2"])


def test_parse_state_groups_all_leaf_stages_dedupes_across_groups():
    groups = phi._parse_state_groups(["N2", "N2+N3"])
    assert phi._all_leaf_stages(groups) == ["N2", "N3"]


# --------------------------------------------------------------------------- #
# subject_isfs_spectrum_features -- pooling a merged group's ISFS spectra must
# tolerate a component stage with zero bouts at all
# --------------------------------------------------------------------------- #
def test_subject_isfs_spectrum_features_pools_when_one_stage_has_no_bouts(monkeypatch):
    # preprocessing.py's compute_bout_spectra saves an empty (0, 0)-shaped
    # freqs/psds for a stage with zero bouts at all -- not the real (0, n_freqs)
    # grid -- so pooling a real stage (N2) with an all-empty one (N3) must not
    # crash on the resulting shape mismatch (regression: caught by a real dry
    # run against usleep_dryrun_100 data).
    freqs = np.linspace(0.0025, 0.1, 40)
    n2_bouts = np.array([[0.0, 300.0], [400.0, 700.0]])

    def fake_load_bouts(subject_dir, channel, stage, hypno_epoch_sec):
        if stage == "N2":
            return {"all": n2_bouts, "spindle": n2_bouts}
        return {"all": np.empty((0, 2)), "spindle": np.empty((0, 2))}

    def fake_load_isfs(subject_dir, channel, stage, band, hypno_epoch_sec):
        if stage == "N2":
            return {"freqs": freqs, "psds": np.ones((2, 40)), "bout_start": np.array([0.0, 400.0])}
        return {"freqs": np.empty(0), "psds": np.empty((0, 0)), "bout_start": np.empty(0)}

    # _EVENT_SPECS["spindle"]["load_bouts"] captured pio.load_stage_bouts at
    # import time, so patch the spec entry itself, not the pio module attribute.
    monkeypatch.setitem(phi._EVENT_SPECS["spindle"], "load_bouts", fake_load_bouts)
    monkeypatch.setattr(phi.pio, "load_stage_isfs_spectra", fake_load_isfs)
    monkeypatch.setattr(
        phi.psp, "select_event_bouts",
        lambda bouts, isfs: {"freqs": isfs["freqs"], "psds": isfs["psds"]},
    )
    monkeypatch.setattr(
        phi.psp, "compute_subject_spectrum_features",
        lambda freqs, psds, return_curves: (
            {"peak_freq_hz": 0.02},
            {"freqs": freqs, "rel": psds.mean(axis=0), "corrected": psds.mean(axis=0)},
        ),
    )
    monkeypatch.setattr(phi.psp, "compute_bout_peak_freqs", lambda freqs, psds: np.array([0.02, 0.02]))

    result = phi.subject_isfs_spectrum_features(
        Path("/fake"), "SUBJ001", "C3", ("N2", "N3"), event="spindle",
        min_bout_sec=200.0, hypno_epoch_sec=3.0,
    )
    assert result is not None
    assert result["feats"]["peak_freq_hz"] == 0.02
