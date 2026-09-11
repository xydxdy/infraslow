from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import preprocessing as pp


def test_valid_usleep_subject_ids_requires_both_edf_and_usleep_dir():
    ids = ["A", "B", "C"]
    edf_names = {"A.edf", "B.edf"}
    usleep_names = {"A", "C"}
    assert pp._valid_usleep_subject_ids(ids, edf_names, usleep_names) == ["A"]


def test_valid_usleep_subject_ids_sorted_by_id():
    ids = ["B", "A"]
    edf_names = {"A.edf", "B.edf"}
    usleep_names = {"A", "B"}
    assert pp._valid_usleep_subject_ids(ids, edf_names, usleep_names) == ["A", "B"]


def test_valid_usleep_subject_ids_edf_only_is_excluded():
    ids = ["A"]
    edf_names = {"A.edf"}
    usleep_names: set = set()
    assert pp._valid_usleep_subject_ids(ids, edf_names, usleep_names) == []


def test_valid_usleep_subject_ids_usleep_only_is_excluded():
    ids = ["A"]
    edf_names: set = set()
    usleep_names = {"A"}
    assert pp._valid_usleep_subject_ids(ids, edf_names, usleep_names) == []


def test_save_usleep_hypnogram_writes_argmax_and_average(tmp_path: Path):
    ch_dir = tmp_path / "SUBJ001" / "C3"
    stage_epoch = np.array(["N2", "N2", "N3"], dtype=object)
    probs_epoch = np.array([
        [0.05, 0.05, 0.80, 0.05, 0.05],
        [0.05, 0.05, 0.80, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.80, 0.05],
    ])
    out_dir = pp.save_usleep_hypnogram(ch_dir, stage_epoch, probs_epoch, hypno_epoch_sec=3.0)

    assert out_dir == ch_dir / "usleep" / "3s"
    argmax = np.load(out_dir / "argmax.npy")
    average = np.load(out_dir / "average.npy")
    assert argmax.dtype.kind == "U"
    assert list(argmax) == ["N2", "N2", "N3"]
    assert average.shape == (3, 5)
    assert np.allclose(average, probs_epoch)


def test_check_usleep_alignment_within_tolerance_returns_none():
    assert pp._check_usleep_alignment(1000.0, 1050.0, tolerance_sec=120.0) is None


def test_check_usleep_alignment_beyond_tolerance_returns_message_with_diff():
    msg = pp._check_usleep_alignment(1000.0, 1200.0, tolerance_sec=120.0)
    assert msg is not None
    assert "200.0s" in msg


def test_check_usleep_file_usable_missing_file_raises(tmp_path: Path):
    usleep_dir = tmp_path / "usleep"
    (usleep_dir / "SUBJ001").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No U-Sleep hypnodensity file"):
        pp._check_usleep_file_usable("SUBJ001", "C3", str(usleep_dir))


def test_check_usleep_file_usable_empty_file_raises(tmp_path: Path):
    usleep_dir = tmp_path / "usleep"
    subject_dir = usleep_dir / "SUBJ001"
    subject_dir.mkdir(parents=True)
    (subject_dir / "C3.npy").touch()  # 0 bytes -- e.g. a killed write
    with pytest.raises(FileNotFoundError, match="empty"):
        pp._check_usleep_file_usable("SUBJ001", "C3", str(usleep_dir))


def test_check_usleep_file_usable_nonempty_file_returns_its_path(tmp_path: Path):
    usleep_dir = tmp_path / "usleep"
    subject_dir = usleep_dir / "SUBJ001"
    subject_dir.mkdir(parents=True)
    np.save(subject_dir / "C3.npy", np.zeros((5, 5)))
    path = pp._check_usleep_file_usable("SUBJ001", "C3", str(usleep_dir))
    assert path == subject_dir / "C3.npy"


# --------------------------------------------------------------------------- #
# <output-dir>/<N>s/ -- root for data/, events_<channel>*.csv and
# sleep_stats_<channel>*.csv, N = --hypno-epoch-sec
# --------------------------------------------------------------------------- #
def test_epoch_root_names_directory_by_hypno_epoch_sec():
    out_dir = Path("/tmp/out")
    assert pp._epoch_root(out_dir, 3.0) == out_dir / "3s"


def test_epoch_root_non_integer_epoch_sec():
    out_dir = Path("/tmp/out")
    assert pp._epoch_root(out_dir, 2.5) == out_dir / "2.5s"


# --------------------------------------------------------------------------- #
# events_<channel>.csv (preprocess subcommand)
# --------------------------------------------------------------------------- #
def test_event_fieldnames_two_stages():
    assert pp._event_fieldnames(["N2", "N3"]) == [
        "id", "N2_Spindle", "N2_SW", "N3_Spindle", "N3_SW",
    ]


def test_event_fieldnames_never_includes_a_channel_name():
    # Channel now lives in the events file's name, not its columns.
    assert pp._event_fieldnames(["N2"]) == ["id", "N2_Spindle", "N2_SW"]


def test_event_row_reflects_detected_event_counts():
    channel_events = {"N2": {"spindle": 3, "sw": 0}, "N3": {"spindle": 0, "sw": 1}}
    row = pp._event_row("SUBJ001", channel_events, stages=["N2", "N3"])
    assert row == {
        "id": "SUBJ001",
        "N2_Spindle": 3, "N2_SW": 0,
        "N3_Spindle": 0, "N3_SW": 1,
    }


def test_event_row_missing_channel_is_none_not_zero():
    # The channel raised outright for this subject in preprocess_subject --
    # absent from `events`, so its columns must be None (blank in the CSV),
    # never a guessed 0.
    row = pp._event_row("SUBJ001", None, stages=["N2"])
    assert row == {"id": "SUBJ001", "N2_Spindle": None, "N2_SW": None}


def test_event_row_missing_stage_is_none_not_zero():
    channel_events = {"N2": {"spindle": 2, "sw": 1}}  # N3 never populated
    row = pp._event_row("SUBJ001", channel_events, stages=["N2", "N3"])
    assert row["N3_Spindle"] is None
    assert row["N3_SW"] is None


def test_event_row_zero_events_is_not_treated_as_missing():
    # A channel that ran fine but detected zero spindles/slow waves must
    # still report 0, not None -- only an absent channel/stage is None.
    channel_events = {"N2": {"spindle": 0, "sw": 0}}
    row = pp._event_row("SUBJ001", channel_events, stages=["N2"])
    assert row == {"id": "SUBJ001", "N2_Spindle": 0, "N2_SW": 0}


def test_events_path_single_subject_has_no_shard_suffix():
    out_dir = Path("/tmp/out")
    assert pp._events_path(out_dir, "C3", shard_index=None) == out_dir / "events_C3.csv"


def test_events_path_sharded_run_includes_shard_index():
    out_dir = Path("/tmp/out")
    assert pp._events_path(out_dir, "C3", shard_index=7) == out_dir / "events_C3_shard7.csv"


# --------------------------------------------------------------------------- #
# sleep_stats_<channel>.csv (written by preprocessing itself, from the
# hypnogram preprocess_channel already loaded)
# --------------------------------------------------------------------------- #
def test_sleep_stats_fieldnames_is_id_then_every_yasa_key():
    assert pp._sleep_stats_fieldnames() == ["id", *pp.SLEEP_STATS_KEYS]


def test_sleep_stats_row_reflects_computed_stats():
    stats = {key: float(i) for i, key in enumerate(pp.SLEEP_STATS_KEYS)}
    row = pp._sleep_stats_row("SUBJ001", stats)
    assert row["id"] == "SUBJ001"
    for i, key in enumerate(pp.SLEEP_STATS_KEYS):
        assert row[key] == float(i)


def test_sleep_stats_row_missing_channel_is_none_not_zero():
    # The channel raised outright for this subject in preprocess_subject --
    # absent from `sleep_stats`, so its columns must be None (blank in the
    # CSV), never a guessed 0.
    row = pp._sleep_stats_row("SUBJ001", None)
    assert row["id"] == "SUBJ001"
    for key in pp.SLEEP_STATS_KEYS:
        assert row[key] is None


def test_sleep_stats_path_single_subject_has_no_shard_suffix():
    out_dir = Path("/tmp/out")
    assert pp._sleep_stats_path(out_dir, "C3", shard_index=None) == out_dir / "sleep_stats_C3.csv"


def test_sleep_stats_path_sharded_run_includes_shard_index():
    out_dir = Path("/tmp/out")
    assert pp._sleep_stats_path(out_dir, "C3", shard_index=7) == out_dir / "sleep_stats_C3_shard7.csv"
