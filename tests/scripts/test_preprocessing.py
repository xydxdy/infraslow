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
# sleep-stats subcommand
# --------------------------------------------------------------------------- #
def test_parse_shard_indices_range():
    assert pp._parse_shard_indices("0-9") == list(range(10))


def test_parse_shard_indices_comma_list():
    assert pp._parse_shard_indices("0,3,7") == [0, 3, 7]


def test_parse_shard_indices_single():
    assert pp._parse_shard_indices("5") == [5]


def test_select_shard_subjects_default_matches_plain_limit():
    subjects = [str(i) for i in range(10)]
    assert pp._select_shard_subjects(
        subjects, num_shards=1, shard_indices=[0], per_shard_limit=3,
    ) == subjects[:3]


def test_select_shard_subjects_matches_preprocessing_array_selection():
    # subjects[i::num_shards][:per_shard_limit] unioned across shard_indices --
    # must equal exactly what a `preprocess` job array with the same
    # --num-shards/--limit would process across those --shard-index values.
    subjects = [str(i) for i in range(30)]
    got = pp._select_shard_subjects(
        subjects, num_shards=10, shard_indices=[0, 1, 2], per_shard_limit=2,
    )
    expected = []
    for shard in (0, 1, 2):
        expected.extend(subjects[shard::10][:2])
    assert got == expected
    assert got == ["0", "10", "1", "11", "2", "12"]


def test_select_shard_subjects_no_limit_takes_whole_shard():
    subjects = [str(i) for i in range(6)]
    got = pp._select_shard_subjects(
        subjects, num_shards=2, shard_indices=[0], per_shard_limit=None,
    )
    assert got == ["0", "2", "4"]
