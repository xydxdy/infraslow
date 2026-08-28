from __future__ import annotations

from scripts import sleep_statistics as ss


def test_parse_shard_indices_range():
    assert ss._parse_shard_indices("0-9") == list(range(10))


def test_parse_shard_indices_comma_list():
    assert ss._parse_shard_indices("0,3,7") == [0, 3, 7]


def test_parse_shard_indices_single():
    assert ss._parse_shard_indices("5") == [5]


def test_select_shard_subjects_default_matches_plain_limit():
    subjects = [str(i) for i in range(10)]
    assert ss._select_shard_subjects(
        subjects, num_shards=1, shard_indices=[0], per_shard_limit=3,
    ) == subjects[:3]


def test_select_shard_subjects_matches_preprocessing_array_selection():
    # subjects[i::num_shards][:per_shard_limit] unioned across shard_indices --
    # must equal exactly what a preprocessing.py job array with the same
    # --num-shards/--limit would process across those --shard-index values.
    subjects = [str(i) for i in range(30)]
    got = ss._select_shard_subjects(
        subjects, num_shards=10, shard_indices=[0, 1, 2], per_shard_limit=2,
    )
    expected = []
    for shard in (0, 1, 2):
        expected.extend(subjects[shard::10][:2])
    assert got == expected
    assert got == ["0", "10", "1", "11", "2", "12"]


def test_select_shard_subjects_no_limit_takes_whole_shard():
    subjects = [str(i) for i in range(6)]
    got = ss._select_shard_subjects(
        subjects, num_shards=2, shard_indices=[0], per_shard_limit=None,
    )
    assert got == ["0", "2", "4"]
