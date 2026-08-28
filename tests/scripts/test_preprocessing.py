from __future__ import annotations

from scripts import preprocessing as pp


def test_valid_usleep_subject_ids_requires_both_edf_and_usleep_dir():
    ids = ["A", "B", "C"]
    edf_names = {"A.edf", "B.edf"}
    usleep_names = {"A", "C"}
    assert pp._valid_usleep_subject_ids(ids, edf_names, usleep_names) == ["A"]


def test_valid_usleep_subject_ids_sorted_and_deduplicated_by_input():
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
