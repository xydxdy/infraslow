from __future__ import annotations

import pytest

from infraslow.io import hypnodensity as hd


def test_load_subject_hypnogram_returns_stages_in_row_order(fake_hypno_dir):
    stages, epoch_sec = hd.load_subject_hypnogram("SUBJ001", hypno_dir=str(fake_hypno_dir))
    assert epoch_sec == 30.0
    assert list(stages) == ["N2", "N2", "N2", "N3", "N3", "Wake"]


def test_load_subject_hypnogram_missing_subject_raises(fake_hypno_dir):
    with pytest.raises(FileNotFoundError):
        hd.load_subject_hypnogram("NOBODY", hypno_dir=str(fake_hypno_dir))


def test_load_subject_hypnogram_custom_epoch_sec(fake_hypno_dir):
    _stages, epoch_sec = hd.load_subject_hypnogram(
        "SUBJ001", hypno_dir=str(fake_hypno_dir), epoch_sec=15.0,
    )
    assert epoch_sec == 15.0
