from __future__ import annotations

import numpy as np
import pytest

from infraslow.io import usleep_hypnodensity as uh


def test_resolve_usleep_channel_path_finds_raw_channel_name(fake_usleep_tree):
    path = uh.resolve_usleep_channel_path(fake_usleep_tree / "SUBJ001", "C3")
    assert path == fake_usleep_tree / "SUBJ001" / "C3.npy"


def test_resolve_usleep_channel_path_finds_aliased_name(fake_usleep_tree):
    path = uh.resolve_usleep_channel_path(fake_usleep_tree / "SUBJ002", "C3")
    assert path == fake_usleep_tree / "SUBJ002" / "EEG C3-A2.npy"


def test_resolve_usleep_channel_path_missing_returns_none(fake_usleep_tree):
    assert uh.resolve_usleep_channel_path(fake_usleep_tree / "SUBJ001", "O1") is None


def test_load_usleep_hypnodensity_shapes_and_time_convention(fake_usleep_tree):
    t_hyp, probs = uh.load_usleep_hypnodensity("SUBJ001", "C3", base_dir=str(fake_usleep_tree))
    assert t_hyp.shape == (30,)
    assert probs.shape == (30, 5)
    assert np.isclose(t_hyp[0], 0.5)
    assert np.isclose(t_hyp[-1], 29.5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_load_usleep_hypnodensity_missing_subject_raises(fake_usleep_tree):
    with pytest.raises(FileNotFoundError):
        uh.load_usleep_hypnodensity("NOBODY", "C3", base_dir=str(fake_usleep_tree))


def test_load_usleep_hypnodensity_missing_channel_raises(fake_usleep_tree):
    with pytest.raises(FileNotFoundError):
        uh.load_usleep_hypnodensity("SUBJ001", "O1", base_dir=str(fake_usleep_tree))


def test_hypnodensity_to_epoch_hypnogram_averages_before_argmax():
    # 3 one-second rows: Wake wins each second individually (0.9 each), but N1 wins
    # the *average* (0.05+0.9+0.9)/3 > (0.9+0.05+0.05)/3 -- proves averaging happens
    # before argmax, not after.
    probs = np.array([
        [0.9, 0.05, 0.02, 0.02, 0.01],
        [0.05, 0.9, 0.02, 0.02, 0.01],
        [0.05, 0.9, 0.02, 0.02, 0.01],
    ])
    t_hyp = np.array([0.5, 1.5, 2.5])
    t_epoch, probs_epoch, stage_epoch = uh.hypnodensity_to_epoch_hypnogram(
        t_hyp, probs, epoch_sec=3.0, src_epoch_sec=1.0,
    )
    assert t_epoch.shape == (1,)
    assert np.isclose(t_epoch[0], 1.5)
    assert stage_epoch[0] == "N1"


def test_hypnodensity_to_epoch_hypnogram_drops_trailing_partial_window():
    probs = np.zeros((7, 5))
    probs[:, 0] = 1.0
    t_hyp = np.arange(7) + 0.5
    t_epoch, probs_epoch, stage_epoch = uh.hypnodensity_to_epoch_hypnogram(
        t_hyp, probs, epoch_sec=3.0, src_epoch_sec=1.0,
    )
    assert t_epoch.shape == (2,)  # 7 // 3 == 2 full windows, remainder dropped


def test_hypnodensity_to_epoch_hypnogram_rejects_non_integer_ratio():
    probs = np.zeros((5, 5))
    t_hyp = np.arange(5) + 0.5
    with pytest.raises(ValueError):
        uh.hypnodensity_to_epoch_hypnogram(t_hyp, probs, epoch_sec=2.5, src_epoch_sec=1.0)


def test_hypnodensity_probs_at_times_nearest_neighbour():
    t_hyp = np.array([0.5, 1.5, 2.5, 3.5])
    probs = np.array([[1, 0, 0, 0, 0.0]] * 4)
    probs[2] = [0, 0, 1, 0, 0]
    out = uh.hypnodensity_probs_at_times(np.array([2.4, 0.0]), t_hyp, probs)
    assert np.array_equal(out[0], [0, 0, 1, 0, 0])
    assert np.array_equal(out[1], [1, 0, 0, 0, 0])
