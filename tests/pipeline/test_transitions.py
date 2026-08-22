from __future__ import annotations

import numpy as np

from infraslow.pipeline import transitions as ptr


def test_is_transition_bout_true_on_stage_change():
    stage_epochs = np.array(["N2", "N2", "N3", "N3"])
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2") is True


def test_is_transition_bout_false_when_next_epoch_is_same_stage():
    stage_epochs = np.array(["N2", "N2", "N2", "N3"])
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2") is False


def test_is_transition_bout_false_at_end_of_scored_night():
    stage_epochs = np.array(["N2", "N2"])  # stop=60 -> idx=2, out of range
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2") is False


def test_select_transition_tails_drops_too_short_bouts():
    stage_epochs = np.array(["N2"] * 5 + ["N3"])
    bouts = np.array([[0.0, 100.0]])  # 100s < window_sec=200 -- dropped
    out = ptr.select_transition_tails(bouts, stage_epochs, 30.0, "N2", window_sec=200.0)
    assert out.shape == (0, 2)


def test_select_transition_tails_drops_non_transition_bouts():
    stage_epochs = np.array(["N2"] * 20)  # never changes stage
    bouts = np.array([[0.0, 300.0]])
    out = ptr.select_transition_tails(bouts, stage_epochs, 30.0, "N2", window_sec=200.0)
    assert out.shape == (0, 2)


def test_select_transition_tails_truncates_real_transitions_and_preserves_order():
    # idx 0-9 N2, idx 10-19 N3, idx 20-29 N2, idx 30-39 N3 (epoch_sec=30 -> 30s/epoch)
    stage_epochs = np.array(["N2"] * 10 + ["N3"] * 10 + ["N2"] * 10 + ["N3"] * 10)
    bouts = np.array([
        [0.0, 300.0],    # stop=300 -> idx=10 -> "N3" -> real transition
        [600.0, 900.0],  # stop=900 -> idx=30 -> "N3" -> real transition
    ])
    out = ptr.select_transition_tails(bouts, stage_epochs, 30.0, "N2", window_sec=200.0)
    assert out.shape == (2, 2)
    assert np.allclose(out[0], [100.0, 300.0])
    assert np.allclose(out[1], [700.0, 900.0])


def test_select_transition_tails_empty_bouts_returns_empty():
    stage_epochs = np.array(["N2"] * 5)
    out = ptr.select_transition_tails(
        np.empty((0, 2)), stage_epochs, 30.0, "N2", window_sec=200.0,
    )
    assert out.shape == (0, 2)
