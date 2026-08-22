from __future__ import annotations

import numpy as np

from infraslow.pipeline import transitions as ptr
from infraslow.processing.utils import find_stage_bouts


def test_is_transition_bout_true_on_stage_change():
    stage_epochs = np.array(["N2", "N2", "N3", "N3"])
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2") is True


def test_is_transition_bout_false_when_next_epoch_is_same_stage():
    stage_epochs = np.array(["N2", "N2", "N2", "N3"])
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2") is False


def test_is_transition_bout_false_at_end_of_scored_night():
    stage_epochs = np.array(["N2", "N2"])  # stop=60 -> idx=2, out of range
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2") is False


def test_is_transition_bout_false_when_reverts_within_persist_window():
    # idx=2 -> "N3", but idx=3 reverts to "N2" -- with persist_epochs=2, not sustained
    stage_epochs = np.array(["N2", "N2", "N3", "N2", "N2"])
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2", persist_epochs=2) is False


def test_is_transition_bout_true_when_sustained_through_persist_window():
    stage_epochs = np.array(["N2", "N2", "N3", "N3", "N3"])
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2", persist_epochs=2) is True


def test_is_transition_bout_true_when_persist_window_exceeds_remaining_night():
    # only 1 epoch remains after stop -- persistence can't be disproven with the
    # data available, so it counts as a sustained transition
    stage_epochs = np.array(["N2", "N2", "N3"])
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2", persist_epochs=5) is True


def test_is_transition_bout_case_insensitive_stage_comparison():
    stage_epochs = np.array(["N2", "N2", "n2", "N3"])  # lowercase blip == same stage
    assert ptr.is_transition_bout(60.0, stage_epochs, 30.0, "N2", persist_epochs=1) is False


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


def test_select_transition_tails_drops_blip_and_revert_with_persist_epochs():
    # idx 10 flips to N3 for one epoch, then reverts to N2 for idx 11-19 --
    # without persistence this bout would count as a "transition"; with
    # persist_epochs=2 it must not, since the very next epoch after idx 10
    # (idx 11) reverts back to N2.
    stage_epochs = np.array(["N2"] * 10 + ["N3"] + ["N2"] * 9)
    bouts = np.array([[0.0, 300.0]])  # stop=300 -> idx=10
    out = ptr.select_transition_tails(
        bouts, stage_epochs, 30.0, "N2", window_sec=200.0, persist_epochs=2,
    )
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


def test_select_transition_tails_against_real_find_stage_bouts_output():
    """Regression guard: find_stage_bouts builds each bout as a *maximal* run
    of same-stage epochs, so the epoch right after any bout's stop already
    differs from the bout's own stage by construction (persist_epochs=1 is
    true for nearly every bout, not a meaningful filter). Build a real
    hypnogram, run the actual find_stage_bouts (unmodified) on it, and
    confirm select_transition_tails with persist_epochs > 1 distinguishes a
    blip-and-revert (dropped) from a genuinely sustained transition (kept)."""
    # N2 x10, N3 x1 (blip that reverts), N2 x10 (splits into two N2 bouts
    # under find_stage_bouts), N3 x10 (a real, sustained transition).
    stage_epochs = np.array(["N2"] * 10 + ["N3"] + ["N2"] * 10 + ["N3"] * 10)
    codes = np.where(stage_epochs == "N2", 2, 3)  # DEFAULT_STAGE_MAP: N2=2, N3=3
    bouts = np.asarray(find_stage_bouts(codes, (2,), epoch_sec=30.0, min_dur=200.0))
    assert bouts.shape[0] == 2  # two N2 bouts, split by the 1-epoch N3 blip

    out = ptr.select_transition_tails(
        bouts, stage_epochs, 30.0, "N2", window_sec=200.0, persist_epochs=3,
    )
    assert out.shape[0] == 1
    assert np.allclose(out[0], [bouts[1][1] - 200.0, bouts[1][1]])
