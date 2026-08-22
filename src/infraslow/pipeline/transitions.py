"""Filter N2/N3 bouts down to just their pre-transition tail: the last
`window_sec` seconds of a bout that actually ends in a change to a different
scored sleep stage, as opposed to more of the same stage (a short scoring
interruption `find_stage_bouts` already absorbed) or the end of the scored
night. Used by `plot_hypnodensity_isfs_phase_transition.py` to restrict the
event/phase analysis to real pre-transition activity instead of every bout in
full (see `plot_hypnodensity_isfs_phase.py` for that unrestricted version).
"""
from __future__ import annotations

import numpy as np


def is_transition_bout(
    stop: float, stage_epochs: np.ndarray, epoch_sec: float, current_state: str,
) -> bool:
    """True iff the epoch immediately after `stop` exists and its scored stage
    (`stage_epochs`, in the same positional order/second-offset convention as
    `infraslow.io.hypnodensity.load_subject_hypnogram`) differs from
    `current_state` -- i.e. `stop` is a real transition out of `current_state`,
    not simply the end of the scored night (no next epoch -> False)."""
    idx = int(round(stop / epoch_sec))
    if idx < 0 or idx >= len(stage_epochs):
        return False
    return str(stage_epochs[idx]) != current_state


def select_transition_tails(
    bouts: np.ndarray, stage_epochs: np.ndarray, epoch_sec: float,
    current_state: str, window_sec: float,
) -> np.ndarray:
    """`bouts` (n,2) that both last >= `window_sec` and end in a real
    transition (`is_transition_bout`), each replaced by its own last
    `window_sec` seconds: `(stop - window_sec, stop)`. Every other bout (too
    short, or not a real transition) is dropped, not truncated. Bout order is
    preserved. Returns a `(m, 2)` array, `m <= n` (empty `(0, 2)` if none
    qualify)."""
    bouts = np.asarray(bouts, dtype=float).reshape(-1, 2)
    kept = []
    for a, b in bouts:
        if (b - a) < window_sec:
            continue
        if not is_transition_bout(b, stage_epochs, epoch_sec, current_state):
            continue
        kept.append((b - window_sec, b))
    return np.asarray(kept, dtype=float).reshape(-1, 2)


__all__ = ["is_transition_bout", "select_transition_tails"]
