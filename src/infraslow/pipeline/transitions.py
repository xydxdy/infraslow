"""Filter N2/N3 bouts down to just their pre-transition tail: the last
`window_sec` seconds of a bout that actually ends in a real, sustained
transition to a different scored sleep stage, as opposed to a single-epoch
scoring blip that reverts right back, or the end of the scored night.

`find_stage_bouts` (infraslow.processing.utils) builds each bout as a
*maximal* run of same-stage epochs -- so the epoch immediately after any
bout's `stop` is, by construction, always a different stage (the run would
not have stopped there otherwise), except when the bout runs to the very
last scored epoch of the night. Checking only the single next epoch is
therefore not a meaningful transition filter -- it is true for nearly every
bout. `persist_epochs` is what actually distinguishes a lasting transition
from a blip: the new stage must hold for at least `persist_epochs`
consecutive epochs (or through to the end of the scored night, if fewer
remain) without reverting back to `current_state`. A one-epoch scoring flip
to a different stage that then reverts back to `current_state` shows up as
two adjacent, same-stage bouts under `find_stage_bouts`'s maximal-run
construction -- `persist_epochs > 1` is what filters that case out. See
docs/superpowers/specs/2026-08-22-hypnodensity-isfs-phase-transition-design.md
for the full derivation.

Used by `plot_hypnodensity_isfs_phase_transition.py` to restrict the
event/phase analysis to real pre-transition activity instead of every bout
in full (see `plot_hypnodensity_isfs_phase.py` for that unrestricted
version).
"""
from __future__ import annotations

import numpy as np


def is_transition_bout(
    stop: float, stage_epochs: np.ndarray, epoch_sec: float, current_state: str,
    persist_epochs: int = 1,
) -> bool:
    """True iff the epoch immediately after `stop` exists and the scored
    stage stays different from `current_state` for at least
    `persist_epochs` consecutive epochs (clamped to however many epochs
    remain in the scored night) -- i.e. `stop` is a real, sustained
    transition out of `current_state`, not a single-epoch scoring blip that
    reverts right back. No next epoch at all (bout ran to the end of the
    scored night) -> False. Stage comparison is case-insensitive.
    """
    idx = int(round(stop / epoch_sec))
    if idx < 0 or idx >= len(stage_epochs):
        return False
    end = min(idx + max(1, persist_epochs), len(stage_epochs))
    window = np.asarray([str(s).strip().upper() for s in stage_epochs[idx:end]])
    return bool(np.all(window != current_state.strip().upper()))


def select_transition_tails(
    bouts: np.ndarray, stage_epochs: np.ndarray, epoch_sec: float,
    current_state: str, window_sec: float, *, persist_epochs: int = 1,
) -> np.ndarray:
    """`bouts` (n,2) that both last >= `window_sec` and end in a real,
    sustained transition (`is_transition_bout`, `persist_epochs`), each
    replaced by its own last `window_sec` seconds: `(stop - window_sec,
    stop)`. Every other bout (too short, or not a real transition) is
    dropped, not truncated. Bout order is preserved. Returns a `(m, 2)`
    array, `m <= n` (empty `(0, 2)` if none qualify)."""
    bouts = np.asarray(bouts, dtype=float).reshape(-1, 2)
    kept = []
    for a, b in bouts:
        if (b - a) < window_sec:
            continue
        if not is_transition_bout(b, stage_epochs, epoch_sec, current_state, persist_epochs):
            continue
        kept.append((b - window_sec, b))
    return np.asarray(kept, dtype=float).reshape(-1, 2)


__all__ = ["is_transition_bout", "select_transition_tails"]
