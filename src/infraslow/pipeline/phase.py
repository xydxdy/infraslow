"""Subject/state/channel-level ISFS phase-bin features.

Reproduces `demo_infraslow_phase.ipynb`'s per-bout construction exactly
(standardize each bout's own slice of the whole-night `isfs_lowpass`
course, run `isfs_phase_bins`, slice spindle peak times to that bout), then
aggregates with the existing `isfs_event_phase_distribution` -- both reused
unmodified from `infraslow.processing.infraslow`.

`preferred_phase`, `mean_phase`, and `resultant_length` do not exist
anywhere in this codebase or in the reference notebook (confirmed by a
repo-wide search for `circmean`/`resultant`/`preferred_phase` before writing
this module) -- the notebook's own "phase" is a discrete 1-8 bin label, not
a continuous angle. This module adds the smallest possible extension needed
to report those three spec-required values: each of the 8 bins is assigned
an evenly-spaced center angle in `(-pi, pi]` (`bin_center_angle`), and the
usual circular-statistics formulas (mean resultant vector, its angle and
length) are applied over each in-cycle event's bin-center angle. This is
disclosed here, not silently invented -- there is no other definition to
reproduce, and no new signal processing (no Hilbert transform, no new
filtering) is introduced; it is a categorical-to-angular re-labelling of the
notebook's own bin assignments.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..constants import DEFAULT_ISFS_MIN_EVENTS, DEFAULT_ISFS_PERIOD
from ..processing.infraslow import isfs_event_phase_distribution, isfs_phase_bins


def bin_center_angle(bin_idx) -> np.ndarray:
    """Evenly-spaced center angle in `(-pi, pi]` for ISFS phase bin(s) `1..8`."""
    b = np.asarray(bin_idx, dtype=float)
    return -np.pi + (2 * np.pi / 8) * (b - 0.5)


def build_bout_phase_data(
    t_env: np.ndarray, filtered: np.ndarray, bouts: np.ndarray, event_times: np.ndarray, *,
    isfs_period=DEFAULT_ISFS_PERIOD,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One `(t, phase_bins, event_times)` tuple per bout, bout-relative time base.

    `filtered` must already be the whole-night `isfs_lowpass` output (from
    `pipeline.io.load_temporal_isfs`) -- filtering per-bout instead of once
    over the whole night would reintroduce the edge-taper artifacts
    `isfs_lowpass`'s docstring explicitly warns against. Each bout's slice is
    standardized (z-scored) independently, matching the notebook
    (`std_b = (filt_b - filt_b.mean()) / filt_b.std()`).
    """
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for a, b in bouts:
        m0 = (t_env >= a) & (t_env < b)
        tt_b = t_env[m0] - a
        filt_b = filtered[m0]
        std = filt_b.std()
        std_b = (filt_b - filt_b.mean()) / std if std > 0 else filt_b - filt_b.mean()
        phase_bins_b, _cycles = isfs_phase_bins(tt_b, std_b, isfs_period=isfs_period)
        evt_b = event_times[(event_times >= a) & (event_times < b)] - a
        out.append((tt_b, phase_bins_b, evt_b))
    return out


def compute_subject_phase_features(
    bouts_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], *,
    min_events: int = DEFAULT_ISFS_MIN_EVENTS,
) -> Optional[Dict[str, object]]:
    """One subject/state/channel's phase features, or `None` if fewer than
    `min_events` total events were passed in (matches
    `isfs_event_phase_distribution`'s own exclusion rule -- a subject with
    too few spindles for a meaningful phase distribution is excluded rather
    than reported on a near-empty denominator)."""
    dist = isfs_event_phase_distribution(bouts_data, min_events=min_events)
    if dist is None:
        return None

    events_per_bout = [events for _t, _bins, events in bouts_data]
    bins_per_bout = [bins for _t, bins, _events in bouts_data]
    ts_per_bout = [t for t, _bins, _events in bouts_data]
    angles: List[float] = []
    for t, bins, events in zip(ts_per_bout, bins_per_bout, events_per_bout):
        if events.size == 0:
            continue
        idx = np.clip(np.searchsorted(t, events), 0, t.size - 1)
        left = np.clip(idx - 1, 0, t.size - 1)
        idx = np.where(np.abs(t[left] - events) <= np.abs(t[idx] - events), left, idx)
        b = bins[idx]
        in_cycle = b > 0
        angles.extend(bin_center_angle(b[in_cycle]).tolist())

    if angles:
        angles_arr = np.asarray(angles)
        c, s = np.cos(angles_arr).mean(), np.sin(angles_arr).mean()
        mean_phase = float(np.arctan2(s, c))
        resultant_length = float(np.hypot(c, s))
    else:
        mean_phase = float("nan")
        resultant_length = float("nan")

    return dict(
        event_count=int(dist["n_total"]),
        n_in_isfs=int(dist["n_in_isfs"]),
        phase_bin_counts=dist["counts"].tolist(),
        phase_bin_rates=dist["pct"].tolist(),
        # No separate "preferred phase" concept exists in the source notebook;
        # both are the circular mean of each in-cycle event's bin-center angle.
        preferred_phase=mean_phase,
        mean_phase=mean_phase,
        resultant_length=resultant_length,
    )


def pool_phase_distributions(dists: List[Dict[str, object]]) -> Dict[str, object]:
    """Cohort-level pooling: sum `phase_bin_counts`/`event_count`/`n_in_isfs`
    across subjects and recompute `phase_bin_rates` from the pooled counts
    (percentage of the pooled total events, same denominator convention as
    `isfs_event_phase_distribution`) -- used for the cohort Figure 3."""
    counts = np.zeros(8, dtype=int)
    event_count = 0
    n_in_isfs = 0
    for d in dists:
        counts += np.asarray(d["phase_bin_counts"], dtype=int)
        event_count += int(d["event_count"])
        n_in_isfs += int(d["n_in_isfs"])
    rates = (100.0 * counts / event_count) if event_count > 0 else np.zeros(8)
    return dict(
        event_count=event_count, n_in_isfs=n_in_isfs,
        phase_bin_counts=counts.tolist(), phase_bin_rates=rates.tolist(),
    )


__all__ = [
    "bin_center_angle",
    "build_bout_phase_data",
    "compute_subject_phase_features",
    "pool_phase_distributions",
]
