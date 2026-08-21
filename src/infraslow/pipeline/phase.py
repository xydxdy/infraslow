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
from ..processing.infraslow import (
    _nearest_sample_index,
    isfs_event_phase_distribution,
    isfs_phase_bins,
)

#: Angle (rad) of each of the 8 landmark-segment boundaries `isfs_phase_bins` already
#: computes (its `edges`, sample-index space) -- `start/trough/mid/peak/end` map to
#: `-pi, -pi/2, 0, pi/2, pi` with one more boundary bisecting each half, i.e. the same
#: fixed pi/4-wide bins `bin_center_angle` reports the centers of.
_PHASE_BIN_BOUNDARIES = -np.pi + (np.pi / 4) * np.arange(9)


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


def build_subject_phase_timeseries(
    t_env: np.ndarray, filtered: np.ndarray, bouts: np.ndarray, *,
    isfs_period=DEFAULT_ISFS_PERIOD,
) -> Dict[str, np.ndarray]:
    """Whole-recording ISFS phase time series, built from every bout in `bouts`
    (absolute recording-relative time, sorted).

    Thin orchestration over `build_bout_phase_data` (unmodified) and `bin_center_angle`
    (unmodified) -- no new phase math. Each bout's bout-relative `(t, phase_bins)` is
    shifted back to absolute time (`+ bout_start`) and concatenated across every bout,
    then sorted by time (bouts are disjoint by construction, so this is a stable
    concatenation, not a merge). No events are needed for this call (unlike
    `build_bout_phase_data`'s original spindle/SW-alignment use), so an empty
    `event_times` is passed through.

    Returns:
        `{"t": (n,) absolute seconds, "phase_bin": (n,) int 0-8, "phase_angle": (n,)
        float, NaN where phase_bin == 0 ("Not ISFS")}`, all sorted by `t`. Empty (but
        correctly-shaped) arrays if `bouts` is empty.
    """
    bouts = np.asarray(bouts, dtype=float).reshape(-1, 2)
    if bouts.shape[0] == 0:
        return dict(t=np.empty(0), phase_bin=np.empty(0, dtype=int), phase_angle=np.empty(0))

    bouts_data = build_bout_phase_data(t_env, filtered, bouts, np.empty(0), isfs_period=isfs_period)
    t_abs = np.concatenate([tt + a for (a, _b), (tt, _bins, _evt) in zip(bouts, bouts_data)])
    phase_bin = np.concatenate([bins for _tt, bins, _evt in bouts_data])

    order = np.argsort(t_abs, kind="stable")
    t_abs = t_abs[order]
    phase_bin = phase_bin[order]

    phase_angle = np.full(phase_bin.shape, np.nan)
    in_cycle = phase_bin > 0
    phase_angle[in_cycle] = bin_center_angle(phase_bin[in_cycle])

    return dict(t=t_abs, phase_bin=phase_bin, phase_angle=phase_angle)


def continuous_phase_samples(t: np.ndarray, x: np.ndarray, *, isfs_period=DEFAULT_ISFS_PERIOD) -> np.ndarray:
    """Continuous per-sample ISFS phase in `(-pi, pi]`, NaN outside any valid cycle.

    Calls `isfs_phase_bins` unmodified for cycle detection (no new zero-crossing/
    trough/peak math) and reuses its own per-cycle `edges` (the same 9 landmark
    sample indices bounding the 8 `pi/4`-wide bins `bin_center_angle` already
    centers) -- this only linearly interpolates each sample's position *within*
    its landmark segment by sample-index fraction, so a segment's own bin label
    (e.g. bin 3) is refined from one integer to a continuum spanning that bin's
    `pi/4` angular width instead of collapsing to its center. No new cycle
    detection or phase definition is introduced.

    Args:
        t, x: same contract as `isfs_phase_bins` (`x` already `isfs_lowpass`-filtered
            and mean-centered, e.g. z-scored per bout).
        isfs_period: forwarded to `isfs_phase_bins`.

    Returns:
        `(n,)` float array, same length as `x`; NaN where `isfs_phase_bins` would
        report bin `0` ("Not ISFS").
    """
    _bins, cycles = isfs_phase_bins(t, x, isfs_period=isfs_period)
    phase = np.full(x.shape, np.nan)
    for c in cycles:
        edges = c["edges"]
        for k in range(8):
            i0, i1 = int(edges[k]), int(edges[k + 1])
            if i1 <= i0:
                continue
            frac = (np.arange(i0, i1) - i0) / (i1 - i0)
            phase[i0:i1] = _PHASE_BIN_BOUNDARIES[k] + frac * (
                _PHASE_BIN_BOUNDARIES[k + 1] - _PHASE_BIN_BOUNDARIES[k]
            )
        phase[int(edges[-1])] = np.pi  # closed on the right, matches isfs_phase_bins's bins[c1] = 8
    return phase


def build_bout_continuous_phase_data(
    t_env: np.ndarray, filtered: np.ndarray, bouts: np.ndarray, *,
    isfs_period=DEFAULT_ISFS_PERIOD,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """One `(t, continuous_phase)` tuple per bout, bout-relative time base.

    Same per-bout slicing/standardization as `build_bout_phase_data` (independent
    z-score of each bout's own `filtered` slice), but reports `continuous_phase_samples`
    instead of `isfs_phase_bins`'s discrete 1-8 label.
    """
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for a, b in bouts:
        m0 = (t_env >= a) & (t_env < b)
        tt_b = t_env[m0] - a
        filt_b = filtered[m0]
        std = filt_b.std()
        std_b = (filt_b - filt_b.mean()) / std if std > 0 else filt_b - filt_b.mean()
        phase_b = continuous_phase_samples(tt_b, std_b, isfs_period=isfs_period)
        out.append((tt_b, phase_b))
    return out


def build_subject_continuous_phase_timeseries(
    t_env: np.ndarray, filtered: np.ndarray, bouts: np.ndarray, *,
    isfs_period=DEFAULT_ISFS_PERIOD,
) -> Dict[str, np.ndarray]:
    """Whole-recording continuous ISFS phase time series (absolute time, sorted).

    Mirrors `build_subject_phase_timeseries` exactly, but built from
    `build_bout_continuous_phase_data` (continuous phase) instead of the discrete
    1-8 bin label -- same bout-disjoint concatenate-then-sort construction, no new
    phase math beyond `continuous_phase_samples`.

    Returns:
        `{"t": (n,) absolute seconds, "phase_angle": (n,) float, NaN outside any
        valid cycle}`, sorted by `t`. Empty (but correctly-shaped) arrays if `bouts`
        is empty.
    """
    bouts = np.asarray(bouts, dtype=float).reshape(-1, 2)
    if bouts.shape[0] == 0:
        return dict(t=np.empty(0), phase_angle=np.empty(0))

    bouts_data = build_bout_continuous_phase_data(t_env, filtered, bouts, isfs_period=isfs_period)
    t_abs = np.concatenate([tt + a for (a, _b), (tt, _ph) in zip(bouts, bouts_data)])
    phase = np.concatenate([ph for _tt, ph in bouts_data])

    order = np.argsort(t_abs, kind="stable")
    return dict(t=t_abs[order], phase_angle=phase[order])


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
        if events.size == 0 or t.size == 0:
            continue
        idx = _nearest_sample_index(t, events)
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
    "build_subject_phase_timeseries",
    "continuous_phase_samples",
    "build_bout_continuous_phase_data",
    "build_subject_continuous_phase_timeseries",
    "compute_subject_phase_features",
    "pool_phase_distributions",
]
