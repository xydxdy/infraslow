#!/usr/bin/env python
"""Group-level (subject-balanced) U-Sleep stage probability across continuous ISFS
phase, per --states group (N2, N3, or a merged "N2+N3" -- see _parse_state_groups),
plus each group's event phase-bin distribution.

Converted from ``demo_hypnodensity_isfs_phase.ipynb`` -- same computation (per-subject
loader, Section 4.1's event phase-bin rates, Section 4.2's continuous phase-locked
hypnodensity curve, Section 4.3's subject-balanced group mean +/- SEM, one figure
per state), but farmed out across ``--workers`` processes via ``ProcessPoolExecutor``
instead of the notebook's single-threaded scan (see ``_process_subject``: one worker
call does one subject's load + both states' Section 4.1/4.2 computation, since 4.2 is
only ever computed for a subject/state Section 4.1 already deemed usable -- the same
"never do 4.2's extra work for a subject 4.1 already excluded" rule the notebook
documents).

Candidate subjects come from ``pio.discover_subjects()`` (sorted) run against
``<preprocess-dir>/<hypno-epoch-sec>s/data`` (preprocessing.py's own ``_epoch_root``
layout -- see :func:`_resolve_data_dir`; pass ``--data-dir`` to point directly at a
``data/`` folder instead), matching the notebook's ``DATA_DIR`` scan order. Unlike the
notebook -- which stops scanning as
soon as ``N_SUBJECTS`` are included -- this script submits candidates to the worker
pool in fixed-size chunks (``--workers * 4`` subjects at a time) and stops once
``--n-subjects`` are included (default 100, matching the notebook's ``N_SUBJECTS``)
or the candidate list (optionally capped by ``--limit``) is exhausted. This keeps the
notebook's "don't scan more than needed" behavior while still using every worker
within each chunk.

A subject/group is included in the group figure only if it has both a usable event
(spindle/slow-wave) phase-bin distribution (Section 4.1: >= 1 in-cycle event) and
complete ``--n-phase-points``-point continuous phase coverage (Section 4.2) -- see
each group's printed usable/excluded counts. A multi-stage group (e.g. "N2+N3")
pools its component stages' bouts/events first, then applies these same checks to
the pooled result, so it is judged as one merged group throughout, not two.

Every subject's result (included or excluded) is written to ``cache/<subject_id>.npz``
as soon as it's computed (atomic temp-file + rename, so a killed job never leaves a
corrupt entry). On the next invocation with the same ``--output-dir``/``--event``,
already-cached subjects are loaded straight from disk instead of being reprocessed --
so a run interrupted by a timeout, OOM, or any other crash can simply be resubmitted
and it picks up only the subjects it hadn't gotten to yet. A missing, empty (e.g. a
0-byte npz left by a previously killed job), or otherwise corrupt per-subject artifact
excludes just that one subject (or, for a per-state metric, just that one state)
instead of crashing the run -- see ``_process_subject``. To force a full recompute,
delete ``<output-dir>/<event>/cache/`` first.

Run via Slurm, not the login node, from this file's own directory (``src/scripts/``)
with ``src/`` on ``PYTHONPATH`` so ``infraslow`` resolves (the package is not
pip-installed), e.g.::

    export PYTHONPATH=/home/users/chaisaen/infraslow/src
    srun -p normal --time=00:30:00 --mem=8G --cpus-per-task=4 \\
        python3 plot_hypnodensity_isfs_phase.py --n-subjects 100 --workers 4

See ``run_hypnodensity_isfs_phase.sbatch`` to submit this as a Slurm job.

Saves, under ``--output-dir/--event`` (e.g. ``.../hypnodensity_isfs_phase/spindle`` vs.
``.../hypnodensity_isfs_phase/sw``) -- namespaced by event so a spindle run and a sw run
against the same ``--output-dir`` can be submitted as concurrent jobs without
clobbering each other's output:
    hypnodensity_isfs_phase_{state}.png -- Section 4.3's figure, one file per --states group
                                            ({state} is that group's label, e.g. "N2" or a
                                            merged "N2+N3" -- always separate files, never a
                                            shared image)
    exclusions.csv          -- every excluded subject id + categorized reason
    group_stats/{state}.npz -- group_mean, group_sem, phase_points, phase_bin_rates
                                (subject x 8), bin_centers, n_complete
    stat_summary_{state}.csv -- per-stage test of whether stage probability tracks the
                                {event} rate across the ISFS phase cycle: per-subject
                                Pearson r between the {event}-rate curve (resampled onto
                                the same phase grid, see infraslow.pipeline.phase.
                                resample_bin_rates_to_points) and each stage's
                                phase-locked probability curve, then a one-sample t-test
                                on Fisher-z(r) across subjects (H0: mean r == 0), BH-FDR
                                across stages (infraslow.stat.state_comparison.
                                per_subject_curve_correlation / phase_curve_correlation_summary)
    usable_subjects_{state}.csv -- subject_id of every subject contributing to that
                                state's figure/stat_summary (i.e. common_ids)
    {event}_stage_phase_corr.png -- diverging bar chart of stat_summary_{state}'s
                                mean_r per stage, one panel per state
    isfs_spectrum_{state}.png -- group-level (subject-balanced) baseline-corrected
                                ISFS spectrum + bi-Gaussian fit, one file per state --
                                the group-level analog of demo_infraslow_yasa_average.
                                ipynb's Fig C-i right panel (see build_isfs_spectrum_figure)
    isfs_spectrum_features_{state}.csv -- every per-subject ISFS spectral fit value
                                (peak_freq_hz, bandwidth_hz, auc, bi_gaussian_*,
                                real_peak_freq_hz, etc. -- see subject_isfs_spectrum_features/
                                infraslow.pipeline.spectrum.compute_subject_spectrum_features),
                                one row per subject usable for that state's spectrum
    isfs_peak_distribution_{state}.png -- histogram of each subject's own real/
                                empirical ISFS peak frequency (subject-balanced: one
                                value per subject, from that subject's own bout-averaged
                                spectrum -- see build_peak_distribution_figure)
    isfs_bout_peak_freqs_{state}.csv -- every individual event-containing bout's own
                                empirical peak frequency, one row per subject (columns
                                subject_id, n_bouts, peak_1..peak_N, NaN-padded to the
                                max bout count across subjects) -- the full per-bout
                                detail behind isfs_peak_distribution_{state}'s subject-
                                balanced histogram
    group_stats/{state}_isfs_spectrum.npz -- freqs, rel_mean, corrected_mean/_sem,
                                real_peak_freq_hz, fit_* (bi-Gaussian popt/mu/lo/hi/
                                bandwidth/auc/threshold/detected), n_subjects, subject_ids
    summary.txt             -- Section 5's subject-inclusion/usable-count summary
    cache/{subject_id}.npz  -- one per scanned subject, enables resuming after a
                                timeout/OOM/crash without reprocessing finished subjects
    progress.log
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yasa

from infraslow.pipeline import io as pio
from infraslow.pipeline import phase as pph
from infraslow.pipeline import spectrum as psp
from infraslow.io import usleep_hypnodensity as uh
from infraslow.stat import state_comparison as stc
from infraslow.processing.infraslow import (
    DEFAULT_BASELINE_BAND,
    DEFAULT_INFRASLOW_BAND,
    bigaussian,
    fit_isfs,
)
from infraslow.constants import (
    DEFAULT_HYPNOGRAM_EPOCH_SEC,
    DEFAULT_MIN_BOUT_SEC,
    DEFAULT_USLEEP_EPOCH_SEC,
    DEFAULT_USLEEP_HYPNODENSITY_DIR,
    USLEEP_STAGE_ORDER,
)

logger = logging.getLogger(__name__)

plt.rcParams["figure.dpi"] = 110
PURPLE, LPURPLE = "#5b2a86", "#c9b3e6"
ORANGE, LORANGE = "#d2691e", "#f2c14e"
# Diverging pair for the stage-probability-vs-{event}-rate correlation chart (dataviz
# skill's validated default diverging pair: blue/red, worst adjacent CVD dE 21.6,
# normal-vision dE 32.3 -- a different color job than PURPLE/ORANGE's half-wave shading
# above, so kept as its own pair rather than reused).
CORR_POS, CORR_NEG, CORR_NEUTRAL = "#2a78d6", "#e34948", "#9a9a95"

# "spindle"/"sw" artifact-pathway switch (band + bouts key + bouts loader + summary
# loader + peak column) -- this script's own scan. events_col is preprocessing.py's
# events_<channel>.csv column suffix (<stage>_Spindle / <stage>_SW) for the matching
# event type -- see _filter_subjects_with_events.
_EVENT_SPECS = {
    "spindle": dict(band="sigma", bouts_key="spindle", event_label="spindles",
                     load_bouts=pio.load_stage_bouts, load_summary=pio.load_spindle_summary,
                     peak_col="Peak", events_col="Spindle"),
    "sw": dict(band="delta", bouts_key="sw", event_label="slow waves",
               load_bouts=pio.load_stage_sw_bouts, load_summary=pio.load_sw_summary,
               peak_col="NegPeak", events_col="SW"),
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None else val


def _resolve_data_dir(
    preprocess_dir: Path, data_dir: Optional[Path], hypno_epoch_sec: float,
) -> Path:
    """``data_dir`` if explicitly given (a direct override, e.g. a non-standard
    layout), else ``<preprocess_dir>/<hypno_epoch_sec>s/data`` -- matching
    ``preprocessing.py``'s own ``_epoch_root``-nested layout (data/,
    events_<channel>*.csv and sleep_stats_<channel>*.csv all live under
    ``<output-dir>/<N>s/``), so this script and preprocessing.py can never
    disagree about where a given epoch width's ``data/`` tree lives."""
    if data_dir is not None:
        return data_dir
    return preprocess_dir / pio.epoch_dirname(hypno_epoch_sec) / "data"


def _events_csv_path(preprocess_dir: Path, channel: str, hypno_epoch_sec: float) -> Path:
    """``<preprocess_dir>/<hypno_epoch_sec>s/events_<channel>.csv`` -- matches
    preprocessing.py's ``_events_path`` naming for its (already merged,
    unsharded) events CSV."""
    return preprocess_dir / pio.epoch_dirname(hypno_epoch_sec) / f"events_{channel}.csv"


def _filter_subjects_with_events(
    candidates: Sequence[str], events_df: pd.DataFrame, states: Sequence[str], event: str,
) -> List[str]:
    """``candidates``, in the same order, restricted to subjects with a confirmed
    nonzero preprocessing.py-detected ``event`` count in at least one of ``states``
    (``events_df``'s ``<stage>_{Spindle,SW}`` columns -- see preprocessing.py's
    ``_event_row``).

    A subject with zero detected events in every requested stage can never pass
    Section 4.1's ">= 1 in-cycle event" usable check downstream, so dropping it
    here changes no result -- only which subjects get submitted to the worker
    pool, saving the cost of loading/processing ones already known to be excluded.
    A subject missing from ``events_df`` entirely, or blank (``NaN``) in every
    relevant column (that channel raised outright during preprocessing), is
    dropped too -- never treated as a guessed ``0``, but neither is it a
    confirmed detected event.
    """
    col_suffix = _EVENT_SPECS[event]["events_col"]
    columns = [f"{state}_{col_suffix}" for state in states]
    by_id = events_df.set_index("id")
    kept = []
    for subject_id in candidates:
        if subject_id not in by_id.index:
            continue
        row = by_id.loc[subject_id]
        if any(pd.notna(row.get(col)) and row.get(col) > 0 for col in columns):
            kept.append(subject_id)
    return kept


def _parse_state_groups(raw_states: Sequence[str]) -> List[Tuple[str, Tuple[str, ...]]]:
    """Parse ``--states`` tokens into ``(label, components)`` groups: a bare
    stage (``"N2"``) is its own singleton group; a ``"+"``-joined token
    (``"N2+N3"``) pools those stages' bouts/events into one merged group --
    see ``subject_event_phase_bin_rates``/``subject_isfs_spectrum_features``/
    ``subject_phase_locked_probs_continuous``, the only functions that treat
    ``components`` as more than an opaque label. Order is preserved, both
    across groups and within a group's components, exactly as given.

    Raises:
        ValueError: a group repeats one of its own components (``"N2+N2"``),
            or two groups produce the same label (``"N2", "N2"`` or
            ``"N2+N3", "N3+N2"``) -- either would silently collide on the
            same output filename/cache key.
    """
    groups: List[Tuple[str, Tuple[str, ...]]] = []
    seen_labels: set = set()
    for token in raw_states:
        components = tuple(token.split("+"))
        if len(set(components)) != len(components):
            raise ValueError(f"--states group {token!r} has a duplicate component")
        if token in seen_labels:
            raise ValueError(f"--states has a duplicate group {token!r}")
        seen_labels.add(token)
        groups.append((token, components))
    return groups


def _all_leaf_stages(groups: Sequence[Tuple[str, Tuple[str, ...]]]) -> List[str]:
    """Every distinct leaf stage referenced by any group, sorted -- the set
    ``load_subject`` must pool bouts from and ``_filter_subjects_with_events``
    must check columns for, regardless of how those leaf stages are grouped
    into figures."""
    return sorted({stage for _, components in groups for stage in components})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preprocess-dir", type=Path,
                    default=Path(os.path.expandvars(
                        _env_str("PREPROCESS_DIR", "$SCRATCH/infraslow_outputs/preprocessed"))),
                    help="preprocessing.py's --output-dir root -- this script reads "
                         "<preprocess-dir>/<hypno-epoch-sec>s/data (env: PREPROCESS_DIR); "
                         "ignored if --data-dir is given")
    _data_dir_env = os.environ.get("DATA_DIR")
    p.add_argument("--data-dir", type=Path,
                    default=Path(os.path.expandvars(_data_dir_env)) if _data_dir_env else None,
                    help="Override: point directly at a preprocessing.py data/ folder "
                         "instead of deriving <preprocess-dir>/<hypno-epoch-sec>s/data "
                         "from --preprocess-dir/--hypno-epoch-sec (env: DATA_DIR)")
    p.add_argument("--usleep-dir", default=os.path.expandvars(
                        _env_str("USLEEP_DIR", DEFAULT_USLEEP_HYPNODENSITY_DIR)),
                    help="U-Sleep hypnodensity directory (env: USLEEP_DIR)")
    p.add_argument("--channel", default=_env_str("CHANNEL", "C3"), help="EEG channel (env: CHANNEL)")
    p.add_argument("--states", nargs="+", default=["N2", "N3"],
                    help="Groups to analyze, one figure/analysis per group (see "
                         "_parse_state_groups). A bare stage (e.g. N2) is its own group; "
                         "join stages with '+' (e.g. N2+N3) to pool their bouts/events "
                         "into one merged group instead. Mix both in one run, e.g. "
                         "--states N2 N3 N2+N3 for three groups.")
    p.add_argument("--event", choices=sorted(_EVENT_SPECS), default="spindle",
                    help="Which event type's phase-bin distribution/band to use "
                         "(sets BAND: spindle -> sigma, sw -> delta)")
    p.add_argument("--min-bout-sec", type=float,
                    default=float(_env_str("MIN_BOUT_SEC", str(DEFAULT_MIN_BOUT_SEC))),
                    help="Minimum consecutive-stage bout length (s) (env: MIN_BOUT_SEC)")
    p.add_argument("--hypno-epoch-sec", type=float,
                    default=float(_env_str("HYPNO_EPOCH_SEC", str(DEFAULT_HYPNOGRAM_EPOCH_SEC))),
                    help="Must match whatever --hypno-epoch-sec preprocessing.py was run "
                         "with -- selects both --preprocess-dir's <N>s/data root (see "
                         "--data-dir/_resolve_data_dir) and which <stage>/<N>s/ directory "
                         "to read each subject's bout/event artifacts from within it "
                         "(env: HYPNO_EPOCH_SEC). Not the same knob as --usleep-epoch-sec below.")
    p.add_argument("--usleep-epoch-sec", type=float,
                    default=float(_env_str("USLEEP_EPOCH_SEC", str(DEFAULT_USLEEP_EPOCH_SEC))),
                    help="Average the native 1-s U-Sleep hypnodensity into windows this "
                         "many seconds wide before matching it to phase samples; must be "
                         "a positive integer multiple of 1s (default: no averaging) "
                         "(env: USLEEP_EPOCH_SEC)")
    p.add_argument("--n-phase-points", type=int, default=50,
                    help="Resolution of the continuous phase-locked group curve")
    p.add_argument("--n-subjects", type=int, default=100,
                    help="Stop once this many subjects are included (matches the notebook's "
                         "N_SUBJECTS); pass 0/negative to disable and use every candidate")
    p.add_argument("--limit", type=int, default=None,
                    help="Cap the candidate pool to the first N discovered subjects "
                         "(quick tests); default scans every subject under the resolved "
                         "data dir (see --preprocess-dir/--data-dir)")
    p.add_argument("--filter-by-events", action="store_true",
                    help="Before submitting candidates to the worker pool, drop any with "
                         "zero detected --event events (in preprocessing.py's "
                         "events_<channel>.csv, across every requested --states -- see "
                         "_filter_subjects_with_events) -- a processing-time optimization "
                         "only: such a subject could never pass Section 4.1's usable-event "
                         "check anyway. Reads <preprocess-dir>/<hypno-epoch-sec>s/"
                         "events_<channel>.csv (see --events-csv to override); errors if "
                         "that file doesn't exist.")
    p.add_argument("--events-csv", type=Path, default=None,
                    help="Override: events_<channel>.csv path to use with "
                         "--filter-by-events, instead of deriving it from "
                         "--preprocess-dir/--channel/--hypno-epoch-sec")
    p.add_argument("--workers", type=int,
                    default=int(_env_str("WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
                    help="Parallel worker processes (env: WORKERS, falls back to "
                         "$SLURM_CPUS_PER_TASK, else 1)")
    p.add_argument("--output-dir", type=Path,
                    default=Path(os.path.expandvars(
                        _env_str("OUTPUT_DIR", "$SCRATCH/outputs/hypnodensity_isfs_phase"))),
                    help="Where to save the PNGs, exclusions.csv, group_stats/, and "
                         "progress.log (env: OUTPUT_DIR)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Per-subject computation (module-level so ProcessPoolExecutor can pickle it)
# --------------------------------------------------------------------------- #
def load_subject(data_dir: Path, usleep_dir: str, subject_id: str, channel: str,
                  states: Sequence[str], band: str, *,
                  min_bout_sec: float = DEFAULT_MIN_BOUT_SEC,
                  usleep_epoch_sec: float = DEFAULT_USLEEP_EPOCH_SEC,
                  hypno_epoch_sec: float = DEFAULT_HYPNOGRAM_EPOCH_SEC) -> dict:
    """Load and align one subject's U-Sleep hypnodensity with its infraslow phase time
    series, or raise with a short, categorized reason.

    `usleep_epoch_sec` averages the native 1-s hypnodensity into coarser
    `usleep_epoch_sec`-wide windows (via `uh.hypnodensity_to_epoch_hypnogram`) before
    it's matched to phase samples -- a no-op at the default 1.0 (each "window" is a
    single native row). `hypno_epoch_sec` is a different knob: it must match whatever
    `--hypno-epoch-sec` `preprocessing.py` was run with, since that's what selects
    which `<stage>/<N>s/` directory this subject's bout artifacts live under.

    Returns a dict with keys `t_hyp`, `probs` (hypnodensity, `usleep_epoch_sec`-wide
    rows), `phase` (this subject's whole-recording phase series, pooled across every
    state in `states` -- N2 and N3 bouts are time-disjoint by construction, so
    concatenating and re-sorting is safe; carries both the discrete 1-8
    `phase_bin`/`phase_angle` and the continuous `phase_continuous`).

    Raises:
        FileNotFoundError: missing U-Sleep hypnodensity, or missing temporal_ISFS/bouts
            artifacts.
        ValueError: no usable bouts, or `usleep_epoch_sec` is not a positive integer
            multiple of the native 1-s U-Sleep cadence.
    """
    subject_dir = Path(data_dir) / subject_id
    t_hyp, probs = uh.load_usleep_hypnodensity(subject_id, channel, base_dir=usleep_dir)
    t_hyp, probs, _ = uh.hypnodensity_to_epoch_hypnogram(
        t_hyp, probs, epoch_sec=usleep_epoch_sec, src_epoch_sec=DEFAULT_USLEEP_EPOCH_SEC,
    )
    t_env, filtered = pio.load_temporal_isfs(subject_dir, channel, band)

    phase_parts = []
    for state in states:
        bouts = pio.load_stage_bouts(subject_dir, channel, state, hypno_epoch_sec=hypno_epoch_sec)["all"]
        if bouts.shape[0]:
            durations = bouts[:, 1] - bouts[:, 0]
            bouts = bouts[durations >= min_bout_sec]
        if bouts.shape[0]:
            series = pph.build_subject_phase_timeseries(t_env, filtered, bouts)
            if series["t"].size:
                continuous = pph.build_subject_continuous_phase_timeseries(t_env, filtered, bouts)
                series["phase_continuous"] = continuous["phase_angle"]
                phase_parts.append(pd.DataFrame(series).assign(state=state))

    if not phase_parts:
        raise ValueError(f"no usable {'/'.join(states)} bouts (>= {min_bout_sec}s) for this subject")

    phase = pd.concat(phase_parts, ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)
    return dict(t_hyp=t_hyp, probs=probs, phase=phase)


def _pooled_event_bouts(
    subject_dir: Path, channel: str, components: Sequence[str], spec: dict,
    hypno_epoch_sec: float,
) -> np.ndarray:
    """``spec["bouts_key"]``'s bouts (n, 2) from every stage in ``components``,
    concatenated -- a single-element ``components`` is bit-identical to
    loading that one stage directly (N2 and N3 bouts are time-disjoint by
    construction, so concatenating never needs deduplication)."""
    parts = [
        spec["load_bouts"](subject_dir, channel, stage, hypno_epoch_sec=hypno_epoch_sec)[spec["bouts_key"]]
        for stage in components
    ]
    parts = [p for p in parts if p.shape[0]]
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 2))


def subject_event_phase_bin_rates(data_dir: Path, subject_id: str, channel: str,
                                   components: Sequence[str], *,
                                   event: str, min_bout_sec: float = DEFAULT_MIN_BOUT_SEC,
                                   hypno_epoch_sec: float = DEFAULT_HYPNOGRAM_EPOCH_SEC):
    """This subject/group's (``components`` pooled) `phase_bin_rates` (% of
    `event`s landing inside a valid ISFS cycle in each of the 8 discrete phase
    bins -- "Not ISFS" events are excluded from the denominator, see
    `denominator="in_cycle"` in `isfs_event_phase_distribution`), or `None` if
    there are no usable `event`-containing bouts across every stage in
    ``components``."""
    spec = _EVENT_SPECS[event]
    subject_dir = Path(data_dir) / subject_id
    bouts = _pooled_event_bouts(subject_dir, channel, components, spec, hypno_epoch_sec)
    if bouts.shape[0]:
        durations = bouts[:, 1] - bouts[:, 0]
        bouts = bouts[durations >= min_bout_sec]
    if bouts.shape[0] == 0:
        return None

    t_env, filtered = pio.load_temporal_isfs(subject_dir, channel, spec["band"])
    summaries = [
        spec["load_summary"](subject_dir, channel, stage, hypno_epoch_sec=hypno_epoch_sec)
        for stage in components
    ]
    event_summary = pd.concat(summaries, ignore_index=True)
    peak_col = spec["peak_col"]
    event_times = event_summary[peak_col].to_numpy() if peak_col in event_summary.columns else np.empty(0)

    bouts_data = pph.build_bout_phase_data(t_env, filtered, bouts, event_times)
    # Default denominator="in_cycle": phase_bin_rates is a % of events landing
    # inside a valid ISFS cycle (bins 1-8 only) -- "Not ISFS" events are
    # excluded from the denominator entirely.
    return pph.compute_subject_phase_features(bouts_data)


def subject_isfs_spectrum_features(data_dir: Path, subject_id: str, channel: str,
                                    components: Sequence[str], *,
                                    event: str, min_bout_sec: float = DEFAULT_MIN_BOUT_SEC,
                                    hypno_epoch_sec: float = DEFAULT_HYPNOGRAM_EPOCH_SEC) -> Optional[dict]:
    """This subject/group's (``components`` pooled) ISFS spectral fit -- the
    bi-Gaussian ``fit_isfs``/relative-power pathway ``infraslow.pipeline.
    spectrum``/``pipeline.py`` already use (matches ``demo_infraslow_yasa_
    average.ipynb``'s Fig C-i right panel in spirit, but reuses this project's
    standard bi-Gaussian fit instead of that notebook's own simplified
    symmetric-Gaussian one). Returns ``None`` if there are no usable `event`-
    containing bouts across every stage in ``components``, mirroring
    ``subject_event_phase_bin_rates``.

    Returns a dict with keys `freqs`, `rel`, `corrected` (this subject's own mean
    relative-power / baseline-corrected spectrum across its `event`-containing bouts),
    `feats` (`infraslow.pipeline.spectrum.compute_subject_spectrum_features`'s dict --
    `peak_freq_hz`, `bandwidth_hz`, `auc`, `bi_gaussian_*`, `real_peak_freq_hz`, etc.),
    and `bout_peak_freqs` (this subject's own per-bout empirical peak frequencies,
    `infraslow.pipeline.spectrum.compute_bout_peak_freqs`)."""
    spec = _EVENT_SPECS[event]
    subject_dir = Path(data_dir) / subject_id
    bouts = _pooled_event_bouts(subject_dir, channel, components, spec, hypno_epoch_sec)
    if bouts.shape[0]:
        durations = bouts[:, 1] - bouts[:, 0]
        bouts = bouts[durations >= min_bout_sec]
    if bouts.shape[0] == 0:
        return None

    isfs_parts = [
        pio.load_stage_isfs_spectra(subject_dir, channel, stage, spec["band"],
                                     hypno_epoch_sec=hypno_epoch_sec)
        for stage in components
    ]
    # A stage with zero bouts at all (not just zero event-containing ones) saves an
    # empty (0, 0)-shaped psds/freqs instead of the real (0, n_freqs) grid (see
    # preprocessing.py's compute_bout_spectra: "All empty if bouts is empty") -- drop
    # those parts before concatenating, using freqs from any part that actually has
    # bouts (freqs is otherwise the same shared grid for every stage, see
    # preprocessing.py's module docstring, so any one non-empty part's freqs is as
    # good as another's).
    isfs_parts = [p for p in isfs_parts if p["psds"].shape[0] > 0]
    if not isfs_parts:
        return None
    isfs = dict(
        freqs=isfs_parts[0]["freqs"],
        psds=np.concatenate([p["psds"] for p in isfs_parts], axis=0),
        bout_start=np.concatenate([p["bout_start"] for p in isfs_parts], axis=0),
    )
    selected = psp.select_event_bouts(bouts, isfs)
    if selected["psds"].shape[0] == 0:
        return None

    feats, curves = psp.compute_subject_spectrum_features(
        selected["freqs"], selected["psds"], return_curves=True,
    )
    bout_peak_freqs = psp.compute_bout_peak_freqs(selected["freqs"], selected["psds"])
    return dict(freqs=curves["freqs"], rel=curves["rel"], corrected=curves["corrected"],
                feats=feats, bout_peak_freqs=bout_peak_freqs)


def subject_phase_locked_probs_continuous(
    data: dict, *, components: Sequence[str], n_points: int,
) -> np.ndarray:
    """This subject's own mean hypnodensity probability vector, pooled across
    every stage in ``components``, across `n_points` equal-width bins spanning
    the continuous ISFS phase `(-pi, pi]` -- (n_points, n_stages), NaN row for
    any bin with zero in-cycle samples."""
    phase = data["phase"]
    phase = phase[phase["state"].isin(components)]
    in_cycle = phase["phase_continuous"].notna()
    t_in = phase.loc[in_cycle, "t"].to_numpy()
    phase_in = phase.loc[in_cycle, "phase_continuous"].to_numpy()
    probs_at = uh.hypnodensity_probs_at_times(t_in, data["t_hyp"], data["probs"])

    edges = np.linspace(-np.pi, np.pi, n_points + 1)
    bin_idx = np.digitize(phase_in, edges[1:-1])  # 0 .. n_points-1

    out = np.full((n_points, len(USLEEP_STAGE_ORDER)), np.nan)
    for b in range(n_points):
        mask = bin_idx == b
        if mask.any():
            out[b] = probs_at[mask].mean(axis=0)
    return out


def _process_subject(
    task: Tuple[str, Path, str, str, Tuple[Tuple[str, Tuple[str, ...]], ...], str, float, int,
                float, float],
) -> Tuple[str, bool, Optional[str], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, dict]]:
    """One subject's full Section 3/4.1/4.2 pipeline. Returns
    ``(subject_id, included, exclusion_reason, rates_by_group, curves_by_group,
    spectrum_by_group)`` -- ``rates_by_group``/``curves_by_group``/``spectrum_by_group``
    only ever contain a group label key when that group was usable for it
    (``curves_by_group`` is always a subset of ``rates_by_group``, same as the
    notebook's Section 4.2 restricting itself to Section 4.1's usable subject
    set; ``spectrum_by_group`` is computed alongside ``rates_by_group`` -- it
    depends only on event-containing bouts, not on Section 4.2's continuous-
    phase coverage, so it is not restricted to ``curves_by_group``). A
    multi-stage group (see ``_parse_state_groups``) pools its components'
    bouts/events before computing any of these, so it behaves as one merged
    stage throughout, not two separate ones."""
    (subject_id, data_dir, usleep_dir, channel, groups, event, min_bout_sec, n_phase_points,
     usleep_epoch_sec, hypno_epoch_sec) = task
    band = _EVENT_SPECS[event]["band"]
    all_leaf_stages = _all_leaf_stages(groups)

    try:
        data = load_subject(data_dir, usleep_dir, subject_id, channel, all_leaf_stages, band,
                             min_bout_sec=min_bout_sec, usleep_epoch_sec=usleep_epoch_sec,
                             hypno_epoch_sec=hypno_epoch_sec)
    except Exception as exc:  # noqa: BLE001 - a missing, empty, or corrupt artifact
        # (e.g. a 0-byte npz from a killed prior job raises EOFError, not caught by
        # the FileNotFoundError/ValueError/OSError this used to only catch) excludes
        # this subject, it must never crash the whole worker pool.
        return subject_id, False, f"{type(exc).__name__}: {exc}", {}, {}, {}

    rates: Dict[str, np.ndarray] = {}
    curves: Dict[str, np.ndarray] = {}
    spectrum: Dict[str, dict] = {}
    for label, components in groups:
        try:
            feats = subject_event_phase_bin_rates(data_dir, subject_id, channel, components,
                                                    event=event, min_bout_sec=min_bout_sec,
                                                    hypno_epoch_sec=hypno_epoch_sec)
        except Exception:  # noqa: BLE001 - same as above, scoped to this metric only
            feats = None
        # n_in_isfs (not sum(phase_bin_rates), which is all-NaN whenever no event
        # landed in a valid cycle under denominator="in_cycle"'s zero-division guard)
        # is the denominator-agnostic "no usable phase-binned events" signal.
        if feats is None or feats["n_in_isfs"] <= 0:
            continue
        rates[label] = np.asarray(feats["phase_bin_rates"])

        try:
            spec_data = subject_isfs_spectrum_features(data_dir, subject_id, channel, components,
                                                         event=event, min_bout_sec=min_bout_sec,
                                                         hypno_epoch_sec=hypno_epoch_sec)
        except Exception:  # noqa: BLE001 - ditto
            spec_data = None
        if spec_data is not None:
            spectrum[label] = spec_data

        try:
            curve = subject_phase_locked_probs_continuous(data, components=components, n_points=n_phase_points)
        except Exception:  # noqa: BLE001 - ditto
            continue
        if not np.isnan(curve).any():
            curves[label] = curve

    return subject_id, True, None, rates, curves, spectrum


def _categorize(reason: str) -> str:
    if "No U-Sleep hypnodensity directory" in reason:
        return "missing_usleep_subject_dir"
    if "No U-Sleep hypnodensity file for channel" in reason:
        return "missing_usleep_channel_file"
    if "temporal_ISFS" in reason:
        return "missing_temporal_isfs"
    if "bouts.npz" in reason:
        return "missing_bouts_file"
    if "no usable" in reason:
        return "no_usable_bouts"
    if "EOFError" in reason or "BadZipFile" in reason or "No data left in file" in reason:
        return "corrupt_or_empty_file"
    return "other"


def _mean_sem(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """`(mean, sem)` of `x` along `axis` (SEM uses `ddof=1`) -- the project's standard
    "group average" convention, no hypothesis testing (see `infraslow.stat.
    state_comparison.mean_sem`, which this mirrors but does not import)."""
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=axis)
    sem = x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])
    return mean, sem


# --------------------------------------------------------------------------- #
# Figure (Section 4.3's figure, one page per state)
# --------------------------------------------------------------------------- #
def _wrap_phase(phase: np.ndarray, center: float) -> np.ndarray:
    """Unwrap circular `phase` (values in `[-pi, pi]`) onto the branch centered on
    `center`, i.e. every value lands in `[center - pi, center + pi)` -- e.g. with
    `center=pi/2`, a bin at `-3*pi/4` (physically just before `pi`) is relabeled
    `5*pi/4` so it sorts after `pi` instead of wrapping back to the start."""
    return center + ((phase - center + np.pi) % (2 * np.pi) - np.pi)


def _phase_tick_label(val: float) -> str:
    """`val` (expected to be a multiple of `pi/2`) as e.g. `-pi`, `pi/2`, `3*pi/2`."""
    n = int(round(val / (np.pi / 2)))
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n_abs = abs(n)
    if n_abs % 2 == 0:
        k = n_abs // 2
        return f"{sign}{'' if k == 1 else k}pi"
    return f"{sign}{'' if n_abs == 1 else n_abs}pi/2"


def build_state_figure(
    state: str, *, channel: str, event_label: str, group_mean: np.ndarray, n_complete: int,
    phase_points: np.ndarray, bin_centers: np.ndarray, mean_r: np.ndarray, sem_r: np.ndarray,
    center: float = 0.0,
) -> plt.Figure:
    """`center` re-centers the circular phase axis on an arbitrary angle instead of
    `0` (e.g. `pi/2` puts both the ascending, phase=0, and descending, phase=+-pi,
    zero-crossings in view at once instead of splitting the descending one across
    the two edges of the default phase=0-centered axis)."""
    order = np.argsort(_wrap_phase(phase_points, center))
    phase_points = _wrap_phase(phase_points, center)[order]
    group_mean = group_mean[order]

    order_b = np.argsort(_wrap_phase(bin_centers, center))
    bin_centers = _wrap_phase(bin_centers, center)[order_b]
    mean_r = mean_r[order_b]
    sem_r = sem_r[order_b]

    stage_cols = [s.upper() for s in USLEEP_STAGE_ORDER]
    group_labels = [stage_cols[i] for i in np.argmax(group_mean, axis=1)]
    proba_df = pd.DataFrame(group_mean, columns=stage_cols)
    hyp_phase = yasa.Hypnogram(group_labels, n_stages=5, freq="1s", proba=proba_df)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1, 1.2]})
    hyp_phase.plot_hypnodensity(ax=ax_top)

    x_all = hyp_phase.timedelta.total_seconds() / 60
    if hyp_phase.duration > 90:
        x_all = x_all / 60
    tick_phases = center + np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    tick_idx = np.array([np.argmin(np.abs(phase_points - p)) for p in tick_phases])
    ax_top.set_xticks(x_all[tick_idx])
    ax_top.set_xticklabels([_phase_tick_label(t) for t in tick_phases])
    ax_top.set_xlabel("ISFS phase (rad)")
    ax_top.set_title(f"A) U-Sleep stage probability across ISFS phase "
                      f"({len(phase_points)} points, n={n_complete} subjects)")

    label_bbox = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8)
    bar_width = (2 * np.pi / 8) * 0.5
    # Color by the sign of the schematic sine (not the raw phase value), so this
    # stays correct once `center != 0` unwraps some bins past +-pi.
    bar_colors = [LPURPLE if np.sin(c) < 0 else LORANGE for c in bin_centers]
    ax_bottom.bar(bin_centers, mean_r, width=bar_width, yerr=sem_r, capsize=3,
                  color=bar_colors, edgecolor="0.3", zorder=2)
    for c, pct, err in zip(bin_centers, mean_r, sem_r):
        ax_bottom.text(c, pct + err, f"{pct:.1f}%", ha="center", va="bottom", fontsize=10,
                        fontweight="bold", bbox=label_bbox, zorder=6)

    y0, y1 = ax_bottom.get_ylim()
    span = y1 - y0
    amp = 0.50 * span
    x_smooth = np.linspace(center - np.pi, center + np.pi, 200)
    y_smooth = amp * np.sin(x_smooth)
    ax_bottom.plot(x_smooth, y_smooth, color="k", lw=2.0, ls="-", zorder=1, label="ISFS phase (schematic)")
    ax_bottom.axhline(0, color="0.5", lw=1.0, zorder=0)
    ax_bottom.set_ylim(min(y0, -amp - 0.03 * span), max(y1, amp + 0.03 * span))

    ax_bottom.set(xlim=(center - np.pi, center + np.pi),
                  xticks=list(tick_phases),
                  xticklabels=[_phase_tick_label(t) for t in tick_phases],
                  xlabel="ISFS phase (rad)", ylabel=f"% of {event_label} (in a valid ISFS cycle)")
    ax_bottom.set_title(f"B) {event_label.capitalize()} distribution across ISFS phase "
                         f"(n={n_complete} subjects)", fontsize=12, fontweight="bold")
    ax_bottom.tick_params(labelsize=10)

    yticks = ax_bottom.get_yticks()
    ax_bottom.set_yticks(yticks)
    ax_bottom.set_yticklabels([f"{t:g}" if t >= 0 else "" for t in yticks])

    ax_bottom.legend(handles=ax_bottom.get_legend_handles_labels()[0] + [
        Patch(color=LPURPLE, label="negative half of ISFS cycle (schematic sine < 0)"),
        Patch(color=LORANGE, label="positive half of ISFS cycle (schematic sine >= 0)")],
        loc="lower right", frameon=True, framealpha=0.9, fontsize=9)

    center_note = f" (centered on phase={_phase_tick_label(center)})" if center != 0 else ""
    fig.suptitle(f"{state} ({channel}): U-Sleep stage probability & {event_label} rate across "
                 f"ISFS phase{center_note}", fontweight="bold")
    fig.tight_layout()
    return fig


def build_correlation_figure(
    corr_summaries: Dict[str, pd.DataFrame], *, channel: str, event_label: str, stage_order: Sequence[str],
) -> plt.Figure:
    """Diverging bar chart of `phase_curve_correlation_summary`'s `mean_r` per stage,
    one panel per state in `corr_summaries` -- answers "does stage-X probability track
    (or anti-track) the {event}-rate curve across the ISFS phase cycle?" per stage."""
    # barh plots its first row at the bottom, so reverse stage_order here to get the
    # conventional top-to-bottom Wake..REM reading order (stage_order itself, e.g.
    # USLEEP_STAGE_ORDER, is left in its original order for every other caller).
    stage_order = list(reversed(stage_order))
    fig, axes = plt.subplots(1, len(corr_summaries), figsize=(5.5 * len(corr_summaries), 5.2),
                              sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    xlim = max(0.1, float(np.nanmax(np.abs(np.concatenate(
        [s["mean_r"].to_numpy() for s in corr_summaries.values()]
    )))) * 1.35)

    for ax, (state, summary) in zip(axes, corr_summaries.items()):
        summary = summary.set_index("stage").reindex(stage_order)
        y = np.arange(len(stage_order))
        colors = [CORR_POS if r >= 0 else CORR_NEG for r in summary["mean_r"]]
        ax.barh(y, summary["mean_r"], xerr=summary["sem_r"], color=colors,
                edgecolor="white", linewidth=0.6, height=0.62, capsize=3,
                error_kw=dict(ecolor=CORR_NEUTRAL, elinewidth=1.0))
        ax.axvline(0, color=CORR_NEUTRAL, linewidth=1.2, zorder=0)

        for yi, r, n, sig in zip(y, summary["mean_r"], summary["n"], summary["significant_FDR"]):
            if np.isnan(r):
                continue
            label_x = r + (0.02 * xlim if r >= 0 else -0.02 * xlim)
            label = f"{r:+.2f}" + (" *" if bool(sig) else "")
            ax.text(label_x, yi, label, va="center", ha="left" if r >= 0 else "right",
                    fontsize=10, fontweight="bold", color="#2a2a28")

        ax.set_yticks(y)
        ax.set_yticklabels(stage_order, fontsize=11)
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel(f"correlation r\n({event_label} % vs. stage probability, across ISFS phase)", fontsize=9)
        n_subjects = int(np.nanmax(summary["n"])) if len(summary) else 0
        ax.set_title(f"{state}-restricted phase\n(n = {n_subjects:,} subjects)",
                     fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(CORR_NEUTRAL)
        ax.tick_params(colors="#4a4a46")

    handles = [
        Patch(color=CORR_POS, label=f"tracks {event_label} rate (r > 0)"),
        Patch(color=CORR_NEG, label=f"anti-tracks {event_label} rate (r < 0)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.995,
              f"Does stage probability track {event_label} density across the ISFS phase cycle? ({channel})",
              ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.93,
              f"per-subject Pearson r ({event_label}-% curve vs. stage-probability curve over the phase cycle),\n"
              "one-sample t-test on Fisher-z(r), BH-FDR across stages; error bars = SEM; "
              "* = significant after FDR (q < 0.05)",
              ha="center", va="top", fontsize=9, color="#4a4a46")
    fig.tight_layout(rect=[0, 0.04, 1, 0.84])
    return fig


def build_isfs_spectrum_figure(
    state: str, *, channel: str, event_label: str, freqs: np.ndarray,
    corrected_mean: np.ndarray, corrected_sem: np.ndarray, fit: dict,
    real_peak_freq: float, n_subjects: int,
    infraslow_band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND,
) -> plt.Figure:
    """Group-level (subject-balanced) baseline-corrected ISFS spectrum with its
    bi-Gaussian fit -- the group-level counterpart of
    `demo_infraslow_yasa_average.ipynb`'s Fig C-i right panel (there: one cohort's
    grand-mean spectrum + a simple symmetric-Gaussian fit; here: each subject's own
    mean spectrum averaged first -- `_mean_sem` across `corrected_mean`'s per-subject
    rows -- then the bi-Gaussian `fit_isfs` this project's pipeline already uses
    elsewhere, matching `demo_infraslow_phase.ipynb` Figure 5B's styling)."""
    band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
    popt, mu = fit["popt"], fit["mu"]
    lo, hi = fit["lo"], fit["hi"]
    threshold, detected = fit["threshold"], fit["detected"]
    f_auc = np.linspace(lo, hi, 400)  # matches fit_isfs's own AUC integration grid

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(freqs[band_m], (corrected_mean - corrected_sem)[band_m],
                     (corrected_mean + corrected_sem)[band_m], color=LPURPLE, alpha=0.4,
                     zorder=2, label="mean +/- SEM (across subjects)")
    ax.plot(freqs[band_m], corrected_mean[band_m], color="0.15", lw=1.8, zorder=3,
             label="baseline-corrected")
    fg = np.linspace(*infraslow_band, 400)
    ax.plot(fg, bigaussian(fg, *popt), color=PURPLE, lw=2, zorder=4, label="bi-Gaussian fit")
    ax.axhline(threshold, ls=":", color="0.3", zorder=1, label="threshold (1.5x SD)")
    ax.fill_between(f_auc, bigaussian(f_auc, *popt), color=LPURPLE, alpha=0.7, zorder=2,
                     label="area under curve (+/-1 SD)")
    ax.plot([mu], [bigaussian(mu, *popt)], "o", color="k", ms=7, zorder=5,
             label="peak frequency (fit)")
    real_peak_y = corrected_mean[band_m][int(np.argmin(np.abs(freqs[band_m] - real_peak_freq)))]
    ax.plot([real_peak_freq], [real_peak_y], "x", color="crimson", ms=9, mew=2, zorder=6,
             label=f"peak at {real_peak_freq:.4f} Hz")
    ax.hlines(bigaussian(lo, *popt), lo, hi, color="k", lw=2, zorder=5, label="bandwidth (+/-1 SD)")
    ax.set(xlim=infraslow_band, xlabel="Frequency (Hz)", ylabel="Relative spectral power",
           title=f"{state} ({channel}): group ISFS spectrum -- {event_label} bouts\n"
                 f"(n={n_subjects} subjects, ISFS detected: {detected})")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    return fig


def build_peak_distribution_figure(
    state: str, *, channel: str, event_label: str, subject_peak_freqs: np.ndarray,
    group_fit_peak: float, group_real_peak: float,
    infraslow_band: Tuple[float, float] = DEFAULT_INFRASLOW_BAND,
) -> plt.Figure:
    """Histogram of each subject's own empirical ISFS peak frequency
    (`feats["real_peak_freq_hz"]`, argmax of that subject's own mean relative-power
    spectrum across their `event`-containing bouts) -- one value per subject, i.e.
    each subject's own bouts are already averaged together before this histogram
    pools across subjects, never raw per-bout peaks pooled directly (that per-bout
    detail is instead saved in full to `isfs_bout_peak_freqs_{state}.csv`, one row
    per subject)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bin_w = 0.005
    bins = np.arange(infraslow_band[0], infraslow_band[1] + bin_w, bin_w)
    ax.hist(subject_peak_freqs, bins=bins, color=LPURPLE, edgecolor="0.3", zorder=2)
    ax.axvline(group_real_peak, color="crimson", lw=2, ls="--", zorder=4,
               label=f"mean-spectrum peak ({group_real_peak:.4f} Hz)")
    ax.axvline(group_fit_peak, color=PURPLE, lw=2, ls="-", zorder=4,
               label=f"bi-Gaussian fit peak ({group_fit_peak:.4f} Hz)")
    ax.set(xlim=infraslow_band, xlabel="Peak frequency (Hz)", ylabel="Number of subjects",
           title=f"{state} ({channel}): distribution of subject-level ISFS peak frequency\n"
                 f"({event_label}, n={len(subject_peak_freqs)} subjects)")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Per-subject checkpoint cache -- lets a resumed run skip subjects a prior,
# killed/timed-out/OOM'd invocation already finished, instead of reprocessing
# the whole candidate list from scratch.
# --------------------------------------------------------------------------- #
# Fixed key order for flattening a subject_isfs_spectrum_features "feats" dict into a
# single float array for the npz cache (and back) -- must match
# infraslow.pipeline.spectrum.compute_subject_spectrum_features's returned keys exactly.
_SPECTRUM_FEAT_KEYS: Tuple[str, ...] = (
    "peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc", "chromatogram_peak_area",
    "bi_gaussian_amp", "bi_gaussian_mu", "bi_gaussian_sd_l", "bi_gaussian_sd_r",
    "isfs_detected", "isfs_threshold", "real_peak_freq_hz", "real_peak_period_s",
)


def _cache_path(cache_dir: Path, subject_id: str) -> Path:
    return cache_dir / f"{subject_id}.npz"


def _write_cache(
    cache_dir: Path, subject_id: str, ok: bool, reason: Optional[str],
    rates: Dict[str, np.ndarray], curves: Dict[str, np.ndarray],
    spectrum: Optional[Dict[str, dict]] = None,
) -> None:
    """Persist one subject's `_process_subject` result, written atomically (temp
    file + rename) so a job killed mid-write never leaves a corrupt cache entry
    that a later resume would have to guard against. `spectrum` defaults to `{}`
    (optional) so callers that never compute it -- e.g.
    `plot_hypnodensity_isfs_phase_transition.py`, which imports this function
    unchanged -- don't need to pass it."""
    payload = {"ok": np.array(ok), "reason": np.array(reason or "")}
    for state, arr in rates.items():
        payload[f"rates_{state}"] = arr
    for state, arr in curves.items():
        payload[f"curves_{state}"] = arr
    for state, spec_data in (spectrum or {}).items():
        payload[f"spec_freqs_{state}"] = spec_data["freqs"]
        payload[f"spec_rel_{state}"] = spec_data["rel"]
        payload[f"spec_corrected_{state}"] = spec_data["corrected"]
        payload[f"spec_bout_peaks_{state}"] = spec_data["bout_peak_freqs"]
        payload[f"spec_feats_{state}"] = np.array(
            [float(spec_data["feats"][k]) for k in _SPECTRUM_FEAT_KEYS]
        )
    # Must already end in ".npz" -- np.savez silently appends ".npz" to any filename
    # that doesn't, which would otherwise write "<subject_id>.tmp.npz" while this
    # variable still points at "<subject_id>.npz.tmp", breaking the rename below.
    tmp_path = cache_dir / f"{subject_id}.tmp.npz"
    np.savez(tmp_path, **payload)
    tmp_path.replace(_cache_path(cache_dir, subject_id))


def _read_cache(
    cache_dir: Path, subject_id: str, group_labels: Sequence[str],
) -> Optional[Tuple[bool, Optional[str], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, dict]]]:
    """This subject's cached `_process_subject` result, or `None` if there is no
    cache entry yet -- or it's unreadable (e.g. left mid-write by a killed job
    before the atomic rename in `_write_cache` took effect), in which case the
    subject is simply reprocessed instead of failing the whole run. Always returns
    a 5-tuple (`spectrum` is `{}` for a cache entry written before spectrum support
    was added, or by a caller -- e.g. the transition script -- that never passes it).

    `group_labels` must match the current run's `--states` groups (see
    `_parse_state_groups`) -- a label not present in this cache entry (e.g. a
    merged group requested for the first time against an existing cache) is
    simply absent from the returned dicts, not an error; that group is then
    reprocessed for this subject the next time it's actually resubmitted (see
    the module docstring: delete `cache/` to force a full recompute)."""
    path = _cache_path(cache_dir, subject_id)
    if not path.exists():
        return None
    try:
        with np.load(path) as npz:
            ok = bool(npz["ok"])
            reason = str(npz["reason"][()]) or None
            rates = {s: npz[f"rates_{s}"] for s in group_labels if f"rates_{s}" in npz.files}
            curves = {s: npz[f"curves_{s}"] for s in group_labels if f"curves_{s}" in npz.files}
            spectrum = {}
            for s in group_labels:
                if f"spec_freqs_{s}" not in npz.files:
                    continue
                spectrum[s] = dict(
                    freqs=npz[f"spec_freqs_{s}"], rel=npz[f"spec_rel_{s}"],
                    corrected=npz[f"spec_corrected_{s}"], bout_peak_freqs=npz[f"spec_bout_peaks_{s}"],
                    feats=dict(zip(_SPECTRUM_FEAT_KEYS, npz[f"spec_feats_{s}"].tolist())),
                )
        return ok, reason, rates, curves, spectrum
    except Exception:  # noqa: BLE001 - corrupt/partial cache entry -- reprocess, don't crash
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    data_dir = _resolve_data_dir(args.preprocess_dir, args.data_dir, args.hypno_epoch_sec)
    # Namespaced by event ("spindle"/"sw") so a spindle run and a sw run against the
    # same --output-dir never share a hypnodensity_isfs_phase.png/exclusions.csv/group_stats
    # -- lets both be submitted as concurrent jobs.
    args.output_dir = args.output_dir / args.event
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "group_stats").mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(args.output_dir / "progress.log", mode="a"),
            logging.StreamHandler(),
        ],
    )

    groups = _parse_state_groups(args.states)
    group_labels = tuple(label for label, _ in groups)
    all_leaf_stages = _all_leaf_stages(groups)
    n_subjects = args.n_subjects if args.n_subjects and args.n_subjects > 0 else None
    workers = max(1, args.workers)

    candidates = pio.discover_subjects(data_dir)
    if args.filter_by_events:
        events_csv_path = args.events_csv or _events_csv_path(
            args.preprocess_dir, args.channel, args.hypno_epoch_sec,
        )
        events_df = pd.read_csv(events_csv_path, dtype={"id": str})
        n_before = len(candidates)
        candidates = _filter_subjects_with_events(candidates, events_df, all_leaf_stages, args.event)
        logger.info(f"--filter-by-events: {n_before} -> {len(candidates)} candidate(s) with "
                    f">=1 detected {args.event} event in {all_leaf_stages} (from {events_csv_path})")
    if args.limit:
        candidates = candidates[: args.limit]
    logger.info(f"{len(candidates)} candidate subject(s) under {data_dir}; "
                f"channel={args.channel} groups={group_labels} event={args.event!r} "
                f"n_subjects={n_subjects or 'unlimited'} workers={workers} "
                f"hypno_epoch_sec={args.hypno_epoch_sec} usleep_epoch_sec={args.usleep_epoch_sec}")

    included: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, dict]]] = {}
    exclusion_reasons: Dict[str, str] = {}
    n_scanned = 0
    chunk_size = max(workers * 4, workers)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(candidates), chunk_size):
            if n_subjects is not None and len(included) >= n_subjects:
                break
            chunk = candidates[start:start + chunk_size]

            # Subjects a prior (killed/timed-out/OOM'd) run already finished are
            # served straight from cache/, never resubmitted to the pool -- this is
            # what makes a resumed run only process subjects "not yet run".
            to_process = []
            n_from_cache = 0
            for sid in chunk:
                cached = _read_cache(cache_dir, sid, group_labels)
                if cached is None:
                    to_process.append(sid)
                    continue
                ok, reason, rates, curves, spectrum = cached
                n_scanned += 1
                n_from_cache += 1
                if ok:
                    included[sid] = (rates, curves, spectrum)
                else:
                    exclusion_reasons[sid] = reason

            if to_process:
                tasks = [
                    (sid, data_dir, args.usleep_dir, args.channel, groups, args.event,
                     args.min_bout_sec, args.n_phase_points, args.usleep_epoch_sec,
                     args.hypno_epoch_sec)
                    for sid in to_process
                ]
                for subject_id, ok, reason, rates, curves, spectrum in pool.map(_process_subject, tasks):
                    n_scanned += 1
                    # Written immediately, one subject at a time, so a crash/OOM/timeout
                    # partway through this chunk only costs the in-flight subjects, not
                    # the ones already finished.
                    _write_cache(cache_dir, subject_id, ok, reason, rates, curves, spectrum)
                    if ok:
                        included[subject_id] = (rates, curves, spectrum)
                    else:
                        exclusion_reasons[subject_id] = reason

            logger.info(f"scanned {n_scanned}; included {len(included)}/{n_subjects or '?'} so far "
                        f"({n_from_cache} from cache this chunk)")

    subject_ids = list(included)
    if n_subjects is not None:
        subject_ids = subject_ids[:n_subjects]
    logger.info(f"scanned {n_scanned} subject(s); included {len(included)}, "
                f"excluded {len(exclusion_reasons)}; using {len(subject_ids)} for the group figure")

    exclusion_df = pd.DataFrame(
        [{"subject_id": sid, "reason_category": _categorize(reason), "detail": reason}
         for sid, reason in exclusion_reasons.items()]
    )
    exclusion_df.to_csv(args.output_dir / "exclusions.csv", index=False)
    if len(exclusion_df):
        logger.info("Excluded, by reason:\n" + exclusion_df["reason_category"].value_counts().to_string())

    assert subject_ids, "no usable subjects found -- check --data-dir/--usleep-dir/--channel"

    bin_centers = pph.bin_center_angle(np.arange(1, 9))
    phase_edges = np.linspace(-np.pi, np.pi, args.n_phase_points + 1)
    phase_points = (phase_edges[:-1] + phase_edges[1:]) / 2
    event_label = _EVENT_SPECS[args.event]["event_label"]

    state_summary: Dict[str, Dict[str, int]] = {}
    corr_summaries: Dict[str, pd.DataFrame] = {}
    stage_order = list(USLEEP_STAGE_ORDER)

    for state in group_labels:
        rates_by_sid = {sid: included[sid][0][state] for sid in subject_ids if state in included[sid][0]}
        curves_by_sid = {sid: included[sid][1][state] for sid in subject_ids if state in included[sid][1]}
        n_rates = len(rates_by_sid)
        logger.info(f"{state}: {n_rates}/{len(subject_ids)} subjects have a usable "
                    f"{args.event} phase distribution.")
        assert rates_by_sid, f"no subject has a usable {state} {args.event} phase distribution"

        # curves_by_sid is already a subset of rates_by_sid (_process_subject only
        # computes a curve for a state it already has rates for) -- no intersection needed.
        common_ids = sorted(curves_by_sid)
        logger.info(f"{state}: {len(common_ids)}/{n_rates} {args.event}-usable subjects also "
                    f"have complete {args.n_phase_points}-point phase coverage.")
        assert common_ids, (
            f"no subject has complete {args.n_phase_points}-point {state} phase coverage "
            f"-- try raising --n-subjects/--limit"
        )
        state_summary[state] = dict(n_rates=n_rates, n_common=len(common_ids))

        group_stack = np.stack([curves_by_sid[sid] for sid in common_ids])
        group_mean, group_sem = _mean_sem(group_stack, axis=0)

        rates_stack = np.stack([rates_by_sid[sid] for sid in common_ids])
        mean_r, sem_r = _mean_sem(rates_stack)

        np.savez(
            args.output_dir / "group_stats" / f"{state}.npz",
            group_mean=group_mean, group_sem=group_sem, phase_points=phase_points,
            phase_bin_rates=rates_stack, bin_centers=bin_centers,
            n_complete=len(common_ids), subject_ids=np.asarray(common_ids),
        )

        # Does stage probability track (or anti-track) the event-rate curve across the
        # ISFS phase cycle? Per-subject Pearson r (event-rate curve, resampled onto the
        # same phase_points grid, vs. each stage's own phase-locked probability curve --
        # group_stack is already that curve stack), then a group-level one-sample test.
        rate_curves = np.stack([
            pph.resample_bin_rates_to_points(rates_by_sid[sid], bin_centers, phase_points)
            for sid in common_ids
        ])
        r_matrix = stc.per_subject_curve_correlation(rate_curves, group_stack)
        corr_summary = stc.phase_curve_correlation_summary(r_matrix, stage_order)
        corr_summary.to_csv(args.output_dir / f"stat_summary_{state}.csv", index=False)
        logger.info(f"{state}: saved stat_summary_{state}.csv\n" + corr_summary.to_string(index=False))
        corr_summaries[state] = corr_summary

        pd.DataFrame({"subject_id": common_ids}).to_csv(
            args.output_dir / f"usable_subjects_{state}.csv", index=False
        )

        fig = build_state_figure(
            state, channel=args.channel, event_label=event_label, group_mean=group_mean,
            n_complete=len(common_ids), phase_points=phase_points, bin_centers=bin_centers,
            mean_r=mean_r, sem_r=sem_r,
        )
        # One PNG per --states group -- always separate files, never combined
        # into a shared multi-page figure.
        png_path = args.output_dir / f"hypnodensity_isfs_phase_{state}.png"
        fig.savefig(png_path)
        plt.close(fig)
        logger.info(f"saved {state} figure to {png_path}")

        # Same data, phase axis re-centered on pi/2 -- puts both the ascending
        # (phase=0) and descending (phase=+-pi) zero-crossings in view at once,
        # instead of splitting the descending one across the default figure's edges.
        fig_c = build_state_figure(
            state, channel=args.channel, event_label=event_label, group_mean=group_mean,
            n_complete=len(common_ids), phase_points=phase_points, bin_centers=bin_centers,
            mean_r=mean_r, sem_r=sem_r, center=np.pi / 2,
        )
        png_path_c = args.output_dir / f"hypnodensity_isfs_phase_{state}_centered_pi2.png"
        fig_c.savefig(png_path_c)
        plt.close(fig_c)
        logger.info(f"saved {state} pi/2-centered figure to {png_path_c}")

        # --- ISFS spectral peak-frequency analysis (group-level, subject-balanced) ---
        # Its own subject set (spectrum_ids), not common_ids: this only needs a usable
        # event-containing-bout spectrum, not Section 4.2's continuous-phase coverage.
        spectrum_by_sid = {sid: included[sid][2][state] for sid in subject_ids if state in included[sid][2]}
        if not spectrum_by_sid:
            logger.warning(f"{state}: no subject has a usable ISFS spectrum -- skipping "
                            "spectrum figure/CSV outputs")
        else:
            spectrum_ids = sorted(spectrum_by_sid)
            freqs = spectrum_by_sid[spectrum_ids[0]]["freqs"]
            rel_stack = np.stack([spectrum_by_sid[sid]["rel"] for sid in spectrum_ids])
            corrected_stack = np.stack([spectrum_by_sid[sid]["corrected"] for sid in spectrum_ids])
            # Subject-balanced: each subject's own mean spectrum (already averaged across
            # their event-containing bouts in subject_isfs_spectrum_features) first, then
            # mean +/- SEM across subjects -- matches demo_infraslow_yasa_average.ipynb's
            # Fig C-i methodology, never raw per-bout PSDs pooled across subjects.
            rel_mean, _rel_sem = _mean_sem(rel_stack)
            corrected_mean, corrected_sem = _mean_sem(corrected_stack)

            spec_band_m = (freqs >= DEFAULT_INFRASLOW_BAND[0]) & (freqs <= DEFAULT_INFRASLOW_BAND[1])
            real_peak_idx = int(np.argmax(rel_mean[spec_band_m]))
            real_peak_freq = float(freqs[spec_band_m][real_peak_idx])
            group_fit = fit_isfs(freqs, corrected_mean, infraslow_band=DEFAULT_INFRASLOW_BAND,
                                  baseline_band=DEFAULT_BASELINE_BAND)
            logger.info(f"{state}: group ISFS fit -- detected={group_fit['detected']} "
                        f"peak={group_fit['mu']:.4f} Hz auc={group_fit['auc']:.3g} "
                        f"(n={len(spectrum_ids)} subjects)")

            np.savez(
                args.output_dir / "group_stats" / f"{state}_isfs_spectrum.npz",
                freqs=freqs, rel_mean=rel_mean, corrected_mean=corrected_mean,
                corrected_sem=corrected_sem, real_peak_freq_hz=real_peak_freq,
                fit_popt=group_fit["popt"], fit_mu=group_fit["mu"], fit_lo=group_fit["lo"],
                fit_hi=group_fit["hi"], fit_bandwidth=group_fit["bandwidth"], fit_auc=group_fit["auc"],
                fit_threshold=group_fit["threshold"], fit_detected=group_fit["detected"],
                n_subjects=len(spectrum_ids), subject_ids=np.asarray(spectrum_ids),
            )

            spec_fig = build_isfs_spectrum_figure(
                state, channel=args.channel, event_label=event_label, freqs=freqs,
                corrected_mean=corrected_mean, corrected_sem=corrected_sem, fit=group_fit,
                real_peak_freq=real_peak_freq, n_subjects=len(spectrum_ids),
            )
            spec_png_path = args.output_dir / f"isfs_spectrum_{state}.png"
            spec_fig.savefig(spec_png_path)
            plt.close(spec_fig)
            logger.info(f"saved {state} ISFS spectrum figure to {spec_png_path}")

            # Every per-subject spectral fit value, one row per subject.
            feat_df = pd.DataFrame(
                [{"subject_id": sid, **spectrum_by_sid[sid]["feats"]} for sid in spectrum_ids]
            )
            feat_csv_path = args.output_dir / f"isfs_spectrum_features_{state}.csv"
            feat_df.to_csv(feat_csv_path, index=False)
            logger.info(f"saved {state} per-subject ISFS spectral features to {feat_csv_path}")

            # Subject-balanced peak-frequency distribution: one value per subject (that
            # subject's own real/empirical peak, from their own bout-averaged spectrum),
            # never raw per-bout peaks pooled directly across subjects.
            subject_peak_freqs = np.array(
                [spectrum_by_sid[sid]["feats"]["real_peak_freq_hz"] for sid in spectrum_ids]
            )
            peak_fig = build_peak_distribution_figure(
                state, channel=args.channel, event_label=event_label,
                subject_peak_freqs=subject_peak_freqs, group_fit_peak=group_fit["mu"],
                group_real_peak=real_peak_freq,
            )
            peak_png_path = args.output_dir / f"isfs_peak_distribution_{state}.png"
            peak_fig.savefig(peak_png_path)
            plt.close(peak_fig)
            logger.info(f"saved {state} peak-frequency distribution figure to {peak_png_path}")

            # Every individual event-containing bout's own empirical peak frequency, one
            # row per subject -- unlike the plot above (one subject-balanced value per
            # subject), this keeps every bout's peak, ragged/NaN-padded to the max bout
            # count across subjects since subjects have differing bout counts.
            max_bouts = max(len(spectrum_by_sid[sid]["bout_peak_freqs"]) for sid in spectrum_ids)
            peak_cols = [f"peak_{i + 1}" for i in range(max_bouts)]
            bout_peak_rows = []
            for sid in spectrum_ids:
                peaks_sid = spectrum_by_sid[sid]["bout_peak_freqs"]
                row = {"subject_id": sid, "n_bouts": len(peaks_sid)}
                row.update({col: (peaks_sid[i] if i < len(peaks_sid) else np.nan)
                             for i, col in enumerate(peak_cols)})
                bout_peak_rows.append(row)
            bout_peak_df = pd.DataFrame(bout_peak_rows, columns=["subject_id", "n_bouts"] + peak_cols)
            bout_peak_csv_path = args.output_dir / f"isfs_bout_peak_freqs_{state}.csv"
            bout_peak_df.to_csv(bout_peak_csv_path, index=False)
            logger.info(f"saved {state} per-subject per-bout peak frequencies to {bout_peak_csv_path}")

    corr_fig = build_correlation_figure(
        corr_summaries, channel=args.channel, event_label=event_label, stage_order=stage_order,
    )
    corr_png_path = args.output_dir / f"{args.event}_stage_phase_corr.png"
    corr_fig.savefig(corr_png_path, dpi=150, bbox_inches="tight")
    plt.close(corr_fig)
    logger.info(f"saved stage/phase correlation figure to {corr_png_path}")

    summary_lines = [
        "=== Subject inclusion summary ===",
        f"Scanned:  {n_scanned}",
        f"Included: {len(included)}",
        f"Excluded: {len(exclusion_reasons)}",
    ]
    if len(exclusion_df):
        summary_lines.append("\nExclusion reasons:")
        for reason, count in exclusion_df["reason_category"].value_counts().items():
            summary_lines.append(f"  {reason}: {count}")
    summary_lines.append(f"\nEVENT = {args.event!r} ({event_label}, band={_EVENT_SPECS[args.event]['band']!r})")

    summary_lines.append(f"\nSection 4.1 ({args.event} phase-bin distribution) usable/excluded, out of "
                          f"{len(subject_ids)} included subjects:")
    for state in group_labels:
        n_usable = state_summary[state]["n_rates"]
        summary_lines.append(f"  {state}: {n_usable} usable, {len(subject_ids) - n_usable} excluded")

    summary_lines.append("\nSection 4.2 (continuous phase-locked hypnodensity) usable/excluded, out of each "
                          "state's Section 4.1-usable subjects:")
    for state in group_labels:
        n_attempted = state_summary[state]["n_rates"]
        n_usable = state_summary[state]["n_common"]
        summary_lines.append(f"  {state}: {n_usable} usable, {n_attempted - n_usable} excluded "
                              f"(of {n_attempted} attempted)")

    summary_lines.append("\nSubjects contributing to each state's figure (usable for both Section 4.1's "
                          "event distribution and Section 4.2's continuous curve):")
    for state in group_labels:
        summary_lines.append(f"  {state}: {state_summary[state]['n_common']}/{len(subject_ids)}")

    summary_text = "\n".join(summary_lines) + "\n"
    (args.output_dir / "summary.txt").write_text(summary_text)
    logger.info("\n" + summary_text)


if __name__ == "__main__":
    main()
