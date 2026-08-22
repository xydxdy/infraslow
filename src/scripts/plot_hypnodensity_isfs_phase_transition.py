#!/usr/bin/env python
"""Pre-transition variant of plot_hypnodensity_isfs_phase.py: restricts every
N2/N3 bout to just its last --window-sec seconds (default 200s), and only
keeps bouts that end in a real, SUSTAINED transition to a different scored
sleep stage -- see infraslow.pipeline.transitions.select_transition_tails.
`find_stage_bouts` builds each bout as a *maximal* run of same-stage epochs,
so the epoch right after any bout's end already differs from the bout's own
stage by construction -- checking only that one epoch is therefore not a
meaningful filter (true for nearly every bout). --persist-sec (default 60s)
is what actually distinguishes a lasting transition from a single-epoch
scoring blip that reverts right back (which shows up as two adjacent,
same-stage bouts under find_stage_bouts's maximal-run construction, not one
bout followed by "more of the same stage"): the new stage must hold for at
least --persist-sec seconds after the bout's end (or through to the end of
the scored night, if less remains) without reverting to the bout's own
stage. Non-transition and too-short bouts are dropped entirely, not
truncated -- see
docs/superpowers/specs/2026-08-22-hypnodensity-isfs-phase-transition-design.md.

Everything except bout selection is unchanged from plot_hypnodensity_isfs_phase.py
and is imported from it directly: subject_phase_locked_probs_continuous,
build_state_figure, build_correlation_figure, _EVENT_SPECS, _mean_sem,
_write_cache, _read_cache. Only load_subject,
subject_event_phase_bin_rates, _process_subject, and _categorize are
reimplemented here, since the original inlines bout-loading exactly where
this script must change it. Requires plot_hypnodensity_isfs_phase.py to sit
in the same directory (relies on Python adding the invoked script's own
directory to sys.path[0]) -- run this from src/scripts/, same as the original.

Run via Slurm, not the login node, from this file's own directory
(``src/scripts/``) with ``src/`` on ``PYTHONPATH``, e.g.::

    export PYTHONPATH=/home/users/chaisaen/infraslow/src
    srun -p normal --time=00:30:00 --mem=8G --cpus-per-task=4 \\
        python3 plot_hypnodensity_isfs_phase_transition.py --n-subjects 100 --workers 4

See ``run_hypnodensity_isfs_phase_transition.sbatch`` to submit this as a Slurm job.

Saves the same files as plot_hypnodensity_isfs_phase.py (see that script's
module docstring for the full list), under --output-dir/--event, but
--output-dir defaults to $SCRATCH/outputs/hypnodensity_isfs_phase_transition
-- a different root, so this never collides with the original's production
outputs. cache/ is namespaced by --window-sec and --persist-sec (e.g.
cache/w200_p60/) so switching either parameter against the same --output-dir
never silently reuses a prior parameter combination's results (same caveat
the original documents for --n-phase-points/--min-bout-sec, but enforced
here rather than left to the operator to remember).
"""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infraslow.pipeline import io as pio
from infraslow.pipeline import phase as pph
from infraslow.pipeline import transitions as ptr
from infraslow.io import usleep_hypnodensity as uh
from infraslow.io.hypnodensity import load_subject_hypnogram
from infraslow.stat import state_comparison as stc
from infraslow.constants import (
    DEFAULT_HYPNO_DIR,
    DEFAULT_MIN_BOUT_SEC,
    DEFAULT_USLEEP_HYPNODENSITY_DIR,
    USLEEP_STAGE_ORDER,
)
from plot_hypnodensity_isfs_phase import (
    _EVENT_SPECS,
    _mean_sem,
    _read_cache,
    _write_cache,
    build_correlation_figure,
    build_state_figure,
    subject_phase_locked_probs_continuous,
)

logger = logging.getLogger(__name__)

plt.rcParams["figure.dpi"] = 110

#: Pre-transition tail length -- deliberately a local constant, not
#: infraslow.constants.DEFAULT_WINDOW_SEC (an unrelated ISFS-spectrum
#: frequency-grid constant of the same generic name) or DEFAULT_MIN_BOUT_SEC
#: (a different, coincidentally-equal-valued knob) -- so changing either of
#: those elsewhere in the codebase can never silently move this default.
DEFAULT_TRANSITION_WINDOW_SEC = 200.0
#: Minimum time (s) the new stage must persist after a bout's end, without
#: reverting to the bout's own stage, to count as a real transition -- see
#: infraslow.pipeline.transitions.is_transition_bout. 60s (2 epochs at the
#: scored hypnogram's 30s resolution) filters out a single-epoch scoring
#: blip that reverts right back, without requiring an implausibly long
#: stable stretch.
DEFAULT_TRANSITION_PERSIST_SEC = 60.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None else val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                    default=Path(os.path.expandvars(
                        _env_str("DATA_DIR", "$SCRATCH/processed_data/data"))),
                    help="preprocessing.py output tree, one subdir per subject (env: DATA_DIR)")
    p.add_argument("--usleep-dir", default=os.path.expandvars(
                        _env_str("USLEEP_DIR", DEFAULT_USLEEP_HYPNODENSITY_DIR)),
                    help="U-Sleep hypnodensity directory (env: USLEEP_DIR)")
    p.add_argument("--hypno-dir", default=os.path.expandvars(
                        _env_str("HYPNO_DIR", DEFAULT_HYPNO_DIR)),
                    help="Raw Bioserenity Hypnodensity CSV directory, used to detect real "
                         "stage transitions (env: HYPNO_DIR)")
    p.add_argument("--channel", default=_env_str("CHANNEL", "C3"), help="EEG channel (env: CHANNEL)")
    p.add_argument("--states", nargs="+", default=["N2", "N3"],
                    help="Stages to analyze, each treated as its own group/figure, never pooled")
    p.add_argument("--event", choices=sorted(_EVENT_SPECS), default="spindle",
                    help="Which event type's phase-bin distribution/band to use "
                         "(sets BAND: spindle -> sigma, sw -> delta)")
    p.add_argument("--min-bout-sec", type=float,
                    default=float(_env_str("MIN_BOUT_SEC", str(DEFAULT_MIN_BOUT_SEC))),
                    help="Minimum consecutive-stage bout length (s) before transition "
                         "selection (env: MIN_BOUT_SEC)")
    p.add_argument("--window-sec", type=float,
                    default=float(_env_str("TRANSITION_WINDOW_SEC", str(DEFAULT_TRANSITION_WINDOW_SEC))),
                    help="Pre-transition tail length (s): a bout must be at least this "
                         "long and end in a real, sustained stage transition to be "
                         "included, and only its last --window-sec seconds are used "
                         "(env: TRANSITION_WINDOW_SEC)")
    p.add_argument("--persist-sec", type=float,
                    default=float(_env_str("TRANSITION_PERSIST_SEC", str(DEFAULT_TRANSITION_PERSIST_SEC))),
                    help="Minimum time (s) the new stage must persist after a bout's "
                         "end, without reverting back to the bout's own stage, to "
                         "count as a real transition rather than a single-epoch "
                         "scoring blip -- see infraslow.pipeline.transitions."
                         "is_transition_bout (env: TRANSITION_PERSIST_SEC)")
    p.add_argument("--n-phase-points", type=int, default=50,
                    help="Resolution of the continuous phase-locked group curve")
    p.add_argument("--n-subjects", type=int, default=100,
                    help="Stop once this many subjects are included; pass 0/negative to "
                         "disable and use every candidate")
    p.add_argument("--limit", type=int, default=None,
                    help="Cap the candidate pool to the first N discovered subjects "
                         "(quick tests); default scans every subject under --data-dir")
    p.add_argument("--workers", type=int,
                    default=int(_env_str("WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
                    help="Parallel worker processes (env: WORKERS, falls back to "
                         "$SLURM_CPUS_PER_TASK, else 1)")
    p.add_argument("--output-dir", type=Path,
                    default=Path(os.path.expandvars(
                        _env_str("OUTPUT_DIR", "$SCRATCH/outputs/hypnodensity_isfs_phase_transition"))),
                    help="Where to save the PDF, exclusions.csv, group_stats/, and progress.log "
                         "(env: OUTPUT_DIR)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Per-subject computation (module-level so ProcessPoolExecutor can pickle it)
# --------------------------------------------------------------------------- #
def load_subject(data_dir: Path, usleep_dir: str, subject_id: str, channel: str,
                  states: Sequence[str], band: str, stage_epochs: np.ndarray, epoch_sec: float, *,
                  window_sec: float, persist_epochs: int,
                  min_bout_sec: float = DEFAULT_MIN_BOUT_SEC) -> dict:
    """Same contract as `plot_hypnodensity_isfs_phase.load_subject`, but each
    state's bouts are first reduced to their pre-transition tails via
    `infraslow.pipeline.transitions.select_transition_tails` before the phase
    timeseries is built.

    Raises:
        FileNotFoundError: missing U-Sleep hypnodensity, or missing temporal_ISFS/bouts
            artifacts.
        ValueError: no usable transition-tail bouts.
    """
    subject_dir = Path(data_dir) / subject_id
    t_hyp, probs = uh.load_usleep_hypnodensity(subject_id, channel, base_dir=usleep_dir)
    t_env, filtered = pio.load_temporal_isfs(subject_dir, channel, band)

    phase_parts = []
    for state in states:
        bouts = pio.load_stage_bouts(subject_dir, channel, state)["all"]
        if bouts.shape[0]:
            durations = bouts[:, 1] - bouts[:, 0]
            bouts = bouts[durations >= min_bout_sec]
        bouts = ptr.select_transition_tails(bouts, stage_epochs, epoch_sec, state, window_sec,
                                             persist_epochs=persist_epochs)
        if bouts.shape[0]:
            series = pph.build_subject_phase_timeseries(t_env, filtered, bouts)
            if series["t"].size:
                continuous = pph.build_subject_continuous_phase_timeseries(t_env, filtered, bouts)
                series["phase_continuous"] = continuous["phase_angle"]
                phase_parts.append(pd.DataFrame(series).assign(state=state))

    if not phase_parts:
        raise ValueError(
            f"no usable {'/'.join(states)} transition-tail bouts (>= {window_sec}s, "
            "ending in a real stage transition) for this subject"
        )

    phase = pd.concat(phase_parts, ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)
    return dict(t_hyp=t_hyp, probs=probs, phase=phase)


def subject_event_phase_bin_rates(data_dir: Path, subject_id: str, channel: str, state: str,
                                   stage_epochs: np.ndarray, epoch_sec: float, *,
                                   event: str, window_sec: float, persist_epochs: int,
                                   min_bout_sec: float = DEFAULT_MIN_BOUT_SEC):
    """Same contract as `plot_hypnodensity_isfs_phase.subject_event_phase_bin_rates`,
    restricted to this state's pre-transition-tail bouts."""
    spec = _EVENT_SPECS[event]
    subject_dir = Path(data_dir) / subject_id
    bouts = spec["load_bouts"](subject_dir, channel, state)[spec["bouts_key"]]
    if bouts.shape[0]:
        durations = bouts[:, 1] - bouts[:, 0]
        bouts = bouts[durations >= min_bout_sec]
    bouts = ptr.select_transition_tails(bouts, stage_epochs, epoch_sec, state, window_sec,
                                         persist_epochs=persist_epochs)
    if bouts.shape[0] == 0:
        return None

    t_env, filtered = pio.load_temporal_isfs(subject_dir, channel, spec["band"])
    event_summary = spec["load_summary"](subject_dir, channel, state)
    peak_col = spec["peak_col"]
    event_times = event_summary[peak_col].to_numpy() if peak_col in event_summary.columns else np.empty(0)

    bouts_data = pph.build_bout_phase_data(t_env, filtered, bouts, event_times)
    return pph.compute_subject_phase_features(bouts_data)


def _process_subject(
    task: Tuple[str, Path, str, str, str, Tuple[str, ...], str, float, float, float, int],
) -> Tuple[str, bool, Optional[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Same contract as `plot_hypnodensity_isfs_phase._process_subject`, but
    loads this subject's scored hypnogram once (`load_subject_hypnogram`) and
    threads it through both `load_subject` and `subject_event_phase_bin_rates`
    so the raw Hypnodensity CSV is only read once per subject."""
    (subject_id, data_dir, usleep_dir, hypno_dir, channel, states, event,
     min_bout_sec, window_sec, persist_sec, n_phase_points) = task
    band = _EVENT_SPECS[event]["band"]

    try:
        stage_epochs, epoch_sec = load_subject_hypnogram(subject_id, hypno_dir=hypno_dir)
    except Exception as exc:  # noqa: BLE001 - missing/corrupt Hypnodensity CSV excludes
        # this subject, same as any other missing per-subject artifact -- must never
        # crash the whole worker pool.
        return subject_id, False, f"{type(exc).__name__}: {exc}", {}, {}

    # >= 1 always -- persist_sec=0 would otherwise pass persist_epochs=0 to
    # select_transition_tails, which is_transition_bout treats identically to 1
    # (see its own max(1, persist_epochs) clamp), so this keeps the two call
    # sites' effective behavior visibly in sync at the point persist_epochs is
    # computed, not hidden inside a second clamp downstream.
    persist_epochs = max(1, int(round(persist_sec / epoch_sec)))

    try:
        data = load_subject(data_dir, usleep_dir, subject_id, channel, states, band,
                             stage_epochs, epoch_sec, window_sec=window_sec,
                             persist_epochs=persist_epochs, min_bout_sec=min_bout_sec)
    except Exception as exc:  # noqa: BLE001 - see plot_hypnodensity_isfs_phase._process_subject
        return subject_id, False, f"{type(exc).__name__}: {exc}", {}, {}

    rates: Dict[str, np.ndarray] = {}
    curves: Dict[str, np.ndarray] = {}
    for state in states:
        try:
            feats = subject_event_phase_bin_rates(
                data_dir, subject_id, channel, state, stage_epochs, epoch_sec,
                event=event, window_sec=window_sec, persist_epochs=persist_epochs,
                min_bout_sec=min_bout_sec,
            )
        except Exception:  # noqa: BLE001 - same as above, scoped to this metric only
            feats = None
        if feats is None or sum(feats["phase_bin_rates"]) <= 0:
            continue
        rates[state] = np.asarray(feats["phase_bin_rates"])

        try:
            curve = subject_phase_locked_probs_continuous(data, state=state, n_points=n_phase_points)
        except Exception:  # noqa: BLE001 - ditto
            continue
        if not np.isnan(curve).any():
            curves[state] = curve

    return subject_id, True, None, rates, curves


def _categorize(reason: str) -> str:
    if "Hypnodensity file does not exist" in reason:
        return "missing_scored_hypnogram"
    if "No U-Sleep hypnodensity directory" in reason:
        return "missing_usleep_subject_dir"
    if "No U-Sleep hypnodensity file for channel" in reason:
        return "missing_usleep_channel_file"
    if "temporal_ISFS" in reason:
        return "missing_temporal_isfs"
    if "bouts.npz" in reason:
        return "missing_bouts_file"
    if "no usable" in reason:
        return "no_transition_bouts"
    if "EOFError" in reason or "BadZipFile" in reason or "No data left in file" in reason:
        return "corrupt_or_empty_file"
    return "other"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir / args.event
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "group_stats").mkdir(parents=True, exist_ok=True)
    # Namespaced by window_sec/persist_sec so switching either parameter against the
    # same --output-dir can never silently reuse a prior parameter combination's cache.
    cache_dir = args.output_dir / "cache" / f"w{args.window_sec:g}_p{args.persist_sec:g}"
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

    states = tuple(args.states)
    n_subjects = args.n_subjects if args.n_subjects and args.n_subjects > 0 else None
    workers = max(1, args.workers)

    candidates = pio.discover_subjects(args.data_dir)
    if args.limit:
        candidates = candidates[: args.limit]
    logger.info(f"{len(candidates)} candidate subject(s) under {args.data_dir}; "
                f"channel={args.channel} states={states} event={args.event!r} "
                f"window_sec={args.window_sec} persist_sec={args.persist_sec} "
                f"n_subjects={n_subjects or 'unlimited'} workers={workers}")

    included: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
    exclusion_reasons: Dict[str, str] = {}
    n_scanned = 0
    chunk_size = max(workers * 4, workers)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(candidates), chunk_size):
            if n_subjects is not None and len(included) >= n_subjects:
                break
            chunk = candidates[start:start + chunk_size]

            to_process = []
            n_from_cache = 0
            for sid in chunk:
                cached = _read_cache(cache_dir, sid, states)
                if cached is None:
                    to_process.append(sid)
                    continue
                ok, reason, rates, curves = cached
                n_scanned += 1
                n_from_cache += 1
                if ok:
                    included[sid] = (rates, curves)
                else:
                    exclusion_reasons[sid] = reason

            if to_process:
                tasks = [
                    (sid, args.data_dir, args.usleep_dir, args.hypno_dir, args.channel, states,
                     args.event, args.min_bout_sec, args.window_sec, args.persist_sec,
                     args.n_phase_points)
                    for sid in to_process
                ]
                for subject_id, ok, reason, rates, curves in pool.map(_process_subject, tasks):
                    n_scanned += 1
                    _write_cache(cache_dir, subject_id, ok, reason, rates, curves)
                    if ok:
                        included[subject_id] = (rates, curves)
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

    assert subject_ids, (
        "no usable subjects found -- check --data-dir/--usleep-dir/--hypno-dir/--window-sec"
    )

    bin_centers = pph.bin_center_angle(np.arange(1, 9))
    phase_edges = np.linspace(-np.pi, np.pi, args.n_phase_points + 1)
    phase_points = (phase_edges[:-1] + phase_edges[1:]) / 2
    event_label = _EVENT_SPECS[args.event]["event_label"]

    state_summary: Dict[str, Dict[str, int]] = {}
    corr_summaries: Dict[str, pd.DataFrame] = {}
    stage_order = list(USLEEP_STAGE_ORDER)

    for state in states:
        rates_by_sid = {sid: included[sid][0][state] for sid in subject_ids if state in included[sid][0]}
        curves_by_sid = {sid: included[sid][1][state] for sid in subject_ids if state in included[sid][1]}
        n_rates = len(rates_by_sid)
        logger.info(f"{state}: {n_rates}/{len(subject_ids)} subjects have a usable "
                    f"{args.event} transition-tail phase distribution.")
        assert rates_by_sid, f"no subject has a usable {state} {args.event} transition-tail phase distribution"

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
        pdf_path = args.output_dir / f"hypnodensity_isfs_phase_{state}.pdf"
        png_path = args.output_dir / f"hypnodensity_isfs_phase_{state}.png"
        fig.savefig(pdf_path)
        fig.savefig(png_path)
        plt.close(fig)
        logger.info(f"saved {state} figure to {pdf_path} and {png_path}")

    corr_fig = build_correlation_figure(
        corr_summaries, channel=args.channel, event_label=event_label, stage_order=stage_order,
    )
    corr_png_path = args.output_dir / f"{args.event}_stage_phase_corr.png"
    corr_pdf_path = args.output_dir / f"{args.event}_stage_phase_corr.pdf"
    corr_fig.savefig(corr_png_path, dpi=150, bbox_inches="tight")
    corr_fig.savefig(corr_pdf_path, bbox_inches="tight")
    plt.close(corr_fig)
    logger.info(f"saved stage/phase correlation figure to {corr_png_path} and {corr_pdf_path}")

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
    summary_lines.append(
        f"\nEVENT = {args.event!r} ({event_label}, band={_EVENT_SPECS[args.event]['band']!r}), "
        f"WINDOW_SEC = {args.window_sec}, PERSIST_SEC = {args.persist_sec}"
    )

    summary_lines.append(f"\n{args.event} transition-tail phase-bin distribution usable/excluded, out of "
                          f"{len(subject_ids)} included subjects:")
    for state in states:
        n_usable = state_summary[state]["n_rates"]
        summary_lines.append(f"  {state}: {n_usable} usable, {len(subject_ids) - n_usable} excluded")

    summary_lines.append("\nContinuous phase-locked hypnodensity usable/excluded, out of each "
                          "state's transition-tail-usable subjects:")
    for state in states:
        n_attempted = state_summary[state]["n_rates"]
        n_usable = state_summary[state]["n_common"]
        summary_lines.append(f"  {state}: {n_usable} usable, {n_attempted - n_usable} excluded "
                              f"(of {n_attempted} attempted)")

    summary_lines.append("\nSubjects contributing to each state's figure (usable for both the "
                          "event distribution and the continuous curve):")
    for state in states:
        summary_lines.append(f"  {state}: {state_summary[state]['n_common']}/{len(subject_ids)}")

    summary_text = "\n".join(summary_lines) + "\n"
    (args.output_dir / "summary.txt").write_text(summary_text)
    logger.info("\n" + summary_text)


if __name__ == "__main__":
    main()
