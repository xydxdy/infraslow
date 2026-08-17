#!/usr/bin/env python
"""N2 vs N3 infraslow spectrum + spindle-phase comparison across a cohort.

Converted from ``demo_plot_and_compared.ipynb``. Downstream analysis/visualization
over `infraslow.pipeline`'s already-computed per-subject ISFS spectrum and phase
features (see ``src/scripts/preprocessing.py`` for how the underlying `.npz`/`.csv`
artifacts under ``<data-dir>`` were generated). No EEG filtering, spindle/slow-wave
detection, or envelope computation happens here -- every number below comes from
``infraslow.pipeline.pipeline.run_subject_state_channel`` and
``infraslow.stat.state_comparison``, reused unmodified from the rest of the project.

Compares **N2 vs N3** on one channel across every eligible subject in ``<data-dir>``
that has a finite, usable ISFS spectrum AND non-degenerate, finite spindle-phase data
in both states (see ``_is_usable``) -- not just a fixed demo-sized subset. Pass
``--n-subjects`` to cap the count (first N by subject id, as the source notebook did
with 10) or leave it unset to use every eligible subject found.

Saves per-state feature tables, group-comparison result tables, and every figure the
notebook produced to ``<output-dir>`` (see ``main`` for the full file list), plus a
``progress.log``.

Run via Slurm, not the login node, from this file's own directory
(``src/scripts/``) with ``src/`` on ``PYTHONPATH`` so ``infraslow`` resolves
(the package is not pip-installed), e.g.:
    export PYTHONPATH=/home/users/chaisaen/infraslow/src
    srun -p normal --time=00:30:00 --mem=8G --cpus-per-task=4 \\
        python3 plot_n2_vs_n3_compare.py --channel C3 --workers 4

Every parameter can be set via CLI flag or equivalent env var (CLI wins if both are
given): --data-dir/DATA_DIR, --channel/CHANNEL, --n-subjects/N_SUBJECTS,
--candidate-limit/CANDIDATE_LIMIT, --output-dir/OUTPUT_DIR, --verbose/VERBOSE,
--workers/WORKERS. Run `--help` for details.

Subject selection (``select_subjects``) scans the cohort in id order and stops as soon
as ``--n-subjects`` *usable* subjects are found (see ``_is_usable``), instead of
scanning a fixed ``--candidate-limit`` margin up front -- so ``--candidate-limit`` unset
("no cap") stays fast for a small ``--n-subjects`` on a large cohort: it only scans as
far as it actually needs to. ``--candidate-limit`` only matters as a safety cap when
``--n-subjects`` is also unset (nothing to stop early on -- that combination does mean
"scan everything"), or to bound runtime if usable subjects turn out to be rarer than
expected.

``--workers`` parallelizes that scan across subject-level worker *processes*
(``concurrent.futures.ProcessPoolExecutor`` -- each subject/state's
``run_subject_state_channel`` call is independent I/O + curve-fitting work, so this
scales close to linearly with worker count). Match it to ``--cpus-per-task`` in the
Slurm submission; the default (1) is sequential, identical to not using workers at all.
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from infraslow.pipeline import pipeline as ppl
from infraslow.pipeline import phase as pph
from infraslow.processing.infraslow import (
    DEFAULT_INFRASLOW_BAND,
    DEFAULT_BASELINE_BAND,
    bigaussian,
    fit_isfs,
)
from infraslow.stat import state_comparison as stc

logger = logging.getLogger(__name__)

plt.rcParams['figure.dpi'] = 110
PURPLE, LPURPLE = '#5b2a86', '#c9b3e6'
ORANGE, LORANGE = '#d2691e', '#f2c14e'
COLORS = {"N2": (PURPLE, LPURPLE), "N3": (ORANGE, LORANGE)}

STATES = ("N2", "N3")
# "spindle" selects the sigma-band pathway in run_subject_state_channel (band="sigma",
# paired against bouts.npz["spindle"]/spindel_yasa.csv["Peak"]) -- it's also the
# function's own default, but named explicitly here for parity with
# plot_n2_vs_n3_compare_delta.py's EVENT = "sw" (band="delta", slow-wave pathway).
EVENT = "spindle"
REQUIRED_RECORD_KEYS = {
    "subject_id", "sleep_state", "peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc",
    "chromatogram_peak_area", "bi_gaussian_amp", "bi_gaussian_mu", "bi_gaussian_sd_l",
    "bi_gaussian_sd_r", "phase_bin_rates", "mean_phase", "resultant_length",
}
ISFS_METRICS = {
    "peak_freq_hz": "peak_frequency_hz",
    "peak_period_s": "peak_period_s",
    "bi_gaussian_amp": "amplitude",
    "bi_gaussian_sd_l": "sd_left",
    "bi_gaussian_sd_r": "sd_right",
    "bandwidth_hz": "bandwidth_hz",
    "auc": "auc",
    "chromatogram_peak_area": "chromatogram_peak_area",
}


def _env_int(name, default):
    """Int env var; unset -> default."""
    val = os.environ.get(name)
    return default if val is None else int(val)


def _env_bool(name, default):
    """Bool env var ('1'/'true'/'yes'/'on', case-insensitive); unset -> default."""
    val = os.environ.get(name)
    return default if val is None else val.strip().lower() in ('1', 'true', 'yes', 'on')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path,
                    default=Path(os.path.expandvars(os.environ.get(
                        'DATA_DIR', '/scratch/users/chaisaen/processed_data/data'))),
                    help='Root of preprocessing.py\'s per-subject output (env: DATA_DIR)')
    p.add_argument('--channel', default=os.environ.get('CHANNEL', 'C3'),
                    help='EEG channel (env: CHANNEL)')
    p.add_argument('--n-subjects', type=int,
                    default=_env_int('N_SUBJECTS', 0) or None,
                    help='Cap on eligible subjects used, first N by subject id '
                         '(env: N_SUBJECTS); unset/0 = every eligible subject found')
    p.add_argument('--candidate-limit', type=int,
                    default=_env_int('CANDIDATE_LIMIT', 0) or None,
                    help='Cap on candidate subjects scanned before the usability filter '
                         '(env: CANDIDATE_LIMIT); unset/0 = scan the whole cohort')
    p.add_argument('--output-dir', type=Path,
                    default=Path(os.path.expandvars(
                        os.environ.get('OUTPUT_DIR', '/scratch/users/chaisaen/outputs/n2_vs_n3_compare'))),
                    help='Where to save figures, result tables, and progress.log (env: OUTPUT_DIR)')
    p.add_argument('--verbose', action='store_true', default=_env_bool('VERBOSE', True),
                    help='Log one line per subject scanned during selection (env: VERBOSE)')
    p.add_argument('--workers', type=int, default=_env_int('WORKERS', 1),
                    help='Parallel worker processes for subject scanning (env: WORKERS); '
                         'match --cpus-per-task in the Slurm submission; default 1 = sequential')
    return p.parse_args()


def _is_usable(entry, states=STATES):
    """A non-`None` spectrum record from `find_paired_subject_data` is not sufficient on
    its own: `run_subject_state_channel` can still return an all-NaN spectrum record (zero
    ISFS rows matched -- `infraslow.pipeline.spectrum._nan_spectrum_features`), or a
    record with all-zero `phase_bin_rates` and NaN `mean_phase` (zero in-cycle-ISFS events
    despite >= 5 total detected spindles -- `phase_bin_rates` is `[0.0]*8`, NOT `None`, in
    that case, so it would silently pass a `phase_bin_rates is not None`-only check).
    Requires a finite spectrum plus finite, non-degenerate phase data in every state."""
    for state in states:
        record = entry[state][0]
        phase_bin_rates = record.get("phase_bin_rates")
        if phase_bin_rates is None:
            return False
        if not np.isfinite(record["peak_freq_hz"]):
            return False
        if sum(phase_bin_rates) <= 0:
            return False
        if not np.isfinite(record["mean_phase"]):
            return False
    return True


def _process_subject_state_pair(data_dir, channel, subject_id):
    """`(subject_id, per_state)` -- both states' `run_subject_state_channel(..., event=EVENT)`
    result for one subject; `per_state` is `None` if either state failed to produce a record.
    Module-level (not a closure) so it's picklable for `ProcessPoolExecutor`."""
    per_state = {}
    for state in STATES:
        record, bout_records, _failure, curves = ppl.run_subject_state_channel(
            data_dir, subject_id, channel, state, event=EVENT, return_curves=True,
        )
        if record is None:
            return subject_id, None
        per_state[state] = (record, bout_records, curves)
    return subject_id, per_state


def select_subjects(data_dir, channel, *, n_subjects=None, candidate_limit=None, verbose=False, workers=1):
    """`(paired_data, n_candidates)` -- every eligible subject (see `_is_usable`), sorted
    by id, capped at `n_subjects` if given.

    Scans subjects in batches of `workers` at a time -- one subject per worker process
    via `ProcessPoolExecutor` when `workers > 1` (each subject/state's
    `run_subject_state_channel` call is independent I/O + curve-fitting work, so this
    parallelizes cleanly), or sequentially in-process when `workers <= 1` (no pool
    overhead, identical behavior to before `--workers` existed). `Executor.map` returns
    results in input order regardless of which worker finishes first, so `workers` has
    no effect on subject order/"first N by id" semantics.

    Stops as soon as `n_subjects` usable subjects are found -- so `candidate_limit=None`
    ("no cap") stays fast for a small `n_subjects` on a large cohort: it only scans as
    far as it actually needs to. `candidate_limit` bounds the number of *paired*
    candidates accumulated (matches `find_paired_subject_data`'s own `limit` semantics),
    relevant when `n_subjects` is also `None` (nothing to stop early on)."""
    candidates = {}
    n_usable = 0
    n_scanned = 0
    subject_iter = iter(ppl.pio.discover_subjects(data_dir))
    batch_size = max(workers, 1)

    def _take_batch():
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(subject_iter))
            except StopIteration:
                break
        return batch

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while True:
            if candidate_limit is not None and len(candidates) >= candidate_limit:
                break
            batch = _take_batch()
            if not batch:
                break
            n_scanned += len(batch)

            if executor is not None:
                results = list(executor.map(
                    _process_subject_state_pair, [data_dir] * len(batch), [channel] * len(batch), batch,
                ))
            else:
                results = [_process_subject_state_pair(data_dir, channel, sid) for sid in batch]

            stop = False
            for subject_id, per_state in results:
                if candidate_limit is not None and len(candidates) >= candidate_limit:
                    break
                if per_state is None:
                    if verbose:
                        logger.info(f"[{n_scanned}] {subject_id}: not paired (missing/failed a state) "
                                    f"-- {len(candidates)} paired, {n_usable} usable so far")
                    continue
                candidates[subject_id] = per_state
                usable = _is_usable(per_state)
                if usable:
                    n_usable += 1
                if verbose:
                    target = f"/{n_subjects}" if n_subjects is not None else ""
                    logger.info(f"[{n_scanned}] {subject_id}: paired, {'usable' if usable else 'not usable'} "
                                f"-- {len(candidates)} paired, {n_usable}{target} usable so far")
                if usable and n_subjects is not None and n_usable >= n_subjects:
                    stop = True
                    break
            if stop:
                break
    finally:
        if executor is not None:
            executor.shutdown()

    phase_complete = {sid: data for sid, data in candidates.items() if _is_usable(data)}
    paired_data = dict(list(phase_complete.items())[:n_subjects])
    return paired_data, len(candidates)


def build_feature_tables(paired_data, subject_ids):
    """Validates every selected subject/state's record/curves and returns `(n2_df, n3_df)`."""
    for sid in subject_ids:
        for state in STATES:
            record, _bout_records, curves = paired_data[sid][state]
            missing = REQUIRED_RECORD_KEYS - record.keys()
            assert not missing, f"{sid}/{state} missing keys: {missing}"
            assert curves["freqs"].shape == curves["rel"].shape == curves["corrected"].shape
            assert np.isfinite(curves["rel"]).all(), f"{sid}/{state}: non-finite curves['rel']"
            assert np.isfinite(curves["corrected"]).all(), f"{sid}/{state}: non-finite curves['corrected']"
            assert len(record["phase_bin_rates"]) == 8

    n2_df = pd.DataFrame([paired_data[sid]["N2"][0] for sid in subject_ids])
    n3_df = pd.DataFrame([paired_data[sid]["N3"][0] for sid in subject_ids])
    return n2_df, n3_df


def plot_spectrum_figure(freqs, band_m, rel_mean, corrected_mean, group_fit, *, channel, n_subjects):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

    for state in STATES:
        dark, light = COLORS[state]
        mean_rel, sem_rel = rel_mean[state]
        axL.fill_between(freqs[band_m], (mean_rel - sem_rel)[band_m], (mean_rel + sem_rel)[band_m],
                          color=light, alpha=0.5)
        axL.plot(freqs[band_m], mean_rel[band_m], color=dark, lw=2, label=f"{state} mean +/- SEM")
    axL.set(xlim=DEFAULT_INFRASLOW_BAND, xlabel="Frequency (Hz)", ylabel="Relative spectral power",
            title="A) Average relative spectral power")
    axL.legend(frameon=True, fontsize=9)

    fg = np.linspace(*DEFAULT_INFRASLOW_BAND, 400)
    for state in STATES:
        dark, light = COLORS[state]
        mean_corr, sem_corr = corrected_mean[state]
        fit = group_fit[state]
        axR.fill_between(freqs[band_m], (mean_corr - sem_corr)[band_m], (mean_corr + sem_corr)[band_m],
                          color=light, alpha=0.5)
        axR.plot(freqs[band_m], mean_corr[band_m], color=dark, lw=1.5, label=f"{state} baseline-corrected")
        axR.plot(fg, bigaussian(fg, *fit["popt"]), color=dark, lw=2.5, ls="--",
                  label=f"{state} bi-Gaussian fit (peak={fit['mu']:.4f} Hz)")
        axR.axvline(fit["mu"], color=dark, lw=1, ls=":")
    axR.set(xlim=DEFAULT_INFRASLOW_BAND, xlabel="Frequency (Hz)", ylabel="Relative spectral power",
            title="B) Baseline-corrected spectrum + bi-Gaussian fit")
    axR.legend(frameon=True, fontsize=8)

    fig.suptitle(f"N2 vs N3 average ISFS spectrum ({channel}, n={n_subjects} subjects)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_phase_distribution_figure(bin_centers, phase_mean, *, channel, n_subjects):
    fig, ax = plt.subplots(figsize=(8, 5))
    for state in STATES:
        dark, light = COLORS[state]
        mean_r, sem_r = phase_mean[state]
        ax.errorbar(bin_centers, mean_r, yerr=sem_r, color=dark, marker="o", ms=6,
                    capsize=3, lw=1.8, label=f"{state} mean +/- SEM")
    ax.axvline(0, color="0.7", lw=0.8, ls="--")
    ax.set(xlabel="ISFS phase (rad)", ylabel="% of spindles (of total detected)",
           xticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
           xticklabels=["-pi", "-pi/2", "0", "pi/2", "pi"],
           title=f"N2 vs N3 spindle distribution across ISFS phase ({channel}, n={n_subjects} subjects)")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    return fig


def plot_peak_freq_violin_figure(peak_freqs):
    fig, ax = plt.subplots(figsize=(6, 5))
    positions = [1, 2]
    parts = ax.violinplot([peak_freqs["N2"], peak_freqs["N3"]], positions=positions,
                           showmeans=False, showextrema=False)
    for pc, state in zip(parts["bodies"], STATES):
        pc.set_facecolor(COLORS[state][1])
        pc.set_edgecolor(COLORS[state][0])
        pc.set_alpha(0.6)

    rng = np.random.default_rng(0)
    for pos, state in zip(positions, STATES):
        y = peak_freqs[state]
        x = rng.normal(pos, 0.04, size=y.size)
        ax.plot(x, y, "o", color=COLORS[state][0], ms=6, mec="white", mew=0.6, zorder=3)

    ax.set(xticks=positions, xticklabels=STATES, ylabel="Peak frequency (Hz)",
           title="Peak-frequency distribution: N2 vs N3")
    fig.tight_layout()
    return fig


def plot_peak_freq_paired_figure(peak_freqs, *, n_subjects):
    fig, ax = plt.subplots(figsize=(5, 5))
    for n2_v, n3_v in zip(peak_freqs["N2"], peak_freqs["N3"]):
        ax.plot([1, 2], [n2_v, n3_v], color="0.6", lw=1.0, zorder=1)
    ax.plot(np.ones_like(peak_freqs["N2"]), peak_freqs["N2"], "o", color=COLORS["N2"][0], ms=8,
            zorder=3, label="N2")
    ax.plot(np.full_like(peak_freqs["N3"], 2), peak_freqs["N3"], "o", color=COLORS["N3"][0], ms=8,
            zorder=3, label="N3")
    ax.set(xlim=(0.7, 2.3), xticks=[1, 2], xticklabels=STATES, ylabel="Peak frequency (Hz)",
           title=f"Paired N2 vs N3 peak frequency (n={n_subjects} subjects)")
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.output_dir / 'progress.log', mode='a'),
            logging.StreamHandler(),
        ],
    )
    logger.info(f'starting run: event={EVENT} data_dir={args.data_dir} channel={args.channel} '
                f'n_subjects={args.n_subjects} candidate_limit={args.candidate_limit} '
                f'verbose={args.verbose} workers={args.workers} output_dir={args.output_dir}')

    # --- 2. Find eligible subjects ---
    paired_data, n_candidates = select_subjects(
        args.data_dir, args.channel, n_subjects=args.n_subjects, candidate_limit=args.candidate_limit,
        verbose=args.verbose, workers=args.workers,
    )
    subject_ids = list(paired_data)
    if args.n_subjects is not None:
        assert len(subject_ids) == args.n_subjects, (
            f"expected {args.n_subjects} paired subjects with usable, finite spectrum AND phase "
            f"data, found {len(subject_ids)} among {n_candidates} spectrum-paired candidates "
            f"(try raising --candidate-limit)"
        )
    n_subjects = len(subject_ids)
    logger.info(f'selected {n_subjects} subjects with usable, finite {"/".join(STATES)} spectrum + '
                f'phase data on {args.channel} (out of {n_candidates} spectrum-paired candidates checked)')
    (args.output_dir / 'selected_subjects.txt').write_text('\n'.join(subject_ids) + '\n')

    # --- 3. Load subject data / build feature tables ---
    n2_df, n3_df = build_feature_tables(paired_data, subject_ids)
    n2_df.to_csv(args.output_dir / 'n2_features.csv', index=False)
    n3_df.to_csv(args.output_dir / 'n3_features.csv', index=False)
    logger.info(f'n2_df: {n2_df.shape}, n3_df: {n3_df.shape}')

    # --- 4. N2 vs N3 ISFS spectral analysis ---
    freqs = paired_data[subject_ids[0]]["N2"][2]["freqs"]
    for sid in subject_ids:
        for state in STATES:
            assert np.allclose(paired_data[sid][state][2]["freqs"], freqs)

    rel_by_state = {
        state: np.stack([paired_data[sid][state][2]["rel"] for sid in subject_ids])
        for state in STATES
    }
    corrected_by_state = {
        state: np.stack([paired_data[sid][state][2]["corrected"] for sid in subject_ids])
        for state in STATES
    }
    band_m = (freqs >= DEFAULT_INFRASLOW_BAND[0]) & (freqs <= DEFAULT_INFRASLOW_BAND[1])

    rel_mean = {state: stc.mean_sem(rel_by_state[state]) for state in STATES}
    corrected_mean = {state: stc.mean_sem(corrected_by_state[state]) for state in STATES}

    group_fit = {}
    for state in STATES:
        mean_corrected = corrected_mean[state][0]
        group_fit[state] = fit_isfs(
            freqs, mean_corrected, infraslow_band=DEFAULT_INFRASLOW_BAND, baseline_band=DEFAULT_BASELINE_BAND,
        )
        fit = group_fit[state]
        logger.info(f"{state} group fit: peak={fit['mu']:.4f} Hz (~{1/fit['mu']:.0f} s) | "
                    f"bandwidth={fit['bandwidth']:.4f} Hz | detected={fit['detected']}")

    fig = plot_spectrum_figure(freqs, band_m, rel_mean, corrected_mean, group_fit,
                                channel=args.channel, n_subjects=n_subjects)
    fig.savefig(args.output_dir / 'fig_isfs_spectrum.png', bbox_inches='tight')
    plt.close(fig)

    # --- 5. N2 vs N3 spindle phase analysis ---
    phase_rates = {
        state: np.stack([paired_data[sid][state][0]["phase_bin_rates"] for sid in subject_ids])
        for state in STATES
    }
    bin_centers = pph.bin_center_angle(np.arange(1, 9))
    phase_mean = {state: stc.mean_sem(phase_rates[state]) for state in STATES}

    fig = plot_phase_distribution_figure(bin_centers, phase_mean, channel=args.channel, n_subjects=n_subjects)
    fig.savefig(args.output_dir / 'fig_phase_distribution.png', bbox_inches='tight')
    plt.close(fig)

    # --- 6. Peak-frequency distribution ---
    peak_freqs = {state: (n2_df if state == "N2" else n3_df)["peak_freq_hz"].to_numpy() for state in STATES}

    fig = plot_peak_freq_violin_figure(peak_freqs)
    fig.savefig(args.output_dir / 'fig_peak_freq_violin.png', bbox_inches='tight')
    plt.close(fig)

    fig = plot_peak_freq_paired_figure(peak_freqs, n_subjects=n_subjects)
    fig.savefig(args.output_dir / 'fig_peak_freq_paired.png', bbox_inches='tight')
    plt.close(fig)

    peak_freq_summary = pd.DataFrame({state: stc.describe(peak_freqs[state]) for state in STATES}).T
    peak_freq_summary.to_csv(args.output_dir / 'peak_freq_summary.csv')

    # --- 7. Statistical comparison ---
    peak_ttest = stc.paired_ttest(n2_df["peak_freq_hz"].to_numpy(), n3_df["peak_freq_hz"].to_numpy())
    peak_dz = stc.paired_effect_size(n2_df["peak_freq_hz"].to_numpy(), n3_df["peak_freq_hz"].to_numpy())
    logger.info(f"n={peak_ttest['n_pairs']} | N2 {peak_ttest['x_mean']:.4f}+/-{peak_ttest['x_sd']:.4f} Hz | "
                f"N3 {peak_ttest['y_mean']:.4f}+/-{peak_ttest['y_sd']:.4f} Hz | "
                f"mean diff={peak_ttest['mean_diff']:.4f} Hz | t={peak_ttest['t_stat']:.3f} | "
                f"p={peak_ttest['p_value']:.4g} | Cohen's dz={peak_dz:.3f}")

    phase_bin_results = stc.compare_phase_bins(phase_rates["N2"], phase_rates["N3"], bin_centers)
    phase_bin_results.to_csv(args.output_dir / 'phase_bin_results.csv', index=False)

    isfs_metric_results = stc.compare_isfs_metrics(n2_df, n3_df, ISFS_METRICS)
    isfs_metric_results.to_csv(args.output_dir / 'isfs_metric_results.csv', index=False)

    n2_valid_phase = int(n2_df["mean_phase"].notna().sum())
    n3_valid_phase = int(n3_df["mean_phase"].notna().sum())
    logger.info(f"Valid (non-NaN) mean_phase: N2 {n2_valid_phase}/{len(n2_df)} subjects, "
                f"N3 {n3_valid_phase}/{len(n3_df)} subjects")

    n2_circ = stc.circular_mean_and_resultant(n2_df["mean_phase"].to_numpy())
    n3_circ = stc.circular_mean_and_resultant(n3_df["mean_phase"].to_numpy())
    logger.info(f"N2 circular mean phase = {n2_circ[0]:.3f} rad, resultant length = {n2_circ[1]:.3f}")
    logger.info(f"N3 circular mean phase = {n3_circ[0]:.3f} rad, resultant length = {n3_circ[1]:.3f}")

    # --- 8. Summary ---
    n_sig_bins = int(phase_bin_results["significant_FDR"].sum())
    n_sig_metrics = int((isfs_metric_results["q"] < 0.05).sum())

    summary_lines = [
        "Summary",
        "=======",
        f"Subjects: {n_subjects} (channel {args.channel}), paired N2/N3 usable data",
        f"Peak frequency: N2 {peak_ttest['x_mean']:.4f} Hz vs N3 {peak_ttest['y_mean']:.4f} Hz, "
        f"t({peak_ttest['n_pairs'] - 1})={peak_ttest['t_stat']:.3f}, p={peak_ttest['p_value']:.4g}, "
        f"Cohen's dz={peak_dz:.3f}",
        f"Phase bins significant after FDR (q<0.05): {n_sig_bins}/8",
        f"ISFS metrics significant after FDR (q<0.05): {n_sig_metrics}/{len(isfs_metric_results)}",
        f"N2 circular mean phase: {n2_circ[0]:.3f} rad (resultant {n2_circ[1]:.3f})",
        f"N3 circular mean phase: {n3_circ[0]:.3f} rad (resultant {n3_circ[1]:.3f})",
    ]
    summary_text = '\n'.join(summary_lines)
    (args.output_dir / 'summary.txt').write_text(summary_text + '\n')
    logger.info('\n' + summary_text)
    logger.info(f'done: results + figures saved to {args.output_dir}')


if __name__ == '__main__':
    main()
