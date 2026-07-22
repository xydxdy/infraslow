"""Infraslow oscillation — relative spectral power across channels and stages (results_v3).

Converted from demo_infraslow_yasa_compare.ipynb. Reads the pipeline's own
bout-averaged, baseline-corrected infraslow spectrum (`{channel}__spectra__corr_mean`
vs. `{channel}__spectra__freqs`) directly from
`{RESULTS_DIR}/{STAGE}/{SUBJECT}.npz` for every subject per stage (pass
--n-subjects/N_SUBJECTS to cap the cohort to the first N sorted subjects instead,
useful while iterating). For each stage (N1/N2/N3/NREM) produces one 6-panel figure (one
subplot per channel: F3/F4/C3/C4/O1/O2) with individual-subject lines, the
across-subject average +/- SEM, a bi-Gaussian ISFS fit, and a red dot marking the
channel with the highest fitted peak power. Also computes each channel's spindle
rate (spindles/min, averaged per-subject before averaging across subjects) and a
chromatogram-style peak area.

Every parameter can be set via CLI flag or equivalent env var (CLI wins if both are
given): --show-individual/SHOW_INDIVIDUAL, --n-subjects/N_SUBJECTS,
--results-dir/RESULTS_DIR, --output-dir/OUTPUT_DIR. Run `--help` for details.

Run via Slurm, not the login node, e.g.:
    srun -p normal --time=00:10:00 --mem=4G --cpus-per-task=1 \\
        python3 demo_infraslow_yasa_compare.py --n-subjects 50

Saves one PNG per stage, a summary.csv, and a progress.log (loading progress,
appended across runs) to $SCRATCH/infraslow_outputs/demo_infraslow_yasa_compare/
by default (override with --output-dir/OUTPUT_DIR).
"""

import argparse
import logging
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from infraslow.processing.subject_pipeline import INFRASLOW_BAND, BASELINE_BAND

logger = logging.getLogger(__name__)

plt.rcParams['figure.dpi'] = 110
CHANNELS = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
STAGES = ['N1', 'N2', 'N3', 'NREM']
STAGE_COLORS = {'N1': '#ff7f0e', 'N2': '#5b2a86', 'N3': '#1f77b4', 'NREM': '#2ca02c'}
SUBJ_COLOR = '0.6'
_trapz = getattr(np, 'trapezoid', np.trapz)   # NumPy 2.0 renamed trapz

# UI styling (scaled down from plot_infraslow.ipynb's for this notebook's denser 2x3 grid)
TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 9
TICK_FONTSIZE = 8
LEGEND_FONTSIZE = 7
ANNOTATION_FONTSIZE = 7
SUPTITLE_FONTSIZE = 15

# Opening/decoding a .npz is dominated by Lustre I/O latency, not CPU, and a
# blocking read releases the GIL, so more threads than cores still overlaps more
# latency (see plot_infraslow.ipynb) -- matters once N_SUBJECTS is None and a
# stage has 70k+ subject files.
N_IO_WORKERS = min(32, len(os.sched_getaffinity(0)) * 4)
LOG_EVERY = 1000   # log a progress line every this many npz files loaded, per stage


def _env_flag(name, default):
    """Boolean env var: '1'/'true'/'yes'/'on' (case-insensitive) -> True; unset -> default."""
    val = os.environ.get(name)
    return default if val is None else val.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int_or_none(name, default):
    """Int env var, with '' or 'none' (case-insensitive) meaning None; unset -> default."""
    val = os.environ.get(name)
    if val is None:
        return default
    val = val.strip()
    return None if val == '' or val.lower() == 'none' else int(val)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--show-individual', action='store_true',
                    default=_env_flag('SHOW_INDIVIDUAL', False),
                    help='Plot each subject\'s spectrum behind the average (env: SHOW_INDIVIDUAL)')
    p.add_argument('--n-subjects', type=int, default=_env_int_or_none('N_SUBJECTS', None),
                    help='Cap each stage\'s cohort to the first N sorted subjects; '
                         'omit/None = every subject npz found (env: N_SUBJECTS)')
    p.add_argument('--results-dir', type=Path,
                    default=Path(os.path.expandvars(
                        os.environ.get('RESULTS_DIR', '$SCRATCH/results_v3/npz'))),
                    help='Directory containing {stage}/{subject}.npz files (env: RESULTS_DIR)')
    p.add_argument('--output-dir', type=Path,
                    default=Path(os.path.expandvars(
                        os.environ.get('OUTPUT_DIR', '$SCRATCH/infraslow_outputs/demo_infraslow_yasa_compare'))),
                    help='Where to save PNGs, summary.csv, and progress.log (env: OUTPUT_DIR)')
    return p.parse_args()


def _spindle_rate_per_min(npz, ch):
    """Spindles/min for ONE subject/channel: total `{ch}__bouts__n_spindles` over
    total bout duration (`{ch}__bouts__stop` - `{ch}__bouts__start`, seconds), or NaN
    if there are no bouts for this channel. This collapses all of a subject's bouts
    into a single per-subject rate -- callers must average these per-subject rates
    across subjects (not pool raw per-bout counts across subjects), otherwise
    subjects with more/longer bouts would be over-weighted relative to subjects with
    fewer bouts."""
    nkey, skey, ekey = f'{ch}__bouts__n_spindles', f'{ch}__bouts__start', f'{ch}__bouts__stop'
    if nkey not in npz.files or skey not in npz.files or ekey not in npz.files:
        return np.nan
    n_spindles, start, stop = npz[nkey], npz[skey], npz[ekey]
    if n_spindles.size == 0:
        return np.nan
    total_sec = float((stop - start).sum())
    if total_sec <= 0:
        return np.nan
    return float(n_spindles.sum()) / (total_sec / 60.0)


def load_subject_spectra(npz_path, channels):
    """{channel: (freqs, corr_mean, spindle_rate_per_min)} for channels with a
    non-empty spectrum in this npz."""
    out = {}
    with np.load(npz_path) as npz:
        for ch in channels:
            fkey, ckey = f'{ch}__spectra__freqs', f'{ch}__spectra__corr_mean'
            if fkey not in npz.files or ckey not in npz.files:
                continue
            freqs = npz[fkey]
            if freqs.size == 0:
                continue
            out[ch] = (freqs, npz[ckey], _spindle_rate_per_min(npz, ch))
    return out


def load_stage_cohort(stage, channels, n_subjects, results_dir):
    """by_channel[ch] = {'freqs': shared freq grid, 'subjects': {subject_id: corr_mean},
    'spindle_rate': {subject_id: spindles/min}} for the first n_subjects subject npz
    files (sorted by id) found for this stage, or every file if n_subjects is None.
    Loads run across N_IO_WORKERS threads (I/O-bound, a blocking read releases the
    GIL) -- matters once a stage has 70k+ subject files. by_channel['_n_files'] holds
    the actual file count used (for titles/labels), since it can differ from
    n_subjects and even between stages when n_subjects is None. Logs a progress line
    every LOG_EVERY files loaded, since a full stage can be 70k+ files and take a
    while -- useful for watching a running Slurm job via `tail -f progress.log`."""
    stage_dir = results_dir / stage
    all_paths = sorted(stage_dir.glob('*.npz'))
    subject_paths = all_paths[:n_subjects] if n_subjects else all_paths
    total = len(subject_paths)
    logger.info(f'{stage}: loading {total} of {len(all_paths)} available subject npz files from {stage_dir}')

    by_channel = {ch: {'freqs': None, 'subjects': {}, 'spindle_rate': {}} for ch in channels}
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_IO_WORKERS) as pool:
        loaded = pool.map(lambda p: (p.stem, load_subject_spectra(p, channels)), subject_paths)
        for i, (sid, per_channel) in enumerate(loaded, start=1):
            for ch, (freqs, corr, rate) in per_channel.items():
                entry = by_channel[ch]
                if entry['freqs'] is None:
                    entry['freqs'] = freqs
                elif not np.array_equal(entry['freqs'], freqs):
                    logger.warning(f'skip {stage}/{sid} {ch}: frequency grid differs from the rest of the cohort')
                    continue
                entry['subjects'][sid] = corr
                entry['spindle_rate'][sid] = rate
            if i % LOG_EVERY == 0 or i == total:
                elapsed = time.monotonic() - start
                rate_per_s = i / elapsed if elapsed > 0 else 0.0
                logger.info(f'{stage}: loaded {i}/{total} files ({rate_per_s:.1f} files/s, {elapsed:.0f}s elapsed)')
    by_channel['_n_files'] = total
    logger.info(f'{stage}: done loading in {time.monotonic() - start:.0f}s')
    return by_channel


def bigaussian(f, amp, mu, sd_l, sd_r):
    """Two Gaussian halves sharing one peak but independent left/right widths --
    captures the asymmetric shape (steep rise, slow decay) real infraslow spectra
    show, which a symmetric Gaussian would pull away from the true peak to compromise on."""
    sd = np.where(f < mu, sd_l, sd_r)
    return amp * np.exp(-0.5 * ((f - mu) / sd) ** 2)


def fit_isfs(freqs, corrected, infraslow_band=INFRASLOW_BAND, baseline_band=BASELINE_BAND):
    """Bi-Gaussian ISFS fit (peak, bandwidth, AUC, detection) with `mu` fixed at the
    empirical argmax -- see plot_infraslow.ipynb's fit_isfs for why (too few points
    in the fit window to also let a 4th free parameter float)."""
    base_m = (freqs >= baseline_band[0]) & (freqs <= baseline_band[1])
    fit_m = (freqs >= infraslow_band[0]) & (freqs < baseline_band[0])
    ff, yy = freqs[fit_m], corrected[fit_m]
    mu = float(ff[np.argmax(yy)])

    def _bigaussian_fixed_mu(f, amp, sd_l, sd_r):
        return bigaussian(f, amp, mu, sd_l, sd_r)

    p0 = [max(yy.max(), 1e-9), 0.01, 0.01]
    (amp, sd_l, sd_r), _ = curve_fit(_bigaussian_fixed_mu, ff, yy, p0=p0,
                                     bounds=([0, 1e-3, 1e-3], [np.inf, 0.05, 0.05]),
                                     maxfev=10000)
    popt = (amp, mu, sd_l, sd_r)
    lo, hi = mu - sd_l, mu + sd_r
    bandwidth = hi - lo
    f_auc = np.linspace(lo, hi, 400)
    auc = float(_trapz(bigaussian(f_auc, *popt), f_auc))
    threshold = 1.5 * corrected[base_m].std()
    return dict(popt=popt, amp=amp, mu=mu, sd_l=sd_l, sd_r=sd_r, lo=lo, hi=hi,
                bandwidth=bandwidth, auc=auc, threshold=threshold,
                detected=bool(amp > threshold))


def _threshold_crossing(curve_freqs, curve, start_idx, threshold=0.0):
    """First frequency, at or after `start_idx`, where `curve` drops from >=
    `threshold` to < `threshold`, linearly interpolated between the two
    bracketing samples (see plot_infraslow.ipynb)."""
    seg = curve[start_idx:]
    crossings = np.flatnonzero((seg[:-1] >= threshold) & (seg[1:] < threshold))
    if crossings.size == 0:
        return float(curve_freqs[-1])
    i = start_idx + int(crossings[0])
    f_a, f_b = curve_freqs[i], curve_freqs[i + 1]
    y_a, y_b = curve[i], curve[i + 1]
    return float(f_a + (threshold - y_a) * (f_b - f_a) / (y_b - y_a))


def chromatogram_peak_area(curve_freqs, curve, threshold=0.0, infraslow_band=INFRASLOW_BAND):
    """Chromatogram-style peak area: `curve` integrated above a sloped baseline
    from (infraslow_band[0], curve there) down to where `curve` drops to
    `threshold` (see plot_infraslow.ipynb)."""
    x0 = infraslow_band[0]
    y0 = float(np.interp(x0, curve_freqs, curve))
    peak_idx = int(np.argmax(curve))
    x1 = _threshold_crossing(curve_freqs, curve, peak_idx, threshold)

    peak_m = (curve_freqs >= x0) & (curve_freqs <= x1)
    xf, yf = curve_freqs[peak_m], curve[peak_m]
    incline = threshold + (y0 - threshold) * (x1 - xf) / (x1 - x0)
    above = np.clip(yf - incline, 0, None)
    area = float(_trapz(above, xf))
    return dict(area=area, freqs=xf, curve=yf, incline=incline, x0=x0, y0=y0, x1=x1,
                threshold=threshold)


def plot_channel_subplot(ax, freqs, subj_corr, title, *, color, spindle_rates=None, show_individual=False, show_legend=False):
    """Subject lines behind, average +/- SEM on top, plus a bi-Gaussian ISFS fit
    (on the average curve) with its peak/bandwidth/AUC annotated. Returns
    {'fit': fit_isfs(...), 'peak_area': chromatogram_peak_area(...)['area'], 'n': n,
    'spindle_per_min': mean spindles/min across subjects, 'spindle_per_min_sem': its
    SEM} (or None if there's no data / the fit failed) so the caller can build a
    summary table and compare peak power across channels."""
    if freqs is None or not subj_corr:
        ax.set_title(f'{title} (no data)', fontsize=TITLE_FONTSIZE)
        ax.axis('off')
        return None

    band_m = (freqs >= INFRASLOW_BAND[0]) & (freqs <= INFRASLOW_BAND[1])
    stack = np.vstack(list(subj_corr.values()))
    mean = stack.mean(0)
    n = len(stack)
    sem = stack.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)

    if show_individual:
        for i, corr in enumerate(subj_corr.values()):
            ax.plot(freqs[band_m], corr[band_m], color=SUBJ_COLOR, lw=0.8, alpha=0.5,
                    zorder=1, label='individual subjects' if i == 0 else None)
    ax.fill_between(freqs[band_m], (mean - sem)[band_m], (mean + sem)[band_m],
                     color=color, alpha=0.2, zorder=2, label='SEM')
    ax.axhline(0.0, ls=":", color="0.3", label="Baseline (0)",
    )
    ax.plot(freqs[band_m], mean[band_m], color=color, lw=2.0, zorder=3,
            label=f'Average (n={n})')

    # `spindle_rates` already holds ONE rate per subject (bouts averaged within that
    # subject by `_spindle_rate_per_min`) -- averaging those per-subject rates here
    # (rather than pooling raw per-bout spindle counts across subjects) keeps every
    # subject weighted equally regardless of how many bouts it contributed.
    spindle_vals = np.array(list(spindle_rates.values()), dtype=float) if spindle_rates else np.array([])
    spindle_vals = spindle_vals[np.isfinite(spindle_vals)]
    if spindle_vals.size > 0:
        spindle_per_min = float(spindle_vals.mean())
        spindle_per_min_sem = (float(spindle_vals.std(ddof=1) / np.sqrt(spindle_vals.size))
                                if spindle_vals.size > 1 else 0.0)
    else:
        spindle_per_min = spindle_per_min_sem = np.nan

    try:
        fit = fit_isfs(freqs, mean)
    except RuntimeError:
        fit = None

    result = None
    if fit is not None:
        fg = np.linspace(*INFRASLOW_BAND, 200)
        fitted_curve = bigaussian(fg, *fit['popt'])
        ax.plot(fg, fitted_curve, color='k', lw=1.6, zorder=4, label='bi-Gaussian fit')
        ax.plot([fit['mu']], [fit['amp']], 'o', color='k', ms=5, zorder=5)
        rate_line = (f"{spindle_per_min:.2f}±{spindle_per_min_sem:.2f} spindles/min"
                     if np.isfinite(spindle_per_min) else "")
        ax.text(0.97, 0.75,
                f"peak {fit['mu']:.4f} Hz (~{1 / fit['mu']:.0f} s)\n{rate_line}",
                transform=ax.transAxes, ha='right', va='top', fontsize=ANNOTATION_FONTSIZE,
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
        peak = chromatogram_peak_area(fg, fitted_curve, threshold=fit['threshold'])
        result = dict(fit=fit, peak_area=peak['area'], n=n,
                      spindle_per_min=spindle_per_min, spindle_per_min_sem=spindle_per_min_sem)

    ax.set_xlim(INFRASLOW_BAND)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='semibold', pad=8)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, length=4, width=1)
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.6, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if show_legend:
        ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, borderpad=0.7,
                  labelspacing=0.4, loc='upper right')
    return result


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
    logger.info(f'starting run: show_individual={args.show_individual} n_subjects={args.n_subjects} '
                f'results_dir={args.results_dir} output_dir={args.output_dir}')

    by_stage = {stage: load_stage_cohort(stage, CHANNELS, args.n_subjects, args.results_dir)
                for stage in STAGES}

    all_results = {}
    for stage in STAGES:
        by_channel = by_stage[stage]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
        results = {}
        for i, (ax, ch) in enumerate(zip(axes.flat, CHANNELS)):
            results[ch] = (ax, plot_channel_subplot(
                ax, by_channel[ch]['freqs'], by_channel[ch]['subjects'], ch,
                color=STAGE_COLORS[stage], spindle_rates=by_channel[ch]['spindle_rate'],
                show_individual=args.show_individual,
                show_legend=True))
        all_results[stage] = {ch: r for ch, (ax, r) in results.items()}

        for ax in axes[-1]:
            ax.set_xlabel('Frequency (Hz)', fontsize=LABEL_FONTSIZE, labelpad=6)
        for ax in axes[:, 0]:
            ax.set_ylabel('Baseline-corrected\nrelative power', fontsize=LABEL_FONTSIZE, labelpad=6)

        # Mark the channel with the highest fitted peak power (bi-Gaussian amplitude)
        # with a red dot on top of its (black) peak marker.
        valid_results = {ch: (ax, r) for ch, (ax, r) in results.items() if r is not None}
        if valid_results:
            max_ch, (max_ax, max_r) = max(valid_results.items(), key=lambda kv: kv[1][1]['fit']['amp'])
            max_ax.plot([max_r['fit']['mu']], [max_r['fit']['amp']], 'o', color='red', ms=9, zorder=6,
                        # label='max power channel'
                        )
            max_ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, borderpad=0.7,
                          labelspacing=0.4, loc='upper right')

        fig.suptitle(f'Infraslow oscillation — stage {stage}, {by_channel["_n_files"]} subjects',
                     y=1.02, fontsize=SUPTITLE_FONTSIZE, fontweight='bold')
        fig.tight_layout()

        fig_path = args.output_dir / f'{stage}.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'saved {fig_path}')

    summary = pd.DataFrame([
        {
            'stage': stage,
            'channel': ch,
            'n': r['n'],
            'peak_freq_hz': r['fit']['mu'],
            'peak_period_s': 1 / r['fit']['mu'],
            'bandwidth_hz': r['fit']['bandwidth'],
            'auc_pm1sd': r['fit']['auc'],
            'chromatogram_peak_area': r['peak_area'],
            'spindle_per_min': r['spindle_per_min'],
            'spindle_per_min_SEM': r['spindle_per_min_sem'],
            'detected': r['fit']['detected'],
        }
        for stage, by_ch in all_results.items()
        for ch, r in by_ch.items()
        if r is not None
    ]).set_index(['stage', 'channel'])

    summary_path = args.output_dir / 'summary.csv'
    summary.to_csv(summary_path)
    logger.info(f'saved {summary_path}')
    logger.info(f'\n{summary}')


if __name__ == '__main__':
    main()
