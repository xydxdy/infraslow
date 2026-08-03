"""Per-bout ISFS phase-bin figures for one subject/channel, saved as one multi-page PDF.

Converted from demo_infraslow_phase.ipynb, generalized from one hand-picked bout to
every spindle-containing N2 bout >= --min-bout-sec. For each bout: computes the
sigma-power envelope once (eeg_envelope), low-pass filters it once on the whole
continuous overnight course (isfs_lowpass -- avoids re-tapering/re-padding every
bout's own edges, see isfs_lowpass's docstring), then finds every valid ISFS cycle in
that bout and its 8 phase bins (isfs_phase_bins). Each bout is paginated into
consecutive, non-overlapping --window-sec windows (default 400 s) -- a bout longer
than one window gets multiple pages so the whole bout is shown, not just its start.
Every page shows the sigma-power + filtered course (top) and the standardized course
with peaks/troughs/zero-crossings, every detected cycle's 8 shaded/numbered phase
bins, and explicit grey-hatched "Not ISFS" spans (bottom). A final summary page
(isfs_event_phase_distribution) shows what percentage of this subject/channel's
detected spindles fall in each of the 8 ISFS phase bins, across every bout -- omitted
if fewer than --min-events spindles were detected in total (the reference's own
per-participant inclusion criterion) -- overlaid with a smooth schematic sine curve
(not real amplitude data) marking where bins 1-4/5-8 sit on a phase cycle.

Every parameter can be set via CLI flag or equivalent env var (CLI wins if both are
given): --subject/SUBJECT, --channel/CHANNEL, --sf/SF, --min-bout-sec/MIN_BOUT_SEC,
--window-sec/WINDOW_SEC, --output-dir/OUTPUT_DIR. Run `--help` for details.

Run via Slurm, not the login node, e.g.:
    srun -p normal --time=00:15:00 --mem=4G --cpus-per-task=1 \\
        python3 plot_isfs_phase_bins.py --subject 318679

Saves {SUBJECT}_{CHANNEL}_isfs_phase_bins.pdf and a progress.log to
$SCRATCH/infraslow_outputs/isfs_phase_bins/ by default (override with
--output-dir/OUTPUT_DIR).
"""

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy.signal import find_peaks

from infraslow import BioserenityPSGLoader
from infraslow.processing.spindle import (
    spindles_detect,
    _extract_epoch_stages,
    _stages_to_int,
    DEFAULT_STAGE_MAP,
    DEFAULT_EPOCH_SEC,
)
from infraslow.processing.infraslow import (
    eeg_envelope,
    isfs_lowpass,
    isfs_phase_bins,
    isfs_event_phase_distribution,
    DEFAULT_SF_ENV,
    DEFAULT_ISFS_PERIOD,
    DEFAULT_SIGMA_BAND,
    DEFAULT_ISFS_MIN_EVENTS,
)

logger = logging.getLogger(__name__)

plt.rcParams['figure.dpi'] = 110
LPURPLE, BLUE, GOLD = '#c9b3e6', 'tab:blue', '#f2c14e'

DEFAULT_MIN_BOUT_SEC = 200.0
DEFAULT_WINDOW_SEC = 400.0


def _env_float(name, default):
    """Float env var; unset -> default."""
    val = os.environ.get(name)
    return default if val is None else float(val)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', default=os.environ.get('SUBJECT', '318679'),
                    help='Subject id / EDF stem (env: SUBJECT)')
    p.add_argument('--channel', default=os.environ.get('CHANNEL', 'C3'),
                    help='EEG channel (env: CHANNEL)')
    p.add_argument('--sf', type=float, default=_env_float('SF', 200.0),
                    help='Loader resample rate (Hz) (env: SF)')
    p.add_argument('--min-bout-sec', type=float,
                    default=_env_float('MIN_BOUT_SEC', DEFAULT_MIN_BOUT_SEC),
                    help='Minimum consecutive-N2 bout length (s) (env: MIN_BOUT_SEC)')
    p.add_argument('--window-sec', type=float,
                    default=_env_float('WINDOW_SEC', DEFAULT_WINDOW_SEC),
                    help='Per-page window width (s); bouts longer than this are '
                         'paginated into consecutive windows (env: WINDOW_SEC)')
    p.add_argument('--min-events', type=int,
                    default=int(_env_float('MIN_EVENTS', DEFAULT_ISFS_MIN_EVENTS)),
                    help='Minimum total spindle count required for the final '
                         'phase-distribution summary page (env: MIN_EVENTS)')
    p.add_argument('--output-dir', type=Path,
                    default=Path(os.path.expandvars(
                        os.environ.get('OUTPUT_DIR', '$SCRATCH/infraslow_outputs/isfs_phase_bins'))),
                    help='Where to save the PDF and progress.log (env: OUTPUT_DIR)')
    return p.parse_args()


def nrem2_bouts(loader, *, epoch_sec, min_dur, stage_code=2):
    """(start, stop) times (s) of runs of consecutive N2 epochs lasting >= min_dur."""
    codes = _stages_to_int(
        _extract_epoch_stages(loader.annotations, stage_column='stage'),
        DEFAULT_STAGE_MAP,
    )
    bouts, i, n = [], 0, len(codes)
    while i < n:
        if codes[i] == stage_code:
            j = i
            while j < n and codes[j] == stage_code:
                j += 1
            if (j - i) * epoch_sec >= min_dur:
                bouts.append((i * epoch_sec, j * epoch_sec))
            i = j
        else:
            i += 1
    return bouts


def plot_bout_window(axes, *, tt, raw, filt, std, phase_bins, pk_t, pk_y,
                      w0, w1, bout_idx, window_idx, n_windows):
    """One PDF page: top = sigma power + filtered course + spindles; bottom =
    standardized course + peaks/troughs/zero-crossings + every detected cycle's 8
    phase bins (shaded/numbered) plus explicit "Not ISFS" spans -- all restricted to
    the [w0, w1) window via xlim (the underlying arrays cover the whole bout)."""
    axT, axB = axes

    axT.plot(tt, raw, color='0.6', lw=1.0, label='sigma power')
    axT.plot(pk_t, pk_y, 'o', color=BLUE, ms=6, mec='white', mew=0.8,
              zorder=5, label='detected spindle')
    axT2 = axT.twinx()
    axT2.plot(tt, filt, color='k', lw=1.8, label='filtered (infraslow band)')
    axT2.set_yticks([])
    axT.set(xlim=(w0, w1))
    axT.set_ylabel('Sigma power (dB)', fontsize=13)
    axT.set_title('A) Sigma-power course + infraslow-filtered', fontsize=12, fontweight='bold')
    axT.tick_params(labelsize=12)
    l1, la1 = axT.get_legend_handles_labels()
    l2, la2 = axT2.get_legend_handles_labels()
    axT.legend(l1 + l2, la1 + la2, loc='upper right', frameon=True, framealpha=0.9, fontsize=9)

    pk_i, _ = find_peaks(std)
    tr_i, _ = find_peaks(-std)
    zc_i = np.where(np.diff(np.signbit(std)))[0]

    axB.axhline(0, color='0.7', lw=0.8)
    axB.plot(tt, std, color='k', lw=1.6)
    axB.plot(tt[pk_i], std[pk_i], '^', color='crimson', ms=8, mec='white', mew=0.8,
              zorder=5, label='peak')
    axB.plot(tt[tr_i], std[tr_i], 'v', color='tab:blue', ms=8, mec='white', mew=0.8,
              zorder=5, label='trough')
    axB.plot(tt[zc_i], std[zc_i], 'o', color='0.3', ms=6, mec='white', mew=0.6,
              zorder=5, label='zero-crossing')
    axB.set(xlim=(w0, w1))
    axB.tick_params(labelsize=12)

    run_edges = np.flatnonzero(np.diff(phase_bins) != 0)
    run_starts = np.concatenate([[0], run_edges + 1])
    run_ends = np.concatenate([run_edges + 1, [phase_bins.size]])
    y_top = axB.get_ylim()[1] if axB.get_ylim()[1] > 0 else 1.0
    for s_i, e_i in zip(run_starts, run_ends):
        x0, x1 = tt[s_i], tt[min(e_i, phase_bins.size - 1)]
        if x1 < w0 or x0 > w1:
            continue  # entirely outside this page's window
        label = int(phase_bins[s_i])
        if label == 0:
            axB.axvspan(x0, x1, color='0.85', alpha=0.6, hatch='//', zorder=0)
            continue
        col = LPURPLE if label <= 4 else GOLD
        axB.axvspan(x0, x1, color=col, alpha=0.5, zorder=0)
        label_x = (max(w0, x0) + min(w1, x1)) / 2  # clamp to the visible slice
        axB.text(label_x, y_top * 0.82, str(label), ha='center', va='top',
                  fontsize=11, fontweight='bold', color='0.1', zorder=6)

    legB = axB.legend(handles=axB.get_legend_handles_labels()[0] + [
        Patch(color=LPURPLE, alpha=0.5, label='bins 1-4 (negative half wave)'),
        Patch(color=GOLD, alpha=0.5, label='bins 5-8 (positive half wave)'),
        Patch(facecolor='0.85', alpha=0.6, hatch='//', label='Not ISFS')],
        loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    legB.set_zorder(10)
    axB.set_xlabel('Time within bout (s)', fontsize=13)
    axB.set_ylabel('Standardized power', fontsize=13)
    axB.set_title(f'B) Bout #{bout_idx}, window {window_idx + 1}/{n_windows} '
                  f'({w0:.0f}-{w1:.0f}s)', fontsize=12, fontweight='bold')


def plot_phase_distribution(ax, dist, *, event_label, min_events):
    """Bar chart of an event type's percentage distribution across the 8 ISFS phase
    bins (isfs_event_phase_distribution's output) plus a 9th "Not ISFS" bar right
    after bin 8 (the complement: events that fell outside any valid ISFS cycle, as
    a % of the same total), colored by half-wave, with a smooth schematic sine
    curve underneath the 8 phase bins as a phase reference (not real amplitude data
    -- just where bins 1-4/5-8 sit on a cycle), mirrored left-right from the real
    negative-half-wave-first ISFS shape. Plotted on the *same* %-axis as the bars
    (a shared 0 line) rather than an independent secondary axis, shifted below 0 so
    it reads as a phase strip under the bars instead of competing with them for
    vertical space. ``dist`` may be ``None`` (fewer than ``min_events`` events
    detected), in which case the axes just report that instead of a bar chart."""
    if dist is None:
        ax.axis('off')
        ax.text(0.5, 0.5, f'Excluded: fewer than {min_events} {event_label} detected',
                 ha='center', va='center', fontsize=13, transform=ax.transAxes)
        return

    label_bbox = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8)
    bins = np.arange(1, 9)
    colors = [LPURPLE if b <= 4 else GOLD for b in bins]
    ax.bar(bins, dist['pct'], color=colors, edgecolor='0.3', zorder=2)
    for b, pct in zip(bins, dist['pct']):
        ax.text(b, pct, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10,
                 fontweight='bold', bbox=label_bbox, zorder=6)

    not_isfs_pct = 100.0 * (dist['n_total'] - dist['n_in_isfs']) / dist['n_total']
    ax.bar(9, not_isfs_pct, color='0.85', edgecolor='0.3', hatch='//', zorder=2)
    ax.text(9, not_isfs_pct, f'{not_isfs_pct:.1f}%', ha='center', va='bottom', fontsize=10,
             fontweight='bold', bbox=label_bbox, zorder=6)

    # No tick labels here for bins 1-8 or "Not ISFS" (x=9) -- all placed manually at
    # y=0 below, instead of at the default tick position (which would sit far below
    # the bars once the y-axis is extended downward to fit the phase curve).
    ax.set(xticks=list(bins) + [9], xticklabels=[''] * (len(bins) + 1))
    # ax.set_xlabel('ISFS phase bin', fontsize=11)
    ax.set_ylabel(f'% of {event_label} (of total detected)', fontsize=11)
    ax.set_title(f'Distribution of {event_label} across ISFS phase bins -- '
                 f"{dist['n_in_isfs']}/{dist['n_total']} in a valid ISFS cycle",
                 fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=10)

    # Schematic phase reference, same %-axis as the bars: one smooth sine period
    # across the 8 bins (bin edges 0.5-8.5), amplitude kept small (10% of the bar
    # chart's own range) so it reads as a thin strip near the bottom rather than
    # competing with the bars -- but *not* offset by a baseline, so its first
    # point (x=0.5, the start of bin 1) sits exactly on the shared 0% line.
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    amp = 0.50 * span
    x_smooth = np.linspace(0.5, 8.5, 200)
    y_smooth = -amp * np.sin(2 * np.pi * (x_smooth - 0.5) / 8)
    ax.plot(x_smooth, y_smooth, color='k', lw=2.0, ls='-', zorder=1,
            label='ISFS phase (schematic)')
    ax.axhline(0, color='0.5', lw=1.0, zorder=0)
    ax.set_ylim(min(y0, -amp - 0.03 * span), max(y1, amp + 0.03 * span))

    # Bin-number (1-8) and "Not ISFS" (9) tick labels, placed at the shared y=0
    # line instead of the default tick position (now far below the bars, since the
    # axis extends down to fit the phase curve above).
    for b in bins:
        ax.text(b, 0, str(b), ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(9, 0, 'Not\nISFS', ha='center', va='top', fontsize=10, fontweight='bold')

    # Negative y-tick labels are meaningless (the phase curve is schematic, not
    # real %) -- blank them, keeping 0 and the positive (real, % of events) ticks.
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{t:g}' if t >= 0 else '' for t in yticks])

    # Drop the x-axis frame (bottom spine) -- the axhline(0) above already marks
    # the real zero line -- and clip the left/right spines to start at 0 instead of
    # running down through the (schematic-curve-only) negative region below it.
    _, y1f = ax.get_ylim()
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_bounds(0, y1f)
    ax.spines['right'].set_bounds(0, y1f)
    ax.tick_params(axis='x', bottom=False)

    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
        Patch(color=LPURPLE, label='bins 1-4 (negative half wave)'),
        Patch(color=GOLD, label='bins 5-8 (positive half wave)'),
        Patch(facecolor='0.85', edgecolor='0.3', hatch='//', label='Not ISFS')],
        loc='lower right', frameon=True, framealpha=0.9, fontsize=9)


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
    logger.info(f'starting run: subject={args.subject} channel={args.channel} '
                f'min_bout_sec={args.min_bout_sec} window_sec={args.window_sec} '
                f'output_dir={args.output_dir}')

    loader = BioserenityPSGLoader(
        subject_id=args.subject, sf=args.sf, requested_channels=[args.channel]
    ).load()
    data = np.asarray(loader.get_channel(args.channel), dtype=float)
    sf = float(loader.sf)

    bouts_all = nrem2_bouts(loader, epoch_sec=DEFAULT_EPOCH_SEC, min_dur=args.min_bout_sec)
    sp = spindles_detect(loader, ch_names=args.channel, include=(2,))
    peaks = sp.summary()['Peak'].to_numpy()
    bouts = [(a, b) for a, b in bouts_all if np.any((peaks >= a) & (peaks < b))]
    logger.info(f'{len(bouts)}/{len(bouts_all)} N2 bout(s) >= {args.min_bout_sec:g}s contain >= 1 spindle')

    t_env, sigma_db = eeg_envelope(
        data, sf, band=DEFAULT_SIGMA_BAND, sf_env=DEFAULT_SF_ENV, smooth_sec=1,
        wavelet=True, to_db=True, kind='power',
    )

    def env_at(times):
        return np.interp(times, t_env, sigma_db)

    # Filtered once, on the whole continuous overnight course, before any bout
    # slicing -- see isfs_lowpass's docstring for why (edge effects).
    sigma_lp_full = isfs_lowpass(sigma_db, DEFAULT_SF_ENV)

    pdf_path = args.output_dir / f'{args.subject}_{args.channel}_isfs_phase_bins.pdf'
    n_pages = 0
    bouts_data = []  # (tt, phase_bins, pk_t) per bout, for the final distribution page
    with PdfPages(pdf_path) as pdf:
        for bout_idx, (a, b) in enumerate(bouts):
            m0 = (t_env >= a) & (t_env < b)
            tt = t_env[m0] - a                       # bout-relative time (s)
            raw = sigma_db[m0]
            filt = sigma_lp_full[m0]
            std = (filt - filt.mean()) / filt.std()  # standardized (z-scored)

            phase_bins, isfs_cycles = isfs_phase_bins(tt, std, isfs_period=DEFAULT_ISFS_PERIOD)

            pk_abs = peaks[(peaks >= a) & (peaks < b)]
            pk_t = pk_abs - a
            pk_y = env_at(pk_abs)
            bouts_data.append((tt, phase_bins, pk_t))

            duration = b - a
            n_windows = max(1, math.ceil(duration / args.window_sec))
            for w in range(n_windows):
                w0 = w * args.window_sec
                w1 = min(w0 + args.window_sec, duration)
                fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
                plot_bout_window(
                    axes, tt=tt, raw=raw, filt=filt, std=std, phase_bins=phase_bins,
                    pk_t=pk_t, pk_y=pk_y, w0=w0, w1=w1, bout_idx=bout_idx,
                    window_idx=w, n_windows=n_windows,
                )
                fig.suptitle(f'Subject {args.subject} {args.channel} -- bout #{bout_idx} '
                             f'({a:.0f}-{b:.0f}s, {duration:.0f}s total) -- '
                             f'{len(isfs_cycles)} ISFS cycle(s) in bout',
                             y=1.0, fontsize=11)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                n_pages += 1

            logger.info(f'bout #{bout_idx} ({a:.0f}-{b:.0f}s, {duration:.0f}s): '
                        f'{n_windows} page(s), {len(isfs_cycles)} ISFS cycle(s)')

        # --- Final page: spindle distribution across the 8 ISFS phase bins, across
        # every bout (isfs_event_phase_distribution), plus a schematic phase curve ---
        dist = isfs_event_phase_distribution(bouts_data, min_events=args.min_events)
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_phase_distribution(ax, dist, event_label='spindles', min_events=args.min_events)
        fig.suptitle(f'Subject {args.subject} {args.channel} -- spindle/ISFS phase distribution',
                     y=1.0, fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        n_pages += 1
        if dist is None:
            logger.info(f'distribution summary: excluded (fewer than {args.min_events} spindles detected)')
        else:
            logger.info(f"distribution summary: {dist['n_in_isfs']}/{dist['n_total']} spindles in a valid "
                        f"ISFS cycle; pct per bin = {np.round(dist['pct'], 1).tolist()}")

    logger.info(f'saved {n_pages} page(s) across {len(bouts)} bout(s) to {pdf_path}')


if __name__ == '__main__':
    main()
