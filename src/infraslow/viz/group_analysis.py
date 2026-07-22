"""N2-C3 spindle-rate group-comparison plots (``src/group_analysis.py``).

Pure plotting: every number these functions draw (fits, chromatogram peak
areas, cutoff thresholds, q-values, ...) is computed by the caller with the
same scientific functions ``demo_infraslow_yasa_compare.py`` uses
(``bigaussian``, ``fit_isfs``, ``chromatogram_peak_area``) -- this module only
owns the visual layout, so the plotting logic is not duplicated between the
per-stage/per-channel demo figure and this group comparison (md/group_analysis.md
Step 6, "Do not duplicate plotting logic in multiple files").

matplotlib/scipy are imported lazily inside each function, matching the rest of
:mod:`infraslow.viz`, so importing this module has no plotting-library cost
until a figure is actually drawn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "LOW_COLOR",
    "HIGH_COLOR",
    "MID_COLOR",
    "ALL_COLOR",
    "SUBJECT_COLOR",
    "plot_spindle_rate_pretransform",
    "plot_spindle_rate_distribution",
    "plot_group_infraslow_compare",
    "plot_group_spectrum_clean",
    "plot_parameter_comparisons",
    "plot_cohort_infraslow_compare",
    "plot_cohort_spectrum_clean",
    "plot_parameter_distributions",
]

LOW_COLOR = "#1f77b4"
HIGH_COLOR = "#d62728"
MID_COLOR = "#7f7f7f"
#: Whole-cohort ("before" any low/high split) curve/violin color.
ALL_COLOR = "#2ca02c"
#: Individual-subject line color -- matches infraslow_yasa_compare.py's SUBJ_COLOR.
SUBJECT_COLOR = "0.6"

TITLE_FONTSIZE = 12
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 8
ANNOTATION_FONTSIZE = 8
SUPTITLE_FONTSIZE = 15


def _save(fig, output_png: Path, output_pdf: Optional[Path] = None) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")


def plot_spindle_rate_pretransform(
    *,
    spindle_rate: np.ndarray,
    output_png: Path,
    rate_unit: str = "min",
) -> None:
    """Raw- vs. log1p-scale spindle-rate histogram, drawn *before* grouping.

    :func:`~infraslow.stats.group_assignment.assign_spindle_rate_groups` groups
    subjects from ``log1p(spindle_rate)``'s mean +/- std -- this figure shows
    both candidate scales side by side (no group coloring, since no cutoff has
    been computed yet), each with a single-Gaussian reference fit overlaid as
    a dotted line, so how far each scale departs from Gaussian can be
    eyeballed before the cutoff is computed (md/group_analysis.md Step 4).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    values = np.asarray(spindle_rate, dtype=float)
    values = values[np.isfinite(values)]
    log_values = np.log1p(values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, xlabel, title in (
        (axes[0], values, f"spindle_per_{rate_unit}", "Raw scale"),
        (axes[1], log_values, f"log1p(spindle_per_{rate_unit})", "log1p scale"),
    ):
        ax.hist(data, bins=40, color="0.35", alpha=0.75, density=True)
        if data.size > 1 and data.std() > 0:
            mu, sigma = sp_stats.norm.fit(data)
            grid = np.linspace(data.min(), data.max(), 200)
            ax.plot(grid, sp_stats.norm.pdf(grid, loc=mu, scale=sigma), color="0.1", lw=1.6, ls=":",
                    label=f"Gaussian fit (μ={mu:.2f}, σ={sigma:.2f})")
            ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, loc="upper right")
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="semibold")
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"N2-C3 spindle-rate distribution (n={values.size}) — pre-grouping",
        y=1.02, fontsize=SUPTITLE_FONTSIZE, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_png)
    plt.close(fig)


def plot_spindle_rate_distribution(
    *,
    spindle_rate: np.ndarray,
    spindle_group: np.ndarray,
    low_threshold_original_scale: float,
    high_threshold_original_scale: float,
    low_label: str,
    high_label: str,
    mid_label: str,
    output_png: Path,
    rate_unit: str = "min",
) -> None:
    """N2-C3 spindle-rate histogram split by the log1p mean +/- std cutoff groups.

    Shows the distribution split into low/mid/high group membership
    (:func:`~infraslow.stats.group_assignment.assign_spindle_rate_groups`) plus
    the two cutoff thresholds -- mapped back from log1p scale to the original
    spindle-rate scale via ``expm1`` -- as vertical dotted lines
    (md/group_analysis.md Step 4). ``rate_unit`` (e.g. ``"min"`` or ``"hr"``)
    only affects axis/label text -- every value here must already be expressed
    in that unit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(spindle_rate, dtype=float)
    spindle_group = np.asarray(spindle_group)
    n_low = int((spindle_group == low_label).sum())
    n_mid = int((spindle_group == mid_label).sum())
    n_high = int((spindle_group == high_label).sum())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(values[spindle_group == low_label], bins=40, color=LOW_COLOR, alpha=0.5,
            density=True, label=f"{low_label} (n={n_low})")
    ax.hist(values[spindle_group == mid_label], bins=40, color=MID_COLOR, alpha=0.4,
            density=True, label=f"{mid_label} (n={n_mid})")
    ax.hist(values[spindle_group == high_label], bins=40, color=HIGH_COLOR, alpha=0.5,
            density=True, label=f"{high_label} (n={n_high})")

    ax.axvline(low_threshold_original_scale, color=LOW_COLOR, lw=1.6, ls=":",
               label=f"low cutoff = {low_threshold_original_scale:.2f}/{rate_unit}")
    ax.axvline(high_threshold_original_scale, color=HIGH_COLOR, lw=1.6, ls=":",
               label=f"high cutoff = {high_threshold_original_scale:.2f}/{rate_unit}")

    ax.set_xlabel(f"N2-C3 spindle_per_{rate_unit}", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        "N2-C3 spindle-rate grouping (log1p mean ± std cutoff)",
        fontsize=TITLE_FONTSIZE, fontweight="semibold",
    )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, loc="upper right")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_png)
    plt.close(fig)


def _draw_one_group_curve(
    ax, freqs: np.ndarray, mean: np.ndarray, sem: np.ndarray, *,
    fit: Optional[Dict[str, float]], fitted_curve: Optional[np.ndarray], fitted_freqs: Optional[np.ndarray],
    color: str, label: str, n: int, rate: float, rate_sem: float, rate_unit: str,
    infraslow_band: Tuple[float, float], annotate_xy: Tuple[float, float],
    individual_curves: Optional[np.ndarray] = None, individual_label: Optional[str] = None,
) -> None:
    """Draw one group's individual-subject lines, SEM band, mean curve, bi-Gaussian
    fit, and peak/spindle annotation onto `ax`.

    Mirrors ``demo_infraslow_yasa_compare.plot_channel_subplot``'s visual
    components (individual-subject lines, SEM band, average line, bi-Gaussian
    fit, peak marker, spindle annotation) for one group instead of one
    channel, so both groups can be drawn on the same shared axes.

    ``individual_curves`` is an optional ``(n, len(freqs))`` array (one row
    per subject, see ``group_analysis.py``'s ``_build_group_plot_data``); each
    row is drawn as a thin, low-alpha :data:`SUBJECT_COLOR` (grey) line behind
    the SEM band -- a neutral background texture that doesn't compete with
    either group's own color. ``individual_label`` legends only one such line
    (pass it for at most one of the two groups sharing this axes, to avoid a
    duplicate legend entry).
    """
    band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
    # SEM band goes down first (zorder=1) so the individual lines (zorder=2) sit on
    # top of it instead of being washed out underneath a semi-transparent fill.
    ax.fill_between(freqs[band_m], (mean - sem)[band_m], (mean + sem)[band_m],
                     color=color, alpha=0.15, zorder=1)
    if individual_curves is not None and individual_curves.size:
        for i, curve in enumerate(individual_curves):
            ax.plot(freqs[band_m], curve[band_m], color=SUBJECT_COLOR, lw=1.0, alpha=0.55,
                     zorder=2, label=individual_label if i == 0 else None)
    ax.plot(freqs[band_m], mean[band_m], color=color, lw=2.2, zorder=3, label=f"{label} (n={n})")

    if fit is None or fitted_curve is None or fitted_freqs is None:
        return
    ax.axvspan(fit["lo"], fit["hi"], color=color, alpha=0.08, zorder=0)
    ax.plot(fitted_freqs, fitted_curve, color=color, lw=1.4, ls="--", zorder=4,
            label=f"{label} bi-Gaussian fit")
    ax.plot([fit["mu"]], [fit["amp"]], "o", color=color, ms=6, zorder=5, mec="k", mew=0.6)

    rate_line = (f"{rate:.2f}±{rate_sem:.2f} spindles/{rate_unit}"
                 if np.isfinite(rate) else "")
    ax.text(
        *annotate_xy,
        f"{label}\npeak {fit['mu']:.4f} Hz (~{1 / fit['mu']:.0f} s)\n"
        f"bandwidth {fit['bandwidth']:.4f} Hz\nAUC {fit['auc']:.4g}\n{rate_line}",
        transform=ax.transAxes, ha="right", va="top", fontsize=ANNOTATION_FONTSIZE, color=color,
        bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.85),
    )


def plot_group_infraslow_compare(
    *,
    low: Dict[str, object],
    high: Dict[str, object],
    infraslow_band: Tuple[float, float],
    sleep_stage: str,
    channel: str,
    output_png: Path,
    output_pdf: Path,
    rate_unit: str = "min",
) -> None:
    """The main N2-C3 low- vs. high-spindle-rate comparison figure.

    Reproduces every visually-relevant component of
    ``demo_infraslow_yasa_compare.plot_channel_subplot`` (individual-subject
    lines, baseline-at-0 line, per-group SEM band + mean curve, bi-Gaussian
    fit, peak-frequency marker, shaded bandwidth span, AUC/spindle-rate
    annotation), adapted to compare two groups on one shared axes instead of
    one channel per subplot (md/group_analysis.md Step 6).

    Args:
        low, high: Dicts with keys ``freqs, mean, sem, n, fit, fitted_curve,
            fitted_freqs, individual_curves, rate, rate_sem`` -- see
            ``group_analysis.py``'s ``_build_group_plot_data``. ``rate``/
            ``rate_sem`` must already be expressed in ``rate_unit`` (only used
            here for the annotation text).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0.0, ls=":", color="0.3", label="Baseline (0)")

    # Draw high before low so the legend lists Baseline/individual subjects,
    # then every high_spindle_rate entry, then every low_spindle_rate entry,
    # in that order. The legend itself is anchored just outside the axes
    # (top-right of the figure) so it never overlaps the curves/annotations,
    # which stay stacked inside the axes' own top-right corner.
    _draw_one_group_curve(
        ax, high["freqs"], high["mean"], high["sem"], fit=high["fit"],
        fitted_curve=high["fitted_curve"], fitted_freqs=high["fitted_freqs"],
        color=HIGH_COLOR, label="high_spindle_rate", n=high["n"],
        rate=high["rate"], rate_sem=high["rate_sem"], rate_unit=rate_unit,
        individual_curves=high.get("individual_curves"), individual_label="individual subjects",
        infraslow_band=infraslow_band, annotate_xy=(0.97, 0.97),
    )
    _draw_one_group_curve(
        ax, low["freqs"], low["mean"], low["sem"], fit=low["fit"],
        fitted_curve=low["fitted_curve"], fitted_freqs=low["fitted_freqs"],
        color=LOW_COLOR, label="low_spindle_rate", n=low["n"],
        rate=low["rate"], rate_sem=low["rate_sem"], rate_unit=rate_unit,
        individual_curves=low.get("individual_curves"),
        infraslow_band=infraslow_band, annotate_xy=(0.97, 0.55),
    )

    ax.set_xlim(infraslow_band)
    ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Baseline-corrected relative power", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    fig.suptitle(
        f"Infraslow oscillation — {sleep_stage}/{channel}: low- vs. high-spindle-rate group",
        y=1.02, fontsize=SUPTITLE_FONTSIZE, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_png, output_pdf)
    plt.close(fig)


def plot_cohort_infraslow_compare(
    *,
    cohort: Dict[str, object],
    infraslow_band: Tuple[float, float],
    sleep_stage: str,
    channel: str,
    output_png: Path,
    output_pdf: Path,
    rate_unit: str = "min",
) -> None:
    """The whole-cohort N2-C3 infraslow figure, *before* any low/high spindle-rate split.

    The "before" counterpart to :func:`plot_group_infraslow_compare`: same
    visual components (individual-subject lines, SEM band, mean curve,
    bi-Gaussian fit, peak/spindle annotation), drawn once for every validated
    subject as a single group -- a baseline reference to compare against the
    after-grouping low_spindle_rate vs. high_spindle_rate figure
    (md/group_analysis.md Step 6).

    Args:
        cohort: Dict with keys ``freqs, mean, sem, n, fit, fitted_curve,
            fitted_freqs, individual_curves, rate, rate_sem`` -- see
            ``group_analysis.py``'s ``_build_group_plot_data``, called on the
            whole validated cohort (not a low/high subset).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0.0, ls=":", color="0.3", label="Baseline (0)")

    _draw_one_group_curve(
        ax, cohort["freqs"], cohort["mean"], cohort["sem"], fit=cohort["fit"],
        fitted_curve=cohort["fitted_curve"], fitted_freqs=cohort["fitted_freqs"],
        color=ALL_COLOR, label="all_subjects", n=cohort["n"],
        rate=cohort["rate"], rate_sem=cohort["rate_sem"], rate_unit=rate_unit,
        individual_curves=cohort.get("individual_curves"), individual_label="individual subjects",
        infraslow_band=infraslow_band, annotate_xy=(0.97, 0.97),
    )

    ax.set_xlim(infraslow_band)
    ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Baseline-corrected relative power", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    fig.suptitle(
        f"Infraslow oscillation — {sleep_stage}/{channel}: whole cohort (before grouping)",
        y=1.02, fontsize=SUPTITLE_FONTSIZE, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_png, output_pdf)
    plt.close(fig)


def _draw_ci_band(ax, sp_stats, group: Dict[str, object], *, color: str, label: str,
                   infraslow_band: Tuple[float, float], ci: float) -> None:
    """One group's mean curve + Normal-approximation CI band (from per-frequency SEM) onto `ax`."""
    freqs, mean, sem, n = group["freqs"], group["mean"], group["sem"], group["n"]
    band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
    z = float(sp_stats.norm.ppf(0.5 + ci / 2)) if n > 1 else 0.0
    half_width = z * sem
    ax.fill_between(freqs[band_m], (mean - half_width)[band_m], (mean + half_width)[band_m],
                     color=color, alpha=0.25)
    ax.plot(freqs[band_m], mean[band_m], color=color, lw=2.0, label=f"{label} (n={n})")


def plot_group_spectrum_clean(
    *,
    low: Dict[str, object],
    high: Dict[str, object],
    infraslow_band: Tuple[float, float],
    sleep_stage: str,
    channel: str,
    output_png: Path,
    output_pdf: Path,
    ci: float = 0.95,
) -> None:
    """A clean group-level spectrum figure: mean +/- CI per group, no fit clutter.

    Shows low- and high-spindle-rate mean spectra with a ``ci``-level
    confidence band (Normal approximation from each group's per-frequency
    SEM), group sample sizes, and ``sleep_stage``/``channel`` stated in the
    title (md/group_analysis.md Step 6).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for group, color, label in ((low, LOW_COLOR, "low_spindle_rate"), (high, HIGH_COLOR, "high_spindle_rate")):
        _draw_ci_band(ax, sp_stats, group, color=color, label=label, infraslow_band=infraslow_band, ci=ci)

    ax.axhline(0.0, ls=":", color="0.3")
    ax.set_xlim(infraslow_band)
    ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Baseline-corrected relative power", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        f"{sleep_stage}, {channel} infraslow spectrum by spindle-rate group ({int(ci * 100)}% CI)",
        fontsize=TITLE_FONTSIZE, fontweight="semibold",
    )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_png, output_pdf)
    plt.close(fig)


def plot_cohort_spectrum_clean(
    *,
    cohort: Dict[str, object],
    infraslow_band: Tuple[float, float],
    sleep_stage: str,
    channel: str,
    output_png: Path,
    output_pdf: Path,
    ci: float = 0.95,
) -> None:
    """A clean whole-cohort spectrum figure, *before* any low/high spindle-rate split.

    The "before" counterpart to :func:`plot_group_spectrum_clean`: the same
    mean +/- CI band (no fit clutter), drawn once for every validated subject
    as a single group.

    Args:
        cohort: Dict with keys ``freqs, mean, sem, n`` -- see
            ``group_analysis.py``'s ``_build_group_plot_data``, called on the
            whole validated cohort (not a low/high subset).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _draw_ci_band(ax, sp_stats, cohort, color=ALL_COLOR, label="all_subjects",
                  infraslow_band=infraslow_band, ci=ci)

    ax.axhline(0.0, ls=":", color="0.3")
    ax.set_xlim(infraslow_band)
    ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Baseline-corrected relative power", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        f"{sleep_stage}, {channel} infraslow spectrum, whole cohort ({int(ci * 100)}% CI, before grouping)",
        fontsize=TITLE_FONTSIZE, fontweight="semibold",
    )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_png, output_pdf)
    plt.close(fig)


def plot_parameter_comparisons(
    *,
    df: pd.DataFrame,
    group_col: str,
    parameters: Sequence[str],
    comparison_df: pd.DataFrame,
    low_label: str,
    high_label: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Violin + individual-subject-point plots for each N2-C3 summary parameter.

    One subplot per entry in ``parameters``; each subplot's title is annotated
    with its FDR-adjusted q-value from ``comparison_df`` (md/group_analysis.md
    Step 6). ``comparison_df`` must have ``parameter``/``q_value`` columns
    (see ``infraslow.stats.group_comparison.compare_parameters``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = len(parameters)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    q_by_param = comparison_df.set_index("parameter")["q_value"].to_dict()
    rng = np.random.default_rng(0)

    for ax, parameter in zip(axes, parameters):
        low_vals = df.loc[df[group_col] == low_label, parameter].dropna().to_numpy(dtype=float)
        high_vals = df.loc[df[group_col] == high_label, parameter].dropna().to_numpy(dtype=float)

        parts = ax.violinplot([low_vals, high_vals], showmedians=True)
        for body, color in zip(parts["bodies"], (LOW_COLOR, HIGH_COLOR)):
            body.set_facecolor(color)
            body.set_alpha(0.4)

        for position, (values, color) in enumerate(((low_vals, LOW_COLOR), (high_vals, HIGH_COLOR)), start=1):
            jitter = rng.normal(0, 0.04, size=values.size)
            ax.scatter(np.full(values.size, position) + jitter, values, color=color, s=10, alpha=0.5, zorder=3)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([low_label, high_label], fontsize=TICK_FONTSIZE)
        q_value = q_by_param.get(parameter, np.nan)
        title = parameter if not np.isfinite(q_value) else f"{parameter} (q={q_value:.3g})"
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n_params:]:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, output_png, output_pdf)
    plt.close(fig)


def plot_parameter_distributions(
    *,
    df: pd.DataFrame,
    parameters: Sequence[str],
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Violin + individual-subject-point plot for each N2-C3 summary parameter,
    whole cohort, *before* any low/high spindle-rate split.

    The "before" counterpart to :func:`plot_parameter_comparisons`: one violin
    per parameter (not two), no q-value annotation -- there is no low-vs-high
    comparison to run before the groups exist (md/group_analysis.md Step 6).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = len(parameters)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.2 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    rng = np.random.default_rng(0)

    for ax, parameter in zip(axes, parameters):
        values = df[parameter].dropna().to_numpy(dtype=float)

        parts = ax.violinplot([values], showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(ALL_COLOR)
            body.set_alpha(0.4)

        jitter = rng.normal(0, 0.04, size=values.size)
        ax.scatter(np.full(values.size, 1) + jitter, values, color=ALL_COLOR, s=10, alpha=0.5, zorder=3)

        ax.set_xticks([1])
        ax.set_xticklabels(["all_subjects"], fontsize=TICK_FONTSIZE)
        ax.set_title(parameter, fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n_params:]:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, output_png, output_pdf)
    plt.close(fig)
