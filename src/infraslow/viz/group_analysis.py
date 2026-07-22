"""N2-C3 spindle-rate group-comparison plots (``src/group_analysis.py``).

Pure plotting: every number these functions draw (fits, chromatogram peak
areas, GMM components, q-values, ...) is computed by the caller with the same
scientific functions ``demo_infraslow_yasa_compare.py`` uses
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
    "plot_spindle_rate_distribution",
    "plot_group_infraslow_compare",
    "plot_group_spectrum_clean",
    "plot_parameter_comparisons",
]

LOW_COLOR = "#1f77b4"
HIGH_COLOR = "#d62728"

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


def plot_spindle_rate_distribution(
    *,
    spindle_per_min: np.ndarray,
    spindle_group: np.ndarray,
    uncertain: np.ndarray,
    gmm_scale: str,
    gmm_model,
    centers_original_scale: np.ndarray,
    low_label: str,
    high_label: str,
    output_png: Path,
) -> None:
    """N2-C3 ``spindle_per_min`` histogram with the fitted 2-component GMM overlay.

    Shows: the distribution split by final group membership, each fitted
    Gaussian component's density (transformed back to the original
    spindles/min scale via the log1p Jacobian when ``gmm_scale == "log1p"``),
    both component centers, the estimated decision boundary, and the group /
    uncertain-assignment counts (md/group_analysis.md Step 4).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    values = np.asarray(spindle_per_min, dtype=float)
    spindle_group = np.asarray(spindle_group)
    n_low = int((spindle_group == low_label).sum())
    n_high = int((spindle_group == high_label).sum())
    n_uncertain = int(np.sum(uncertain))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(values[spindle_group == low_label], bins=40, color=LOW_COLOR, alpha=0.5,
            density=True, label=f"{low_label} (n={n_low})")
    ax.hist(values[spindle_group == high_label], bins=40, color=HIGH_COLOR, alpha=0.5,
            density=True, label=f"{high_label} (n={n_high})")

    # Component order on the *original* scale (ascending): index 0 is low, 1 is high --
    # matches infraslow.stats.group_assignment.assign_spindle_rate_groups's own ordering.
    order = np.argsort(centers_original_scale)
    means = gmm_model.means_.ravel()
    covariances = np.asarray(gmm_model.covariances_).reshape(-1)
    weights = gmm_model.weights_

    grid_original = np.linspace(values.min(), values.max(), 400)
    grid_fit = np.log1p(grid_original) if gmm_scale == "log1p" else grid_original
    # d(log1p(x))/dx = 1/(1+x); required to keep the overlaid densities on the
    # original spindles/min scale integrate to (approximately) each component's weight.
    jacobian = (1.0 / (1.0 + grid_original)) if gmm_scale == "log1p" else np.ones_like(grid_original)

    component_densities = np.zeros((2, grid_original.size))
    for raw_idx in range(2):
        pdf = sp_stats.norm.pdf(grid_fit, loc=means[raw_idx], scale=np.sqrt(covariances[raw_idx]))
        component_densities[raw_idx] = weights[raw_idx] * pdf * jacobian

    for rank, raw_idx in enumerate(order):
        color, name = (LOW_COLOR, low_label) if rank == 0 else (HIGH_COLOR, high_label)
        ax.plot(grid_original, component_densities[raw_idx], color=color, lw=2.0, ls="--",
                label=f"GMM component ({name})")
        ax.axvline(centers_original_scale[raw_idx], color=color, lw=1.4, ls=":",
                   label=f"{name} center = {centers_original_scale[raw_idx]:.2f}/min")

    # Decision boundary: where the higher-density component along the grid flips
    # from low- to high-assigned (component order matches the `order` ranking above).
    winner = np.argmax(component_densities, axis=0)
    ordered_winner = np.array([0 if raw_idx == order[0] else 1 for raw_idx in winner])
    flips = np.flatnonzero(np.diff(ordered_winner) != 0)
    if flips.size:
        boundary = grid_original[flips[0]]
        ax.axvline(boundary, color="0.2", lw=1.6, label=f"decision boundary ≈ {boundary:.2f}/min")

    ax.set_xlabel("N2-C3 spindle_per_min", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        f"N2-C3 spindle-rate grouping ({gmm_scale}-scale GMM) — {n_uncertain} uncertain assignment(s)",
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
    color: str, label: str, n: int, spindle_per_min: float, spindle_per_min_sem: float,
    infraslow_band: Tuple[float, float], annotate_xy: Tuple[float, float],
) -> None:
    """Draw one group's SEM band, mean curve, bi-Gaussian fit, and peak/spindle annotation onto `ax`.

    Mirrors ``demo_infraslow_yasa_compare.plot_channel_subplot``'s visual
    components (SEM band, average line, bi-Gaussian fit, peak marker, spindle
    annotation) for one group instead of one channel, so both groups can be
    drawn on the same shared axes.
    """
    band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
    ax.fill_between(freqs[band_m], (mean - sem)[band_m], (mean + sem)[band_m],
                     color=color, alpha=0.2, zorder=2)
    ax.plot(freqs[band_m], mean[band_m], color=color, lw=2.0, zorder=3, label=f"{label} (n={n})")

    if fit is None or fitted_curve is None or fitted_freqs is None:
        return
    ax.axvspan(fit["lo"], fit["hi"], color=color, alpha=0.08, zorder=0)
    ax.plot(fitted_freqs, fitted_curve, color=color, lw=1.4, ls="--", zorder=4,
            label=f"{label} bi-Gaussian fit")
    ax.plot([fit["mu"]], [fit["amp"]], "o", color=color, ms=6, zorder=5, mec="k", mew=0.6)

    rate_line = (f"{spindle_per_min:.2f}±{spindle_per_min_sem:.2f} spindles/min"
                 if np.isfinite(spindle_per_min) else "")
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
) -> None:
    """The main N2-C3 low- vs. high-spindle-rate comparison figure.

    Reproduces every visually-relevant component of
    ``demo_infraslow_yasa_compare.plot_channel_subplot`` (baseline-at-0 line,
    per-group SEM band + mean curve, bi-Gaussian fit, peak-frequency marker,
    shaded bandwidth span, AUC/spindle-rate annotation), adapted to compare two
    groups on one shared axes instead of one channel per subplot
    (md/group_analysis.md Step 6).

    Args:
        low, high: Dicts with keys ``freqs, mean, sem, n, fit, fitted_curve,
            fitted_freqs, spindle_per_min, spindle_per_min_sem`` -- see
            ``group_analysis.py``'s ``_build_group_plot_data``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0.0, ls=":", color="0.3", label="Baseline (0)")

    _draw_one_group_curve(
        ax, low["freqs"], low["mean"], low["sem"], fit=low["fit"],
        fitted_curve=low["fitted_curve"], fitted_freqs=low["fitted_freqs"],
        color=LOW_COLOR, label="low_spindle_rate", n=low["n"],
        spindle_per_min=low["spindle_per_min"], spindle_per_min_sem=low["spindle_per_min_sem"],
        infraslow_band=infraslow_band, annotate_xy=(0.97, 0.97),
    )
    _draw_one_group_curve(
        ax, high["freqs"], high["mean"], high["sem"], fit=high["fit"],
        fitted_curve=high["fitted_curve"], fitted_freqs=high["fitted_freqs"],
        color=HIGH_COLOR, label="high_spindle_rate", n=high["n"],
        spindle_per_min=high["spindle_per_min"], spindle_per_min_sem=high["spindle_per_min_sem"],
        infraslow_band=infraslow_band, annotate_xy=(0.97, 0.55),
    )

    ax.set_xlim(infraslow_band)
    ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Baseline-corrected relative power", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, loc="upper left")
    fig.suptitle(
        f"Infraslow oscillation — {sleep_stage}/{channel}: low- vs. high-spindle-rate group",
        y=1.02, fontsize=SUPTITLE_FONTSIZE, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_png, output_pdf)
    plt.close(fig)


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
        freqs, mean, sem, n = group["freqs"], group["mean"], group["sem"], group["n"]
        band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
        z = float(sp_stats.norm.ppf(0.5 + ci / 2)) if n > 1 else 0.0
        half_width = z * sem
        ax.fill_between(freqs[band_m], (mean - half_width)[band_m], (mean + half_width)[band_m],
                         color=color, alpha=0.25)
        ax.plot(freqs[band_m], mean[band_m], color=color, lw=2.0, label=f"{label} (n={n})")

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
