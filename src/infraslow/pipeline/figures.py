"""The 6 required per-state figures (3 figure types x N2/N3).

Matplotlib only, `Agg` backend (headless Sherlock compute nodes have no
display) -- set once at import time, matching how a batch script should
render figures without a `$DISPLAY`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - backend must be set first
import numpy as np  # noqa: E402

from ..processing.infraslow import bigaussian  # noqa: E402


def plot_relative_spectrum_bigaussian(
    freqs: np.ndarray, rel: np.ndarray, corrected: np.ndarray, fit_features: dict, popt, *,
    state: str, out_path: Path,
) -> None:
    """Relative spectral power, baseline-corrected spectrum, bi-Gaussian fit,
    and detected peak -- annotated with peak_freq_hz/peak_period_s/
    bandwidth_hz/auc/chromatogram_peak_area (Figure 1)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, rel, color="tab:blue", label="relative spectral power")
    ax.plot(freqs, corrected, color="tab:orange", label="baseline-corrected")
    if not any(np.isnan(popt)):
        fg = np.linspace(freqs.min(), freqs.max(), 200)
        ax.plot(fg, bigaussian(fg, *popt), color="tab:purple", lw=2, label="bi-Gaussian fit")
    peak_freq = fit_features.get("peak_freq_hz", np.nan)
    if not np.isnan(peak_freq):
        ax.axvline(peak_freq, color="k", ls="--", lw=1, label="detected peak")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"{state}: relative spectral power + bi-Gaussian fit")
    annotation = "\n".join(
        f"{k} = {fit_features[k]:.4g}"
        for k in ("peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc", "chromatogram_peak_area")
        if k in fit_features and not (isinstance(fit_features[k], float) and np.isnan(fit_features[k]))
    )
    if annotation:
        ax.text(0.98, 0.98, annotation, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_peak_frequency_distribution(bout_peak_freqs: np.ndarray, *, state: str, out_path: Path) -> None:
    """Distribution of per-bout `peak_freq_hz` across every valid bout for
    this sleep state (Figure 2) -- not forced to center at any fixed value."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if bout_peak_freqs.size > 0:
        ax.hist(bout_peak_freqs, bins=20, color="tab:blue", alpha=0.8)
    else:
        ax.text(0.5, 0.5, "no valid bouts", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Peak frequency (Hz)")
    ax.set_ylabel("Bout count")
    ax.set_title(f"{state}: peak frequency distribution across bouts")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spindle_phase_distribution(pooled_dist: dict, *, state: str, out_path: Path) -> None:
    """Spindle occurrence rate across the 8 ISFS phase bins, pooled across
    every subject in the cohort for this sleep state (Figure 3)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(1, 9)
    ax.bar(bins, pooled_dist["phase_bin_rates"], color="tab:green")
    ax.set_xticks(bins)
    ax.set_xlabel("ISFS phase bin")
    ax.set_ylabel("% of events")
    ax.set_title(f"{state}: spindle distribution across ISFS phase bins (n={pooled_dist['event_count']})")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


__all__ = [
    "plot_relative_spectrum_bigaussian",
    "plot_peak_frequency_distribution",
    "plot_spindle_phase_distribution",
]
