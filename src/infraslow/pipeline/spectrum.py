"""Subject/state/channel-level infraslow spectrum features.

Reproduces `demo_infraslow_yasa_recheck.ipynb`'s Figure 5 pipeline exactly,
generalized from its one hardcoded subject/channel/bout to any selection of
bouts, and run against `preprocessing.py`'s already-computed per-bout PSDs
(`<stage>/ISFS/<band>.npz`) instead of re-running `infraslow_spectrum`. The
notebook's bout-selection criterion -- N2/N3 runs >= min_dur AND containing
>= 1 detected spindle -- is exactly `bouts["spindle"]` from
`<stage>/bouts.npz` (see `preprocessing.py`'s `preprocess_channel`, which
saves that exact subset). All fitting/integration math (`fit_isfs`,
`bigaussian`, `chromatogram_peak_area`) is reused unmodified from
`infraslow.processing.infraslow` -- nothing here redefines it.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

from ..processing.infraslow import (
    DEFAULT_BASELINE_BAND,
    DEFAULT_INFRASLOW_BAND,
    baseline_correct,
    bigaussian,
    chromatogram_peak_area,
    fit_isfs,
    relative_power,
)

_CHROMATOGRAM_GRID_POINTS = 200  # matches demo_infraslow_yasa_compare.py's `fg = np.linspace(*band, 200)`


def select_event_bouts(
    event_bouts: np.ndarray, isfs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """The subset of `isfs`'s per-bout rows whose bout matches one of `event_bouts`
    (e.g. `bouts["spindle"]` or `sw_bouts["sw"]`).

    `isfs["bout_start"]` and `bouts["all"][:, 0]`/`sw_bouts["all"][:, 0]` come from the
    exact same `all_bouts` list inside `preprocess_channel` (same order, same float
    values, no recomputation), so matching against `event_bouts[:, 0]` by exact float
    equality recovers the notebook's own "N2/N3 bouts >= min_dur AND containing >= 1
    event" selection (spindle or slow-wave, whichever `event_bouts` came from) without
    ever touching the raw EEG signal.
    """
    starts = event_bouts[:, 0] if event_bouts.size else np.empty(0)
    mask = np.isin(isfs["bout_start"], starts)
    return {
        "freqs": isfs["freqs"],
        "psds": isfs["psds"][mask],
        "bout_start": isfs["bout_start"][mask],
    }


def select_spindle_bouts(
    bouts: Dict[str, np.ndarray], isfs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Backward-compatible spindle-specific wrapper around `select_event_bouts`
    (see its docstring) -- equivalent to `select_event_bouts(bouts["spindle"], isfs)`."""
    return select_event_bouts(bouts["spindle"], isfs)


def _nan_spectrum_features() -> Dict[str, float]:
    keys = (
        "peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc", "chromatogram_peak_area",
        "bi_gaussian_amp", "bi_gaussian_mu", "bi_gaussian_sd_l", "bi_gaussian_sd_r",
        "isfs_threshold", "real_peak_freq_hz", "real_peak_period_s",
    )
    out = {k: float("nan") for k in keys}
    out["isfs_detected"] = False
    return out


def compute_subject_spectrum_features(
    freqs: np.ndarray, psds: np.ndarray, *,
    infraslow_band=DEFAULT_INFRASLOW_BAND, baseline_band=DEFAULT_BASELINE_BAND,
    return_curves: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    """One subject/state/channel's spectrum features, from its selected bouts' PSDs.

    Mirrors the notebook exactly: `mean_psd` across the selected bouts ->
    `relative_power` -> `baseline_correct` -> `fit_isfs` (bi-Gaussian) ->
    `chromatogram_peak_area` on the fit curve sampled on a dense grid (the
    notebook's own methodology in `demo_infraslow_yasa_compare.ipynb`, which
    evaluates `chromatogram_peak_area` on `bigaussian(fg, *fit['popt'])`,
    not on the raw/corrected spectrum directly). The "real" (empirical, not
    fitted) peak is `argmax` of the *relative* (not baseline-corrected)
    spectrum within `infraslow_band`, exactly as the notebook computes it.

    Returns an all-NaN, `isfs_detected=False` dict if `psds` has zero rows
    (this subject/state/channel had no spindle-containing bouts) -- the
    caller (pipeline.py) decides whether that means "state unavailable".

    If `return_curves` is set, additionally returns a second dict with the
    `freqs`/`rel`/`corrected` intermediate arrays (e.g. for figure plotting)
    -- the plain-dict return stays the default so existing call sites are
    unaffected.
    """
    if psds.shape[0] == 0:
        if return_curves:
            return _nan_spectrum_features(), dict(
                freqs=freqs, rel=np.full_like(freqs, np.nan), corrected=np.full_like(freqs, np.nan),
            )
        return _nan_spectrum_features()

    mean_psd = psds.mean(axis=0)
    rel = relative_power(mean_psd, freqs, infraslow_band)
    corrected = baseline_correct(rel, freqs, baseline_band)

    band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
    real_peak_idx = int(np.argmax(rel[band_m]))
    real_peak_freq = float(freqs[band_m][real_peak_idx])

    fit = fit_isfs(freqs, corrected, infraslow_band=infraslow_band, baseline_band=baseline_band)

    fg = np.linspace(infraslow_band[0], infraslow_band[1], _CHROMATOGRAM_GRID_POINTS)
    fitted_curve = bigaussian(fg, *fit["popt"])
    chrom = chromatogram_peak_area(fg, fitted_curve, threshold=fit["threshold"], infraslow_band=infraslow_band)

    feats = dict(
        peak_freq_hz=fit["mu"],
        peak_period_s=1.0 / fit["mu"],
        bandwidth_hz=fit["bandwidth"],
        auc=fit["auc"],
        chromatogram_peak_area=chrom["area"],
        bi_gaussian_amp=fit["amp"],
        bi_gaussian_mu=fit["mu"],
        bi_gaussian_sd_l=fit["sd_l"],
        bi_gaussian_sd_r=fit["sd_r"],
        isfs_detected=fit["detected"],
        isfs_threshold=fit["threshold"],
        real_peak_freq_hz=real_peak_freq,
        real_peak_period_s=1.0 / real_peak_freq,
    )
    if return_curves:
        return feats, dict(freqs=freqs, rel=rel, corrected=corrected)
    return feats


def compute_bout_peak_freqs(
    freqs: np.ndarray, psds: np.ndarray, *, infraslow_band=DEFAULT_INFRASLOW_BAND,
) -> np.ndarray:
    """Per-bout empirical peak frequency (Hz): `argmax` of each bout's own
    relative-power spectrum within `infraslow_band` (Figure 2's distribution)."""
    if psds.shape[0] == 0:
        return np.empty(0)
    rel_all = relative_power(psds, freqs, infraslow_band)
    band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
    idx = np.argmax(rel_all[:, band_m], axis=1)
    return freqs[band_m][idx]


__all__ = [
    "select_event_bouts",
    "select_spindle_bouts",
    "compute_subject_spectrum_features",
    "compute_bout_peak_freqs",
]
