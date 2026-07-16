"""Subject-level sleep + infraslow metrics for the Bioserenity cohort.

This module industrialises the single-subject analysis prototyped in
``src/demo_infraslow_yasa_average.ipynb`` so it can run over *every* subject that
has all three required inputs on ``$OAK``:

* metadata row (``ID, Age, Gender, BMI``) from *either* of two source CSVs --
  ``Morpheus_Data_All5.csv`` and ``bioserenity_metadata3.csv`` -- combined by
  :func:`combine_bioserenity_metadata` (see ``src/match_cohort.py``),
* an EDF signal file ``$OAK/psg/Bioserenity/edf/{id}.edf``,
* a hypnodensity CSV ``$OAK/psg/Bioserenity/Sleep_Staging/{id}_Hypnodensity.csv``.

For each valid subject it produces one flat row combining three metric families:

1. **Metadata** -- ``ID, Age, Gender, BMI`` (column-name tolerant on input).
2. **YASA sleep statistics** -- the full :func:`yasa.sleep_statistics` output
   (``TIB, SPT, WASO, TST, N1, N2, N3, REM, NREM, SOL, Lat_*, %*, SE, SME``),
   computed from the argmax hypnogram at 30-s epochs.
3. **Infraslow sigma-power metrics** -- for each of the ``N2``, ``N3`` and
   ``NREM`` stage groups, the infraslow (~0.02 Hz) oscillation of sigma-band
   power estimated over long, spindle-bearing bouts (the reference notebook's
   Fig C-i analysis), reported as ``{stage}_peak_freq``, ``{stage}_fit_peak_freq``
   (each with a ``_sem`` across bouts), ``{stage}_bandwidth_hz``, ``{stage}_auc``,
   ``{stage}_detected``, plus the consolidated-bout burden ``{stage}_bouts`` (min)
   and ``%{stage}_bouts`` (of TST).

Design notes
------------
* **Two peak estimators.** ``{stage}_peak_freq`` is the *empirical* argmax of the
  baseline-corrected relative spectrum inside the fit band; ``{stage}_fit_peak_freq``
  is the Gaussian-fit centre ``mu``. Both are averaged **across bouts**, with the
  ``_sem`` giving the standard error across those bouts (``NaN`` when <2 bouts).
* **Bandwidth / AUC / detection** are read off the fit of the **bout-averaged**
  spectrum (exactly as the reference notebook), which is more stable than a
  per-bout fit; ``{stage}_detected`` is that fit's ``amp > 1.5*SD(baseline)`` test.
* **NREM** here means N1+N2+N3 (clinical NREM, matching the YASA ``NREM`` column).
  Spindle detection for the bout filter is restricted to N2+N3, where spindles
  occur -- an N1 epoch inside an NREM bout simply carries no spindle.
* **``{stage}_bouts``** is the total duration (minutes) of consecutive stage-X
  runs of at least :data:`MIN_BOUT_SEC` -- a *consolidated-bout* burden distinct
  from the YASA per-stage totals; ``%{stage}_bouts`` expresses it as a percentage
  of TST.
* Anything that cannot be computed becomes ``numpy.nan`` (or ``False`` for
  ``detected``) rather than raising, so one bad stage never sinks a subject and
  one bad subject never sinks the run.

Everything here is pure NumPy/SciPy/pandas plus the repo's own io/processing
helpers; ``yasa`` is imported lazily (it is only needed at call time, on a
compute node -- never on the login node).
"""

from __future__ import annotations

import logging
import traceback as _traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..io.hypnodensity import (
    DEFAULT_HYPNODENSITY_SUFFIX,
    hypnodensity_to_annotations,
)
from ..io.psg_loader import BioserenityPSGLoader
from ..io.utils import list_dir_filenames, progress_iter
from .detection import (
    DEFAULT_EEG_CHANNELS,
    DEFAULT_EPOCH_SEC,
    DEFAULT_STAGE_MAP,
    _extract_epoch_stages,
    _stages_to_int,
    spindles_detect,
)
from .infraslow import (
    DEFAULT_INFRASLOW_BAND,
    DEFAULT_SF_ENV,
    infraslow_spectrum,
    sigma_power_envelope,
    spindle_rate_series,
)

logger = logging.getLogger(__name__)

# ``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2.0; use whichever exists.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# Analysis constants (kept identical to demo_infraslow_yasa_average.ipynb)
# --------------------------------------------------------------------------- #
SF: float = 128.0                       # common resample rate (Hz) per subject
CHANNEL: str = "C3"                     # EEG channel for the sigma-power envelope
SIGMA_BAND: Tuple[float, float] = (10.0, 16.0)   # sigma (spindle) power band
INFRASLOW_BAND: Tuple[float, float] = DEFAULT_INFRASLOW_BAND  # (0.01, 0.1) Hz
BASELINE_BAND: Tuple[float, float] = (0.06, 0.1)  # baseline-correction band
MIN_BOUT_SEC: float = 200.0             # min consecutive-stage bout length (s)
WINDOW_SEC: float = 100.0               # fixed Welch window -> shared freq grid
EPOCH_SEC: float = DEFAULT_EPOCH_SEC     # 30 s scored epochs
SF_ENV: float = DEFAULT_SF_ENV           # 1 Hz sigma-power envelope rate

# Stage groups analysed for infraslow metrics -> the YASA integer codes that
# make up each group (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM).
STAGE_GROUP_CODES: Dict[str, Tuple[int, ...]] = {
    "N1": (1,),
    "N2": (2,),
    "N3": (3,),
    "NREM": (1, 2, 3),
}
INFRASLOW_STAGES: Tuple[str, ...] = ("N1", "N2", "N3", "NREM")
# Stage groups written to the per-subject, per-channel npz files (run_all_metrics.py
# v2 pipeline; see calculate_subject_channel_events). Same three stage groups as
# INFRASLOW_STAGES -- kept as a separate name since the v2 pipeline's per-channel
# npz layout is a distinct concept from the legacy single-channel scalar metrics.
NPZ_STAGES: Tuple[str, ...] = INFRASLOW_STAGES
# Sleep stages spindles are detected within for the bout filter (N1+N2+N3).
SPINDLE_INCLUDE: Tuple[int, ...] = (1, 2, 3)
# EEG channels analysed per subject in the v2 (per-channel) npz pipeline.
CHANNELS: Tuple[str, ...] = DEFAULT_EEG_CHANNELS

# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #
METADATA_COLUMNS: List[str] = ["ID", "Age", "Gender", "BMI"]

#: Full YASA ``sleep_statistics`` output, in canonical order. Any key a given
#: YASA build omits is filled with ``NaN`` so the schema is stable across versions.
YASA_STAT_COLUMNS: List[str] = [
    "TIB", "SPT", "WASO", "TST",
    "N1", "N2", "N3", "REM", "NREM",
    "SOL", "Lat_N1", "Lat_N2", "Lat_N3", "Lat_REM",
    "%N1", "%N2", "%N3", "%REM", "%NREM",
    "SE", "SME",
]

#: Per-stage infraslow *spectral* metric suffixes emitted by
#: :func:`calculate_stage_infraslow`, in output order. The ``{stage}_bouts`` and
#: ``%{stage}_bouts`` columns are appended separately (see :func:`_stage_columns`)
#: because they come from :func:`calculate_stage_bouts`, not the spectral fit.
_STAGE_METRIC_KEYS: List[str] = [
    "peak_freq", "peak_freq_sem",
    "fit_peak_freq", "fit_peak_freq_sem",
    "bandwidth_hz", "auc", "detected",
]

def _stage_columns(stage: str) -> List[str]:
    """Ordered output columns for one infraslow stage group."""
    cols = [f"{stage}_{key}" for key in _STAGE_METRIC_KEYS]
    cols.append(f"{stage}_bouts")
    cols.append(f"%{stage}_bouts")
    return cols


def all_metric_columns() -> List[str]:
    """The full, ordered column schema of the final metrics table."""
    cols = list(METADATA_COLUMNS) + list(YASA_STAT_COLUMNS)
    for stage in INFRASLOW_STAGES:
        cols += _stage_columns(stage)
    return cols


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def load_bioserenity_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load metadata and standardise columns to ``ID, Age, Gender, BMI``.

    Only the four needed columns are read (the CSV is wide and long). ``ID`` is
    read as a string (subject ids are brace-wrapped GUIDs); ``Age`` and ``BMI``
    are coerced to numeric with unparseable cells becoming ``NaN``.

    Args:
        metadata_path: Path to a metadata CSV (e.g. ``Morpheus_Data_All5.csv``
            or ``bioserenity_metadata3.csv``) with columns named exactly
            ``ID, Age, Gender, BMI``.

    Returns:
        DataFrame with exactly the columns ``["ID", "Age", "Gender", "BMI"]``,
        rows with a missing/blank ``ID`` dropped.

    Raises:
        FileNotFoundError: if ``metadata_path`` does not exist.
        KeyError: if the ``ID`` column is missing.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")

    # Read only the header first to check which columns are present.
    header = pd.read_csv(metadata_path, nrows=0)
    available = list(header.columns)
    if "ID" not in available:
        raise KeyError(
            f"Could not find an 'ID' column in {metadata_path.name}; "
            f"Columns: {available[:10]}..."
        )

    usecols = [c for c in METADATA_COLUMNS if c in available]
    out = pd.read_csv(metadata_path, usecols=usecols, dtype={"ID": str})
    for canonical in METADATA_COLUMNS:
        if canonical not in out.columns:
            out[canonical] = np.nan
    out = out[METADATA_COLUMNS].copy()

    out["ID"] = out["ID"].astype(str).str.strip()
    # Some source exports (e.g. Morpheus) write integer ids as floats ("8631.0");
    # the on-disk EDF/hypnodensity files use the bare integer ("8631.edf"), so
    # strip a spurious trailing ".0" before matching. GUID-style ids are untouched.
    out["ID"] = out["ID"].str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    out = out[out["ID"].notna() & (out["ID"] != "") & (out["ID"].str.lower() != "nan")]
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["BMI"] = pd.to_numeric(out["BMI"], errors="coerce")
    out["Gender"] = out["Gender"].astype(str).str.strip()
    return out.reset_index(drop=True)


def combine_bioserenity_metadata(*frames: pd.DataFrame) -> pd.DataFrame:
    """Union metadata rows from multiple sources, keyed by ``ID``.

    A subject is kept if its ``ID`` appears in *any* of ``frames`` (OR, not
    intersection). Where the same ``ID`` appears in more than one frame, the
    first non-null value per column wins.

    Args:
        *frames: Standardised metadata frames (see :func:`load_bioserenity_metadata`).

    Returns:
        DataFrame with columns ``["ID", "Age", "Gender", "BMI"]``, one row per
        distinct ``ID``.
    """
    combined = pd.concat(frames, ignore_index=True)
    coalesce_first = lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
    out = combined.groupby("ID", as_index=False).agg(
        {col: coalesce_first for col in METADATA_COLUMNS if col != "ID"}
    )
    return out[METADATA_COLUMNS].reset_index(drop=True)


def find_valid_bioserenity_subjects(
    metadata: pd.DataFrame,
    edf_dir: Path,
    hypnodensity_dir: Path,
    *,
    id_column: str = "ID",
    edf_suffix: str = ".edf",
    hypnodensity_suffix: str = DEFAULT_HYPNODENSITY_SUFFIX,
) -> pd.DataFrame:
    """Return metadata rows whose EDF *and* hypnodensity files both exist.

    Both directories are listed once (a single ``readdir`` each via
    :func:`~infraslow.io.utils.list_dir_filenames`) and membership is tested in
    memory -- this avoids a per-subject ``stat`` storm against the ``$OAK`` Lustre
    metadata server.

    The returned frame carries an ``availability`` summary dict in ``df.attrs``
    (counts of metadata / with-EDF / with-hypnodensity / valid subjects) for the
    notebook's processing summary.

    Args:
        metadata: Standardised metadata (see :func:`load_bioserenity_metadata`).
        edf_dir: Directory of ``{id}.edf`` files.
        hypnodensity_dir: Directory of ``{id}_Hypnodensity.csv`` files.
        id_column: Column of subject ids in ``metadata``.
        edf_suffix, hypnodensity_suffix: Filename suffixes appended to the id.

    Returns:
        A copy of the matching metadata rows (order preserved), reindexed.
    """
    edf_dir = Path(edf_dir)
    hypnodensity_dir = Path(hypnodensity_dir)
    edf_names = list_dir_filenames(edf_dir)
    hypno_names = list_dir_filenames(hypnodensity_dir)

    ids = metadata[id_column].astype(str)
    has_edf = ids.map(lambda s: f"{s}{edf_suffix}" in edf_names)
    has_hypno = ids.map(lambda s: f"{s}{hypnodensity_suffix}" in hypno_names)
    valid_mask = has_edf & has_hypno

    valid = metadata.loc[valid_mask].reset_index(drop=True).copy()
    valid.attrs["availability"] = {
        "n_metadata": int(len(metadata)),
        "n_with_edf": int(has_edf.sum()),
        "n_with_hypnodensity": int(has_hypno.sum()),
        "n_valid": int(valid_mask.sum()),
    }
    return valid


# --------------------------------------------------------------------------- #
# Hypnogram / YASA sleep statistics
# --------------------------------------------------------------------------- #
def load_hypnodensity_as_hypnogram(
    hypnodensity_path: Path,
    *,
    stage_map: Mapping[str, int] = DEFAULT_STAGE_MAP,
) -> np.ndarray:
    """Load a hypnodensity CSV and convert it to a YASA integer hypnogram.

    The hypnodensity gives per-epoch stage probabilities
    (``Timestamp,Wake,N1,N2,N3,REM``). The discrete stage per epoch is the
    **argmax** column (via :func:`~infraslow.io.hypnodensity.hypnodensity_to_annotations`),
    then mapped to YASA integer codes (``Wake=0, N1=1, N2=2, N3=3, REM=4``;
    unknown labels raise rather than being silently mis-scored).

    Args:
        hypnodensity_path: Path to ``{id}_Hypnodensity.csv``.
        stage_map: Case-insensitive stage-label -> integer map.

    Returns:
        1-D ``int`` array with one stage code per 30-s epoch.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if a stage label is unrecognised.
    """
    annotations = hypnodensity_to_annotations(hypnodensity_path)
    stages = _extract_epoch_stages(annotations, stage_column="stage")
    return _stages_to_int(stages, stage_map)


def calculate_yasa_sleep_statistics(
    hypnogram: np.ndarray, epoch_length_sec: int = 30
) -> Dict[str, float]:
    """Full YASA ``sleep_statistics`` for an integer hypnogram.

    Args:
        hypnogram: Per-epoch YASA integer stage codes.
        epoch_length_sec: Seconds per epoch (sets ``sf_hyp = 1/epoch_length_sec``).

    Returns:
        Dict keyed by :data:`YASA_STAT_COLUMNS`; keys absent from the installed
        YASA build are filled with ``NaN`` so the schema is stable.

    Raises:
        ImportError: if ``yasa`` is not installed.
        ValueError: if the hypnogram is empty or has no scored sleep.
    """
    try:
        import yasa  # noqa: PLC0415 - lazy, compute-node only
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "calculate_yasa_sleep_statistics requires 'yasa'. Install it in a "
            "Slurm/interactive job (not the login node)."
        ) from exc

    hypno = np.asarray(hypnogram, dtype=int)
    if hypno.size == 0:
        raise ValueError("Empty hypnogram: no epochs to compute sleep statistics from.")
    sf_hyp = 1.0 / float(epoch_length_sec)
    stats = dict(yasa.sleep_statistics(hypno, sf_hyp))
    # Normalise to the canonical schema (missing keys -> NaN).
    return {col: float(stats.get(col, np.nan)) for col in YASA_STAT_COLUMNS}


# --------------------------------------------------------------------------- #
# Bouts + Gaussian ISFS fit (generalised from the reference notebook)
# --------------------------------------------------------------------------- #
def find_stage_bouts(
    hypnogram: np.ndarray,
    stage_codes: Sequence[int],
    *,
    epoch_sec: float = EPOCH_SEC,
    min_dur: float = MIN_BOUT_SEC,
) -> List[Tuple[float, float]]:
    """``(start, stop)`` times (s) of consecutive-stage runs of at least ``min_dur``.

    Generalises the notebook's ``nrem2_bouts`` to any set of stage codes (a run
    is a maximal block of epochs whose code is in ``stage_codes``).
    """
    codes = np.asarray(hypnogram)
    wanted = set(int(c) for c in stage_codes)
    in_stage = np.fromiter((int(c) in wanted for c in codes), dtype=bool, count=codes.size)

    bouts: List[Tuple[float, float]] = []
    i, n = 0, codes.size
    while i < n:
        if in_stage[i]:
            j = i
            while j < n and in_stage[j]:
                j += 1
            if (j - i) * epoch_sec >= min_dur:
                bouts.append((i * epoch_sec, j * epoch_sec))
            i = j
        else:
            i += 1
    return bouts


def calculate_stage_bouts(
    hypnogram: np.ndarray,
    stage: str,
    *,
    epoch_length_sec: float = EPOCH_SEC,
    min_bout_sec: float = MIN_BOUT_SEC,
) -> Dict[str, float]:
    """Consolidated-bout burden for ``stage`` (``N2``/``N3``/``NREM``).

    A *bout* is a consecutive run of the stage group's epochs lasting at least
    ``min_bout_sec``. Returns the total bout duration (minutes) and bout count.
    This is deliberately distinct from the YASA per-stage total (which counts
    every epoch, however fragmented).

    Args:
        hypnogram: Per-epoch YASA integer stage codes.
        stage: One of :data:`INFRASLOW_STAGES`.
        epoch_length_sec: Seconds per epoch.
        min_bout_sec: Minimum bout length to count (s).

    Returns:
        ``{"bouts_min": float, "n_bouts": int}``.
    """
    if stage not in STAGE_GROUP_CODES:
        raise KeyError(f"Unknown stage group {stage!r}; expected one of {INFRASLOW_STAGES}.")
    bouts = find_stage_bouts(
        hypnogram, STAGE_GROUP_CODES[stage], epoch_sec=epoch_length_sec, min_dur=min_bout_sec
    )
    total_sec = float(sum(stop - start for start, stop in bouts))
    return {"bouts_min": total_sec / 60.0, "n_bouts": len(bouts)}


def _gaussian(f: np.ndarray, amp: float, mu: float, sd: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((f - mu) / sd) ** 2)


def _fit_isfs(
    freqs: np.ndarray,
    corrected: np.ndarray,
    *,
    infraslow_band: Tuple[float, float] = INFRASLOW_BAND,
    baseline_band: Tuple[float, float] = BASELINE_BAND,
) -> Dict[str, float]:
    """Fit the Fig C-i Gaussian to a baseline-corrected relative spectrum.

    Mirrors the reference notebook's ``fit_isfs``: fits a Gaussian over the
    ``[infraslow_lo, baseline_lo)`` band, returning the peak frequency ``mu``,
    the +/-1 SD bandwidth (``2*sd``), the area under the curve over +/-1 SD, the
    empirical argmax peak, and a detection flag (peak amplitude above 1.5x the
    baseline-band noise SD).

    Raises:
        RuntimeError: if the fit band is empty or the curve fit does not converge.
    """
    base_m = (freqs >= baseline_band[0]) & (freqs <= baseline_band[1])
    fit_m = (freqs >= infraslow_band[0]) & (freqs < baseline_band[0])
    if not fit_m.any() or not base_m.any():
        raise RuntimeError("Frequency grid has no fit/baseline band bins.")

    ff, yy = freqs[fit_m], corrected[fit_m]
    empirical_peak = float(ff[int(np.argmax(yy))])
    p0 = [max(float(yy.max()), 1e-9), ff[int(np.argmax(yy))], 0.01]
    try:
        popt, _ = curve_fit(
            _gaussian, ff, yy, p0=p0,
            bounds=([0, infraslow_band[0], 1e-3], [np.inf, baseline_band[0], 0.05]),
            maxfev=10000,
        )
    except Exception as exc:  # noqa: BLE001 - curve_fit raises several types
        raise RuntimeError(f"Gaussian ISFS fit failed: {exc}") from exc

    amp, mu, sd = (float(p) for p in popt)
    lo, hi = mu - sd, mu + sd
    bandwidth = hi - lo
    f_auc = np.linspace(lo, hi, 400)
    auc = float(_trapz(_gaussian(f_auc, amp, mu, sd), f_auc))
    threshold = 1.5 * float(corrected[base_m].std())
    return {
        "mu": mu,
        "empirical_peak": empirical_peak,
        "bandwidth": float(bandwidth),
        "auc": auc,
        "threshold": threshold,
        "detected": bool(amp > threshold),
    }


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and standard error of the mean (``NaN`` mean if empty, ``NaN`` sem if <2)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size >= 2 else float("nan")
    return mean, sem


def _nan_stage_metrics() -> Dict[str, float]:
    """All-NaN infraslow metrics for a stage (``detected`` -> ``False``)."""
    out: Dict[str, float] = {key: float("nan") for key in _STAGE_METRIC_KEYS if key != "detected"}
    out["detected"] = False
    return out


# --------------------------------------------------------------------------- #
# Per-stage infraslow analysis
# --------------------------------------------------------------------------- #
def calculate_stage_infraslow(
    hypnogram: np.ndarray,
    sigma_db: np.ndarray,
    t_env: np.ndarray,
    spindle_peaks: np.ndarray,
    stage: str,
    *,
    sf_env: float = SF_ENV,
    epoch_sec: float = EPOCH_SEC,
    infraslow_band: Tuple[float, float] = INFRASLOW_BAND,
    baseline_band: Tuple[float, float] = BASELINE_BAND,
    window_sec: float = WINDOW_SEC,
    min_bout_sec: float = MIN_BOUT_SEC,
    require_spindle: bool = True,
    spectrum_out: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Infraslow sigma-power metrics for one stage group of one subject.

    Reproduces the reference notebook's per-subject Fig C-i analysis, generalised
    to any stage group:

    1. find consecutive stage-``stage`` bouts >= ``min_bout_sec``;
    2. (if ``require_spindle``) keep only bouts containing >= 1 detected spindle;
    3. slice the dB sigma-power envelope to each bout and keep segments long
       enough for the Welch window;
    4. per bout: Welch infraslow spectrum -> unit-band-area relative spectrum ->
       baseline-corrected spectrum;
    5. average the corrected spectra across bouts and fit the Gaussian ISFS for
       ``bandwidth_hz``, ``auc`` and ``detected``; average the per-bout empirical
       and fitted peaks for ``peak_freq`` / ``fit_peak_freq`` (with ``_sem``).

    Args:
        hypnogram: Per-epoch YASA integer stage codes.
        sigma_db: dB sigma-power envelope at ``sf_env`` (see
            :func:`~infraslow.processing.infraslow.sigma_power_envelope`).
        t_env: Bin-centre times (s) aligned with ``sigma_db``.
        spindle_peaks: Detected spindle peak times (s); may be empty.
        stage: One of :data:`INFRASLOW_STAGES`.
        require_spindle: If ``True`` (default, matches the reference), a bout must
            contain >= 1 spindle peak to be used.
        spectrum_out: If given, populated in place with the *empirical* spectrum
            underlying the fit -- ``{"freqs": ..., "corr_mean": ...}``, the
            bout-averaged baseline-corrected relative spectrum -- when one was
            computed. Left untouched (no keys added) if no usable bout data
            exists. Kept out of the returned metrics dict since it holds arrays,
            not scalars, and callers that only want the CSV row can ignore it.

    Returns:
        Dict with keys :data:`_STAGE_METRIC_KEYS` (``peak_freq``, ``peak_freq_sem``,
        ``fit_peak_freq``, ``fit_peak_freq_sem``, ``bandwidth_hz``, ``auc``,
        ``detected``). The ``{stage}_bouts`` burden is computed separately by
        :func:`calculate_stage_bouts`. Never raises: on any shortfall it returns
        all-``NaN`` metrics with ``detected=False``.
    """
    if stage not in STAGE_GROUP_CODES:
        raise KeyError(f"Unknown stage group {stage!r}; expected one of {INFRASLOW_STAGES}.")

    bouts = find_stage_bouts(
        hypnogram, STAGE_GROUP_CODES[stage], epoch_sec=epoch_sec, min_dur=min_bout_sec
    )
    peaks = np.asarray(spindle_peaks, dtype=float).ravel()
    if require_spindle:
        bouts = [
            (a, b) for a, b in bouts if peaks.size and np.any((peaks >= a) & (peaks < b))
        ]
    if not bouts:
        return _nan_stage_metrics()

    # Slice the envelope to each bout; keep only segments long enough for Welch.
    min_len = int(round(window_sec * sf_env))
    segments: List[np.ndarray] = []
    for a, b in bouts:
        seg = np.asarray(sigma_db[(t_env >= a) & (t_env < b)], dtype=float)
        if seg.size >= min_len:
            segments.append(seg)
    if not segments:
        return _nan_stage_metrics()

    # Per-bout relative + baseline-corrected spectra on a shared frequency grid.
    corrected_specs: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None
    band_m = base_m = None
    for seg in segments:
        spec = infraslow_spectrum(seg, sf_env, band=infraslow_band, window_sec=window_sec)
        if freqs is None:
            freqs = spec.freqs
            band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
            base_m = (freqs >= baseline_band[0]) & (freqs <= baseline_band[1])
        denom = float(_trapz(spec.psd[band_m], freqs[band_m]))
        if not np.isfinite(denom) or denom <= 0:
            continue
        rel = spec.psd / denom
        corrected_specs.append(rel - float(rel[base_m].mean()))

    metrics = _nan_stage_metrics()
    if not corrected_specs or freqs is None:
        return metrics

    # Empirical (argmax) and fitted (Gaussian mu) peaks, per bout, for mean +/- SEM.
    fit_m = (freqs >= infraslow_band[0]) & (freqs < baseline_band[0])
    ff = freqs[fit_m]
    empirical_peaks: List[float] = []
    fitted_peaks: List[float] = []
    for corr in corrected_specs:
        empirical_peaks.append(float(ff[int(np.argmax(corr[fit_m]))]))
        try:
            fitted_peaks.append(
                _fit_isfs(freqs, corr, infraslow_band=infraslow_band,
                          baseline_band=baseline_band)["mu"]
            )
        except RuntimeError:
            continue  # a single bout's fit may fail; it just drops out of the SEM

    peak_freq, peak_freq_sem = _mean_sem(empirical_peaks)
    fit_peak_freq, fit_peak_freq_sem = _mean_sem(fitted_peaks)

    # Bandwidth / AUC / detection from the fit of the bout-averaged spectrum
    # (matches the reference notebook; more stable than a single-bout fit).
    corr_mean = np.mean(np.vstack(corrected_specs), axis=0)
    if spectrum_out is not None:
        spectrum_out["freqs"] = freqs
        spectrum_out["corr_mean"] = corr_mean
    bandwidth = auc = float("nan")
    detected = False
    try:
        agg = _fit_isfs(freqs, corr_mean, infraslow_band=infraslow_band,
                        baseline_band=baseline_band)
        bandwidth, auc, detected = agg["bandwidth"], agg["auc"], agg["detected"]
        if not np.isfinite(fit_peak_freq):  # fall back if every per-bout fit failed
            fit_peak_freq = agg["mu"]
    except RuntimeError:
        pass

    metrics.update(
        peak_freq=peak_freq,
        peak_freq_sem=peak_freq_sem,
        fit_peak_freq=fit_peak_freq,
        fit_peak_freq_sem=fit_peak_freq_sem,
        bandwidth_hz=bandwidth,
        auc=auc,
        detected=detected,
    )
    return metrics


def calculate_subject_infraslow_metrics(
    subject_id: str,
    hypnogram: np.ndarray,
    *,
    sf: float = SF,
    channel: str = CHANNEL,
    require_spindle: bool = True,
    loader: Optional[BioserenityPSGLoader] = None,
    spectra_out: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, float]:
    """Infraslow metrics for all stage groups of one subject.

    Builds (or reuses) a :class:`~infraslow.io.psg_loader.BioserenityPSGLoader`
    for ``channel``, computes the dB sigma-power envelope once, detects spindles
    once (restricted to N2+N3), then runs :func:`calculate_stage_infraslow` for
    each of :data:`INFRASLOW_STAGES`.

    Args:
        subject_id: EDF stem / subject id (used to locate the recording on ``$OAK``).
        hypnogram: Per-epoch YASA integer stage codes (for bout finding).
        sf: Common resample rate (Hz).
        channel: Canonical EEG channel for the sigma-power envelope.
        require_spindle: Passed through to :func:`calculate_stage_infraslow`.
        loader: Optional pre-loaded loader (skips construction/loading).
        spectra_out: If given, populated in place with ``{stage: {"freqs":
            ..., "corr_mean": ...}}`` for every stage that had usable bout
            data (see :func:`calculate_stage_infraslow`'s ``spectrum_out``).

    Returns:
        Flat dict ``{f"{stage}_{key}": value}`` for every stage/metric, plus
        ``{stage}_bouts`` in minutes. Never contains ``%{stage}_bouts`` -- that
        needs TST and is added by :func:`calculate_subject_all_metrics`.

    Raises:
        Any loader error (missing/unreadable EDF, unresolved channel) propagates,
        so the caller can record it as a subject-level failure.
    """
    if loader is None:
        loader = BioserenityPSGLoader(
            subject_id=subject_id, sf=sf, requested_channels=[channel]
        ).load()

    data = np.asarray(loader.get_channel(channel), dtype=float)
    sf_actual = float(loader.sf)
    t_env, sigma_db = sigma_power_envelope(
        data, sf_actual, sigma_band=SIGMA_BAND, sf_env=SF_ENV, smooth_sec=1.0, to_db=True
    )

    # Detect spindles once (N2+N3); reuse the peaks for every stage's bout filter.
    spindle_peaks = np.empty(0, dtype=float)
    if require_spindle:
        try:
            result = spindles_detect(loader, ch_names=channel, include=SPINDLE_INCLUDE)
            if result is not None:
                spindle_peaks = result.summary()["Peak"].to_numpy(dtype=float)
        except Exception as exc:  # noqa: BLE001 - detection failure -> no spindle filter data
            logger.warning("Spindle detection failed for %s: %s", subject_id, exc)

    out: Dict[str, float] = {}
    for stage in INFRASLOW_STAGES:
        stage_spectrum: Dict[str, np.ndarray] = {}
        try:
            stage_metrics = calculate_stage_infraslow(
                hypnogram, sigma_db, t_env, spindle_peaks, stage,
                require_spindle=require_spindle,
                spectrum_out=stage_spectrum,
            )
        except Exception as exc:  # noqa: BLE001 - one stage must not sink the subject
            logger.warning("Infraslow failed for %s stage %s: %s", subject_id, stage, exc)
            stage_metrics = _nan_stage_metrics()
            stage_spectrum = {}
        for key, value in stage_metrics.items():
            out[f"{stage}_{key}"] = value
        if spectra_out is not None and stage_spectrum:
            spectra_out[stage] = stage_spectrum
    return out


# --------------------------------------------------------------------------- #
# v2 pipeline: per-channel, per-stage spectra + bout/spindle detail (no fit)
# --------------------------------------------------------------------------- #
#: Every array key written into an npz for one channel/stage (see
#: :func:`calculate_stage_events`); npz-key suffix each maps to.
STAGE_EVENT_FIELDS: Dict[str, str] = {
    "freqs": "spectra__freqs",
    "corr_mean": "spectra__corr_mean",
    "spindle_start": "spindles__start",
    "spindle_stop": "spindles__stop",
    "spindle_peak": "spindles__peak",
    "bout_start": "bouts__start",
    "bout_stop": "bouts__stop",
    "bout_n_spindles": "bouts__n_spindles",
}
#: dtype for each field above -- float64 throughout except the spindle count.
_STAGE_EVENT_DTYPES: Dict[str, Any] = {
    "freqs": np.float64, "corr_mean": np.float64,
    "spindle_start": np.float64, "spindle_stop": np.float64, "spindle_peak": np.float64,
    "bout_start": np.float64, "bout_stop": np.float64, "bout_n_spindles": np.int64,
}


def empty_stage_events() -> Dict[str, np.ndarray]:
    """All-empty, correctly-typed arrays for one channel/stage (see :data:`STAGE_EVENT_FIELDS`)."""
    return {key: np.empty(0, dtype=dtype) for key, dtype in _STAGE_EVENT_DTYPES.items()}


def _issue(
    *, channel: str = "", stage: str = "", error_type: str = "", error_message: str = "",
    include_traceback: bool = False,
) -> Dict[str, str]:
    """One structured, sub-subject-level failure record (channel/stage granularity)."""
    return {
        "channel": channel,
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
        "traceback": _traceback.format_exc() if include_traceback else "",
    }


def calculate_stage_events(
    hypnogram: np.ndarray,
    spindle_starts: np.ndarray,
    spindle_stops: np.ndarray,
    spindle_peaks: np.ndarray,
    stage: str,
    *,
    sf_env: float = SF_ENV,
    epoch_sec: float = EPOCH_SEC,
    infraslow_band: Tuple[float, float] = INFRASLOW_BAND,
    baseline_band: Tuple[float, float] = BASELINE_BAND,
    window_sec: float = WINDOW_SEC,
    min_bout_sec: float = MIN_BOUT_SEC,
    smooth_sec: float = 10.0,
    require_spindle: bool = True,
) -> Dict[str, np.ndarray]:
    """Bout-averaged spindle-event infraslow spectrum + bout/spindle detail for one stage.

    Implements steps 7-10 of the v2 pipeline for one channel's already-detected
    spindles and one stage group:

    1. Find consecutive ``stage`` bouts (:func:`find_stage_bouts`) of at least
       ``min_bout_sec`` -- N2/N3 bouts split on any stage change; a combined NREM
       bout is not split by N1<->N2<->N3 transitions (only by Wake/REM/gaps),
       since :data:`STAGE_GROUP_CODES` treats NREM as one merged code set.
    2. Assign each already-detected spindle to a bout by its **peak** time
       (``bout_start <= peak < bout_stop``); a spindle assigned to no bout is
       dropped, and (bouts being disjoint) no spindle is assigned twice.
    3. Keep only bouts with >= 1 assigned spindle (when ``require_spindle``).
    4. Per kept bout, build the *spindle-event time series* from its assigned
       spindles' peak times (:func:`~infraslow.processing.infraslow.
       spindle_rate_series`, peak times shifted to be bout-relative) and take its
       infraslow spectrum (:func:`~infraslow.processing.infraslow.infraslow_spectrum`)
       -- the detector-derived view, not the continuous-EEG sigma envelope.
    5. Normalise to unit band-area and baseline-correct (subtract the
       ``baseline_band`` mean), matching the reference notebook's spectrum
       post-processing; average the corrected spectra across bouts.

    Args:
        hypnogram: Per-epoch YASA integer stage codes.
        spindle_starts, spindle_stops, spindle_peaks: This channel's *entire*
            detected-spindle set (all stages -- see :func:`calculate_subject_channel_events`,
            which detects once per channel, not once per stage). Arrays, not a
            DataFrame, so this function has no pandas dependency.
        stage: One of :data:`NPZ_STAGES` (any key of :data:`STAGE_GROUP_CODES`).
        smooth_sec: Gaussian smoothing width (s) for the spindle-rate series
            (passed to ``spindle_rate_series``).
        require_spindle: If ``True`` (default; the pipeline's hard requirement),
            a bout must contain >= 1 assigned spindle to be kept.

    Returns:
        Dict with :data:`STAGE_EVENT_FIELDS` keys: ``freqs``, ``corr_mean``
        (bout-averaged baseline-corrected relative spectrum), ``bout_start``,
        ``bout_stop``, ``bout_n_spindles`` (one entry per qualifying bout), and
        ``spindle_start``, ``spindle_stop``, ``spindle_peak`` (every spindle
        assigned to a qualifying bout). All empty (never NaN-filled -- absence is
        meaningful) when no bout qualifies.
    """
    if stage not in STAGE_GROUP_CODES:
        raise KeyError(f"Unknown stage group {stage!r}; expected one of {tuple(STAGE_GROUP_CODES)}.")

    bouts = find_stage_bouts(
        hypnogram, STAGE_GROUP_CODES[stage], epoch_sec=epoch_sec, min_dur=min_bout_sec
    )
    peaks = np.asarray(spindle_peaks, dtype=float).ravel()
    starts = np.asarray(spindle_starts, dtype=float).ravel()
    stops = np.asarray(spindle_stops, dtype=float).ravel()

    def _assigned(a: float, b: float) -> np.ndarray:
        """Boolean mask of spindles whose peak falls in [a, b) -- this bout's own."""
        return (peaks >= a) & (peaks < b) if peaks.size else np.zeros(0, dtype=bool)

    if require_spindle:
        bouts = [(a, b) for a, b in bouts if _assigned(a, b).any()]
    if not bouts:
        return empty_stage_events()

    bout_start: List[float] = []
    bout_stop: List[float] = []
    bout_n_spindles: List[int] = []
    spindle_start_chunks: List[np.ndarray] = []
    spindle_stop_chunks: List[np.ndarray] = []
    spindle_peak_chunks: List[np.ndarray] = []
    corrected_specs: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None
    band_m = base_m = None

    for a, b in bouts:
        mask = _assigned(a, b)
        n = int(mask.sum())
        bout_start.append(a)
        bout_stop.append(b)
        bout_n_spindles.append(n)
        if not n:
            continue
        spindle_start_chunks.append(starts[mask])
        spindle_stop_chunks.append(stops[mask])
        spindle_peak_chunks.append(peaks[mask])

        rel_peaks = peaks[mask] - a  # bout-relative time, matching spindle_rate_series's [0, duration)
        duration = b - a
        try:
            _, rate = spindle_rate_series(
                rel_peaks, duration_sec=duration, sf_env=sf_env, smooth_sec=smooth_sec
            )
            spec = infraslow_spectrum(rate, sf_env, band=infraslow_band, window_sec=window_sec)
        except Exception:  # noqa: BLE001 - one bout's spectrum failing must not drop the others
            continue
        if freqs is None:
            freqs = spec.freqs
            band_m = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
            base_m = (freqs >= baseline_band[0]) & (freqs <= baseline_band[1])
        denom = float(_trapz(spec.psd[band_m], freqs[band_m]))
        if not np.isfinite(denom) or denom <= 0:
            continue
        rel = spec.psd / denom
        corrected = rel - float(rel[base_m].mean())
        if np.all(np.isfinite(corrected)):
            corrected_specs.append(corrected)

    out_freqs = freqs if (corrected_specs and freqs is not None) else np.empty(0)
    corr_mean = np.mean(np.vstack(corrected_specs), axis=0) if corrected_specs else np.empty(0)
    spindle_start = np.concatenate(spindle_start_chunks) if spindle_start_chunks else np.empty(0)
    spindle_stop = np.concatenate(spindle_stop_chunks) if spindle_stop_chunks else np.empty(0)
    spindle_peak = np.concatenate(spindle_peak_chunks) if spindle_peak_chunks else np.empty(0)
    return {
        "freqs": np.asarray(out_freqs, dtype=np.float64),
        "corr_mean": np.asarray(corr_mean, dtype=np.float64),
        "bout_start": np.asarray(bout_start, dtype=np.float64),
        "bout_stop": np.asarray(bout_stop, dtype=np.float64),
        "bout_n_spindles": np.asarray(bout_n_spindles, dtype=np.int64),
        "spindle_start": np.asarray(spindle_start, dtype=np.float64),
        "spindle_stop": np.asarray(spindle_stop, dtype=np.float64),
        "spindle_peak": np.asarray(spindle_peak, dtype=np.float64),
    }


def calculate_subject_channel_events(
    subject_id: str,
    hypnogram: np.ndarray,
    *,
    sf: float = SF,
    channels: Sequence[str] = CHANNELS,
    stages: Sequence[str] = NPZ_STAGES,
    require_spindle: bool = True,
    loader: Optional[BioserenityPSGLoader] = None,
    issues_out: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Per-channel, per-stage infraslow spectra + bout/spindle detail for one subject.

    Per channel: detect spindles **once**, restricted to all NREM epochs
    (``include=STAGE_GROUP_CODES["NREM"]`` = N1+N2+N3, matching the YASA
    ``03_spindles_detection_NREM_only`` convention) -- not once per stage. Then,
    for every stage in ``stages``, :func:`calculate_stage_events` finds that
    stage's own bouts and assigns spindles to them purely by peak-time interval
    membership (not by which stage YASA itself scored the spindle epoch as), so
    the same detected-spindle set feeds the N2, N3, and NREM analyses.

    Builds (or reuses) a :class:`~infraslow.io.psg_loader.BioserenityPSGLoader`
    for every channel in ``channels``. If a channel cannot be resolved (missing
    from the EDF) or its spindle detection fails, that failure is recorded to
    ``issues_out`` (if given) and logged, but every stage for that channel still
    gets an entry (all-empty arrays) -- one bad channel never sinks the other
    channels or the subject, and every channel key stays present for a
    predictable npz schema (see :func:`~infraslow.processing.all_metrics.
    calculate_stage_events`'s empty-array convention).

    Args:
        subject_id: EDF stem / subject id (used to locate the recording on ``$OAK``).
        hypnogram: Per-epoch YASA integer stage codes (for bout finding).
        sf: Common resample rate (Hz).
        channels: Canonical EEG channels to analyse (default :data:`CHANNELS`).
        stages: Stage groups to analyse per channel (default :data:`NPZ_STAGES`).
        require_spindle: If ``True`` (default; the pipeline's hard requirement),
            a bout must contain >= 1 assigned spindle to be kept.
        loader: Optional pre-loaded loader (skips construction/loading).
        issues_out: If given, appended in place with one dict per channel- or
            stage-level failure (see :func:`_issue`) -- subject-id-free, since
            the caller already knows the subject; channel/stage-level only.

    Returns:
        ``{channel: {stage: {...}}}`` -- every requested channel and stage is
        always present (see :data:`STAGE_EVENT_FIELDS`), so the caller can write
        a schema-stable npz per stage.

    Raises:
        Any loader error (missing/unreadable EDF) propagates, so the caller can
        record it as a subject-level failure.
    """
    if loader is None:
        loader = BioserenityPSGLoader(
            subject_id=subject_id, sf=sf, requested_channels=list(channels)
        ).load()

    nrem_codes = STAGE_GROUP_CODES["NREM"]  # (1, 2, 3): spindle detection scope, all channels

    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for ch in channels:
        starts = stops = peaks = np.empty(0, dtype=float)
        if require_spindle:
            try:
                result = spindles_detect(loader, ch_names=ch, include=nrem_codes)
                if result is not None:
                    summary = result.summary()
                    starts = summary["Start"].to_numpy(dtype=float)
                    stops = summary["End"].to_numpy(dtype=float)
                    peaks = summary["Peak"].to_numpy(dtype=float)
            except Exception as exc:  # noqa: BLE001 - a missing/failed channel must not sink the subject
                logger.warning("Channel %s spindle detection failed for %s: %s", ch, subject_id, exc)
                if issues_out is not None:
                    issues_out.append(
                        _issue(channel=ch, error_type=type(exc).__name__, error_message=str(exc),
                               include_traceback=True)
                    )

        out[ch] = {}
        for stage in stages:
            try:
                out[ch][stage] = calculate_stage_events(
                    hypnogram, starts, stops, peaks, stage, require_spindle=require_spindle,
                )
            except Exception as exc:  # noqa: BLE001 - one stage must not sink the channel/subject
                logger.warning(
                    "Events failed for %s channel=%s stage=%s: %s", subject_id, ch, stage, exc
                )
                if issues_out is not None:
                    issues_out.append(
                        _issue(channel=ch, stage=stage, error_type=type(exc).__name__,
                               error_message=str(exc), include_traceback=True)
                    )
                out[ch][stage] = empty_stage_events()
    return out


#: Output column order for the v2 (metadata + YASA stats only) CSV -- infraslow
#: detail now lives entirely in the per-channel/per-stage npz files (see
#: calculate_subject_channel_events), not in this row.
def metadata_row_columns() -> List[str]:
    """Column order for the v2 metadata CSV: metadata + full YASA sleep stats."""
    return list(METADATA_COLUMNS) + list(YASA_STAT_COLUMNS)


def calculate_subject_metadata_row(
    subject_row: Mapping[str, Any], hypnogram: np.ndarray
) -> Dict[str, float]:
    """Metadata + YASA sleep statistics for one subject (v2 row; no infraslow scalars).

    Args:
        subject_row: A standardised metadata row (``ID, Age, Gender, BMI``).
        hypnogram: Per-epoch YASA integer stage codes (see
            :func:`load_hypnodensity_as_hypnogram`).

    Returns:
        Flat dict keyed by :func:`metadata_row_columns`.
    """
    subject_id = str(subject_row["ID"]).strip()
    row: Dict[str, float] = {col: subject_row.get(col, np.nan) for col in METADATA_COLUMNS}
    row["ID"] = subject_id
    row.update(calculate_yasa_sleep_statistics(hypnogram, epoch_length_sec=int(EPOCH_SEC)))
    return row


def calculate_subject_v2(
    subject_row: Mapping[str, Any],
    edf_path: Path,
    hypnodensity_path: Path,
    *,
    sf: float = SF,
    channels: Sequence[str] = CHANNELS,
    stages: Sequence[str] = NPZ_STAGES,
    require_spindle: bool = True,
    issues_out: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Metadata/YASA row + per-channel/per-stage infraslow events for one subject.

    The v2 counterpart of :func:`calculate_subject_all_metrics`: the row carries
    only metadata + YASA sleep statistics (:func:`metadata_row_columns`); all
    infraslow spectra/bout/spindle detail -- now per :data:`CHANNELS` channel and
    per :data:`NPZ_STAGES` stage -- is returned separately for the caller to
    persist as npz (not flattened into the row), matching the columns actually
    requested for the metadata CSV.

    Args:
        subject_row: A standardised metadata row (``ID, Age, Gender, BMI``).
        edf_path: Path to the subject's EDF (validated to exist).
        hypnodensity_path: Path to the subject's hypnodensity CSV.
        sf, channels, stages, require_spindle: Analysis parameters.
        issues_out: Passed through to :func:`calculate_subject_channel_events`
            -- appended with any channel/stage-level failures.

    Returns:
        ``(row, channel_events)`` -- see :func:`calculate_subject_metadata_row`
        and :func:`calculate_subject_channel_events`.

    Raises:
        FileNotFoundError: if the EDF or hypnodensity file is missing.
        Any load/compute error from the EDF or YASA propagates to the caller so it
        can be recorded as a subject-level failure.
    """
    edf_path = Path(edf_path)
    hypnodensity_path = Path(hypnodensity_path)
    if not edf_path.is_file():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    if not hypnodensity_path.is_file():
        raise FileNotFoundError(f"Hypnodensity file not found: {hypnodensity_path}")

    subject_id = str(subject_row["ID"]).strip()
    hypnogram = load_hypnodensity_as_hypnogram(hypnodensity_path)
    row = calculate_subject_metadata_row(subject_row, hypnogram)
    events = calculate_subject_channel_events(
        subject_id, hypnogram, sf=sf, channels=channels, stages=stages,
        require_spindle=require_spindle, issues_out=issues_out,
    )
    return row, events


# --------------------------------------------------------------------------- #
# Full per-subject assembly + cohort runner
# --------------------------------------------------------------------------- #
def calculate_subject_all_metrics(
    subject_row: pd.Series,
    edf_path: Path,
    hypnodensity_path: Path,
    *,
    sf: float = SF,
    channel: str = CHANNEL,
    require_spindle: bool = True,
    spectra_out: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, float]:
    """Metadata + YASA sleep statistics + infraslow metrics for one subject.

    Args:
        subject_row: A standardised metadata row (``ID, Age, Gender, BMI``).
        edf_path: Path to the subject's EDF (validated to exist).
        hypnodensity_path: Path to the subject's hypnodensity CSV.
        sf, channel, require_spindle: Analysis parameters (see the module constants).
        spectra_out: If given, populated in place with this subject's empirical
            spectra -- see :func:`calculate_subject_infraslow_metrics`. Lets a
            caller persist the true (pre-fit) grand-average curve alongside the
            scalar metrics without it polluting the CSV row.

    Returns:
        A flat dict keyed by :func:`all_metric_columns` (one row of the final table).

    Raises:
        FileNotFoundError: if the EDF or hypnodensity file is missing.
        Any load/compute error from the EDF or YASA propagates to the caller so it
        can be recorded as a subject-level failure. Individual infraslow stages
        that fail degrade to ``NaN`` rather than raising.
    """
    subject_id = str(subject_row["ID"]).strip()
    edf_path = Path(edf_path)
    hypnodensity_path = Path(hypnodensity_path)
    if not edf_path.is_file():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    if not hypnodensity_path.is_file():
        raise FileNotFoundError(f"Hypnodensity file not found: {hypnodensity_path}")

    row: Dict[str, float] = {col: subject_row.get(col, np.nan) for col in METADATA_COLUMNS}
    row["ID"] = subject_id

    # YASA sleep statistics from the argmax hypnogram.
    hypnogram = load_hypnodensity_as_hypnogram(hypnodensity_path)
    row.update(calculate_yasa_sleep_statistics(hypnogram, epoch_length_sec=int(EPOCH_SEC)))

    # Infraslow metrics (reads the EDF via the loader).
    infra = calculate_subject_infraslow_metrics(
        subject_id, hypnogram, sf=sf, channel=channel, require_spindle=require_spindle,
        spectra_out=spectra_out,
    )
    row.update(infra)

    # Consolidated-bout burden + its percentage of TST, per stage.
    tst = row.get("TST", np.nan)
    for stage in INFRASLOW_STAGES:
        bouts = calculate_stage_bouts(hypnogram, stage)
        row[f"{stage}_bouts"] = bouts["bouts_min"]
        row[f"%{stage}_bouts"] = (
            100.0 * bouts["bouts_min"] / tst
            if isinstance(tst, (int, float)) and np.isfinite(tst) and tst > 0
            else np.nan
        )
    return row


def _subject_paths(
    subject_id: str,
    edf_dir: Path,
    hypnodensity_dir: Path,
    *,
    edf_suffix: str = ".edf",
    hypnodensity_suffix: str = DEFAULT_HYPNODENSITY_SUFFIX,
) -> Tuple[Path, Path]:
    """Build the ``(edf_path, hypnodensity_path)`` for a subject id."""
    return (
        Path(edf_dir) / f"{subject_id}{edf_suffix}",
        Path(hypnodensity_dir) / f"{subject_id}{hypnodensity_suffix}",
    )


def run_all_subjects(
    valid_metadata: pd.DataFrame,
    edf_dir: Path,
    hypnodensity_dir: Path,
    *,
    sf: float = SF,
    channel: str = CHANNEL,
    require_spindle: bool = True,
    id_column: str = "ID",
    show_progress: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full per-subject pipeline over every valid subject.

    Each subject is processed inside a ``try/except`` so a single failure is
    recorded (``ID, error_type, error_message``) and skipped rather than aborting
    the run.

    Args:
        valid_metadata: Rows for subjects with both files present (see
            :func:`find_valid_bioserenity_subjects`).
        edf_dir, hypnodensity_dir: Input directories.
        sf, channel, require_spindle: Analysis parameters.
        id_column: Subject-id column in ``valid_metadata``.
        show_progress: Show a progress bar/percentage.
        verbose: Print a one-line status per subject.

    Returns:
        ``(results_df, failed_df)`` -- ``results_df`` has one row per successfully
        processed subject with columns :func:`all_metric_columns`; ``failed_df``
        has columns ``["ID", "error_type", "error_message"]``.
    """
    results: List[Dict[str, float]] = []
    failures: List[Dict[str, str]] = []

    rows = [row for _, row in valid_metadata.iterrows()]
    for row in progress_iter(rows, len(rows), enabled=show_progress, desc="Subjects"):
        subject_id = str(row[id_column]).strip()
        edf_path, hypnodensity_path = _subject_paths(subject_id, edf_dir, hypnodensity_dir)
        try:
            metrics = calculate_subject_all_metrics(
                row, edf_path, hypnodensity_path,
                sf=sf, channel=channel, require_spindle=require_spindle,
            )
            results.append(metrics)
            if verbose:
                logger.info("OK   %s", subject_id)
        except Exception as exc:  # noqa: BLE001 - isolate per-subject failures
            failures.append(
                {
                    "ID": subject_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            if verbose:
                logger.warning("FAIL %s: %s: %s", subject_id, type(exc).__name__, exc)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Guarantee the full, ordered schema even if a column never got populated.
        results_df = results_df.reindex(columns=all_metric_columns())
    else:
        results_df = pd.DataFrame(columns=all_metric_columns())
    failed_df = pd.DataFrame(failures, columns=["ID", "error_type", "error_message"])
    return results_df, failed_df


__all__ = [
    "SF", "CHANNEL", "CHANNELS", "SIGMA_BAND", "INFRASLOW_BAND", "BASELINE_BAND",
    "MIN_BOUT_SEC", "WINDOW_SEC", "EPOCH_SEC", "SF_ENV",
    "STAGE_GROUP_CODES", "INFRASLOW_STAGES", "NPZ_STAGES", "SPINDLE_INCLUDE",
    "METADATA_COLUMNS", "YASA_STAT_COLUMNS", "all_metric_columns",
    "load_bioserenity_metadata", "combine_bioserenity_metadata",
    "find_valid_bioserenity_subjects",
    "load_hypnodensity_as_hypnogram", "calculate_yasa_sleep_statistics",
    "find_stage_bouts", "calculate_stage_bouts",
    "calculate_stage_infraslow", "calculate_subject_infraslow_metrics",
    "calculate_subject_all_metrics", "run_all_subjects",
    "calculate_stage_events", "calculate_subject_channel_events", "STAGE_EVENT_FIELDS",
    "empty_stage_events",
    "metadata_row_columns", "calculate_subject_metadata_row", "calculate_subject_v2",
]
