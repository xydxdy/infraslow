"""Per-subject sleep + infraslow metrics pipeline for the Bioserenity cohort.

This module powers ``src/run_all_metrics.py``, which runs it over *every*
subject that has all three required inputs on ``$OAK``:

* metadata row (``ID, Age, Gender, BMI``) from *either* of two source CSVs --
  ``Morpheus_Data_All5.csv`` and ``bioserenity_metadata3.csv`` -- combined by
  :func:`combine_bioserenity_metadata`,
* an EDF signal file ``$OAK/psg/Bioserenity/edf/{id}.edf``,
* a hypnodensity CSV ``$OAK/psg/Bioserenity/Sleep_Staging/{id}_Hypnodensity.csv``.

For each valid subject, :func:`calculate_features` produces two things:

1. A **metadata + YASA row** (``ID, Age, Gender, BMI`` plus the full
   :func:`yasa.sleep_statistics` output, computed from the argmax hypnogram at
   30-s epochs) -- see :func:`calculate_sleep_metadata`.
2. Per-``CHANNELS`` EEG channel and per-``NPZ_STAGES`` stage group, the
   infraslow (~0.02 Hz) sigma-power oscillation spectrum plus the underlying
   spindle/bout detail -- see :func:`calculate_channel_events`, written
   to disk as one ``.npz`` per subject per stage by the caller.

Anything that cannot be computed becomes ``numpy.nan``/empty arrays rather than
raising, so one bad channel/stage never sinks a subject and one bad subject
never sinks the run.

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

from ..io.hypnodensity import (
    DEFAULT_HYPNODENSITY_SUFFIX,
    hypnodensity_to_annotations,
)
from ..io.psg_loader import BioserenityPSGLoader
from ..io.utils import list_dir_filenames, progress_iter
from .spindle import (
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
    power_envelope,
)

logger = logging.getLogger(__name__)

# ``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2.0; use whichever exists.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# Analysis constants (kept identical to demo_infraslow_yasa_average.ipynb)
# --------------------------------------------------------------------------- #
SF: float = 128.0                       # common resample rate (Hz) per subject
SIGMA_BAND: Tuple[float, float] = (10.0, 16.0)  # sigma (spindle) power band
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
# Stage groups written to the per-subject, per-channel npz files (see
# calculate_channel_events). Same stage groups as INFRASLOW_STAGES.
NPZ_STAGES: Tuple[str, ...] = INFRASLOW_STAGES
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
# Bout finding
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


# --------------------------------------------------------------------------- #
# Per-channel, per-stage spectra + bout/spindle detail (no fit)
# --------------------------------------------------------------------------- #
#: Every array key written into an npz for one channel/stage (see
#: :func:`calculate_stage_events`); npz-key suffix each maps to.
STAGE_EVENT_FIELDS: Dict[str, str] = {
    "freqs": "spectra__freqs",
    "raw_mean": "spectra__raw_mean",
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
    "freqs": np.float64, "raw_mean": np.float64, "corr_mean": np.float64,
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
    t_env: np.ndarray,
    sigma_db: np.ndarray,
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
    require_spindle: bool = True,
) -> Dict[str, np.ndarray]:
    """Bout-averaged sigma-power infraslow spectrum + bout/spindle detail for one stage.

    Implements steps 7-10 of the v2 pipeline for one channel's sigma-power
    envelope and already-detected spindles, for one stage group:

    1. Find consecutive ``stage`` bouts (:func:`find_stage_bouts`) of at least
       ``min_bout_sec`` -- N2/N3 bouts split on any stage change; a combined NREM
       bout is not split by N1<->N2<->N3 transitions (only by Wake/REM/gaps),
       since :data:`STAGE_GROUP_CODES` treats NREM as one merged code set.
    2. Assign each already-detected spindle to a bout by its **peak** time
       (``bout_start <= peak < bout_stop``); a spindle assigned to no bout is
       dropped, and (bouts being disjoint) no spindle is assigned twice.
    3. Keep only bouts with >= 1 assigned spindle (when ``require_spindle``).
    4. Per kept bout, slice the channel's dB sigma-power envelope
       (``t_env``/``sigma_db``, see :func:`~infraslow.processing.infraslow.
       power_envelope`) to the bout window and take its infraslow spectrum
       (:func:`~infraslow.processing.infraslow.infraslow_spectrum`) -- the
       continuous-EEG view, matching ``demo_infraslow_yasa_average.ipynb``.
    5. Normalise to unit band-area (the "real"/uncorrected spectrum, averaged
       across bouts into ``raw_mean``) and baseline-correct (subtract the
       ``baseline_band`` mean, matching the reference notebook's spectrum
       post-processing; averaged across bouts into ``corr_mean``).

    Args:
        hypnogram: Per-epoch YASA integer stage codes.
        t_env, sigma_db: This channel's dB sigma-power envelope and its
            bin-centre times (s), computed once per channel (see
            :func:`~infraslow.processing.infraslow.power_envelope`) and reused
            across every stage.
        spindle_starts, spindle_stops, spindle_peaks: This channel's *entire*
            detected-spindle set (all stages -- see :func:`calculate_channel_events`,
            which detects once per channel, not once per stage). Arrays, not a
            DataFrame, so this function has no pandas dependency.
        stage: One of :data:`NPZ_STAGES` (any key of :data:`STAGE_GROUP_CODES`).
        require_spindle: If ``True`` (default; the pipeline's hard requirement),
            a bout must contain >= 1 assigned spindle to be kept.

    Returns:
        Dict with :data:`STAGE_EVENT_FIELDS` keys: ``freqs``, ``raw_mean``
        (bout-averaged unit-band-area-normalised spectrum, *before* baseline
        correction -- the "real" spectrum), ``corr_mean`` (the same, minus the
        ``baseline_band`` mean -- the baseline-corrected relative spectrum),
        ``bout_start``, ``bout_stop``, ``bout_n_spindles`` (one entry per
        qualifying bout), and ``spindle_start``, ``spindle_stop``,
        ``spindle_peak`` (every spindle assigned to a qualifying bout). All
        empty (never NaN-filled -- absence is meaningful) when no bout qualifies.
    """
    if stage not in STAGE_GROUP_CODES:
        raise KeyError(f"Unknown stage group {stage!r}; expected one of {tuple(STAGE_GROUP_CODES)}.")

    bouts = find_stage_bouts(
        hypnogram, STAGE_GROUP_CODES[stage], epoch_sec=epoch_sec, min_dur=min_bout_sec
    )
    peaks = np.asarray(spindle_peaks, dtype=float).ravel()
    starts = np.asarray(spindle_starts, dtype=float).ravel()
    stops = np.asarray(spindle_stops, dtype=float).ravel()
    t_env = np.asarray(t_env, dtype=float).ravel()
    sigma_db = np.asarray(sigma_db, dtype=float).ravel()

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
    raw_specs: List[np.ndarray] = []
    corrected_specs: List[np.ndarray] = []
    freqs: Optional[np.ndarray] = None
    band_m = base_m = None
    min_len = int(round(window_sec * sf_env))

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

        seg = sigma_db[(t_env >= a) & (t_env < b)]
        if seg.size < min_len:
            continue
        try:
            spec = infraslow_spectrum(seg, sf_env, infraslow_band=infraslow_band, window_sec=window_sec)
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
        if np.all(np.isfinite(corrected)) and np.all(np.isfinite(rel)):
            raw_specs.append(rel)
            corrected_specs.append(corrected)

    out_freqs = freqs if (corrected_specs and freqs is not None) else np.empty(0)
    raw_mean = np.mean(np.vstack(raw_specs), axis=0) if raw_specs else np.empty(0)
    corr_mean = np.mean(np.vstack(corrected_specs), axis=0) if corrected_specs else np.empty(0)
    spindle_start = np.concatenate(spindle_start_chunks) if spindle_start_chunks else np.empty(0)
    spindle_stop = np.concatenate(spindle_stop_chunks) if spindle_stop_chunks else np.empty(0)
    spindle_peak = np.concatenate(spindle_peak_chunks) if spindle_peak_chunks else np.empty(0)
    return {
        "freqs": np.asarray(out_freqs, dtype=np.float64),
        "raw_mean": np.asarray(raw_mean, dtype=np.float64),
        "corr_mean": np.asarray(corr_mean, dtype=np.float64),
        "bout_start": np.asarray(bout_start, dtype=np.float64),
        "bout_stop": np.asarray(bout_stop, dtype=np.float64),
        "bout_n_spindles": np.asarray(bout_n_spindles, dtype=np.int64),
        "spindle_start": np.asarray(spindle_start, dtype=np.float64),
        "spindle_stop": np.asarray(spindle_stop, dtype=np.float64),
        "spindle_peak": np.asarray(spindle_peak, dtype=np.float64),
    }


def calculate_channel_events(
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
    ``03_spindles_detection_NREM_only`` convention) -- not once per stage --
    and compute the dB sigma-power envelope **once**
    (:func:`~infraslow.processing.infraslow.power_envelope`, matching
    ``demo_infraslow_yasa_average.ipynb``). Then, for every stage in ``stages``,
    :func:`calculate_stage_events` finds that stage's own bouts, slices the
    shared envelope to each bout for the infraslow spectrum, and assigns
    spindles to bouts purely by peak-time interval membership (not by which
    stage YASA itself scored the spindle epoch as), so the same envelope and
    detected-spindle set feed the N2, N3, and NREM analyses.

    Builds (or reuses) a :class:`~infraslow.io.psg_loader.BioserenityPSGLoader`
    for every channel in ``channels``. If a channel cannot be resolved (missing
    from the EDF), its spindle detection fails, or its sigma-power envelope
    fails, that failure is recorded to ``issues_out`` (if given) and logged, but
    every stage for that channel still gets an entry (all-empty arrays) -- one
    bad channel never sinks the other channels or the subject, and every
    channel key stays present for a predictable npz schema (see
    :func:`~infraslow.processing.subject_pipeline.calculate_stage_events`'s
    empty-array convention).

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

        # dB sigma-power envelope for this channel, computed once and reused
        # across every stage (matches demo_infraslow_yasa_average.ipynb).
        t_env = sigma_db = np.empty(0, dtype=float)
        try:
            data = np.asarray(loader.get_channel(ch), dtype=float)
            t_env, sigma_db = power_envelope(
                data, float(loader.sf), band=SIGMA_BAND, sf_env=SF_ENV, smooth_sec=1.0, to_db=True,
            )
        except Exception as exc:  # noqa: BLE001 - a missing/failed channel must not sink the subject
            logger.warning("Sigma-power envelope failed for %s channel=%s: %s", subject_id, ch, exc)
            if issues_out is not None:
                issues_out.append(
                    _issue(channel=ch, error_type=type(exc).__name__, error_message=str(exc),
                           include_traceback=True)
                )

        out[ch] = {}
        for stage in stages:
            try:
                out[ch][stage] = calculate_stage_events(
                    hypnogram, t_env, sigma_db, starts, stops, peaks, stage,
                    require_spindle=require_spindle,
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
#: calculate_channel_events), not in this row.
def metadata_row_columns() -> List[str]:
    """Column order for the v2 metadata CSV: metadata + full YASA sleep stats."""
    return list(METADATA_COLUMNS) + list(YASA_STAT_COLUMNS)


def calculate_sleep_metadata(
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


def calculate_features(
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

    The row carries only metadata + YASA sleep statistics
    (:func:`metadata_row_columns`); all infraslow spectra/bout/spindle detail --
    per :data:`CHANNELS` channel and per :data:`NPZ_STAGES` stage -- is returned
    separately for the caller to persist as npz (not flattened into the row).

    Args:
        subject_row: A standardised metadata row (``ID, Age, Gender, BMI``).
        edf_path: Path to the subject's EDF (validated to exist).
        hypnodensity_path: Path to the subject's hypnodensity CSV.
        sf, channels, stages, require_spindle: Analysis parameters.
        issues_out: Passed through to :func:`calculate_channel_events`
            -- appended with any channel/stage-level failures.

    Returns:
        ``(row, channel_events)`` -- see :func:`calculate_sleep_metadata`
        and :func:`calculate_channel_events`.

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
    row = calculate_sleep_metadata(subject_row, hypnogram)
    events = calculate_channel_events(
        subject_id, hypnogram, sf=sf, channels=channels, stages=stages,
        require_spindle=require_spindle, issues_out=issues_out,
    )
    return row, events


__all__ = [
    "SF", "CHANNELS", "SIGMA_BAND", "INFRASLOW_BAND", "BASELINE_BAND",
    "MIN_BOUT_SEC", "WINDOW_SEC", "EPOCH_SEC", "SF_ENV",
    "STAGE_GROUP_CODES", "INFRASLOW_STAGES", "NPZ_STAGES",
    "METADATA_COLUMNS", "YASA_STAT_COLUMNS",
    "load_bioserenity_metadata", "combine_bioserenity_metadata",
    "find_valid_bioserenity_subjects",
    "load_hypnodensity_as_hypnogram", "calculate_yasa_sleep_statistics",
    "find_stage_bouts",
    "calculate_stage_events", "calculate_channel_events", "STAGE_EVENT_FIELDS",
    "empty_stage_events",
    "metadata_row_columns", "calculate_sleep_metadata", "calculate_features",
]
