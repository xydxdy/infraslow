#!/usr/bin/env python
"""Per-subject preprocessing: save the raw material needed to average infraslow
spectra and ISFS phase across every subject later, without re-running the
expensive EEG steps (wavelet envelope, whole-night lowpass) each time.

Reproduces, per subject/channel, exactly what ``demo_infraslow_yasa_recheck.ipynb``
(Figures 4/5) and ``demo_infraslow_phase.ipynb`` (Figure 7) compute from a
loaded recording -- but saves the per-bout detail those notebooks only ever
hold in memory for one subject, so a later script can pool it across the whole
cohort. Two computations are stage-independent and run **once per channel**
(never per stage, never per bout):

* :func:`~infraslow.processing.infraslow.eeg_envelope` -- the whole-night
  band-power envelope.
* :func:`~infraslow.processing.infraslow.isfs_lowpass` -- the whole-night
  low-pass, applied to that *whole continuous* envelope before any slicing.
  Filtering a per-bout segment directly instead would let the Tukey taper and
  ``filtfilt`` edge padding contaminate the bout's own edges -- see
  :func:`~infraslow.processing.infraslow.isfs_lowpass`'s docstring.

Everything else -- bout-finding, per-bout spindle counts, per-bout
:func:`~infraslow.processing.infraslow.infraslow_spectrum` -- is genuinely
per-stage (N2, N3) and is computed by slicing those two whole-night arrays,
using each channel's own U-Sleep hypnodensity (not a shared subject-wide
Bioserenity hypnogram) to find N2/N3 bouts and restrict spindle/slow-wave
detection -- see the ``usleep/<N>s/`` artifacts below.

Saved layout (one ``<output-dir>/<N>s/data/<subject>/<channel>/`` tree per
subject/channel, ``N`` = ``--hypno-epoch-sec`` -- see :func:`_epoch_root`;
shard/progress logs go under ``<output-dir>/logs/``, not epoch-scoped)::

    <N>s/data/<subject>/<channel>/
        usleep/
            <N>s/                  # N = --hypno-epoch-sec, default 3
                argmax.npy         # (m,) str stage labels, <N>-s epochs
                                   # (argmax of average.npy's row)
                average.npy        # (m, 5) averaged U-Sleep stage probabilities,
                                   # USLEEP_STAGE_ORDER (Wake,N1,N2,N3,REM) columns
        envelope/
            sigma.npz             # t_env, power -- whole night
            delta.npz
        temporal_ISFS/
            sigma.npz             # t_env, power (isfs_lowpass of envelope) -- whole night
            delta.npz
        N2/
            <N>s/                  # N = --hypno-epoch-sec -- same width as usleep/<N>s/
                                   # above, since every artifact below is restricted by
                                   # that epoch's hypnogram; re-running with a different
                                   # --hypno-epoch-sec adds a sibling <N>s/ dir instead of
                                   # overwriting this one (see infraslow.pipeline.io.
                                   # epoch_dirname, the shared writer/reader formatting)
                spindle_yasa.csv   # spindles_detect(..., include=(2,)).summary()
                spindle_bouts.npz  # all, spindle -- (n, 2) [start, stop] arrays (s)
                sw_yasa.csv        # sw_detect(..., include=(2,)).summary()
                sw_bouts.npz       # all, sw -- (n, 2) [start, stop] arrays (s)
                ISFS/
                    sigma.npz      # freqs (shared grid), psds (n_bouts, n_freqs),
                                   # bout_start (n_bouts,) -- ALL bouts, not just
                                   # spindle-containing ones; select a subset later by
                                   # matching bout_start against spindle_bouts.npz['spindle']
                    delta.npz
        N3/                       # same as N2, include=(3,)

``freqs`` is saved once per ``ISFS/*.npz`` (not once per bout) because
:func:`~infraslow.processing.infraslow.infraslow_spectrum`'s frequency grid
only depends on ``sf_env``/``window_sec`` (fixed pipeline constants), not on
any one bout's length -- every bout already shares it. ``t_env`` is likewise
only saved once per ``envelope``/``temporal_ISFS`` file (not per bout): it is
fully reconstructable from a bout's own ``(start, stop)`` in ``spindle_bouts.npz``
plus ``SF_ENV``, so storing it per bout would just be duplicate data.

Run via Slurm, not the login node, from this file's own directory
(``src/scripts/``) with ``src/`` on ``PYTHONPATH`` so ``infraslow`` resolves
(the package is not pip-installed): the per-subject/channel artifact tree
described above, plus two CSVs per ``--channels`` entry per invocation, both
keyed the same way and living alongside ``data/`` under the same
``<output-dir>/<N>s/`` root (see :func:`_epoch_root`) --
``<output-dir>/<N>s/events_<channel>.csv``/``sleep_stats_<channel>.csv`` for
a ``--subject`` run, ``..._<channel>_shard<N>.csv`` for a sharded run
(mirrors ``progress.log``/``progress_shard<N>.log``'s naming so concurrent
shards, and concurrent channels within a shard, never share a file; see
:func:`_events_path`/:func:`_sleep_stats_path`):

* ``events_<channel>.csv`` -- columns ``id`` and, for every ``--stages``
  entry, ``<stage>_Spindle`` and ``<stage>_SW`` (the total number of
  spindles/slow waves YASA detected across that stage's bouts --
  ``spindle_yasa.csv``/``sw_yasa.csv``'s row count, not the number of bouts
  containing one -- see :func:`preprocess_channel`'s return value).
* ``sleep_stats_<channel>.csv`` -- columns ``id`` and every ``yasa.
  sleep_statistics()`` key (:data:`SLEEP_STATS_KEYS`; TIB, SPT, WASO, TST, N1,
  N2, N3, REM, NREM, SOL, Lat_N1, Lat_N2, Lat_N3, Lat_REM, %N1, %N2, %N3,
  %REM, %NREM, SE, SME), computed from the same reduced hypnogram
  :func:`preprocess_channel` uses for bout-finding -- always produced
  alongside the rest of preprocessing, in the same run.

A cell is blank (not a guessed ``0``) if that channel raised outright for
this subject -- see :func:`preprocess_subject`. Each file is rewritten from
scratch at the start of its shard's run (so resubmitting a shard never
duplicates rows) and appended to one subject at a time as the run progresses
(so a killed/timed-out job keeps every subject it finished before that
point, not just the ones from a final batch write). Merge each channel's
per-shard CSVs together yourself once every shard has finished (plain
concatenation -- ``pandas.concat``/``cat`` all ``events_<channel>_shard*.csv``
into one ``events_<channel>.csv``, same for ``sleep_stats_<channel>``; no
subject id repeats across shards, since shards partition the cohort).
Single-subject dry run::

    export PYTHONPATH=/home/users/chaisaen/infraslow/src
    srun -p normal --time=00:30:00 --mem=8G --cpus-per-task=1 \\
        python3 preprocessing.py --subject 318679 --channels C3 \\
            --output-dir $SCRATCH/processed_data

Whole cohort, split across a job array (``--num-shards``/``--shard-index``
default from ``$SLURM_ARRAY_TASK_COUNT``/``$SLURM_ARRAY_TASK_ID`` when a
``--subject`` is not given -- see ``run_preprocessing.sbatch``)::

    sbatch --array=0-999 run_preprocessing.sbatch
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yasa

from infraslow import BioserenityPSGLoader
from infraslow.constants import (
    DEFAULT_EDF_DIR,
    DEFAULT_EEG_CHANNELS,
    DEFAULT_HYPNOGRAM_EPOCH_SEC,
    DEFAULT_METADATA,
    DEFAULT_METADATA2,
    DEFAULT_SF_ENV,
    DEFAULT_STAGE_MAP,
    DEFAULT_SIGMA_BAND,
    DEFAULT_DELTA_BAND,
    DEFAULT_USLEEP_ALIGN_TOLERANCE_SEC,
    DEFAULT_USLEEP_EPOCH_SEC,
    DEFAULT_USLEEP_HYPNODENSITY_DIR,
    DEFAULT_WINDOW_SEC,
)
from infraslow.processing.spindle import _stages_to_int, spindles_detect
from infraslow.processing.sws import sw_detect
from infraslow.processing.utils import find_stage_bouts
from infraslow.io.metadata import (
    combine_bioserenity_metadata,
    load_bioserenity_metadata,
)
from infraslow.io.utils import list_dir_filenames
from infraslow.io import usleep_hypnodensity as uh
from infraslow.pipeline import io as pio
from infraslow.processing.infraslow import eeg_envelope, infraslow_spectrum, isfs_lowpass

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Analysis constants -- matched to demo_infraslow_yasa_recheck.ipynb /
# demo_infraslow_phase.ipynb (not the retired subject_pipeline.py's slightly
# different SF/SIGMA_BAND -- these two notebooks are this script's ground truth).
# --------------------------------------------------------------------------- #
SF: float = 200.0                          # loader resample rate (Hz)
SIGMA_BAND: Tuple[float, float] = DEFAULT_SIGMA_BAND
# Standard delta band. Not computed anywhere else in this repo yet -- if your
# reference analysis uses a different delta range, override via `bands=`.
DELTA_BAND: Tuple[float, float] = DEFAULT_DELTA_BAND
BANDS: Dict[str, Tuple[float, float]] = {"sigma": SIGMA_BAND, "delta": DELTA_BAND}

SF_ENV: float = DEFAULT_SF_ENV              # 1 Hz envelope rate
HYPNO_EPOCH_SEC: float = DEFAULT_HYPNOGRAM_EPOCH_SEC   # U-Sleep hypnogram epoch width (s)
MIN_BOUT_SEC: float = 200.0                  # consecutive-stage bout length (s)
WINDOW_SEC: float = DEFAULT_WINDOW_SEC       # infraslow_spectrum's fixed freq-grid window

STAGE_CODES: Dict[str, Tuple[int, ...]] = {"N2": (2,), "N3": (3,)}
DEFAULT_STAGES: Tuple[str, ...] = ("N2", "N3")
DEFAULT_CHANNELS: Tuple[str, ...] = DEFAULT_EEG_CHANNELS

# yasa.sleep_statistics()'s return dict is a fixed schema given the fixed
# Wake/N1/N2/N3/REM stage set in DEFAULT_STAGE_MAP -- fixed here too so the
# preprocess subcommand's sleep_stats_<channel>.csv header can be written up
# front, before any subject is processed (see main_preprocess).
SLEEP_STATS_KEYS: Tuple[str, ...] = (
    "TIB", "SPT", "WASO", "TST", "N1", "N2", "N3", "REM", "NREM", "SOL",
    "Lat_N1", "Lat_N2", "Lat_N3", "Lat_REM",
    "%N1", "%N2", "%N3", "%REM", "%NREM", "SE", "SME",
)

# Subdirectories of --output-dir: per-subject trees under DATA_DIRNAME, shard/
# progress logs under LOGS_DIRNAME -- kept apart so a directory listing of one
# never has to skip over 100k+ entries of the other.
DATA_DIRNAME: str = "data"
LOGS_DIRNAME: str = "logs"

# Minimal columns guaranteed present in spindle_yasa.csv even when zero
# spindles are detected (a real detection's summary() has YASA's full schema;
# only these three are ever consumed downstream -- see bout/spindle assignment).
_EMPTY_SPINDLE_COLUMNS: Tuple[str, ...] = ("Start", "Peak", "End")

# Same idea for sw_yasa.csv -- "NegPeak" is the slow wave's defining timestamp
# (see yasa.sw_detect), playing the same per-event role "Peak" plays for
# spindles when assigning events to bouts.
_EMPTY_SW_COLUMNS: Tuple[str, ...] = ("Start", "NegPeak", "End")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None else val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default=os.environ.get("SUBJECT"),
                    help="Process only this one subject id / EDF stem (env: SUBJECT). "
                         "If omitted, every valid subject (has both an EDF and a "
                         "U-Sleep hypnodensity folder) is processed -- see "
                         "--num-shards/--shard-index to split that cohort across parallel jobs.")
    p.add_argument("--metadata", default=DEFAULT_METADATA, help="Primary metadata CSV path.")
    p.add_argument("--metadata2", default=DEFAULT_METADATA2,
                    help="Second metadata CSV path, combined with --metadata by ID.")
    p.add_argument("--edf-dir", default=DEFAULT_EDF_DIR, help="Directory of {id}.edf files.")
    p.add_argument("--usleep-dir", default=os.path.expandvars(
                        _env_str("USLEEP_DIR", DEFAULT_USLEEP_HYPNODENSITY_DIR)),
                    help="U-Sleep hypnodensity directory, one subfolder per subject "
                         "(env: USLEEP_DIR)")
    p.add_argument("--hypno-epoch-sec", type=float,
                    default=float(_env_str("HYPNO_EPOCH_SEC", str(HYPNO_EPOCH_SEC))),
                    help="Width (s) of the epoch the native 1-s U-Sleep hypnodensity is "
                         "averaged into before bout-finding/event detection; also names "
                         "the saved usleep/<N>s/ directory (env: HYPNO_EPOCH_SEC)")
    p.add_argument("--num-shards", type=int, default=None,
                    help="Split valid subjects into this many disjoint shards, one per "
                         "parallel job (see --shard-index). Defaults to "
                         "$SLURM_ARRAY_TASK_COUNT, else 1. Ignored with --subject.")
    p.add_argument("--shard-index", type=int, default=None,
                    help="This job's shard in [0, --num-shards) -- processes every "
                         "--num-shards-th subject (sorted by ID), starting here. Defaults "
                         "to $SLURM_ARRAY_TASK_ID, else 0. Ignored with --subject.")
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most this many subjects from this shard (quick tests).")
    p.add_argument("--channels", nargs="+",
                    default=_env_str("CHANNELS", ",".join(DEFAULT_CHANNELS)).split(","),
                    help=f"EEG channels to process (env: CHANNELS, comma-separated; "
                         f"default {','.join(DEFAULT_CHANNELS)})")
    p.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES),
                    choices=list(STAGE_CODES), help="Stage groups to process")
    p.add_argument("--sf", type=float, default=float(_env_str("SF", str(SF))),
                    help="Loader resample rate (Hz) (env: SF)")
    p.add_argument("--min-bout-sec", type=float,
                    default=float(_env_str("MIN_BOUT_SEC", str(MIN_BOUT_SEC))),
                    help="Minimum consecutive-stage bout length (s) (env: MIN_BOUT_SEC)")
    p.add_argument("--window-sec", type=float,
                    default=float(_env_str("WINDOW_SEC", str(WINDOW_SEC))),
                    help="infraslow_spectrum frequency-grid window (s) (env: WINDOW_SEC)")
    p.add_argument("--output-dir", type=Path,
                    default=Path(os.path.expandvars(
                        _env_str("OUTPUT_DIR", "$SCRATCH/infraslow_outputs/preprocessed"))),
                    help="Root output directory, one subfolder per subject (env: OUTPUT_DIR)")
    return p.parse_args()


def resolve_shard(cli_num_shards: Optional[int], cli_shard_index: Optional[int]) -> Tuple[int, int]:
    """``(num_shards, shard_index)``, defaulting to the Slurm array env vars when unset.

    Lets a job array split the subject list across tasks with no explicit CLI
    flags at all.
    """
    num_shards = cli_num_shards
    if num_shards is None:
        env = os.environ.get("SLURM_ARRAY_TASK_COUNT", "").strip()
        num_shards = int(env) if env.isdigit() and int(env) > 0 else 1
    shard_index = cli_shard_index
    if shard_index is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
        shard_index = int(env) if env.isdigit() else 0
    if num_shards < 1:
        raise SystemExit(f"--num-shards must be >= 1 (got {num_shards})")
    if not (0 <= shard_index < num_shards):
        raise SystemExit(f"--shard-index {shard_index} out of range for --num-shards {num_shards}")
    return num_shards, shard_index


def _validate_hypno_epoch_sec(hypno_epoch_sec: float, sf: float) -> None:
    """Fail fast if ``hypno_epoch_sec`` would blow up every channel of every
    subject deep inside :func:`preprocess_channel` instead of before any
    Slurm-job time is spent.

    Checks the same two whole-number constraints
    :func:`infraslow.io.usleep_hypnodensity.hypnodensity_to_epoch_hypnogram`
    and :func:`infraslow.processing.spindle._build_sample_hypno` each need but
    only discover deep inside per-channel processing (where the resulting
    ``ValueError`` is swallowed by :func:`preprocess_subject`'s per-channel
    ``try/except``, letting the run still exit 0):

    1. ``hypno_epoch_sec`` must be a positive integer multiple of the native
       U-Sleep cadence ``DEFAULT_USLEEP_EPOCH_SEC``.
    2. ``sf * hypno_epoch_sec`` -- the number of EEG samples per hypnogram
       epoch -- must be a whole number, or ``yasa.hypno_upsample_to_data``
       raises inside every spindle/slow-wave detection call.
    """
    ratio = hypno_epoch_sec / DEFAULT_USLEEP_EPOCH_SEC
    n_per_window = round(ratio)
    if n_per_window < 1 or not np.isclose(ratio, n_per_window):
        raise SystemExit(
            f"--hypno-epoch-sec {hypno_epoch_sec} must be a positive integer multiple of "
            f"the native U-Sleep cadence DEFAULT_USLEEP_EPOCH_SEC ({DEFAULT_USLEEP_EPOCH_SEC})"
        )

    samples_per_epoch = sf * hypno_epoch_sec
    if not np.isclose(samples_per_epoch, round(samples_per_epoch)):
        raise SystemExit(
            f"--sf {sf} * --hypno-epoch-sec {hypno_epoch_sec} = {samples_per_epoch} must be "
            f"a whole number of EEG samples per hypnogram epoch (required by "
            f"yasa.hypno_upsample_to_data)"
        )


def _valid_usleep_subject_ids(
    ids: Sequence[str], edf_names: Set[str], usleep_names: Set[str], *,
    edf_suffix: str = ".edf",
) -> List[str]:
    """Sorted subject ids present in both ``edf_names`` (as ``f"{id}{edf_suffix}"``)
    and ``usleep_names`` (as a bare directory-entry name, ``id`` itself) -- the
    pure membership test behind :func:`list_valid_subjects`."""
    return sorted(
        sid for sid in ids
        if f"{sid}{edf_suffix}" in edf_names and sid in usleep_names
    )


def list_valid_subjects(metadata_path: str, metadata2_path: str, edf_dir: str, usleep_dir: str) -> List[str]:
    """Every subject id with both an EDF and a U-Sleep hypnodensity subject
    folder, sorted.

    Metadata CSVs still supply the master candidate id list (Age/Gender/BMI);
    unlike the retired Bioserenity-Hypnodensity-CSV check, this no longer reads
    or requires the Bioserenity Hypnodensity CSV at all -- see
    ``infraslow.io.usleep_hypnodensity`` for the U-Sleep data this pipeline
    stages from instead.
    """
    metadata = combine_bioserenity_metadata(
        load_bioserenity_metadata(Path(metadata_path)),
        load_bioserenity_metadata(Path(metadata2_path)),
    )
    edf_names = list_dir_filenames(Path(os.path.expandvars(edf_dir)))
    usleep_names = list_dir_filenames(Path(os.path.expandvars(usleep_dir)))
    ids = metadata["ID"].astype(str).tolist()
    return _valid_usleep_subject_ids(ids, edf_names, usleep_names)


# --------------------------------------------------------------------------- #
# Save helpers
# --------------------------------------------------------------------------- #
def _save_timeseries(path: Path, t_env: np.ndarray, power: np.ndarray) -> None:
    np.savez(path, t_env=np.asarray(t_env, dtype=np.float64),
              power=np.asarray(power, dtype=np.float64))


def _save_bouts(path: Path, all_bouts: Sequence[Tuple[float, float]],
                 event_bouts: Sequence[Tuple[float, float]], *,
                 event_key: str = "spindle") -> None:
    all_arr = np.asarray(all_bouts, dtype=np.float64).reshape(-1, 2)
    event_arr = np.asarray(event_bouts, dtype=np.float64).reshape(-1, 2)
    np.savez(path, all=all_arr, **{event_key: event_arr})


def _save_spectra(path: Path, freqs: np.ndarray, psds: np.ndarray, bout_start: np.ndarray) -> None:
    np.savez(path, freqs=np.asarray(freqs, dtype=np.float64),
              psds=np.asarray(psds, dtype=np.float64),
              bout_start=np.asarray(bout_start, dtype=np.float64))


# --------------------------------------------------------------------------- #
# U-Sleep hypnogram helpers -- per-channel staging (replaces the retired
# subject-wide Bioserenity Hypnodensity CSV hypnogram).
# --------------------------------------------------------------------------- #
def save_usleep_hypnogram(
    ch_dir: Path, stage_epoch: np.ndarray, probs_epoch: np.ndarray, *,
    hypno_epoch_sec: float = HYPNO_EPOCH_SEC,
) -> Path:
    """Save one channel's reduced U-Sleep hypnogram under
    ``ch_dir/usleep/<N>s/argmax.npy`` (str stage labels) and ``average.npy``
    (averaged stage probabilities). Returns the ``usleep/<N>s`` directory."""
    usleep_dir = ch_dir / "usleep" / pio.epoch_dirname(hypno_epoch_sec)
    usleep_dir.mkdir(parents=True, exist_ok=True)
    np.save(usleep_dir / "argmax.npy", np.asarray(stage_epoch, dtype=str))
    np.save(usleep_dir / "average.npy", np.asarray(probs_epoch, dtype=np.float64))
    return usleep_dir


def _check_usleep_alignment(
    usleep_duration_sec: float, eeg_duration_sec: float, *,
    tolerance_sec: float = DEFAULT_USLEEP_ALIGN_TOLERANCE_SEC,
) -> Optional[str]:
    """``None`` if the U-Sleep hypnodensity and EEG signal durations agree
    within ``tolerance_sec``, else a ready-to-log warning message."""
    diff = abs(usleep_duration_sec - eeg_duration_sec)
    if diff <= tolerance_sec:
        return None
    return (
        f"U-Sleep hypnodensity duration ({usleep_duration_sec:.1f}s) and EEG "
        f"signal duration ({eeg_duration_sec:.1f}s) differ by {diff:.1f}s "
        f"(> {tolerance_sec:.1f}s tolerance)."
    )


def _check_usleep_file_usable(subject_id: str, channel: str, usleep_dir: str) -> Path:
    """The resolved U-Sleep hypnodensity ``.npy`` path for this subject/channel,
    checked usable (exists, non-empty) before it's ever opened.

    Runs ahead of :func:`preprocess_channel`'s more expensive work (envelope/ISFS
    computation, spindle/slow-wave detection) so a truncated or killed write --
    the common failure mode behind a 0-byte ``.npy`` file -- fails fast with a
    clear message instead of surfacing as a raw ``numpy``/pickle exception deep
    inside :func:`~infraslow.io.usleep_hypnodensity.load_usleep_hypnodensity`.

    Raises:
        FileNotFoundError: no U-Sleep hypnodensity file exists for this channel
            (see :func:`~infraslow.io.usleep_hypnodensity.resolve_usleep_channel_path`),
            or the resolved file exists but is empty (0 bytes).
    """
    subject_dir = Path(os.path.expandvars(str(usleep_dir))) / subject_id
    path = uh.resolve_usleep_channel_path(subject_dir, channel)
    if path is None:
        raise FileNotFoundError(
            f"No U-Sleep hypnodensity file for channel '{channel}' under {subject_dir}"
        )
    if path.stat().st_size == 0:
        raise FileNotFoundError(f"U-Sleep hypnodensity file is empty (0 bytes): {path}")
    return path


# --------------------------------------------------------------------------- #
# Per-channel computation
# --------------------------------------------------------------------------- #
def compute_envelopes(
    data: np.ndarray, sf: float, *, bands: Mapping[str, Tuple[float, float]] = BANDS,
    sf_env: float = SF_ENV,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``{band_name: (t_env, power)}`` -- one whole-night wavelet envelope per band."""
    return {
        name: eeg_envelope(
            data, sf, band=band, sf_env=sf_env, smooth_sec=1.0, wavelet=True,
            to_db=True, kind="power",
        )
        for name, band in bands.items()
    }


def compute_temporal_isfs(
    envelopes: Mapping[str, Tuple[np.ndarray, np.ndarray]], *, sf_env: float = SF_ENV,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``{band_name: (t_env, filtered)}`` -- ``isfs_lowpass`` of each whole-night envelope.

    Filters the *whole* continuous ``power`` course for each band once (the
    same ``t_env`` carries over unchanged) -- never a per-bout slice; see this
    module's docstring for why.
    """
    return {
        name: (t_env, isfs_lowpass(power, sf_env))
        for name, (t_env, power) in envelopes.items()
    }


def compute_bout_spectra(
    t_env: np.ndarray, power: np.ndarray, bouts: Sequence[Tuple[float, float]], *,
    sf_env: float = SF_ENV, window_sec: float = WINDOW_SEC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(freqs, psds, bout_start)`` -- one :func:`infraslow_spectrum` per bout,
    stacked, from *every* bout passed in (not just spindle-containing ones --
    filter later using ``bout_start``, e.g. by matching it against
    ``spindle_bouts.npz['spindle']``'s own start times).

    ``freqs`` is shared across every bout (see :func:`infraslow_spectrum`'s
    fixed, length-independent grid) so ``psds`` is a plain ``(n_bouts,
    n_freqs)`` array; ``bout_start`` (``(n_bouts,)``) gives each row's source
    bout start time. All empty if ``bouts`` is empty.
    """
    if not bouts:
        return (np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.float64))
    segs = [power[(t_env >= a) & (t_env < b)] for a, b in bouts]
    specs = [infraslow_spectrum(s, sf_env, window_sec=window_sec) for s in segs]
    freqs = specs[0].freqs
    psds = np.stack([s.psd for s in specs])
    bout_start = np.asarray([a for a, _ in bouts], dtype=np.float64)
    return freqs, psds, bout_start


def _detect_stage_events(
    detect_fn, loader: BioserenityPSGLoader, channel: str, codes: Tuple[int, ...],
    all_bouts: Sequence[Tuple[float, float]], *, peak_column: str, empty_columns: Sequence[str],
    epoch_sec: float = HYPNO_EPOCH_SEC,
) -> Tuple[pd.DataFrame, List[Tuple[float, float]]]:
    """Run a YASA event detector (``spindles_detect`` or ``sw_detect``) for one
    stage/channel and pair its events back up with ``all_bouts``.

    Skips the detector call entirely when ``all_bouts`` is empty -- YASA
    asserts rather than returning ``None`` when the hypnogram has no epochs of
    the requested stage -- and returns an empty, schema-stable summary instead.

    Returns:
        ``(summary, event_bouts)`` where ``summary`` is the detector's
        ``.summary()`` DataFrame (or an empty one with ``empty_columns``) and
        ``event_bouts`` is the subset of ``all_bouts`` containing >= 1 event
        (matched via ``peak_column``).
    """
    if not all_bouts:
        summary = pd.DataFrame(columns=list(empty_columns))
        peaks = np.empty(0, dtype=float)
    else:
        result = detect_fn(loader, ch_names=channel, include=codes, epoch_sec=epoch_sec)
        if result is not None:
            summary = result.summary()
            peaks = summary[peak_column].to_numpy(dtype=float)
        else:
            summary = pd.DataFrame(columns=list(empty_columns))
            peaks = np.empty(0, dtype=float)
    event_bouts = [(a, b) for a, b in all_bouts if np.any((peaks >= a) & (peaks < b))]
    return summary, event_bouts


def preprocess_channel(
    loader: BioserenityPSGLoader, subject_id: str, channel: str,
    output_dir: Path, *,
    usleep_dir: str = DEFAULT_USLEEP_HYPNODENSITY_DIR,
    bands: Mapping[str, Tuple[float, float]] = BANDS,
    stages: Sequence[str] = DEFAULT_STAGES,
    stage_codes: Mapping[str, Tuple[int, ...]] = STAGE_CODES,
    sf_env: float = SF_ENV, hypno_epoch_sec: float = HYPNO_EPOCH_SEC,
    min_bout_sec: float = MIN_BOUT_SEC, window_sec: float = WINDOW_SEC,
    align_tolerance_sec: float = DEFAULT_USLEEP_ALIGN_TOLERANCE_SEC,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float]]:
    """Compute and save every artifact for one subject/channel (see module docstring).

    The hypnogram used for bout-finding and to restrict spindle/slow-wave
    detection is this channel's own U-Sleep hypnodensity, reduced from its
    native 1-s resolution to ``hypno_epoch_sec``-wide epochs -- not a
    subject-wide Bioserenity hypnogram (U-Sleep hypnodensity is scored
    per-channel; see ``infraslow.io.usleep_hypnodensity``'s module docstring).
    ``yasa.sleep_statistics`` is computed from that same reduced hypnogram
    instead of loading the U-Sleep file a second time.

    Returns:
        ``({stage: {"spindle": int, "sw": int}}, sleep_stats)`` -- the first
        for :func:`main_preprocess`'s ``events_<channel>.csv`` (one entry per
        ``stages``: the total number of spindles/slow waves YASA detected in
        that stage's bouts -- ``spindle_yasa.csv``/``sw_yasa.csv``'s row
        count, not the number of bouts containing one), the second (``yasa.
        sleep_statistics()``'s return dict) for its ``sleep_stats_<channel>.csv``.
    """
    ch_dir = output_dir / DATA_DIRNAME / subject_id / channel
    (ch_dir / "envelope").mkdir(parents=True, exist_ok=True)
    (ch_dir / "temporal_ISFS").mkdir(parents=True, exist_ok=True)

    data = np.asarray(loader.get_channel(channel), dtype=float)
    sf = float(loader.sf)

    _check_usleep_file_usable(subject_id, channel, usleep_dir)
    t_hyp, probs = uh.load_usleep_hypnodensity(subject_id, channel, base_dir=usleep_dir)
    _t_epoch, probs_epoch, stage_epoch = uh.hypnodensity_to_epoch_hypnogram(
        t_hyp, probs, epoch_sec=hypno_epoch_sec, src_epoch_sec=DEFAULT_USLEEP_EPOCH_SEC,
    )
    alignment_warning = _check_usleep_alignment(
        probs.shape[0] * DEFAULT_USLEEP_EPOCH_SEC, data.shape[-1] / sf,
        tolerance_sec=align_tolerance_sec,
    )
    if alignment_warning:
        logger.warning("subject %s channel %s: %s", subject_id, channel, alignment_warning)
    save_usleep_hypnogram(ch_dir, stage_epoch, probs_epoch, hypno_epoch_sec=hypno_epoch_sec)

    hypnogram = _stages_to_int(stage_epoch, DEFAULT_STAGE_MAP)
    sleep_stats = yasa.sleep_statistics(hypnogram, sf_hyp=1.0 / hypno_epoch_sec)
    # BioserenityPSGLoader.annotations has no public setter (read-only property
    # over the ``_annotations`` field); U-Sleep hypnodensity is scored
    # per-channel, so each channel's own reduced hypnogram must override the
    # loader's (here always-empty, see preprocess_subject) annotations before
    # spindle/slow-wave detection runs for this channel.
    loader._annotations = pd.DataFrame({"stage": stage_epoch})

    envelopes = compute_envelopes(data, sf, bands=bands, sf_env=sf_env)
    for name, (t_env, power) in envelopes.items():
        _save_timeseries(ch_dir / "envelope" / f"{name}.npz", t_env, power)

    temporal_isfs = compute_temporal_isfs(envelopes, sf_env=sf_env)
    for name, (t_env, filtered) in temporal_isfs.items():
        _save_timeseries(ch_dir / "temporal_ISFS" / f"{name}.npz", t_env, filtered)

    stage_events: Dict[str, Dict[str, int]] = {}
    for stage in stages:
        # Nested under <N>s/ (N = hypno_epoch_sec), mirroring usleep/<N>s/ above --
        # every artifact saved below is restricted by that epoch's hypnogram, so a
        # rerun with a different --hypno-epoch-sec adds a sibling <N>s/ dir instead
        # of overwriting this one (see infraslow.pipeline.io.epoch_dirname).
        stage_dir = ch_dir / stage / pio.epoch_dirname(hypno_epoch_sec)
        (stage_dir / "ISFS").mkdir(parents=True, exist_ok=True)
        codes = stage_codes[stage]

        all_bouts = find_stage_bouts(hypnogram, codes, epoch_sec=hypno_epoch_sec, min_dur=min_bout_sec)

        if not all_bouts:
            # No epochs of this stage in the hypnogram -- yasa.spindles_detect/
            # sw_detect assert on this rather than returning None, so skip the
            # call entirely instead of letting it abort the whole channel.
            logger.info(
                "No %s epochs for subject %s channel %s; skipping spindle/slow-wave detection.",
                stage, subject_id, channel,
            )

        spindle_summary, spindle_bouts = _detect_stage_events(
            spindles_detect, loader, channel, codes, all_bouts,
            peak_column="Peak", empty_columns=_EMPTY_SPINDLE_COLUMNS,
            epoch_sec=hypno_epoch_sec,
        )
        spindle_summary.to_csv(stage_dir / "spindle_yasa.csv", index=False)
        _save_bouts(stage_dir / "spindle_bouts.npz", all_bouts, spindle_bouts, event_key="spindle")

        sw_summary, sw_bouts = _detect_stage_events(
            sw_detect, loader, channel, codes, all_bouts,
            peak_column="NegPeak", empty_columns=_EMPTY_SW_COLUMNS,
            epoch_sec=hypno_epoch_sec,
        )
        sw_summary.to_csv(stage_dir / "sw_yasa.csv", index=False)
        _save_bouts(stage_dir / "sw_bouts.npz", all_bouts, sw_bouts, event_key="sw")

        for name, (t_env, power) in envelopes.items():
            freqs, psds, bout_start = compute_bout_spectra(
                t_env, power, all_bouts, sf_env=sf_env, window_sec=window_sec,
            )
            _save_spectra(stage_dir / "ISFS" / f"{name}.npz", freqs, psds, bout_start)

        stage_events[stage] = dict(spindle=len(spindle_summary), sw=len(sw_summary))

    return stage_events, sleep_stats


# --------------------------------------------------------------------------- #
# Per-subject entry point
# --------------------------------------------------------------------------- #
def preprocess_subject(
    subject_id: str, output_dir: Path, *,
    sf: float = SF, channels: Sequence[str] = DEFAULT_CHANNELS,
    usleep_dir: str = DEFAULT_USLEEP_HYPNODENSITY_DIR,
    bands: Mapping[str, Tuple[float, float]] = BANDS,
    stages: Sequence[str] = DEFAULT_STAGES,
    stage_codes: Mapping[str, Tuple[int, ...]] = STAGE_CODES,
    hypno_epoch_sec: float = HYPNO_EPOCH_SEC,
    min_bout_sec: float = MIN_BOUT_SEC, window_sec: float = WINDOW_SEC,
) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], Dict[str, Dict[str, float]]]:
    """Load one subject once and preprocess every requested channel.

    No subject-wide hypnogram is loaded here: the Bioserenity Hypnodensity CSV
    is never read (``annotation_loader`` is a no-op below) -- each channel
    sources, and saves, its own hypnogram from its own U-Sleep hypnodensity
    file inside :func:`preprocess_channel`, which also computes that
    channel's ``yasa.sleep_statistics`` from it -- sleep statistics are
    therefore always produced alongside the rest of preprocessing, in the
    same pass, not via a separate command.

    Returns:
        ``({channel: {stage: {"spindle": int, "sw": int}}}, {channel:
        sleep_stats})`` for every channel that completed without raising -- a
        channel that raised is simply absent from both (see
        :func:`main_preprocess`'s ``events_<channel>.csv``/
        ``sleep_stats_<channel>.csv``, which leave that channel's row blank
        for this subject rather than guessing).
    """
    loader = BioserenityPSGLoader(
        subject_id=subject_id, sf=sf, requested_channels=list(channels),
        annotation_loader=lambda inst, edf_path: None,
    ).load()

    events: Dict[str, Dict[str, Dict[str, int]]] = {}
    sleep_stats: Dict[str, Dict[str, float]] = {}
    for ch in channels:
        try:
            events[ch], sleep_stats[ch] = preprocess_channel(
                loader, subject_id, ch, output_dir,
                usleep_dir=usleep_dir, bands=bands, stages=stages, stage_codes=stage_codes,
                hypno_epoch_sec=hypno_epoch_sec, min_bout_sec=min_bout_sec, window_sec=window_sec,
            )
        except Exception:  # noqa: BLE001 - one bad channel must not sink the others
            logger.exception("Channel %s failed for subject %s", ch, subject_id)
    return events, sleep_stats


def _event_fieldnames(stages: Sequence[str]) -> List[str]:
    """One channel's events CSV header: ``id`` then ``<stage>_{Spindle,SW}``
    for every requested stage, in that order. The channel itself lives in the
    file's name (see :func:`_events_path`), not in these columns."""
    fields = ["id"]
    for stage in stages:
        fields.append(f"{stage}_Spindle")
        fields.append(f"{stage}_SW")
    return fields


def _event_row(
    subject_id: str, channel_events: Optional[Mapping[str, Mapping[str, int]]], *,
    stages: Sequence[str],
) -> Dict[str, object]:
    """One row of one channel's events CSV, from that channel's entry in
    :func:`preprocess_subject`'s return value (``None`` if the channel raised
    outright for this subject). ``None`` (blank in the CSV) for a stage that
    isn't in ``channel_events``, never a guessed ``0``."""
    row: Dict[str, object] = {"id": subject_id}
    for stage in stages:
        stage_events = channel_events.get(stage) if channel_events else None
        row[f"{stage}_Spindle"] = stage_events["spindle"] if stage_events else None
        row[f"{stage}_SW"] = stage_events["sw"] if stage_events else None
    return row


def _epoch_root(output_dir: Path, hypno_epoch_sec: float) -> Path:
    """``<output-dir>/<N>s/`` -- the root every preprocess artifact (the
    ``data/`` tree, ``events_<channel>*.csv``, ``sleep_stats_<channel>*.csv``)
    is saved under for a given ``--hypno-epoch-sec`` (reusing
    :func:`infraslow.pipeline.io.epoch_dirname`'s ``<N>s`` formatting), so
    rerunning against the same ``--output-dir`` with a different epoch width
    adds a sibling tree instead of overwriting this one."""
    return output_dir / pio.epoch_dirname(hypno_epoch_sec)


def _events_path(output_dir: Path, channel: str, *, shard_index: Optional[int]) -> Path:
    """One channel's events CSV path -- ``events_<channel>.csv`` for a
    ``--subject`` run, ``events_<channel>_shard<N>.csv`` for a sharded run
    (mirrors ``progress.log``/``progress_shard<N>.log``'s naming so concurrent
    shards, and concurrent channels within a shard, never share a file)."""
    if shard_index is None:
        return output_dir / f"events_{channel}.csv"
    return output_dir / f"events_{channel}_shard{shard_index}.csv"


def _sleep_stats_fieldnames() -> List[str]:
    """One channel's sleep-stats CSV header: ``id`` then every
    :data:`SLEEP_STATS_KEYS` entry, in that order."""
    return ["id", *SLEEP_STATS_KEYS]


def _sleep_stats_row(
    subject_id: str, channel_stats: Optional[Mapping[str, float]],
) -> Dict[str, object]:
    """One row of one channel's sleep-stats CSV, from that channel's entry in
    :func:`preprocess_subject`'s sleep-stats return value (``None`` if the
    channel raised outright for this subject). ``None`` (blank in the CSV)
    for every column in that case, never a guessed ``0``."""
    row: Dict[str, object] = {"id": subject_id}
    for key in SLEEP_STATS_KEYS:
        row[key] = channel_stats[key] if channel_stats else None
    return row


def _sleep_stats_path(output_dir: Path, channel: str, *, shard_index: Optional[int]) -> Path:
    """One channel's sleep-stats CSV path -- ``sleep_stats_<channel>.csv`` for
    a ``--subject`` run, ``sleep_stats_<channel>_shard<N>.csv`` for a sharded
    run (mirrors :func:`_events_path`'s naming)."""
    if shard_index is None:
        return output_dir / f"sleep_stats_{channel}.csv"
    return output_dir / f"sleep_stats_{channel}_shard{shard_index}.csv"


def main_preprocess(args: argparse.Namespace) -> None:
    _validate_hypno_epoch_sec(args.hypno_epoch_sec, args.sf)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # data/, events_<channel>*.csv and sleep_stats_<channel>*.csv all nest
    # under <output-dir>/<N>s/ (see _epoch_root) -- every one of them is
    # restricted by that epoch's hypnogram, so a rerun with a different
    # --hypno-epoch-sec against the same --output-dir adds a sibling <N>s/
    # tree instead of overwriting this one.
    epoch_root = _epoch_root(args.output_dir, args.hypno_epoch_sec)
    epoch_root.mkdir(parents=True, exist_ok=True)

    # Shard resolution happens before logging setup so each shard gets its own
    # log file -- a thousand array tasks all appending to one shared
    # progress.log would interleave/contend on the same file.
    logs_dir = args.output_dir / LOGS_DIRNAME
    if args.subject:
        log_path = logs_dir / "progress.log"
        shard_index = None
    else:
        num_shards, shard_index = resolve_shard(args.num_shards, args.shard_index)
        log_path = logs_dir / f"progress_shard{shard_index}.log"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    if args.subject:
        subjects = [args.subject]
        logger.info(f"single-subject run: subject={args.subject}")
    else:
        subjects = list_valid_subjects(args.metadata, args.metadata2, args.edf_dir, args.usleep_dir)
        subjects = subjects[shard_index::num_shards]
        if args.limit:
            subjects = subjects[: args.limit]
        logger.info(f"shard {shard_index}/{num_shards}: {len(subjects)} subject(s)")

    logger.info(f"channels={args.channels} stages={args.stages} sf={args.sf} "
                f"hypno_epoch_sec={args.hypno_epoch_sec} epoch_root={epoch_root}")

    # One events CSV and one sleep-stats CSV per channel (see _events_path/
    # _sleep_stats_path) -- each rewritten from scratch here (never appended
    # to a stale file left by a prior run of this same shard), then appended
    # to one subject at a time below so a killed/timed-out job keeps every
    # subject it finished so far. Sleep stats reuse the hypnogram
    # preprocess_channel already loaded, so no extra U-Sleep file read --
    # merge each channel's per-shard CSVs together yourself, same as
    # events_<channel>_shard<N>.csv.
    event_fields = _event_fieldnames(args.stages)
    events_paths = {ch: _events_path(epoch_root, ch, shard_index=shard_index) for ch in args.channels}
    for events_path in events_paths.values():
        with events_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=event_fields).writeheader()

    sleep_stats_fields = _sleep_stats_fieldnames()
    sleep_stats_paths = {
        ch: _sleep_stats_path(epoch_root, ch, shard_index=shard_index) for ch in args.channels
    }
    for sleep_stats_path in sleep_stats_paths.values():
        with sleep_stats_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=sleep_stats_fields).writeheader()

    for subject_id in subjects:
        try:
            events, sleep_stats = preprocess_subject(
                subject_id, epoch_root, sf=args.sf, channels=args.channels,
                usleep_dir=args.usleep_dir, stages=args.stages, hypno_epoch_sec=args.hypno_epoch_sec,
                min_bout_sec=args.min_bout_sec, window_sec=args.window_sec,
            )
            for ch, events_path in events_paths.items():
                row = _event_row(subject_id, events.get(ch), stages=args.stages)
                with events_path.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=event_fields).writerow(row)
            for ch, sleep_stats_path in sleep_stats_paths.items():
                row = _sleep_stats_row(subject_id, sleep_stats.get(ch))
                with sleep_stats_path.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=sleep_stats_fields).writerow(row)
            logger.info(f"done: subject={subject_id} -> {epoch_root / DATA_DIRNAME / subject_id}")
        except Exception:  # noqa: BLE001 - one bad subject must not sink the shard
            logger.exception(f"Subject {subject_id} failed")

    logger.info(
        f"finished: {len(subjects)} subject(s) attempted -> "
        f"{sorted(events_paths.values())} {sorted(sleep_stats_paths.values())}"
    )


def main() -> None:
    args = parse_args()
    main_preprocess(args)


if __name__ == "__main__":
    main()
