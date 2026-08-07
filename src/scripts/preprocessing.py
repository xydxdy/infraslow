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
per-stage (N2, N3) and is computed by slicing those two whole-night arrays.

Saved layout (one ``<output-dir>/data/<subject>/<channel>/`` tree per
subject/channel; shard/progress logs go under ``<output-dir>/logs/``)::

    data/<subject>/<channel>/
        envelope/
            sigma.npz             # t_env, power -- whole night
            delta.npz
        temporal_ISFS/
            sigma.npz             # t_env, power (isfs_lowpass of envelope) -- whole night
            delta.npz
        N2/
            spindel_yasa.csv      # spindles_detect(..., include=(2,)).summary()
            bouts.npz             # all, spindle -- (n, 2) [start, stop] arrays (s)
            sw_yasa.csv           # sw_detect(..., include=(2,)).summary()
            sw_bouts.npz          # all, sw -- (n, 2) [start, stop] arrays (s)
            ISFS/
                sigma.npz         # freqs (shared grid), psds (n_bouts, n_freqs),
                                   # bout_start (n_bouts,) -- ALL bouts, not just
                                   # spindle-containing ones; select a subset later
                                   # by matching bout_start against bouts.npz['spindle']
                delta.npz
        N3/                       # same as N2, include=(3,)

``freqs`` is saved once per ``ISFS/*.npz`` (not once per bout) because
:func:`~infraslow.processing.infraslow.infraslow_spectrum`'s frequency grid
only depends on ``sf_env``/``window_sec`` (fixed pipeline constants), not on
any one bout's length -- every bout already shares it. ``t_env`` is likewise
only saved once per ``envelope``/``temporal_ISFS`` file (not per bout): it is
fully reconstructable from a bout's own ``(start, stop)`` in ``bouts.npz`` plus
``SF_ENV``, so storing it per bout would just be duplicate data.

Run via Slurm, not the login node, from this file's own directory
(``src/scripts/``) with ``src/`` on ``PYTHONPATH`` so ``infraslow`` resolves
(the package is not pip-installed). Single-subject dry run::

    export PYTHONPATH=/home/users/chaisaen/infraslow/src
    srun -p normal --time=00:30:00 --mem=8G --cpus-per-task=1 \\
        python3 preprocessing.py --subject 318679 --channels C3 --output-dir $SCRATCH/processed_data

Whole cohort, split across a job array (``--num-shards``/``--shard-index``
default from ``$SLURM_ARRAY_TASK_COUNT``/``$SLURM_ARRAY_TASK_ID`` when a
``--subject`` is not given -- see ``run_preprocessing.sbatch``)::

    sbatch --array=0-999 run_preprocessing.sbatch
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from infraslow import BioserenityPSGLoader
from infraslow.constants import (
    DEFAULT_EDF_DIR,
    DEFAULT_EEG_CHANNELS,
    DEFAULT_EPOCH_SEC,
    DEFAULT_HYPNO_DIR,
    DEFAULT_METADATA,
    DEFAULT_METADATA2,
    DEFAULT_SF_ENV,
    DEFAULT_STAGE_MAP,
    DEFAULT_SIGMA_BAND,
    DEFAULT_DELTA_BAND,
    DEFAULT_WINDOW_SEC,
)
from infraslow.processing.spindle import _extract_epoch_stages, _stages_to_int, spindles_detect
from infraslow.processing.sws import sw_detect
from infraslow.processing.utils import find_stage_bouts
from infraslow.io.metadata import (
    combine_bioserenity_metadata,
    find_valid_bioserenity_subjects,
    load_bioserenity_metadata,
)
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
EPOCH_SEC: float = DEFAULT_EPOCH_SEC         # 30 s scored epochs
MIN_BOUT_SEC: float = 200.0                  # consecutive-stage bout length (s)
WINDOW_SEC: float = DEFAULT_WINDOW_SEC       # infraslow_spectrum's fixed freq-grid window

STAGE_CODES: Dict[str, Tuple[int, ...]] = {"N2": (2,), "N3": (3,)}
DEFAULT_STAGES: Tuple[str, ...] = ("N2", "N3")
DEFAULT_CHANNELS: Tuple[str, ...] = DEFAULT_EEG_CHANNELS

# Subdirectories of --output-dir: per-subject trees under DATA_DIRNAME, shard/
# progress logs under LOGS_DIRNAME -- kept apart so a directory listing of one
# never has to skip over 100k+ entries of the other.
DATA_DIRNAME: str = "data"
LOGS_DIRNAME: str = "logs"

# Minimal columns guaranteed present in spindel_yasa.csv even when zero
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", default=os.environ.get("SUBJECT"),
                    help="Process only this one subject id / EDF stem (env: SUBJECT). "
                         "If omitted, every valid subject (has both an EDF and a "
                         "Hypnodensity CSV) is processed -- see --num-shards/--shard-index "
                         "to split that cohort across parallel jobs.")
    p.add_argument("--metadata", default=DEFAULT_METADATA, help="Primary metadata CSV path.")
    p.add_argument("--metadata2", default=DEFAULT_METADATA2,
                    help="Second metadata CSV path, combined with --metadata by ID.")
    p.add_argument("--edf-dir", default=DEFAULT_EDF_DIR, help="Directory of {id}.edf files.")
    p.add_argument("--hypno-dir", default=DEFAULT_HYPNO_DIR,
                    help="Directory of {id}_Hypnodensity.csv files.")
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


def list_valid_subjects(metadata_path: str, metadata2_path: str, edf_dir: str, hypno_dir: str) -> List[str]:
    """Every subject id with both an EDF and a Hypnodensity CSV, sorted.

    Reuses :mod:`infraslow.io.metadata`'s cohort-discovery (same two metadata
    CSVs, same EDF/Hypnodensity directories) so "every subject" here means
    every subject with usable data.
    """
    metadata = combine_bioserenity_metadata(
        load_bioserenity_metadata(Path(metadata_path)),
        load_bioserenity_metadata(Path(metadata2_path)),
    )
    valid = find_valid_bioserenity_subjects(
        metadata, Path(os.path.expandvars(edf_dir)), Path(os.path.expandvars(hypno_dir)),
    )
    return sorted(valid["ID"].astype(str).tolist())


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
    ``bouts.npz['spindle']``'s own start times).

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
        result = detect_fn(loader, ch_names=channel, include=codes)
        if result is not None:
            summary = result.summary()
            peaks = summary[peak_column].to_numpy(dtype=float)
        else:
            summary = pd.DataFrame(columns=list(empty_columns))
            peaks = np.empty(0, dtype=float)
    event_bouts = [(a, b) for a, b in all_bouts if np.any((peaks >= a) & (peaks < b))]
    return summary, event_bouts


def preprocess_channel(
    loader: BioserenityPSGLoader, subject_id: str, channel: str, hypnogram: np.ndarray,
    output_dir: Path, *,
    bands: Mapping[str, Tuple[float, float]] = BANDS,
    stages: Sequence[str] = DEFAULT_STAGES,
    stage_codes: Mapping[str, Tuple[int, ...]] = STAGE_CODES,
    sf_env: float = SF_ENV, epoch_sec: float = EPOCH_SEC,
    min_bout_sec: float = MIN_BOUT_SEC, window_sec: float = WINDOW_SEC,
) -> None:
    """Compute and save every artifact for one subject/channel (see module docstring)."""
    ch_dir = output_dir / DATA_DIRNAME / subject_id / channel
    (ch_dir / "envelope").mkdir(parents=True, exist_ok=True)
    (ch_dir / "temporal_ISFS").mkdir(parents=True, exist_ok=True)

    data = np.asarray(loader.get_channel(channel), dtype=float)
    sf = float(loader.sf)

    envelopes = compute_envelopes(data, sf, bands=bands, sf_env=sf_env)
    for name, (t_env, power) in envelopes.items():
        _save_timeseries(ch_dir / "envelope" / f"{name}.npz", t_env, power)

    temporal_isfs = compute_temporal_isfs(envelopes, sf_env=sf_env)
    for name, (t_env, filtered) in temporal_isfs.items():
        _save_timeseries(ch_dir / "temporal_ISFS" / f"{name}.npz", t_env, filtered)

    for stage in stages:
        stage_dir = ch_dir / stage
        (stage_dir / "ISFS").mkdir(parents=True, exist_ok=True)
        codes = stage_codes[stage]

        all_bouts = find_stage_bouts(hypnogram, codes, epoch_sec=epoch_sec, min_dur=min_bout_sec)

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
        )
        spindle_summary.to_csv(stage_dir / "spindel_yasa.csv", index=False)
        _save_bouts(stage_dir / "bouts.npz", all_bouts, spindle_bouts, event_key="spindle")

        sw_summary, sw_bouts = _detect_stage_events(
            sw_detect, loader, channel, codes, all_bouts,
            peak_column="NegPeak", empty_columns=_EMPTY_SW_COLUMNS,
        )
        sw_summary.to_csv(stage_dir / "sw_yasa.csv", index=False)
        _save_bouts(stage_dir / "sw_bouts.npz", all_bouts, sw_bouts, event_key="sw")

        for name, (t_env, power) in envelopes.items():
            freqs, psds, bout_start = compute_bout_spectra(
                t_env, power, all_bouts, sf_env=sf_env, window_sec=window_sec,
            )
            _save_spectra(stage_dir / "ISFS" / f"{name}.npz", freqs, psds, bout_start)


# --------------------------------------------------------------------------- #
# Per-subject entry point
# --------------------------------------------------------------------------- #
def preprocess_subject(
    subject_id: str, output_dir: Path, *,
    sf: float = SF, channels: Sequence[str] = DEFAULT_CHANNELS,
    bands: Mapping[str, Tuple[float, float]] = BANDS,
    stages: Sequence[str] = DEFAULT_STAGES,
    stage_codes: Mapping[str, Tuple[int, ...]] = STAGE_CODES,
    min_bout_sec: float = MIN_BOUT_SEC, window_sec: float = WINDOW_SEC,
) -> None:
    """Load one subject once and preprocess every requested channel.

    The hypnogram comes from ``loader.annotations`` -- by default this is
    already sourced from the subject's Hypnodensity CSV (see
    :class:`~infraslow.io.psg_loader.BioserenityPSGLoader`'s default
    ``annotation_loader``) -- not raw EDF-embedded annotations.
    """
    loader = BioserenityPSGLoader(
        subject_id=subject_id, sf=sf, requested_channels=list(channels)
    ).load()
    hypnogram = _stages_to_int(
        _extract_epoch_stages(loader.annotations, stage_column="stage"), DEFAULT_STAGE_MAP
    )

    for ch in channels:
        try:
            preprocess_channel(
                loader, subject_id, ch, hypnogram, output_dir,
                bands=bands, stages=stages, stage_codes=stage_codes,
                min_bout_sec=min_bout_sec, window_sec=window_sec,
            )
        except Exception:  # noqa: BLE001 - one bad channel must not sink the others
            logger.exception("Channel %s failed for subject %s", ch, subject_id)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Shard resolution happens before logging setup so each shard gets its own
    # log file -- a thousand array tasks all appending to one shared
    # progress.log would interleave/contend on the same file.
    logs_dir = args.output_dir / LOGS_DIRNAME
    if args.subject:
        log_path = logs_dir / "progress.log"
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
        subjects = list_valid_subjects(args.metadata, args.metadata2, args.edf_dir, args.hypno_dir)
        subjects = subjects[shard_index::num_shards]
        if args.limit:
            subjects = subjects[: args.limit]
        logger.info(f"shard {shard_index}/{num_shards}: {len(subjects)} subject(s)")

    logger.info(f"channels={args.channels} stages={args.stages} sf={args.sf} "
                f"output_dir={args.output_dir}")

    for subject_id in subjects:
        try:
            preprocess_subject(
                subject_id, args.output_dir, sf=args.sf, channels=args.channels,
                stages=args.stages, min_bout_sec=args.min_bout_sec, window_sec=args.window_sec,
            )
            logger.info(f"done: subject={subject_id} -> {args.output_dir / DATA_DIRNAME / subject_id}")
        except Exception:  # noqa: BLE001 - one bad subject must not sink the shard
            logger.exception(f"Subject {subject_id} failed")

    logger.info(f"finished: {len(subjects)} subject(s) attempted")


if __name__ == "__main__":
    main()
