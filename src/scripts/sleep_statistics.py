#!/usr/bin/env python
"""Per-subject YASA sleep statistics for every valid subject (one row per
subject id).

Subjects are discovered the same way as ``preprocessing.py`` -- via
:func:`preprocessing.list_valid_subjects` (same two metadata CSVs, same
EDF/U-Sleep hypnodensity directories) -- so this script's cohort always
matches ``preprocessing.py``'s, instead of drifting from whatever happens to
be sitting in an output directory.

Only the hypnogram is needed to compute sleep statistics, so this reads each
subject's ``--channel`` U-Sleep hypnodensity directly
(:func:`~infraslow.io.usleep_hypnodensity.load_usleep_hypnodensity`, reduced
to ``--hypno-epoch-sec``-wide epochs the same way ``preprocessing.py`` does)
instead of loading the subject's EDF through ``BioserenityPSGLoader`` -- far
cheaper, since it skips opening 100k+ EEG recordings just to read their
attached hypnogram. U-Sleep hypnodensity is scored per-channel (channels can
disagree), so this reports one channel's statistics per subject, not a
subject-wide average across channels -- pick a different ``--channel`` to
compare.

Saves one CSV to ``--output`` with columns ``[id, <yasa.sleep_statistics()
keys>]`` (TIB, SPT, WASO, TST, N1, N2, N3, REM, NREM, SOL, Lat_N1, Lat_N2,
Lat_N3, Lat_REM, %N1, %N2, %N3, %REM, %NREM, SE, SME -- see
``yasa.sleep_statistics``'s docstring for definitions). A subject with a
missing/empty/corrupt U-Sleep hypnodensity file for ``--channel`` is logged
and skipped, not fatal to the run.

Subjects are farmed out across ``--workers`` processes (default:
``$SLURM_CPUS_PER_TASK``, else 1) via ``ProcessPoolExecutor`` -- each
subject's CSV read + ``yasa.sleep_statistics`` call is independent, so this
scales close to linearly with CPU count; pass ``--cpus-per-task`` > 1 to
``run_sleep_statistics.sbatch`` to use it.

Run via Slurm, not the login node, from this file's own directory
(``src/scripts/``) with ``src/`` on ``PYTHONPATH`` so ``infraslow`` resolves
(the package is not pip-installed), e.g.::

    export PYTHONPATH=/home/users/chaisaen/infraslow/src
    python3 sleep_statistics.py \\
        --output /scratch/users/chaisaen/processed_data/sleep_statistics.csv

See ``run_sleep_statistics.sbatch`` to submit this as a Slurm job.
"""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yasa

from infraslow.constants import (
    DEFAULT_EDF_DIR,
    DEFAULT_HYPNOGRAM_EPOCH_SEC,
    DEFAULT_METADATA,
    DEFAULT_METADATA2,
    DEFAULT_STAGE_MAP,
    DEFAULT_USLEEP_EPOCH_SEC,
    DEFAULT_USLEEP_HYPNODENSITY_DIR,
)
from infraslow.io import usleep_hypnodensity as uh
from infraslow.io.utils import progress_iter
from infraslow.processing.spindle import _stages_to_int
from preprocessing import _check_usleep_file_usable, list_valid_subjects

logger = logging.getLogger(__name__)


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None else val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", default=DEFAULT_METADATA, help="Primary metadata CSV path.")
    p.add_argument("--metadata2", default=DEFAULT_METADATA2,
                    help="Second metadata CSV path, combined with --metadata by ID.")
    p.add_argument("--edf-dir", default=DEFAULT_EDF_DIR, help="Directory of {id}.edf files.")
    p.add_argument("--usleep-dir", default=os.path.expandvars(
                        _env_str("USLEEP_DIR", DEFAULT_USLEEP_HYPNODENSITY_DIR)),
                    help="U-Sleep hypnodensity directory, one subfolder per subject "
                         "(env: USLEEP_DIR)")
    p.add_argument("--channel", default=_env_str("CHANNEL", "C3"),
                    help="EEG channel whose U-Sleep hypnodensity to use -- statistics are "
                         "per-channel, not a subject-wide average (env: CHANNEL)")
    p.add_argument("--hypno-epoch-sec", type=float,
                    default=float(_env_str("HYPNO_EPOCH_SEC", str(DEFAULT_HYPNOGRAM_EPOCH_SEC))),
                    help="Width (s) of the epoch the native 1-s U-Sleep hypnodensity is "
                         "averaged into before computing statistics; sets "
                         "yasa.sleep_statistics's sf_hyp (1/hypno_epoch_sec) (env: HYPNO_EPOCH_SEC)")
    p.add_argument(
        "--output", type=Path,
        default=Path(os.path.expandvars(
            _env_str("OUTPUT", "/scratch/users/chaisaen/processed_data/sleep_statistics.csv"))),
        help="Output CSV path (env: OUTPUT).",
    )
    p.add_argument("--num-shards", type=int, default=1,
                    help="Split the valid-subject list into this many pieces, matching "
                         "preprocessing.py's --num-shards convention -- use with "
                         "--shard-indices to reproduce the exact subject set a "
                         "preprocessing.py job array touched. Default 1 (no sharding).")
    p.add_argument("--shard-indices", default="0",
                    help="Which shard(s) to include: an inclusive range ('0-9') or a "
                         "comma-separated list ('0,3,7'). Each shard i contributes "
                         "subjects[i::num_shards], --limit applied per shard -- same "
                         "per-shard semantics as preprocessing.py's --limit. Default '0' "
                         "(with the default --num-shards=1, this is the whole list, "
                         "unchanged from --limit alone).")
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most this many subjects per shard (quick tests).")
    p.add_argument(
        "--workers", type=int,
        default=int(_env_str("WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="Parallel worker processes (env: WORKERS, falls back to "
             "$SLURM_CPUS_PER_TASK, else 1).",
    )
    return p.parse_args()


def _parse_shard_indices(spec: str) -> List[int]:
    """Parse a ``--shard-indices`` spec into a sorted list of ints: an inclusive
    range (``"0-9"`` -> ``[0, 1, ..., 9]``) or a comma-separated list
    (``"0,3,7"`` -> ``[0, 3, 7]``); a bare number (``"5"``) is a single shard."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(part) for part in spec.split(","))


def _select_shard_subjects(
    subjects: Sequence[str], *, num_shards: int, shard_indices: Sequence[int],
    per_shard_limit: Optional[int],
) -> List[str]:
    """Union, in ``shard_indices`` order, of ``subjects[i::num_shards][:per_shard_limit]``
    for each shard ``i`` -- replicates exactly what a preprocessing.py job array with
    the same ``--num-shards``/``--limit`` selects across those shard indices, so a
    single-task ``sleep_statistics.py`` run can be pointed at the identical subject
    set. With the defaults (``num_shards=1``, ``shard_indices=[0]``), this reduces to
    plain ``subjects[:per_shard_limit]`` -- unchanged from ``--limit`` alone."""
    selected: List[str] = []
    for shard in shard_indices:
        shard_subjects = list(subjects[shard::num_shards])
        if per_shard_limit:
            shard_subjects = shard_subjects[:per_shard_limit]
        selected.extend(shard_subjects)
    return selected


def compute_subject_stats(
    subject_id: str, usleep_dir: str, channel: str, *, hypno_epoch_sec: float,
) -> Dict[str, float]:
    """One subject's ``yasa.sleep_statistics`` dict, sourced from ``channel``'s own
    U-Sleep hypnodensity, reduced from its native 1-s resolution to
    ``hypno_epoch_sec``-wide epochs (matching ``preprocessing.py``'s convention)."""
    _check_usleep_file_usable(subject_id, channel, usleep_dir)
    t_hyp, probs = uh.load_usleep_hypnodensity(subject_id, channel, base_dir=usleep_dir)
    _t_epoch, _probs_epoch, stage_epoch = uh.hypnodensity_to_epoch_hypnogram(
        t_hyp, probs, epoch_sec=hypno_epoch_sec, src_epoch_sec=DEFAULT_USLEEP_EPOCH_SEC,
    )
    hypno_int = _stages_to_int(stage_epoch, DEFAULT_STAGE_MAP)
    return yasa.sleep_statistics(hypno_int, sf_hyp=1.0 / hypno_epoch_sec)


def _compute_or_error(
    args: Tuple[str, str, str, float],
) -> Tuple[str, Optional[Dict[str, float]], Optional[str]]:
    """``ProcessPoolExecutor``-friendly wrapper: exceptions can't cross the
    process boundary as live objects, so catch here and ship back the
    formatted traceback text instead for the parent to log.
    """
    subject_id, usleep_dir, channel, hypno_epoch_sec = args
    try:
        stats = compute_subject_stats(
            subject_id, usleep_dir, channel, hypno_epoch_sec=hypno_epoch_sec,
        )
        return subject_id, stats, None
    except Exception:  # noqa: BLE001 - one bad subject must not sink the run
        return subject_id, None, traceback.format_exc()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    subjects = list_valid_subjects(args.metadata, args.metadata2, args.edf_dir, args.usleep_dir)
    shard_indices = _parse_shard_indices(args.shard_indices)
    subjects = _select_shard_subjects(
        subjects, num_shards=args.num_shards, shard_indices=shard_indices,
        per_shard_limit=args.limit,
    )
    workers = max(1, args.workers)
    logger.info(f"{len(subjects)} valid subject(s) (num_shards={args.num_shards} "
                f"shard_indices={shard_indices} limit={args.limit}); channel={args.channel} "
                f"hypno_epoch_sec={args.hypno_epoch_sec} workers={workers}")

    tasks = [
        (subject_id, args.usleep_dir, args.channel, args.hypno_epoch_sec)
        for subject_id in subjects
    ]
    # Each task is one small .npy read + a cheap yasa call, so a large chunksize
    # keeps IPC overhead from dominating over ~170k subjects.
    chunksize = max(1, len(tasks) // (workers * 20)) if tasks else 1

    rows: List[Dict[str, object]] = []
    n_failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_compute_or_error, tasks, chunksize=chunksize)
        for subject_id, stats, error in progress_iter(
            results, len(tasks), enabled=True, desc="sleep_statistics",
        ):
            if error is not None:
                logger.error(f"Subject {subject_id} failed:\n{error}")
                n_failed += 1
                continue
            rows.append({"id": subject_id, **stats})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    logger.info(f"wrote {len(rows)} subject(s) ({n_failed} failed) -> {args.output}")


if __name__ == "__main__":
    main()
