#!/usr/bin/env python
"""Per-subject YASA sleep statistics for every valid subject (one row per
subject id).

Subjects are discovered the same way as ``preprocessing.py`` -- via
:func:`preprocessing.list_valid_subjects` (same two metadata CSVs, same
EDF/Hypnodensity directories) -- so this script's cohort always matches
``preprocessing.py``'s, instead of drifting from whatever happens to be
sitting in an output directory.

Only the hypnogram is needed to compute sleep statistics, so this reads each
subject's Hypnodensity CSV directly (:func:`~infraslow.io.hypnodensity.
hypnodensity_to_annotations`) instead of loading the subject's EDF through
``BioserenityPSGLoader`` -- far cheaper, since it skips opening 100k+ EEG
recordings just to read their attached hypnogram.

Saves one CSV to ``--output`` with columns ``[id, <yasa.sleep_statistics()
keys>]`` (TIB, SPT, WASO, TST, N1, N2, N3, REM, NREM, SOL, Lat_N1, Lat_N2,
Lat_N3, Lat_REM, %N1, %N2, %N3, %REM, %NREM, SE, SME -- see
``yasa.sleep_statistics``'s docstring for definitions). A subject with a
missing/corrupt Hypnodensity CSV is logged and skipped, not fatal to the run.

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
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yasa

from infraslow.constants import (
    DEFAULT_EDF_DIR,
    DEFAULT_EPOCH_SEC,
    DEFAULT_HYPNO_DIR,
    DEFAULT_HYPNODENSITY_SUFFIX,
    DEFAULT_METADATA,
    DEFAULT_METADATA2,
    DEFAULT_STAGE_MAP,
)
from infraslow.io.hypnodensity import hypnodensity_to_annotations
from infraslow.io.utils import progress_iter
from infraslow.processing.spindle import _extract_epoch_stages, _stages_to_int
from preprocessing import list_valid_subjects

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
    p.add_argument("--hypno-dir", default=DEFAULT_HYPNO_DIR,
                    help="Directory of {id}_Hypnodensity.csv files.")
    p.add_argument("--epoch-sec", type=float, default=DEFAULT_EPOCH_SEC,
                    help="Seconds per scored epoch -- sets yasa.sleep_statistics's "
                         "sf_hyp (1/epoch_sec).")
    p.add_argument(
        "--output", type=Path,
        default=Path(os.path.expandvars(
            _env_str("OUTPUT", "/scratch/users/chaisaen/processed_data/sleep_statistics.csv"))),
        help="Output CSV path (env: OUTPUT).",
    )
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most this many subjects (quick tests).")
    p.add_argument(
        "--workers", type=int,
        default=int(_env_str("WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="Parallel worker processes (env: WORKERS, falls back to "
             "$SLURM_CPUS_PER_TASK, else 1).",
    )
    return p.parse_args()


def compute_subject_stats(
    subject_id: str, hypno_dir: Path, *, epoch_sec: float,
) -> Dict[str, float]:
    """One subject's ``yasa.sleep_statistics`` dict, sourced from its Hypnodensity CSV."""
    csv_path = hypno_dir / f"{subject_id}{DEFAULT_HYPNODENSITY_SUFFIX}"
    annotations = hypnodensity_to_annotations(csv_path)
    stages = _extract_epoch_stages(annotations, stage_column="stage")
    hypno_int = _stages_to_int(stages, DEFAULT_STAGE_MAP)
    return yasa.sleep_statistics(hypno_int, sf_hyp=1.0 / epoch_sec)


def _compute_or_error(
    args: Tuple[str, Path, float],
) -> Tuple[str, Optional[Dict[str, float]], Optional[str]]:
    """``ProcessPoolExecutor``-friendly wrapper: exceptions can't cross the
    process boundary as live objects, so catch here and ship back the
    formatted traceback text instead for the parent to log.
    """
    subject_id, hypno_dir, epoch_sec = args
    try:
        return subject_id, compute_subject_stats(subject_id, hypno_dir, epoch_sec=epoch_sec), None
    except Exception:  # noqa: BLE001 - one bad subject must not sink the run
        return subject_id, None, traceback.format_exc()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    hypno_dir = Path(os.path.expandvars(args.hypno_dir))
    subjects = list_valid_subjects(args.metadata, args.metadata2, args.edf_dir, args.hypno_dir)
    if args.limit:
        subjects = subjects[: args.limit]
    workers = max(1, args.workers)
    logger.info(f"{len(subjects)} valid subject(s); workers={workers}")

    tasks = [(subject_id, hypno_dir, args.epoch_sec) for subject_id in subjects]
    # Each task is one small CSV read + a cheap yasa call, so a large chunksize
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
