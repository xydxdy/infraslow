#!/usr/bin/env python
"""Slurm-ready runner for the Bioserenity all-metrics pipeline.

Command-line, parallel reproduction of ``src/all_metrics_calculation.ipynb``.
For every subject that has both an EDF and a hypnodensity file on ``$OAK`` it
computes metadata + YASA sleep statistics + N2/N3/NREM infraslow metrics, and
writes one merged CSV (plus a CSV of per-subject failures) -- and, alongside
it, each subject's *empirical* (pre-fit) infraslow spectrum per stage as its
own ``{ID}_spectra.npz`` under ``--spectra-dir``, so a true grand-average curve
can be plotted later instead of one reconstructed from the fitted Gaussian
parameters alone. One file per subject (not one consolidated file) because the
cohort is 100k+ subjects -- a single growing ``.npz`` rewritten on every
checkpoint would be O(n) work per checkpoint; a small per-subject file is a
one-time write with no rewrite cost as the cohort grows.

Metadata is read from *two* source CSVs and combined by subject ``ID`` (a
subject is kept if it appears in *either* file; see :func:`~infraslow.processing.
all_metrics.combine_bioserenity_metadata`), matching ``src/match_cohort.py``.

The analysis itself is unchanged from the notebook: all computation is delegated
to :mod:`infraslow.processing.all_metrics` (the same functions the notebook
calls). This script only adds a CLI, per-subject process-level parallelism,
resumable checkpointing, OOM-resilient execution, and HPC-friendly logging.

Run (matches the accompanying Slurm script)::

    python -u src/run_all_metrics.py --workers ${SLURM_CPUS_PER_TASK}

or, with everything defaulted::

    python -u src/run_all_metrics.py --workers 10

Memory / OOM safety
-------------------
Each subject reads a full-night EDF; 10 such loads at once can exceed a 20 GB
job. To stay alive under memory pressure:

* **All heavy work runs in worker subprocesses** -- the driver process never
  loads an EDF, so a subject that OOMs kills only its worker, never the run.
* Concurrency is **capped to fit the job's memory** (``--mem-per-worker-gb``,
  read from ``$SLURM_MEM_PER_NODE``); set ``--mem-per-worker-gb 0`` to disable.
* Work is done in **recycled chunks** (a fresh pool per chunk) so worker memory
  is released periodically (Python 3.10 has no ``max_tasks_per_child``).
* A worker killed mid-subject (``BrokenProcessPool``) is caught; that subject is
  **retried in isolation** (its own 1-worker pool with the full node memory) and,
  if it still fails, recorded as an error while the run continues.

Notes on differences from the prompt:
* The hypnodensity files on disk are ``{id}_Hypnodensity.csv`` (verified), not
  ``{id}._Hypnodensity.csv``; the latter would match no files, so the correct
  suffix is used.
* The notebook/module percentage columns ``%{stage}_bouts`` are renamed to
  ``{stage}_percent_bouts`` in the output CSV, as requested.
"""

from __future__ import annotations

# Keep every worker single-threaded for BLAS/OpenMP so process-level parallelism
# does not oversubscribe cores (and inflate memory). Set before numpy is imported
# (directly or via infraslow). The Slurm script also exports these; ``setdefault``
# respects an existing value.
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# The package lives beside this script (``src/infraslow``); make sure it is
# importable no matter the current working directory or multiprocessing start
# method.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from infraslow.io.hypnodensity import DEFAULT_HYPNODENSITY_SUFFIX
from infraslow.processing.all_metrics import (
    CHANNEL,
    INFRASLOW_STAGES,
    METADATA_COLUMNS,
    SF,
    all_metric_columns,
    calculate_subject_all_metrics,
    combine_bioserenity_metadata,
    find_valid_bioserenity_subjects,
    load_bioserenity_metadata,
)

logger = logging.getLogger("run_all_metrics")

# --------------------------------------------------------------------------- #
# Defaults / constants
# --------------------------------------------------------------------------- #
DEFAULT_METADATA = "$OAK/psg/Bioserenity/Excel/Morpheus_Data_All5.csv"
DEFAULT_METADATA2 = "$OAK/psg/Bioserenity/Excel/bioserenity_metadata3.csv"
DEFAULT_EDF_DIR = "$OAK/psg/Bioserenity/edf"
DEFAULT_HYPNO_DIR = "$OAK/psg/Bioserenity/Sleep_Staging"
DEFAULT_OUTPUT = "$SCRATCH/results/all_metrics.csv"
DEFAULT_ERROR_OUTPUT = "$SCRATCH/results/all_metrics_errors.csv"
DEFAULT_SPECTRA_DIR = "$SCRATCH/results/spectra"
EDF_SUFFIX = ".edf"
HYPNO_SUFFIX = DEFAULT_HYPNODENSITY_SUFFIX  # "_Hypnodensity.csv"

LOG_EVERY = 25              # progress log cadence (completed subjects)
DEFAULT_CHUNK_SIZE = 40     # subjects per recycled worker pool
DEFAULT_MEM_PER_WORKER_GB = 3.0  # concurrency cap so workers fit job memory

ERROR_COLUMNS = ["ID", "error_type", "error_message"]

# ``%{stage}_bouts`` (module/notebook) -> ``{stage}_percent_bouts`` (this CSV).
_PERCENT_RENAME: Dict[str, str] = {
    f"%{stage}_bouts": f"{stage}_percent_bouts" for stage in INFRASLOW_STAGES
}

# A single unit of work: (subject_id, metadata_row, edf_path, hypno_path, params, spectra_dir).
Task = Tuple[str, Dict[str, Any], str, str, Dict[str, Any], str]
# A worker outcome: (status, subject_id, payload).
Outcome = Tuple[str, str, Dict[str, Any]]


def final_columns() -> List[str]:
    """Output column order: the module schema with percent columns renamed."""
    return [_PERCENT_RENAME.get(col, col) for col in all_metric_columns()]


# --------------------------------------------------------------------------- #
# Worker (top-level so it is picklable by ProcessPoolExecutor)
# --------------------------------------------------------------------------- #
def process_subject(task: Task) -> Outcome:
    """Compute all metrics for one subject; never raises inside the worker.

    Returns ``("ok", subject_id, metrics_dict)`` on success, or
    ``("error", subject_id, error_record_dict)`` on any handled failure.
    (A hard crash such as an OOM kill cannot be caught here -- the driver
    handles that as a ``BrokenProcessPool`` casualty.)

    On success, this subject's empirical (pre-fit) infraslow spectra are also
    written directly to their own ``.npz`` (see :func:`write_subject_spectra`)
    -- done here, inside the worker, so the tiny arrays never need to be shipped
    back through the process pool just to be written by the driver.
    """
    subject_id, metadata_row, edf_path, hypno_path, params, spectra_dir = task
    try:
        spectra: Dict[str, Dict[str, Any]] = {}
        metrics = calculate_subject_all_metrics(
            metadata_row, Path(edf_path), Path(hypno_path), spectra_out=spectra, **params
        )
        try:
            write_subject_spectra(Path(spectra_dir), subject_id, spectra)
        except Exception as exc:  # noqa: BLE001 - a spectra-write failure must not lose the metrics
            logger.warning("Could not write spectra for %s: %s", subject_id, exc)
        return ("ok", subject_id, metrics)
    except Exception as exc:  # noqa: BLE001 - isolate every per-subject failure
        return ("error", subject_id, _error_record(subject_id, exc))


def _error_record(subject_id: str, exc: BaseException, *, prefix: str = "") -> Dict[str, str]:
    return {
        "ID": subject_id,
        "error_type": type(exc).__name__,
        "error_message": f"{prefix}{exc}",
    }


# --------------------------------------------------------------------------- #
# Parallel execution helpers (chunked, recycled, OOM-resilient)
# --------------------------------------------------------------------------- #
def _chunks(seq: List[Task], size: int) -> Iterator[List[Task]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _run_chunk(chunk: List[Task], workers: int) -> Tuple[List[Outcome], List[Task]]:
    """Run one chunk in a fresh pool. Return (outcomes, casualties).

    ``casualties`` are tasks whose worker died mid-run (e.g. OOM kill -> the
    future raises ``BrokenProcessPool``); the pool is discarded afterwards so its
    memory is fully released.
    """
    outcomes: List[Outcome] = []
    casualties: List[Task] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(process_subject, t): t for t in chunk}
        for future, task in future_to_task.items():
            try:
                outcomes.append(future.result())
            except Exception:  # noqa: BLE001 - BrokenProcessPool / unpickling: worker died
                casualties.append(task)
    return outcomes, casualties


def _run_isolated(task: Task) -> Outcome:
    """Run a single task in its own 1-worker pool (full node memory, crash-safe).

    Used to retry a casualty: an OOM here kills only this transient subprocess,
    never the driver, and is recorded as an error.
    """
    subject_id = task[0]
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            return executor.submit(process_subject, task).result()
    except Exception as exc:  # noqa: BLE001 - the isolated worker also died
        return (
            "error",
            subject_id,
            _error_record(subject_id, exc, prefix="worker terminated (likely OOM): "),
        )


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def build_results_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Assemble result dicts into the final, renamed, ordered DataFrame."""
    base_cols = all_metric_columns()
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=base_cols)
    df = df.reindex(columns=base_cols).rename(columns=_PERCENT_RENAME)
    return df.reindex(columns=final_columns())


def write_outputs(
    output_path: Path,
    error_path: Path,
    prior_df: Optional[pd.DataFrame],
    result_records: List[Dict[str, Any]],
    error_records: List[Dict[str, str]],
) -> None:
    """Write (resumed prior + new) results and the current errors to disk.

    Writes to a temporary sibling first, then atomically replaces the target, so
    a crash mid-write never truncates an existing checkpoint.
    """
    new_df = build_results_frame(result_records)
    if prior_df is not None and not prior_df.empty:
        combined = pd.concat([prior_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.reindex(columns=final_columns())
    _atomic_to_csv(combined, output_path)
    _atomic_to_csv(pd.DataFrame(error_records, columns=ERROR_COLUMNS), error_path)


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Spectra (.npz) helper -- the empirical (pre-fit) grand-average curve
# --------------------------------------------------------------------------- #
def write_subject_spectra(
    spectra_dir: Path, subject_id: str, spectra: Dict[str, Dict[str, Any]]
) -> None:
    """Write one subject's empirical infraslow spectra to ``{spectra_dir}/{ID}_spectra.npz``.

    One file per subject, not one consolidated file: with a 100k+-subject
    cohort, rewriting a single growing ``.npz`` on every checkpoint would be
    O(n) work per checkpoint. Each subject's file is written exactly once and
    never touched again, so total write cost stays O(1) per subject regardless
    of cohort size (see :func:`process_subject`, which calls this).

    Keys are ``{stage}_freqs`` / ``{stage}_corr_mean`` for whichever stages had
    usable bout data (see :func:`~infraslow.processing.all_metrics.
    calculate_stage_infraslow`'s ``spectrum_out``); a subject with no usable
    stage data writes nothing.
    """
    if not spectra:
        return
    arrays: Dict[str, np.ndarray] = {}
    for stage, entry in spectra.items():
        arrays[f"{stage}_freqs"] = entry["freqs"]
        arrays[f"{stage}_corr_mean"] = entry["corr_mean"]

    path = spectra_dir / f"{subject_id}_spectra.npz"
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# CLI / configuration
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Bioserenity sleep + infraslow metrics for all valid subjects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes. Defaults to $SLURM_CPUS_PER_TASK, else 1.",
    )
    parser.add_argument("--metadata", default=DEFAULT_METADATA, help="Primary metadata CSV path.")
    parser.add_argument(
        "--metadata2", default=DEFAULT_METADATA2,
        help="Second metadata CSV path, combined with --metadata by ID (OR, not intersection).",
    )
    parser.add_argument("--edf-dir", default=DEFAULT_EDF_DIR, help="Directory of {id}.edf files.")
    parser.add_argument(
        "--hypno-dir", default=DEFAULT_HYPNO_DIR,
        help="Directory of {id}_Hypnodensity.csv files.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output metrics CSV.")
    parser.add_argument(
        "--error-output", default=DEFAULT_ERROR_OUTPUT, help="Failed-subjects CSV.",
    )
    parser.add_argument(
        "--spectra-dir", default=DEFAULT_SPECTRA_DIR,
        help="Directory of one {ID}_spectra.npz per subject (freqs + baseline-corrected "
             "relative power per stage), for the true grand-average curve.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many valid subjects (for quick tests).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute everything, ignoring any existing output CSV.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Subjects per recycled worker pool (memory is released between chunks).",
    )
    parser.add_argument(
        "--mem-per-worker-gb", type=float, default=DEFAULT_MEM_PER_WORKER_GB,
        help="Cap concurrency so workers fit job memory. 0 disables the cap.",
    )
    return parser.parse_args(argv)


def resolve_workers(cli_workers: Optional[int]) -> int:
    """Workers = --workers, else $SLURM_CPUS_PER_TASK, else 1."""
    if cli_workers and cli_workers > 0:
        return cli_workers
    env = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return 1


def detect_job_mem_gb() -> Optional[float]:
    """Total memory (GB) Slurm granted this job, if discoverable from the env."""
    per_node = os.environ.get("SLURM_MEM_PER_NODE", "").strip()  # MB
    if per_node.isdigit() and int(per_node) > 0:
        return int(per_node) / 1024.0
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "").strip()      # MB
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if per_cpu.isdigit() and cpus.isdigit() and int(per_cpu) > 0 and int(cpus) > 0:
        return int(per_cpu) * int(cpus) / 1024.0
    return None


def cap_workers_for_memory(workers: int, mem_per_worker_gb: float) -> int:
    """Reduce ``workers`` so ``workers * mem_per_worker_gb`` fits the job memory."""
    if mem_per_worker_gb <= 0:
        return workers
    job_mem_gb = detect_job_mem_gb()
    if job_mem_gb is None:
        logger.info(
            "could not detect job memory; not capping workers "
            "(set --mem-per-worker-gb 0 to silence)"
        )
        return workers
    cap = max(1, int(job_mem_gb // mem_per_worker_gb))
    if cap < workers:
        logger.warning(
            "capping workers %d -> %d to fit %.1f GB at %.1f GB/worker "
            "(override with --mem-per-worker-gb)",
            workers, cap, job_mem_gb, mem_per_worker_gb,
        )
        return cap
    logger.info("job memory %.1f GB fits %d worker(s) at %.1f GB each",
                job_mem_gb, workers, mem_per_worker_gb)
    return workers


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("infraslow").setLevel(logging.INFO)


def _expand(path_str: str) -> Path:
    """Expand ``$VARS`` and ``~`` in a path string."""
    return Path(os.path.expandvars(os.path.expanduser(path_str)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()

    workers = resolve_workers(args.workers)
    workers = cap_workers_for_memory(workers, args.mem_per_worker_gb)
    chunk_size = max(1, args.chunk_size)

    metadata_path = _expand(args.metadata)
    metadata_path2 = _expand(args.metadata2)
    edf_dir = _expand(args.edf_dir)
    hypno_dir = _expand(args.hypno_dir)
    output_path = _expand(args.output)
    error_path = _expand(args.error_output)
    spectra_dir = _expand(args.spectra_dir)

    # Requirement: create results/ and logs/ up front.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger.info("workers        : %d", workers)
    logger.info("chunk size     : %d", chunk_size)
    logger.info("metadata       : %s", metadata_path)
    logger.info("metadata2      : %s", metadata_path2)
    logger.info("edf dir        : %s", edf_dir)
    logger.info("hypno dir      : %s", hypno_dir)
    logger.info("output         : %s", output_path)
    logger.info("error output   : %s", error_path)
    logger.info("spectra dir    : %s", spectra_dir)

    # --- 1. Metadata (union of both sources by ID) + valid-subject discovery #
    metadata = combine_bioserenity_metadata(
        load_bioserenity_metadata(metadata_path),
        load_bioserenity_metadata(metadata_path2),
    )
    valid = find_valid_bioserenity_subjects(
        metadata, edf_dir, hypno_dir, hypnodensity_suffix=HYPNO_SUFFIX,
    )
    availability = valid.attrs.get("availability", {})
    n_metadata = availability.get("n_metadata", len(metadata))
    n_valid = availability.get("n_valid", len(valid))
    logger.info("subjects in metadata      : %d", n_metadata)
    logger.info("subjects with both files  : %d", n_valid)

    if args.limit is not None:
        valid = valid.head(args.limit)
        logger.info("limited to first %d valid subject(s)", len(valid))

    # --- 2. Resume: skip subjects already in an existing output ----------- #
    prior_df: Optional[pd.DataFrame] = None
    done_ids: set = set()
    if output_path.exists() and not args.overwrite:
        try:
            prior_df = pd.read_csv(output_path, dtype={"ID": str})
            done_ids = set(prior_df["ID"].astype(str))
            logger.info(
                "resuming: %d subject(s) already in %s will be skipped",
                len(done_ids), output_path,
            )
        except Exception as exc:  # noqa: BLE001 - a corrupt prior file must not abort
            logger.warning("could not read existing output (%s); starting fresh", exc)
            prior_df = None
            done_ids = set()
    elif output_path.exists() and args.overwrite:
        logger.info("--overwrite set: existing %s will be replaced", output_path)

    # --- 3. Build the task list ------------------------------------------- #
    # (Spectra need no separate resume handling: each subject's {ID}_spectra.npz
    # is written once by the worker; a subject skipped via done_ids just leaves
    # its prior-run file untouched on disk.)
    params = {"sf": SF, "channel": CHANNEL, "require_spindle": True}
    tasks: List[Task] = []
    for _, row in valid.iterrows():
        subject_id = str(row["ID"]).strip()
        if subject_id in done_ids:
            continue
        metadata_row = {col: row.get(col) for col in METADATA_COLUMNS}
        metadata_row["ID"] = subject_id
        edf_path = str(edf_dir / f"{subject_id}{EDF_SUFFIX}")
        hypno_path = str(hypno_dir / f"{subject_id}{HYPNO_SUFFIX}")
        tasks.append((subject_id, metadata_row, edf_path, hypno_path, params, str(spectra_dir)))

    n_to_process = len(tasks)
    logger.info("subjects to process this run: %d", n_to_process)

    # --- 4. Process (chunked + recycled + OOM-resilient) ------------------ #
    result_records: List[Dict[str, Any]] = []
    error_records: List[Dict[str, str]] = []
    processed = 0

    def handle(outcome: Outcome) -> None:
        nonlocal processed
        processed += 1
        status, subject_id, payload = outcome
        if status == "ok":
            result_records.append(payload)
        else:
            error_records.append(payload)
            logger.warning(
                "FAIL %s: %s: %s",
                subject_id, payload["error_type"], payload["error_message"],
            )
        if processed % LOG_EVERY == 0:
            logger.info(
                "progress %d/%d  (ok=%d, failed=%d)",
                processed, n_to_process, len(result_records), len(error_records),
            )

    def checkpoint() -> None:
        write_outputs(output_path, error_path, prior_df, result_records, error_records)

    casualties: List[Task] = []
    if n_to_process == 0:
        logger.info("nothing to process")
    elif workers <= 1:
        # Even "sequential" work runs in an isolated subprocess so an OOM cannot
        # kill the driver; the run always finishes and checkpoints.
        logger.info("running one subject at a time (isolated subprocesses)")
        for task in tasks:
            handle(_run_isolated(task))
            if processed % chunk_size == 0:
                checkpoint()
                logger.info("checkpoint written at %d/%d", processed, n_to_process)
    else:
        logger.info("running with %d worker(s) in chunks of %d", workers, chunk_size)
        for chunk in _chunks(tasks, chunk_size):
            outcomes, chunk_casualties = _run_chunk(chunk, workers)
            for outcome in outcomes:
                handle(outcome)
            if chunk_casualties:
                casualties.extend(chunk_casualties)
                logger.warning(
                    "%d subject(s) lost to worker termination this chunk "
                    "(likely OOM); queued for isolated retry",
                    len(chunk_casualties),
                )
            # Checkpoint after every chunk so a killed job resumes cleanly.
            checkpoint()
            logger.info("checkpoint written at %d/%d", processed, n_to_process)

    # --- 4b. Retry casualties one at a time, fully isolated --------------- #
    if casualties:
        logger.warning("retrying %d casualty subject(s) in isolation...", len(casualties))
        for task in casualties:
            handle(_run_isolated(task))
            checkpoint()

    # --- 5. Final write --------------------------------------------------- #
    checkpoint()

    # --- 6. Summary ------------------------------------------------------- #
    n_prior = 0 if prior_df is None else len(prior_df)
    total_success = n_prior + len(result_records)
    print("=" * 60)
    print(f"Total subjects in metadata:    {n_metadata}")
    print(f"Subjects with required files:  {n_valid}")
    print(f"Successful:                    {total_success}")
    print(f"Failed:                        {len(error_records)}")
    print(f"Output CSV:                    {output_path}")
    print(f"Error CSV:                     {error_path}")
    print(f"Spectra dir ({{ID}}_spectra.npz): {spectra_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
