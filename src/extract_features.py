#!/usr/bin/env python
"""Slurm-ready runner for the Bioserenity all-metrics pipeline (v2).

For every subject that has both an EDF and a hypnodensity file on ``$OAK`` it
computes metadata + YASA sleep statistics, and writes one merged
``metadata.csv`` (plus a CSV of per-subject failures) -- and, alongside it,
per-channel infraslow (~0.02 Hz sigma-power) spectra, spindles, and bouts for
six EEG channels (``F3, F4, C3, C4, O1, O2``) across three sleep-stage groups
(``N2, N3, NREM``), one ``{ID}.npz`` per subject under ``--npz-dir/{stage}/``.
One file per subject per stage (not one consolidated file) because the cohort
is 100k+ subjects -- a single growing ``.npz`` rewritten on every checkpoint
would be O(n) work per checkpoint; a small per-subject file is a one-time
write with no rewrite cost as the cohort grows.

Metadata is read from *two* source CSVs and combined by subject ``ID`` (a
subject is kept if it appears in *either* file; see :func:`~infraslow.processing.
subject_pipeline.combine_bioserenity_metadata`), matching ``src/match_cohort.py``.

The analysis itself is delegated to :mod:`infraslow.processing.subject_pipeline`
(:func:`~infraslow.processing.subject_pipeline.calculate_features`). This script
only adds a CLI, per-subject process-level parallelism, resumable
checkpointing, OOM-resilient execution, and HPC-friendly logging.

Run (matches the accompanying Slurm script)::

    python -u src/run_all_metrics.py --workers ${SLURM_CPUS_PER_TASK}

or, with everything defaulted::

    python -u src/run_all_metrics.py --workers 10

Memory / OOM safety
-------------------
Each subject reads a full-night, 6-channel EDF; several such loads at once can
exceed a job's memory. To stay alive under memory pressure:

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


# job 1: mignot, 10 CPU / 30G, shard 0 of 3
sbatch --partition=mignot --cpus-per-task=10 --mem=30G --time=4-00:00:00 \
--export=ALL,NUM_SHARDS=3,SHARD_INDEX=0 \
run_all_metrics.sbatch

# job 2: normal, shard 1 of 3 (normal's ceiling is 2 days, below the script's 72h default)
sbatch --partition=normal --cpus-per-task=10 --mem=30G --time=2-00:00:00 \
--export=ALL,NUM_SHARDS=3,SHARD_INDEX=1 \
run_all_metrics.sbatch

# job 3: normal, shard 2 of 3
sbatch --partition=normal --cpus-per-task=10 --mem=30G --time=2-00:00:00 \
--export=ALL,NUM_SHARDS=3,SHARD_INDEX=2 \
run_all_metrics.sbatch
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
import traceback as _traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# The package lives beside this script (``src/infraslow``); make sure it is
# importable no matter the current working directory or multiprocessing start
# method.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from infraslow.io.hypnodensity import DEFAULT_HYPNODENSITY_SUFFIX
from infraslow.processing.subject_pipeline import (
    CHANNELS,
    METADATA_COLUMNS,
    MIN_BOUT_SEC,
    NPZ_STAGES,
    SF,
    STAGE_EVENT_FIELDS,
    calculate_features,
    combine_bioserenity_metadata,
    empty_stage_events,
    find_valid_bioserenity_subjects,
    load_bioserenity_metadata,
    metadata_row_columns,
)

logger = logging.getLogger("run_all_metrics")

# --------------------------------------------------------------------------- #
# Defaults / constants
# --------------------------------------------------------------------------- #
DEFAULT_METADATA = "$OAK/psg/Bioserenity/Excel/Morpheus_Data_All5.csv"
DEFAULT_METADATA2 = "$OAK/psg/Bioserenity/Excel/bioserenity_metadata3.csv"
DEFAULT_EDF_DIR = "$OAK/psg/Bioserenity/edf"
DEFAULT_HYPNO_DIR = "$OAK/psg/Bioserenity/Sleep_Staging"
DEFAULT_OUTPUT = "$SCRATCH/results_v2/metadata.csv"
DEFAULT_ERROR_OUTPUT = "$SCRATCH/results_v2/errors.csv"
DEFAULT_NPZ_DIR = "$SCRATCH/results_v2/npz"
EDF_SUFFIX = ".edf"
HYPNO_SUFFIX = DEFAULT_HYPNODENSITY_SUFFIX  # "_Hypnodensity.csv"

LOG_EVERY = 25              # progress log cadence (completed subjects)
DEFAULT_CHUNK_SIZE = 40     # subjects per recycled worker pool
# Concurrency cap so workers fit job memory. Six channels are now loaded and
# analysed per subject (vs. one previously), so this is sized well above a
# single-channel run; tune with --mem-per-worker-gb for your job's --mem.
DEFAULT_MEM_PER_WORKER_GB = 5.0

# errors.csv schema: subject-level failures leave state/channel/traceback blank;
# channel- or stage-level issues (a subject that otherwise succeeded) fill them in.
ERROR_COLUMNS = ["subject_id", "state", "channel", "error_type", "error_message", "traceback"]

# A single unit of work: (subject_id, metadata_row, edf_path, hypno_path, params, npz_dir).
Task = Tuple[str, Dict[str, Any], str, str, Dict[str, Any], str]
# A worker outcome: (status, subject_id, payload, issues) -- issues is always a
# list of channel/stage-level error records (see subject_pipeline._issue), even on
# an "ok" outcome (a subject can succeed overall with some channels/stages failed).
Outcome = Tuple[str, str, Dict[str, Any], List[Dict[str, str]]]


# --------------------------------------------------------------------------- #
# Worker (top-level so it is picklable by ProcessPoolExecutor)
# --------------------------------------------------------------------------- #
def process_subject(task: Task) -> Outcome:
    """Compute metadata+YASA row and per-channel events for one subject.

    Never raises inside the worker. Returns ``("ok", subject_id, row_dict,
    issues)`` on success, or ``("error", subject_id, error_record_dict, [])`` on
    a subject-level failure. (A hard crash such as an OOM kill cannot be caught
    here -- the driver handles that as a ``BrokenProcessPool`` casualty.)
    ``issues`` (always a list, possibly empty) carries channel/stage-level
    failures that did *not* sink the subject -- e.g. one missing EEG channel --
    logged separately to errors.csv alongside (not instead of) the metadata row.

    On success, this subject's per-channel/per-stage infraslow spectra, bouts
    and spindles are also written directly to their npz files (see
    :func:`write_subject_channel_events`) -- done here, inside the worker, so
    the arrays never need to be shipped back through the process pool just to
    be written by the driver.
    """
    subject_id, metadata_row, edf_path, hypno_path, params, npz_dir = task
    raw_issues: List[Dict[str, str]] = []
    try:
        row, events = calculate_features(
            metadata_row, Path(edf_path), Path(hypno_path), issues_out=raw_issues, **params
        )
        try:
            write_subject_channel_events(Path(npz_dir), subject_id, events)
        except Exception as exc:  # noqa: BLE001 - an npz-write failure must not lose the row
            logger.warning("Could not write npz for %s: %s", subject_id, exc)
            raw_issues.append({
                "channel": "", "stage": "", "error_type": type(exc).__name__,
                "error_message": str(exc), "traceback": _traceback.format_exc(),
            })
        issues = [_issue_record(subject_id, issue) for issue in raw_issues]
        return ("ok", subject_id, row, issues)
    except Exception as exc:  # noqa: BLE001 - isolate every per-subject failure
        return ("error", subject_id, _error_record(subject_id, exc), [])


def _issue_record(subject_id: str, issue: Dict[str, str]) -> Dict[str, str]:
    """Map a subject_pipeline channel/stage issue (key ``stage``) to the errors.csv row (key ``state``)."""
    return {
        "subject_id": subject_id,
        "state": issue.get("stage", ""),
        "channel": issue.get("channel", ""),
        "error_type": issue.get("error_type", ""),
        "error_message": issue.get("error_message", ""),
        "traceback": issue.get("traceback", ""),
    }


def _error_record(subject_id: str, exc: BaseException, *, prefix: str = "") -> Dict[str, str]:
    """A subject-level failure -- state/channel blank (see ERROR_COLUMNS)."""
    return {
        "subject_id": subject_id, "state": "", "channel": "",
        "error_type": type(exc).__name__, "error_message": f"{prefix}{exc}",
        "traceback": _traceback.format_exc(),
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
            [],
        )


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def build_results_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Assemble result dicts into the final, ordered DataFrame."""
    cols = metadata_row_columns()
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=cols)
    return df.reindex(columns=cols)


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
    combined = combined.reindex(columns=metadata_row_columns())
    _atomic_to_csv(combined, output_path)
    _atomic_to_csv(pd.DataFrame(error_records, columns=ERROR_COLUMNS), error_path)


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Per-channel events (.npz) helper -- spectra + bouts + spindles, per stage
# --------------------------------------------------------------------------- #
def npz_key(channel: str, field: str) -> str:
    """The flattened npz key for one channel/field, e.g. ``F3__spectra__freqs``."""
    return f"{channel}__{STAGE_EVENT_FIELDS[field]}"


def expected_npz_keys(channels: Sequence[str] = CHANNELS) -> List[str]:
    """Every key one stage's npz is expected to have (see :func:`npz_key`)."""
    return [npz_key(ch, field) for ch in channels for field in STAGE_EVENT_FIELDS]


def write_subject_channel_events(
    npz_dir: Path,
    subject_id: str,
    events: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    *,
    stages: Sequence[str] = NPZ_STAGES,
    channels: Sequence[str] = CHANNELS,
) -> None:
    """Write one subject's per-channel infraslow events to ``{npz_dir}/{stage}/{ID}.npz``.

    ``events`` is ``{channel: {stage: {...}}}`` (see :func:`~infraslow.processing.
    subject_pipeline.calculate_features_channel_events`; keys are :data:`STAGE_EVENT_FIELDS`).

    One ``{ID}.npz`` per subject *per stage* (not one consolidated file): with a
    100k+-subject cohort, rewriting a single growing ``.npz`` on every checkpoint
    would be O(n) work per checkpoint. Each subject's file is written exactly
    once and never touched again, so total write cost stays O(1) per subject
    regardless of cohort size (see :func:`process_subject`, which calls this).

    One npz is always written per ``(subject, stage)`` -- **every** channel's
    keys are always present (see :func:`expected_npz_keys`), correctly-typed but
    empty where a channel had no data, so every subject has the same predictable
    schema whether or not any bout qualified (a subject with no NREM sleep still
    gets an ``NREM/{ID}.npz`` with all-empty arrays, not a missing file).

    Each write goes to a temporary sibling first, is loaded back and validated
    (see :func:`validate_subject_npz`), and only then atomically replaces the
    target -- a crash mid-write, or a corrupt write, never leaves a bad file at
    the final path; a temp file that fails validation is removed, not renamed.
    """
    for stage in stages:
        arrays: Dict[str, np.ndarray] = {}
        for ch in channels:
            stage_events = events.get(ch, {}).get(stage)
            fields = stage_events if stage_events is not None else empty_stage_events()
            for field in STAGE_EVENT_FIELDS:
                arrays[npz_key(ch, field)] = fields[field]

        stage_dir = npz_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        path = stage_dir / f"{subject_id}.npz"
        tmp = path.with_suffix(path.suffix + ".tmp")
        # ``np.savez_compressed`` silently appends ".npz" to a *filename* that
        # doesn't already end in ".npz" -- "{tmp}.npz" would be created instead
        # of "{tmp}", and the rename below would then fail to find it. Passing
        # an open file object instead of a path bypasses that auto-append.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        problems = validate_subject_npz(tmp, channels=channels)
        if problems:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"Wrote an invalid npz for {subject_id}/{stage}: {'; '.join(problems)}")
        os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# NPZ validation (schema, dtypes, and the pipeline's own invariants)
# --------------------------------------------------------------------------- #
def validate_subject_npz(
    path: Path, *, channels: Sequence[str] = CHANNELS, min_bout_sec: float = MIN_BOUT_SEC,
) -> List[str]:
    """Validate one stage's ``{ID}.npz`` against the v2 schema and pipeline invariants.

    Returns a list of human-readable problem descriptions (empty = valid). Never
    raises for a structurally-broken file -- a load failure itself becomes one
    problem string -- so callers can use this both to gate a fresh write and to
    decide whether a prior-run file is safe to skip on resume (see ``main``'s
    ``--overwrite``-free resume path).

    Checks (see the pipeline's ``md/update-metrics.md`` Sec. 19):

    * the file loads with ``np.load(path, allow_pickle=False)`` and every array
      is present under the expected ``{channel}__{category}__{field}`` keys;
    * no array has an ``object`` dtype;
    * per channel: ``len(freqs) == len(corr_mean)``, both finite when non-empty;
    * per channel: ``len(spindle_start) == len(spindle_stop) == len(spindle_peak)``,
      each spindle satisfies ``start <= peak <= stop``, all finite;
    * per channel: ``len(bout_start) == len(bout_stop) == len(bout_n_spindles)``,
      each bout satisfies ``stop > start``, ``stop - start >= min_bout_sec``, and
      ``n_spindles >= 1``;
    * per channel: ``sum(bout_n_spindles) == len(spindle_peak)`` -- every saved
      spindle is accounted for by exactly the saved bouts, and none twice.
    """
    problems: List[str] = []
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any load failure is itself the one problem to report
        return [f"could not load {path}: {exc}"]

    with data:
        keys = set(data.files)
        missing = set(expected_npz_keys(channels)) - keys
        if missing:
            problems.append(f"missing key(s): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

        for ch in channels:
            arr = {}
            for field in STAGE_EVENT_FIELDS:
                key = npz_key(ch, field)
                if key not in keys:
                    continue
                a = data[key]
                if a.dtype == object:
                    problems.append(f"{key}: object dtype not allowed")
                arr[field] = a
            if len(arr) < len(STAGE_EVENT_FIELDS):
                continue  # already reported as missing above

            if len(arr["freqs"]) != len(arr["corr_mean"]):
                problems.append(f"{ch}: freqs/corr_mean length mismatch")
            elif arr["freqs"].size and not (np.all(np.isfinite(arr["freqs"])) and np.all(np.isfinite(arr["corr_mean"]))):
                problems.append(f"{ch}: non-finite spectrum values")

            n_sp = {len(arr["spindle_start"]), len(arr["spindle_stop"]), len(arr["spindle_peak"])}
            if len(n_sp) != 1:
                problems.append(f"{ch}: spindle start/stop/peak length mismatch")
            elif arr["spindle_start"].size:
                ok = (
                    np.all(np.isfinite(arr["spindle_start"]))
                    and np.all(np.isfinite(arr["spindle_stop"]))
                    and np.all(np.isfinite(arr["spindle_peak"]))
                    and np.all(arr["spindle_start"] <= arr["spindle_peak"])
                    and np.all(arr["spindle_peak"] <= arr["spindle_stop"])
                )
                if not ok:
                    problems.append(f"{ch}: spindle start<=peak<=stop violated")

            n_b = {len(arr["bout_start"]), len(arr["bout_stop"]), len(arr["bout_n_spindles"])}
            if len(n_b) != 1:
                problems.append(f"{ch}: bout start/stop/n_spindles length mismatch")
            elif arr["bout_start"].size:
                duration = arr["bout_stop"] - arr["bout_start"]
                if not np.all(duration > 0):
                    problems.append(f"{ch}: a bout has stop <= start")
                elif not np.all(duration >= min_bout_sec - 1e-6):
                    problems.append(f"{ch}: a bout is shorter than {min_bout_sec}s")
                if not np.all(arr["bout_n_spindles"] >= 1):
                    problems.append(f"{ch}: a retained bout has n_spindles < 1")
                if int(arr["bout_n_spindles"].sum()) != len(arr["spindle_peak"]):
                    problems.append(f"{ch}: sum(bout_n_spindles) != len(spindle_peak)")

    return problems


def load_channel_events(npz_path: Path, channel: str) -> Dict[str, np.ndarray]:
    """Load one channel's arrays from a stage npz written by :func:`write_subject_channel_events`.

    Example::

        from pathlib import Path
        path = Path("$SCRATCH/results_v2/npz/N2/12345.npz")
        events = load_channel_events(path, "F3")
        print(events["freqs"].shape, events["corr_mean"].shape)
        print(events["bout_start"].shape, events["spindle_peak"].shape)

    Returns:
        Dict keyed by :data:`~infraslow.processing.subject_pipeline.STAGE_EVENT_FIELDS`
        (``freqs``, ``corr_mean``, ``spindle_start/stop/peak``, ``bout_start/stop/n_spindles``).
    """
    with np.load(npz_path, allow_pickle=False) as data:
        return {field: data[npz_key(channel, field)] for field in STAGE_EVENT_FIELDS}


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
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output metadata+YASA-stats CSV.")
    parser.add_argument(
        "--error-output", default=DEFAULT_ERROR_OUTPUT, help="Failed-subjects CSV.",
    )
    parser.add_argument(
        "--npz-dir", default=DEFAULT_NPZ_DIR,
        help="Root directory of {npz-dir}/{stage}/{ID}.npz -- one file per subject per "
             f"stage ({', '.join(NPZ_STAGES)}), holding every channel's ({', '.join(CHANNELS)}) "
             "infraslow spectra, bouts, and spindles for that stage.",
    )
    parser.add_argument(
        "--subject-id", default=None,
        help="Process only this one subject ID (a one-subject dry run); overrides --limit.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many valid subjects (for quick tests).",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Split valid subjects into this many disjoint shards, one per parallel job "
             "(see --shard-index). Defaults to $SLURM_ARRAY_TASK_COUNT, else 1.",
    )
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="This job's shard in [0, --num-shards) -- processes every "
             "--num-shards-th subject (sorted by ID), starting here. Defaults to "
             "$SLURM_ARRAY_TASK_ID, else 0.",
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
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve subjects and print the processing plan; compute/write nothing.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate every existing subject's npz files (see validate_subject_npz) "
             "against --output's ID list and report problems; compute/write nothing.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logging level.",
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


def resolve_shard(cli_num_shards: Optional[int], cli_shard_index: Optional[int]) -> Tuple[int, int]:
    """(num_shards, shard_index), defaulting to the Slurm array env vars when unset.

    Lets a job array split the subject list across tasks without any explicit
    CLI flags: each array task's ``$SLURM_ARRAY_TASK_ID``/``_COUNT`` picks its
    slice automatically, mirroring how :func:`resolve_workers` defaults from
    ``$SLURM_CPUS_PER_TASK``.
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


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("infraslow").setLevel(level)


def _expand(path_str: str) -> Path:
    """Expand ``$VARS`` and ``~`` in a path string."""
    return Path(os.path.expandvars(os.path.expanduser(path_str)))


def validated_done_ids(prior_df: pd.DataFrame, npz_dir: Path, stages: Sequence[str] = NPZ_STAGES) -> set:
    """Subject IDs from a prior ``metadata.csv`` whose npz outputs are all present and valid.

    Implements the resume/rerun rule (md/update-metrics.md Sec. 16): a subject is
    only skipped on resume if *every* stage's npz file exists and passes
    :func:`validate_subject_npz`; a missing, incomplete, or corrupted file means
    the subject is rebuilt, not silently skipped. Logs one summary line per
    outcome bucket rather than one line per subject, to keep this cheap-ish over
    a 100k+-subject resume and avoid flooding the log.
    """
    ids = set(prior_df["ID"].astype(str))
    valid_ids: set = set()
    rebuild = 0
    for subject_id in ids:
        ok = True
        for stage in stages:
            path = npz_dir / stage / f"{subject_id}.npz"
            if not path.is_file() or validate_subject_npz(path):
                ok = False
                break
        if ok:
            valid_ids.add(subject_id)
        else:
            rebuild += 1
    logger.info(
        "resume validation: %d/%d subject(s) have complete, valid npz outputs "
        "(will be skipped); %d will be rebuilt",
        len(valid_ids), len(ids), rebuild,
    )
    return valid_ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    workers = resolve_workers(args.workers)
    workers = cap_workers_for_memory(workers, args.mem_per_worker_gb)
    chunk_size = max(1, args.chunk_size)

    metadata_path = _expand(args.metadata)
    metadata_path2 = _expand(args.metadata2)
    edf_dir = _expand(args.edf_dir)
    hypno_dir = _expand(args.hypno_dir)
    output_path = _expand(args.output)
    error_path = _expand(args.error_output)
    npz_dir = _expand(args.npz_dir)

    # Requirement: create results_v2/ (+ per-stage npz/ subdirs) and logs/ up front.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    for stage in NPZ_STAGES:
        (npz_dir / stage).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger.info("workers        : %d", workers)
    logger.info("chunk size     : %d", chunk_size)
    logger.info("metadata       : %s", metadata_path)
    logger.info("metadata2      : %s", metadata_path2)
    logger.info("edf dir        : %s", edf_dir)
    logger.info("hypno dir      : %s", hypno_dir)
    logger.info("output         : %s", output_path)
    logger.info("error output   : %s", error_path)
    logger.info("npz dir        : %s", npz_dir)
    logger.info("channels       : %s", ", ".join(CHANNELS))
    logger.info("npz stages     : %s", ", ".join(NPZ_STAGES))

    # --- 0. --validate-only: check existing outputs and exit, no processing --
    if args.validate_only:
        if not output_path.exists():
            logger.error("--validate-only: %s does not exist", output_path)
            return 1
        prior_df = pd.read_csv(output_path, dtype={"ID": str})
        valid_ids = validated_done_ids(prior_df, npz_dir)
        print("=" * 60)
        print(f"Validated {len(prior_df)} subject(s) listed in {output_path}")
        print(f"  valid npz outputs   : {len(valid_ids)}")
        print(f"  invalid/missing     : {len(prior_df) - len(valid_ids)}")
        print("=" * 60)
        return 0

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

    if args.subject_id is not None:
        valid = valid[valid["ID"].astype(str).str.strip() == args.subject_id]
        logger.info("--subject-id set: restricted to %d subject(s)", len(valid))
    elif args.limit is not None:
        valid = valid.head(args.limit)
        logger.info("limited to first %d valid subject(s)", len(valid))

    # Shard across parallel jobs (e.g. a Slurm array): each shard gets a disjoint,
    # deterministic slice of the ID-sorted subject list. --output/--error-output
    # must be per-shard paths (e.g. metadata_shard{N}.csv) -- checkpointing does a
    # full-file overwrite, so two shards sharing one output path would clobber
    # each other; merge the per-shard CSVs into one after the array completes.
    num_shards, shard_index = resolve_shard(args.num_shards, args.shard_index)
    if num_shards > 1:
        valid = valid.sort_values("ID", kind="stable").reset_index(drop=True)
        valid = valid.iloc[shard_index::num_shards].reset_index(drop=True)
        logger.info(
            "shard %d/%d: %d subject(s) assigned to this job",
            shard_index, num_shards, len(valid),
        )

    # --- 2. Resume: skip subjects already in an existing output ----------- #
    # A subject is only skipped if its metadata row *and* every stage's npz
    # file are present and pass validate_subject_npz -- a missing, incomplete,
    # or corrupted file means the subject is rebuilt, not silently skipped
    # (md/update-metrics.md Sec. 16).
    prior_df: Optional[pd.DataFrame] = None
    done_ids: set = set()
    if output_path.exists() and not args.overwrite:
        try:
            prior_df = pd.read_csv(output_path, dtype={"ID": str})
            done_ids = validated_done_ids(prior_df, npz_dir)
            # A subject being rebuilt (invalid/missing npz) must not keep its stale
            # row too, or it would be duplicated once its fresh row is appended.
            prior_df = prior_df[prior_df["ID"].astype(str).isin(done_ids)].reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001 - a corrupt prior file must not abort
            logger.warning("could not read existing output (%s); starting fresh", exc)
            prior_df = None
            done_ids = set()
    elif output_path.exists() and args.overwrite:
        logger.info("--overwrite set: existing %s will be replaced", output_path)

    # --- 3. Build the task list ------------------------------------------- #
    params = {"sf": SF, "channels": CHANNELS, "stages": NPZ_STAGES, "require_spindle": True}
    tasks: List[Task] = []
    for _, row in valid.iterrows():
        subject_id = str(row["ID"]).strip()
        if subject_id in done_ids:
            continue
        metadata_row = {col: row.get(col) for col in METADATA_COLUMNS}
        metadata_row["ID"] = subject_id
        edf_path = str(edf_dir / f"{subject_id}{EDF_SUFFIX}")
        hypno_path = str(hypno_dir / f"{subject_id}{HYPNO_SUFFIX}")
        tasks.append((subject_id, metadata_row, edf_path, hypno_path, params, str(npz_dir)))

    n_to_process = len(tasks)
    logger.info("subjects to process this run: %d", n_to_process)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN -- nothing computed or written")
        print(f"Subjects with required files:  {n_valid}")
        print(f"Already valid/skipped:         {len(done_ids)}")
        print(f"Would process this run:        {n_to_process}")
        if tasks:
            preview = ", ".join(t[0] for t in tasks[:10])
            print(f"First subject(s):               {preview}{' ...' if n_to_process > 10 else ''}")
        print("=" * 60)
        return 0

    # --- 4. Process (chunked + recycled + OOM-resilient) ------------------ #
    result_records: List[Dict[str, Any]] = []
    error_records: List[Dict[str, str]] = []
    processed = 0

    def handle(outcome: Outcome) -> None:
        nonlocal processed
        processed += 1
        status, subject_id, payload, issues = outcome
        error_records.extend(issues)
        if status == "ok":
            result_records.append(payload)
            if issues:
                logger.warning("%s: %d channel/stage issue(s) (subject still recorded)",
                                subject_id, len(issues))
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
    subject_failures = sum(1 for r in error_records if not r["state"] and not r["channel"])
    channel_stage_issues = len(error_records) - subject_failures
    print("=" * 60)
    print(f"Total subjects in metadata:    {n_metadata}")
    print(f"Subjects with required files:  {n_valid}")
    print(f"Successful (metadata rows):    {total_success}")
    print(f"Subject-level failures:        {subject_failures}")
    print(f"Channel/stage-level issues:    {channel_stage_issues}")
    print(f"Output CSV:                    {output_path}")
    print(f"Error CSV:                     {error_path}")
    print(f"NPZ dir ({{stage}}/{{ID}}.npz):    {npz_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
