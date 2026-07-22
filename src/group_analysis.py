"""N2-C3 spindle-rate group analysis (md/group_analysis.md).

Restricted, throughout, to::

    sleep_stage = "N2"
    channel = "C3"

No other stage/channel is ever combined in, and no metadata (age/gender/BMI,
sleep architecture) is loaded here.

This script builds *subject-level* N2-C3 results on top of
``demo_infraslow_yasa_compare.py``, which only ever fits a bi-Gaussian ISFS
curve to the *cross-subject average* spectrum per stage/channel -- it has no
notion of a per-subject summary row. Per subject/channel/stage, the pipeline
(``infraslow.processing.subject_pipeline``) writes only
``{channel}__spectra__freqs`` / ``{channel}__spectra__corr_mean`` (an
already bout-averaged, baseline-corrected, unit-band-area relative spectrum --
there is no separately-stored raw/baseline spectrum) plus that channel's bout
and spindle arrays, to ``{results}/{stage}/{subject_id}.npz``. So this script
reuses ``demo_infraslow_yasa_compare``'s own ``fit_isfs`` / ``bigaussian`` /
``chromatogram_peak_area`` functions, applied to *each subject's own*
``corr_mean`` curve (instead of the cross-subject average) to derive
``peak_freq_hz, peak_period_s, bandwidth_hz, auc, chromatogram_peak_area`` and
``power`` (the fitted bi-Gaussian peak amplitude -- the same "power" the demo
script itself uses to pick its "max power channel"). ``spindle_per_min`` reuses
the demo's own ``_spindle_rate_per_min`` verbatim; ``spindle_per_min_SEM`` (not
computed anywhere upstream) is a per-subject extension: the SEM, across that
subject's own bouts, of each bout's own spindles/min rate.

Per-subject loading (Step 2) is both I/O-bound (one npz open per subject) and
CPU-bound (a ``scipy.optimize.curve_fit`` bi-Gaussian fit per subject) -- unlike
``demo_infraslow_yasa_compare.py``'s cohort loading, which is I/O-only and uses
a thread pool. So this script loads across a **process** pool
(:func:`load_subject_records_from_npz`) instead: threads would serialize the
curve-fit work behind the GIL, while separate processes get true multi-core
speedup for it (and still overlap each other's I/O waits). Size the pool with
``--workers`` (defaults to ``$SLURM_CPUS_PER_TASK``) and request that many CPUs
via Slurm's ``--cpus-per-task`` for it to matter.

Run via Slurm, not the login node, e.g.::

    srun -p normal --time=00:30:00 --mem=32G --cpus-per-task=16 \\
        python3 group_analysis.py --results $SCRATCH/data/npz \\
        --output-dir infraslow/results/group_analysis/N2_C3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Keep every worker process single-threaded for BLAS/OpenMP so the
# ProcessPoolExecutor below (one process per core) doesn't oversubscribe
# cores -- must be set before numpy/scipy (and demo_infraslow_yasa_compare,
# which imports both) are imported. Matches run_all_metrics.py.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402 - see env-var shim above
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ``demo_infraslow_yasa_compare`` lives beside this script; make it importable
# regardless of the current working directory (matches run_all_metrics.py).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import demo_infraslow_yasa_compare as demo  # noqa: E402 - see sys.path shim above
from infraslow.stats.group_assignment import (  # noqa: E402
    HIGH_LABEL,
    LOW_LABEL,
    assign_spindle_rate_groups,
)
from infraslow.stats.group_comparison import compare_parameters  # noqa: E402
from infraslow.viz.group_analysis import (  # noqa: E402
    plot_group_infraslow_compare,
    plot_group_spectrum_clean,
    plot_parameter_comparisons,
    plot_spindle_rate_distribution,
)

logger = logging.getLogger(__name__)

#: Required per-subject summary columns (md/group_analysis.md Step 2/3).
REQUIRED_SUMMARY_PARAMS: List[str] = [
    "peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc",
    "chromatogram_peak_area", "spindle_per_min", "spindle_per_min_SEM",
]
#: Parameters compared between spindle-rate groups (Step 5). `spindle_per_min`
#: is deliberately excluded -- the groups are *constructed* from it, so testing
#: it again would not be independent evidence (md/group_analysis.md Step 4/5).
COMPARISON_PARAMETERS: List[str] = [
    "power", "peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc", "chromatogram_peak_area",
]
#: Documented tolerance for the peak_period_s ≈ 1/peak_freq_hz consistency check (Step 3).
PERIOD_FREQ_TOLERANCE = 1e-6


# --------------------------------------------------------------------------- #
# Step 2: load subject-level N2-C3 results
# --------------------------------------------------------------------------- #
def _subject_channel_spindle_rate(npz, channel: str) -> Tuple[float, float, int]:
    """``(spindle_per_min, spindle_per_min_SEM, n_bouts)`` for one subject/channel.

    ``spindle_per_min`` reuses
    ``demo_infraslow_yasa_compare._spindle_rate_per_min`` verbatim (pools this
    subject's own bouts into one rate: total spindles / total bout duration).
    ``spindle_per_min_SEM`` extends it -- the existing pipeline has no
    per-subject SEM -- as the SEM, across this subject's own bouts, of each
    bout's own spindles/min rate: ``0.0`` with exactly one bout, ``NaN`` with none.
    """
    n_key = f"{channel}__bouts__n_spindles"
    start_key = f"{channel}__bouts__start"
    stop_key = f"{channel}__bouts__stop"
    if n_key not in npz.files or start_key not in npz.files or stop_key not in npz.files:
        return np.nan, np.nan, 0
    n_spindles, start, stop = npz[n_key], npz[start_key], npz[stop_key]
    if n_spindles.size == 0:
        return np.nan, np.nan, 0
    duration_min = (stop - start) / 60.0
    valid = duration_min > 0
    if not valid.any():
        return np.nan, np.nan, 0
    per_bout_rate = n_spindles[valid] / duration_min[valid]
    overall_rate = demo._spindle_rate_per_min(npz, channel)
    sem = float(per_bout_rate.std(ddof=1) / np.sqrt(per_bout_rate.size)) if per_bout_rate.size > 1 else 0.0
    return overall_rate, sem, int(per_bout_rate.size)


def _empty_subject_record(subject_id: str, stage: str, channel: str) -> Dict[str, object]:
    return dict(
        subject_id=subject_id, sleep_stage=stage, channel=channel,
        peak_freq_hz=np.nan, peak_period_s=np.nan, bandwidth_hz=np.nan, power=np.nan,
        auc=np.nan, chromatogram_peak_area=np.nan, spindle_per_min=np.nan, spindle_per_min_SEM=np.nan,
        freqs=None, corr_mean=None, fitted_curve=None, fitted_freqs=None, exclusion_reason="",
    )


def _load_subject_record(npz_path: Path, stage: str, channel: str) -> Dict[str, object]:
    """One raw (pre-validation) N2-C3 record for one subject, from its per-stage npz.

    Never raises: any problem is captured in ``exclusion_reason`` for
    :func:`validate_records` to report -- a missing key or an empty/failed
    spectrum is an expected outcome for some subjects at this cohort's scale,
    not a hard error (md/group_analysis.md Step 2/3).
    """
    subject_id = npz_path.stem
    record = _empty_subject_record(subject_id, stage, channel)
    try:
        with np.load(npz_path) as npz:
            freq_key, corr_key = f"{channel}__spectra__freqs", f"{channel}__spectra__corr_mean"
            if freq_key not in npz.files or corr_key not in npz.files:
                record["exclusion_reason"] = f"missing {channel} spectrum arrays"
                return record
            freqs, corr_mean = npz[freq_key], npz[corr_key]
            if freqs.size == 0 or corr_mean.size == 0:
                record["exclusion_reason"] = f"empty {channel} spectrum arrays"
                return record
            if freqs.size != corr_mean.size:
                record["exclusion_reason"] = "unequal frequency and spectrum-array lengths"
                return record
            if not (np.all(np.isfinite(freqs)) and np.all(np.isfinite(corr_mean))):
                record["exclusion_reason"] = "non-finite frequency or power values"
                return record
            if np.any(np.diff(freqs) <= 0):
                record["exclusion_reason"] = "non-increasing frequency array"
                return record

            record["freqs"], record["corr_mean"] = freqs, corr_mean

            rate, rate_sem, _n_bouts = _subject_channel_spindle_rate(npz, channel)
            record["spindle_per_min"], record["spindle_per_min_SEM"] = rate, rate_sem

            try:
                fit = demo.fit_isfs(freqs, corr_mean)
            except RuntimeError as exc:
                record["exclusion_reason"] = f"bi-Gaussian fit failed: {exc}"
                return record

            fitted_freqs = np.linspace(*demo.INFRASLOW_BAND, 200)
            fitted_curve = demo.bigaussian(fitted_freqs, *fit["popt"])
            peak = demo.chromatogram_peak_area(fitted_freqs, fitted_curve, threshold=fit["threshold"])

            record["peak_freq_hz"] = fit["mu"]
            record["peak_period_s"] = 1.0 / fit["mu"] if fit["mu"] else np.nan
            record["bandwidth_hz"] = fit["bandwidth"]
            record["power"] = fit["amp"]
            record["auc"] = fit["auc"]
            record["chromatogram_peak_area"] = peak["area"]
            record["fitted_curve"], record["fitted_freqs"] = fitted_curve, fitted_freqs
    except Exception as exc:  # noqa: BLE001 - one bad npz must not sink the whole load
        record["exclusion_reason"] = f"could not load npz: {exc}"
    return record


def resolve_workers(cli_workers: Optional[int]) -> int:
    """Worker-process count: ``cli_workers``, else ``$SLURM_CPUS_PER_TASK``, else all visible CPUs."""
    if cli_workers and cli_workers > 0:
        return cli_workers
    env = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return len(os.sched_getaffinity(0))


def load_subject_records_from_npz(
    results_dir: Path, stage: str, channel: str, *,
    n_subjects: Optional[int] = None, workers: Optional[int] = None,
) -> pd.DataFrame:
    """Every subject's raw N2-C3 record from ``{results_dir}/{stage}/*.npz`` (Step 2).

    Same directory layout as ``demo_infraslow_yasa_compare.py``'s
    ``--results-dir``, but loaded across a **process** pool rather than a
    thread pool: each subject's record also runs a ``scipy.optimize.curve_fit``
    bi-Gaussian fit (CPU-bound), which would serialize behind the GIL under
    threads. Separate processes get real multi-core speedup for that fit while
    still overlapping each other's npz-read I/O waits.
    """
    stage_dir = results_dir / stage
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"No {stage} results directory found at {stage_dir}")
    paths = sorted(stage_dir.glob("*.npz"))
    if n_subjects:
        paths = paths[:n_subjects]
    if not paths:
        raise FileNotFoundError(f"No subject npz files found under {stage_dir}")

    n_workers = resolve_workers(workers)
    # Small per-task chunks amortize IPC overhead without letting one worker's
    # queue run dry while another still has a long tail of files.
    chunksize = max(1, min(200, len(paths) // (n_workers * 4) or 1))
    logger.info(
        "loading %d subject npz file(s) from %s (channel=%s) across %d worker process(es), chunksize=%d",
        len(paths), stage_dir, channel, n_workers, chunksize,
    )

    worker = partial(_load_subject_record, stage=stage, channel=channel)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        records = list(pool.map(worker, paths, chunksize=chunksize))
    logger.info("loaded %d subject record(s)", len(records))
    return pd.DataFrame(records)


def _load_spectrum_arrays(
    npz_path: Path, channel: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``(freqs, corr_mean)`` for one subject/channel, or ``(None, None)`` if unavailable."""
    if not npz_path.is_file():
        return None, None
    with np.load(npz_path) as npz:
        freq_key, corr_key = f"{channel}__spectra__freqs", f"{channel}__spectra__corr_mean"
        if freq_key in npz.files and corr_key in npz.files and npz[freq_key].size:
            return npz[freq_key], npz[corr_key]
    return None, None


def load_subject_records_from_files(
    summary_path: Path, spectrum_dir: Optional[Path], stage: str, channel: str,
) -> pd.DataFrame:
    """Alternate load path: a precomputed subject-level summary CSV, optionally
    paired with a separate npz root for the Step 6 spectrum arrays.

    Args:
        summary_path: CSV with at least ``subject_id, sleep_stage, channel``
            plus :data:`REQUIRED_SUMMARY_PARAMS` (``power`` optional; filled
            with NaN if absent).
        spectrum_dir: A ``{stage}/{subject_id}.npz`` root providing
            ``{channel}__spectra__freqs`` / ``{channel}__spectra__corr_mean``.
            If omitted, every subject fails Step 3's spectrum-array checks and
            is excluded -- Step 6's plots need real spectra, so pass
            ``--spectrum-results`` alongside ``--summary-results`` whenever the
            spectrum plots are wanted (see the script's `Limitations` note).

    Raises:
        ValueError: if a required column is missing from ``summary_path``.
    """
    df = pd.read_csv(summary_path, dtype={"subject_id": str})
    missing_cols = [c for c in ["subject_id", "sleep_stage", "channel", *REQUIRED_SUMMARY_PARAMS] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{summary_path} is missing required column(s): {missing_cols}")
    if "power" not in df.columns:
        df["power"] = np.nan
    df["exclusion_reason"] = ""
    df["freqs"] = None
    df["corr_mean"] = None
    df["fitted_curve"] = None
    df["fitted_freqs"] = None

    if spectrum_dir is not None:
        stage_dir = spectrum_dir / stage
        paths = [stage_dir / f"{subject_id}.npz" for subject_id in df["subject_id"]]
        # Pure I/O here (no per-subject curve fit, unlike the --results/npz path),
        # so a thread pool -- whose blocking npz reads release the GIL -- is enough.
        with ThreadPoolExecutor(max_workers=demo.N_IO_WORKERS) as pool:
            spectra = list(pool.map(lambda p: _load_spectrum_arrays(p, channel), paths))
        for idx, (freqs, corr_mean) in zip(df.index, spectra):
            if freqs is not None:
                df.at[idx, "freqs"] = freqs
                df.at[idx, "corr_mean"] = corr_mean
    return df


# --------------------------------------------------------------------------- #
# Step 3: validate the N2-C3 dataset
# --------------------------------------------------------------------------- #
def _spectrum_problem(freqs, corr_mean) -> str:
    if freqs is None or corr_mean is None:
        return "missing spectrum arrays"
    freqs, corr_mean = np.asarray(freqs), np.asarray(corr_mean)
    if freqs.size == 0:
        return "empty frequency array"
    if corr_mean.size == 0:
        return "empty power array"
    if freqs.size != corr_mean.size:
        return "unequal frequency and spectrum-array lengths"
    if not (np.all(np.isfinite(freqs)) and np.all(np.isfinite(corr_mean))):
        return "non-finite frequency or power values"
    if np.any(np.diff(freqs) <= 0):
        return "non-increasing frequency array"
    return ""


def validate_records(
    df: pd.DataFrame, sleep_stage: str, channel: str, *, tolerance: float = PERIOD_FREQ_TOLERANCE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Split raw per-subject records into (validated, excluded) + a validation report (Step 3).

    Every check in md/group_analysis.md Step 3 is applied in sequence; a
    subject keeps only its *first* applicable exclusion reason (checks run
    duplicate/stage/channel -> value checks -> missingness -> infinities ->
    peak_period_s consistency -> spectrum-array checks -> fitted-curve
    validity), and no invalid value is ever silently recomputed or dropped in
    place -- the whole subject is excluded and reported instead.
    """
    df = df.copy()
    n_total = len(df)
    df["exclusion_reason"] = df["exclusion_reason"].fillna("").astype(str)

    def _set_reason(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False)
        df.loc[mask & (df["exclusion_reason"] == ""), "exclusion_reason"] = reason

    dup_mask = df["subject_id"].duplicated(keep=False)
    n_duplicates = int(dup_mask.sum())
    _set_reason(dup_mask, "duplicate subject_id record")

    _set_reason(df["sleep_stage"].astype(str) != sleep_stage, "incorrect sleep state")
    _set_reason(df["channel"].astype(str) != channel, "incorrect channel")

    _set_reason(df["spindle_per_min"] < 0, "negative spindle_per_min")
    _set_reason(df["spindle_per_min_SEM"] < 0, "negative spindle_per_min_SEM")
    _set_reason(df["peak_freq_hz"] <= 0, "nonpositive peak_freq_hz")
    _set_reason(df["peak_period_s"] <= 0, "nonpositive peak_period_s")
    _set_reason(df["bandwidth_hz"] < 0, "negative bandwidth_hz")
    _set_reason(df["auc"] < 0, "negative auc")
    _set_reason(df["chromatogram_peak_area"] < 0, "negative chromatogram_peak_area")

    missingness = {}
    for col in REQUIRED_SUMMARY_PARAMS:
        missing = df[col].isna()
        missingness[col] = int(missing.sum())
        _set_reason(missing, f"missing {col}")

    inf_mask = pd.Series(
        np.isinf(df[REQUIRED_SUMMARY_PARAMS].to_numpy(dtype=float)).any(axis=1), index=df.index,
    )
    _set_reason(inf_mask, "infinite value in a required parameter")

    consistent_domain = (df["peak_freq_hz"] > 0) & df["peak_freq_hz"].notna() & df["peak_period_s"].notna()
    inconsistent = consistent_domain & ((df["peak_period_s"] - 1.0 / df["peak_freq_hz"]).abs() > tolerance)
    _set_reason(inconsistent, f"peak_period_s inconsistent with 1/peak_freq_hz (tolerance={tolerance})")

    spectrum_problems = df.apply(lambda row: _spectrum_problem(row["freqs"], row["corr_mean"]), axis=1)
    has_valid_spectrum = spectrum_problems == ""
    _set_reason(~has_valid_spectrum, spectrum_problems)  # aligned Series -> per-row reason text

    fitted_curve_invalid = df["fitted_curve"].apply(
        lambda curve: curve is not None and not np.all(np.isfinite(np.asarray(curve)))
    )
    _set_reason(fitted_curve_invalid, "invalid fitted curve")

    is_valid = df["exclusion_reason"] == ""
    validated = df.loc[is_valid].reset_index(drop=True)
    excluded = df.loc[~is_valid].reset_index(drop=True)

    report = {
        "total_subjects_loaded": n_total,
        "n_with_stage": int((df["sleep_stage"].astype(str) == sleep_stage).sum()),
        "n_with_channel": int((df["channel"].astype(str) == channel).sum()),
        "n_valid_stage_channel": int(is_valid.sum()),
        "n_excluded": int((~is_valid).sum()),
        "n_duplicates": n_duplicates,
        "missingness": missingness,
        "n_valid_spectrum": int(has_valid_spectrum.sum()),
        "exclusion_reason_counts": excluded["exclusion_reason"].value_counts().to_dict(),
    }
    return validated, excluded, report


def format_validation_report(report: Dict[str, object], sleep_stage: str, channel: str) -> str:
    lines = [
        f"{sleep_stage}-{channel} Validation Report",
        "========================",
        f"Total subjects loaded          : {report['total_subjects_loaded']}",
        f"Subjects with {sleep_stage} data          : {report['n_with_stage']}",
        f"Subjects with {channel} data          : {report['n_with_channel']}",
        f"Subjects with valid {sleep_stage}-{channel} data : {report['n_valid_stage_channel']}",
        f"Subjects excluded              : {report['n_excluded']}",
        f"Duplicate subject records      : {report['n_duplicates']}",
        f"Subjects with valid spectrum   : {report['n_valid_spectrum']}",
        "",
        "Missingness per required parameter:",
    ]
    for col, n in report["missingness"].items():
        lines.append(f"  {col:<24}: {n}")
    lines.append("")
    lines.append("Exclusion reasons:")
    for reason, n in report["exclusion_reason_counts"].items():
        lines.append(f"  {reason:<50}: {n}")
    return "\n".join(lines)


def format_grouping_report(
    fit_result, assignments: pd.DataFrame, threshold: float, sleep_stage: str, channel: str,
) -> str:
    n_low = int((assignments["spindle_group"] == LOW_LABEL).sum())
    n_high = int((assignments["spindle_group"] == HIGH_LABEL).sum())
    n_uncertain = int(assignments["uncertain_assignment"].sum())
    centers = np.sort(fit_result.centers_original_scale)
    return "\n".join([
        f"{sleep_stage}-{channel} Spindle-Rate Grouping Report",
        "===================================",
        f"GMM input scale selected : {fit_result.scale}",
        f"Raw-scale BIC            : {fit_result.bic_raw:.2f}",
        f"log1p-scale BIC          : {fit_result.bic_log1p:.2f}",
        f"Component centers (spindles/min, low->high): {centers[0]:.3f}, {centers[1]:.3f}",
        f"Uncertainty threshold    : group_probability < {threshold}",
        f"low_spindle_rate n       : {n_low}",
        f"high_spindle_rate n      : {n_high}",
        f"uncertain assignments    : {n_uncertain}",
    ])


# --------------------------------------------------------------------------- #
# Step 6: group-level spectrum curve (mean/SEM across subjects + a fresh
# group-level bi-Gaussian fit, matching demo_infraslow_yasa_compare's own
# fit-on-the-average-curve approach -- distinct from Step 5's per-subject fits).
# --------------------------------------------------------------------------- #
def _group_mean_curve(freqs_list, corr_list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(freqs, mean, sem, n)`` across one group's subjects.

    One row per subject is guaranteed by validation (Step 3), so no
    within-subject duplicate-averaging is needed here. A subject whose
    frequency grid differs from the group's first valid grid is skipped
    (logged), never silently averaged onto a foreign grid.
    """
    ref_freqs = None
    stack = []
    for freqs, corr in zip(freqs_list, corr_list):
        if freqs is None or corr is None:
            continue
        freqs, corr = np.asarray(freqs), np.asarray(corr)
        if ref_freqs is None:
            ref_freqs = freqs
        elif not (freqs.shape == ref_freqs.shape and np.allclose(freqs, ref_freqs)):
            logger.warning("skipping one subject's spectrum for the group curve: frequency grid mismatch")
            continue
        stack.append(corr)
    if not stack:
        return np.empty(0), np.empty(0), np.empty(0), 0
    arr = np.vstack(stack)
    n = arr.shape[0]
    mean = arr.mean(0)
    sem = arr.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return ref_freqs, mean, sem, n


def _build_group_plot_data(df_group: pd.DataFrame, infraslow_band: Tuple[float, float]) -> Dict[str, object]:
    """freqs/mean/sem/n + a group-level bi-Gaussian fit, for one spindle-rate group."""
    freqs, mean, sem, n = _group_mean_curve(df_group["freqs"].tolist(), df_group["corr_mean"].tolist())
    if n == 0:
        raise ValueError("No valid spectra available to build this group's spectrum curve.")

    try:
        fit = demo.fit_isfs(freqs, mean, infraslow_band=infraslow_band)
    except RuntimeError:
        fit = None
    fitted_curve = fitted_freqs = None
    if fit is not None:
        fitted_freqs = np.linspace(*infraslow_band, 200)
        fitted_curve = demo.bigaussian(fitted_freqs, *fit["popt"])

    rates = df_group["spindle_per_min"].to_numpy(dtype=float)
    rates = rates[np.isfinite(rates)]
    spindle_per_min = float(rates.mean()) if rates.size else np.nan
    spindle_per_min_sem = float(rates.std(ddof=1) / np.sqrt(rates.size)) if rates.size > 1 else 0.0

    return dict(
        freqs=freqs, mean=mean, sem=sem, n=n, fit=fit,
        fitted_curve=fitted_curve, fitted_freqs=fitted_freqs,
        spindle_per_min=spindle_per_min, spindle_per_min_sem=spindle_per_min_sem,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Root of {sleep-stage}/{subject_id}.npz files (same layout as "
             "demo_infraslow_yasa_compare.py's --results-dir). Used unless --summary-results is given.",
    )
    parser.add_argument(
        "--summary-results", type=Path, default=None,
        help="Alternate input: a CSV of precomputed subject-level N2-C3 summary parameters "
             "(subject_id, sleep_stage, channel, plus the required summary columns).",
    )
    parser.add_argument(
        "--spectrum-results", type=Path, default=None,
        help="Alternate input: root of {sleep-stage}/{subject_id}.npz files providing "
             "freqs/corr_mean, used together with --summary-results for the Step 6 spectrum plots.",
    )
    parser.add_argument(
        "--sleep-stage", default="N2",
        help="Sleep stage to restrict every step to (defaults to N2 per md/group_analysis.md, "
             "but is not hardcoded -- e.g. N1/N3/NREM also work if the results dir has that stage).",
    )
    parser.add_argument(
        "--channel", default="C3",
        help="EEG channel to restrict every step to (defaults to C3 per md/group_analysis.md, "
             "but is not hardcoded -- e.g. F3/F4/C4/O1/O2 also work).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory outputs are written to. Defaults to "
             "infraslow/results/group_analysis/{sleep-stage}_{channel}.",
    )
    parser.add_argument(
        "--subject-id-column", default="subject_id",
        help="Subject-id column name in --summary-results, if not already 'subject_id'.",
    )
    parser.add_argument("--group-probability-threshold", type=float, default=0.70)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-subjects", type=int, default=None,
        help="Cap the cohort to the first N sorted subjects (--results/npz loading path only); "
             "omit to use every subject found.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes for the --results/npz loading + per-subject curve fit "
             "(--results/npz loading path only). Defaults to $SLURM_CPUS_PER_TASK, else all visible CPUs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args(argv)
    if args.results is None and args.summary_results is None:
        parser.error("one of --results or --summary-results is required")
    if args.output_dir is None:
        args.output_dir = Path(f"infraslow/results/group_analysis/{args.sleep_stage}_{args.channel}")
    return args


def _output_paths(output_dir: Path, sleep_stage: str, channel: str) -> Dict[str, Path]:
    """Output filenames, tokenized on the actual ``{sleep_stage}_{channel}`` in use --
    never hardcoded to N2/C3 -- so different stage/channel runs never collide
    and every filename reflects what it actually contains."""
    prefix = f"{sleep_stage}_{channel}"
    return {
        "validated": output_dir / f"validated_{prefix}_subject_results.csv",
        "excluded": output_dir / f"invalid_or_excluded_{prefix}_subjects.csv",
        "validation_report": output_dir / f"validation_report_{prefix}.txt",
        "assignments": output_dir / f"{prefix}_subject_group_assignments.csv",
        "grouping_report": output_dir / f"grouping_report_{prefix}.txt",
        "comparison": output_dir / f"{prefix}_infraslow_group_comparison.csv",
        "distribution_png": output_dir / f"{prefix}_spindle_rate_group_distribution.png",
        "param_png": output_dir / f"{prefix}_parameter_group_comparisons.png",
        "param_pdf": output_dir / f"{prefix}_parameter_group_comparisons.pdf",
        "compare_png": output_dir / f"{prefix}_infraslow_group_compare.png",
        "compare_pdf": output_dir / f"{prefix}_infraslow_group_compare.pdf",
        "clean_png": output_dir / f"{prefix}_infraslow_power_by_spindle_group.png",
        "clean_pdf": output_dir / f"{prefix}_infraslow_power_by_spindle_group.pdf",
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        "group analysis starting: sleep_stage=%s channel=%s output_dir=%s",
        args.sleep_stage, args.channel, args.output_dir,
    )

    outputs = _output_paths(args.output_dir, args.sleep_stage, args.channel)
    if not args.overwrite:
        existing = [p for p in outputs.values() if p.exists()]
        if existing:
            raise SystemExit(
                f"{len(existing)} output file(s) already exist in {args.output_dir} "
                f"(e.g. {existing[0]}); pass --overwrite to replace them."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 2: load --------------------------------------------------------
    if args.summary_results is not None:
        raw_df = load_subject_records_from_files(
            args.summary_results, args.spectrum_results, args.sleep_stage, args.channel,
        )
        if args.subject_id_column != "subject_id" and args.subject_id_column in raw_df.columns:
            raw_df = raw_df.rename(columns={args.subject_id_column: "subject_id"})
    else:
        raw_df = load_subject_records_from_npz(
            args.results, args.sleep_stage, args.channel,
            n_subjects=args.n_subjects, workers=args.workers,
        )
    raw_df["subject_id"] = raw_df["subject_id"].astype(str)

    # --- Step 3: validate -----------------------------------------------------
    validated, excluded, report = validate_records(raw_df, args.sleep_stage, args.channel)
    if validated.empty:
        raise SystemExit(
            f"No valid {args.sleep_stage}-{args.channel} records available after validation "
            f"({report['n_excluded']}/{report['total_subjects_loaded']} excluded); aborting."
        )

    save_cols = ["subject_id", "sleep_stage", "channel", *REQUIRED_SUMMARY_PARAMS, "power"]
    validated[save_cols].to_csv(outputs["validated"], index=False)
    excluded_save_cols = [c for c in save_cols if c in excluded.columns] + ["exclusion_reason"]
    excluded[excluded_save_cols].to_csv(outputs["excluded"], index=False)
    outputs["validation_report"].write_text(
        format_validation_report(report, args.sleep_stage, args.channel)
    )
    logger.info(
        "validated %d/%d subject(s); %d excluded",
        len(validated), report["total_subjects_loaded"], report["n_excluded"],
    )

    # --- Step 4: GMM group assignment -----------------------------------------
    assignments, fit_result = assign_spindle_rate_groups(
        validated["subject_id"], validated["spindle_per_min"], validated["spindle_per_min_SEM"],
        random_state=args.random_state, probability_threshold=args.group_probability_threshold,
    )
    assignments.insert(1, "sleep_stage", args.sleep_stage)
    assignments.insert(2, "channel", args.channel)
    assignments.to_csv(outputs["assignments"], index=False)
    outputs["grouping_report"].write_text(
        format_grouping_report(
            fit_result, assignments, args.group_probability_threshold, args.sleep_stage, args.channel,
        )
    )
    plot_spindle_rate_distribution(
        spindle_per_min=assignments["spindle_per_min"].to_numpy(),
        spindle_group=assignments["spindle_group"].to_numpy(),
        uncertain=assignments["uncertain_assignment"].to_numpy(),
        gmm_scale=fit_result.scale, gmm_model=fit_result.model,
        centers_original_scale=fit_result.centers_original_scale,
        low_label=LOW_LABEL, high_label=HIGH_LABEL,
        output_png=outputs["distribution_png"],
    )
    logger.info("saved %s", outputs["distribution_png"])

    # --- Step 5: compare N2-C3 infraslow summary parameters -------------------
    compare_df = validated.merge(
        assignments[["subject_id", "spindle_group", "group_probability", "uncertain_assignment"]],
        on="subject_id", how="inner",
    )
    comparison = compare_parameters(
        compare_df, "spindle_group", COMPARISON_PARAMETERS, fdr_alpha=args.fdr_alpha,
    )
    comparison.insert(0, "channel", args.channel)
    comparison.insert(0, "sleep_stage", args.sleep_stage)
    comparison.to_csv(outputs["comparison"], index=False)
    logger.info("saved %s", outputs["comparison"])

    # --- Step 6: reproduce the comparison plot + group figures ----------------
    low_df = compare_df[compare_df["spindle_group"] == LOW_LABEL]
    high_df = compare_df[compare_df["spindle_group"] == HIGH_LABEL]
    infraslow_band = demo.INFRASLOW_BAND

    low_plot = _build_group_plot_data(low_df, infraslow_band)
    high_plot = _build_group_plot_data(high_df, infraslow_band)

    plot_group_infraslow_compare(
        low=low_plot, high=high_plot, infraslow_band=infraslow_band,
        sleep_stage=args.sleep_stage, channel=args.channel,
        output_png=outputs["compare_png"], output_pdf=outputs["compare_pdf"],
    )
    plot_group_spectrum_clean(
        low=low_plot, high=high_plot, infraslow_band=infraslow_band,
        sleep_stage=args.sleep_stage, channel=args.channel,
        output_png=outputs["clean_png"], output_pdf=outputs["clean_pdf"],
    )
    plot_parameter_comparisons(
        df=compare_df, group_col="spindle_group", parameters=COMPARISON_PARAMETERS,
        comparison_df=comparison, low_label=LOW_LABEL, high_label=HIGH_LABEL,
        output_png=outputs["param_png"], output_pdf=outputs["param_pdf"],
    )
    logger.info("saved %s, %s, %s", outputs["compare_png"], outputs["clean_png"], outputs["param_png"])

    logger.info("done: outputs written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
