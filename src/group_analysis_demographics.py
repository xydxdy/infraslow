"""Sex / age / BMI infraslow-parameter group comparisons (md/group_analysis.md style).

``group_analysis.py`` already accepts ``--sex``/``--age-group``/``--bmi-group``,
but those *filter* the cohort down to one demographic band and then still run
the usual low/high **spindle-rate** comparison inside that band (see
``run_group_analysis_by_demographic.sh``, which submits one independent job per
band and never compares the bands against each other).

This script instead makes the demographic variable itself the grouping
variable: Male vs Female, Age low/mid/high, BMI low/mid/high, each compared
directly on the same N2-C3 infraslow summary parameters
``group_analysis.py``'s Step 5 uses -- plus the spindle-rate column itself,
since it isn't used to build these groups (no double-dipping concern the way
there is for ``group_analysis.py``'s spindle-rate grouping).

Reuses ``group_analysis.py``'s Step 2 (load) and Step 3 (validate) verbatim
(imported as a sibling module -- this script must stay in the same directory,
``src/``, which Python puts on ``sys.path`` when the script is run directly).

Age/BMI have three bands; ``infraslow.stats.group_comparison.compare_parameter``
only compares two groups at a time, so this script runs all 3 pairwise
comparisons (low-mid, mid-high, low-high) per parameter and applies **one**
BH-FDR correction jointly across every (pair, parameter) test together, rather
than correcting each pair independently -- the pairs share data (the mid band
appears in two of the three), so correcting them separately would understate
how many comparisons are actually being drawn from this cohort.

The existing ``infraslow.viz.group_analysis.plot_group_infraslow_compare`` /
``plot_group_spectrum_clean`` hardcode "low_spindle_rate"/"high_spindle_rate"
in their legends and titles, so they are not reused here (that would mislabel
a Male-vs-Female or Age-low-vs-high plot as a spindle-rate comparison); this
script draws its own generically-labeled version of that spectrum figure
instead. ``plot_parameter_comparisons``/``plot_parameter_distributions`` take
their group labels as arguments and are reused as-is.

Run via Slurm, not the login node, e.g.::

    srun -p normal --time=00:30:00 --mem=32G --cpus-per-task=16 \\
        python3 src/group_analysis_demographics.py --results $SCRATCH/data/npz \\
        --output-dir infraslow/results/group_analysis_demographics/N2_C3
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os

# Keep every worker process single-threaded for BLAS/OpenMP so group_analysis's
# ProcessPoolExecutor (one process per core) doesn't oversubscribe cores --
# must be set before numpy/scipy are imported. Matches group_analysis.py.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import group_analysis as ga  # noqa: E402 - sibling script; see module docstring
from infraslow.stats.group_comparison import compare_parameter  # noqa: E402
from infraslow.viz.group_analysis import (  # noqa: E402
    group_palette,
    plot_cohort_infraslow_compare,
    plot_cohort_spectrum_clean,
    plot_parameter_comparisons,
    plot_parameter_distributions,
)

logger = logging.getLogger(__name__)

#: One entry per demographic variable compared. ``band_order`` is every band
#: the variable can take, in report/plot order -- unlike group_analysis.py's
#: own spindle-rate grouping (which drops its mid band from the "after"
#: plots), every band here is both compared *and* plotted: sex has 2 bands,
#: age/BMI have 3 (low/mid/high), all of them shown together on one figure.
GROUPINGS: Tuple[Dict[str, object], ...] = (
    dict(name="sex", column="sex_group", band_order=("Female", "Male")),
    dict(name="age", column="age_group", band_order=("low", "mid", "high")),
    dict(name="bmi", column="bmi_group", band_order=("low", "mid", "high")),
)


def demographic_comparison_parameters(rate_unit: str) -> List[str]:
    """:data:`group_analysis.COMPARISON_PARAMETERS` plus the spindle-rate column
    itself -- safe to include here since sex/age/BMI groups aren't constructed
    from spindle rate, unlike ``group_analysis.py``'s spindle-rate grouping
    (see module docstring)."""
    rate_col, _ = ga._rate_columns(rate_unit)
    return [*ga.COMPARISON_PARAMETERS, rate_col]


# --------------------------------------------------------------------------- #
# Demographic band assignment (reuses group_analysis.py's cutoff/matching logic)
# --------------------------------------------------------------------------- #
def _band_map(
    subject_ids: pd.Series, values: pd.Series, bands: Sequence[str], *, valid_range: Tuple[float, float], label: str,
) -> Dict[str, str]:
    """``{subject_id: band}`` across every band in ``bands`` (each computed via
    :func:`group_analysis.select_demographic_group`, so the log1p mean +/- std
    cutoff is computed identically to ``group_analysis.py``'s own ``--age-group``/
    ``--bmi-group`` filters)."""
    band_of_id: Dict[str, str] = {}
    for band in bands:
        for subject_id in ga.select_demographic_group(subject_ids, values, band, valid_range=valid_range, label=label):
            band_of_id[subject_id] = band
    return band_of_id


def assign_demographic_groups(validated: pd.DataFrame) -> pd.DataFrame:
    """Adds ``sex_group``/``age_group``/``bmi_group`` columns to ``validated``
    (subject ids with no demographic match get ``None`` in that column)."""
    demographics = ga.load_subject_demographics()
    demographics["ID"] = demographics["ID"].astype(str)
    merged = validated[["subject_id"]].merge(demographics, left_on="subject_id", right_on="ID", how="left")

    sex_by_id = dict(zip(merged["subject_id"], merged["Gender"].map(ga.normalize_sex_label)))
    age_by_id = _band_map(merged["subject_id"], merged["Age"], ("low", "mid", "high"),
                           valid_range=ga.AGE_VALID_RANGE, label="Age")
    bmi_by_id = _band_map(merged["subject_id"], merged["BMI"], ("low", "mid", "high"),
                           valid_range=ga.BMI_VALID_RANGE, label="BMI")

    validated = validated.copy()
    validated["sex_group"] = validated["subject_id"].map(sex_by_id)
    validated["age_group"] = validated["subject_id"].map(age_by_id)
    validated["bmi_group"] = validated["subject_id"].map(bmi_by_id)
    return validated


def format_band_counts_report(name: str, df: pd.DataFrame, column: str, band_order: Sequence[str]) -> str:
    counts = df[column].value_counts()
    lines = [f"{name} grouping (column={column})", "=" * (len(name) + 20 + len(column))]
    for band in band_order:
        lines.append(f"  {band:<10}: {int(counts.get(band, 0))}")
    lines.append(f"  {'(unmatched)':<10}: {int(df[column].isna().sum())}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Pairwise comparison across a demographic variable's bands, with one BH-FDR
# correction applied jointly across every (pair, parameter) test.
# --------------------------------------------------------------------------- #
def compare_group_bands(
    df: pd.DataFrame, group_col: str, band_order: Sequence[str], parameters: Sequence[str], *, fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """All pairwise comparisons between ``band_order``'s bands (2 bands -> 1
    pair, 3 bands -> 3 pairs: low-mid, mid-high, low-high), one BH-FDR
    correction applied jointly across every (pair, parameter) test -- see
    module docstring for why not per-pair.

    Returns:
        One row per (pair, parameter): ``group_a, group_b, parameter,
        group_a_n/mean/sd/median/q1/q3, group_b_n/mean/sd/median/q1/q3, test,
        statistic, p_value, q_value, effect_size_name, effect_size,
        significant_fdr``. ``effect_size``/``statistic`` direction is always
        ``group_b - group_a`` (:func:`infraslow.stats.group_comparison.compare_parameter`'s
        fixed ``high - low`` convention, with ``group_a``/``group_b`` in the
        ``low``/``high`` argument positions respectively).
    """
    rows: List[Dict[str, object]] = []
    p_values: List[float] = []
    for group_a, group_b in itertools.combinations(band_order, 2):
        for parameter in parameters:
            a_values = df.loc[df[group_col] == group_a, parameter].to_numpy(dtype=float)
            b_values = df.loc[df[group_col] == group_b, parameter].to_numpy(dtype=float)
            comparison = compare_parameter(parameter, a_values, b_values)
            rows.append({
                "group_a": group_a, "group_b": group_b, "parameter": parameter,
                "group_a_n": comparison.low["n"], "group_b_n": comparison.high["n"],
                "group_a_mean": comparison.low["mean"], "group_a_sd": comparison.low["standard_deviation"],
                "group_a_median": comparison.low["median"], "group_a_q1": comparison.low["q1"],
                "group_a_q3": comparison.low["q3"],
                "group_b_mean": comparison.high["mean"], "group_b_sd": comparison.high["standard_deviation"],
                "group_b_median": comparison.high["median"], "group_b_q1": comparison.high["q1"],
                "group_b_q3": comparison.high["q3"],
                "test": comparison.test, "statistic": comparison.statistic, "p_value": comparison.p_value,
                "effect_size_name": comparison.effect_size_name, "effect_size": comparison.effect_size,
            })
            p_values.append(comparison.p_value)

    result = pd.DataFrame(rows)
    p_array = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p_array)
    q_values = np.full_like(p_array, np.nan)
    if valid.any():
        _, q_valid, _, _ = multipletests(p_array[valid], alpha=fdr_alpha, method="fdr_bh")
        q_values[valid] = q_valid
    result["q_value"] = q_values
    result["significant_fdr"] = np.isfinite(q_values) & (q_values < fdr_alpha)
    return result


# --------------------------------------------------------------------------- #
# Generic (non-spindle-rate-labeled) two-group spectrum plot.
# --------------------------------------------------------------------------- #
def plot_demographic_spectrum_compare(
    *, groups: Dict[str, Dict[str, object]], band_order: Sequence[str],
    infraslow_band: Tuple[float, float], sleep_stage: str, channel: str, grouping_name: str,
    output_png: Path, output_pdf: Path, ci: float = 0.95,
) -> None:
    """Mean +/- CI spectrum for every one of a demographic variable's bands
    (2 for sex, 3 for age/BMI) on one figure -- a genuinely generic-label
    counterpart to ``infraslow.viz.group_analysis.plot_group_spectrum_clean``,
    which hardcodes "low_spindle_rate"/"high_spindle_rate" text (see module docstring).

    Args:
        groups: ``{band: group_plot_data}`` for every band in ``band_order``
            (see ``group_analysis.py``'s ``_build_group_plot_data``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, color in zip(band_order, group_palette(len(band_order))):
        group = groups[label]
        freqs, mean, sem, n = group["freqs"], group["mean"], group["sem"], group["n"]
        band_mask = (freqs >= infraslow_band[0]) & (freqs <= infraslow_band[1])
        z = float(sp_stats.norm.ppf(0.5 + ci / 2)) if n > 1 else 0.0
        half_width = z * sem
        ax.fill_between(freqs[band_mask], (mean - half_width)[band_mask], (mean + half_width)[band_mask],
                         color=color, alpha=0.25)
        ax.plot(freqs[band_mask], mean[band_mask], color=color, lw=2.0, label=f"{label} (n={n})")

    ax.axhline(0.0, ls=":", color="0.3")
    ax.set_xlim(infraslow_band)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Baseline-corrected relative power")
    ax.set_title(f"{sleep_stage}, {channel} infraslow spectrum by {grouping_name} ({int(ci * 100)}% CI)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Root of {sleep-stage}/{subject_id}.npz files. Used unless --summary-results is given.",
    )
    parser.add_argument(
        "--summary-results", type=Path, default=None,
        help="Alternate input: a CSV of precomputed subject-level N2-C3 summary parameters.",
    )
    parser.add_argument(
        "--spectrum-results", type=Path, default=None,
        help="Alternate input: root of {sleep-stage}/{subject_id}.npz files providing "
             "freqs/corr_mean, used together with --summary-results for the spectrum plots.",
    )
    parser.add_argument("--sleep-stage", default="N2", help="Sleep stage to restrict every step to.")
    parser.add_argument("--channel", default="C3", help="EEG channel to restrict every step to.")
    parser.add_argument(
        "--rate-unit", choices=["min", "hr"], default="min",
        help="Unit for the spindle-rate comparison parameter/column ('min' or 'hr').",
    )
    parser.add_argument(
        "--dominant-freq-hz", type=float, default=None,
        help="Restrict to subjects whose own dominant_freq_hz equals this value before "
             "computing any demographic comparison (see group_analysis.py). Omit for the whole cohort.",
    )
    parser.add_argument(
        "--exclude-drug-users", nargs="*", default=None, metavar="DRUG",
        help="Exclude subjects flagged as using any of the given drug(s) (see group_analysis.py). "
             "Pass with no values, or 'All', to exclude every drug in --drug-exclude-list.",
    )
    parser.add_argument(
        "--drug-exclude-list", type=Path, default=Path(ga.DEFAULT_DRUG_EXCLUDE_LIST),
        help="drug/drug_exclude.csv-style file used when --exclude-drug-users is passed with no values or 'All'.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory outputs are written to. Defaults to "
             "infraslow/results/group_analysis_demographics/{sleep-stage}_{channel}.",
    )
    parser.add_argument(
        "--subject-id-column", default="subject_id",
        help="Subject-id column name in --summary-results, if not already 'subject_id'.",
    )
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument(
        "--n-subjects", type=int, default=None,
        help="Cap the cohort to the first N sorted subjects (--results/npz loading path only).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes for the --results/npz loading + per-subject curve fit. "
             "Defaults to $SLURM_CPUS_PER_TASK, else all visible CPUs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args(argv)
    if args.results is None and args.summary_results is None:
        parser.error("one of --results or --summary-results is required")
    if args.output_dir is None:
        args.output_dir = Path(f"infraslow/results/group_analysis_demographics/{args.sleep_stage}_{args.channel}")
    return args


def _output_paths(output_dir: Path, sleep_stage: str, channel: str) -> Dict[str, Path]:
    prefix = f"{sleep_stage}_{channel}"
    outputs = {
        "validated": output_dir / f"validated_{prefix}_subject_results.csv",
        "excluded": output_dir / f"invalid_or_excluded_{prefix}_subjects.csv",
        "validation_report": output_dir / f"validation_report_{prefix}.txt",
        "param_before_png": output_dir / f"{prefix}_parameter_distributions_before.png",
        "param_before_pdf": output_dir / f"{prefix}_parameter_distributions_before.pdf",
        "compare_before_png": output_dir / f"{prefix}_infraslow_compare_before.png",
        "compare_before_pdf": output_dir / f"{prefix}_infraslow_compare_before.pdf",
        "clean_before_png": output_dir / f"{prefix}_infraslow_power_before.png",
        "clean_before_pdf": output_dir / f"{prefix}_infraslow_power_before.pdf",
    }
    for grouping in GROUPINGS:
        name = grouping["name"]
        outputs[f"{name}_comparison"] = output_dir / f"{prefix}_{name}_group_comparison.csv"
        outputs[f"{name}_counts"] = output_dir / f"{prefix}_{name}_group_counts.txt"
        outputs[f"{name}_param_png"] = output_dir / f"{prefix}_{name}_parameter_group_comparisons.png"
        outputs[f"{name}_param_pdf"] = output_dir / f"{prefix}_{name}_parameter_group_comparisons.pdf"
        outputs[f"{name}_spectrum_png"] = output_dir / f"{prefix}_{name}_infraslow_spectrum_compare.png"
        outputs[f"{name}_spectrum_pdf"] = output_dir / f"{prefix}_{name}_infraslow_spectrum_compare.pdf"
    return outputs


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        "demographic group analysis starting: sleep_stage=%s channel=%s rate_unit=%s output_dir=%s",
        args.sleep_stage, args.channel, args.rate_unit, args.output_dir,
    )
    parameters = demographic_comparison_parameters(args.rate_unit)

    outputs = _output_paths(args.output_dir, args.sleep_stage, args.channel)
    if not args.overwrite:
        existing = [p for p in outputs.values() if p.exists()]
        if existing:
            raise SystemExit(
                f"{len(existing)} output file(s) already exist in {args.output_dir} "
                f"(e.g. {existing[0]}); pass --overwrite to replace them."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 2: load (reuses group_analysis.py verbatim) ---------------------
    if args.summary_results is not None:
        raw_df = ga.load_subject_records_from_files(
            args.summary_results, args.spectrum_results, args.sleep_stage, args.channel, rate_unit=args.rate_unit,
        )
        if args.subject_id_column != "subject_id" and args.subject_id_column in raw_df.columns:
            raw_df = raw_df.rename(columns={args.subject_id_column: "subject_id"})
    else:
        raw_df = ga.load_subject_records_from_npz(
            args.results, args.sleep_stage, args.channel,
            n_subjects=args.n_subjects, workers=args.workers, rate_unit=args.rate_unit,
        )
    raw_df["subject_id"] = raw_df["subject_id"].astype(str)

    # --- Optional pre-filters, before Step 3 (same as group_analysis.py) ------
    if args.dominant_freq_hz is not None:
        dominant = pd.to_numeric(raw_df["dominant_freq_hz"], errors="coerce")
        dominant_mask = np.isclose(dominant, args.dominant_freq_hz, atol=ga.DOMINANT_FREQ_TOLERANCE, rtol=0.0)
        n_before = len(raw_df)
        raw_df = raw_df.loc[dominant_mask].reset_index(drop=True)
        logger.info("dominant_freq_hz == %s filter: %d/%d subject(s) kept", args.dominant_freq_hz, len(raw_df), n_before)
        if raw_df.empty:
            raise SystemExit(f"No subjects with dominant_freq_hz == {args.dominant_freq_hz}; aborting.")

    if args.exclude_drug_users is not None:
        drug_names = ga.resolve_requested_drug_names(args.exclude_drug_users, args.drug_exclude_list)
        excluded_ids = ga.load_drug_excluded_subject_ids(drug_names)
        n_before = len(raw_df)
        raw_df = raw_df.loc[~raw_df["subject_id"].isin(excluded_ids)].reset_index(drop=True)
        logger.info(
            "drug exclusion (%s): %d/%d subject(s) kept", ", ".join(drug_names), len(raw_df), n_before,
        )
        if raw_df.empty:
            raise SystemExit(f"No subjects remain after excluding drug users ({', '.join(drug_names)}); aborting.")

    # --- Step 3: validate (reuses group_analysis.py verbatim) -----------------
    validated, excluded, report = ga.validate_records(raw_df, args.sleep_stage, args.channel, rate_unit=args.rate_unit)
    if validated.empty:
        raise SystemExit(
            f"No valid {args.sleep_stage}-{args.channel} records available after validation "
            f"({report['n_excluded']}/{report['total_subjects_loaded']} excluded); aborting."
        )

    save_cols = [
        "subject_id", "sleep_stage", "channel", *ga.required_summary_params(args.rate_unit),
        "power", "dominant_freq_hz",
    ]
    validated[save_cols].to_csv(outputs["validated"], index=False)
    excluded_save_cols = [c for c in save_cols if c in excluded.columns] + ["exclusion_reason"]
    excluded[excluded_save_cols].to_csv(outputs["excluded"], index=False)
    outputs["validation_report"].write_text(ga.format_validation_report(report, args.sleep_stage, args.channel))
    logger.info("validated %d/%d subject(s); %d excluded", len(validated), report["total_subjects_loaded"], report["n_excluded"])

    # --- Whole-cohort reference plots, before any demographic split -----------
    infraslow_band = ga.INFRASLOW_BAND
    cohort_plot = ga._build_group_plot_data(validated, infraslow_band, rate_unit=args.rate_unit)
    plot_cohort_infraslow_compare(
        cohort=cohort_plot, infraslow_band=infraslow_band, sleep_stage=args.sleep_stage, channel=args.channel,
        output_png=outputs["compare_before_png"], output_pdf=outputs["compare_before_pdf"], rate_unit=args.rate_unit,
    )
    plot_cohort_spectrum_clean(
        cohort=cohort_plot, infraslow_band=infraslow_band, sleep_stage=args.sleep_stage, channel=args.channel,
        output_png=outputs["clean_before_png"], output_pdf=outputs["clean_before_pdf"],
    )
    plot_parameter_distributions(
        df=validated, parameters=parameters, output_png=outputs["param_before_png"], output_pdf=outputs["param_before_pdf"],
    )
    logger.info("saved whole-cohort reference plots (before any demographic split)")

    # --- Assign sex_group/age_group/bmi_group ---------------------------------
    validated = assign_demographic_groups(validated)

    # --- Per-demographic-variable comparison + plots --------------------------
    for grouping in GROUPINGS:
        name, column, band_order = grouping["name"], grouping["column"], grouping["band_order"]

        outputs[f"{name}_counts"].write_text(format_band_counts_report(name, validated, column, band_order))

        comparison = compare_group_bands(validated, column, band_order, parameters, fdr_alpha=args.fdr_alpha)
        comparison.insert(0, "channel", args.channel)
        comparison.insert(0, "sleep_stage", args.sleep_stage)
        comparison.insert(0, "grouping", name)
        comparison.to_csv(outputs[f"{name}_comparison"], index=False)
        logger.info("saved %s", outputs[f"{name}_comparison"])

        try:
            plot_parameter_comparisons(
                df=validated, group_col=column, parameters=parameters, comparison_df=comparison,
                group_labels=band_order,
                output_png=outputs[f"{name}_param_png"], output_pdf=outputs[f"{name}_param_pdf"],
            )

            group_plots = {
                band: ga._build_group_plot_data(validated[validated[column] == band], infraslow_band, rate_unit=args.rate_unit)
                for band in band_order
            }
            plot_demographic_spectrum_compare(
                groups=group_plots, band_order=band_order,
                infraslow_band=infraslow_band, sleep_stage=args.sleep_stage, channel=args.channel,
                grouping_name=name, output_png=outputs[f"{name}_spectrum_png"], output_pdf=outputs[f"{name}_spectrum_pdf"],
            )
            logger.info("saved %s, %s", outputs[f"{name}_param_png"], outputs[f"{name}_spectrum_png"])
        except ValueError as exc:
            logger.warning("skipping %s plots (%s): %s", name, ", ".join(band_order), exc)

    logger.info("done: outputs written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
