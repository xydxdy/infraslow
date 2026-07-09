"""Sleep-parameter calculations derived from merged PSG data.

Functions here take the merged DataFrame (or a written CSV) and either add
computed sleep metrics as new columns or derive subject groupings from them.
They are tolerant of the all-string columns produced by the merge step (values
are coerced to numeric, with invalid/missing entries becoming NaN).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..utils import ensure_output_dir, normalize_ids


def compute_ahi(
    df: pd.DataFrame,
    *,
    oa_column: str = "Numbers_OA_Total",
    hi_column: str = "Numbers_HI_Total",
    tst_column: str = "TST",
    output_column: str = "AHI",
) -> pd.DataFrame:
    """Compute the Apnea-Hypopnea Index (AHI) and add it as a new column.

    AHI is the number of apnea and hypopnea events per hour of sleep::

        AHI = (Numbers_OA_Total + Numbers_HI_Total) / (TST_hours)

    ``TST`` (total sleep time) is stored in **minutes**, so it is divided by 60
    to convert to hours. Rows with a missing or non-positive TST yield ``NaN``
    (AHI is undefined without sleep time), as do rows with missing event counts.

    Args:
        df: Merged DataFrame containing the OA, HI, and TST columns.
        oa_column: Column with the obstructive-apnea event count.
        hi_column: Column with the hypopnea event count.
        tst_column: Column with total sleep time, in minutes.
        output_column: Name of the AHI column to add.

    Returns:
        The same DataFrame with the AHI column added (modified in place and
        also returned for chaining).
    """
    missing = [c for c in (oa_column, hi_column, tst_column) if c not in df.columns]
    if missing:
        raise KeyError(
            f"compute_ahi: missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    # Columns arrive as strings from the merge step; coerce to numeric safely.
    oa = pd.to_numeric(df[oa_column], errors="coerce")
    hi = pd.to_numeric(df[hi_column], errors="coerce")
    tst_minutes = pd.to_numeric(df[tst_column], errors="coerce")

    # Convert TST to hours; treat non-positive sleep time as invalid (NaN).
    tst_hours = tst_minutes / 60.0
    tst_hours = tst_hours.where(tst_hours > 0)

    df[output_column] = (oa + hi) / tst_hours
    return df


# --------------------------------------------------------------------------- #
# Two-threshold subject grouping
# --------------------------------------------------------------------------- #
def _resolve_value_column(df: pd.DataFrame, column: str) -> None:
    """Validate that the requested value column exists, raising clearly if not.

    We never silently pick a different column: if the requested one is missing,
    the caller must pass the exact column name.
    """
    if column in df.columns:
        return
    raise ValueError(
        f"Column '{column}' not found in the CSV. "
        f"Available columns: {', '.join(df.columns)}. "
        f"Please pass the exact column name — it will not be guessed."
    )


def _validate_gap(
    lower_max: float, upper_min: float, minimum_separation_gap: float
) -> float:
    """Validate the two thresholds and return the actual separation gap.

    ``upper_min`` must sit strictly above ``lower_max``, and the gap between them
    must be at least ``minimum_separation_gap``.
    """
    if upper_min <= lower_max:
        raise ValueError(
            f"The upper threshold ({upper_min}) must be greater than the lower "
            f"threshold ({lower_max}); the two bands must not overlap."
        )
    actual_gap = upper_min - lower_max
    if actual_gap < minimum_separation_gap:
        raise ValueError(
            f"Separation gap {actual_gap} (upper - lower threshold) is smaller than "
            f"the required minimum of {minimum_separation_gap}. Widen the thresholds "
            f"so the upper boundary is at least {minimum_separation_gap} units above "
            f"the lower one."
        )
    return actual_gap


def _select_two_threshold_groups(
    df: pd.DataFrame,
    *,
    id_column: str,
    value_column: str,
    lower_max: float,
    upper_min: float,
    lower_group: str,
    upper_group: str,
    metric_label: str,
    value_header: str,
    minimum_separation_gap: float,
    duplicate_policy: str,
    exclude_missing: bool,
    output_csv: Optional[str],
    excluded_output_csv: Optional[str],
) -> Dict[str, List[str]]:
    """Generic two-threshold subject grouping shared by all metrics.

    Operates on an in-memory DataFrame (no CSV round-trip), so the same frame can
    be threaded through several grouping steps. The input ``df`` is never mutated.

    Subjects are split into two bands by a numeric ``value_column``::

        lower band (value <= lower_max)             -> ``lower_group``
        excluded intermediate (lower < v < upper)
        upper band (value >= upper_min)             -> ``upper_group``

    Intermediate subjects are intentionally removed from both groups to widen the
    separation. The returned dict is always keyed ``control``/``focus``.
    """
    if duplicate_policy not in {"error", "first", "last", "mean"}:
        raise ValueError(
            f"Invalid duplicate_policy '{duplicate_policy}'. "
            "Choose one of: error, first, last, mean."
        )

    # Validate thresholds before touching the data so misconfiguration fails fast.
    actual_gap = _validate_gap(lower_max, upper_min, minimum_separation_gap)

    total_input_rows = len(df)

    if id_column not in df.columns:
        raise ValueError(
            f"ID column '{id_column}' not found in the data. "
            f"Available columns: {', '.join(df.columns)}"
        )
    _resolve_value_column(df, value_column)

    # Normalize IDs locally (strip whitespace, drop float artifacts, preserve
    # zeros) without mutating the caller's DataFrame.
    ids = normalize_ids(df[id_column])

    # Classify values: missing (blank/NA) vs non-numeric (present but unparseable).
    raw = df[value_column].astype(str).str.strip()
    missing_mask = df[value_column].isna() | raw.isin(
        {"", "nan", "NaN", "None", "<NA>"}
    )
    numeric = pd.to_numeric(raw.where(~missing_mask), errors="coerce")
    nonnumeric_mask = (~missing_mask) & numeric.isna()

    n_missing = int(missing_mask.sum())
    n_nonnumeric = int(nonnumeric_mask.sum())

    if not exclude_missing and (n_missing or n_nonnumeric):
        raise ValueError(
            f"Found {n_missing} missing and {n_nonnumeric} non-numeric value(s) in "
            f"'{value_column}', but exclude_missing=False. Clean the data or set "
            "exclude_missing=True to drop them."
        )

    # Keep only rows with a usable ID and a valid numeric value.
    valid_mask = ids.notna() & ~missing_mask & ~nonnumeric_mask
    work = pd.DataFrame(
        {id_column: ids[valid_mask], value_column: numeric[valid_mask]}
    )

    # Resolve duplicate subject IDs per policy.
    dup_ids = work.loc[work[id_column].duplicated(keep=False), id_column].nunique()
    if duplicate_policy == "error" and dup_ids:
        raise ValueError(
            f"Found {dup_ids} duplicated subject ID(s) in '{id_column}'. "
            "Use duplicate_policy='first', 'last', or 'mean' to resolve them."
        )
    if duplicate_policy == "mean":
        work = work.groupby(id_column, as_index=False)[value_column].mean()
    elif duplicate_policy in {"first", "last"}:
        work = work.drop_duplicates(subset=[id_column], keep=duplicate_policy)
    # 'error' with no duplicates: nothing to do.

    total_unique_subjects = int(work[id_column].nunique())

    # Build the selection masks (the heart of the two-threshold method).
    value = work[value_column]
    lower_mask = value <= lower_max
    upper_mask = value >= upper_min
    intermediate_mask = (value > lower_max) & (value < upper_min)

    lower_ids = sorted(work.loc[lower_mask, id_column].unique())
    upper_ids = sorted(work.loc[upper_mask, id_column].unique())

    # Map each band onto its control/focus label.
    groups: Dict[str, List[str]] = {lower_group: lower_ids, upper_group: upper_ids}
    control_ids = groups["control"]
    focus_ids = groups["focus"]

    # Defensive guarantee: the two groups must never share a subject.
    overlap = set(control_ids) & set(focus_ids)
    if overlap:
        raise AssertionError(
            f"Control and focus groups overlap on {len(overlap)} subject(s); "
            "check the thresholds."
        )

    selected_groups: Dict[str, List[str]] = {
        "control": control_ids,
        "focus": focus_ids,
    }

    intermediate = work.loc[intermediate_mask, [id_column, value_column]].sort_values(
        id_column
    )
    n_intermediate = len(intermediate)

    # Write outputs if requested.
    if output_csv:
        ensure_output_dir(output_csv)
        rows = [{"subject_id": sid, "group": lower_group} for sid in lower_ids]
        rows += [{"subject_id": sid, "group": upper_group} for sid in upper_ids]
        pd.DataFrame(rows, columns=["subject_id", "group"]).to_csv(
            output_csv, index=False
        )

    if excluded_output_csv:
        ensure_output_dir(excluded_output_csv)
        excluded = pd.DataFrame(
            {
                "subject_id": intermediate[id_column].values,
                value_header: intermediate[value_column].values,
                "exclusion_reason": "intermediate_between_control_and_focus",
            }
        )
        excluded.to_csv(excluded_output_csv, index=False)

    _print_group_summary(
        metric_label=metric_label,
        value_column=value_column,
        total_input_rows=total_input_rows,
        total_unique_subjects=total_unique_subjects,
        n_control=len(control_ids),
        n_focus=len(focus_ids),
        n_intermediate=n_intermediate,
        n_missing=n_missing,
        n_nonnumeric=n_nonnumeric,
        n_duplicates=int(dup_ids),
        lower_max=lower_max,
        upper_min=upper_min,
        lower_group=lower_group,
        upper_group=upper_group,
        minimum_separation_gap=minimum_separation_gap,
        actual_gap=actual_gap,
        duplicate_policy=duplicate_policy,
    )

    return selected_groups


def selected_groups_by_ahi(
    df: pd.DataFrame,
    id_column: str = "ID",
    apnea_column: str = "AHI",
    control_max: float = 10.0,
    focus_min: float = 30.0,
    minimum_separation_gap: float = 20.0,
    duplicate_policy: str = "error",
    exclude_missing: bool = True,
    output_csv: Optional[str] = None,
    excluded_output_csv: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Build two strongly separated groups from an apnea severity index (AHI).

    Higher AHI means more severe apnea, so::

        control:                AHI <= control_max   (least severe)
        excluded intermediate:  control_max < AHI < focus_min
        focus:                  AHI >= focus_min     (most severe)

    Intermediate subjects are intentionally removed to widen the separation.

    Returns ``{"control": [...], "focus": [...]}`` with sorted, unique, disjoint
    IDs. See :func:`_select_two_threshold_groups` for the shared mechanics.
    """
    return _select_two_threshold_groups(
        df,
        id_column=id_column,
        value_column=apnea_column,
        lower_max=control_max,
        upper_min=focus_min,
        lower_group="control",
        upper_group="focus",
        metric_label="AHI",
        value_header="apnea_value",
        minimum_separation_gap=minimum_separation_gap,
        duplicate_policy=duplicate_policy,
        exclude_missing=exclude_missing,
        output_csv=output_csv,
        excluded_output_csv=excluded_output_csv,
    )


def selected_groups_by_se(
    df: pd.DataFrame,
    id_column: str = "ID",
    se_column: str = "SE",
    focus_max: float = 70.0,
    control_min: float = 90.0,
    minimum_separation_gap: float = 20.0,
    duplicate_policy: str = "error",
    exclude_missing: bool = True,
    output_csv: Optional[str] = None,
    excluded_output_csv: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Build two strongly separated groups from sleep efficiency (SE).

    Sleep efficiency runs the *opposite* way to AHI: a higher SE means healthier
    sleep, so the high band is the control group and the low band is the focus
    group::

        focus:                  SE <= focus_max      (poor sleepers)
        excluded intermediate:  focus_max < SE < control_min
        control:                SE >= control_min    (good sleepers)

    Intermediate subjects are intentionally removed to widen the separation.

    Returns ``{"control": [...], "focus": [...]}`` with sorted, unique, disjoint
    IDs. See :func:`_select_two_threshold_groups` for the shared mechanics.
    """
    return _select_two_threshold_groups(
        df,
        id_column=id_column,
        value_column=se_column,
        lower_max=focus_max,
        upper_min=control_min,
        lower_group="focus",
        upper_group="control",
        metric_label="SE",
        value_header="se_value",
        minimum_separation_gap=minimum_separation_gap,
        duplicate_policy=duplicate_policy,
        exclude_missing=exclude_missing,
        output_csv=output_csv,
        excluded_output_csv=excluded_output_csv,
    )


def _print_group_summary(**s: object) -> None:
    """Print a human-friendly summary of a group-selection run."""
    metric = s["metric_label"]
    lower_group = s["lower_group"]
    upper_group = s["upper_group"]
    print("\n" + "=" * 60)
    print(f"SUBJECT GROUP SELECTION COMPLETE ({metric})")
    print("=" * 60)
    print(f"Total input rows               : {s['total_input_rows']}")
    print(f"Total unique subjects          : {s['total_unique_subjects']}")
    print(f"Control subjects               : {s['n_control']}")
    print(f"Focus subjects                 : {s['n_focus']}")
    print(f"Excluded intermediate subjects : {s['n_intermediate']}")
    print(f"Excluded - missing {metric:<11} : {s['n_missing']}")
    print(f"Excluded - non-numeric {metric:<7} : {s['n_nonnumeric']}")
    print(f"Duplicate subject IDs          : {s['n_duplicates']}")
    print("-" * 60)
    print(f"Value column                   : {s['value_column']}")
    print(f"Lower band ({metric} <= {s['lower_max']:<6}) : group '{lower_group}'")
    print(f"Upper band ({metric} >= {s['upper_min']:<6}) : group '{upper_group}'")
    print(f"Required separation gap        : {s['minimum_separation_gap']}")
    print(f"Actual separation gap          : {s['actual_gap']}")
    print(f"Duplicate policy               : {s['duplicate_policy']}")
    print("-" * 60)
    print(
        f"Note: intermediate subjects ({s['n_intermediate']}) with "
        f"{s['lower_max']} < {metric} < {s['upper_min']} were intentionally removed "
        "to strengthen the separation between the control and focus groups."
    )
    print("=" * 60)
