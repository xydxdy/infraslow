"""CSV reading, column selection, and duplicate-handling tools.

This module groups every operation that works directly on a single CSV/
DataFrame before it is merged: reading from disk, validating and slicing
columns, and resolving duplicate IDs.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

from ..utils import ensure_id_first, fail


def read_csv_strict(path: str, label: str, id_column: str = "ID") -> pd.DataFrame:
    """Read a CSV file, reading every column as a string.

    Reading as ``str`` preserves leading zeros in IDs and avoids surprising
    type coercion. Raises a clear error if the file cannot be read. Exact
    duplicate IDs are collapsed (keeping the last occurrence) so that downstream
    merges do not fan out unexpectedly.
    """
    if not os.path.isfile(path):
        fail(f"{label} file does not exist: {path}")
    try:
        # keep_default_na=True so blank cells become NaN; dtype=str preserves text like '00123'.
        df = pd.read_csv(path, dtype=str)
    except Exception as exc:  # noqa: BLE001 - surface any pandas/IO error clearly
        fail(f"Failed to read {label} file '{path}': {exc}")

    if id_column in df.columns:
        df = df.drop_duplicates(subset=[id_column], keep="last")
    return df


def validate_columns(
    df: pd.DataFrame, requested: List[str], path: str, label: str
) -> None:
    """Ensure every requested column exists in ``df``."""
    missing = [col for col in requested if col not in df.columns]
    if missing:
        fail(
            f"{label} file '{path}' is missing requested column(s): {', '.join(missing)}.\n"
            f"       Available columns: {', '.join(df.columns)}"
        )


def select_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]],
    id_column: str,
    path: str,
    label: str,
) -> pd.DataFrame:
    """Validate the ID column, then return the requested subset of ``df``.

    If ``columns`` is None, all columns are kept (only the ID column is
    validated). Otherwise the ID column is auto-included and every column is
    validated before slicing.
    """
    if id_column not in df.columns:
        fail(
            f"{label} file '{path}' has no ID column '{id_column}'.\n"
            f"       Available columns: {', '.join(df.columns)}"
        )

    if columns is None:
        # Keep all columns from this file (used for stages when --stages-columns omitted).
        return df

    if len(columns) == 0:
        fail(f"{label} column list is empty. Provide at least one column or omit the flag.")

    wanted = ensure_id_first(columns, id_column)
    validate_columns(df, wanted, path, label)
    return df[wanted].copy()


def handle_duplicates(
    df: pd.DataFrame,
    id_column: str,
    policy: str,
    label: str,
) -> pd.DataFrame:
    """Detect and resolve duplicate IDs according to ``policy``."""
    # Count duplicate IDs (ignoring missing IDs, which are not real duplicates).
    non_null = df[df[id_column].notna()]
    dup_mask = non_null[id_column].duplicated(keep=False)
    dup_count = int(dup_mask.sum())

    if dup_count == 0:
        return df

    distinct_dups = non_null.loc[dup_mask, id_column].nunique()
    print(
        f"WARNING: {label} contains {dup_count} duplicated row(s) across "
        f"{distinct_dups} repeated ID value(s) in column '{id_column}'."
    )

    if policy == "error":
        fail(
            f"{label} has duplicate IDs and --duplicate-policy=error. "
            f"Use 'first', 'last', or 'allow' to proceed."
        )
    if policy == "first":
        return df.drop_duplicates(subset=[id_column], keep="first")
    if policy == "last":
        return df.drop_duplicates(subset=[id_column], keep="last")
    # policy == "allow": keep every row as-is.
    return df
