"""Core merge logic exposed as a reusable function.

The public entry point here is :func:`merge_csv_columns`, a plain function that
takes typed arguments and returns the merged :class:`pandas.DataFrame`. It has
no dependency on argparse, so it can be imported and called from notebooks,
tests, or other scripts as well as from the CLI.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

from ..io.csv_tools import handle_duplicates, read_csv_strict, select_columns
from ..utils import ensure_output_dir, fail, normalize_ids


def merge_frames(
    stages: pd.DataFrame,
    metadata: pd.DataFrame,
    apnea: pd.DataFrame,
    id_column: str,
    join_type: str,
) -> pd.DataFrame:
    """Merge metadata and apnea onto the stages base table.

    Overlapping (non-ID) column names are disambiguated with ``_metadata`` and
    ``_apnea`` suffixes. The ID column is never duplicated.
    """
    # First merge: stages (base) + metadata.
    merged = stages.merge(
        metadata,
        on=id_column,
        how=join_type,
        suffixes=("", "_metadata"),
    )

    # Second merge: result + apnea.
    merged = merged.merge(
        apnea,
        on=id_column,
        how=join_type,
        suffixes=("", "_apnea"),
    )
    return merged


def print_summary(
    stages_rows: int,
    metadata_rows: int,
    apnea_rows: int,
    output: pd.DataFrame,
    id_column: str,
    stages: pd.DataFrame,
    metadata: pd.DataFrame,
    apnea: pd.DataFrame,
    output_file: str,
) -> None:
    """Print a human-friendly completion summary."""
    stage_ids = set(stages[id_column].dropna())
    meta_ids = set(metadata[id_column].dropna())
    apnea_ids = set(apnea[id_column].dropna())

    matched_meta = len(stage_ids & meta_ids)
    matched_apnea = len(stage_ids & apnea_ids)
    unmatched_meta = len(stage_ids - meta_ids)
    unmatched_apnea = len(stage_ids - apnea_ids)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Rows read - stages   : {stages_rows}")
    print(f"Rows read - metadata : {metadata_rows}")
    print(f"Rows read - apnea    : {apnea_rows}")
    print(f"Rows in output       : {len(output)}")
    print(f"Columns in output    : {len(output.columns)}")
    print("-" * 60)
    print(f"Stage IDs matched in metadata : {matched_meta} (unmatched: {unmatched_meta})")
    print(f"Stage IDs matched in apnea    : {matched_apnea} (unmatched: {unmatched_apnea})")
    print("-" * 60)
    print(f"Output written to    : {os.path.abspath(output_file)}")
    print("=" * 60)


def merge_csv_columns(
    *,
    stages_file: str,
    metadata_file: str,
    metadata_columns: List[str],
    apnea_file: str,
    apnea_columns: List[str],
    output_file: str,
    stages_columns: Optional[List[str]] = None,
    id_column: str = "ID",
    join_type: str = "left",
    duplicate_policy: str = "error",
    summary: bool = True,
) -> pd.DataFrame:
    """Merge selected columns from the stages, metadata, and apnea CSVs.

    This is the reusable heart of the tool. It reads each file, selects the
    requested columns, normalizes IDs, resolves duplicates, merges everything
    onto the stages base table, writes the result, and returns the merged
    DataFrame.

    Args:
        stages_file: Path to the base/left stages CSV.
        metadata_file: Path to the metadata CSV.
        metadata_columns: Columns to keep from the metadata file (ID auto-added).
        apnea_file: Path to the apnea CSV.
        apnea_columns: Columns to keep from the apnea file (ID auto-added).
        output_file: Path for the merged output CSV.
        stages_columns: Columns to keep from the stages file; None keeps all.
        id_column: Name of the shared ID column used to match rows.
        join_type: One of ``left``, ``inner``, ``right``, ``outer``.
        duplicate_policy: One of ``error``, ``first``, ``last``, ``allow``.
        summary: When True, print a completion summary to stdout.

    Returns:
        The merged :class:`pandas.DataFrame`.
    """
    # 1. Read each input file (everything as string to protect IDs like '00123').
    stages_raw = read_csv_strict(stages_file, "Stages", id_column)
    metadata_raw = read_csv_strict(metadata_file, "Metadata", id_column)
    apnea_raw = read_csv_strict(apnea_file, "Apnea", id_column)

    stages_rows = len(stages_raw)
    metadata_rows = len(metadata_raw)
    apnea_rows = len(apnea_raw)

    # 2. Select requested columns (ID auto-included; stages keeps all if omitted).
    stages = select_columns(stages_raw, stages_columns, id_column, stages_file, "Stages")
    metadata = select_columns(metadata_raw, metadata_columns, id_column, metadata_file, "Metadata")
    apnea = select_columns(apnea_raw, apnea_columns, id_column, apnea_file, "Apnea")

    # 3. Normalize IDs for reliable matching.
    stages[id_column] = normalize_ids(stages[id_column])
    metadata[id_column] = normalize_ids(metadata[id_column])
    apnea[id_column] = normalize_ids(apnea[id_column])

    # 4. Resolve duplicate IDs per the chosen policy.
    stages = handle_duplicates(stages, id_column, duplicate_policy, "Stages")
    metadata = handle_duplicates(metadata, id_column, duplicate_policy, "Metadata")
    apnea = handle_duplicates(apnea, id_column, duplicate_policy, "Apnea")

    # 5. Merge everything onto the stages base table.
    merged = merge_frames(stages, metadata, apnea, id_column, join_type)

    # 6. Ensure the output directory exists, then write.
    ensure_output_dir(output_file)
    try:
        merged.to_csv(output_file, index=False)
    except Exception as exc:  # noqa: BLE001
        fail(f"Failed to write output file '{output_file}': {exc}")

    # 7. Report what happened.
    if summary:
        print_summary(
            stages_rows,
            metadata_rows,
            apnea_rows,
            merged,
            id_column,
            stages,
            metadata,
            apnea,
            output_file,
        )

    return merged
