"""Generic, file-format-agnostic helper utilities.

These functions know nothing about pandas merging or the CLI — they are small,
reusable building blocks (error reporting, ID handling, filesystem helpers).
"""

from __future__ import annotations

import os
import sys
from typing import List, NoReturn

import pandas as pd


def fail(message: str) -> NoReturn:
    """Print a clear error message to stderr and exit with a non-zero status."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def ensure_id_first(columns: List[str], id_column: str) -> List[str]:
    """Return ``columns`` with ``id_column`` present exactly once, listed first.

    The ID column is required for merging, so we add it automatically when the
    user forgets it. Duplicates are removed while preserving order.
    """
    ordered: List[str] = [id_column]
    for col in columns:
        if col != id_column and col not in ordered:
            ordered.append(col)
    return ordered


def normalize_ids(series: pd.Series) -> pd.Series:
    """Normalize an ID column for safe matching.

    - Convert IDs to strings.
    - Strip leading/trailing whitespace.
    - Drop a trailing float artifact (e.g. ``8631.0`` -> ``8631``). Some files
      store integer IDs as floats, so they read back as ``"8631.0"`` and would
      never match a plain ``"8631"`` from another file.
    - Preserve missing IDs as NA (not the literal string ``"nan"``).
    - Preserve leading zeros (e.g. ``00123`` stays ``00123``).

    Because we operate on the values as-read by pandas, this does not by itself
    re-create leading zeros that pandas already dropped. To guarantee IDs such
    as ``00123`` survive, all files are read with ``dtype=str`` (see
    :func:`infraslow.io.csv_tools.read_csv_strict`), so the values arrive here as strings
    untouched.
    """
    # Identify missing values before string conversion so we can restore them.
    missing_mask = series.isna()
    normalized = series.astype(str).str.strip()
    # Collapse pure-integer floats like "8631.0" -> "8631". Only matches all-digit
    # values with a zero fractional part, so GUIDs and zero-padded IDs are untouched.
    normalized = normalized.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    normalized = normalized.mask(missing_mask, other=pd.NA)
    # Treat pandas' string spellings of missing values as missing, too.
    normalized = normalized.mask(
        normalized.isin({"nan", "NaN", "None", "<NA>", ""}), other=pd.NA
    )
    return normalized


def ensure_output_dir(output_file: str) -> None:
    """Ensure the output file's parent directory exists, creating it if needed."""
    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
            print(f"Created output directory: {out_dir}")
        except OSError as exc:
            fail(f"Could not create output directory '{out_dir}': {exc}")
