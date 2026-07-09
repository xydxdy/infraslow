#!/usr/bin/env python3
"""Run the full CSV processing pipeline.

This is the main entry point for processing. Instead of driving the merge from
the command line, configuration is set here as plain values and
``merge_csv_columns`` is called directly. The merged DataFrame is returned
in-memory so additional processing steps can run on it afterwards.

Run with:
    python -m infraslow.pipeline      # or the ``infraslow-process`` console script
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from infraslow.io import filter_groups_by_available_files
from infraslow.processing.merge import merge_csv_columns
from infraslow.processing.sleep_params import (
    compute_ahi,
    selected_groups_by_ahi,
    selected_groups_by_se,
)
from infraslow.utils import normalize_ids

# --------------------------------------------------------------------------- #
# Configuration  (edit these instead of passing CLI flags)
# --------------------------------------------------------------------------- #
DATA_DIR = Path(os.path.expandvars("$OAK/psg/Bioserenity/Excel/"))
OUTPUT_DIR = Path("output")

STAGES_FILE = DATA_DIR / "Stages_PSG_all.csv"
METADATA_FILE = DATA_DIR / "bioserenity_metadata3.csv"
APNEA_FILE = DATA_DIR / "N_Apnea_Types.csv"

MERGED_OUTPUT = OUTPUT_DIR / "merged_output.csv"
PROCESSED_OUTPUT = OUTPUT_DIR / "processed_output.csv"

# Group-selection outputs. The two metrics run as a chain (AHI first, then SE on
# the AHI-selected subjects), so the SE step produces the final saved result.
AHI_GROUPS_OUTPUT = OUTPUT_DIR / "selected_groups_ahi.csv"
AHI_EXCLUDED_OUTPUT = OUTPUT_DIR / "excluded_intermediate_ahi.csv"
FINAL_GROUPS_OUTPUT = OUTPUT_DIR / "selected_groups_final.csv"
FINAL_EXCLUDED_OUTPUT = OUTPUT_DIR / "excluded_intermediate_final.csv"

# After group selection, keep only subjects whose raw recordings exist on disk
# (an EDF under $OAK/psg/Bioserenity/edf AND a Hypnodensity.csv under
# $OAK/psg/Bioserenity/Sleep_Staging). Subjects missing either file are dropped.
AVAILABLE_GROUPS_OUTPUT = OUTPUT_DIR / "selected_groups_available.csv"
UNAVAILABLE_GROUPS_OUTPUT = OUTPUT_DIR / "excluded_missing_files.csv"

# Columns to keep from each file (the ID column is added automatically).
METADATA_COLUMNS = ["ID", "Age", "Gender", "BMI"]
APNEA_COLUMNS = ["ID", "Numbers_OA_Total", "Numbers_HI_Total"]
STAGES_COLUMNS = None  # None keeps ALL columns from the stages file.

ID_COLUMN = "ID"
JOIN_TYPE = "left"
DUPLICATE_POLICY = "last"

# Two-threshold group selection (configurable research-selection criteria, not
# universal medical thresholds). See md/Separated_Apnea_Groups.md.
GROUP_DUPLICATE_POLICY = "error"

# AHI grouping: higher AHI = more severe apnea (control = low, focus = high).
APNEA_COLUMN = "AHI"
AHI_CONTROL_MAX = 5.0
AHI_FOCUS_MIN = 30.0
AHI_MINIMUM_SEPARATION_GAP = 20.0

# SE grouping: higher sleep efficiency = healthier sleep, so the direction is
# inverted (control = high SE, focus = low SE).
SE_COLUMN = "SE"
SE_FOCUS_MAX = 80.0
SE_CONTROL_MIN = 90.0
SE_MINIMUM_SEPARATION_GAP = 10.0


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def run_merge() -> pd.DataFrame:
    """Step 1 — merge the stages, metadata, and apnea CSVs into one DataFrame."""
    return merge_csv_columns(
        stages_file=str(STAGES_FILE),
        metadata_file=str(METADATA_FILE),
        metadata_columns=METADATA_COLUMNS,
        apnea_file=str(APNEA_FILE),
        apnea_columns=APNEA_COLUMNS,
        output_file=str(MERGED_OUTPUT),
        stages_columns=STAGES_COLUMNS,
        id_column=ID_COLUMN,
        join_type=JOIN_TYPE,
        duplicate_policy=DUPLICATE_POLICY,
    )


def process(df: pd.DataFrame) -> dict:
    """Step 2 — post-merge processing, with the two groupings run as a chain.

    The processed DataFrame is kept in memory and threaded through each step —
    no CSV is written or re-read between steps. Only the final processed frame is
    saved, at the end.

    Steps:
        1. Run the existing processing logic (compute AHI).
        2. Group by apnea severity (AHI) -> select control + focus subjects.
        3. Keep only those AHI-selected subjects and group them by sleep
           efficiency (SE). The SE step therefore runs *on the AHI result*, not
           independently, and produces the final result.
        4. Drop subjects whose raw recordings (EDF + Hypnodensity.csv) are not
           present on disk, so only processable subjects remain.
        5. Save the processed DataFrame once, at the last.

    Returns a backward-compatible dict holding the processed DataFrame, the
    intermediate AHI groups, the final (SE-on-AHI-selected) groups restricted to
    subjects with files on disk (``selected_groups``), and the unrestricted final
    groups (``selected_groups_all``).
    """
    # Compute AHI = (Numbers_OA_Total + Numbers_HI_Total) / (TST hours).
    df = compute_ahi(df)

    # Step 2 — two-threshold grouping by apnea severity (AHI), on the in-memory df.
    groups_ahi: Dict[str, List[str]] = selected_groups_by_ahi(
        df,
        id_column=ID_COLUMN,
        apnea_column=APNEA_COLUMN,
        control_max=AHI_CONTROL_MAX,
        focus_min=AHI_FOCUS_MIN,
        minimum_separation_gap=AHI_MINIMUM_SEPARATION_GAP,
        duplicate_policy=GROUP_DUPLICATE_POLICY,
        output_csv=str(AHI_GROUPS_OUTPUT),
        excluded_output_csv=str(AHI_EXCLUDED_OUTPUT),
    )

    # Keep only the subjects the AHI step selected (control + focus) as an
    # in-memory subset, then feed that subset into the SE step so the two run in
    # order, not separately. No intermediate CSV is written.
    ahi_selected_ids = set(groups_ahi["control"]) | set(groups_ahi["focus"])
    ids = normalize_ids(df[ID_COLUMN])
    df_ahi_selected = df[ids.isin(ahi_selected_ids)]

    # Step 3 — two-threshold grouping by sleep efficiency (SE) on the AHI subset.
    # This is the final result.
    groups_final: Dict[str, List[str]] = selected_groups_by_se(
        df_ahi_selected,
        id_column=ID_COLUMN,
        se_column=SE_COLUMN,
        focus_max=SE_FOCUS_MAX,
        control_min=SE_CONTROL_MIN,
        minimum_separation_gap=SE_MINIMUM_SEPARATION_GAP,
        duplicate_policy=GROUP_DUPLICATE_POLICY,
        output_csv=str(FINAL_GROUPS_OUTPUT),
        excluded_output_csv=str(FINAL_EXCLUDED_OUTPUT),
    )

    # Step 4 — keep only subjects whose raw recordings actually exist on disk.
    # Some selected IDs have no EDF under $OAK/psg/Bioserenity/edf and/or no
    # Hypnodensity.csv under $OAK/psg/Bioserenity/Sleep_Staging; drop them so the
    # final groups contain only subjects that can be processed downstream.
    groups_available: Dict[str, List[str]] = filter_groups_by_available_files(
        groups_final,
        output_csv=str(AVAILABLE_GROUPS_OUTPUT),
        dropped_output_csv=str(UNAVAILABLE_GROUPS_OUTPUT),
    )

    # Step 5 — save the processed DataFrame once, at the last.
    df.to_csv(PROCESSED_OUTPUT, index=False)
    print(f"\nProcessed output written to : {PROCESSED_OUTPUT.resolve()}")
    print(f"Final selected groups written to : {FINAL_GROUPS_OUTPUT.resolve()}")
    print(f"Available (with files) groups written to : {AVAILABLE_GROUPS_OUTPUT.resolve()}")

    return {
        "data": df,
        "selected_groups": groups_available,
        "selected_groups_all": groups_final,
        "groups_ahi": groups_ahi,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> dict:
    """Run the merge, then the post-merge processing and group selection."""
    df = run_merge()
    result = process(df)
    return result


if __name__ == "__main__":
    main()
