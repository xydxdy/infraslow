import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List


DEFAULT_HYPNODENSITY_SUFFIX = "_Hypnodensity.csv"

METADATA_COLUMNS: List[str] = ["ID", "Age", "Gender", "BMI"]

# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def list_dir_filenames(directory: Path) -> set:
    """Return the set of entry names directly under ``directory`` (one readdir).

    Listing the directory once and testing membership in memory turns a bulk
    file-existence check from ``N`` per-file ``stat`` calls -- slow and hard on
    the Lustre metadata server for ``$OAK`` -- into a single ``readdir``.
    ``os.scandir`` is used so we never ``stat`` each entry.
    """
    with os.scandir(directory) as it:
        return {entry.name for entry in it}


def load_bioserenity_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load metadata and standardise columns to ``ID, Age, Gender, BMI``.

    Only the four needed columns are read (the CSV is wide and long).
    ``ID`` is read as a string (subject ids are brace-wrapped GUIDs); ``Age``
    and ``BMI`` are coerced to numeric with unparseable cells becoming
    ``NaN``.

    Args:
        metadata_path: Path to ``bioserenity_metadata3.csv``.

    Returns:
        DataFrame with exactly the columns ``["ID", "Age", "Gender", "BMI"]``,
        rows with a missing/blank ``ID`` dropped.

    Raises:
        FileNotFoundError: if ``metadata_path`` does not exist.
        KeyError: if the ``ID`` column is missing.
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_path}")

    # Read only the header first to check which columns are present.
    header = pd.read_csv(metadata_path, nrows=0)
    available = list(header.columns)
    if "ID" not in available:
        raise KeyError(
            f"Could not find an 'ID' column in {metadata_path.name}; "
            f"Columns: {available[:10]}..."
        )

    usecols = [c for c in METADATA_COLUMNS if c in available]
    out = pd.read_csv(metadata_path, usecols=usecols, dtype={"ID": str})
    for canonical in METADATA_COLUMNS:
        if canonical not in out.columns:
            out[canonical] = np.nan
    out = out[METADATA_COLUMNS].copy()

    out["ID"] = out["ID"].astype(str).str.strip()
    # Some source exports (e.g. Morpheus) write integer ids as floats ("8631.0");
    # the on-disk EDF/hypnodensity files use the bare integer ("8631.edf"), so
    # strip a spurious trailing ".0" before matching. GUID-style ids are untouched.
    out["ID"] = out["ID"].str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    out = out[out["ID"].notna() & (out["ID"] != "") & (out["ID"].str.lower() != "nan")]
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["BMI"] = pd.to_numeric(out["BMI"], errors="coerce")
    out["Gender"] = out["Gender"].astype(str).str.strip()
    return out.reset_index(drop=True)

def combine_bioserenity_metadata(*frames: pd.DataFrame) -> pd.DataFrame:
    """Union metadata rows from multiple sources, keyed by ``ID``.

    A subject is kept if its ``ID`` appears in *any* of ``frames`` (OR, not
    intersection). Where the same ``ID`` appears in more than one frame, the
    first non-null value per column wins.

    Args:
        *frames: Standardised metadata frames (see :func:`load_bioserenity_metadata`).

    Returns:
        DataFrame with columns ``["ID", "Age", "Gender", "BMI"]``, one row per
        distinct ``ID``.
    """
    combined = pd.concat(frames, ignore_index=True)
    coalesce_first = lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
    out = combined.groupby("ID", as_index=False).agg(
        {col: coalesce_first for col in METADATA_COLUMNS if col != "ID"}
    )
    return out[METADATA_COLUMNS].reset_index(drop=True)

def find_valid_bioserenity_subjects(
    metadata: pd.DataFrame,
    edf_dir: Path,
    hypnodensity_dir: Path,
    *,
    id_column: str = "ID",
    edf_suffix: str = ".edf",
    hypnodensity_suffix: str = DEFAULT_HYPNODENSITY_SUFFIX,
) -> pd.DataFrame:
    """Return metadata rows whose EDF *and* hypnodensity files both exist.

    Both directories are listed once (a single ``readdir`` each via
    :func:`~infraslow.io.utils.list_dir_filenames`) and membership is tested in
    memory -- this avoids a per-subject ``stat`` storm against the ``$OAK`` Lustre
    metadata server.

    The returned frame carries an ``availability`` summary dict in ``df.attrs``
    (counts of metadata / with-EDF / with-hypnodensity / valid subjects) for the
    notebook's processing summary.

    Args:
        metadata: Standardised metadata (see :func:`load_bioserenity_metadata`).
        edf_dir: Directory of ``{id}.edf`` files.
        hypnodensity_dir: Directory of ``{id}_Hypnodensity.csv`` files.
        id_column: Column of subject ids in ``metadata``.
        edf_suffix, hypnodensity_suffix: Filename suffixes appended to the id.

    Returns:
        A copy of the matching metadata rows (order preserved), reindexed.
    """
    edf_dir = Path(edf_dir)
    hypnodensity_dir = Path(hypnodensity_dir)
    edf_names = list_dir_filenames(edf_dir)
    hypno_names = list_dir_filenames(hypnodensity_dir)

    ids = metadata[id_column].astype(str)
    has_edf = ids.map(lambda s: f"{s}{edf_suffix}" in edf_names)
    has_hypno = ids.map(lambda s: f"{s}{hypnodensity_suffix}" in hypno_names)
    valid_mask = has_edf & has_hypno

    valid = metadata.loc[valid_mask].reset_index(drop=True).copy()
    valid.attrs["availability"] = {
        "n_metadata": int(len(metadata)),
        "n_with_edf": int(has_edf.sum()),
        "n_with_hypnodensity": int(has_hypno.sum()),
        "n_valid": int(valid_mask.sum()),
    }
    return valid


oak = Path(os.environ["OAK"])
metadata_path = oak / "psg/Bioserenity/Excel/Morpheus_Data_All5.csv"
metadata_path2 = oak / "psg/Bioserenity/Excel/bioserenity_metadata3.csv"
edf_dir = oak / "psg/Bioserenity/edf"
hypno_dir = oak / "psg/Bioserenity/Sleep_Staging"
HYPNO_SUFFIX =  "_Hypnodensity.csv"

# --- 1. Metadata + valid-subject discovery ---------------------------- #
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
print("subjects in metadata      :", n_metadata)
print("subjects with both files  :", n_valid)
