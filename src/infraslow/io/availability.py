"""Keep only subjects whose raw recording files actually exist on disk.

Group selection (:mod:`infraslow.processing.sleep_params`) picks subjects purely
from the merged metadata CSVs. But a selected subject is only usable downstream
if its raw recordings are present: the **EDF** signal file *and* the
**hypnodensity** staging CSV. Some IDs in ``groups_final`` have no file under
``$OAK/psg/Bioserenity/edf`` or ``$OAK/psg/Bioserenity/Sleep_Staging``; this
module drops those so the returned groups contain only processable subjects.

Path convention (matches :class:`infraslow.io.psg_loader.BioserenityPSGLoader`
and :func:`infraslow.io.hypnodensity.make_hypnodensity_annotation_loader`)::

    EDF:          $OAK/psg/Bioserenity/edf/{id}.edf
    Hypnodensity: $OAK/psg/Bioserenity/Sleep_Staging/{id}_Hypnodensity.csv

The subject id is the EDF file stem, so it maps directly onto both filenames.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..utils import ensure_output_dir
from .hypnodensity import DEFAULT_HYPNODENSITY_SUFFIX, DEFAULT_STAGING_DIRNAME
from .psg_loader import DirectoryNotFoundError, EnvironmentVariableError
from .utils import list_dir_filenames, progress_iter

DEFAULT_EDF_SUBDIR = "psg/Bioserenity/edf"
DEFAULT_STAGING_SUBDIR = f"psg/Bioserenity/{DEFAULT_STAGING_DIRNAME}"
DEFAULT_EDF_SUFFIX = ".edf"


def _resolve_storage_root(oak_env_var: str) -> Path:
    """Resolve the storage root from an environment variable (e.g. ``$OAK``)."""
    raw = os.environ.get(oak_env_var)
    if not raw:
        raise EnvironmentVariableError(
            f"Environment variable ${oak_env_var} is not set."
        )
    root = Path(raw)
    if not root.is_dir():
        raise DirectoryNotFoundError(
            f"${oak_env_var} does not point at a directory: {root}"
        )
    return root


def _missing_reason(
    subject_id: str,
    *,
    edf_names: set,
    hypno_names: set,
    edf_suffix: str,
    hypnodensity_suffix: str,
) -> Optional[str]:
    """Return why a subject is unusable, or ``None`` if both files exist.

    Membership is tested against pre-listed directory contents (see
    :func:`list_dir_filenames`). Reasons are ``"missing_edf"``,
    ``"missing_hypnodensity"``, or ``"missing_edf_and_hypnodensity"``.
    """
    has_edf = f"{subject_id}{edf_suffix}" in edf_names
    has_hypno = f"{subject_id}{hypnodensity_suffix}" in hypno_names
    if has_edf and has_hypno:
        return None
    if not has_edf and not has_hypno:
        return "missing_edf_and_hypnodensity"
    return "missing_edf" if not has_edf else "missing_hypnodensity"


def filter_groups_by_available_files(
    groups: Dict[str, List[str]],
    *,
    oak_env_var: str = "OAK",
    edf_subdir: str = DEFAULT_EDF_SUBDIR,
    staging_subdir: str = DEFAULT_STAGING_SUBDIR,
    edf_suffix: str = DEFAULT_EDF_SUFFIX,
    hypnodensity_suffix: str = DEFAULT_HYPNODENSITY_SUFFIX,
    output_csv: Optional[str] = None,
    dropped_output_csv: Optional[str] = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """Drop subjects that lack an EDF *and/or* hypnodensity file on disk.

    Each subject id is kept only when **both** files exist::

        $OAK/<edf_subdir>/{id}<edf_suffix>
        $OAK/<staging_subdir>/{id}<hypnodensity_suffix>

    Args:
        groups: Group mapping such as ``{"control": [...], "focus": [...]}``.
            Never mutated; a new dict with the same keys is returned.
        oak_env_var: Environment variable holding the storage root.
        edf_subdir: EDF directory relative to the storage root.
        staging_subdir: Hypnodensity (staging) directory relative to the root.
        edf_suffix: Filename suffix appended to the id for the EDF file.
        hypnodensity_suffix: Filename suffix appended for the hypnodensity CSV.
        output_csv: If given, write the kept ``subject_id,group`` rows here.
        dropped_output_csv: If given, write dropped subjects with the column
            ``subject_id,group,exclusion_reason``.
        verbose: Print a per-group kept/dropped summary.
        show_progress: Show a percentage progress bar (``tqdm`` if installed,
            otherwise a plain stderr percentage) while checking subjects, and log
            the directory-listing steps.

    Returns:
        A new dict, same keys as ``groups``, each list filtered to ids whose EDF
        and hypnodensity files both exist (order preserved).
    """
    root = _resolve_storage_root(oak_env_var)
    edf_dir = root / edf_subdir
    staging_dir = root / staging_subdir
    for label, path in (("EDF", edf_dir), ("Sleep_Staging", staging_dir)):
        if not path.is_dir():
            raise DirectoryNotFoundError(f"{label} directory does not exist: {path}")

    # List both directories once up front, then test membership in memory. This
    # avoids a per-subject stat storm against Lustre (see list_dir_filenames).
    # The listing is the slow part on $OAK, so announce it before each scan.
    if show_progress:
        print(f"Listing EDF directory {edf_dir} ...", file=sys.stderr, flush=True)
    edf_names = list_dir_filenames(edf_dir)
    if show_progress:
        print(f"  -> {len(edf_names)} entries", file=sys.stderr, flush=True)
        print(
            f"Listing Sleep_Staging directory {staging_dir} ...",
            file=sys.stderr,
            flush=True,
        )
    hypno_names = list_dir_filenames(staging_dir)
    if show_progress:
        print(f"  -> {len(hypno_names)} entries", file=sys.stderr, flush=True)

    total = sum(len(ids) for ids in groups.values())
    pairs = ((group, sid) for group, ids in groups.items() for sid in ids)

    # Preserve group order/keys even if a group ends up empty.
    kept: Dict[str, List[str]] = {group: [] for group in groups}
    dropped_rows: List[Dict[str, str]] = []
    for group, subject_id in progress_iter(
        pairs, total, enabled=show_progress, desc="Checking files"
    ):
        reason = _missing_reason(
            subject_id,
            edf_names=edf_names,
            hypno_names=hypno_names,
            edf_suffix=edf_suffix,
            hypnodensity_suffix=hypnodensity_suffix,
        )
        if reason is None:
            kept[group].append(subject_id)
        else:
            dropped_rows.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "exclusion_reason": reason,
                }
            )

    if output_csv:
        ensure_output_dir(output_csv)
        rows = [
            {"subject_id": sid, "group": group}
            for group, ids in kept.items()
            for sid in ids
        ]
        pd.DataFrame(rows, columns=["subject_id", "group"]).to_csv(
            output_csv, index=False
        )

    if dropped_output_csv:
        ensure_output_dir(dropped_output_csv)
        pd.DataFrame(
            dropped_rows, columns=["subject_id", "group", "exclusion_reason"]
        ).to_csv(dropped_output_csv, index=False)

    if verbose:
        _print_availability_summary(groups, kept, dropped_rows, edf_dir, staging_dir)

    return kept


def _print_availability_summary(
    groups: Dict[str, List[str]],
    kept: Dict[str, List[str]],
    dropped_rows: List[Dict[str, str]],
    edf_dir: Path,
    staging_dir: Path,
) -> None:
    """Print a human-friendly kept/dropped summary."""
    print("\n" + "=" * 60)
    print("FILE-AVAILABILITY FILTER (EDF + hypnodensity)")
    print("=" * 60)
    print(f"EDF directory           : {edf_dir}")
    print(f"Sleep_Staging directory : {staging_dir}")
    print("-" * 60)
    for group, ids in groups.items():
        n_in = len(ids)
        n_kept = len(kept.get(group, []))
        print(f"{group:<10} : kept {n_kept}/{n_in} (dropped {n_in - n_kept})")
    if dropped_rows:
        by_reason: Dict[str, int] = {}
        for row in dropped_rows:
            by_reason[row["exclusion_reason"]] = (
                by_reason.get(row["exclusion_reason"], 0) + 1
            )
        print("-" * 60)
        for reason, count in sorted(by_reason.items()):
            print(f"Dropped - {reason:<28} : {count}")
    print("=" * 60)
