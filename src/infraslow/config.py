"""Default paths for locating Bioserenity pipeline inputs.

Metadata CSVs are small and checked into the repo (``infraslow/metadata/``),
so they no longer depend on ``$OAK`` being mounted. EDF/hypnodensity data is
far too large to check in and stays on ``$OAK``.
"""

from pathlib import Path

METADATA_DIR = Path(__file__).resolve().parent / "metadata"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_METADATA = str(METADATA_DIR / "Morpheus_Data_All5.csv")
DEFAULT_METADATA2 = str(METADATA_DIR / "bioserenity_metadata3.csv")
DEFAULT_DRUG_METADATA = str(METADATA_DIR / "drug_usage_metadata.csv")
DEFAULT_EDF_DIR = "$OAK/psg/Bioserenity/edf"
DEFAULT_HYPNO_DIR = "$OAK/psg/Bioserenity/Sleep_Staging"
DEFAULT_DRUG_EXCLUDE_LIST = str(REPO_ROOT / "drug" / "drug_exclude.csv")
