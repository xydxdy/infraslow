#!/bin/bash
# Submits one independent run_group_analysis.sbatch job per (peak group x
# drug-exclusion group) combination below: partitions the {STAGE}-{CHANNEL}
# cohort by each subject's own exact dominant infraslow peak (dominant_freq_hz
# -- the frequency bin, on the fixed Welch grid, with the highest
# baseline-corrected relative power in that subject's spectrum; see
# group_analysis.py Step 2) and/or by drug usage (--exclude-drug-users; see
# group_analysis.py) before running the usual low/high spindle-rate grouping +
# comparison pipeline separately on each combination. Each submission is its
# own Slurm job, so every combination runs in parallel rather than serially.
#
# Peak groups:
#   All -> no dominant_freq_hz filter (the full validated cohort)
#   001 -> dominant_freq_hz == 0.01 Hz
#   002 -> dominant_freq_hz == 0.02 Hz
#   003 -> dominant_freq_hz == 0.03 Hz
#   004 -> dominant_freq_hz == 0.04 Hz
#   005 -> dominant_freq_hz == 0.05 Hz
#
# Drug-exclusion groups:
#   z_benzo   -> exclude subjects flagged for Z_Drugs or Benzodiazepine
#   all_drugs -> exclude subjects flagged for any drug in drug/drug_exclude.csv
#
# Outputs land under
#   $SCRATCH/infraslow_outputs/group_analysis_/{STAGE}_{CHANNEL}/{peak_label}/{drug_label}/
#
# Usage: ./run_group_analysis_by_peak.sh [STAGE] [CHANNEL]
#   e.g.: ./run_group_analysis_by_peak.sh N2 C3

set -euo pipefail

STAGE="${1:-N2}"
CHANNEL="${2:-C3}"
BASE_OUTPUT_DIR="$SCRATCH/infraslow_outputs/group_analysis_/${STAGE}_${CHANNEL}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# label:dominant_freq_hz ("none" = no filter, i.e. the "All" group).
PEAK_GROUPS=(
    # "All:none"
    # "001:0.01"
    "002:0.02"
    # "003:0.03"
    # "004:0.04"
    # "005:0.05"
)

# label:drug-exclude-arg ("All" = every drug in drug/drug_exclude.csv;
# otherwise a space-separated list of (single-word) drug names -- this is
# passed straight through to run_group_analysis.sbatch's DRUG_EXCLUDE, which
# word-splits it, so only single-word drug names work here).
DRUG_GROUPS=(
    "z_benzo:Z_Drugs Benzodiazepine"
    "all_drugs:All"
)

for peak_spec in "${PEAK_GROUPS[@]}"; do
    IFS=':' read -r peak_label dominant_freq_hz <<< "$peak_spec"
    for drug_spec in "${DRUG_GROUPS[@]}"; do
        IFS=':' read -r drug_label drug_exclude <<< "$drug_spec"
        out_dir="${BASE_OUTPUT_DIR}/${peak_label}/${drug_label}"
        echo "submitting group ${peak_label} (dominant_freq_hz == ${dominant_freq_hz}), drug exclude ${drug_label} (${drug_exclude}) -> ${out_dir}"
        sbatch --export=ALL,STAGE="$STAGE",CHANNEL="$CHANNEL",OUTPUT_DIR="$out_dir",DOMINANT_FREQ_HZ="$dominant_freq_hz",DRUG_EXCLUDE="$drug_exclude" \
            run_group_analysis.sbatch
    done
done
