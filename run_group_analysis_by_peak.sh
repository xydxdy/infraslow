#!/bin/bash
# Submits 6 independent run_group_analysis.sbatch jobs, partitioning the
# {STAGE}-{CHANNEL} cohort by each subject's own exact dominant infraslow
# peak (dominant_freq_hz -- the frequency bin, on the fixed Welch grid, with
# the highest baseline-corrected relative power in that subject's spectrum;
# see group_analysis.py Step 2) before running the usual low/high
# spindle-rate grouping + comparison pipeline separately on each partition.
# Each submission is its own Slurm job, so the 6 runs execute in parallel
# rather than serially.
#
# Groups:
#   All -> no dominant_freq_hz filter (the full validated cohort)
#   001 -> dominant_freq_hz == 0.01 Hz
#   002 -> dominant_freq_hz == 0.02 Hz
#   003 -> dominant_freq_hz == 0.03 Hz
#   004 -> dominant_freq_hz == 0.04 Hz
#   005 -> dominant_freq_hz == 0.05 Hz
#
# Outputs land under
#   $SCRATCH/infraslow_outputs/group_analysis_before_remove_outlier/{STAGE}_{CHANNEL}/{group}/
#
# Usage: ./run_group_analysis_by_peak.sh [STAGE] [CHANNEL]
#   e.g.: ./run_group_analysis_by_peak.sh N2 C3

set -euo pipefail

STAGE="${1:-N2}"
CHANNEL="${2:-C3}"
BASE_OUTPUT_DIR="$SCRATCH/infraslow_outputs/group_analysis_before_remove_outlier/${STAGE}_${CHANNEL}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# label:dominant_freq_hz ("none" = no filter, i.e. the "All" group).
PEAK_GROUPS=(
    "All:none"
    "001:0.01"
    "002:0.02"
    "003:0.03"
    "004:0.04"
    "005:0.05"
)

for spec in "${PEAK_GROUPS[@]}"; do
    IFS=':' read -r label dominant_freq_hz <<< "$spec"
    echo "submitting group ${label} (dominant_freq_hz == ${dominant_freq_hz}) -> ${BASE_OUTPUT_DIR}/${label}"
    sbatch --export=ALL,STAGE="$STAGE",CHANNEL="$CHANNEL",OUTPUT_DIR="${BASE_OUTPUT_DIR}/${label}",DOMINANT_FREQ_HZ="$dominant_freq_hz" \
        run_group_analysis.sbatch
done
