#!/bin/bash
# Submits one independent run_group_analysis.sbatch job per (peak group x
# demographic group) combination below: partitions the {STAGE}-{CHANNEL}
# cohort by each subject's own exact dominant infraslow peak (dominant_freq_hz;
# see group_analysis.py Step 2) and by one demographic variable at a time
# (--sex / --age-group / --bmi-group; see group_analysis.py) before running
# the usual low/high spindle-rate grouping + comparison pipeline separately on
# each combination. Each submission is its own Slurm job, so every
# combination runs in parallel rather than serially.
#
# Peak groups:
#   All -> no dominant_freq_hz filter (the full validated cohort)
#   002 -> dominant_freq_hz == 0.02 Hz
#
# Demographic groups (each filters on exactly one variable; unfiltered on the
# other two):
#   sex_male / sex_female -> --sex Male / --sex Female
#   age_low / age_mid / age_high -> --age-group low / mid / high
#   bmi_low / bmi_mid / bmi_high -> --bmi-group low / mid / high
#
# Outputs land under
#   $SCRATCH/infraslow_outputs/group_analysis_/{STAGE}_{CHANNEL}/{peak_label}/{demo_label}/
#
# Usage: ./run_group_analysis_by_demographic.sh [STAGE] [CHANNEL]
#   e.g.: ./run_group_analysis_by_demographic.sh N2 C3

set -euo pipefail

STAGE="${1:-N2}"
CHANNEL="${2:-C3}"
BASE_OUTPUT_DIR="$SCRATCH/infraslow_outputs/group_analysis_/${STAGE}_${CHANNEL}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# label:dominant_freq_hz ("none" = no filter, i.e. the "All" group).
PEAK_GROUPS=(
    "All:none"
    "002:0.02"
)

# label:env_var:value -- env_var is one of SEX/AGE_GROUP/BMI_GROUP, forwarded
# to run_group_analysis.sbatch (see its --sex/--age-group/--bmi-group).
DEMOGRAPHIC_GROUPS=(
    "sex_male:SEX:Male"
    "sex_female:SEX:Female"
    "age_low:AGE_GROUP:low"
    "age_mid:AGE_GROUP:mid"
    "age_high:AGE_GROUP:high"
    "bmi_low:BMI_GROUP:low"
    "bmi_mid:BMI_GROUP:mid"
    "bmi_high:BMI_GROUP:high"
)

for peak_spec in "${PEAK_GROUPS[@]}"; do
    IFS=':' read -r peak_label dominant_freq_hz <<< "$peak_spec"
    for demo_spec in "${DEMOGRAPHIC_GROUPS[@]}"; do
        IFS=':' read -r demo_label demo_var demo_value <<< "$demo_spec"
        out_dir="${BASE_OUTPUT_DIR}/${peak_label}/${demo_label}"
        echo "submitting group ${peak_label} (dominant_freq_hz == ${dominant_freq_hz}), ${demo_var}=${demo_value} -> ${out_dir}"
        sbatch --export=ALL,STAGE="$STAGE",CHANNEL="$CHANNEL",OUTPUT_DIR="$out_dir",DOMINANT_FREQ_HZ="$dominant_freq_hz","${demo_var}=${demo_value}" \
            run_group_analysis.sbatch
    done
done
