#!/bin/bash
#SBATCH --job-name=group_analysis_demographics
#SBATCH --partition=mignot
#SBATCH --output=/scratch/users/chaisaen/logs/group_analysis_demographics_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

# Runs src/group_analysis_demographics.py (see its module docstring): loads
# per-subject {STAGE}/{SUBJECT}.npz results, restricts to one sleep stage +
# one EEG channel, validates, then -- unlike run_group_analysis.sbatch, which
# filters to one demographic band and compares low/high spindle-rate groups
# *inside* it -- compares Male vs Female, Age low/mid/high, and BMI
# low/mid/high directly against each other on the same infraslow summary
# parameters. Outputs go to
# $SCRATCH/infraslow_outputs/group_analysis_demographics/{STAGE}_{CHANNEL}/ by
# default.
#
# Per-subject loading also fits a bi-Gaussian curve per subject (CPU-bound),
# so group_analysis_demographics.py loads across a process pool sized to
# $SLURM_CPUS_PER_TASK by default -- --cpus-per-task above is what actually
# controls load speed; raise it (and MEM roughly in step, ~2G/core) for a
# bigger/faster run.
#
# All knobs below are sent to the script explicitly as CLI flags -- edit the
# values directly in this file to change them, or override at submit time,
# e.g.:
#   sbatch --export=ALL,STAGE=N3,CHANNEL=F3 run_group_analysis_demographics.sbatch
#   sbatch --export=ALL,N_SUBJECTS=500 run_group_analysis_demographics.sbatch     # quick test
#   sbatch --cpus-per-task=32 --mem=64G --export=ALL,WORKERS=32 run_group_analysis_demographics.sbatch
#   sbatch --export=ALL,OUTPUT_DIR=$SCRATCH/.../001,DOMINANT_FREQ_HZ=0.01 \
#       run_group_analysis_demographics.sbatch   # one dominant-infraslow-peak group
# Run `python3 src/group_analysis_demographics.py --help` for what each flag does.
#
# Logs go to $SCRATCH (not $HOME/logs) since this can be resubmitted often
# while iterating and $HOME's 15 GB quota is easy to fill with accumulated
# run logs.

set -euo pipefail

mkdir -p /scratch/users/chaisaen/logs

# --- Parameters (edit values here, or override via --export at submit time) ---
STAGE="${STAGE:-N2}"                       # sleep stage: N1 | N2 | N3 | NREM
CHANNEL="${CHANNEL:-C3}"                   # EEG channel: F3 | F4 | C3 | C4 | O1 | O2
N_SUBJECTS="${N_SUBJECTS:-none}"           # none = every subject npz found for STAGE
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH/data/npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/infraslow_outputs/group_analysis_demographics_v2/${STAGE}_${CHANNEL}}"
RATE_UNIT="${RATE_UNIT:-hr}"                # min | hr -- spindle-rate column unit (also compared here)
FDR_ALPHA="${FDR_ALPHA:-0.05}"
OVERWRITE="${OVERWRITE:-true}"
WORKERS="${WORKERS:-none}"                 # none = defaults to $SLURM_CPUS_PER_TASK
DOMINANT_FREQ_HZ="${DOMINANT_FREQ_HZ:-none}"  # none = no dominant_freq_hz filter (whole cohort)
DRUG_EXCLUDE="${DRUG_EXCLUDE:-none}"       # none = no drug filter; "All" = every drug in
                                            # drug/drug_exclude.csv; else a space-separated list
                                            # of (single-word) drug names, e.g. "Z_Drugs Benzodiazepine"

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow

ARGS=(
    --results "$RESULTS_DIR"
    --sleep-stage "$STAGE"
    --channel "$CHANNEL"
    --output-dir "$OUTPUT_DIR"
    --fdr-alpha "$FDR_ALPHA"
    --rate-unit "$RATE_UNIT"
)
[[ "$N_SUBJECTS" != "none" ]] && ARGS+=(--n-subjects "$N_SUBJECTS")
[[ "$WORKERS" != "none" ]] && ARGS+=(--workers "$WORKERS")
[[ "$OVERWRITE" == "true" ]] && ARGS+=(--overwrite)
[[ "$DOMINANT_FREQ_HZ" != "none" ]] && ARGS+=(--dominant-freq-hz "$DOMINANT_FREQ_HZ")
if [[ "$DRUG_EXCLUDE" != "none" ]]; then
    if [[ "$DRUG_EXCLUDE" == "All" ]]; then
        ARGS+=(--exclude-drug-users All)
    else
        # Intentional word-splitting: DRUG_EXCLUDE is a space-separated list of
        # (single-word) drug names, e.g. "Z_Drugs Benzodiazepine".
        read -ra DRUG_NAMES <<< "$DRUG_EXCLUDE"
        ARGS+=(--exclude-drug-users "${DRUG_NAMES[@]}")
    fi
fi

python3 src/group_analysis_demographics.py "${ARGS[@]}"
