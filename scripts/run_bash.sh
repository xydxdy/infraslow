#!/bin/bash

set -euo pipefail

mkdir -p /scratch/users/chaisaen/logs

# --- Parameters (edit values here) ---
DATA_DIR=$SCRATCH/processed_data/data
CHANNEL=C3
N_SUBJECTS=none                        # none = every eligible subject found
CANDIDATE_LIMIT=none                   # none = scan the whole cohort, no cap
WORKERS=8                              # match --cpus-per-task above
OUTPUT_DIR=$SCRATCH/outputs/n2_vs_n3_compare_v3

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src/scripts
export PYTHONPATH=/home/users/chaisaen/infraslow/src

ARGS=(--data-dir "$DATA_DIR" --channel "$CHANNEL" --output-dir "$OUTPUT_DIR" --workers "$WORKERS" --verbose)
[[ "$N_SUBJECTS" != "none" ]] && ARGS+=(--n-subjects "$N_SUBJECTS")
[[ "$CANDIDATE_LIMIT" != "none" ]] && ARGS+=(--candidate-limit "$CANDIDATE_LIMIT")

python3 plot_n2_vs_n3_compare.py "${ARGS[@]}"