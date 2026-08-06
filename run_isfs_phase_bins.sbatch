#!/bin/bash
#SBATCH --job-name=isfs_phase_bins
#SBATCH --partition=mignot
#SBATCH --output=/scratch/users/chaisaen/logs/isfs_phase_bins_%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Runs plot_isfs_phase_bins.py for one subject/channel: every spindle-containing
# N2 bout >= MIN_BOUT_SEC, each paginated into WINDOW_SEC-second pages, saved as one
# multi-page PDF plus a progress.log to $SCRATCH/infraslow_outputs/isfs_phase_bins/.
#
# All 5 knobs below are sent to the script explicitly as CLI flags -- edit the
# values directly in this file to change them.
# Run `python3 plot_isfs_phase_bins.py --help` for what each flag does.

set -euo pipefail

mkdir -p /scratch/users/chaisaen/logs

# --- Parameters (edit values here) ---
SUBJECT=318679
CHANNEL=C3
MIN_BOUT_SEC=200
WINDOW_SEC=400
OUTPUT_DIR=$SCRATCH/infraslow_outputs/isfs_phase_bins

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src/scripts
export PYTHONPATH=/home/users/chaisaen/infraslow/src

python3 plot_isfs_phase_bins.py \
    --subject "$SUBJECT" \
    --channel "$CHANNEL" \
    --min-bout-sec "$MIN_BOUT_SEC" \
    --window-sec "$WINDOW_SEC" \
    --output-dir "$OUTPUT_DIR"
