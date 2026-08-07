#!/bin/bash
#SBATCH --job-name=sleep_statistics
#SBATCH --partition=mignot
#SBATCH --output=/scratch/users/chaisaen/logs/sleep_statistics_%j.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G

# Runs src/scripts/sleep_statistics.py over every subject already
# preprocessed under ${DATA_DIR} (its subdirectory names are the subject
# list), computing YASA sleep statistics straight from each subject's
# Hypnodensity CSV -- no EDF load needed -- and saving one row per subject to
# ${OUTPUT}.
#
# Single task, no job array: this only reads small per-subject Hypnodensity
# CSVs (not full EEG recordings), so it's far cheaper than
# run_preprocessing.sbatch's per-channel EEG work.
#
# Override at submit time, e.g.:
#   sbatch --export=ALL,DATA_DIR=/scratch/users/chaisaen/processed_data/data,\
#OUTPUT=/scratch/users/chaisaen/processed_data/sleep_statistics.csv \
#       run_sleep_statistics.sbatch
#   sbatch --export=ALL,LIMIT=50 run_sleep_statistics.sbatch   # quick test: 50 subjects

set -euo pipefail

DATA_DIR="${DATA_DIR:-/scratch/users/chaisaen/processed_data/data}"
OUTPUT="${OUTPUT:-/scratch/users/chaisaen/processed_data/sleep_statistics.csv}"
LIMIT="${LIMIT:-}"

mkdir -p /scratch/users/chaisaen/logs

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src/scripts
export PYTHONPATH=/home/users/chaisaen/infraslow/src

python3 sleep_statistics.py \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT}" \
    ${LIMIT:+--limit "${LIMIT}"}
