#!/bin/bash
#SBATCH --job-name=sleep_statistics
#SBATCH --partition=normal
#SBATCH --output=/scratch/users/chaisaen/logs/sleep_statistics_%j.out
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G

# Runs `src/scripts/preprocessing.py sleep-stats` over the same valid cohort
# as run_preprocessing.sbatch (every subject with both an EDF and a U-Sleep
# hypnodensity folder, via preprocessing.py's list_valid_subjects -- see the
# sleep-stats section of preprocessing.py's module docstring), computing YASA
# sleep statistics straight from each subject's --channel (default C3)
# U-Sleep hypnodensity file -- no EDF load needed -- and saving one row per
# subject to ${OUTPUT}.
#
# Single task, no job array: this only reads small per-subject U-Sleep .npy
# files (not full EEG recordings), so it's far cheaper than
# run_preprocessing.sbatch's per-channel EEG work.
#
# Override at submit time, e.g.:
#   sbatch --export=ALL,OUTPUT=/scratch/users/chaisaen/processed_data/sleep_statistics.csv run_sleep_statistics.sbatch
#   sbatch --export=ALL,LIMIT=50 run_sleep_statistics.sbatch   # quick test: 50 subjects
#   sbatch --export=ALL,NUM_SHARDS=1000,SHARD_INDICES=0-9,LIMIT=50,HYPNO_EPOCH_SEC=3,OUTPUT=/scratch/users/chaisaen/usleep_dryrun/sleep_statistics.csv \
#       run_sleep_statistics.sbatch   # dry run: same 500 subjects a
#       # `run_preprocessing.sbatch --array=0-9 --export=ALL,NUM_SHARDS=1000,LIMIT=50,...`
#       # job array would touch (see preprocessing.py sleep-stats --num-shards/--shard-indices)

set -euo pipefail

OUTPUT="${OUTPUT:-/scratch/users/chaisaen/processed_data/sleep_statistics.csv}"
LIMIT="${LIMIT:-}"
HYPNO_EPOCH_SEC="${HYPNO_EPOCH_SEC:-}"
NUM_SHARDS="${NUM_SHARDS:-}"
SHARD_INDICES="${SHARD_INDICES:-}"

mkdir -p /scratch/users/chaisaen/logs

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src/scripts
export PYTHONPATH=/home/users/chaisaen/infraslow/src

python3 preprocessing.py sleep-stats \
    --output "${OUTPUT}" \
    ${LIMIT:+--limit "${LIMIT}"} \
    ${HYPNO_EPOCH_SEC:+--hypno-epoch-sec "${HYPNO_EPOCH_SEC}"} \
    ${NUM_SHARDS:+--num-shards "${NUM_SHARDS}"} \
    ${SHARD_INDICES:+--shard-indices "${SHARD_INDICES}"}
