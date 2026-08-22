#!/bin/bash
#SBATCH --job-name=hypnodensity_isfs_phase_transition
#SBATCH --partition=normal
#SBATCH --output=/scratch/users/chaisaen/logs/hypnodensity_isfs_phase_transition_%j.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB

# Runs src/scripts/plot_hypnodensity_isfs_phase_transition.py: same group-level
# U-Sleep stage probability / event phase-bin analysis as
# run_hypnodensity_isfs_phase.sbatch, but restricted to each N2/N3 bout's last
# TRANSITION_WINDOW_SEC seconds before a real, sustained transition (the new
# stage must persist >= TRANSITION_PERSIST_SEC without reverting) to a
# different scored sleep stage (see plot_hypnodensity_isfs_phase_transition.py's
# module docstring and
# docs/superpowers/specs/2026-08-22-hypnodensity-isfs-phase-transition-design.md).
# --mem is 40GB, not the original job's 32GB -- that job's two most recent full
# runs both hit exactly 100% of their 32GB request (no headroom to copy as-is).
#
# All knobs below are sent to the script explicitly as CLI flags -- edit the
# values directly in this file to change them, or override at submit time, e.g.:
#   sbatch --export=ALL,N_SUBJECTS=200 run_hypnodensity_isfs_phase_transition.sbatch
#   sbatch --export=ALL,LIMIT=500 run_hypnodensity_isfs_phase_transition.sbatch   # quick test
#
# Namespaced by --event ("spindle"/"sw") the same way the original job is, so a
# spindle run and a sw run against the same OUTPUT_DIR never collide:
#   sbatch --job-name=hypnodensity_isfs_phase_transition_spindle --export=ALL,EVENT=spindle run_hypnodensity_isfs_phase_transition.sbatch
#   sbatch --job-name=hypnodensity_isfs_phase_transition_sw      --export=ALL,EVENT=sw      run_hypnodensity_isfs_phase_transition.sbatch
#
# Run `python3 plot_hypnodensity_isfs_phase_transition.py --help` for what each flag does.

set -euo pipefail

mkdir -p /scratch/users/chaisaen/logs

# --- Parameters (edit values here, or override via --export=ALL,VAR=value) ---
DATA_DIR="${DATA_DIR:-$SCRATCH/processed_data/data}"
USLEEP_DIR="${USLEEP_DIR:-$OAK/AISleepScientist/data/bioserenity/usleep_hypnodensities}"
HYPNO_DIR="${HYPNO_DIR:-$OAK/psg/Bioserenity/Sleep_Staging}"
CHANNEL="${CHANNEL:-C3}"
STATES="${STATES:-N2 N3}"
EVENT="${EVENT:-sw}"
TRANSITION_WINDOW_SEC="${TRANSITION_WINDOW_SEC:-200}"
TRANSITION_PERSIST_SEC="${TRANSITION_PERSIST_SEC:-60}"
N_PHASE_POINTS="${N_PHASE_POINTS:-50}"
N_SUBJECTS="${N_SUBJECTS:-0}"
LIMIT="${LIMIT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/outputs/hypnodensity_isfs_phase_transition}"

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src/scripts
export PYTHONPATH=/home/users/chaisaen/infraslow/src

python3 plot_hypnodensity_isfs_phase_transition.py \
    --data-dir "$DATA_DIR" \
    --usleep-dir "$USLEEP_DIR" \
    --hypno-dir "$HYPNO_DIR" \
    --channel "$CHANNEL" \
    --states $STATES \
    --event "$EVENT" \
    --window-sec "$TRANSITION_WINDOW_SEC" \
    --persist-sec "$TRANSITION_PERSIST_SEC" \
    --n-phase-points "$N_PHASE_POINTS" \
    --n-subjects "$N_SUBJECTS" \
    --workers "${SLURM_CPUS_PER_TASK:-1}" \
    --output-dir "$OUTPUT_DIR" \
    ${LIMIT:+--limit "$LIMIT"}
