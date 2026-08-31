#!/bin/bash
#SBATCH --job-name=hypnodensity_isfs_phase
#SBATCH --partition=mignot
#SBATCH --output=/scratch/users/chaisaen/logs/hypnodensity_isfs_phase_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB

# Runs src/scripts/plot_hypnodensity_isfs_phase.py: subject-balanced average U-Sleep
# stage probability across continuous ISFS phase, per state (N2, N3), plus each
# state's event phase-bin distribution. Converted from
# demo_hypnodensity_isfs_phase.ipynb -- see that script's module docstring for the
# chunked-candidate/early-stop scanning strategy.
#
# Subjects are farmed out across --cpus-per-task worker processes
# (ProcessPoolExecutor) -- raise --cpus-per-task above to use more.
#
# All knobs below are sent to the script explicitly as CLI flags -- edit the
# values directly in this file to change them, or override at submit time, e.g.:
#   sbatch --export=ALL,N_SUBJECTS=200 run_hypnodensity_isfs_phase.sbatch
#   sbatch --export=ALL,LIMIT=500 run_hypnodensity_isfs_phase.sbatch   # quick test
#
# plot_hypnodensity_isfs_phase.py namespaces --output-dir by --event ("spindle"/"sw"), so
# a spindle run and a sw run against the same OUTPUT_DIR never collide -- submit
# both at once as two separate jobs (--job-name is just for squeue readability):
#   sbatch --job-name=hypnodensity_isfs_phase_spindle --export=ALL,EVENT=spindle run_hypnodensity_isfs_phase.sbatch
#   sbatch --job-name=hypnodensity_isfs_phase_sw      --export=ALL,EVENT=sw      run_hypnodensity_isfs_phase.sbatch
#
# This script additionally namespaces OUTPUT_DIR by HYPNO_EPOCH_SEC and
# USLEEP_EPOCH_SEC (the python script itself does not), so sweeping either
# epoch width against the same OUTPUT_DIR never collides. HYPNO_EPOCH_SEC
# selects which <stage>/<N>s/ directory preprocessing.py wrote each subject's
# bout/event artifacts under -- it must match whatever --hypno-epoch-sec that
# preprocessing.py run used, or every subject will be excluded as
# missing_bouts_file. USLEEP_EPOCH_SEC is unrelated: this script's own
# post-hoc averaging of the native 1-s U-Sleep hypnodensity before matching it
# to phase samples. Sweep 3/5/10 s USLEEP_EPOCH_SEC on 100 subjects as three
# separate jobs (against a fixed HYPNO_EPOCH_SEC matching your preprocessing run):
#   sbatch --job-name=hypnodensity_isfs_phase_3s_sp  --export=ALL,USLEEP_EPOCH_SEC=3,EVENT=spindle run_hypnodensity_isfs_phase.sbatch
#   sbatch --job-name=hypnodensity_isfs_phase_5s_sp  --export=ALL,USLEEP_EPOCH_SEC=5,EVENT=spindle run_hypnodensity_isfs_phase.sbatch
#   sbatch --job-name=hypnodensity_isfs_phase_10s_sp --export=ALL,USLEEP_EPOCH_SEC=10,EVENT=spindle run_hypnodensity_isfs_phase.sbatch
#   sbatch --job-name=hypnodensity_isfs_phase_3s_sw  --export=ALL,USLEEP_EPOCH_SEC=3,EVENT=sw run_hypnodensity_isfs_phase.sbatch
#   sbatch --job-name=hypnodensity_isfs_phase_5s_sw  --export=ALL,USLEEP_EPOCH_SEC=5,EVENT=sw run_hypnodensity_isfs_phase.sbatch
#   sbatch --job-name=hypnodensity_isfs_phase_10s_sw --export=ALL,USLEEP_EPOCH_SEC=10,EVENT=sw run_hypnodensity_isfs_phase.sbatch
#
# Run `python3 plot_hypnodensity_isfs_phase.py --help` for what each flag does.

set -euo pipefail

mkdir -p /scratch/users/chaisaen/logs

# --- Parameters (edit values here, or override via --export=ALL,VAR=value) ---
DATA_DIR="${DATA_DIR:-$SCRATCH/usleep_dryrun/data}"
CHANNEL="${CHANNEL:-C3}"
STATES="${STATES:-N2 N3}"
EVENT="${EVENT:-spindle}"
N_PHASE_POINTS="${N_PHASE_POINTS:-8}"
N_SUBJECTS="${N_SUBJECTS:-0}"
HYPNO_EPOCH_SEC="${HYPNO_EPOCH_SEC:-3}"
USLEEP_EPOCH_SEC="${USLEEP_EPOCH_SEC:-3}"
LIMIT="${LIMIT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/usleep_dryrun/outputs}/hypno${HYPNO_EPOCH_SEC}s/epoch${USLEEP_EPOCH_SEC}s"

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src/scripts
export PYTHONPATH=/home/users/chaisaen/infraslow/src

python3 plot_hypnodensity_isfs_phase.py \
    --data-dir "$DATA_DIR" \
    --channel "$CHANNEL" \
    --states $STATES \
    --event "$EVENT" \
    --n-phase-points "$N_PHASE_POINTS" \
    --n-subjects "$N_SUBJECTS" \
    --hypno-epoch-sec "$HYPNO_EPOCH_SEC" \
    --usleep-epoch-sec "$USLEEP_EPOCH_SEC" \
    --workers "${SLURM_CPUS_PER_TASK:-1}" \
    --output-dir "$OUTPUT_DIR" \
    ${LIMIT:+--limit "$LIMIT"}
