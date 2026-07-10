#!/bin/bash
#SBATCH --job-name=all_metrics
#SBATCH --partition=mignot
#SBATCH --output=logs/all_metrics_%j.log
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G

mkdir -p logs
mkdir -p "$SCRATCH/results/spectra"

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=6
export METADATA=$OAK/psg/Bioserenity/Excel/Morpheus_Data_All5.csv
export METADATA2=$OAK/psg/Bioserenity/Excel/bioserenity_metadata3.csv
export OUTPUT=$SCRATCH/results/all_metrics.csv
export ERROR_OUTPUT=$SCRATCH/results/all_metrics_errors.csv
export SPECTRA_DIR=$SCRATCH/results/spectra

python -u src/run_all_metrics.py \
    --workers ${SLURM_CPUS_PER_TASK} \
    --metadata ${METADATA} \
    --metadata2 ${METADATA2} \
    --output ${OUTPUT} \
    --error-output ${ERROR_OUTPUT} \
    --spectra-dir ${SPECTRA_DIR}
