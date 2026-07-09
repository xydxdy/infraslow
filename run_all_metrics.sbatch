#!/bin/bash
#SBATCH --job-name=all_metrics
#SBATCH --partition=mignot
#SBATCH --output=logs/all_metrics_%j.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G

mkdir -p logs
mkdir -p results

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK=16
export METADATA=$OAK/psg/Bioserenity/Excel/Morpheus_Data_All5.csv
export OUTPUT=results/all_metrics.csv
export ERROR_OUTPUT=results/all_metrics_errors.csv

python -u src/run_all_metrics.py \
    --workers ${SLURM_CPUS_PER_TASK} \
    --metadata ${METADATA} \
    --output ${OUTPUT} \
    --error-output ${ERROR_OUTPUT}
