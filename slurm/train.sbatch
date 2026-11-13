#!/bin/bash
# PLANTILLA: ajustar partición/GPU/módulos tras la capacitación del clúster (11-nov).
#SBATCH --job-name=lsc5
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
# uso: sbatch slurm/train.sbatch "gru lstm tcn" aug     (o ... noaug)
MODELS=${1:-"gru"}; AUG=${2:-aug}
mkdir -p logs
# module load python/3.x   # <-- según el clúster
source venv/bin/activate
FLAG=""; [ "$AUG" = "noaug" ] && FLAG="--no_aug"
python train.py --mode loso --models $MODELS $FLAG --tag loso_${AUG}
