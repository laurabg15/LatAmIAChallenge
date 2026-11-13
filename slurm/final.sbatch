#!/bin/bash
# Reentrenamiento final (15-nov) con CodeCarbon activo. uso: sbatch slurm/final.sbatch gru aug
#SBATCH --job-name=lsc5_final
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
MODEL=${1:?modelo}; AUG=${2:-aug}
mkdir -p logs
source venv/bin/activate
FLAG=""; [ "$AUG" = "noaug" ] && FLAG="--no_aug"
python train.py --mode final --final_model $MODEL $FLAG --tag final
