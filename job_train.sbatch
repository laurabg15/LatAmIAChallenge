#!/bin/bash
# Entrenamiento en el servidor cronos (Slurm): LOSO de dummy y linear + modelo final linear.
# Lanzar SIEMPRE desde la raíz del proyecto:
#   cd ~/LatAmAIChallenge
#   sbatch job_train.sbatch                    # valores por defecto (abajo)
#   sbatch job_train.sbatch --epochs 5         # prueba corta; los argumentos extra reemplazan a los de abajo
# Salida: train_<JOBID>.out, reports/cronos/ y models/cronos.joblib
#
#SBATCH --job-name=train
#SBATCH --partition=estudiantes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

set -e

source ~/venv_entrenamiento/bin/activate     # el trabajo no hereda el entorno de la sesión SSH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # numpy/BLAS y PyTorch: solo las CPU asignadas
export PYTHONUNBUFFERED=1                    # los print aparecen al instante en el .out
cd "$SLURM_SUBMIT_DIR"                       # el código usa rutas relativas a la raíz del proyecto

python train.py \
    --data_dir "$HOME/Dataset/train/landmarks" \
    --models dummy linear \
    --final_model linear \
    --tag cronos \
    --tracking_mode process \
    "$@"
