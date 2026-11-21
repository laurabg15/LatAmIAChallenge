#!/bin/bash
# Predicción del test en el servidor cronos (Slurm) con el modelo entrenado por job_train.sbatch.
# Lanzar SIEMPRE desde la raíz del proyecto, cuando job_train.sbatch haya terminado:
#   cd ~/LatAmAIChallenge
#   sbatch job_submission.sbatch
# Salida: submission_<JOBID>.out y reports/cronos/submission.csv
#
#SBATCH --job-name=submission
#SBATCH --partition=estudiantes
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -e

source ~/venv_entrenamiento/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

# En Linux "test" va en minúscula (en el PC local la carpeta es data/Test)
python generate_submission.py \
    --model models/cronos.joblib \
    --test_dir "$HOME/Dataset/test/landmarks" \
    --template "$HOME/Dataset/test/submission_template.csv" \
    --out reports/cronos/submission.csv \
    "$@"
