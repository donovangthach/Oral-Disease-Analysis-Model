#!/bin/bash
#SBATCH --job-name=odam_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_train_%j.txt
#SBATCH --error=logs/slurm_train_err_%j.txt

# Load required modules
module load slurm
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source venv/bin/activate

# Ensure log directory exists
mkdir -p logs
echo "Starting ODAM training job $SLURM_JOB_ID"

# Run training
python src/train.py
echo "Training complete"
