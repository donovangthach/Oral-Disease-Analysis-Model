#!/bin/bash
#SBATCH --job-name=odam_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_eval_%j.txt
#SBATCH --error=logs/slurm_eval_err_%j.txt

# Load required modules
module load slurm
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source venv/bin/activate

# Run evaluation on the locked test split
echo "Starting ODAM evaluation job $SLURM_JOB_ID"
rm -f logs/evaluation_results.json  # remove stale results so cp can't copy an old file
python src/evaluate.py

# Keep a copy of the results, since evaluate.py overwrites logs/evaluation_results.json
cp logs/evaluation_results.json "logs/eval_results_${SLURM_JOB_ID}.json"
echo "Evaluation complete"
