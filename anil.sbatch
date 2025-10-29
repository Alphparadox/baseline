#!/bin/bash
# ===========================================================
# LLaVA Job Runner (H100 Shard Cluster)
# Author: Anil Prajapati
# ===========================================================

#SBATCH --job-name=llava_test
#SBATCH --output=../logs/llava_test_%j.out
#SBATCH --error=../logs/llava_test_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:1      # ✅ correct for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# (Removed exclusion — both node1 and node2 are valid)

echo "=============================="
echo "Starting job $SLURM_JOB_ID on $(hostname)"
date
echo "=============================="


# -----------------------------
# 1️⃣ Activate Conda environment
# -----------------------------
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kiva_env
echo "Environment loaded."

# -----------------------------
# 2️⃣ Clean up any old GPU jobs
# -----------------------------
echo "Checking for old GPU jobs..."
nvidia-smi
OLD_PIDS=$(nvidia-smi | grep python | awk '{print $5}')
if [ ! -z "$OLD_PIDS" ]; then
  echo "Killing old Python processes: $OLD_PIDS"
  kill -9 $OLD_PIDS
fi

# -----------------------------
# 3️⃣ Run your Python benchmark
# -----------------------------
echo "Running LLaVA benchmark..."
python /home/naveenkumar/baseline/sample.py

echo "=============================="
echo "Job finished at $(date)"
echo "=============================="