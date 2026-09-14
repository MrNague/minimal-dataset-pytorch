#!/bin/bash

#SBATCH --job-name=final_scaling
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/netscratch/%u/final_scaling_%j.out
#SBATCH --error=/netscratch/%u/final_scaling_%j.err

set -euo pipefail

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project


echo "=============================================="
echo "BASELINE"
echo "=============================================="

python3 -u experiments/scaling/benchmark_final_scaling.py \
    --mode baseline


echo
echo
echo "=============================================="
echo "OPTIMIZED"
echo "=============================================="

# Separate Python process:
# the baseline Arrow table is released before
# loading the optimized dataset.

python3 -u experiments/scaling/benchmark_final_scaling.py \
    --mode optimized


echo
echo "FINAL BENCHMARK COMPLETE"
