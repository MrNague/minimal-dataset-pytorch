#!/bin/bash
#SBATCH --job-name=deep_stages
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/netscratch/%u/deep_stages_%j.out
#SBATCH --error=/netscratch/%u/deep_stages_%j.err

source ~/venv/torch_env/bin/activate

for nw in 1 2 4 8 16 32; do
    python3 -u ~/bachelor-project/experiments/profiling/benchmark_deep_stages.py --num-workers ${nw}
done

echo "DONE"
