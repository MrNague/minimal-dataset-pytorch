#!/bin/bash
#SBATCH --job-name=deep_mp
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/netscratch/%u/deep_mp_%j.out
#SBATCH --error=/netscratch/%u/deep_mp_%j.err

source ~/venv/torch_env/bin/activate

for nw in 1 2 4 8 16; do
    python3 -u ~/bachelor-project/experiments/multiprocessing/benchmark_deep_mp.py --num-workers ${nw}
done

echo "DONE"
