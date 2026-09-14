#!/bin/bash
#SBATCH --job-name=bench_stages
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/netscratch/%u/bench_stages_%j.out
#SBATCH --error=/netscratch/%u/bench_stages_%j.err

source ~/venv/torch_env/bin/activate

for nw in 1 2 4 8 16; do
    python3 -u ~/bachelor-project/experiments/profiling/benchmark_stages.py --num-workers ${nw}
done

echo "DONE"
