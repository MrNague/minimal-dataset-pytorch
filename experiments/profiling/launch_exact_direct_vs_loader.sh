#!/bin/bash

#SBATCH --job-name=exact_dl
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/exact_dl_%j.out
#SBATCH --error=/netscratch/%u/exact_dl_%j.err

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project

python3 -u experiments/profiling/benchmark_exact_direct_vs_loader.py
