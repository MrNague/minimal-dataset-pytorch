#!/bin/bash

#SBATCH --job-name=staging_chunks
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/staging_chunks_%j.out
#SBATCH --error=/netscratch/%u/staging_chunks_%j.err

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project

python3 -u experiments/optimization/benchmark_staging_chunks.py
