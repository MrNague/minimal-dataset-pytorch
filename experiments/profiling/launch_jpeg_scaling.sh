#!/bin/bash

#SBATCH --job-name=jpeg_scaling
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/jpeg_scaling_%j.out
#SBATCH --error=/netscratch/%u/jpeg_scaling_%j.err

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project

python3 -u experiments/profiling/benchmark_jpeg_scaling.py
