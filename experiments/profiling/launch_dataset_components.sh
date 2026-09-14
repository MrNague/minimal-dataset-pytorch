#!/bin/bash

#SBATCH --job-name=dataset_parts
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/dataset_parts_%j.out
#SBATCH --error=/netscratch/%u/dataset_parts_%j.err

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project

python3 -u experiments/profiling/benchmark_dataset_components.py
