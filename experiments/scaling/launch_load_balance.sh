#!/bin/bash

#SBATCH --job-name=load_balance
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/load_balance_%j.out
#SBATCH --error=/netscratch/%u/load_balance_%j.err

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project

python3 -u experiments/scaling/benchmark_load_balance.py
