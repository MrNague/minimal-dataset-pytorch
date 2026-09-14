#!/bin/bash

#SBATCH --job-name=decoder_swap
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/decoder_swap_%j.out
#SBATCH --error=/netscratch/%u/decoder_swap_%j.err

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project

python3 -u experiments/optimization/benchmark_decoder_swap.py
