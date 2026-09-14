#!/bin/bash

#SBATCH --job-name=input_conv
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/input_conv_%j.out
#SBATCH --error=/netscratch/%u/input_conv_%j.err

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project

python3 -u experiments/optimization/benchmark_input_conversion.py
