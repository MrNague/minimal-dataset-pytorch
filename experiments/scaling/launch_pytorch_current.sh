#!/bin/bash
#SBATCH --job-name=pytorch_current
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/netscratch/%u/pytorch_current_%j.out
#SBATCH --error=/netscratch/%u/pytorch_current_%j.err

source ~/venv/torch_env/bin/activate

cd ~/bachelor-project

python3 -u experiments/scaling/benchmark_pytorch_current.py
