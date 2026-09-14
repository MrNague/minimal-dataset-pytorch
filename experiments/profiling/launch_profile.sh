#!/bin/bash
#SBATCH --job-name=profile_dl
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/profile_%j.out
#SBATCH --error=/netscratch/%u/profile_%j.err

source ~/venv/torch_env/bin/activate
python3 -u ~/bachelor-project/experiments/profiling/profile_dataloader.py
