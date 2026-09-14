#!/bin/bash
#SBATCH --job-name=decode_lock
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/decode_lock_%j.out
#SBATCH --error=/netscratch/%u/decode_lock_%j.err

source ~/venv/torch_env/bin/activate
python3 -u ~/bachelor-project/experiments/profiling/test_decode_lock.py
