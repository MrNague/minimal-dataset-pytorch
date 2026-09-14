#!/bin/bash

#SBATCH --job-name=gil_diag
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/gil_diag_%j.out
#SBATCH --error=/netscratch/%u/gil_diag_%j.err

source ~/venv/torch_env/bin/activate
cd ~/bachelor-project

echo "============================================================"
echo "GIL diagnostic"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "============================================================"

python3 -u experiments/profiling/benchmark_gil_diagnostic.py \
    --parquet /fscratch/nague/storage_benchmarks/images.parquet \
    --samples 9984 \
    --output /netscratch/$USER/gil_diag_${SLURM_JOB_ID}.csv

echo "DONE"
