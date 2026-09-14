#!/bin/bash
#SBATCH --job-name=dl_pytorch
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/netscratch/%u/dl_pytorch_%j.out
#SBATCH --error=/netscratch/%u/dl_pytorch_%j.err

source ~/venv/torch_env/bin/activate

BATCH_SIZES=(16 32 64 128 256 512 1024 2048 4096)
NUM_WORKERS=(1 2 4 8 16 32)

echo "type,batch_size,num_workers,batches,time_s,throughput,worker_util,staging_empty,staging_full,batch_wait_s,mem_gb"

for bs in "${BATCH_SIZES[@]}"; do
    for nw in "${NUM_WORKERS[@]}"; do
        python3 -u ~/bachelor-project/experiments/scaling/benchmark_pytorch.py \
            --batch-size ${bs} --num-workers ${nw}
    done
done

echo "DONE"
