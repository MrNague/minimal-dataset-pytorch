#!/bin/bash
#SBATCH --job-name=bench_mp_msg
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/netscratch/%u/bench_mp_msg_%j.out
#SBATCH --error=/netscratch/%u/bench_mp_msg_%j.err

source ~/venv/torch_env/bin/activate

echo "type,batch_size,num_workers,batches,time_s,throughput"

for nw in 1 2 4 8 16; do
    python3 -u ~/bachelor-project/experiments/multiprocessing/benchmark_mp_msgpack.py --batch-size 256 --num-workers ${nw}
done

echo "DONE"
