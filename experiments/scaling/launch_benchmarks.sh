#!/bin/bash
BATCH_SIZES=(16 32 64)
NUM_WORKERS=(1 2 4)

for bs in "${BATCH_SIZES[@]}"; do
    for nw in "${NUM_WORKERS[@]}"; do
        cat > /tmp/bench_bs${bs}_w${nw}.sbatch << INNER
#!/bin/bash
#SBATCH --job-name=dl_bs${bs}_w${nw}
#SBATCH --partition=batch
#SBATCH --cpus-per-task=$((nw + 2))
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/dl_bs${bs}_w${nw}_%j.out
#SBATCH --error=/netscratch/%u/dl_bs${bs}_w${nw}_%j.err

source ~/venv/torch_env/bin/activate
python3 -u ~/bachelor-project/experiments/scaling/benchmark_dataloader.py --batch-size ${bs} --num-workers ${nw} --num-samples 500
INNER
        sbatch /tmp/bench_bs${bs}_w${nw}.sbatch
        echo "Submitted: bs=${bs}, workers=${nw}"
    done
done
