#!/usr/bin/env python3
"""Benchmark throughput at each pipeline stage vs number of workers."""
import sys, os, time, argparse

sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import ParquetDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, required=True)
args = parser.parse_args()

dataset = ParquetDataset("/fscratch/nague/storage_benchmarks/images.parquet", max_samples=10000)
loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=args.num_workers,
    metrics_dir="/netscratch/nague/stage_metrics"
)

batch_count = 0
start = time.time()
for batch in loader:
    batch_count += 1
elapsed = time.time() - start

summary = loader._cleanup()

# Print CSV header once
if args.num_workers == 1:
    print("workers,batches,time_s,throughput,dataset_ms,staging_put_ms,collate_ms,batch_put_ms")

st = summary["stage_times"]
print(f"{args.num_workers},{batch_count},{elapsed:.2f},{batch_count*256/elapsed:.1f},"
      f"{st['dataset']['mean_ms']},{st['staging_put']['mean_ms']},"
      f"{st['collate']['mean_ms']},{st['batch_put']['mean_ms']}")
