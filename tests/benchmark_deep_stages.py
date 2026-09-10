#!/usr/bin/env python3
"""Deep instrumentation benchmark with multiple orchestrators."""
import sys, os, time, argparse, threading

sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import ParquetDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, required=True)
parser.add_argument("--num-orchestrators", type=int, default=1)
args = parser.parse_args()

dataset = ParquetDataset("/fscratch/nague/storage_benchmarks/images.parquet", max_samples=10000)
loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=args.num_workers,
    num_orchestrators=args.num_orchestrators,
    metrics_dir="/netscratch/nague/deep_metrics"
)

batch_count = 0
start = time.time()
for batch in loader:
    batch_count += 1
elapsed = time.time() - start

summary = loader._cleanup()

st = summary["stage_times"]
sq = summary["staging_queue"]

print(f"{args.num_workers},{args.num_orchestrators},{batch_count},{elapsed:.2f},{batch_count*256/elapsed:.1f},"
      f"{st['collate']['mean_ms']},{sq['full_events']}", flush=True)
