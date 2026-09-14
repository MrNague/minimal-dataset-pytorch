#!/usr/bin/env python3
"""Benchmark DataLoader across batch sizes and worker counts."""
import sys
import os
import time
import argparse

sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import ParquetDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--num-workers", type=int, required=True)
parser.add_argument("--parquet-path", type=str,
                    default="/fscratch/nague/storage_benchmarks/images.parquet")
parser.add_argument("--metrics-dir", type=str,
                    default="/netscratch/nague/dataloader_benchmarks")
args = parser.parse_args()

dataset = ParquetDataset(args.parquet_path)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    metrics_dir=args.metrics_dir
)

batch_count = 0
start = time.time()

for batch in loader:
    batch_count += 1
    if batch_count % 500 == 0:
        print(f"  [{args.batch_size}/{args.num_workers}] batch {batch_count}", flush=True)

elapsed = time.time() - start
summary = loader._cleanup()
samples = batch_count * args.batch_size

# Memory from /proc
try:
    with open(f'/proc/{os.getpid()}/status') as f:
        for line in f:
            if 'VmRSS' in line:
                mem_gb = int(line.split()[1]) / 1e6
                break
        else:
            mem_gb = 0
except:
    mem_gb = 0

print(f"{args.batch_size},{args.num_workers},{batch_count},{elapsed:.2f},"
      f"{samples/elapsed:.1f},"
      f"{summary['aggregate']['avg_utilization_pct']:.1f},"
      f"{summary['staging_queue']['empty_events']},"
      f"{summary['staging_queue']['full_events']},"
      f"{summary['batch_queue']['total_get_wait_s']:.2f},"
      f"{mem_gb:.2f}", flush=True)
