#!/usr/bin/env python3
"""Benchmark PyTorch DataLoader for comparison with minimal_dataset."""
import sys
import os
import time
import argparse

import torch
from torch.utils.data import DataLoader as TorchDataLoader

sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import ParquetDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--num-workers", type=int, required=True)
parser.add_argument("--parquet-path", type=str,
                    default="/fscratch/nague/storage_benchmarks/images.parquet")
args = parser.parse_args()

dataset = ParquetDataset(args.parquet_path)

def collate_fn(samples):
    images = torch.stack([s[0] for s in samples])
    labels = torch.tensor([int(s[1].replace('n', '')) if isinstance(s[1], str) else s[1] for s in samples])
    return images, labels

loader = TorchDataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=False
)

batch_count = 0
start = time.time()

for batch in loader:
    batch_count += 1
    if batch_count % 500 == 0:
        print(f"  [PyTorch {args.batch_size}/{args.num_workers}] batch {batch_count}", flush=True)

elapsed = time.time() - start
samples = batch_count * args.batch_size

# Memory
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

print(f"PYTORCH,{args.batch_size},{args.num_workers},{batch_count},{elapsed:.2f},"
      f"{samples/elapsed:.1f},0,0,0,0,{mem_gb:.2f}", flush=True)
