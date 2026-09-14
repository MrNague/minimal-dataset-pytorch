#!/usr/bin/env python3
"""Profile DataLoader to find exact bottleneck lines."""
import sys, cProfile, pstats, io
sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import ParquetDataset, DataLoader

dataset = ParquetDataset("/fscratch/nague/storage_benchmarks/images.parquet", max_samples=5000)
loader = DataLoader(dataset, batch_size=256, num_workers=16)

# Profiler
pr = cProfile.Profile()
pr.enable()

batch_count = 0
for batch in loader:
    batch_count += 1

pr.disable()

# Sort by cumulative time
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Top 30 functions

print(s.getvalue())
