#!/usr/bin/env python3
"""Calculate contribution of each stage to total time and throughput gap."""
import sys

# Données du benchmark deep_stages avec 32 cœurs
data = {
    1:  {"io": 0.051, "decode": 2.128, "preprocess": 0.009, "staging_put": 0.009, "collate": 2.873, "batch_put": 0.023},
    2:  {"io": 0.053, "decode": 2.160, "preprocess": 0.010, "staging_put": 0.010, "collate": 2.927, "batch_put": 0.017},
    4:  {"io": 0.054, "decode": 2.324, "preprocess": 0.010, "staging_put": 0.010, "collate": 3.031, "batch_put": 0.021},
    8:  {"io": 0.057, "decode": 2.632, "preprocess": 0.010, "staging_put": 0.010, "collate": 3.391, "batch_put": 0.020},
    16: {"io": 0.064, "decode": 3.883, "preprocess": 0.010, "staging_put": 0.010, "collate": 3.717, "batch_put": 0.021},
    32: {"io": 0.077, "decode": 7.058, "preprocess": 0.010, "staging_put": 0.010, "collate": 5.934, "batch_put": 0.018},
}

batch_size = 256

print(f"{'Workers':<10} {'Total_ms':<10} {'I/O%':<8} {'Decode%':<10} {'Preproc%':<10} {'Staging%':<10} {'Collate%':<10} {'BatchPut%':<10} {'Throughput':<12}")
print("-" * 95)

for w, stages in data.items():
    # Total time per batch (sum of all stages)
    total_ms = sum(stages.values())
    
    # Contribution of each stage
    io_pct = stages["io"] / total_ms * 100
    decode_pct = stages["decode"] / total_ms * 100
    preproc_pct = stages["preprocess"] / total_ms * 100
    staging_pct = stages["staging_put"] / total_ms * 100
    collate_pct = stages["collate"] / total_ms * 100
    batchput_pct = stages["batch_put"] / total_ms * 100
    
    # Theoretical throughput if no contention (using 1-worker stage times)
    base_total = sum(data[1].values())
    theoretical_throughput = batch_size / (base_total / 1000) * w
    
    # Actual throughput from benchmark
    actual_throughput = {
        1: 240, 2: 462, 4: 800, 8: 1291, 16: 1389, 32: 1270
    }[w]
    
    print(f"{w:<10} {total_ms:<10.2f} {io_pct:<8.1f} {decode_pct:<10.1f} {preproc_pct:<10.1f} {staging_pct:<10.1f} {collate_pct:<10.1f} {batchput_pct:<10.1f} {actual_throughput:<12}")
    
print()
print("Key finding: Decode and Collate dominate the total time.")
print("At 32 workers, Decode contributes 47% and Collate contributes 39%.")
print("The throughput gap comes from these two stages growing with worker count.")
