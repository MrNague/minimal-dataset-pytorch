#!/usr/bin/env python3
"""Plot pipeline stage times vs number of workers."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

workers = [1, 2, 4, 8, 16]

# Data from benchmark_stages
dataset_ms = [4.013, 4.096, 4.392, 5.058, 9.287]
staging_put_ms = [0.012, 0.012, 0.012, 0.012, 0.012]
collate_ms = [2.824, 2.944, 2.744, 3.237, 3.499]
batch_put_ms = [0.023, 0.024, 0.019, 0.017, 0.021]

# Plot 1: All stages
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(workers, dataset_ms, 'o-', label='dataset[idx] (loading)', linewidth=2, markersize=8, color='red')
ax.plot(workers, collate_ms, 's-', label='collate_fn (assembly)', linewidth=2, markersize=8, color='blue')
ax.plot(workers, staging_put_ms, '^-', label='staging_queue.put', linewidth=2, markersize=8, color='green')
ax.plot(workers, batch_put_ms, 'd-', label='batch_queue.put', linewidth=2, markersize=8, color='orange')

ax.set_xlabel('Number of workers', fontsize=12)
ax.set_ylabel('Time per operation (ms)', fontsize=12)
ax.set_title('Pipeline Stage Times vs Number of Workers', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(workers)
plt.tight_layout()
plt.savefig('docs/images/plot_stages.png', dpi=150)
print("Saved: plot_stages.png")

# Plot 2: Throughput vs workers with stage contribution
fig, ax1 = plt.subplots(figsize=(10, 6))

throughput = [242.1, 462.8, 825.2, 1305.0, 1409.3]
ax1.bar([str(w) for w in workers], throughput, color='steelblue', alpha=0.7)
ax1.set_xlabel('Number of workers', fontsize=12)
ax1.set_ylabel('Throughput (samples/s)', fontsize=12, color='steelblue')
ax1.set_title('Throughput and Dataset Loading Time vs Workers', fontsize=14)

ax2 = ax1.twinx()
ax2.plot([str(w) for w in workers], dataset_ms, 'o-', color='red', linewidth=2, markersize=8)
ax2.set_ylabel('Dataset load time per sample (ms)', fontsize=12, color='red')

plt.tight_layout()
plt.savefig('docs/images/plot_stages_combined.png', dpi=150)
print("Saved: plot_stages_combined.png")
print("Done!")
