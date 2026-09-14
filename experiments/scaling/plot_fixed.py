#!/usr/bin/env python3
"""Corrected plots with linear X-axis."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

workers = [1, 2, 4, 8, 16, 32]
ours = {
    16:  [260.9, 529.2, 949.1, 1653.6, 1709.8, 1539.0],
    256: [274.2, 540.0, 987.5, 1681.0, 1676.6, 1547.2],
    512: [272.5, 533.3, 1012.8, 1723.4, 1787.0, 1501.2],
}

# Plot 1: Throughput vs Workers (linear X)
fig, ax = plt.subplots(figsize=(10, 6))
for bs, vals in ours.items():
    ax.plot(workers, vals, 'o-', label=f'BS={bs}', linewidth=2, markersize=8)

# Add ideal linear scaling line from 1-worker baseline
ideal_x = [1, 2, 4, 8, 16, 32]
ideal_y = [ours[16][0] * w for w in ideal_x]
ax.plot(ideal_x, ideal_y, '--', color='gray', alpha=0.6, label='Linear scaling')

ax.set_xlabel('Number of Workers', fontsize=12)
ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('DataLoader Throughput vs Workers (Parquet, 100k images, A100)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(workers)
plt.tight_layout()
plt.savefig('/netscratch/nague/plot_throughput_linear.png', dpi=150)
print("Saved: plot_throughput_linear.png")

# Plot 2: Speedup ratio vs ideal
fig, ax = plt.subplots(figsize=(10, 6))
for bs, vals in ours.items():
    base = vals[0]
    speedup = [v / base for v in vals]
    ax.plot(workers, speedup, 'o-', label=f'BS={bs}', linewidth=2, markersize=8)

ax.plot(ideal_x, ideal_x, '--', color='gray', alpha=0.6, label='Linear scaling')
ax.set_xlabel('Number of Workers', fontsize=12)
ax.set_ylabel('Speedup (x vs 1 worker)', fontsize=12)
ax.set_title('Scaling Efficiency (linear X-axis)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(workers)
plt.tight_layout()
plt.savefig('/netscratch/nague/plot_speedup_linear.png', dpi=150)
print("Saved: plot_speedup_linear.png")

# Plot 3: Local vs Remote (same workers: 8)
fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Local SSD', 'Remote BeeGFS']
ours_vals = [1681, 1291]
pytorch_vals = [1700, 1661]

x = np.arange(len(categories))
width = 0.3
ax.bar(x - width/2, ours_vals, width, label='Our DataLoader', color='#2196F3')
ax.bar(x + width/2, pytorch_vals, width, label='PyTorch DataLoader', color='#4CAF50')

for i, (o, p) in enumerate(zip(ours_vals, pytorch_vals)):
    ax.text(x[i], max(o, p) + 30, f'{p/o:.2f}x', ha='center', fontweight='bold')

ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('DataLoader Comparison: Local vs Remote (8 workers)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/netscratch/nague/plot_local_vs_remote.png', dpi=150)
print("Saved: plot_local_vs_remote.png")

import os
os.system('cp /netscratch/nague/plot_*.png ~/bachelor-project/docs/images/')
print("Copied to docs/images/")
