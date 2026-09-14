#!/usr/bin/env python3

"""Corrected plots with linear X-axis."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

workers = [1, 2, 4, 8, 16, 32]

# Data
ours = {
    16:  [260.9, 529.2, 949.1, 1653.6, 1709.8, 1539.0],
    256: [274.2, 540.0, 987.5, 1681.0, 1676.6, 1547.2],
    512: [272.5, 533.3, 1012.8, 1723.4, 1787.0, 1501.2],
}

pytorch = {
    16:  [260.0, 525.0, 1020.0, 1718.6, 2558.0, 2775.7],
    512: [275.0, 540.0, 1010.0, 1700.0, 2831.2, 2864.8],
}

# ============================================================
# Plot 1: Our DataLoader throughput vs workers (linear X)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for bs, vals in ours.items():
    ax.plot(workers, vals, 'o-', label=f'Batch size {bs}', linewidth=2, markersize=8)

# Ideal linear scaling from 1 worker
ideal = [ours[16][0] * w for w in workers]
ax.plot(workers, ideal, '--', color='gray', alpha=0.5, label='Linear scaling')

ax.set_xlabel('Number of workers', fontsize=12)
ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('Our DataLoader: throughput vs workers', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(workers)
plt.tight_layout()
plt.savefig('docs/images/plot_ours_throughput.png', dpi=150)
print("Saved: plot_ours_throughput.png")

# ============================================================
# Plot 2: Our vs PyTorch at BS=16
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(workers, ours[16], 'o-', color='blue', label='Our DataLoader', linewidth=2, markersize=8)
ax.plot(workers, pytorch[16], 's--', color='green', label='PyTorch DataLoader', linewidth=2, markersize=8)

ax.set_xlabel('Number of workers', fontsize=12)
ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('Comparison: Our DataLoader vs PyTorch (batch size 16)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(workers)
plt.tight_layout()
plt.savefig('docs/images/plot_comparison_bs16.png', dpi=150)
print("Saved: plot_comparison_bs16.png")

# ============================================================
# Plot 3: Speedup vs ideal linear
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for bs, vals in ours.items():
    base = vals[0]
    speedup = [v / base for v in vals]
    ax.plot(workers, speedup, 'o-', label=f'Batch size {bs}', linewidth=2, markersize=8)

ax.plot(workers, workers, '--', color='gray', alpha=0.5, label='Linear scaling')
ax.set_xlabel('Number of workers', fontsize=12)
ax.set_ylabel('Speedup (times vs 1 worker)', fontsize=12)
ax.set_title('Scaling efficiency: deviation from linear beyond 4 workers', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(workers)
plt.tight_layout()
plt.savefig('docs/images/plot_speedup.png', dpi=150)
print("Saved: plot_speedup.png")

# ============================================================
# Plot 4: Local vs Remote (8 workers, BS=256)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Local SSD', 'Remote BeeGFS']
ours_vals = [1681, 1291]
pytorch_vals = [1700, 1661]

x = np.arange(len(categories))
width = 0.3
bars1 = ax.bar(x - width/2, ours_vals, width, label='Our DataLoader', color='#2196F3')
bars2 = ax.bar(x + width/2, pytorch_vals, width, label='PyTorch DataLoader', color='#4CAF50')

for i, (o, p) in enumerate(zip(ours_vals, pytorch_vals)):
    ax.text(x[i], max(o, p) + 20, f'{p/o:.2f}x', ha='center', fontweight='bold')
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha='center', fontsize=9)

ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('DataLoader: Local SSD vs Remote Storage (8 workers, BS=256)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('docs/images/plot_local_vs_remote.png', dpi=150)
print("Saved: plot_local_vs_remote.png")

# Copy to docs
#os.system('cp /netscratch/nague/plot_*.png ~/bachelor-project/docs/images/')
print("Copied to docs/images/")
print("Done!")