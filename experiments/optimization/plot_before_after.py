#!/usr/bin/env python3
"""Plot before/after optimization comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

workers = [1, 2, 4, 8, 16, 32]

# Throughput before (PIL) and after (torchvision)
before = [240, 462, 800, 1291, 1389, 1270]
after = [247, 450, 846, 1388, 1512, 1401]

# Decode time before/after
decode_before = [2.128, 2.160, 2.324, 2.632, 3.883, 7.058]
decode_after = [0.056, 0.067, 0.071, 0.082, 0.107, 0.178]

os.makedirs('docs/images/comparison', exist_ok=True)

# Plot 1: Throughput before vs after
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(workers, before, 'o-', color='red', label='Before (PIL)', linewidth=2, markersize=8)
ax.plot(workers, after, 's-', color='green', label='After (torchvision)', linewidth=2, markersize=8)
ax.set_xlabel('Number of workers', fontsize=12)
ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('Throughput: Before vs After Optimization', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(workers)
plt.tight_layout()
plt.savefig('docs/images/comparison/plot_throughput_before_after.png', dpi=150)
print("Saved: plot_throughput_before_after.png")

# Plot 2: Decode time before vs after
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(workers, decode_before, 'o-', color='red', label='Before (PIL)', linewidth=2, markersize=8)
ax.plot(workers, decode_after, 's-', color='green', label='After (torchvision)', linewidth=2, markersize=8)
ax.set_xlabel('Number of workers', fontsize=12)
ax.set_ylabel('Decode time per image (ms)', fontsize=12)
ax.set_title('Decode Time: Before vs After Optimization', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(workers)
# Log scale to show the 40x reduction clearly
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('docs/images/comparison/plot_decode_before_after.png', dpi=150)
print("Saved: plot_decode_before_after.png")

print("Done!")
