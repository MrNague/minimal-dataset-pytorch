#!/usr/bin/env python3
"""Plot MessagePack threading vs PyTorch."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

workers = [1, 2, 4, 8, 16, 32]

# MessagePack threading (from benchmark_msgpack.py - sequential test)
# We only have single-thread sequential data for msgpack
# Using our DataLoader with msgpack (threading)
msgpack_data = {
    256: [274.2, 540.0, 987.5, 1681.0, 1676.6, 1547.2],  # Placeholder - need real data
}

# PyTorch DataLoader
pytorch = {
    256: [275.0, 540.0, 1010.0, 1700.0, 2831.2, 2864.8],
}

# Local vs Remote storage
categories = ['Local SSD (Parquet)', 'Local SSD (MsgPack)', 'Remote BeeGFS']
ours_vals = [1681, 470, 1291]
pytorch_vals = [1700, 0, 1661]

# Plot 1: Storage format comparison at 8 workers
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(categories))
width = 0.3
ax.bar(x - width/2, ours_vals, width, label='Our DataLoader', color='#2196F3')
ax.bar(x + width/2, pytorch_vals, width, label='PyTorch DataLoader', color='#4CAF50')
ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('DataLoader Performance by Format (8 workers, BS=256)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
os.makedirs('docs/images/msgpack', exist_ok=True)
plt.savefig('docs/images/msgpack/plot_format_comparison.png', dpi=150)
print("Saved: docs/images/msgpack/plot_format_comparison.png")
print("Done!")
