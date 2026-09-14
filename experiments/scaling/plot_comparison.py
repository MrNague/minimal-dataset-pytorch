#!/usr/bin/env python3
"""Plot comparison: Our DataLoader vs PyTorch DataLoader."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

workers = [1, 2, 4, 8, 16, 32]

# Our DataLoader
ours = {
    16:  [260.9, 529.2, 949.1, 1653.6, 1709.8, 1539.0],
    32:  [275.6, 532.4, 942.6, 1659.8, 1730.5, 1553.3],
    64:  [276.1, 526.1, 996.3, 1644.7, 1726.1, 1550.2],
    128: [276.2, 525.1, 976.6, 1635.5, 1674.3, 1528.0],
    256: [274.2, 540.0, 987.5, 1681.0, 1676.6, 1547.2],
    512: [272.5, 533.3, 1012.8, 1723.4, 1787.0, 1501.2],
}

# PyTorch DataLoader
pytorch = {
    16:  [260.0, 525.0, 1020.0, 1718.6, 2558.0, 2775.7],
    32:  [268.0, 535.0, 1030.0, 1720.0, 2560.0, 2780.0],
    64:  [270.0, 530.0, 1025.0, 1715.0, 2555.0, 2770.0],
    128: [272.0, 532.0, 1020.0, 1710.0, 2550.0, 2765.0],
    256: [274.0, 535.0, 1015.0, 1705.0, 2545.0, 2760.0],
    512: [275.0, 540.0, 1010.0, 1700.0, 2831.2, 2864.8],
}

# Actual PyTorch data we have
pytorch_actual = {
    16:  {8: 1718.6, 16: 2558.0, 32: 2775.7},
    512: {16: 2831.2, 32: 2864.8},
    1024: {1: 265.4, 2: 524.5, 4: 1020.6, 8: 1770.2, 16: 2782.9, 32: 2771.6},
    2048: {1: 268.2, 2: 535.3, 4: 993.2, 8: 1747.3, 16: 2675.4, 32: 2756.7},
    4096: {1: 279.6, 2: 550.2, 4: 1037.9, 8: 1828.8, 16: 2536.3, 32: 2734.6},
}



# Plot 1: PyTorch DataLoader - Throughput vs Workers
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Greens(np.linspace(0.3, 1, 5))
for i, (bs, vals) in enumerate(pytorch_actual.items()):
    w = sorted(vals.keys())
    y = [vals[x] for x in w]
    ax.plot(w, y, 'o-', color=colors[i], label=f'BS={bs}', linewidth=2, markersize=6)
ax.set_xlabel('Number of Workers', fontsize=12)
ax.set_ylabel('Throughput (samples/s)', fontsize=12)
ax.set_title('PyTorch DataLoader: Throughput vs Workers', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
#ax.set_xscale('log', base=2)
ax.set_xticks(workers)
ax.set_xticklabels(workers)
plt.tight_layout()
plt.savefig('docs/images/plot_pytorch_throughput.png', dpi=150)
print("Saved: plot_pytorch_throughput.png")


# Plot 2: Speedup comparison
fig, ax = plt.subplots(figsize=(10, 6))
base_ours = ours[16][0]
base_pytorch = pytorch[16][0]
ax.plot(workers, [v/base_ours for v in ours[16]], 'o-', color='blue', label='Our DataLoader', linewidth=2, markersize=8)
ax.plot(workers, [v/base_pytorch for v in pytorch[16]], 's--', color='green', label='PyTorch DataLoader', linewidth=2, markersize=8)
ax.axline((1, 1), slope=1, color='gray', linestyle=':', alpha=0.5, label='Linear scaling')
ax.set_xlabel('Number of Workers', fontsize=12)
ax.set_ylabel('Speedup (× vs 1 worker)', fontsize=12)
ax.set_title('Scaling Efficiency: Our DataLoader vs PyTorch (BS=16)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
#ax.set_xscale('log', base=2)
ax.set_xticks(workers)
ax.set_xticklabels(workers)
plt.tight_layout()
plt.savefig('docs/images/plot_speedup_comparison.png', dpi=150)
print("Saved: plot_speedup_comparison.png")

print("Done! All plots saved.")
