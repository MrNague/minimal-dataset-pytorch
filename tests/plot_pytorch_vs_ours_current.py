#!/usr/bin/env python3

import matplotlib.pyplot as plt

workers = [1, 2, 4, 8, 16, 32]

pytorch_mean = [439.4, 815.6, 1342.0, 1709.3, 1388.4, 857.8]
pytorch_std  = [13.7, 3.7, 18.2, 25.2, 15.0, 3.9]

ours_mean = [454.3, 859.2, 1535.6, 2560.7, 3245.3, 3099.8]
ours_std  = [1.0, 7.7, 10.7, 18.3, 94.4, 17.1]

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(
    workers,
    pytorch_mean,
    yerr=pytorch_std,
    marker="o",
    linewidth=2,
    capsize=4,
    label="PyTorch DataLoader"
)

ax.errorbar(
    workers,
    ours_mean,
    yerr=ours_std,
    marker="o",
    linewidth=2,
    capsize=4,
    label="Our optimized DataLoader"
)

ax.set_xlabel("Number of workers")
ax.set_ylabel("Throughput (samples/s)")
ax.set_title("PyTorch DataLoader vs Our Optimized DataLoader")

ax.set_xticks(workers)
ax.set_xlim(0, 33)

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

output = "docs/images/plot_pytorch_vs_ours_current.png"
plt.savefig(output, dpi=180)

print(f"Saved: {output}")
