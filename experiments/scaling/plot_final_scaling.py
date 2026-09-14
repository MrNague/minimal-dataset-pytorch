import matplotlib.pyplot as plt

workers = [1, 2, 4, 8, 16, 32]

baseline = [
    246.5,
    469.5,
    829.4,
    1364.3,
    1517.4,
    1383.6,
]

optimized = [
    454.3,
    859.2,
    1535.6,
    2560.7,
    3245.3,
    3099.8,
]

baseline_std = [
    1.3,
    0.6,
    15.1,
    30.1,
    37.3,
    8.4,
]

optimized_std = [
    1.0,
    7.7,
    10.7,
    18.3,
    94.4,
    17.1,
]

plt.figure(figsize=(8, 5))

plt.errorbar(
    workers,
    baseline,
    yerr=baseline_std,
    marker="o",
    capsize=4,
    label="Baseline",
)

plt.errorbar(
    workers,
    optimized,
    yerr=optimized_std,
    marker="o",
    capsize=4,
    label="Optimized",
)

plt.xlabel("Number of workers")
plt.ylabel("Throughput [samples/s]")
plt.title("DataLoader Scaling: Baseline vs Optimized")

plt.xticks(workers)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "/netscratch/nague/final_scaling.png",
    dpi=300,
)

print(
    "Saved: /netscratch/nague/final_scaling.png"
)
