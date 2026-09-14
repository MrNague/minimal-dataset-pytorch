#!/usr/bin/env python3

import argparse
import gc
import resource
import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/nague/bachelor-project")


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"
SAMPLES = 9984
BATCH_SIZE = 256

WORKERS = [1, 2, 4, 8, 16, 32]
REPEATS = 3


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["baseline", "optimized"],
    required=True,
)

args = parser.parse_args()


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# Select implementation
# ============================================================

if args.mode == "baseline":

    from experiments.legacy.parquet_dataset_before_final_optimization import (
        ParquetDataset,
    )

    from experiments.legacy.dataloader_before_final_optimization import (
        DataLoader,
    )

else:

    from minimal_dataset.parquet_dataset import (
        ParquetDataset,
    )

    from minimal_dataset.dataloader import (
        DataLoader,
    )


def cpu_time():

    usage = resource.getrusage(
        resource.RUSAGE_SELF
    )

    return (
        usage.ru_utime
        + usage.ru_stime
    )


# ============================================================
# Dataset construction
# ============================================================

print(
    f"MODE={args.mode}",
    flush=True,
)

print(
    "Loading dataset...",
    flush=True,
)

load_start = time.perf_counter()

dataset = ParquetDataset(
    PARQUET,
    max_samples=SAMPLES,
)

load_time = (
    time.perf_counter()
    - load_start
)

print(
    f"Dataset ready in {load_time:.2f}s",
    flush=True,
)

print(
    f"Dataset length: {len(dataset)}",
    flush=True,
)


# ============================================================
# Benchmark
# ============================================================

print()
print("=" * 100)

print(
    f"FINAL SCALING — {args.mode.upper()}"
)

print("=" * 100)

print(
    "mode,workers,repeat,"
    "samples,batches,"
    "elapsed_s,throughput,cpu_cores",
    flush=True,
)


summary = {}


for workers in WORKERS:

    throughputs = []
    cpu_cores_values = []


    for repeat in range(
        1,
        REPEATS + 1
    ):

        if args.mode == "baseline":

            loader = DataLoader(
                dataset=dataset,
                batch_size=BATCH_SIZE,
                num_workers=workers,
                num_orchestrators=1,
            )

        else:

            loader = DataLoader(
                dataset=dataset,
                batch_size=BATCH_SIZE,
                num_workers=workers,

                # Values selected from the
                # staging queue experiments.
                staging_chunk_size=32,
                max_staging_size=512,
            )


        # Same deterministic seed for the
        # corresponding repetitions.
        loader.set_epoch(
            seed=12345 + repeat
        )


        before_cpu = cpu_time()

        start = time.perf_counter()


        total_samples = 0
        batches = 0


        for images, labels in loader:

            batches += 1

            total_samples += (
                images.shape[0]
            )


        elapsed = (
            time.perf_counter()
            - start
        )

        after_cpu = cpu_time()


        throughput = (
            total_samples
            / elapsed
        )


        cpu_cores = (
            after_cpu
            - before_cpu
        ) / elapsed


        throughputs.append(
            throughput
        )

        cpu_cores_values.append(
            cpu_cores
        )


        print(
            f"{args.mode},"
            f"{workers},"
            f"{repeat},"
            f"{total_samples},"
            f"{batches},"
            f"{elapsed:.3f},"
            f"{throughput:.1f},"
            f"{cpu_cores:.2f}",
            flush=True,
        )


        del loader
        gc.collect()


    summary[workers] = {
        "mean": statistics.mean(
            throughputs
        ),
        "std": statistics.stdev(
            throughputs
        ),
        "cpu": statistics.mean(
            cpu_cores_values
        ),
    }


# ============================================================
# Summary
# ============================================================

print()
print("-" * 100)

print(
    "SUMMARY"
)

print("-" * 100)

print(
    "mode,workers,"
    "mean_throughput,"
    "std_throughput,"
    "mean_cpu_cores,"
    "speedup_vs_1",
    flush=True,
)


baseline_1 = summary[1]["mean"]


for workers in WORKERS:

    result = summary[workers]

    speedup = (
        result["mean"]
        / baseline_1
    )

    print(
        f"{args.mode},"
        f"{workers},"
        f"{result['mean']:.1f},"
        f"{result['std']:.1f},"
        f"{result['cpu']:.2f},"
        f"{speedup:.2f}",
        flush=True,
    )


best_workers = max(
    WORKERS,
    key=lambda w: summary[w]["mean"]
)

best_throughput = (
    summary[best_workers]["mean"]
)


print()
print(
    f"BEST={best_workers} workers, "
    f"{best_throughput:.1f} samples/s",
    flush=True,
)

print(
    f"MODE {args.mode} DONE",
    flush=True,
)
