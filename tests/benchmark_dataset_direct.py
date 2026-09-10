#!/usr/bin/env python3

import sys
import time
import threading
import resource
import inspect
import gc

import torch

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset, DataLoader


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"
SAMPLES = 9984
BATCH_SIZE = 256
WORKERS = [1, 2, 4, 8, 16, 32]


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


# ---------------------------------------------------------
# Load dataset ONCE
# ---------------------------------------------------------

print("Loading Parquet dataset once...", flush=True)

dataset = ParquetDataset(
    PARQUET,
    max_samples=SAMPLES
)

print(f"Dataset ready: {len(dataset)} samples used", flush=True)


# ---------------------------------------------------------
# Dataset-only benchmark
# No DataLoader
# No queue
# No orchestrator
# No collate
# ---------------------------------------------------------

def benchmark_direct(num_workers):

    start_event = threading.Event()
    counts = [0] * num_workers


    def worker(worker_id):

        start_event.wait()

        count = 0

        for idx in range(worker_id, SAMPLES, num_workers):
            dataset[idx]
            count += 1

        counts[worker_id] = count


    threads = []

    for worker_id in range(num_workers):

        thread = threading.Thread(
            target=worker,
            args=(worker_id,)
        )

        thread.start()
        threads.append(thread)


    cpu_before = cpu_time()
    start = time.perf_counter()

    start_event.set()

    for thread in threads:
        thread.join()

    elapsed = time.perf_counter() - start
    cpu_after = cpu_time()

    processed = sum(counts)

    throughput = processed / elapsed
    cpu_cores = (cpu_after - cpu_before) / elapsed

    return throughput, cpu_cores


# ---------------------------------------------------------
# Full DataLoader benchmark
# Same dataset object
# ---------------------------------------------------------

def benchmark_loader(num_workers):

    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "num_workers": num_workers,
    }

    signature = inspect.signature(DataLoader.__init__)

    if "num_orchestrators" in signature.parameters:
        kwargs["num_orchestrators"] = 1

    loader = DataLoader(**kwargs)

    cpu_before = cpu_time()
    start = time.perf_counter()

    processed = 0

    for images, labels in loader:
        processed += images.shape[0]

    elapsed = time.perf_counter() - start
    cpu_after = cpu_time()

    cpu_cores = (cpu_after - cpu_before) / elapsed
    throughput = processed / elapsed

    del loader
    gc.collect()

    return throughput, cpu_cores


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

print("=" * 76)
print("DATASET DIRECT vs FULL DATALOADER")
print("=" * 76)

print(
    "workers,"
    "direct_throughput,"
    "direct_cpu,"
    "loader_throughput,"
    "loader_cpu",
    flush=True
)


for workers in WORKERS:

    print(
        f"\nTesting {workers} worker(s)...",
        flush=True
    )

    direct_tp, direct_cpu = benchmark_direct(workers)

    print(
        f"  direct done: {direct_tp:.1f} samples/s",
        flush=True
    )

    loader_tp, loader_cpu = benchmark_loader(workers)

    print(
        f"  loader done: {loader_tp:.1f} samples/s",
        flush=True
    )

    print(
        f"RESULT,"
        f"{workers},"
        f"{direct_tp:.1f},"
        f"{direct_cpu:.2f},"
        f"{loader_tp:.1f},"
        f"{loader_cpu:.2f}",
        flush=True
    )


print("\nDONE", flush=True)
