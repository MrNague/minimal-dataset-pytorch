#!/usr/bin/env python3

import sys
import time
import statistics
import torch

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset
from torch.utils.data import DataLoader

PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"
NUM_SAMPLES = 9984
BATCH_SIZE = 256
WORKERS = [1, 2, 4, 8, 16, 32]
REPEATS = 3

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def collate_fn(samples):
    images = torch.stack([s[0] for s in samples])

    if images.dtype == torch.uint8:
        images = images.float()
        images.div_(255.0)

    labels = [s[1] for s in samples]

    if isinstance(labels[0], str):
        labels = torch.tensor(
            [int(label.replace("n", "")) for label in labels]
        )
    else:
        labels = torch.tensor(labels)

    return images, labels


print("Loading dataset...", flush=True)
t0 = time.time()

dataset = ParquetDataset(
    PARQUET,
    max_samples=NUM_SAMPLES
)

print(f"Dataset loaded in {time.time() - t0:.2f}s", flush=True)

print("workers,mean_throughput,std_throughput", flush=True)

for workers in WORKERS:
    throughputs = []

    for repeat in range(REPEATS):
        generator = torch.Generator()
        generator.manual_seed(12345 + repeat)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=workers,
            shuffle=True,
            generator=generator,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        start = time.perf_counter()

        batches = 0
        for _ in loader:
            batches += 1

        elapsed = time.perf_counter() - start
        samples = batches * BATCH_SIZE
        throughput = samples / elapsed

        throughputs.append(throughput)

        print(
            f"workers={workers}, repeat={repeat+1}, "
            f"throughput={throughput:.1f}",
            flush=True
        )

    mean = statistics.mean(throughputs)
    std = statistics.stdev(throughputs)

    print(
        f"RESULT,{workers},{mean:.1f},{std:.1f}",
        flush=True
    )
