#!/usr/bin/env python3

import sys
import time
import threading
import resource

import torch
import torch.nn.functional as F
import torchvision.io as tvio

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"

MAX_SAMPLES = 9984
OPERATIONS = 20000

WORKERS = [16, 32]

WORKING_SETS = [
    256,
    1024,
    2048,
    4096,
    9984,
]


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


print("Loading table once...", flush=True)

dataset = ParquetDataset(
    PARQUET,
    max_samples=MAX_SAMPLES
)

table = dataset._table


print("Preparing 9984 encoded JPEG tensors...", flush=True)

encoded = []

for i in range(MAX_SAMPLES):

    row = table.slice(i, 1).to_pylist()[0]

    encoded.append(
        torch.frombuffer(
            bytearray(row["image"]),
            dtype=torch.uint8
        )
    )


print("Cache ready.", flush=True)


def process(index, working_set):

    x = encoded[
        index % working_set
    ]

    img = tvio.decode_jpeg(
        x,
        mode=tvio.ImageReadMode.RGB
    )

    if img.shape[-2:] != (64, 64):

        img = F.interpolate(
            img.unsqueeze(0),
            size=(64, 64),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

    return img


def benchmark(workers, working_set):

    start_event = threading.Event()

    counts = [0] * workers


    def worker(worker_id):

        start_event.wait()

        count = 0

        for i in range(
            worker_id,
            OPERATIONS,
            workers
        ):

            _ = process(
                i,
                working_set
            )

            count += 1

        counts[worker_id] = count


    threads = []

    for wid in range(workers):

        t = threading.Thread(
            target=worker,
            args=(wid,)
        )

        t.start()
        threads.append(t)


    cpu_before = cpu_time()
    start = time.perf_counter()

    start_event.set()


    for t in threads:
        t.join()


    elapsed = time.perf_counter() - start
    cpu_after = cpu_time()


    throughput = sum(counts) / elapsed

    cores = (
        cpu_after - cpu_before
    ) / elapsed


    return throughput, cores


print()
print("=" * 90)
print("WORKING SET SCALING")
print("=" * 90)

print(
    "working_set,workers,throughput,cpu_cores",
    flush=True
)


for working_set in WORKING_SETS:

    print(
        f"\n--- WORKING SET {working_set} ---",
        flush=True
    )

    for workers in WORKERS:

        tp, cores = benchmark(
            workers,
            working_set
        )

        print(
            f"{working_set},"
            f"{workers},"
            f"{tp:.1f},"
            f"{cores:.2f}",
            flush=True
        )


print("\nDONE", flush=True)
