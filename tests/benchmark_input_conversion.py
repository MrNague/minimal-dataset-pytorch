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
SAMPLES = 9984
WORKERS = [8, 16, 32]

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
    max_samples=SAMPLES
)

table = dataset._table

print("Table ready.", flush=True)


# ============================================================
# Prepare caches OUTSIDE benchmark
# ============================================================

print("Pre-extracting Python bytes...", flush=True)

cached_bytes = []

for i in range(SAMPLES):
    row = table.slice(i, 1).to_pylist()[0]
    cached_bytes.append(row["image"])

print("Preparing encoded tensors...", flush=True)

cached_encoded = [
    torch.frombuffer(
        bytearray(b),
        dtype=torch.uint8
    )
    for b in cached_bytes
]

print("Caches ready.", flush=True)


# ============================================================
# Small components
# ============================================================

def parquet_only(i):

    row = table.slice(i, 1).to_pylist()[0]

    return row["image"]


def parquet_bytearray(i):

    row = table.slice(i, 1).to_pylist()[0]

    return bytearray(row["image"])


def parquet_frombuffer(i):

    row = table.slice(i, 1).to_pylist()[0]

    b = bytearray(row["image"])

    return torch.frombuffer(
        b,
        dtype=torch.uint8
    )


def cached_bytes_frombuffer(i):

    b = bytearray(
        cached_bytes[i]
    )

    return torch.frombuffer(
        b,
        dtype=torch.uint8
    )


# ============================================================
# Full paths
# ============================================================

def finish(encoded):

    img = tvio.decode_jpeg(
        encoded,
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


def full_exact(i):

    row = table.slice(i, 1).to_pylist()[0]

    encoded = torch.frombuffer(
        bytearray(row["image"]),
        dtype=torch.uint8
    )

    return finish(encoded)


def full_cached_bytes(i):

    encoded = torch.frombuffer(
        bytearray(cached_bytes[i]),
        dtype=torch.uint8
    )

    return finish(encoded)


def full_preencoded(i):

    return finish(
        cached_encoded[i]
    )


MODES = {
    "parquet_only": parquet_only,
    "parquet_bytearray": parquet_bytearray,
    "parquet_frombuffer": parquet_frombuffer,
    "cached_bytes_frombuffer": cached_bytes_frombuffer,
    "full_exact": full_exact,
    "full_cached_bytes": full_cached_bytes,
    "full_preencoded": full_preencoded,
}


# ============================================================
# Benchmark
# ============================================================

def benchmark(fn, workers):

    start_event = threading.Event()
    counts = [0] * workers

    def worker(worker_id):

        start_event.wait()

        count = 0

        for i in range(
            worker_id,
            SAMPLES,
            workers
        ):
            _ = fn(i)
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
print("=" * 95)
print("INPUT CONVERSION SCALING")
print("=" * 95)

print(
    "mode,workers,throughput,cpu_cores",
    flush=True
)


for name, fn in MODES.items():

    print(
        f"\n--- {name.upper()} ---",
        flush=True
    )

    for workers in WORKERS:

        tp, cores = benchmark(
            fn,
            workers
        )

        print(
            f"{name},{workers},"
            f"{tp:.1f},{cores:.2f}",
            flush=True
        )


print("\nDONE", flush=True)
