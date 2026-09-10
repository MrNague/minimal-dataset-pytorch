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

CACHE_SIZE = 2048
OPERATIONS = 20000
WORKERS = [1, 2, 4, 8, 16, 32]


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# ============================================================
# Load compressed JPEGs once
# ============================================================

print("Loading dataset once...", flush=True)

dataset = ParquetDataset(
    PARQUET,
    max_samples=CACHE_SIZE
)

table = dataset._table

print(
    f"Preparing {CACHE_SIZE} encoded JPEG tensors...",
    flush=True
)

encoded_tensors = []

for i in range(CACHE_SIZE):

    row = table.slice(i, 1).to_pylist()[0]

    encoded = torch.frombuffer(
        bytearray(row["image"]),
        dtype=torch.uint8
    )

    encoded_tensors.append(encoded)


print("JPEG cache ready.", flush=True)


# ============================================================
# Operations
# ============================================================

def decode_only(i):

    encoded = encoded_tensors[
        i % CACHE_SIZE
    ]

    return tvio.decode_jpeg(
        encoded,
        mode=tvio.ImageReadMode.RGB
    )


def old_order(i):
    """
    Current torchvision experiment:

    JPEG
      -> uint8 full resolution
      -> float32 full resolution
      -> resize
      -> /255
    """

    encoded = encoded_tensors[
        i % CACHE_SIZE
    ]

    img = tvio.decode_jpeg(
        encoded,
        mode=tvio.ImageReadMode.RGB
    )

    if img.shape[-2:] != (64, 64):

        img = F.interpolate(
            img.unsqueeze(0).float(),
            size=(64, 64),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        img = img / 255.0

    else:

        img = img.float() / 255.0

    return img


def uint8_resize_only(i):
    """
    JPEG
      -> uint8
      -> resize uint8

    No float conversion.
    """

    encoded = encoded_tensors[
        i % CACHE_SIZE
    ]

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


def new_order(i):
    """
    Proposed version:

    JPEG
      -> uint8 full resolution
      -> resize uint8 to 64x64
      -> float32 ONLY at 64x64
      -> /255
    """

    encoded = encoded_tensors[
        i % CACHE_SIZE
    ]

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

    img = img.float() / 255.0

    return img


MODES = {
    "decode_only": decode_only,
    "old_float_before_resize": old_order,
    "uint8_resize_only": uint8_resize_only,
    "new_resize_before_float": new_order,
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
            OPERATIONS,
            workers
        ):

            result = fn(i)
            _ = result

            count += 1

        counts[worker_id] = count


    threads = []

    for worker_id in range(workers):

        t = threading.Thread(
            target=worker,
            args=(worker_id,)
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

    total = sum(counts)

    throughput = total / elapsed

    cores = (
        cpu_after - cpu_before
    ) / elapsed

    return throughput, cores


# ============================================================
# Run
# ============================================================

print()
print("=" * 90)
print("FLOAT-BEFORE-RESIZE vs RESIZE-BEFORE-FLOAT")
print("=" * 90)

print(
    "mode,workers,throughput,cpu_cores",
    flush=True
)


for name, fn in MODES.items():

    print()
    print(
        f"--- {name.upper()} ---",
        flush=True
    )

    for workers in WORKERS:

        tp, cores = benchmark(
            fn,
            workers
        )

        print(
            f"{name},"
            f"{workers},"
            f"{tp:.1f},"
            f"{cores:.2f}",
            flush=True
        )


print()
print("DONE", flush=True)
