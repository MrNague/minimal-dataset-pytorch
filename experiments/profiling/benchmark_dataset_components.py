#!/usr/bin/env python3

import sys
import io
import time
import threading
import resource
from collections import Counter

import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"
SAMPLES = 9984
CACHE_SIZE = 256
WORKERS = [1, 2, 4, 8, 16, 32]

to_tensor = T.ToTensor()

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# ------------------------------------------------------------
# Load table ONCE
# ------------------------------------------------------------

print("Loading Parquet table once...", flush=True)

dataset = ParquetDataset(
    PARQUET,
    max_samples=SAMPLES
)

table = dataset._table

print("Table loaded.", flush=True)


# ------------------------------------------------------------
# Inspect image sizes / modes
# ------------------------------------------------------------

print("\nInspecting first 256 images...", flush=True)

sizes = Counter()
modes = Counter()
compressed_bytes = []

for i in range(CACHE_SIZE):

    row = table.slice(i, 1).to_pylist()[0]

    b = row["image"]

    compressed_bytes.append(b)

    img = Image.open(io.BytesIO(b))

    sizes[img.size] += 1
    modes[img.mode] += 1

print("Sizes:", dict(sizes), flush=True)
print("Modes:", dict(modes), flush=True)


# ------------------------------------------------------------
# Prepare decoded PIL cache
# ------------------------------------------------------------

print("\nPreparing decoded image cache...", flush=True)

decoded_images = []

for b in compressed_bytes:

    img = Image.open(io.BytesIO(b))

    # IMPORTANT:
    # force actual JPEG decoding here
    img.load()

    if img.mode != "RGB":
        img = img.convert("RGB")

    decoded_images.append(img)

print("Decoded cache ready.", flush=True)


# ------------------------------------------------------------
# Component functions
# ------------------------------------------------------------

def current(index):
    """
    Exact current ParquetDataset implementation.
    Includes timing attribute writes.
    """
    dataset[index]


def no_metrics(index):
    """
    Same useful work, without timing instrumentation.
    """
    row = table.slice(index, 1).to_pylist()[0]

    b = row["image"]

    img = Image.open(io.BytesIO(b))

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((64, 64))

    _ = to_tensor(img)


def parquet_only(index):
    """
    Arrow extraction only.
    """
    row = table.slice(index, 1).to_pylist()[0]

    _ = row["image"]
    _ = row["label"]


def pil_open_only(index):
    """
    PIL header/open only.
    Does NOT force pixel decoding.
    """
    b = compressed_bytes[index % CACHE_SIZE]

    _ = Image.open(io.BytesIO(b))


def jpeg_decode(index):
    """
    PIL open + actual JPEG decoding.

    img.load() explicitly forces decoding.
    """
    b = compressed_bytes[index % CACHE_SIZE]

    img = Image.open(io.BytesIO(b))

    img.load()


def resize_only(index):
    """
    Image has already been decoded.
    Measures PIL resize.
    """
    img = decoded_images[index % CACHE_SIZE]

    _ = img.resize((64, 64))


def totensor_only(index):
    """
    Image already decoded.
    Measures PIL -> torch tensor conversion.
    """
    img = decoded_images[index % CACHE_SIZE]

    _ = to_tensor(img)


MODES = {
    "current": current,
    "no_metrics": no_metrics,
    "parquet": parquet_only,
    "pil_open": pil_open_only,
    "jpeg_decode": jpeg_decode,
    "resize": resize_only,
    "totensor": totensor_only,
}


# ------------------------------------------------------------
# Generic threaded benchmark
# ------------------------------------------------------------

def benchmark(fn, workers):

    start_event = threading.Event()

    counts = [0] * workers


    def worker(worker_id):

        start_event.wait()

        count = 0

        for idx in range(
            worker_id,
            SAMPLES,
            workers
        ):

            fn(idx)

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


    processed = sum(counts)

    throughput = processed / elapsed

    cpu_cores = (
        cpu_after - cpu_before
    ) / elapsed


    return throughput, cpu_cores


# ------------------------------------------------------------
# Run benchmarks
# ------------------------------------------------------------

print()
print("=" * 90)
print("PARQUET DATASET COMPONENT SCALING")
print("=" * 90)

print(
    "mode,workers,throughput,cpu_cores",
    flush=True
)


for mode_name, fn in MODES.items():

    print(
        f"\n--- {mode_name.upper()} ---",
        flush=True
    )

    for workers in WORKERS:

        tp, cores = benchmark(
            fn,
            workers
        )

        print(
            f"{mode_name},"
            f"{workers},"
            f"{tp:.1f},"
            f"{cores:.2f}",
            flush=True
        )


print("\nDONE", flush=True)
