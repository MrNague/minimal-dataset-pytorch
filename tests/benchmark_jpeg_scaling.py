#!/usr/bin/env python3

import sys
import io
import time
import threading
import resource

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.io as tvio

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"

CACHE_SIZE = 2048
OPERATIONS = 20000
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


print("Loading dataset once...", flush=True)

dataset = ParquetDataset(
    PARQUET,
    max_samples=CACHE_SIZE
)

table = dataset._table


# ------------------------------------------------------------
# Compressed JPEG cache
# ------------------------------------------------------------

print(
    f"Preparing {CACHE_SIZE} compressed JPEGs...",
    flush=True
)

jpeg_bytes = []

for i in range(CACHE_SIZE):
    row = table.slice(i, 1).to_pylist()[0]
    jpeg_bytes.append(row["image"])


# Prepare encoded tensors ONCE so conversion overhead is
# not included in torchvision JPEG decoder benchmark.
encoded_tensors = [
    torch.frombuffer(
        bytearray(b),
        dtype=torch.uint8
    )
    for b in jpeg_bytes
]


print("JPEG cache ready.", flush=True)


# ------------------------------------------------------------
# Operations
# ------------------------------------------------------------

def pillow_decode(i):
    b = jpeg_bytes[i % CACHE_SIZE]

    img = Image.open(io.BytesIO(b))

    # Actually force JPEG decompression
    img.load()

    return img


def pillow_decode_resize(i):
    b = jpeg_bytes[i % CACHE_SIZE]

    img = Image.open(io.BytesIO(b))
    img.load()

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img.resize((64, 64))


def pillow_full(i):
    """
    Same image path as current dataset, except Parquet extraction
    and instrumentation have been removed.
    """
    b = jpeg_bytes[i % CACHE_SIZE]

    img = Image.open(io.BytesIO(b))
    img.load()

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((64, 64))

    return to_tensor(img)


def torchvision_decode(i):
    encoded = encoded_tensors[i % CACHE_SIZE]

    return tvio.decode_jpeg(
        encoded,
        mode=tvio.ImageReadMode.RGB
    )


MODES = {
    "pillow_decode": pillow_decode,
    "pillow_decode_resize": pillow_decode_resize,
    "pillow_full": pillow_full,
    "torchvision_decode": torchvision_decode,
}


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

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

    wall = time.perf_counter() - start
    cpu_after = cpu_time()

    total = sum(counts)

    throughput = total / wall

    cores = (
        cpu_after - cpu_before
    ) / wall

    return throughput, cores


print()
print("=" * 85)
print("JPEG SCALING DIAGNOSTIC")
print("=" * 85)

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
            f"{name},"
            f"{workers},"
            f"{tp:.1f},"
            f"{cores:.2f}",
            flush=True
        )


print("\nDONE", flush=True)
