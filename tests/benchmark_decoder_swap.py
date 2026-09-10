#!/usr/bin/env python3

import sys
import io
import time
import threading
import resource

import torch
import torch.nn.functional as F
import torchvision.io as tvio
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"
SAMPLES = 9984
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


print("Loading Parquet table once...", flush=True)

dataset = ParquetDataset(
    PARQUET,
    max_samples=SAMPLES
)

table = dataset._table

print("Table ready.", flush=True)


# ============================================================
# CURRENT PILLOW PATH
# ============================================================

def pillow_pipeline(index):

    row = table.slice(index, 1).to_pylist()[0]

    img_bytes = row["image"]
    label = row["label"]

    img = Image.open(
        io.BytesIO(img_bytes)
    )

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((64, 64))

    img = to_tensor(img)

    return img, label


# ============================================================
# TORCHVISION PATH
# ============================================================

def torchvision_pipeline(index):

    row = table.slice(index, 1).to_pylist()[0]

    img_bytes = row["image"]
    label = row["label"]

    # Realistic bytes -> encoded uint8 tensor conversion.
    # Unlike the previous decoder benchmark, this conversion
    # is INCLUDED in the measurement.
    encoded = torch.frombuffer(
        bytearray(img_bytes),
        dtype=torch.uint8
    )

    img = tvio.decode_jpeg(
        encoded,
        mode=tvio.ImageReadMode.RGB
    )

    # decode_jpeg -> uint8 CHW tensor

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

    return img, label


MODES = {
    "pillow": pillow_pipeline,
    "torchvision": torchvision_pipeline,
}


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(fn, workers):

    start_event = threading.Event()

    counts = [0] * workers


    def worker(worker_id):

        start_event.wait()

        count = 0

        for index in range(
            worker_id,
            SAMPLES,
            workers
        ):

            result = fn(index)

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
# RUN
# ============================================================

print()
print("=" * 85)
print("PILLOW vs TORCHVISION — REAL DATASET PATH")
print("=" * 85)

print(
    "implementation,workers,throughput,cpu_cores",
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
