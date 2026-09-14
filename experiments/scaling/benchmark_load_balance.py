#!/usr/bin/env python3

import sys
import time
import threading
import resource
import statistics

import torch
import torch.nn.functional as F
import torchvision.io as tvio

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"
SAMPLES = 9984
WORKERS = [16, 32]
REPEATS = 3
CHUNK = 8

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


print("Preparing encoded JPEG tensors...", flush=True)

encoded = []

for i in range(SAMPLES):

    row = table.slice(i, 1).to_pylist()[0]

    encoded.append(
        torch.frombuffer(
            bytearray(row["image"]),
            dtype=torch.uint8
        )
    )

print("Cache ready.", flush=True)


def process(index):

    img = tvio.decode_jpeg(
        encoded[index],
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


# ============================================================
# Static strided partition
# ============================================================

def benchmark_static(workers):

    start_event = threading.Event()

    worker_times = [0.0] * workers
    counts = [0] * workers


    def worker(worker_id):

        start_event.wait()

        begin = time.perf_counter()
        count = 0

        for index in range(
            worker_id,
            SAMPLES,
            workers
        ):

            _ = process(index)

            count += 1

        worker_times[worker_id] = (
            time.perf_counter() - begin
        )

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
    wall_start = time.perf_counter()

    start_event.set()


    for t in threads:
        t.join()


    wall = time.perf_counter() - wall_start
    cpu_after = cpu_time()


    return {
        "throughput": sum(counts) / wall,
        "cpu": (cpu_after - cpu_before) / wall,
        "min_worker": min(worker_times),
        "max_worker": max(worker_times),
        "mean_worker": statistics.mean(worker_times),
        "std_worker": statistics.stdev(worker_times),
    }


# ============================================================
# Dynamic chunk scheduling
# ============================================================

def benchmark_dynamic(workers):

    start_event = threading.Event()

    worker_times = [0.0] * workers
    counts = [0] * workers

    next_index = 0
    index_lock = threading.Lock()


    def claim_chunk():

        nonlocal next_index

        with index_lock:

            start = next_index

            if start >= SAMPLES:
                return None

            end = min(
                start + CHUNK,
                SAMPLES
            )

            next_index = end

            return start, end


    def worker(worker_id):

        start_event.wait()

        begin = time.perf_counter()
        count = 0

        while True:

            claimed = claim_chunk()

            if claimed is None:
                break

            start, end = claimed

            for index in range(start, end):

                _ = process(index)

                count += 1


        worker_times[worker_id] = (
            time.perf_counter() - begin
        )

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
    wall_start = time.perf_counter()

    start_event.set()


    for t in threads:
        t.join()


    wall = time.perf_counter() - wall_start
    cpu_after = cpu_time()


    return {
        "throughput": sum(counts) / wall,
        "cpu": (cpu_after - cpu_before) / wall,
        "min_worker": min(worker_times),
        "max_worker": max(worker_times),
        "mean_worker": statistics.mean(worker_times),
        "std_worker": statistics.stdev(worker_times),
    }


print()
print("=" * 100)
print("STATIC vs DYNAMIC LOAD BALANCE")
print("=" * 100)

print(
    "mode,workers,repeat,throughput,cpu,"
    "worker_min_s,worker_mean_s,worker_max_s,"
    "worker_std_s,tail_ratio",
    flush=True
)


for workers in WORKERS:

    for repeat in range(1, REPEATS + 1):

        for name, fn in [
            ("static", benchmark_static),
            ("dynamic", benchmark_dynamic),
        ]:

            r = fn(workers)

            tail_ratio = (
                r["max_worker"]
                / r["mean_worker"]
            )

            print(
                f"{name},"
                f"{workers},"
                f"{repeat},"
                f"{r['throughput']:.1f},"
                f"{r['cpu']:.2f},"
                f"{r['min_worker']:.3f},"
                f"{r['mean_worker']:.3f},"
                f"{r['max_worker']:.3f},"
                f"{r['std_worker']:.3f},"
                f"{tail_ratio:.3f}",
                flush=True
            )


print("\nDONE", flush=True)
