#!/usr/bin/env python3

import sys
import time
import threading
import resource
import statistics
import gc

import torch
import torch.nn.functional as F
import torchvision.io as tvio

sys.path.insert(0, "/home/nague/bachelor-project")

from minimal_dataset import ParquetDataset, DataLoader


PARQUET = "/fscratch/nague/storage_benchmarks/images.parquet"

SAMPLES = 9984
BATCH_SIZE = 256
WORKERS = [8, 16, 32]
REPEATS = 3


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cpu_time():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# ============================================================
# Load Parquet ONCE
# ============================================================

print("Loading Parquet table once...", flush=True)

base = ParquetDataset(
    PARQUET,
    max_samples=SAMPLES
)

table = base._table

print("Table ready.", flush=True)


# ============================================================
# EXACT dataset used by BOTH tests
# ============================================================

class UInt8Dataset:

    def __len__(self):
        return SAMPLES

    def __getitem__(self, index):

        row = table.slice(index, 1).to_pylist()[0]

        encoded = torch.frombuffer(
            bytearray(row["image"]),
            dtype=torch.uint8
        )

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

        return img, row["label"]


dataset = UInt8Dataset()


# ============================================================
# DIRECT THREADED
# ============================================================

def benchmark_direct(workers):

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

            _ = dataset[index]

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

    throughput = sum(counts) / elapsed

    cores = (
        cpu_after - cpu_before
    ) / elapsed

    return throughput, cores


# ============================================================
# FULL DATALOADER
# ============================================================

def benchmark_loader(workers):

    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=workers
    )

    cpu_before = cpu_time()
    start = time.perf_counter()

    total = 0

    for images, labels in loader:
        total += images.shape[0]

    elapsed = time.perf_counter() - start
    cpu_after = cpu_time()

    throughput = total / elapsed

    cores = (
        cpu_after - cpu_before
    ) / elapsed


    # Inspect DataLoader's own valid stage measurements
    stage_info = {}

    for name in [
        "staging_put",
        "collate",
        "batch_put"
    ]:

        values = loader._stage_times.get(
            name,
            []
        )

        if values:

            stage_info[name] = (
                statistics.mean(values)
                * 1000.0
            )

        else:

            stage_info[name] = 0.0


    try:
        staging_stats = loader.staging_queue.stats()
    except Exception:
        staging_stats = {}


    del loader
    gc.collect()


    return (
        throughput,
        cores,
        stage_info,
        staging_stats
    )


# ============================================================
# RUN
# ============================================================

print()
print("=" * 95)
print("EXACT DIRECT vs DATALOADER")
print("=" * 95)

print(
    "workers,repeat,"
    "direct_tp,direct_cpu,"
    "loader_tp,loader_cpu,"
    "staging_put_ms,"
    "collate_ms,"
    "batch_put_ms",
    flush=True
)


for workers in WORKERS:

    direct_results = []
    loader_results = []


    for repeat in range(1, REPEATS + 1):

        direct_tp, direct_cpu = benchmark_direct(
            workers
        )

        (
            loader_tp,
            loader_cpu,
            stages,
            queue_stats
        ) = benchmark_loader(
            workers
        )


        direct_results.append(direct_tp)
        loader_results.append(loader_tp)


        print(
            f"{workers},"
            f"{repeat},"
            f"{direct_tp:.1f},"
            f"{direct_cpu:.2f},"
            f"{loader_tp:.1f},"
            f"{loader_cpu:.2f},"
            f"{stages['staging_put']:.3f},"
            f"{stages['collate']:.3f},"
            f"{stages['batch_put']:.3f}",
            flush=True
        )


    print(
        f"MEAN,{workers},"
        f"direct={statistics.mean(direct_results):.1f},"
        f"loader={statistics.mean(loader_results):.1f},"
        f"overhead="
        f"{(1-statistics.mean(loader_results)/statistics.mean(direct_results))*100:.1f}%",
        flush=True
    )


print()
print("DONE", flush=True)
